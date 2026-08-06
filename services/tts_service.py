"""
TTS Service - Handles audio generation with queue management
"""

import asyncio
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from services.audio_output import AudioOutputService

# ── Model type constants ─────────────────────────────────────────────────────
MODEL_VOXCPM = "gpu_model"
MODEL_POCKET = "cpu_model"


def _resolve_voice(entry: str, voices_dir: str) -> str:
    """
    Risolve automaticamente una voce del config.

    - Se è un nome nel catalogo → restituisce il nome (catalog voice, no HF login)
    - Se ha estensione .wav/.safetensors → risolve come file path (clone voice)
    - Altrimenti → prova come file path in voices_dir
    """
    from models.pocket_tts_model import CATALOG_VOICES

    if entry in CATALOG_VOICES:
        return entry  # catalog voice — returned as-is, no path resolution
    if os.path.isabs(entry) or any(
        entry.endswith(ext) for ext in (".wav", ".safetensors")
    ):
        return entry if os.path.isabs(entry) else os.path.join(voices_dir, entry)
    # No extension and not a catalog name → treat as file path anyway
    return os.path.join(voices_dir, entry)


class VoiceRotator:
    """
    Seleziona la reference voice per ogni richiesta TTS.

    Modalità (impostabili via config["model"]["voice_rotation"]):
      - "sequential"  → gira in ordine ciclico (A B C A B C …)  [default]
      - "random"      → sceglie casualmente, evitando di ripetere
                        la stessa voce due volte di fila
      - "disabled"    → usa sempre ref_audio_path (comportamento originale)

    Rilevamento automatico: la lista 'voices' può contenere sia nomi catalog
    (es. "giovanni", "alba") che file path (es. "voice_a.wav"). Il codice
    distingue automaticamente senza bisogno di voice_mode.
    """

    def __init__(self, config: dict):
        model_cfg = config.get("model", {})
        rotation_cfg = config.get("voice_rotation", {})

        self.mode: str = rotation_cfg.get("mode", "sequential")
        voices_from_cfg: List[str] = rotation_cfg.get("voices", [])
        voices_dir: str = rotation_cfg.get("voices_dir", "config/voices")

        if voices_from_cfg:
            self.voices: List[str] = [
                _resolve_voice(v, voices_dir) for v in voices_from_cfg
            ]
        else:
            single = model_cfg.get("ref_audio_path", "")
            self.voices = [single] if single else []
            if self.voices:
                logger.info(
                    "VoiceRotator: nessuna lista voci, uso ref_audio_path singolo"
                )

        if not self.voices:
            raise ValueError(
                "VoiceRotator: nessuna voce configurata. "
                "Imposta model.voice_rotation.voices o model.ref_audio_path nel config."
            )

        # Validation + categorization
        from models.pocket_tts_model import _is_catalog_voice

        catalog_count = 0
        clone_count = 0
        for v in self.voices:
            if _is_catalog_voice(v):
                catalog_count += 1
            elif not os.path.exists(v):
                logger.warning(f"VoiceRotator: voice file not found → {v}")
            else:
                clone_count += 1

        display_names = [
            v if _is_catalog_voice(v) else os.path.basename(v)
            for v in self.voices
        ]
        logger.info(
            f"VoiceRotator pronto: {len(self.voices)} voce/i, "
            f"modalità '{self.mode}' | catalog={catalog_count}, clone={clone_count} | "
            f"{display_names}"
        )

        # State for rotation modes
        self._index = 0
        self._last_idx = -1

    def next(self) -> str:
        from models.pocket_tts_model import _is_catalog_voice

        if len(self.voices) == 1:
            if not _is_catalog_voice(self.voices[0]) and not os.path.exists(self.voices[0]):
                raise FileNotFoundError(f"Voice file missing: {self.voices[0]}")
            return self.voices[0]

        if self.mode == "random":
            candidates = [i for i in range(len(self.voices)) if i != self._last_idx]
            idx = random.choice(candidates)
            self._last_idx = idx
        else:
            idx = self._index
            self._index = (self._index + 1) % len(self.voices)

        chosen = self.voices[idx]
        if not _is_catalog_voice(chosen) and not os.path.exists(chosen):
            logger.error(f"Voice file not found at runtime: {chosen}")
            raise FileNotFoundError(chosen)
        display = chosen if _is_catalog_voice(chosen) else os.path.basename(chosen)
        logger.debug(f"VoiceRotator → {display}")
        return chosen

    @property
    def count(self) -> int:
        return len(self.voices)


# ──────────────────────────────────────────────────────────────────────────────
# TTS Service
# ──────────────────────────────────────────────────────────────────────────────


class TTSService:
    """Async TTS service with queue management.

    Supporto dual-mode: scegli il modello via config["model_type"]:
      - "gpu_model"  → VoxCPM2 (pesante, richiede GPU)
      - "cpu_model"  → Pocket TTS (leggero, gira su CPU)
    """

    def __init__(
        self, config: dict, audio_service: Optional[AudioOutputService] = None
    ):
        self.config = config
        self.model_type = config.get("model_type", MODEL_VOXCPM)
        self.sample_rate = 48000 if self.model_type == MODEL_VOXCPM else 24000

        self.model = None
        self.audio_service = audio_service

        self.queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.get("queue", {}).get("max_size", 10)
        )

        self.worker_tasks: list = []
        self._is_running = False

        # Semaforo: una sola inferenza alla volta
        self._inference_lock = asyncio.Semaphore(1)

        self.voice_rotator = VoiceRotator(config)

        logger.info(f"TTS model type: {self.model_type} ({self.sample_rate / 1000:.0f}kHz)")

    async def start_workers(self, num_workers: int = 1):
        logger.info(
            f"Starting {num_workers} TTS workers | model={self.model_type} | "
            f"voices: {self.voice_rotator.count} | "
            f"rotation: {self.voice_rotator.mode}"
        )
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(worker_id=i))
            self.worker_tasks.append(task)

        self._is_running = True
        logger.info("TTS workers started")

    async def stop_workers(self):
        self._is_running = False

        # Signal shutdown via sentinel first
        for _ in self.worker_tasks:
            try:
                await self.queue.put(None)  # Sentinel per worker
            except asyncio.QueueFull:
                pass

        # Wait for graceful exit before forcing cancel
        for task in self.worker_tasks:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        logger.info("TTS workers stopped")

    # ── Generation + streaming worker ────────────────────────────────────────

    async def _worker_loop(self, worker_id: int):
        logger.info(f"Worker {worker_id} ready.")

        while self._is_running:
            # Track whether we actually dequeued a request,
            # so finally knows if task_done() is needed.
            _got_request = False
            try:
                request = await self.queue.get()

                if request is None:  # Shutdown sentinel
                    break

                _got_request = True

                text = request.get("text", "")
                if not text:
                    continue  # finally handles task_done

                # Lazy-load model on first start (dual-mode)
                if self.model is None:
                    async with self._inference_lock:
                        if self.model is None:
                            if self.model_type == MODEL_POCKET:
                                from models.pocket_tts_model import PocketTTSPipeline
                                logger.info("Initializing Pocket TTS (CPU model, first start)...")
                                self.model = await asyncio.to_thread(
                                    PocketTTSPipeline, self.config
                                )
                            else:
                                from models.voxcpm_tts_model import VoxCPMTTSPipeline
                                logger.info("Initializing VoxCPM2 (GPU model, first start)...")
                                self.model = await asyncio.to_thread(
                                    VoxCPMTTSPipeline, self.config
                                )
                            # Warm-up: pass voice names (catalog) or existing files (clone)
                            from models.pocket_tts_model import _is_catalog_voice
                            warmup_voices = [
                                v for v in self.voice_rotator.voices
                                if _is_catalog_voice(v) or os.path.exists(v)
                            ]
                            await asyncio.to_thread(
                                self.model.warm_up_cache, warmup_voices
                            )

                ref_audio = request.get("ref_audio") or self.voice_rotator.next()
                model_label = "Pocket TTS" if self.model_type == MODEL_POCKET else "VoxCPM2"
                logger.info(
                    f"Worker {worker_id}: {model_label} '{text[:40]}...' "
                    f"| voice: {os.path.basename(ref_audio)}"
                )

                # Limited capacity queue for back-pressure
                audio_buffer = asyncio.Queue(maxsize=100)
                streaming_done = asyncio.Event()
                chunk_count = 0

                async def producer():
                    """Generates audio in the ONNX thread, puts in queue with back-pressure"""
                    try:
                        # Creates the synchronous generator in the thread
                        def make_generator():
                            return self.model.generate_realtime_stream(
                                text=text,

                                ref_audio=ref_audio,
                            )

                        # Initializes the synchronous generator in the ONNX thread
                        sync_gen = await asyncio.to_thread(make_generator)

                        # Consumes the synchronous generator chunk by chunk
                        while True:
                            # Gets next chunk in separate thread
                            def next_chunk(g):
                                try:
                                    return next(g), False
                                except StopIteration:
                                    return None, True

                            chunk, done = await asyncio.to_thread(next_chunk, sync_gen)

                            if done or chunk is None:
                                break

                            # This await BLOCKS if the queue is full (5 chunks)
                            # Creating back-pressure on the ONNX inference!
                            await audio_buffer.put(chunk)

                    except Exception as e:
                        logger.error(f"Producer error: {e}")
                        streaming_done.set()  # Signal consumer to flush and exit
                        return  # Don't re-raise — let gather complete normally
                    finally:
                        streaming_done.set()

                async def consumer():
                    """Consumes from the queue and plays"""
                    nonlocal chunk_count
                    accumulated: list = []

                    TARGET_BUFFER_CHUNKS = 1

                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                audio_buffer.get(), timeout=2.0
                            )
                        except asyncio.TimeoutError:
                            # Flush whatever we have if streaming is done
                            if streaming_done.is_set() and audio_buffer.empty():
                                if accumulated:
                                    combined = np.concatenate(accumulated)
                                    await asyncio.to_thread(
                                        self.audio_service.play_chunk_sync, combined
                                    )
                                    chunk_count += len(accumulated)
                                break
                            continue

                        accumulated.append(chunk)
                        audio_buffer.task_done()

                        # Flusha SOLO quando abbiamo accumulato abbastanza chunk
                        # o quando lo streaming è completato
                        flush = len(accumulated) >= TARGET_BUFFER_CHUNKS or (
                            streaming_done.is_set() and audio_buffer.empty()
                        )

                        if flush:
                            combined = np.concatenate(accumulated)
                            await asyncio.to_thread(
                                self.audio_service.play_chunk_sync, combined
                            )
                            chunk_count += len(accumulated)
                            accumulated = []

                # Runs producer and consumer in parallel
                await asyncio.gather(producer(), consumer())
                await audio_buffer.join()  # Ensure all items processed

                logger.success(
                    f"Worker {worker_id}: streaming completato "
                    f"({chunk_count} chunk riprodotti)."
                )

            except asyncio.CancelledError:
                raise  # let finally run, then propagate
            except Exception as e:
                logger.error(f"Worker {worker_id}: critical error: {e}", exc_info=True)
            finally:
                # Always called exactly once per get(), regardless of success/exception.
                if _got_request:
                    try:
                        self.queue.task_done()
                    except ValueError:
                        # Should never happen, but just in case
                        logger.warning(
                            f"Worker {worker_id}: surplus task_done() ignored"
                        )

    async def submit_request(self, request: Dict[str, Any]) -> bool:
        if not self._is_running or not self.worker_tasks:
            logger.error("TTS service not running or no workers!")
            return False

        try:
            await asyncio.wait_for(self.queue.put(request), timeout=5.0)
            logger.info(f"Request submitted to queue (size: {self.queue.qsize()})")
            return True

        except asyncio.TimeoutError:
            logger.error("Queue is full - request rejected!")
            return False
