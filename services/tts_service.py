"""
TTS Service - Handles audio generation with queue management
"""

import asyncio
import os
import random
from typing import Optional, Dict, Any, List

from loguru import logger
import numpy as np

from services.audio_output import AudioOutputService


class VoiceRotator:
    """
    Seleziona la reference voice per ogni richiesta TTS.

    Modalità (impostabili via config["model"]["voice_rotation"]):
      - "sequential"  → gira in ordine ciclico (A B C A B C …)  [default]
      - "random"      → sceglie casualmente, evitando di ripetere
                        la stessa voce due volte di fila
      - "disabled"    → usa sempre ref_audio_path (comportamento originale)
    """

    def __init__(self, config: dict):
        model_cfg = config.get("model", {})
        rotation_cfg = config.get("voice_rotation", {})

        self.mode: str = rotation_cfg.get("mode", "sequential")

        voices_from_cfg: List[str] = rotation_cfg.get("voices", [])

        if voices_from_cfg:
            base_dir = rotation_cfg.get("voices_dir", "config/voices")
            self.voices: List[str] = [
                v if os.path.isabs(v) else os.path.join(base_dir, v)
                for v in voices_from_cfg
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

        for v in self.voices:
            if not os.path.exists(v):
                logger.warning(f"VoiceRotator: voice file not found → {v}")

        self._index: int = 0
        self._last_idx: int = -1

        logger.info(
            f"VoiceRotator pronto: {len(self.voices)} voce/i, "
            f"modalità '{self.mode}' | {[os.path.basename(v) for v in self.voices]}"
        )

    def next(self) -> str:
        if len(self.voices) == 1:
            return self.voices[0]

        if self.mode == "random":
            candidates = [i for i in range(len(self.voices)) if i != self._last_idx]
            idx = random.choice(candidates)
            self._last_idx = idx
        else:
            idx = self._index
            self._index = (self._index + 1) % len(self.voices)

        chosen = self.voices[idx]
        logger.debug(f"VoiceRotator → {os.path.basename(chosen)}")
        return chosen

    @property
    def count(self) -> int:
        return len(self.voices)


# ──────────────────────────────────────────────────────────────────────────────
# TTS Service
# ──────────────────────────────────────────────────────────────────────────────


class TTSService:
    """Async TTS service with queue management"""

    def __init__(
        self, config: dict, audio_service: Optional[AudioOutputService] = None
    ):
        self.config = config
        self.sample_rate = 48000  # VoxCPM2 nativo

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

    async def start_workers(self, num_workers: int = 1):
        logger.info(
            f"Starting {num_workers} TTS workers | "
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

        for task in self.worker_tasks:
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
                _got_request = True

                text = request.get("text", "")
                if not text:
                    continue  # finally handles task_done

                # Lazy-load VoxCPM2 model on first start
                if self.model is None:
                    async with self._inference_lock:
                        if self.model is None:
                            logger.info("Initializing VoxCPM2 (first start)...")
                            self.model = await asyncio.to_thread(
                                VoxCPMTTSPipeline, self.config
                            )
                            await asyncio.to_thread(
                                self.model.warm_up_cache, self.voice_rotator.voices
                            )

                ref_audio = request.get("ref_audio") or self.voice_rotator.next()
                logger.info(
                    f"Worker {worker_id}: VoxCPM2 '{text[:40]}...' "
                    f"| voice: {os.path.basename(ref_audio)}"
                )

                # Limited capacity queue for back-pressure
                audio_buffer = asyncio.Queue(maxsize=100)  # Max 5 chunks ahead
                streaming_done = asyncio.Event()
                chunk_count = 0

                async def producer():
                    """Generates audio in the ONNX thread, puts in queue with back-pressure"""
                    try:
                        # Creates the synchronous generator in the thread
                        def make_generator():
                            return self.model.generate_realtime_stream(
                                text=text,
                                language="it",
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
                        raise
                    finally:
                        streaming_done.set()

                async def consumer():
                    """Consumes from the queue and plays"""
                    nonlocal chunk_count
                    accumulated: list = []

                    # Buffering basato sul NUMERO di chunk, non sul tempo
                    # Aspetta 3 chunk prima di iniziare il playback per dare vantaggio alla CPU
                    TARGET_BUFFER_CHUNKS = 1

                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                audio_buffer.get(), timeout=0.5
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
                        flush = (
                            len(accumulated) >= TARGET_BUFFER_CHUNKS
                            or (streaming_done.is_set() and audio_buffer.empty())
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
                        logger.warning(f"Worker {worker_id}: surplus task_done() ignored")

    async def submit_request(self, request: Dict[str, Any]) -> bool:
        if not self._is_running:
            logger.error("TTS service not running!")
            return False

        try:
            await asyncio.wait_for(self.queue.put(request), timeout=5.0)
            logger.info(f"Request submitted to queue (size: {self.queue.qsize()})")
            return True

        except asyncio.TimeoutError:
            logger.error("Queue is full - request rejected!")
            return False
