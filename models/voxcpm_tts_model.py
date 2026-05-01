"""
VoxCPM2 TTS Model Wrapper — Unified (CPU/GPU)
Wrapper per VoxCPM2 (openbmb/VoxCPM2) con supporto CPU e GPU.
Mantiene la stessa interfaccia API di XttsTTSPipeline per compatibilità.

Requisiti: pip install voxcpm torch
Python ≥ 3.10, PyTorch ≥ 2.5.0 (con o senza CUDA support)
"""

import os
import re
import numpy as np
import soundfile as sf
import torch
torch.backends.mkldnn.enabled = True
torch.set_num_threads(os.cpu_count())
import torch.nn.functional as F
from typing import Tuple, Optional, Generator, List, Dict
from loguru import logger


def _ensure_voxcpm_installed():
    """Verifica che voxcpm sia installato."""
    try:
        import voxcpm  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "VoxCPM non installato. Esegui: pip install voxcpm\n"
            "Requisiti: Python ≥ 3.10, PyTorch ≥ 2.5.0 (con o senza CUDA)"
        )


def _patch_sdpa_for_cpu():
    """
    Workaround per bug VoxCPM2 su CPU.

    Durante forward_step (decoding autoregressivo token-by-token) i tensori
    Q/K/V possono essere ridotti a 1-2D per un squeeze non intenzionale nel
    codice VoxCPM. scaled_dot_product_attention richiede almeno 4D
    [batch, heads, seq, head_dim] → IndexError: Dimension out of range.

    Questo patch porta i tensori a 4D prima della chiamata e rimuove le
    dimensioni aggiunte dall'output, senza alterare i valori.
    """
    if getattr(F, "_voxcpm_sdpa_patched", False):
        return  # già applicato, evita doppio wrapping

    _orig_sdpa = F.scaled_dot_product_attention

    def _safe_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=None,
        **kwargs,
    ):
        q_dim = query.dim()
        k_dim = key.dim()
        v_dim = value.dim()

        missing_q = max(0, 4 - q_dim)
        missing_k = max(0, 4 - k_dim)
        missing_v = max(0, 4 - v_dim)

        # Aggiungi dimensioni mancanti
        if missing_q:
            for _ in range(missing_q):
                query = query.unsqueeze(0)
        if missing_k:
            for _ in range(missing_k):
                key = key.unsqueeze(0)
        if missing_v:
            for _ in range(missing_v):
                value = value.unsqueeze(0)
        if attn_mask is not None:
            mask_missing = max(0, 4 - attn_mask.dim())
            for _ in range(mask_missing):
                attn_mask = attn_mask.unsqueeze(0)

        out = _orig_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            **kwargs,
        )

        # Rimuovi dimensioni aggiunte dall'output
        if missing_q:
            for _ in range(missing_q):
                out = out.squeeze(0)
        return out

    # Patch globale — quando voxcpm importerà torch.nn.functional prenderà questa
    F.scaled_dot_product_attention = _safe_sdpa
    torch.nn.functional.scaled_dot_product_attention = _safe_sdpa

    # Patch sui riferimenti interni di voxcpm se già caricati/disponibili
    try:
        import voxcpm.core as vc
        if hasattr(vc, "F") and hasattr(vc.F, "scaled_dot_product_attention"):
            vc.F.scaled_dot_product_attention = _safe_sdpa
        if hasattr(vc, "scaled_dot_product_attention"):
            vc._orig_sdpa = vc.scaled_dot_product_attention
            vc.scaled_dot_product_attention = _safe_sdpa
    except Exception:
        pass

    F._voxcpm_sdpa_patched = True


def _unpatch_sdpa_for_cpu():
    """Non fare nulla: la patch deve restare attiva per tutta l'inferenza."""
    pass


class VoxCPMTTSPipeline:
    """VoxCPM2 streaming pipeline wrapper con voice cloning (CPU/GPU)."""

    MAX_CHUNK_CHARS: int = 400
    CHUNK_SILENCE_SEC: float = 0.1

    def __init__(self, config: dict):
        self.config = config
        model_cfg = config["model"]

        # Determina dispositivo
        force_cpu = (
            model_cfg.get("force_cpu", False)
            or model_cfg.get("device") == "cpu"
        )

        if force_cpu:
            self.device = "cpu"
            logger.info("CPU forced via config")
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                logger.info("Nascosto CUDA_VISIBLE_DEVICES=-1 per forzare CPU")
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA available - using GPU")
        else:
            self.device = "cpu"
            logger.info("No CUDA available - using CPU")

        self.sr = 48000  # VoxCPM2 nativo (48kHz)

        model_path = model_cfg.get("pretrained_path", "openbmb/VoxCPM2")
        dtype_str = model_cfg.get("dtype", "float32")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(dtype_str, torch.float32)

        logger.info(f"Loading VoxCPM2 from {model_path} on {self.device.upper()}...")
        logger.info(f"  Device: {self.device} | Dtype: {dtype_str}")

        _ensure_voxcpm_installed()

        if self.device == "cpu":
            logger.info("⚡ Applicazione Patch SDPA per stabilità su CPU...")
            _patch_sdpa_for_cpu()
            num_threads = model_cfg.get("num_threads_cpu", 4)
            torch.set_num_threads(num_threads)
            logger.info(f"CPU threads: {num_threads}")

        if self.device == "cpu":
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
            logger.info("  Monkey-patched torch.cuda → forced CPU visibility")

        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(
            model_path,
            load_denoiser=False,
            optimize=False,
        )

        if self.device == "cpu" and self.dtype == torch.bfloat16:
            logger.warning(
                "  ⚠️  bfloat16 on CPU can cause illegal instruction errors. "
                "Change model.dtype to 'float32' in config if needed."
            )

        logger.info(
            f"VoxCPM2 loaded! sr={self.sr} Hz | max_chunk={self.MAX_CHUNK_CHARS} chars"
        )

        # Cache per le reference voices
        self._latents_cache: Dict[str, dict] = {}

    def _get_latents(self, ref_audio: str) -> dict:
        """Calcola latents di condizionamento per una reference voice."""
        if ref_audio not in self._latents_cache:
            logger.info(f"Latents cache MISS → {os.path.basename(ref_audio)}")
            self._latents_cache[ref_audio] = {"ref_path": ref_audio}
            logger.info(f"Latents cache: {len(self._latents_cache)} voce/i")
        else:
            logger.debug(f"Latents cache HIT → {os.path.basename(ref_audio)}")

        return self._latents_cache[ref_audio]

    def warm_up_cache(self, voice_paths: List[str]):
        """Pre-carica le reference voices (solo latents, senza inferenza)."""
        logger.info(f"Warming up VoxCPM for {len(voice_paths)} voce/i...")
        for path in voice_paths:
            if os.path.exists(path):
                self._get_latents(path)
            else:
                logger.warning(f"Warm-up: file not found → {path}")
        logger.info("VoxCPM warm-up completed (latents ready)")

    def clear_cache(self, ref_audio: Optional[str] = None):
        """Svuota la cache."""
        if ref_audio:
            self._latents_cache.pop(ref_audio, None)
        else:
            self._latents_cache.clear()

    def _split_into_chunks(self, text: str) -> List[str]:
        """Divide il testo in chunk più piccoli."""
        if len(text) <= self.MAX_CHUNK_CHARS:
            return [text.strip()]

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self.MAX_CHUNK_CHARS:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence

        if current:
            while len(current) > self.MAX_CHUNK_CHARS:
                split_at = current.rfind(" ", 0, self.MAX_CHUNK_CHARS)
                if split_at == -1:
                    split_at = self.MAX_CHUNK_CHARS
                chunks.append(current[:split_at].strip())
                current = current[split_at:].strip()
            if current:
                chunks.append(current.strip())

        chunks = [c for c in chunks if c]
        logger.debug(f"Text split into {len(chunks)} chunks")
        return chunks

    def _infer_chunk_stream(
        self,
        text: str,
        ref_audio: str,
        speed: float = 1.0,
        inference_timesteps: int = 4,
    ) -> Generator[np.ndarray, None, None]:
        """Genera audio per un singolo chunk con streaming."""
        try:
            for chunk in self.model.generate_streaming(
                text=text,
                reference_wav_path=ref_audio,
                inference_timesteps=inference_timesteps,
            ):
                if chunk is not None and len(chunk) > 0:
                    yield chunk.astype(np.float32)
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    def stream_voice_clone(
        self,
        text: str,
        language: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        chunk_size_sec: float = 0.5,
        speed: float = 1.0,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Streaming REALE: yield di (audio_chunk, sample_rate)."""
        if not ref_audio or not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        latents = self._get_latents(ref_audio)
        chunks = self._split_into_chunks(text)
        n = len(chunks)
        silence = np.zeros(int(self.CHUNK_SILENCE_SEC * self.sr), dtype=np.float32)

        logger.info(
            f"Streaming VoxCPM2 | voice: {os.path.basename(ref_audio)} | "
            f"{n} chunk/s | {len(text)} chars"
        )

        for idx, chunk in enumerate(chunks, start=1):
            logger.debug(f"  Chunk {idx}/{n}: '{chunk[:50]}...'")
            for audio_piece in self._infer_chunk_stream(
                chunk, ref_audio, speed,
                self.config["model"].get("inference_timesteps", 4)  # ← leggi dalla config
            ):
                if len(audio_piece) > 0:
                    yield audio_piece.astype(np.float32), self.sr

            if idx < n:
                yield silence, self.sr

    def generate_chunked(
        self,
        text: str,
        language: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Genera intero audio concatenando chunk."""
        if not ref_audio or not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        parts: List[np.ndarray] = []
        for audio_chunk, _ in self.stream_voice_clone(
            text, language, ref_audio, ref_text, speed=speed
        ):
            parts.append(audio_chunk)

        if not parts:
            return np.zeros(0, dtype=np.float32), self.sr

        full_audio = np.concatenate(parts)
        full_audio = self._post_process(full_audio)

        logger.info(f"Done: {len(full_audio) / self.sr:.1f}s of audio")
        return full_audio, self.sr

    def generate_voice_clone(
        self,
        text: str,
        language: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Alias di generate_chunked."""
        return self.generate_chunked(text=text, language=language, ref_audio=ref_audio)

    def generate_realtime_stream(
        self,
        text: str,
        language: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        inference_timesteps: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """Generatore realtime per playback immediato."""
        if not ref_audio:
            ref_audio = self.config["model"].get("ref_audio_path")

        if not ref_audio or not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        if inference_timesteps is None:
            inference_timesteps = self.config["model"].get("inference_timesteps", 1)

        logger.info(
            f"Streaming VoxCPM2 | voice: {os.path.basename(ref_audio)} | "
            f"steps: {inference_timesteps}"
        )
        logger.info(f"Streaming TTS started for: {text[:30]}...")

        latents = self._get_latents(ref_audio)
        chunks = self._split_into_chunks(text)
        n = len(chunks)
        silence = np.zeros(int(self.CHUNK_SILENCE_SEC * self.sr), dtype=np.float32)

        logger.info(f"Streaming VoxCPM2 | voice: {os.path.basename(ref_audio)}")

        for idx, chunk in enumerate(chunks, start=1):
            for audio_piece in self._infer_chunk_stream(
                chunk, ref_audio, speed, inference_timesteps
            ):
                if audio_piece is not None and len(audio_piece) > 0:
                    yield audio_piece.astype(np.float32)

            if idx < n:
                yield silence

        logger.info("Streaming TTS completed.")

    def generate_simple(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generazione semplice senza streaming."""
        lang = language or self.config["model"].get("language", "it")
        default_ref = self.config["model"].get("ref_audio_path")
        if not default_ref:
            raise ValueError(
                "VoxCPM2 requires a reference audio. "
                "Configure ref_audio_path in config."
            )
        return self.generate_chunked(text, lang, ref_audio=default_ref)

    def _post_process(self, audio: np.ndarray) -> np.ndarray:
        """Trim silenzio e normalizzazione."""
        threshold = 0.005
        mask = np.abs(audio) > threshold
        if mask.any():
            last = int(np.where(mask)[0][-1])
            padding = int(0.3 * self.sr)
            audio = audio[: min(last + padding, len(audio))]

        # Fade-out finale
        fade_len = int(0.1 * self.sr)
        if len(audio) > fade_len:
            audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

        # Normalizzazione
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.89

        return audio.astype(np.float32)
