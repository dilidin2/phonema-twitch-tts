"""
Audio Output Service - Streaming Edition
Gestisce la riproduzione audio in tempo reale utilizzando sounddevice.
"""

import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger

# Hardware block size in samples at 24kHz.
# Lower values = lower latency but more overhead.
# 2048 samples ≈ 85ms — good compromise for TTS streaming.
_BLOCKSIZE = 2048


class AudioOutputService:
    """Gestisce la riproduzione audio per i chunk TTS e file WAV"""

    def __init__(self, method: str = "direct"):
        """
        Inizializza il servizio di output audio.

        Args:
            method: "direct" usa sounddevice (consigliato per streaming),
                    "streamerbot" per integrazioni esterne.
        """
        self.method = method
        self.samplerate = 48000  # VoxCPM2 nativo (48kHz)
        self._is_playing = False

        self.stream = None
        if self.method == "direct":
            try:
                # latency='low' keeps the hardware ring-buffer small:
                # stream.write() will block when full, creating
                # natural back-pressure on the chunk generator.
                self.stream = sd.OutputStream(
                    samplerate=self.samplerate,
                    channels=1,
                    dtype="float32",
                    blocksize=_BLOCKSIZE,
                    latency="low",
                )
                self.stream.start()
                logger.info(
                    f"SoundDevice stream started (24kHz, Mono, "
                    f"blocksize={_BLOCKSIZE}, latency=low)"
                )
            except Exception as e:
                logger.error(f"SoundDevice initialization error: {e}")

    # ── API sincrona (da usare dentro asyncio.to_thread) ─────────────────────

    def play_chunk_sync(self, chunk: np.ndarray) -> None:
        """
        Riproduce un chunk di audio in modo sincrono.

        Questa è la versione da chiamare dentro un thread (asyncio.to_thread).
        stream.write() con latency='low' si blocca quando il ring-buffer è
        pieno, forzando il generatore a procedere al ritmo della riproduzione.
        """
        if self.method == "direct" and self.stream:
            try:
                self.stream.write(chunk.astype(np.float32))
            except Exception as e:
                logger.error(f"Error during chunk playback: {e}")
                # Attempt to recover
                if not self.stream.active:
                    logger.warning("Audio stream inactive — attempting restart")
                    try:
                        self.stream.start()
                    except Exception:
                        logger.error("Failed to restart audio stream")

    # ── API asincrona (mantenuta per compatibilità) ───────────────────────────

    async def play_chunk(self, chunk: np.ndarray) -> None:
        """
        Wrapper asincrono di play_chunk_sync.
        Preferire la chiamata diretta a play_chunk_sync dentro to_thread.
        """
        self.play_chunk_sync(chunk)

    async def play(self, file_path: str) -> bool:
        """Riproduce un file WAV intero (fallback per compatibilità)."""
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return False

        try:
            if self.method == "direct":
                data, _ = sf.read(file_path)
                self.play_chunk_sync(data)
                return True
            return False
        except Exception as e:
            logger.error(f"File playback error: {e}")
            return False

    async def stop(self):
        """Ferma lo stream audio"""
        self._is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped.")
