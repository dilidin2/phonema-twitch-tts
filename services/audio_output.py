"""
Audio Output Service - Streaming Edition
Handles real-time audio playback using sounddevice.
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
    """Handles audio playback for TTS chunks and WAV files"""

    def __init__(self, method: str = "direct"):
        """
        Initialize the audio output service.

        Args:
            method: "direct" uses sounddevice (recommended for streaming),
                    "streamerbot" for external integrations.
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
                   f"SoundDevice stream started (48kHz, Mono, "
                    f"blocksize={_BLOCKSIZE}, latency=low)"
                )
            except Exception as e:
                logger.error(f"SoundDevice initialization error: {e}")

    # ── Sync API (use inside asyncio.to_thread) ──────────────────────────────

    def play_chunk_sync(self, chunk: np.ndarray) -> None:
        """
        Plays an audio chunk synchronously.

        This is the version to call inside a thread (asyncio.to_thread).
        stream.write() with latency='low' blocks when the ring-buffer is
        full, forcing the generator to proceed at playback speed.
        """
        if self.method == "direct" and self.stream:
            try:
                self.stream.write(chunk.astype(np.float32))
            except Exception as e:
                logger.error(f"Error during chunk playback: {e}")

    # ── Async API (kept for compatibility) ───────────────────────────────────

    async def play_chunk(self, chunk: np.ndarray) -> None:
        """
        Async wrapper for play_chunk_sync.
        Prefer direct call to play_chunk_sync inside to_thread.
        """
        self.play_chunk_sync(chunk)

    async def play(self, file_path: str) -> bool:
        """Plays a complete WAV file (compatibility fallback)."""
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
        """Stops the audio stream"""
        self._is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped.")
