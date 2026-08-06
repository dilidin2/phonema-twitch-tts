"""
Audio Output Service - Streaming Edition
Riproduce l'audio in tempo reale passando dal server audio del sistema
(PulseAudio o PipeWire-pulse) tramite la Pulse Simple API, invece di
scrivere direttamente sull'hardware ALSA.

Perché pasimple e non sounddevice/ALSA diretto:
- ALSA grezzo scrive sull'hardware bypassando PulseAudio/PipeWire: lo stream
  non compare in pavucontrol e può finire su un device fisico diverso da
  quello che l'utente sta effettivamente ascoltando (es. scheda audio
  integrata invece delle cuffie USB), soprattutto se manca il plugin ALSA
  di integrazione (pipewire-alsa / pulseaudio-alsa).
- pasimple parla direttamente col server audio (pulse o l'emulazione pulse
  di PipeWire), che è presente in pratica su ogni distro Linux desktop.
  Lo stream compare correttamente in pavucontrol e può essere instradato
  a un sink specifico (es. un sink virtuale creato per OBS) tramite nome,
  con fallback automatico al sink di default se quel nome non esiste.
"""

import os
import numpy as np
from loguru import logger

try:
    from pasimple import PaSimple, PA_STREAM_PLAYBACK, PA_SAMPLE_FLOAT32LE, PaSimpleError
except ImportError:
    PaSimple = None
    PaSimpleError = Exception
    logger.error(
        "pasimple non installato — esegui `pip install pasimple`. "
        "Serve la libreria di sistema libpulse.so.0 (presente su qualunque "
        "sistema con PulseAudio o PipeWire)."
    )


class AudioOutputService:
    """Gestisce la riproduzione audio per i chunk TTS e file WAV.

    L'audio viene inviato al server audio di sistema (PulseAudio/PipeWire)
    tramite la Pulse Simple API, non scritto direttamente sull'hardware.
    Questo garantisce portabilità: funziona su qualunque sistema Linux con
    un server audio pulse-compatibile attivo, senza dipendere da indici di
    device ALSA specifici della macchina.
    """

    def __init__(
        self,
        method: str = "direct",
        samplerate: int = 48000,
        sink_name: str | None = None,
        app_name: str = "Twitch TTS",
    ):
        """
        Inizializza il servizio di output audio.

        Args:
            method: "direct" usa pasimple/Pulse (consigliato per streaming),
                    "streamerbot" per integrazioni esterne.
            samplerate: sample rate nativo del modello TTS (48000=VoxCPM2, 24000=Pocket)
            sink_name: nome del sink pulse/pipewire di destinazione (es. "virtual_tts").
                       Se None o se il sink non esiste, usa il sink di default di sistema.
            app_name: nome mostrato in pavucontrol per identificare lo stream.
        """
        self.method = method
        self.samplerate = samplerate
        self.sink_name = sink_name
        self.app_name = app_name
        self._is_playing = False

        self.pa = None
        if self.method == "direct":
            self.pa = self._open_stream(sink_name)

    def _open_stream(self, sink_name: str | None):
        """Apre uno stream pulse, con fallback al sink di default se il sink
        nominato non esiste o non è raggiungibile."""
        if PaSimple is None:
            logger.error("Impossibile avviare l'output audio: pasimple non disponibile.")
            return None

        target = sink_name
        candidates = [target] if target is None else [target, None]
        for attempt_name in candidates:
            try:
                stream = PaSimple(
                    direction=PA_STREAM_PLAYBACK,
                    format=PA_SAMPLE_FLOAT32LE,
                    channels=1,
                    rate=self.samplerate,
                    app_name=self.app_name,
                    stream_name="TTS Output",
                    device_name=attempt_name,
                )
                label = attempt_name if attempt_name else "sink di default"
                logger.info(
                    f"Stream audio pulse/pipewire avviato su '{label}' "
                    f"({self.samplerate / 1000:.0f}kHz, Mono)"
                )
                if attempt_name != target:
                    logger.warning(
                        f"Sink '{target}' non trovato — uso il sink di default come fallback."
                    )
                return stream
            except PaSimpleError as e:
                if attempt_name == target and target is not None:
                    logger.warning(
                        f"Impossibile aprire il sink '{target}' ({e}), riprovo con il default..."
                    )
                    continue
                logger.error(f"Inizializzazione audio (pulse/pipewire) fallita: {e}")
                return None
            except Exception as e:
                logger.error(f"Errore inatteso nell'apertura dello stream audio: {e}")
                return None
        return None

    # ── API sincrona (da usare dentro asyncio.to_thread) ─────────────────────

    def play_chunk_sync(self, chunk: np.ndarray) -> None:
        """
        Riproduce un chunk di audio in modo sincrono.

        Questa è la versione da chiamare dentro un thread (asyncio.to_thread).
        pa.write() si blocca quando il buffer interno è pieno, creando una
        naturale back-pressure sul generatore, come faceva prima stream.write().
        """
        if self.method == "direct" and self.pa:
            try:
                self.pa.write(chunk.astype(np.float32).tobytes())
            except Exception as e:
                logger.error(f"Error during chunk playback: {e}")
                # Attempt to recover riaprendo lo stream
                logger.warning("Stream audio non valido — provo a riaprirlo")
                try:
                    if self.pa:
                        self.pa.close()
                except Exception:
                    pass
                self.pa = self._open_stream(self.sink_name)

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
                import soundfile as sf

                data, _ = sf.read(file_path, dtype="float32")
                self.play_chunk_sync(data)
                return True
            return False
        except Exception as e:
            logger.error(f"File playback error: {e}")
            return False

    async def stop(self):
        """Ferma lo stream audio"""
        self._is_playing = False
        if self.pa:
            try:
                self.pa.drain()
            except Exception:
                pass
            try:
                self.pa.close()
            except Exception:
                pass
            logger.info("Audio stream stopped.")
