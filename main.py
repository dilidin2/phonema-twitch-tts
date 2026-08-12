"""
Twitch Channel Points TTS Server - FastAPI Application
Main entry point for the HTTP REST API + WebSocket listener
"""

import os
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any


from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Load configuration
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_blacklist():
    """Carica la blacklist dagli utenti nel file config/blacklist.txt."""
    path = "config/blacklist.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return {line.strip().lower() for line in f if line.strip()}
    return set()


def load_config():
    with open("config/tts_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Public client_id — override via env for forks with their own app
    config["TWITCH_CLIENT_ID"] = os.getenv(
        "TWITCH_CLIENT_ID", "0dy5ss974g4hsuygfbvju265xrhb87"
    )

    return config


CONFIG = load_config()
BLACKLIST = load_blacklist()

logger.info(f"Blacklist caricata: {len(BLACKLIST)} utente/i → {BLACKLIST}")

# Import services
from services.tts_service import TTSService
from services.twitch_service import TwitchService, TOKEN_PATH
from services.audio_output import AudioOutputService


# Pydantic models
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    voice_id: Optional[str] = None


class TTSResponse(BaseModel):
    success: bool
    message: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_type: str
    cuda_available: bool
    cuda_devices: int
    queue_size: int
    workers_active: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Show model type
    model_type = CONFIG.get("model_type", "gpu_model")
    if model_type == "cpu_model":
        model_label = "Pocket TTS (CPU, Kyutai)"
    else:
        model_label = "VoxCPM2 (GPU, BMB/ModelScope)"

    logger.info("=" * 60)
    logger.info("  Starting Twitch Channel Points TTS Server")
    logger.info(f"  Model: {model_label}")
    logger.info("=" * 60)

    # Check CUDA availability
    try:
        import torch

        valid_gpus = []
        cuda_available = False

        if torch.cuda.is_available():
            # Testiamo fisicamente le GPU
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.get_device_properties(i)
                    valid_gpus.append(i)
                except Exception:
                    continue

            if len(valid_gpus) > 0:
                cuda_available = True

            num_gpus = len(valid_gpus)
            logger.info(
                f"  CUDA/HIP available: {cuda_available} (Found {num_gpus} valid GPU/s)"
            )

            for i in valid_gpus:
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    logger.info(f"  ✓ GPU {i}: {gpu_name}")
                except:
                    logger.warning(f"  ⚠ GPU {i} detected but name unreachable")

        if not cuda_available:
            logger.warning(
                "  ⚠️  CUDA/HIP not available or disabled - CPU inference active!"
            )

    except ImportError:
        logger.error("  ✗ PyTorch not installed!")
        raise

    # Initialize services
    logger.info("  Initializing services...")

    # Dynamic sample rate based on model type
    model_type = CONFIG.get("model_type", "gpu_model")
    samplerate = 24000 if model_type == "cpu_model" else 48000

    audio_service = AudioOutputService(
        method=CONFIG.get("AUDIO_OUTPUT_METHOD", "direct"),
        samplerate=samplerate,
        # Nome del sink pulse/pipewire a cui instradare l'audio (es. un sink
        # virtuale collegato a OBS). Se assente o non trovato sul sistema,
        # si usa automaticamente il sink di default — nessuna dipendenza
        # da hardware specifico della macchina.
        sink_name=CONFIG.get("AUDIO_SINK_NAME") or None,
    )
    tts_service = TTSService(CONFIG, audio_service=audio_service)
    await tts_service.start_workers(num_workers=1)

    twitch_service = TwitchService(CONFIG)

    # Auto-connect to Twitch using saved tokens from token.json
    # Broadcaster ID is resolved automatically from the OAuth token (no .env needed)
    logger.info("  Auto-connecting to Twitch...")
    try:
        await twitch_service.connect()
        if TOKEN_PATH.exists():
            await twitch_service.authenticate_user()
            await twitch_service.listen_channel_points_redemption(twitch_service.broadcaster_id)
            logger.info(
                f"  ✓ Twitch connected as {twitch_service.broadcaster_name} "
                f"(ID: {twitch_service.broadcaster_id})"
            )
        else:
            # No saved tokens — start DCF automatically in background
            result = await twitch_service._start_device_flow_async()
            logger.warning("=" * 50)
            logger.warning("  ⚠️  No token.json found — first-time authorization")
            logger.warning(f"  Open: {result['verification_uri']}")
            logger.warning(f"  Enter code: {result['user_code']}")
            logger.warning(f"  Expires in: {result['expires_in']}s")
            logger.warning("=" * 50)
    except Exception as e:
        logger.warning(f"Auto-connect failed: {e}")

    logger.info("  ✓ All services initialized")

    app.state.cuda_available = cuda_available
    app.state.cuda_devices = len(valid_gpus) if 'valid_gpus' in dir() else 0
    app.state.audio_service = audio_service
    app.state.tts_service = tts_service
    app.state.twitch_service = twitch_service

    # Wire redemption callback from TwitchService → TTSService
    async def on_redemption(data):
        text = data.get("user_input", "") or ""
        user_name = data.get("user_name", "A user")
        reward_title = data.get("reward_title", "")

        # Blacklist check
        if user_name.lower() in BLACKLIST:
            logger.info(f"Redemption ignored: {user_name} is blacklisted")
            return

        # Filtro per nome redemption (se configurato)
        required_name = CONFIG.get("redemption_name", "")
        if required_name and reward_title != required_name:
            logger.debug(f"Redemption '{reward_title}' ignorata (attesa: '{required_name}')")
            return

        if text:
            formatted_text = f"{user_name} says: {text}"
            logger.info(f"Processing redemption from {user_name}: '{formatted_text}'")
            await tts_service.submit_request(
                {
                    "text": formatted_text,
                    "ref_text": CONFIG["model"].get("ref_text", ""),
                }
            )

    twitch_service.on_redemption = on_redemption

    # Log resolved broadcaster info
    if twitch_service.broadcaster_name:
        logger.info(f"  Broadcaster: {twitch_service.broadcaster_name} (ID: {twitch_service.broadcaster_id})")

    yield

    # Shutdown
    logger.info("  Shutting down services...")
    await tts_service.stop_workers()
    await audio_service.stop()
    if hasattr(twitch_service, "eventsub") and twitch_service.eventsub:
        await twitch_service.disconnect()

    logger.info("  ✓ Server stopped")


app = FastAPI(
    title="Twitch Channel Points TTS",
    description="Local TTS service for Twitch channel points redemptions using VoxCPM2",
    version="3.0.0",  # Bump version per VoxCPM2
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Get validated CUDA state from lifespan (not raw torch.cuda.is_available())
        cuda_available = getattr(app.state, "cuda_available", False)
        cuda_devices = getattr(app.state, "cuda_devices", 0)

        # Get queue status safely
        queue_size = 0
        workers_active = 0
        model_type = "gpu_model"
        if hasattr(app.state, "tts_service") and app.state.tts_service:
            queue_size = app.state.tts_service.queue.qsize()
            workers_active = len(app.state.tts_service.worker_tasks)
            model_type = app.state.tts_service.model_type

        # CPU model doesn't need CUDA
        healthy = (cuda_available if model_type == "gpu_model" else True)

        return HealthResponse(
            status="ok" if healthy else "degraded",
            model_type=model_type,
            cuda_available=cuda_available,
            cuda_devices=cuda_devices,
            queue_size=queue_size,
            workers_active=workers_active,
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            model_type="unknown",
            cuda_available=False,
            cuda_devices=0,
            queue_size=0,
            workers_active=0,
        )


@app.post("/tts/speak", response_model=TTSResponse)
async def speak(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Generate TTS audio from text using VoxCPM2

    POST /tts/speak
    {
        "text": "Hello world",
        "voice_id": "optional_voice"
    }

    Returns file path for playback
    """
    if not hasattr(app.state, "tts_service"):
        raise HTTPException(status_code=500, detail="TTS service not initialized")

    try:
        tts = app.state.tts_service

        # Create request dict
        req_data = {
            "text": request.text,
            "voice_id": request.voice_id,
            "ref_audio": CONFIG["model"]["ref_audio_path"],
            "ref_text": CONFIG["model"].get("ref_text", ""),
        }

        # Submit to queue
        success = await tts.submit_request(req_data)

        if not success:
            raise HTTPException(
                status_code=503, detail="Queue is full - too many concurrent requests"
            )

        return TTSResponse(
            success=True,
            message="Audio generation started",
        )

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts/status")
async def get_queue_status():
    """Get current queue status"""
    if not hasattr(app.state, "tts_service"):
        return {"status": "not_initialized"}

    queue = app.state.tts_service.queue

    tts = app.state.tts_service
    return {
        "queue_size": queue.qsize(),
        "max_size": queue.maxsize,
        "active_workers": len(tts.worker_tasks),
        "is_running": tts._is_running,
        "model_type": tts.model_type,  # "gpu_model" or "cpu_model"
        "sample_rate": tts.sample_rate,
    }


@app.post("/twitch/connect")
async def connect_twitch():
    """Manually trigger Twitch connection (uses saved tokens + auto-resolved broadcaster)"""
    if not hasattr(app.state, "twitch_service"):
        raise HTTPException(status_code=500, detail="Twitch service not initialized")

    try:
        await app.state.twitch_service.connect()
        await app.state.twitch_service.authenticate_user()

        broadcaster_id = app.state.twitch_service.broadcaster_id
        if not broadcaster_id:
            raise HTTPException(
                status_code=400,
                detail="Broadcaster ID not resolved — ensure tokens are valid",
            )

        await app.state.twitch_service.listen_channel_points_redemption(broadcaster_id)

        return {
            "status": "connected",
            "broadcaster": app.state.twitch_service.broadcaster_name,
            "message": "Listening for redemptions...",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Twitch connection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/twitch/disconnect")
async def disconnect_twitch():
    """Disconnect from Twitch"""
    if not hasattr(app.state, "twitch_service"):
        raise HTTPException(status_code=500, detail="Twitch service not initialized")

    await app.state.twitch_service.disconnect()
    return {"status": "disconnected"}


@app.post("/twitch/reconnect")
async def reconnect_twitch():
    """Reconnect to Twitch using saved tokens (auto-resolves broadcaster)"""
    if not hasattr(app.state, "twitch_service"):
        raise HTTPException(status_code=500, detail="Twitch service not initialized")

    twitch_service = app.state.twitch_service

    try:
        await twitch_service.connect()
        await twitch_service.authenticate_user()

        if twitch_service.broadcaster_id:
            await twitch_service.listen_channel_points_redemption(twitch_service.broadcaster_id)
            return {
                "status": "reconnected",
                "broadcaster": twitch_service.broadcaster_name,
                "message": "Using saved tokens from token.json",
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/twitch/status")
async def get_twitch_status():
    """Get current Twitch connection status"""
    if not hasattr(app.state, "twitch_service"):
        return {"status": "not_initialized"}

    twitch_service = app.state.twitch_service
    eventsub = twitch_service.eventsub
    connected = (
        eventsub is not None
        and hasattr(eventsub, "_running")
        and eventsub._running
        if eventsub
        else False
    )
    return {
        "initialized": eventsub is not None,
        "connected": connected,
        "broadcaster": twitch_service.broadcaster_name,
        "broadcaster_id": twitch_service.broadcaster_id,
    }


@app.post("/twitch/auth/start")
async def start_auth():
    """
    Start the Device Code Flow for first-time or re-authentication.
    Responds immediately with user_code + verification_uri.
    On success the service auto-connects to Twitch.
    """
    if not hasattr(app.state, "twitch_service"):
        raise HTTPException(status_code=500, detail="Twitch service not initialized")

    twitch_service = app.state.twitch_service

    try:
        result = await twitch_service._start_device_flow_async()
        return {
            "status": "pending",
            "verification_uri": result["verification_uri"],
            "user_code": result["user_code"],
            "expires_in": result["expires_in"],
        }
    except Exception as e:
        logger.error(f"Failed to start DCF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/twitch/auth/status")
async def get_auth_status():
    """
    Poll the current DCF authorization status.
    Returns: idle | pending | success | expired | denied
    """
    if not hasattr(app.state, "twitch_service"):
        return {"status": "not_initialized"}

    twitch_service = app.state.twitch_service
    return {
        "auth_status": twitch_service._auth_status,
        "tokens_exist": TOKEN_PATH.exists(),
    }


if __name__ == "__main__":
    import uvicorn

    host = CONFIG.get("host", "127.0.0.1")
    port = CONFIG.get("port", 8000)

    logger.info(f"Server starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
