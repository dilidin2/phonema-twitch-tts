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


def load_config():
    with open("config/tts_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["TWITCH_CLIENT_ID"] = os.getenv("TWITCH_CLIENT_ID", "")
    config["TWITCH_CLIENT_SECRET"] = os.getenv("TWITCH_CLIENT_SECRET", "")
    config["TWITCH_BOT_USERNAME"] = os.getenv("TWITCH_BOT_USERNAME", "")
    config["TWITCH_BOT_OAUTH_TOKEN"] = os.getenv("TWITCH_BOT_OAUTH_TOKEN", "")

    return config


CONFIG = load_config()

# Import services
from services.tts_service import TTSService
from services.twitch_service import TwitchService
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
    cuda_available: bool
    cuda_devices: int
    queue_size: int
    workers_active: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 60)
    logger.info("  Starting Twitch Channel Points TTS Server")
    logger.info("  Model: VoxCPM2 (BMB/ModelScope)")
    logger.info("=" * 60)

    # Check CUDA availability
    try:
        import torch

        # If we are hiding GPUs, don't even try to query them
        gpu_disabilitata = (
            os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            or os.environ.get("HIP_VISIBLE_DEVICES") == "-1"
        )

        valid_gpus = []
        cuda_available = False

        if not gpu_disabilitata and torch.cuda.is_available():
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

    audio_service = AudioOutputService(
        method=CONFIG.get("AUDIO_OUTPUT_METHOD", "direct")
    )
    tts_service = TTSService(CONFIG, audio_service=audio_service)
    await tts_service.start_workers(num_workers=1)

    twitch_service = TwitchService(CONFIG)

    # Auto-connect to Twitch using saved tokens from token.json
    logger.info("  Auto-connecting to Twitch...")
    try:
        await twitch_service.connect()
        await twitch_service.authenticate_user()

        broadcaster_id = os.getenv("TWITCH_BROADCASTER_ID")
        if broadcaster_id:
            await twitch_service.listen_channel_points_redemption(broadcaster_id)
            logger.info("  ✓ Twitch connected automatically (saved tokens loaded)")
    except Exception as e:
        logger.warning(f"Auto-connect failed: {e}")
        logger.warning(
            "  Run 'curl -X POST http://localhost:8100/twitch/connect' to authenticate"
        )

    logger.info("  ✓ All services initialized")

    app.state.audio_service = audio_service
    app.state.tts_service = tts_service
    app.state.twitch_service = twitch_service

    # Connect callback from TwitchService to TTSService
    async def on_redemption(data):
        """Handle redemption events - submit to TTS queue"""
        text = data.get("user_input", "") or ""
        user_name = data.get("user_name", "A user")

        if text:
            formatted_text = f"{user_name} says: {text}"

            logger.info(f"  Processing redemption from {user_name}: '{formatted_text}'")
            await tts_service.submit_request(
                {
                    "text": formatted_text,
                    "ref_text": CONFIG["model"].get("ref_text", ""),
                }
            )

    twitch_service.on_redemption = on_redemption

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        import torch

        cuda_available = torch.cuda.is_available()

        # Get queue status safely
        queue_size = 0
        workers_active = 0
        if hasattr(app.state, "tts_service") and app.state.tts_service:
            queue_size = app.state.tts_service.queue.qsize()
            workers_active = len(app.state.tts_service.worker_tasks)

        return HealthResponse(
            status="ok" if cuda_available else "degraded",
            cuda_available=cuda_available,
            cuda_devices=torch.cuda.device_count(),
            queue_size=queue_size,
            workers_active=workers_active,
        )
    except Exception as e:
        return HealthResponse(
            status="error",
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

    return {
        "queue_size": queue.qsize(),
        "max_size": queue.maxsize,
        "active_workers": len(app.state.tts_service.worker_tasks),
        "is_running": app.state.tts_service._is_running,
        "model": "VoxCPM2",  # Info aggiuntiva
    }


@app.post("/twitch/connect")
async def connect_twitch():
    """Manually trigger Twitch connection"""
    if not hasattr(app.state, "twitch_service"):
        raise HTTPException(status_code=500, detail="Twitch service not initialized")

    try:
        # First connect EventSub
        await app.state.twitch_service.connect()

        # Then authenticate user for OAuth flow
        await app.state.twitch_service.authenticate_user()

        # Get broadcaster_id (numeric ID, not username) from environment
        broadcaster_id = os.getenv("TWITCH_BROADCASTER_ID")
        if not broadcaster_id:
            raise HTTPException(
                status_code=400,
                detail="TWITCH_BROADCASTER_ID not set in .env file. Get it from https://www.streamweasels.com/tools/convert-twitch-username-to-user-id/",
            )

        # Start listening for redemptions
        await app.state.twitch_service.listen_channel_points_redemption(broadcaster_id)

        return {"status": "connected", "message": "Listening for redemptions..."}

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
    """Reconnect to Twitch using saved tokens"""
    if not hasattr(app.state, "twitch_service"):
        raise HTTPException(status_code=500, detail="Twitch service not initialized")

    twitch_service = app.state.twitch_service

    try:
        await twitch_service.connect()
        await twitch_service.authenticate_user()

        broadcaster_id = os.getenv("TWITCH_BROADCASTER_ID")
        if broadcaster_id:
            await twitch_service.listen_channel_points_redemption(broadcaster_id)
            return {"status": "reconnected", "message": "Using saved tokens from token.json"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/twitch/status")
async def get_twitch_status():
    """Get current Twitch connection status"""
    if not hasattr(app.state, "twitch_service"):
        return {"status": "not_initialized"}

    eventsub = app.state.twitch_service.eventsub
    return {
        "initialized": eventsub is not None,
        "connected": eventsub is not None
        and hasattr(eventsub, "_running")
        and eventsub._running
        if eventsub
        else False,
    }


@app.get("/output/list")
async def list_output_files():
    """List recently generated audio files"""
    import glob

    output_dir = CONFIG["output"]["directory"]
    files = sorted(glob.glob(f"{output_dir}/*.wav"), reverse=True)

    return {
        "count": len(files),
        "files": [os.path.basename(f) for f in files[:20]],  # Last 20 files
    }


if __name__ == "__main__":
    import uvicorn

    host = CONFIG.get("host", "0.0.0.0")
    port = CONFIG.get("port", 8000)

    logger.info(f"Server starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
