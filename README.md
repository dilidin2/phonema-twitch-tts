# Phonema

<p align="center">
  <img src="img/image.png" width="480">
</p>

> *Phoneme* — the smallest unit of sound in a spoken language.

Real-time Text-to-Speech service for Twitch using **VoxCPM2**. Listens to channel point redemptions and speaks the user's message instantly via audio streaming.

## Features

- **Real-time streaming** — Audio plays while VoxCPM2 is still generating (no waiting)
- **Voice cloning** — Reference audio determines voice characteristics
- **Voice rotation** — Cycle through multiple voices (random or sequential mode)
- **Queue management** — Back-pressure controlled concurrent request handling
- **Auto-reconnect** — OAuth tokens persisted to `token.json` for seamless resumption
- **Cross-platform** — CPU, CUDA (NVIDIA), and ROCm (AMD) supported

## Architecture

```
Twitch EventSub ──► TwitchService ──► TTS Queue ──► VoxCPM2 Worker
                                                        │
                                                Audio Buffer
                                                        │
                                                  sounddevice ──► Speakers
```

One worker processes inference sequentially. Producer/consumer pattern streams chunks to the audio output with back-pressure control.

## Installation

### Clone the repo

```bash
git clone https://github.com/dilidin2/phonema-twitch-tts.git
cd phonema-twitch-tts
```

### 1. PyTorch (critical for performance)

VoxCPM2 requires PyTorch ≥ 2.5.0. Pick the build matching your hardware:

**NVIDIA GPU (CUDA 12.4–12.6):**
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**AMD GPU (ROCm 7.2):**
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

**Generic CPU:**
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Project dependencies

```bash
pip install -r requirements.txt
# or manually:
pip install voxcpm numpy fastapi uvicorn python-multipart twitchAPI pyyaml \
           python-dotenv loguru sounddevice soundfile
```

## Configuration

### Get Twitch API Credentials

1. Go to https://dev.twitch.tv/console
2. Click **"Register Your Application"**
3. Fill in:
   - **Name**: any name for your app
   - **OAuth Redirect URLs**: `http://localhost:17563`
   - **Category**: `Chat Bot`
   - **Client Type**: `Confidential`
4. Copy the **Client ID**
5. Click **"New Secret"** under Client Secret, then copy it

### Environment Variables (`.env`)

Copy `.env.example` to `.env` and fill in the values:

```ini
# Twitch API credentials (from https://dev.twitch.tv/console)
TWITCH_CLIENT_ID=your_client_id_here
TWITCH_CLIENT_SECRET=your_client_secret_here

# Bot account username (use your channel name if no separate bot account)
TWITCH_BOT_USERNAME=your_bot_username

# Numeric broadcaster ID (your channel's user ID, not username)
# Get it: https://www.streamweasels.com/tools/convert-twitch-username-to-user-id/
TWITCH_BROADCASTER_ID=123456789

# Server
PORT=8100
HOST=127.0.0.1

# Optional: override model path for custom VoxCPM2 deployment
# TTS_MODEL_PATH=/path/to/local/model

# Optional: force CPU even if GPU is available
# TTS_DEVICE=cpu
```

### Model Config (`config/tts_config.yaml`)

```yaml
model:
  pretrained_path: "openbmb/VoxCPM2"   # HuggingFace model ID or local path
  force_cpu: true                       # Set false if using GPU
  dtype: "bfloat16"                     # bfloat16 / float32
  inference_timesteps: 5                # Higher = better quality, slower
  sr: 48000                             # VoxCPM2 native sample rate
  ref_audio_path: "config/reference_voice.wav"

voice_rotation:
  mode: "random"                        # "random" | "sequential" | "disabled"
  voices_dir: "config/voices"
  voices:
    - "voice_a.wav"
    - "voice_b.wav"
```

### Setup Voice Files

1. Place at least one `.wav` file in the root `config/` directory named `reference_voice.wav`
2. For voice rotation, add additional `.wav` files to `config/voices/`
3. Reference audio should be 5-30 seconds of clear speech for best results

## Usage

### Start the server

```bash
python main.py
```

The server starts on port 8100 by default. Open Swagger docs at `http://localhost:8100/docs`.

### Connect to Twitch

First-time connection requires OAuth authentication:

```bash
curl -X POST http://localhost:8100/twitch/connect
```

This opens a browser for Twitch login. Tokens are saved to `token.json` and reused on restart.

For auto-connect on startup, set `TWITCH_BROADCASTER_ID` in `.env` — the service attempts connection on launch.

### API Endpoints

**TTS:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts/speak` | Generate speech from text |
| GET | `/tts/status` | Check queue status |

```bash
# Speak a message
curl -X POST http://localhost:8100/tts/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Twitch TTS!", "voice_id": null}'
```

**Twitch:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/twitch/connect` | Authenticate and start listening |
| POST | `/twitch/reconnect` | Reconnect using saved tokens |
| POST | `/twitch/disconnect` | Disconnect EventSub |
| GET | `/twitch/status` | Connection status |

**Health:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | CUDA status, queue size, worker count |

## License

MIT License.

VoxCPM2 model is licensed under Apache 2.0 (by OpenBMB). Respect their license when using model weights.
