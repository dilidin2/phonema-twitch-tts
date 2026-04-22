"""
Configuration Module
Loads and manages application configuration
"""

import os
from typing import Dict, Any
from loguru import logger


def get_config() -> Dict[str, Any]:
    """Get merged configuration from YAML and environment variables"""

    # Load YAML config
    config_path = "config/tts_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_config = load_yaml(f.read())
    else:
        logger.warning(f"Config file not found: {config_path}")
        yaml_config = {}

    # Load environment variables
    env_config = get_env_config()

    # Merge configs (env overrides YAML)
    merged_config = merge_configs(yaml_config, env_config)

    return merged_config


def load_yaml(content: str) -> Dict[str, Any]:
    """Load YAML content"""
    try:
        import yaml

        return yaml.safe_load(content) or {}
    except ImportError:
        logger.error("PyYAML not installed! Install with: pip install pyyaml")
        raise


def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables"""
    config = {}

    # Twitch config
    config["TWITCH_CLIENT_ID"] = os.getenv("TWITCH_CLIENT_ID", "")
    config["TWITCH_CLIENT_SECRET"] = os.getenv("TWITCH_CLIENT_SECRET", "")
    config["TWITCH_BOT_USERNAME"] = os.getenv("TWITCH_BOT_USERNAME", "")
    config["TWITCH_BOT_OAUTH_TOKEN"] = os.getenv("TWITCH_BOT_OAUTH_TOKEN", "")

    # TTS config
    config["TTS_MODEL_PATH"] = os.getenv(
        "TTS_MODEL_PATH", "openbmb/VoxCPM2"
    )
    config["TTS_DEVICE"] = os.getenv("TTS_DEVICE", "cuda:0")
    config["TTS_DTYPE"] = os.getenv("TTS_DTYPE", "bfloat16")
    config["REF_AUDIO_PATH"] = os.getenv("REF_AUDIO_PATH", "config/reference_voice.wav")
    config["REF_TEXT"] = os.getenv("REF_TEXT", "Hello sample text")

    # Output config
    config["OUTPUT_DIRECTORY"] = os.getenv("OUTPUT_DIRECTORY", "output/tts_files")
    config["AUDIO_FORMAT"] = os.getenv("AUDIO_FORMAT", "wav")
    config["SAMPLE_RATE"] = int(os.getenv("SAMPLE_RATE", "44100"))

    # Server config
    config["PORT"] = int(os.getenv("PORT", "8000"))
    config["HOST"] = os.getenv("HOST", "0.0.0.0")
    config["MAX_QUEUE_SIZE"] = int(os.getenv("MAX_QUEUE_SIZE", "10"))
    config["REQUEST_TIMEOUT"] = int(os.getenv("REQUEST_TIMEOUT", "30"))

    # Audio output method
    config["AUDIO_OUTPUT_METHOD"] = os.getenv("AUDIO_OUTPUT_METHOD", "direct")

    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge two configs"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
