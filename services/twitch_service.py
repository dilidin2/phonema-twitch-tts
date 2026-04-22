"""
Twitch Service - WebSocket integration with Twitch API
Handles real-time Channel Points redemption events using EventSub
"""

import asyncio
from pathlib import Path, PurePath
from typing import Optional, Callable
from loguru import logger
import os

# Import twitchAPI v4 components
try:
    from twitchAPI.twitch import Twitch
    from twitchAPI.oauth import UserAuthenticator, UserAuthenticationStorageHelper
    from twitchAPI.eventsub.websocket import EventSubWebsocket
    from twitchAPI.type import AuthScope
except ImportError:
    raise ImportError("Install twitchAPI: pip install 'twitchAPI>=4.5.0'")


class TwitchService:
    """Manages EventSub connection to Twitch for real-time events"""

    def __init__(self, config: dict):
        self.config = config
        self.client_id = config.get("TWITCH_CLIENT_ID", "")
        self.client_secret = config.get("TWITCH_CLIENT_SECRET", "")
        self.oauth_token = config.get("TWITCH_BOT_OAUTH_TOKEN", "")

        # Twitch API client instance
        self.twitch: Optional[Twitch] = None

        # EventSub WebSocket client instance
        self.eventsub: Optional[EventSubWebsocket] = None

        # Callback for redemption events
        self.on_redemption: Optional[Callable] = None

    async def connect(self):
        """Initialize Twitch client and load existing tokens from token.json if available"""
        logger.info("Initializing Twitch client...")

        # Initialize twitch API client first
        self.twitch = await Twitch(self.client_id, self.client_secret)

        # Try to load existing tokens from token.json
        storage_path = Path("token.json")
        if os.path.exists(storage_path):
            import json
            try:
                with open(storage_path, 'r') as f:
                    creds = json.load(f)

                logger.info("Loading existing tokens from token.json...")
                await self.twitch.set_user_authentication(
                    creds['token'],
                    [AuthScope.CHANNEL_READ_REDEMPTIONS],
                    creds['refresh']
                )
                logger.info("✓ Tokens loaded successfully from token.json")
            except Exception as e:
                logger.warning(f"Failed to load tokens: {e}")

        logger.info("Twitch client initialized")

    async def reauthenticate_if_needed(self):
        """Re-authenticate using browser if tokens are invalid/expired"""
        logger.warning("Token expired or invalid - opening browser for re-authentication...")

        auth = UserAuthenticator(
            self.twitch,
            [AuthScope.CHANNEL_READ_REDEMPTIONS],
            force_verify=False
        )
        token, refresh_token = await auth.authenticate()
        await self.twitch.set_user_authentication(
            token=token,
            scope=[AuthScope.CHANNEL_READ_REDEMPTIONS],
            refresh_token=refresh_token,
        )

        # Save new tokens to file
        import json
        with open("token.json", 'w') as f:
            json.dump({'token': token, 'refresh': refresh_token}, f)

        logger.info("✓ Re-authenticated successfully")

    async def authenticate_user(self):
        """Creates EventSub client after tokens are loaded in connect()."""
        logger.info("Creating EventSub client...")

        # Creates EventSub AFTER authentication (tokens already loaded in connect())
        self.eventsub = EventSubWebsocket(twitch=self.twitch)
        logger.info("EventSub client created")

    async def listen_channel_points_redemption(self, broadcaster_id: str):
        if not self.eventsub:
            raise RuntimeError("Call authenticate_user() first to create EventSub")

        try:
            logger.info(f"Setting up EventSub listener for channel: {broadcaster_id}")
            self.eventsub.start()

            async def redemption_callback(data):
                await self._handle_redemption(data)

            await self.eventsub.listen_channel_points_custom_reward_redemption_add(
                broadcaster_user_id=broadcaster_id,
                callback=redemption_callback,
            )

            logger.info("Listening for redemptions via EventSub...")

        except Exception as e:
            # Check if it's an authentication error
            if "401" in str(e) or "Unauthorized" in str(e):
                logger.warning("Authentication failed - tokens may be expired")
                await self.reauthenticate_if_needed()
                # Retry after re-authentication
                self.eventsub.start()
                await self.eventsub.listen_channel_points_custom_reward_redemption_add(
                    broadcaster_user_id=broadcaster_id,
                    callback=redemption_callback,
                )
            else:
                logger.error(f"Failed to start redemption listener: {e}")
                raise

    async def _handle_redemption(self, data):
        try:
            user_input = data.event.user_input
            user_id = data.event.user_id

            logger.info(f"💰 Redemption received from {user_id}: '{user_input}'")

            if self.on_redemption:
                await self.on_redemption(
                    {
                        "user_input": user_input,
                        "user_id": user_id,
                    }
                )

        except Exception as e:
            logger.error(f"Error handling redemption: {e}")

    async def disconnect(self):
        """Disconnect from EventSub"""
        if self.eventsub:
            logger.info("Stopping EventSub client...")
            await self.eventsub.stop()
            self.eventsub = None
            logger.info("EventSub disconnected")

    async def reconnect(self, max_retries: int = 5):
        """Reconnect with exponential backoff"""
        if not self.eventsub:
            for attempt in range(max_retries):
                try:
                    await self.connect()
                    await self.authenticate_user()
                    return True

                except Exception as e:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Reconnection attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)

            logger.error("Max reconnection attempts reached!")
            return False

        return True
