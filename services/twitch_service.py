"""
Twitch Service - WebSocket integration with Twitch API
Handles real-time Channel Points redemption events using EventSub
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Callable
from loguru import logger
import os

# Import twitchAPI v4 components
try:
    from twitchAPI.twitch import Twitch
    from twitchAPI.oauth import UserAuthenticator
    from twitchAPI.eventsub.websocket import EventSubWebsocket
    from twitchAPI.type import AuthScope
except ImportError:
    raise ImportError("Install twitchAPI: pip install 'twitchAPI>=4.5.0'")

TOKEN_PATH = Path("token.json")
REQUIRED_SCOPES = [AuthScope.CHANNEL_READ_REDEMPTIONS]


def _save_tokens(token: str, refresh_token: str) -> None:
    with open(TOKEN_PATH, "w") as f:
        json.dump({"token": token, "refresh": refresh_token}, f)
    logger.debug("Tokens saved to token.json")


class TwitchService:
    """Manages EventSub connection to Twitch for real-time events"""

    def __init__(self, config: dict):
        self.config = config
        self.client_id = config.get("TWITCH_CLIENT_ID", "")
        self.client_secret = config.get("TWITCH_CLIENT_SECRET", "")

        self.twitch: Optional[Twitch] = None
        self.eventsub: Optional[EventSubWebsocket] = None
        self.on_redemption: Optional[Callable] = None

    # ── Auth helpers ──────────────────────────────────────────────────────────

    async def _user_auth_refresh_callback(self, token: str, refresh_token: str) -> None:
        """
        Called automatically by twitchAPI whenever the OAuth token is refreshed.
        Persisting new tokens here prevents silent expiry after ~4 hours.
        """
        logger.info("OAuth token refreshed — saving new tokens to token.json")
        _save_tokens(token, refresh_token)

    async def _do_browser_auth(self) -> tuple[str, str]:
        """Open browser for first-time or re-authentication, return (token, refresh)."""
        logger.info("Opening browser for OAuth authentication...")
        auth = UserAuthenticator(
            self.twitch,
            REQUIRED_SCOPES,
            force_verify=False,
        )
        token, refresh_token = await auth.authenticate()
        return token, refresh_token

    # ── Public API ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        logger.info("Initializing Twitch client...")
        self.twitch = await Twitch(self.client_id, self.client_secret)

        # Register the refresh callback BEFORE setting any token, so the
        # library can transparently persist renewed tokens on every refresh.
        self.twitch.user_auth_refresh_callback = self._user_auth_refresh_callback

        if TOKEN_PATH.exists():
            try:
                with open(TOKEN_PATH, "r") as f:
                    creds = json.load(f)
                logger.info("Loading existing tokens from token.json...")
                await self.twitch.set_user_authentication(
                    creds["token"],
                    REQUIRED_SCOPES,
                    creds["refresh"],
                )
                logger.info("✓ Tokens loaded from token.json")
            except Exception as e:
                logger.warning(f"Failed to load tokens ({e}) — falling back to browser auth")
                token, refresh_token = await self._do_browser_auth()
                await self.twitch.set_user_authentication(token, REQUIRED_SCOPES, refresh_token)
                _save_tokens(token, refresh_token)
        else:
            logger.info("No token.json found — opening browser for first-time authentication...")
            token, refresh_token = await self._do_browser_auth()
            await self.twitch.set_user_authentication(token, REQUIRED_SCOPES, refresh_token)
            _save_tokens(token, refresh_token)

        logger.info("Twitch client initialized")

    async def reauthenticate_if_needed(self) -> None:
        """Re-authenticate via browser (e.g. after a 401)."""
        logger.warning("Re-authenticating via browser...")
        token, refresh_token = await self._do_browser_auth()
        await self.twitch.set_user_authentication(token, REQUIRED_SCOPES, refresh_token)
        _save_tokens(token, refresh_token)
        logger.info("✓ Re-authenticated successfully")

    async def authenticate_user(self) -> None:
        """Create the EventSub WebSocket client (call after connect())."""
        logger.info("Creating EventSub client...")
        self.eventsub = EventSubWebsocket(twitch=self.twitch)
        logger.info("EventSub client created")

    async def listen_channel_points_redemption(self, broadcaster_id: str) -> None:
        if not self.eventsub:
            raise RuntimeError("Call authenticate_user() first to create EventSub")

        # Define the callback BEFORE starting, so it's always in scope.
        async def redemption_callback(data):
            await self._handle_redemption(data)

        async def _subscribe() -> None:
            await self.eventsub.listen_channel_points_custom_reward_redemption_add(
                broadcaster_user_id=broadcaster_id,
                callback=redemption_callback,
            )

        try:
            logger.info(f"Starting EventSub listener for broadcaster: {broadcaster_id}")
            self.eventsub.start()
            await _subscribe()
            logger.info("✓ Listening for channel point redemptions via EventSub")

        except Exception as e:
            error_str = str(e)
            if any(k in error_str for k in ("401", "Unauthorized", "needs user authentication")):
                logger.warning("Auth error during EventSub subscribe — re-authenticating...")
                await self.reauthenticate_if_needed()
                # EventSub was already started; just re-subscribe.
                await _subscribe()
                logger.info("✓ Re-subscribed after re-authentication")
            else:
                logger.error(f"Failed to start redemption listener: {e}")
                raise

    # ── Internal handlers ─────────────────────────────────────────────────────

    async def _handle_redemption(self, data) -> None:
        try:
            user_input = data.event.user_input
            user_id = data.event.user_id
            user_name = data.event.user_name or "Qualcuno"

            logger.info(
                f"💰 Redemption from {user_name} ({user_id}): '{user_input}'"
            )

            if self.on_redemption:
                await self.on_redemption(
                    {
                        "user_input": user_input,
                        "user_id": user_id,
                        "user_name": user_name,
                    }
                )

        except Exception as e:
            logger.error(f"Error handling redemption: {e}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def disconnect(self) -> None:
        if self.eventsub:
            logger.info("Stopping EventSub client...")
            await self.eventsub.stop()
            self.eventsub = None
            logger.info("EventSub disconnected")

    async def reconnect(self, max_retries: int = 5) -> bool:
        if self.eventsub:
            return True  # Already connected

        for attempt in range(max_retries):
            try:
                await self.connect()
                await self.authenticate_user()
                return True
            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Reconnection attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

        logger.error("Max reconnection attempts reached!")
        return False
