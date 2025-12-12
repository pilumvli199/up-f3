"""
Alerts: Telegram Bot & Message Formatting (fixed)
- robust handling when python-telegram-bot is not installed or sync/async mismatch
- safe send() wrapper with retries and clear logging
"""

import logging
import asyncio
from datetime import datetime

try:
    # we don't assume a particular major version; we only import Bot if available
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except Exception:
    Bot = None
    TelegramError = Exception
    TELEGRAM_AVAILABLE = False

from config import TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils import setup_logger

logger = setup_logger("alerts")


class TelegramBot:
    """Telegram notification service (robust, minimal assumptions)"""

    def __init__(self):
        self.enabled = TELEGRAM_ENABLED
        self.bot = None
        self.chat_id = TELEGRAM_CHAT_ID
        if self.enabled:
            if not TELEGRAM_AVAILABLE or Bot is None:
                logger.warning("⚠️ python-telegram-bot not installed or import failed — disabling Telegram")
                self.enabled = False
            elif not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                logger.warning("⚠️ Telegram credentials missing — disabling Telegram")
                self.enabled = False
            else:
                try:
                    # Bot is sync in many versions; using it in an async environment with run_in_executor
                    self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
                    logger.info("✅ Telegram initialized")
                except Exception as e:
                    logger.error(f"❌ Telegram init failed: {e}")
                    self.enabled = False

    async def _async_send(self, text, parse_mode='HTML'):
        """Internal send: run sync send_message in executor if necessary."""
        if not self.bot:
            return False

        # try 2 attempts
        last_exc = None
        for attempt in range(2):
            try:
                # Many Bot implementations are sync; we run in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=parse_mode)
                )
                return True
            except TelegramError as e:
                last_exc = e
                logger.error(f"❌ Telegram send failed (TelegramError) attempt {attempt+1}: {e}")
                await asyncio.sleep(0.5)
            except Exception as e:
                last_exc = e
                logger.error(f"❌ Telegram send failed (Exception) attempt {attempt+1}: {e}")
                await asyncio.sleep(0.5)

        logger.error(f"❌ Telegram send ultimately failed: {last_exc}")
        return False

    async def send(self, message, parse_mode='HTML'):
        """Public send wrapper"""
        if not self.enabled or not self.bot:
            return False
        # guard: message must be string and non-empty
        if not message:
            return False
        # Trim message to reasonable size for Telegram
        if len(message) > 4000:
            message = message[:3995] + "..."
        return await self._async_send(message, parse_mode=parse_mode)

    async def send_signal(self, message):
        formatted = f"🔔 <b>TRADING SIGNAL</b>\n\n{message}"
        return await self.send(formatted)

    async def send_exit(self, message):
        formatted = f"🚪 <b>EXIT SIGNAL</b>\n\n{message}"
        return await self.send(formatted)

    async def send_update(self, message):
        formatted = f"ℹ️ <b>UPDATE</b>\n\n{message}"
        return await self.send(formatted)

    def is_enabled(self):
        return self.enabled and self.bot is not None
