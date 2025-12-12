"""
Alerts: Telegram Bot + MessageFormatter
- Async-friendly
- Graceful fallback if python-telegram-bot missing or credentials absent
- Exports: TelegramBot, MessageFormatter
"""
import logging
from datetime import datetime
from typing import Optional

# Try to import telegram (async usage assumed)
try:
    # If using python-telegram-bot v20+ the interfaces changed; we'll assume Bot.send_message is awaitable.
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False

from config import TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils import setup_logger

logger = setup_logger("alerts")

# -------------------- Telegram Bot --------------------
class TelegramBot:
    """Telegram notification service (async)."""

    def __init__(self):
        self.enabled = bool(TELEGRAM_ENABLED)
        self.bot: Optional[Bot] = None
        self.chat_id = TELEGRAM_CHAT_ID
        if self.enabled:
            if not TELEGRAM_AVAILABLE:
                logger.warning("⚠️ python-telegram-bot not installed; Telegram disabled")
                self.enabled = False
            elif not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                logger.warning("⚠️ Telegram credentials missing (TELEGRAM_BOT_TOKEN/CHAT_ID); Telegram disabled")
                self.enabled = False
            else:
                try:
                    # instantiate bot (await not required here)
                    self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
                    logger.info("✅ Telegram initialized")
                except Exception as e:
                    logger.error(f"❌ Telegram init failed: {e}")
                    self.enabled = False

    async def send(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send message to configured chat_id. Returns True on success."""
        if not self.enabled or not self.bot:
            logger.debug("Telegram is disabled or bot not initialized; skipping send")
            return False

        try:
            # Bot.send_message is awaited for async-compatible wrappers
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode=parse_mode)
            logger.debug("✅ Telegram message sent")
            return True
        except TelegramError as e:
            logger.error(f"❌ Telegram send failed (TelegramError): {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected Telegram send error: {e}")
            return False

    async def send_signal(self, message: str) -> bool:
        """Send entry signal with a header"""
        formatted = f"🔔 <b>TRADING SIGNAL</b>\n\n{message}"
        return await self.send(formatted)

    async def send_exit(self, message: str) -> bool:
        """Send exit signal with a header"""
        formatted = f"🚪 <b>EXIT SIGNAL</b>\n\n{message}"
        return await self.send(formatted)

    async def send_update(self, message: str) -> bool:
        """Generic update (no extra header)"""
        return await self.send(message)

    def is_enabled(self) -> bool:
        return self.enabled and (self.bot is not None)


# -------------------- Message Formatter --------------------
class MessageFormatter:
    """Format messages for Telegram (HTML)."""

    @staticmethod
    def _format_header(icon: str, title: str, extra: Optional[str] = None) -> str:
        header = f"{icon} <b>{title}</b>"
        if extra:
            header += f" — {extra}"
        header += "\n\n"
        return header

    @staticmethod
    def format_entry_signal(signal) -> str:
        """
        Format entry signal alert with enhanced info.
        Expects `signal` to have attributes used across your codebase (see main.py usage).
        """
        try:
            emoji = "📈" if getattr(signal, "signal_type").value == "CE_BUY" else "📉"
            expiry_note = " ⚡ <b>EXPIRY DAY</b>" if getattr(signal, "is_expiry_day", False) else ""
            ts = getattr(signal, "timestamp", datetime.now())
            ts_str = ts.strftime('%I:%M:%S %p') if hasattr(ts, "strftime") else str(ts)

            # OI strength emoji mapping
            oi_strength = getattr(signal, "oi_strength", "weak") or "weak"
            if oi_strength == 'strong':
                oi_emoji = "🔥"
            elif oi_strength == 'medium':
                oi_emoji = "💪"
            else:
                oi_emoji = "📊"

            # Safe attribute access with defaults
            entry_price = getattr(signal, "entry_price", 0.0)
            target_price = getattr(signal, "target_price", 0.0)
            stop_loss = getattr(signal, "stop_loss", 0.0)
            rr = getattr(signal, "get_rr_ratio", lambda: 0.0)()
            atm = getattr(signal, "atm_strike", "N/A")
            recommended = getattr(signal, "recommended_strike", atm)
            premium = getattr(signal, "option_premium", 0.0)
            premium_sl = getattr(signal, "premium_sl", 0.0)
            vwap = getattr(signal, "vwap", 0.0)
            vwap_distance = getattr(signal, "vwap_distance", 0.0)
            vwap_score = getattr(signal, "vwap_score", 0)
            atr = getattr(signal, "atr", 0.0)
            pcr = getattr(signal, "pcr", "N/A")
            oi_5m = getattr(signal, "oi_5m", 0.0)
            oi_15m = getattr(signal, "oi_15m", 0.0)
            atm_ce = getattr(signal, "atm_ce_change", 0.0)
            atm_pe = getattr(signal, "atm_pe_change", 0.0)
            volume_ratio = getattr(signal, "volume_ratio", 0.0)
            volume_spike = getattr(signal, "volume_spike", False)
            order_flow = getattr(signal, "order_flow", 0.0)
            confidence = getattr(signal, "confidence", 0)
            primary_checks = getattr(signal, "primary_checks", 0)
            bonus_checks = getattr(signal, "bonus_checks", 0)
            is_expiry = getattr(signal, "is_expiry_day", False)

            msg = (
                f"{emoji} <b>{getattr(signal, 'signal_type').value} SIGNAL</b>{expiry_note} ⏰ {ts_str}\n"
                f"💯 Confidence: <b>{confidence}%</b>\n"
                f"{oi_emoji} OI Strength: <b>{str(oi_strength).upper()}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "📊 <b>ENTRY DETAILS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Entry: ₹{entry_price:.2f}\n"
                f"Target: ₹{target_price:.2f} (+{abs(target_price - entry_price):.0f} pts)\n"
                f"Stop Loss: ₹{stop_loss:.2f} (-{abs(entry_price - stop_loss):.0f} pts)\n"
                f"R:R Ratio: <b>1:{rr:.2f}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "🎯 <b>OPTION INFO</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"ATM: {atm}\n"
                f"Strike: {recommended}\n"
                f"Premium: ₹{premium:.2f}\n"
                f"Premium SL: ₹{premium_sl:.2f}\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "📈 <b>ANALYSIS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"VWAP: ₹{vwap:.2f} ({vwap_distance:+.0f} pts)\n"
                f"VWAP Score: {vwap_score}/100 {'✅' if vwap_score >= 80 else '⚠️'}\n"
                f"ATR: {atr:.1f}\n"
                f"PCR: {pcr}\n"
                f"OI Changes: 5m: {oi_5m:+.1f}% 15m: {oi_15m:+.1f}%\n"
                f"ATM {atm}: CE: {atm_ce:+.1f}% PE: {atm_pe:+.1f}%\n"
                f"Volume: {volume_ratio:.1f}x {'🔥' if volume_spike else ''}\n"
                f"Order Flow: {order_flow:.2f}\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ Primary: {primary_checks}/3   🎁 Bonus: {bonus_checks}/9\n"
            )
            return msg
        except Exception as e:
            logger.error(f"❌ format_entry_signal error: {e}")
            return "⚠️ Error formatting entry signal"

    @staticmethod
    def format_exit_signal(position, reason: str, details: str) -> str:
        """Format exit signal alert. Expects Position object with `.signal` inside."""
        try:
            signal = getattr(position, "signal", None)
            exit_time = getattr(position, "exit_time", datetime.now())
            exit_time_str = exit_time.strftime('%I:%M:%S %p') if hasattr(exit_time, "strftime") else str(exit_time)
            entry_premium = getattr(position, "entry_premium", 0.0)
            exit_premium = getattr(position, "exit_premium", entry_premium)
            profit = exit_premium - entry_premium
            profit_pct = (profit / entry_premium * 100) if entry_premium > 0 else 0.0
            hold_time = getattr(position, "get_hold_time_minutes", lambda: 0)()

            profit_emoji = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"

            msg = (
                f"{getattr(signal, 'signal_type').value if signal else 'POSITION'} EXIT\n"
                f"⏰ Time: {exit_time_str}\n"
                f"📝 Reason: <b>{reason}</b>\n"
                f"{details}\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "💰 <b>P&L SUMMARY</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Entry: ₹{entry_premium:.2f}\n"
                f"Exit: ₹{exit_premium:.2f}\n"
                f"{profit_emoji} Profit: <b>₹{profit:+.2f} ({profit_pct:+.1f}%)</b>\n"
                f"⏱️ Hold Time: {hold_time:.0f} minutes\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "📊 <b>POSITION DETAILS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
            )
            if signal:
                msg += (
                    f"Strike: {getattr(signal, 'atm_strike', 'N/A')}\n"
                    f"Entry Price: ₹{getattr(signal, 'entry_price', 0.0):.2f}\n"
                    f"Target: ₹{getattr(signal, 'target_price', 0.0):.2f}\n"
                    f"SL: ₹{getattr(signal, 'stop_loss', 0.0):.2f}\n"
                )
            return msg
        except Exception as e:
            logger.error(f"❌ format_exit_signal error: {e}")
            return "⚠️ Error formatting exit signal"

    @staticmethod
    def format_position_update(position, current_premium: float) -> str:
        """Format a short periodic position update."""
        try:
            signal = getattr(position, "signal", None)
            entry_premium = getattr(position, "entry_premium", 0.0)
            highest = getattr(position, "highest_premium", entry_premium)
            trailing_sl = getattr(position, "trailing_sl", 0.0)
            unrealized_pl = current_premium - entry_premium
            unrealized_pct = (unrealized_pl / entry_premium * 100) if entry_premium > 0 else 0.0
            hold_time = getattr(position, "get_hold_time_minutes", lambda: 0)()

            msg = (
                f"📊 <b>Position Update</b>\n"
                f"Type: {getattr(signal, 'signal_type').value if signal else 'N/A'}\n"
                f"Entry: ₹{entry_premium:.2f}\n"
                f"Current: ₹{current_premium:.2f}\n"
                f"Peak: ₹{highest:.2f}\n"
                f"Trail SL: ₹{trailing_sl:.2f}\n"
                f"Unrealized P&L: ₹{unrealized_pl:+.2f} ({unrealized_pct:+.1f}%)\n"
                f"Hold Time: {hold_time:.0f} min\n"
            )
            return msg
        except Exception as e:
            logger.error(f"❌ format_position_update error: {e}")
            return "⚠️ Error formatting position update"
