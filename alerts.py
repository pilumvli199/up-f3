"""
Alerts: Telegram Bot & Message Formatting
"""

import logging

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

from config import TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils import setup_logger

logger = setup_logger("alerts")


# ==================== Telegram Bot ====================
class TelegramBot:
    """Telegram notification service"""
    
    def __init__(self):
        self.enabled = TELEGRAM_ENABLED
        self.bot = None
        self.chat_id = TELEGRAM_CHAT_ID
        
        if self.enabled:
            if not TELEGRAM_AVAILABLE:
                logger.warning("⚠️ python-telegram-bot not installed")
                self.enabled = False
            elif not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                logger.warning("⚠️ Telegram credentials missing")
                self.enabled = False
            else:
                try:
                    self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
                    logger.info("✅ Telegram initialized")
                except Exception as e:
                    logger.error(f"❌ Telegram init failed: {e}")
                    self.enabled = False
    
    async def send(self, message, parse_mode='HTML'):
        """Send message"""
        if not self.enabled or not self.bot:
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            return True
        except TelegramError as e:
            logger.error(f"❌ Telegram send failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return False
    
    async def send_signal(self, message):
        """Send entry signal alert"""
        formatted = f"🔔 <b>TRADING SIGNAL</b>\n\n{message}"
        return await self.send(formatted)
    
    async def send_exit(self, message):
        """Send exit alert"""
        formatted = f"🚪 <b>EXIT SIGNAL</b>\n\n{message}"
        return await self.send(formatted)
    
    async def send_update(self, message):
        """Send update"""
        formatted = f"📊 <b>Market Update</b>\n\n{message}"
        return await self.send(formatted)
    
    def is_enabled(self):
        return self.enabled and self.bot is not None


# ==================== Message Formatter ====================
class MessageFormatter:
    """Format Telegram messages"""
    
    @staticmethod
    def format_entry_signal(signal):
        """Format entry signal alert"""
        emoji = "📈" if signal.signal_type.value == "CE_BUY" else "📉"
        expiry = " ⚡ <b>EXPIRY DAY</b>" if signal.is_expiry_day else ""
        
        msg = f"""
{emoji} <b>{signal.signal_type.value} SIGNAL</b>{expiry}

⏰ {signal.timestamp.strftime('%I:%M:%S %p')}
💯 Confidence: <b>{signal.confidence}%</b>

━━━━━━━━━━━━━━━━━━━━
📊 <b>ENTRY DETAILS</b>
━━━━━━━━━━━━━━━━━━━━

Entry: ₹{signal.entry_price:.2f}
Target: ₹{signal.target_price:.2f} (+{abs(signal.target_price - signal.entry_price):.0f} pts)
Stop Loss: ₹{signal.stop_loss:.2f} (-{abs(signal.entry_price - signal.stop_loss):.0f} pts)

R:R Ratio: <b>1:{signal.get_rr_ratio():.2f}</b>

━━━━━━━━━━━━━━━━━━━━
🎯 <b>OPTION INFO</b>
━━━━━━━━━━━━━━━━━━━━

ATM: {signal.atm_strike}
Strike: {signal.recommended_strike}
Premium: ₹{signal.option_premium:.2f}
Premium SL: ₹{signal.premium_sl:.2f}

━━━━━━━━━━━━━━━━━━━━
📈 <b>ANALYSIS</b>
━━━━━━━━━━━━━━━━━━━━

VWAP: ₹{signal.vwap:.2f}
ATR: {signal.atr:.1f}
PCR: {signal.pcr}

OI Changes:
  5m:  {signal.oi_5m:+.1f}%
  15m: {signal.oi_15m:+.1f}%

ATM {signal.atm_strike}:
  CE: {signal.atm_ce_change:+.1f}%
  PE: {signal.atm_pe_change:+.1f}%

Volume: {signal.volume_ratio:.1f}x {'✅' if signal.volume_spike else ''}
Order Flow: {signal.order_flow:.2f}

━━━━━━━━━━━━━━━━━━━━
✅ Primary: {signal.primary_checks}/3
🎁 Bonus: {signal.bonus_checks}/8
"""
        return msg
    
    @staticmethod
    def format_exit_signal(position, reason, details):
        """Format exit signal alert"""
        signal = position.signal
        profit = position.get_profit_loss()
        profit_pct = position.get_profit_percent()
        hold_time = position.get_hold_time_minutes()
        
        profit_emoji = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
        
        msg = f"""
{signal.signal_type.value} EXIT

⏰ Time: {position.exit_time.strftime('%I:%M:%S %p')}
📝 Reason: <b>{reason}</b>
{details}

━━━━━━━━━━━━━━━━━━━━
💰 <b>P&L SUMMARY</b>
━━━━━━━━━━━━━━━━━━━━

Entry: ₹{position.entry_premium:.2f}
Exit: ₹{position.exit_premium:.2f}

{profit_emoji} Profit: <b>₹{profit:+.2f} ({profit_pct:+.1f}%)</b>
⏱️ Hold Time: {hold_time:.0f} minutes

━━━━━━━━━━━━━━━━━━━━
📊 <b>POSITION DETAILS</b>
━━━━━━━━━━━━━━━━━━━━

Strike: {signal.atm_strike}
Entry Price: ₹{signal.entry_price:.2f}
Target: ₹{signal.target_price:.2f}
SL: ₹{signal.stop_loss:.2f}
"""
        return msg
    
    @staticmethod
    def format_position_update(position, current_premium):
        """Format position update"""
        signal = position.signal
        unrealized_pl = current_premium - position.entry_premium
        unrealized_pct = (unrealized_pl / position.entry_premium * 100) if position.entry_premium > 0 else 0
        
        msg = f"""
📊 <b>Position Update</b>

Type: {signal.signal_type.value}
Entry: ₹{position.entry_premium:.2f}
Current: ₹{current_premium:.2f}
Peak: ₹{position.highest_premium:.2f}
Trail SL: ₹{position.trailing_sl:.2f}

Unrealized P&L: ₹{unrealized_pl:+.2f} ({unrealized_pct:+.1f}%)
Hold Time: {position.get_hold_time_minutes():.0f} min
"""
        return msg

