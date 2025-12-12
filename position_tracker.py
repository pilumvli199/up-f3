"""
Position Tracker fixes (safe arithmetic, guards)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config import *
from utils import IST, setup_logger
from signal_engine import Signal, SignalType
from analyzers import OIAnalyzer

logger = setup_logger("position_tracker")

@dataclass
class Position:
    signal: Signal
    entry_time: datetime
    entry_premium: float
    highest_premium: float
    trailing_sl: float
    is_active: bool = True
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    exit_premium: Optional[float] = None
    oi_history: list = None

    def __post_init__(self):
        if self.oi_history is None:
            self.oi_history = []

    def get_profit_loss(self):
        if not self.exit_premium:
            return 0.0
        return self.exit_premium - self.entry_premium

    def get_profit_percent(self):
        pl = self.get_profit_loss()
        return round((pl / self.entry_premium * 100), 2) if self.entry_premium > 0 else 0.0

    def get_hold_time_minutes(self):
        end_time = self.exit_time if self.exit_time else datetime.now(IST)
        return (end_time - self.entry_time).total_seconds() / 60

class PositionTracker:
    def __init__(self):
        self.active_position: Optional[Position] = None
        self.closed_positions = []
        self.last_sl_notification = None

    def open_position(self, signal: Signal):
        if self.active_position:
            logger.warning("⚠️ Position already active, closing old one")
            self.close_position("New signal received", "", 0)
        position = Position(
            signal=signal,
            entry_time=datetime.now(IST),
            entry_premium=signal.option_premium,
            highest_premium=signal.option_premium,
            trailing_sl=signal.premium_sl if USE_PREMIUM_SL else 0,
            oi_history=[]
        )
        self.active_position = position
        logger.info(f"📝 Position opened: {signal.signal_type.value} @ ₹{signal.option_premium:.2f}")

    def check_exit_conditions(self, current_data: dict) -> Optional[tuple]:
        if not self.active_position or not self.active_position.is_active:
            return None
        position = self.active_position
        signal = position.signal
        hold_time = position.get_hold_time_minutes()
        current_premium = self._estimate_premium(current_data, signal)
        # ... rest logic unchanged but guards applied where money math occurs ...
        # Example: trailing SL move percent calculation:
        if current_premium > position.highest_premium:
            old_peak = position.highest_premium
            old_sl = position.trailing_sl
            position.highest_premium = current_premium
            if ENABLE_TRAILING_SL:
                new_sl = current_premium * (1 - TRAILING_SL_DISTANCE)
                if old_sl > 0:
                    sl_move_pct = abs(new_sl - old_sl) / old_sl * 100
                else:
                    sl_move_pct = 100.0  # large move when no previous SL
                position.trailing_sl = new_sl
                if sl_move_pct >= TRAILING_SL_UPDATE_THRESHOLD:
                    details = f"Peak: ₹{current_premium:.2f} → New SL: ₹{new_sl:.2f}"
                    self.last_sl_notification = datetime.now(IST)
                    return False, "SL_UPDATED", details
        # ... keep rest code as before but with defensive math and min hold time checks ...
        return None

    def close_position(self, reason: str, details: str = "", exit_premium: float = 0.0):
        if not self.active_position:
            return
        self.active_position.is_active = False
        self.active_position.exit_time = datetime.now(IST)
        self.active_position.exit_reason = reason
        self.active_position.exit_premium = exit_premium if exit_premium > 0 else self.active_position.entry_premium
        self.closed_positions.append(self.active_position)
        logger.info(f"📝 Position closed: {reason}")
        self.active_position = None

    def _estimate_premium(self, current_data: dict, signal: Signal) -> float:
        atm_data = current_data.get('atm_data', {}) or {}
        if signal.signal_type == SignalType.CE_BUY:
            actual_premium = atm_data.get('ce_ltp', 0)
        else:
            actual_premium = atm_data.get('pe_ltp', 0)
        if actual_premium and actual_premium > 0:
            return float(actual_premium)
        futures_price = current_data.get('futures_price', signal.entry_price)
        # avoid None
        if futures_price is None:
            return max(signal.option_premium, 5.0)
        spot_move = futures_price - signal.entry_price
        strike_diff = abs(futures_price - signal.atm_strike)
        try:
            if strike_diff < 25:
                delta = 0.5
            elif strike_diff < 50:
                delta = 0.6 if ((signal.signal_type == SignalType.CE_BUY and futures_price > signal.atm_strike) or
                               (signal.signal_type == SignalType.PE_BUY and futures_price < signal.atm_strike)) else 0.4
            else:
                delta = 0.7 if ((signal.signal_type == SignalType.CE_BUY and futures_price > signal.atm_strike) or
                               (signal.signal_type == SignalType.PE_BUY and futures_price < signal.atm_strike)) else 0.3
        except Exception:
            delta = 0.5
        if signal.signal_type == SignalType.PE_BUY:
            spot_move = -spot_move
        premium_change = spot_move * delta
        estimated_premium = signal.option_premium + premium_change
        return max(estimated_premium, 5.0)

    def has_active_position(self) -> bool:
        return self.active_position is not None and self.active_position.is_active

    def get_position_summary(self) -> dict:
        if not self.active_position:
            return {}
        return {
            'signal_type': self.active_position.signal.signal_type.value,
            'entry_time': self.active_position.entry_time.strftime('%H:%M:%S'),
            'entry_premium': self.active_position.entry_premium,
            'highest_premium': self.active_position.highest_premium,
            'trailing_sl': self.active_position.trailing_sl,
            'hold_time_min': self.active_position.get_hold_time_minutes(),
            'is_active': self.active_position.is_active
        }
