
"""
Position Tracker: Track active positions & monitor exit conditions
Alert-based exit signals (no auto-execution)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config import *
from utils import IST, setup_logger
from signal_engine import Signal, SignalType

logger = setup_logger("position_tracker")


# ==================== Position Model ====================
@dataclass
class Position:
    """Active position tracking"""
    signal: Signal
    entry_time: datetime
    entry_premium: float
    highest_premium: float
    trailing_sl: float
    is_active: bool = True
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    exit_premium: Optional[float] = None
    
    def get_profit_loss(self):
        """Calculate P&L"""
        if not self.exit_premium:
            return 0.0
        return self.exit_premium - self.entry_premium
    
    def get_profit_percent(self):
        """Calculate P&L %"""
        pl = self.get_profit_loss()
        return round((pl / self.entry_premium * 100), 2) if self.entry_premium > 0 else 0.0
    
    def get_hold_time_minutes(self):
        """Get hold time in minutes"""
        end_time = self.exit_time if self.exit_time else datetime.now(IST)
        return (end_time - self.entry_time).total_seconds() / 60


# ==================== Position Tracker ====================
class PositionTracker:
    """Track active positions and generate exit alerts"""
    
    def __init__(self):
        self.active_position: Optional[Position] = None
        self.closed_positions = []
    
    def open_position(self, signal: Signal):
        """Open new position from signal"""
        if self.active_position:
            logger.warning("⚠️ Position already active, closing old one")
            self.close_position("New signal received")
        
        position = Position(
            signal=signal,
            entry_time=datetime.now(IST),
            entry_premium=signal.option_premium,
            highest_premium=signal.option_premium,
            trailing_sl=signal.premium_sl if USE_PREMIUM_SL else 0
        )
        
        self.active_position = position
        logger.info(f"📝 Position opened: {signal.signal_type.value} @ ₹{signal.option_premium:.2f}")
    
    def check_exit_conditions(self, current_data: dict) -> Optional[tuple]:
        """
        Check if position should exit
        Returns: (should_exit, reason, details) or None
        
        current_data = {
            'ce_oi_5m': float,
            'pe_oi_5m': float,
            'volume_ratio': float,
            'candle_data': dict,
            'futures_price': float,
            'atm_data': dict
        }
        """
        if not self.active_position or not self.active_position.is_active:
            return None
        
        position = self.active_position
        signal = position.signal
        
        # Get current premium (estimated from futures movement)
        current_premium = self._estimate_premium(current_data, signal)
        
        # Update highest premium for trailing
        if current_premium > position.highest_premium:
            position.highest_premium = current_premium
            if ENABLE_TRAILING_SL:
                position.trailing_sl = current_premium * (1 - TRAILING_SL_DISTANCE)
        
        # Exit Check 1: OI Reversal
        if signal.signal_type == SignalType.CE_BUY:
            ce_oi = current_data.get('ce_oi_5m', 0)
            if ce_oi > EXIT_OI_REVERSAL_THRESHOLD:
                return True, "OI Reversal", f"CE OI building: {ce_oi:+.1f}%"
        
        elif signal.signal_type == SignalType.PE_BUY:
            pe_oi = current_data.get('pe_oi_5m', 0)
            if pe_oi > EXIT_OI_REVERSAL_THRESHOLD:
                return True, "OI Reversal", f"PE OI building: {pe_oi:+.1f}%"
        
        # Exit Check 2: Volume Dry
        volume_ratio = current_data.get('volume_ratio', 1.0)
        if volume_ratio < EXIT_VOLUME_DRY_THRESHOLD:
            return True, "Volume Dried", f"Volume ratio: {volume_ratio:.1f}x"
        
        # Exit Check 3: Premium Drop (from peak)
        premium_drop_pct = ((position.highest_premium - current_premium) / 
                           position.highest_premium * 100) if position.highest_premium > 0 else 0
        
        if premium_drop_pct >= EXIT_PREMIUM_DROP_PERCENT:
            return True, "Premium Drop", f"Down {premium_drop_pct:.1f}% from peak"
        
        # Exit Check 4: Trailing SL Hit
        if ENABLE_TRAILING_SL and current_premium < position.trailing_sl:
            profit = current_premium - position.entry_premium
            return True, "Trailing SL Hit", f"Locked profit: ₹{profit:.2f}"
        
        # Exit Check 5: Candle Rejection
        candle = current_data.get('candle_data', {})
        if candle.get('rejection'):
            rejection_type = candle.get('rejection_type')
            
            # Bullish position + upper rejection = exit
            if signal.signal_type == SignalType.CE_BUY and rejection_type == 'upper':
                return True, "Candle Rejection", "Long upper wick at resistance"
            
            # Bearish position + lower rejection = exit
            elif signal.signal_type == SignalType.PE_BUY and rejection_type == 'lower':
                return True, "Candle Rejection", "Long lower wick at support"
        
        # Exit Check 6: Time-based (Market close)
        current_time = datetime.now(IST).time()
        if current_time >= time(15, 15):
            return True, "Market Closing", "Exiting before market close"
        
        # No exit condition met
        return None
    
    def close_position(self, reason: str, details: str = "", exit_premium: float = 0.0):
        """Close active position"""
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
        """
        Estimate option premium from futures movement
        Uses Delta approximation (0.5 for ATM)
        """
        futures_price = current_data.get('futures_price', signal.entry_price)
        spot_move = futures_price - signal.entry_price
        
        # Delta approximation for ATM options
        delta = 0.5
        premium_change = spot_move * delta
        
        estimated_premium = signal.option_premium + premium_change
        
        # Ensure premium doesn't go negative
        return max(estimated_premium, 0.0)
    
    def has_active_position(self) -> bool:
        """Check if position is active"""
        return self.active_position is not None and self.active_position.is_active
    
    def get_position_summary(self) -> dict:
        """Get current position summary"""
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

