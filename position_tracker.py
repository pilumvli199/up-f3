"""
Position Tracker: Track active positions & monitor exit conditions
FIXED: Sustained OI reversal, minimum hold time, smarter premium estimate
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config import *
from utils import IST, setup_logger
from signal_engine import Signal, SignalType
from analyzers import OIAnalyzer

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
    
    # Track OI history for exit logic
    oi_history: list = None
    
    def __post_init__(self):
        if self.oi_history is None:
            self.oi_history = []
    
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
        self.last_sl_notification = None
    
    def open_position(self, signal: Signal):
        """Open new position from signal"""
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
        """
        Check if position should exit OR if trailing SL updated
        Returns: (should_exit, reason, details) or None
        
        Special: (False, "SL_UPDATED", details) for SL moves (not exit)
        """
        if not self.active_position or not self.active_position.is_active:
            return None
        
        position = self.active_position
        signal = position.signal
        hold_time = position.get_hold_time_minutes()
        
        # Get current premium (estimated from futures movement + ATM data)
        current_premium = self._estimate_premium(current_data, signal)
        
        # Update highest premium for trailing
        if current_premium > position.highest_premium:
            old_peak = position.highest_premium
            old_sl = position.trailing_sl
            
            position.highest_premium = current_premium
            
            if ENABLE_TRAILING_SL:
                new_sl = current_premium * (1 - TRAILING_SL_DISTANCE)
                sl_move_pct = abs(new_sl - old_sl) / old_sl * 100 if old_sl > 0 else 0
                
                position.trailing_sl = new_sl
                
                # Only notify if significant move (>5%)
                if sl_move_pct >= TRAILING_SL_UPDATE_THRESHOLD:
                    logger.info(f"📈 NEW PEAK: ₹{current_premium:.2f} (was ₹{old_peak:.2f})")
                    logger.info(f"🔒 SL MOVED: ₹{new_sl:.2f} (was ₹{old_sl:.2f}) [{sl_move_pct:.1f}% move]")
                    
                    details = f"Peak: ₹{current_premium:.2f} → New SL: ₹{new_sl:.2f}"
                    self.last_sl_notification = datetime.now(IST)
                    return False, "SL_UPDATED", details
        
        # ========== EXIT PRIORITY ORDER ==========
        
        # EXIT 1: Stop Loss Hit (HIGHEST PRIORITY)
        if signal.signal_type == SignalType.CE_BUY:
            if current_data['futures_price'] <= signal.stop_loss:
                return True, "Stop Loss Hit", f"Price: ₹{current_data['futures_price']:.2f} ≤ SL: ₹{signal.stop_loss:.2f}"
        else:  # PE_BUY
            if current_data['futures_price'] >= signal.stop_loss:
                return True, "Stop Loss Hit", f"Price: ₹{current_data['futures_price']:.2f} ≥ SL: ₹{signal.stop_loss:.2f}"
        
        # EXIT 2: Target Hit
        if signal.signal_type == SignalType.CE_BUY:
            if current_data['futures_price'] >= signal.target_price:
                return True, "Target Hit", f"Price: ₹{current_data['futures_price']:.2f} ≥ Target: ₹{signal.target_price:.2f}"
        else:  # PE_BUY
            if current_data['futures_price'] <= signal.target_price:
                return True, "Target Hit", f"Price: ₹{current_data['futures_price']:.2f} ≤ Target: ₹{signal.target_price:.2f}"
        
        # EXIT 3: Time-based (Market close)
        current_time = datetime.now(IST).time()
        if current_time >= time(15, 15):
            return True, "Market Closing", "Exiting before market close"
        
        # ========== Below exits require MINIMUM HOLD TIME ==========
        
        if hold_time < MIN_HOLD_TIME_MINUTES:
            logger.debug(f"  ⏳ Hold time {hold_time:.1f}min < {MIN_HOLD_TIME_MINUTES}min - blocking early exits")
            return None
        
        # EXIT 4: OI Reversal (SUSTAINED CHECK)
        if hold_time >= MIN_HOLD_BEFORE_OI_EXIT:
            # Track OI changes
            if signal.signal_type == SignalType.CE_BUY:
                ce_oi = current_data.get('ce_oi_5m', 0)
                position.oi_history.append(ce_oi)
            else:  # PE_BUY
                pe_oi = current_data.get('pe_oi_5m', 0)
                position.oi_history.append(pe_oi)
            
            # Keep only last 5 candles
            position.oi_history = position.oi_history[-5:]
            
            # Check sustained reversal
            signal_type_str = 'CE' if signal.signal_type == SignalType.CE_BUY else 'PE'
            is_reversal, strength, avg, message = OIAnalyzer.check_oi_reversal(
                signal_type_str,
                position.oi_history,
                EXIT_OI_REVERSAL_THRESHOLD
            )
            
            if is_reversal:
                return True, "OI Reversal", f"{message} (Avg: {avg:.1f}%)"
        
        # EXIT 5: Trailing SL Hit
        if ENABLE_TRAILING_SL and current_premium < position.trailing_sl:
            profit = current_premium - position.entry_premium
            profit_pct = (profit / position.entry_premium * 100) if position.entry_premium > 0 else 0
            return True, "Trailing SL Hit", f"Locked profit: ₹{profit:.2f} ({profit_pct:+.1f}%)"
        
        # EXIT 6: Premium Drop from Peak
        premium_drop_pct = ((position.highest_premium - current_premium) / 
                           position.highest_premium * 100) if position.highest_premium > 0 else 0
        
        if premium_drop_pct >= EXIT_PREMIUM_DROP_PERCENT:
            return True, "Premium Drop", f"Down {premium_drop_pct:.1f}% from peak ₹{position.highest_premium:.2f}"
        
        # EXIT 7: Volume Dry (if significant time passed)
        if hold_time >= 15:
            volume_ratio = current_data.get('volume_ratio', 1.0)
            if volume_ratio < EXIT_VOLUME_DRY_THRESHOLD:
                return True, "Volume Dried", f"Volume ratio: {volume_ratio:.1f}x"
        
        # EXIT 8: Candle Rejection
        candle = current_data.get('candle_data', {})
        if candle.get('rejection'):
            rejection_type = candle.get('rejection_type')
            
            if signal.signal_type == SignalType.CE_BUY and rejection_type == 'upper':
                return True, "Candle Rejection", "Long upper wick at resistance"
            elif signal.signal_type == SignalType.PE_BUY and rejection_type == 'lower':
                return True, "Candle Rejection", "Long lower wick at support"
        
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
        Estimate option premium - IMPROVED VERSION
        
        Priority:
        1. Use actual premium from atm_data if available
        2. Delta-based estimation from price movement
        3. Fallback to entry premium
        """
        # Try to get actual premium from ATM data
        atm_data = current_data.get('atm_data', {})
        
        if signal.signal_type == SignalType.CE_BUY:
            actual_premium = atm_data.get('ce_ltp', 0)
        else:
            actual_premium = atm_data.get('pe_ltp', 0)
        
        # If we have actual premium, use it
        if actual_premium > 0:
            return float(actual_premium)
        
        # Otherwise, estimate using delta
        futures_price = current_data.get('futures_price', signal.entry_price)
        spot_move = futures_price - signal.entry_price
        
        # Calculate delta based on moneyness
        strike_diff = abs(futures_price - signal.atm_strike)
        
        if strike_diff < 25:  # Near ATM
            delta = 0.5
        elif strike_diff < 50:  # Slightly ITM/OTM
            delta = 0.6 if ((signal.signal_type == SignalType.CE_BUY and futures_price > signal.atm_strike) or
                           (signal.signal_type == SignalType.PE_BUY and futures_price < signal.atm_strike)) else 0.4
        else:  # Deeper ITM/OTM
            delta = 0.7 if ((signal.signal_type == SignalType.CE_BUY and futures_price > signal.atm_strike) or
                           (signal.signal_type == SignalType.PE_BUY and futures_price < signal.atm_strike)) else 0.3
        
        # Adjust delta for direction
        if signal.signal_type == SignalType.PE_BUY:
            spot_move = -spot_move
        
        premium_change = spot_move * delta
        estimated_premium = signal.option_premium + premium_change
        
        # Ensure premium doesn't go negative
        return max(estimated_premium, 5.0)  # Min 5 rupees
    
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
