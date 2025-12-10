"""
Market Analyzers: OI, Volume, Technical, Market Structure
FIXED: 5 strikes deep analysis, AND logic for unwinding, VWAP validation
"""

import pandas as pd
from datetime import datetime
from config import *
from utils import IST, setup_logger

logger = setup_logger("analyzers")


# ==================== OI Analyzer ====================
class OIAnalyzer:
    """Open Interest analysis with 5 strikes deep focus"""
    
    @staticmethod
    def calculate_total_oi(strike_data):
        """Calculate total CE/PE OI (uses ALL 11 strikes)"""
        if not strike_data:
            return 0, 0
        
        total_ce = sum(d.get('ce_oi', 0) for d in strike_data.values())
        total_pe = sum(d.get('pe_oi', 0) for d in strike_data.values())
        
        return total_ce, total_pe
    
    @staticmethod
    def calculate_deep_analysis_oi(strike_data, atm_strike):
        """
        Calculate CE/PE OI for DEEP ANALYSIS strikes only (5 strikes)
        ATM ± 2 = 5 strikes where 90% institutional money flows
        """
        deep_strikes = get_deep_analysis_strikes(atm_strike)
        
        deep_ce = 0
        deep_pe = 0
        
        for strike in deep_strikes:
            if strike in strike_data:
                deep_ce += strike_data[strike].get('ce_oi', 0)
                deep_pe += strike_data[strike].get('pe_oi', 0)
        
        return deep_ce, deep_pe, deep_strikes
    
    @staticmethod
    def calculate_pcr(total_pe, total_ce):
        """Calculate Put-Call Ratio with neutral default"""
        if total_ce == 0:
            if total_pe == 0:
                return 1.0  # Neutral (both zero)
            else:
                return 10.0  # Very high PCR (cap)
        
        pcr = total_pe / total_ce
        return round(min(pcr, 10.0), 2)  # Cap at 10.0
    
    @staticmethod
    def detect_unwinding(ce_5m, ce_15m, pe_5m, pe_15m):
        """
        Detect CE/PE unwinding - AND LOGIC (both timeframes required)
        This ensures quality signals, not just noise
        """
        # CE unwinding - BOTH 5m AND 15m must show unwinding
        ce_unwinding = (ce_15m < -MIN_OI_15M_FOR_ENTRY and ce_5m < -MIN_OI_5M_FOR_ENTRY)
        
        # Strength based on 15m (primary timeframe)
        if ce_15m < -STRONG_OI_15M_THRESHOLD and ce_5m < -STRONG_OI_5M_THRESHOLD:
            ce_strength = 'strong'
        elif ce_15m < -MIN_OI_15M_FOR_ENTRY and ce_5m < -MIN_OI_5M_FOR_ENTRY:
            ce_strength = 'medium'
        else:
            ce_strength = 'weak'
        
        # PE unwinding - BOTH 5m AND 15m must show unwinding
        pe_unwinding = (pe_15m < -MIN_OI_15M_FOR_ENTRY and pe_5m < -MIN_OI_5M_FOR_ENTRY)
        
        # Strength
        if pe_15m < -STRONG_OI_15M_THRESHOLD and pe_5m < -STRONG_OI_5M_THRESHOLD:
            pe_strength = 'strong'
        elif pe_15m < -MIN_OI_15M_FOR_ENTRY and pe_5m < -MIN_OI_5M_FOR_ENTRY:
            pe_strength = 'medium'
        else:
            pe_strength = 'weak'
        
        # Multi-timeframe confirmation (both showing negative)
        multi_tf = (ce_5m < -2.0 and ce_15m < -3.0) or (pe_5m < -2.0 and pe_15m < -3.0)
        
        return {
            'ce_unwinding': ce_unwinding,
            'pe_unwinding': pe_unwinding,
            'ce_strength': ce_strength,
            'pe_strength': pe_strength,
            'multi_timeframe': multi_tf
        }
    
    @staticmethod
    def get_atm_data(strike_data, atm_strike):
        """Get ATM strike data (current values only)"""
        return strike_data.get(atm_strike, {
            'ce_oi': 0,
            'pe_oi': 0,
            'ce_vol': 0,
            'pe_vol': 0,
            'ce_ltp': 0,
            'pe_ltp': 0
        })
    
    @staticmethod
    def get_atm_oi_changes(strike_data, atm_strike, previous_strike_data=None):
        """
        Get ATM strike data WITH OI change calculations
        Compares current vs previous scan
        """
        current = strike_data.get(atm_strike, {
            'ce_oi': 0,
            'pe_oi': 0,
            'ce_vol': 0,
            'pe_vol': 0,
            'ce_ltp': 0,
            'pe_ltp': 0
        })
        
        ce_change_pct = 0.0
        pe_change_pct = 0.0
        
        if previous_strike_data:
            previous = previous_strike_data.get(atm_strike, {
                'ce_oi': 0,
                'pe_oi': 0
            })
            
            # CE OI change
            prev_ce_oi = previous.get('ce_oi', 0)
            curr_ce_oi = current.get('ce_oi', 0)
            
            if prev_ce_oi > 0:
                ce_diff = curr_ce_oi - prev_ce_oi
                ce_change_pct = (ce_diff / prev_ce_oi) * 100
            elif curr_ce_oi > 0:
                ce_change_pct = 100.0
            
            # PE OI change
            prev_pe_oi = previous.get('pe_oi', 0)
            curr_pe_oi = current.get('pe_oi', 0)
            
            if prev_pe_oi > 0:
                pe_diff = curr_pe_oi - prev_pe_oi
                pe_change_pct = (pe_diff / prev_pe_oi) * 100
            elif curr_pe_oi > 0:
                pe_change_pct = 100.0
        
        return {
            'ce_oi': current.get('ce_oi', 0),
            'pe_oi': current.get('pe_oi', 0),
            'ce_vol': current.get('ce_vol', 0),
            'pe_vol': current.get('pe_vol', 0),
            'ce_ltp': current.get('ce_ltp', 0),
            'pe_ltp': current.get('pe_ltp', 0),
            'ce_change_pct': round(ce_change_pct, 1),
            'pe_change_pct': round(pe_change_pct, 1),
            'has_previous_data': previous_strike_data is not None,
            'atm_strike': atm_strike
        }
    
    @staticmethod
    def check_oi_reversal(signal_type, oi_changes_history, threshold=EXIT_OI_REVERSAL_THRESHOLD):
        """
        Check OI reversal with SUSTAINED building (not single candle spike)
        Requires EXIT_OI_CONFIRMATION_CANDLES consecutive candles
        """
        if not oi_changes_history or len(oi_changes_history) < EXIT_OI_CONFIRMATION_CANDLES:
            return False, 'none', 0.0, "Insufficient data"
        
        # Get last N candles
        recent = oi_changes_history[-EXIT_OI_CONFIRMATION_CANDLES:]
        current = recent[-1]
        
        # Count sustained building
        building_count = sum(1 for oi in recent if oi > threshold)
        
        # Strong reversal: ALL recent candles building
        if building_count >= EXIT_OI_CONFIRMATION_CANDLES:
            avg_building = sum(recent) / len(recent)
            strength = 'strong' if avg_building > 5.0 else 'medium'
            return True, strength, avg_building, f"{signal_type} sustained building: {building_count}/{len(recent)} candles"
        
        # Very strong single spike (exception)
        if current > EXIT_OI_SPIKE_THRESHOLD:
            return True, 'spike', current, f"{signal_type} spike: {current:.1f}%"
        
        # Not confirmed
        return False, 'none', current, f"{signal_type} OI change: {current:.1f}% (not sustained)"


# ==================== Volume Analyzer ====================
class VolumeAnalyzer:
    """Volume and order flow analysis"""
    
    @staticmethod
    def calculate_total_volume(strike_data):
        """Calculate total CE/PE volume (uses ALL 11 strikes)"""
        if not strike_data:
            return 0, 0
        
        ce_vol = sum(d.get('ce_vol', 0) for d in strike_data.values())
        pe_vol = sum(d.get('pe_vol', 0) for d in strike_data.values())
        return ce_vol, pe_vol
    
    @staticmethod
    def detect_volume_spike(current, avg):
        """Detect volume spike"""
        if avg == 0:
            return False, 0.0
        ratio = current / avg
        return ratio >= VOL_SPIKE_MULTIPLIER, round(ratio, 2)
    
    @staticmethod
    def calculate_order_flow(strike_data):
        """Calculate order flow ratio (CE vol / PE vol)"""
        ce_vol, pe_vol = VolumeAnalyzer.calculate_total_volume(strike_data)
        
        if ce_vol == 0 and pe_vol == 0:
            return 1.0
        elif pe_vol == 0:
            return 5.0
        elif ce_vol == 0:
            return 0.2
        
        ratio = ce_vol / pe_vol
        return round(max(0.2, min(ratio, 5.0)), 2)
    
    @staticmethod
    def analyze_volume_trend(df, periods=5):
        """Analyze volume trend from futures candles"""
        if df is None or len(df) < periods + 1:
            logger.warning(f"⚠️ VOL DEBUG: Insufficient data - df={'None' if df is None else len(df)} rows")
            return {
                'trend': 'unknown',
                'avg_volume': 0,
                'current_volume': 0,
                'ratio': 1.0
            }
        
        recent = df['volume'].tail(periods + 1)
        avg = recent.iloc[:-1].mean()
        current = recent.iloc[-1]
        ratio = current / avg if avg > 0 else 1.0
        
        # DEBUG: Log actual values
        logger.debug(f"📊 VOL DEBUG: current={current:.0f}, avg={avg:.0f}, ratio={ratio:.2f}")
        
        trend = 'increasing' if ratio > 1.3 else 'decreasing' if ratio < 0.7 else 'stable'
        
        return {
            'trend': trend,
            'avg_volume': round(avg, 2),
            'current_volume': round(current, 2),
            'ratio': round(ratio, 2)
        }


# ==================== Technical Analyzer ====================
class TechnicalAnalyzer:
    """Technical indicators: VWAP, ATR, Candles"""
    
    @staticmethod
    def calculate_vwap(df):
        """Calculate VWAP from futures candles"""
        if df is None or len(df) == 0:
            return None
        
        try:
            df_copy = df.copy()
            df_copy['typical_price'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
            df_copy['vol_price'] = df_copy['typical_price'] * df_copy['volume']
            df_copy['cum_vol_price'] = df_copy['vol_price'].cumsum()
            df_copy['cum_volume'] = df_copy['volume'].cumsum()
            df_copy['vwap'] = df_copy['cum_vol_price'] / df_copy['cum_volume']
            return round(df_copy['vwap'].iloc[-1], 2)
        except Exception as e:
            logger.error(f"❌ VWAP error: {e}")
            return None
    
    @staticmethod
    def calculate_vwap_distance(price, vwap):
        """Calculate distance from VWAP"""
        if not vwap or not price:
            return 0
        return round(price - vwap, 2)
    
    @staticmethod
    def validate_signal_with_vwap(signal_type, spot, vwap, atr):
        """
        Validate if signal should be taken based on VWAP distance
        BLOCKING CHECK - rejects signals too far from fair value
        """
        if not vwap or not spot or not atr:
            return False, "Missing VWAP/Price data", 0
        
        distance = spot - vwap
        
        # Calculate buffer
        if VWAP_STRICT_MODE:
            buffer = atr * VWAP_DISTANCE_MAX_ATR_MULTIPLE
        else:
            buffer = VWAP_BUFFER
        
        if signal_type == "CE_BUY":
            # CE buy: Price should be near or above VWAP
            if distance < -buffer:
                return False, f"Price {abs(distance):.0f} pts below VWAP (too far)", 0
            elif distance > buffer * 3:
                return False, f"Price {distance:.0f} pts above VWAP (overextended)", 0
            else:
                # Score based on proximity
                if distance > 0:
                    score = min(100, 80 + (distance / buffer * 20))
                else:
                    score = max(60, 80 - (abs(distance) / buffer * 20))
                return True, f"VWAP distance OK: {distance:+.0f} pts", int(score)
        
        elif signal_type == "PE_BUY":
            # PE buy: Price should be near or below VWAP
            if distance > buffer:
                return False, f"Price {distance:.0f} pts above VWAP (too far)", 0
            elif distance < -buffer * 3:
                return False, f"Price {abs(distance):.0f} pts below VWAP (overextended)", 0
            else:
                # Score based on proximity
                if distance < 0:
                    score = min(100, 80 + (abs(distance) / buffer * 20))
                else:
                    score = max(60, 80 - (distance / buffer * 20))
                return True, f"VWAP distance OK: {distance:+.0f} pts", int(score)
        
        return False, "Unknown signal type", 0
    
    @staticmethod
    def calculate_atr(df, period=ATR_PERIOD):
        """Calculate ATR from futures candles"""
        if df is None or len(df) < period:
            return ATR_FALLBACK
        
        try:
            df_copy = df.copy()
            df_copy['h_l'] = df_copy['high'] - df_copy['low']
            df_copy['h_cp'] = abs(df_copy['high'] - df_copy['close'].shift(1))
            df_copy['l_cp'] = abs(df_copy['low'] - df_copy['close'].shift(1))
            df_copy['tr'] = df_copy[['h_l', 'h_cp', 'l_cp']].max(axis=1)
            atr = df_copy['tr'].rolling(window=period).mean().iloc[-1]
            return round(atr, 2)
        except Exception as e:
            logger.error(f"❌ ATR error: {e}")
            return ATR_FALLBACK
    
    @staticmethod
    def analyze_candle(df):
        """Analyze current candle"""
        if df is None or len(df) == 0:
            return TechnicalAnalyzer._empty_candle()
        
        try:
            candle = df.iloc[-1]
            o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
            
            total_size = h - l
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            
            color = 'GREEN' if c > o else 'RED' if c < o else 'DOJI'
            
            rejection = False
            rejection_type = None
            
            if upper_wick > body * EXIT_CANDLE_REJECTION_MULTIPLIER and body > 0:
                rejection = True
                rejection_type = 'upper'
            elif lower_wick > body * EXIT_CANDLE_REJECTION_MULTIPLIER and body > 0:
                rejection = True
                rejection_type = 'lower'
            
            return {
                'color': color,
                'size': round(total_size, 2),
                'body_size': round(body, 2),
                'upper_wick': round(upper_wick, 2),
                'lower_wick': round(lower_wick, 2),
                'rejection': rejection,
                'rejection_type': rejection_type,
                'open': o,
                'high': h,
                'low': l,
                'close': c
            }
        except Exception as e:
            logger.error(f"❌ Candle error: {e}")
            return TechnicalAnalyzer._empty_candle()
    
    @staticmethod
    def detect_momentum(df, periods=3):
        """Detect price momentum"""
        if df is None or len(df) < periods:
            return {
                'direction': 'unknown',
                'strength': 0,
                'consecutive_green': 0,
                'consecutive_red': 0
            }
        
        recent = df.tail(periods)
        green = sum(recent['close'] > recent['open'])
        red = sum(recent['close'] < recent['open'])
        
        direction = 'bullish' if green >= 2 else 'bearish' if red >= 2 else 'sideways'
        strength = green if green >= 2 else red if red >= 2 else 0
        
        return {
            'direction': direction,
            'strength': strength,
            'consecutive_green': green,
            'consecutive_red': red
        }
    
    @staticmethod
    def _empty_candle():
        return {
            'color': 'UNKNOWN',
            'size': 0,
            'body_size': 0,
            'upper_wick': 0,
            'lower_wick': 0,
            'rejection': False,
            'rejection_type': None,
            'open': 0,
            'high': 0,
            'low': 0,
            'close': 0
        }


# ==================== Market Analyzer ====================
class MarketAnalyzer:
    """Market structure analysis"""
    
    @staticmethod
    def calculate_max_pain(strike_data, spot_price):
        """Calculate max pain strike (uses all 11 strikes)"""
        if not strike_data:
            return 0, 0.0
        
        strikes = sorted(strike_data.keys())
        if not strikes:
            return 0, 0.0
        
        max_pain_strike = strikes[len(strikes) // 2]
        min_pain = float('inf')
        
        for test_strike in strikes:
            total_pain = 0.0
            
            for strike, data in strike_data.items():
                ce_oi = data.get('ce_oi', 0)
                pe_oi = data.get('pe_oi', 0)
                
                if test_strike > strike:
                    total_pain += ce_oi * (test_strike - strike)
                if test_strike < strike:
                    total_pain += pe_oi * (strike - test_strike)
            
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike
        
        return max_pain_strike, round(min_pain, 2)
    
    @staticmethod
    def detect_gamma_zone():
        """Check if expiry day (weekly options)"""
        try:
            from config import get_next_weekly_expiry
            today = datetime.now(IST).date()
            expiry = datetime.strptime(get_next_weekly_expiry(), '%Y-%m-%d').date()
            return today == expiry
        except:
            return False
    
    @staticmethod
    def calculate_sentiment(pcr, order_flow, ce_change, pe_change):
        """Calculate market sentiment"""
        bullish = 0
        bearish = 0
        
        if pcr > PCR_BULLISH:
            bullish += 1
        elif pcr < PCR_BEARISH:
            bearish += 1
        
        if order_flow < 1.0:
            bullish += 1
        elif order_flow > 1.5:
            bearish += 1
        
        if ce_change < -2.0:
            bullish += 1
        if pe_change < -2.0:
            bearish += 1
        
        if bullish > bearish:
            return "BULLISH"
        elif bearish > bullish:
            return "BEARISH"
        else:
            return "NEUTRAL"
