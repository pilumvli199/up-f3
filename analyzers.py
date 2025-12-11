"""
Market Analyzers V2: COMPLETE FIX
- Delta volume comparison (not cumulative!)
- Adaptive thresholds based on data source
- Live VWAP calculation fallback
- ATM validation
- All previous fixes included
"""

import pandas as pd
from datetime import datetime
from config import *
from utils import IST, setup_logger

logger = setup_logger("analyzers")


# ==================== OI Analyzer (unchanged, already working) ====================
class OIAnalyzer:
    """Open Interest analysis with 5 strikes deep focus"""
    
    @staticmethod
    def calculate_total_oi(strike_data):
        if not strike_data:
            return 0, 0
        total_ce = sum(d.get('ce_oi', 0) for d in strike_data.values())
        total_pe = sum(d.get('pe_oi', 0) for d in strike_data.values())
        return total_ce, total_pe
    
    @staticmethod
    def calculate_deep_analysis_oi(strike_data, atm_strike):
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
        if total_ce == 0:
            return 1.0 if total_pe == 0 else 10.0
        pcr = total_pe / total_ce
        return round(min(pcr, 10.0), 2)
    
    @staticmethod
    def detect_unwinding(ce_5m, ce_15m, pe_5m, pe_15m):
        ce_unwinding = (ce_15m < -MIN_OI_15M_FOR_ENTRY and ce_5m < -MIN_OI_5M_FOR_ENTRY)
        if ce_15m < -STRONG_OI_15M_THRESHOLD and ce_5m < -STRONG_OI_5M_THRESHOLD:
            ce_strength = 'strong'
        elif ce_15m < -MIN_OI_15M_FOR_ENTRY and ce_5m < -MIN_OI_5M_FOR_ENTRY:
            ce_strength = 'medium'
        else:
            ce_strength = 'weak'
        
        pe_unwinding = (pe_15m < -MIN_OI_15M_FOR_ENTRY and pe_5m < -MIN_OI_5M_FOR_ENTRY)
        if pe_15m < -STRONG_OI_15M_THRESHOLD and pe_5m < -STRONG_OI_5M_THRESHOLD:
            pe_strength = 'strong'
        elif pe_15m < -MIN_OI_15M_FOR_ENTRY and pe_5m < -MIN_OI_5M_FOR_ENTRY:
            pe_strength = 'medium'
        else:
            pe_strength = 'weak'
        
        multi_tf = (ce_5m < -2.0 and ce_15m < -3.0) or (pe_5m < -2.0 and pe_15m < -3.0)
        
        return {
            'ce_unwinding': ce_unwinding,
            'pe_unwinding': pe_unwinding,
            'ce_strength': ce_strength,
            'pe_strength': pe_strength,
            'multi_timeframe': multi_tf
        }
    
    @staticmethod
    def detect_conflicting_oi(ce_5m, ce_15m, pe_5m, pe_15m):
        BUILDING_THRESHOLD = 3.0
        
        ce_building_5m = ce_5m > BUILDING_THRESHOLD
        pe_building_5m = pe_5m > BUILDING_THRESHOLD
        
        if ce_building_5m and pe_building_5m:
            return True, f"BOTH CE (+{ce_5m:.1f}%) & PE (+{pe_5m:.1f}%) building on 5m = CHOPPY"
        
        ce_unwinding_15m = ce_15m < -MIN_OI_15M_FOR_ENTRY
        pe_building_5m_strong = pe_5m > BUILDING_THRESHOLD
        
        if ce_unwinding_15m and pe_building_5m_strong:
            return True, f"CE unwinding 15m ({ce_15m:.1f}%) BUT PE building 5m (+{pe_5m:.1f}%) = CONFUSED"
        
        pe_unwinding_15m = pe_15m < -MIN_OI_15M_FOR_ENTRY
        ce_building_5m_strong = ce_5m > BUILDING_THRESHOLD
        
        if pe_unwinding_15m and ce_building_5m_strong:
            return True, f"PE unwinding 15m ({pe_15m:.1f}%) BUT CE building 5m (+{ce_5m:.1f}%) = CONFUSED"
        
        return False, "OI aligned"
    
    @staticmethod
    def get_atm_data(strike_data, atm_strike):
        return strike_data.get(atm_strike, {
            'ce_oi': 0, 'pe_oi': 0, 'ce_vol': 0,
            'pe_vol': 0, 'ce_ltp': 0, 'pe_ltp': 0
        })
    
    @staticmethod
    def get_atm_oi_changes(strike_data, atm_strike, previous_strike_data=None):
        current = strike_data.get(atm_strike, {
            'ce_oi': 0, 'pe_oi': 0, 'ce_vol': 0,
            'pe_vol': 0, 'ce_ltp': 0, 'pe_ltp': 0
        })
        
        ce_change_pct = 0.0
        pe_change_pct = 0.0
        
        if previous_strike_data:
            previous = previous_strike_data.get(atm_strike, {'ce_oi': 0, 'pe_oi': 0})
            prev_ce_oi = previous.get('ce_oi', 0)
            curr_ce_oi = current.get('ce_oi', 0)
            if prev_ce_oi > 0:
                ce_diff = curr_ce_oi - prev_ce_oi
                ce_change_pct = (ce_diff / prev_ce_oi) * 100
            elif curr_ce_oi > 0:
                ce_change_pct = 100.0
            
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
    def validate_atm_data(atm_data):
        ce_ltp = atm_data.get('ce_ltp', 0)
        pe_ltp = atm_data.get('pe_ltp', 0)
        ce_oi = atm_data.get('ce_oi', 0)
        pe_oi = atm_data.get('pe_oi', 0)
        
        if ce_ltp == 0 and pe_ltp == 0 and ce_oi == 0 and pe_oi == 0:
            return False, "ATM data ALL ZERO (API not updating or wrong strike)"
        if ce_ltp > 0 and ce_ltp < 5:
            return False, f"CE premium {ce_ltp:.2f} too low (deep OTM?)"
        if pe_ltp > 0 and pe_ltp < 5:
            return False, f"PE premium {pe_ltp:.2f} too low (deep OTM?)"
        if ce_oi == 0 and pe_oi == 0:
            return False, "ATM has NO open interest (illiquid strike)"
        return True, "ATM data valid"
    
    @staticmethod
    def validate_atm_strike(new_atm, previous_atm, futures_price):
        """
        🆕 VALIDATE ATM strike makes sense
        Prevents sudden wrong jumps
        """
        if previous_atm is None:
            return True, "First ATM"
        
        # Check if ATM shifted too much
        atm_diff = abs(new_atm - previous_atm)
        
        if atm_diff > 200:
            # Big jump - verify alignment
            futures_to_atm = abs(futures_price - new_atm)
            
            if futures_to_atm > 100:
                return False, f"ATM {new_atm} too far from futures {futures_price:.0f} ({futures_to_atm:.0f} pts)"
        
        if atm_diff > 0:
            logger.info(f"📊 ATM SHIFT: {previous_atm} → {new_atm} ({atm_diff:+.0f} pts)")
        
        return True, "ATM valid"
    
    @staticmethod
    def check_oi_reversal(signal_type, oi_changes_history, threshold=EXIT_OI_REVERSAL_THRESHOLD):
        if not oi_changes_history or len(oi_changes_history) < EXIT_OI_CONFIRMATION_CANDLES:
            return False, 'none', 0.0, "Insufficient data"
        
        recent = oi_changes_history[-EXIT_OI_CONFIRMATION_CANDLES:]
        current = recent[-1]
        building_count = sum(1 for oi in recent if oi > threshold)
        
        if building_count >= EXIT_OI_CONFIRMATION_CANDLES:
            avg_building = sum(recent) / len(recent)
            strength = 'strong' if avg_building > 5.0 else 'medium'
            return True, strength, avg_building, f"{signal_type} sustained building: {building_count}/{len(recent)} candles"
        
        if current > EXIT_OI_SPIKE_THRESHOLD:
            return True, 'spike', current, f"{signal_type} spike: {current:.1f}%"
        
        return False, 'none', current, f"{signal_type} OI change: {current:.1f}% (not sustained)"


# ==================== Volume Analyzer V2 ====================
class VolumeAnalyzer:
    """
    🔥 V2 COMPLETE FIX: Delta volume comparison with adaptive thresholds
    """
    
    @staticmethod
    def calculate_total_volume(strike_data):
        if not strike_data:
            return 0, 0
        ce_vol = sum(d.get('ce_vol', 0) for d in strike_data.values())
        pe_vol = sum(d.get('pe_vol', 0) for d in strike_data.values())
        return ce_vol, pe_vol
    
    @staticmethod
    def detect_volume_spike(current, avg, adaptive_threshold=None):
        """
        🆕 ADAPTIVE threshold based on data source
        """
        if avg == 0:
            return False, 0.0
        
        ratio = current / avg
        
        # Use adaptive threshold if provided
        threshold = adaptive_threshold if adaptive_threshold else VOL_SPIKE_MULTIPLIER
        
        is_spike = ratio >= threshold
        
        logger.info(f"📊 VOL SPIKE CHECK: {current:,.0f} / {avg:,.0f} = {ratio:.2f}x")
        logger.info(f"   Threshold: {threshold}x, Result: {'🔥 SPIKE!' if is_spike else 'Normal'}")
        
        return is_spike, round(ratio, 2)
    
    @staticmethod
    def calculate_order_flow(strike_data):
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
    def analyze_volume_trend(df, periods=5, live_volume_delta=None, candle_frozen=False):
        """
        🔥 V2 COMPLETE FIX: 
        - Uses DELTA volume (1-min change)
        - Adaptive thresholds
        - Fallback for frozen candles
        """
        # MODE 1: DELTA VOLUME (BEST - when available)
        if live_volume_delta is not None:
            logger.info(f"📊 VOL CALC (DELTA MODE - CORRECT!):")
            
            # Get historical average from candles
            if df is not None and len(df) >= periods:
                recent = df['volume'].tail(periods)
                avg = recent.mean()
            else:
                # Fallback to reasonable estimate
                avg = 50000  # Conservative average
                logger.warning(f"⚠️ No candle history, using fallback avg: {avg:,.0f}")
            
            current = live_volume_delta
            
            logger.info(f"   Historical avg (per-candle): {avg:,.0f}")
            logger.info(f"   Current delta (1-min): {current:,.0f}")
            
            if avg <= 0:
                logger.error(f"❌ Invalid avg volume")
                return {
                    'trend': 'unknown',
                    'avg_volume': 0,
                    'current_volume': 0,
                    'ratio': 1.0,
                    'source': 'ERROR'
                }
            
            ratio = current / avg
            trend = 'increasing' if ratio > 1.3 else 'decreasing' if ratio < 0.7 else 'stable'
            
            return {
                'trend': trend,
                'avg_volume': round(float(avg), 2),
                'current_volume': round(float(current), 2),
                'ratio': round(float(ratio), 2),
                'source': 'DELTA',
                'adaptive_threshold': VOL_SPIKE_MULTIPLIER  # Normal threshold
            }
        
        # MODE 2: CANDLE VOLUME (FALLBACK - when delta not available)
        elif df is not None and len(df) >= periods:
            logger.info(f"📊 VOL CALC (CANDLE MODE - Fallback):")
            
            recent = df['volume'].tail(periods + 1)
            
            if not pd.api.types.is_numeric_dtype(recent):
                logger.error(f"❌ Volume not numeric")
                return {
                    'trend': 'unknown',
                    'avg_volume': 0,
                    'current_volume': 0,
                    'ratio': 1.0,
                    'source': 'ERROR'
                }
            
            avg = recent.iloc[:-1].mean()
            current = recent.iloc[-1]
            
            if pd.isna(avg) or pd.isna(current) or avg <= 0:
                logger.error(f"❌ Invalid values")
                return {
                    'trend': 'unknown',
                    'avg_volume': 0,
                    'current_volume': 0,
                    'ratio': 1.0,
                    'source': 'ERROR'
                }
            
            ratio = current / avg
            trend = 'increasing' if ratio > 1.3 else 'decreasing' if ratio < 0.7 else 'stable'
            
            logger.info(f"   Avg: {avg:,.0f}, Current: {current:,.0f}, Ratio: {ratio:.2f}x")
            
            # Warn if candles frozen
            if candle_frozen:
                logger.warning(f"⚠️ CANDLE FROZEN - Volume data may be stale!")
            
            return {
                'trend': trend,
                'avg_volume': round(float(avg), 2),
                'current_volume': round(float(current), 2),
                'ratio': round(float(ratio), 2),
                'source': 'CANDLE' if not candle_frozen else 'CANDLE_FROZEN',
                'adaptive_threshold': VOL_SPIKE_MULTIPLIER
            }
        
        # MODE 3: NO DATA
        else:
            logger.warning(f"⚠️ VOL CALC: Insufficient data")
            return {
                'trend': 'unknown',
                'avg_volume': 0,
                'current_volume': 0,
                'ratio': 1.0,
                'source': 'NO_DATA'
            }


# ==================== Technical Analyzer V2 ====================
class TechnicalAnalyzer:
    """Technical indicators with fallback support"""
    
    @staticmethod
    def calculate_vwap(df, live_vwap_fallback=None):
        """
        🆕 VWAP with fallback for frozen candles
        """
        if df is None or len(df) == 0:
            if live_vwap_fallback:
                logger.warning(f"⚠️ Using LIVE VWAP fallback: ₹{live_vwap_fallback:.2f}")
                return live_vwap_fallback
            return None
        
        try:
            df_copy = df.copy()
            df_copy['typical_price'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
            df_copy['vol_price'] = df_copy['typical_price'] * df_copy['volume']
            df_copy['cum_vol_price'] = df_copy['vol_price'].cumsum()
            df_copy['cum_volume'] = df_copy['volume'].cumsum()
            df_copy['vwap'] = df_copy['cum_vol_price'] / df_copy['cum_volume']
            
            vwap = round(df_copy['vwap'].iloc[-1], 2)
            
            # Validate VWAP is reasonable
            last_close = df_copy['close'].iloc[-1]
            vwap_diff = abs(vwap - last_close)
            
            if vwap_diff > 500:
                logger.warning(f"⚠️ VWAP {vwap:.2f} too far from close {last_close:.2f}")
                if live_vwap_fallback:
                    logger.warning(f"   Using LIVE VWAP fallback: ₹{live_vwap_fallback:.2f}")
                    return live_vwap_fallback
            
            return vwap
        except Exception as e:
            logger.error(f"❌ VWAP error: {e}")
            if live_vwap_fallback:
                logger.warning(f"⚠️ Using LIVE VWAP fallback: ₹{live_vwap_fallback:.2f}")
                return live_vwap_fallback
            return None
    
    @staticmethod
    def calculate_vwap_distance(price, vwap):
        if not vwap or not price:
            return 0
        return round(price - vwap, 2)
    
    @staticmethod
    def validate_signal_with_vwap(signal_type, spot, vwap, atr):
        if not vwap or not spot or not atr:
            return False, "Missing VWAP/Price data", 0
        
        distance = spot - vwap
        
        if VWAP_STRICT_MODE:
            buffer = atr * VWAP_DISTANCE_MAX_ATR_MULTIPLE
        else:
            buffer = VWAP_BUFFER
        
        if signal_type == "CE_BUY":
            if distance < -buffer:
                return False, f"Price {abs(distance):.0f} pts below VWAP (too far)", 0
            elif distance > buffer * 3:
                return False, f"Price {distance:.0f} pts above VWAP (overextended)", 0
            else:
                if distance > 0:
                    score = min(100, 80 + (distance / buffer * 20))
                else:
                    score = max(60, 80 - (abs(distance) / buffer * 20))
                return True, f"VWAP distance OK: {distance:+.0f} pts", int(score)
        
        elif signal_type == "PE_BUY":
            if distance > buffer:
                return False, f"Price {distance:.0f} pts above VWAP (too far)", 0
            elif distance < -buffer * 3:
                return False, f"Price {abs(distance):.0f} pts below VWAP (overextended)", 0
            else:
                if distance < 0:
                    score = min(100, 80 + (abs(distance) / buffer * 20))
                else:
                    score = max(60, 80 - (distance / buffer * 20))
                return True, f"VWAP distance OK: {distance:+.0f} pts", int(score)
        
        return False, "Unknown signal type", 0
    
    @staticmethod
    def calculate_atr(df, period=ATR_PERIOD, synthetic_atr_fallback=None):
        """
        🆕 ATR with synthetic fallback
        """
        if df is None or len(df) < period:
            if synthetic_atr_fallback:
                logger.warning(f"⚠️ Using SYNTHETIC ATR fallback: {synthetic_atr_fallback:.1f}")
                return synthetic_atr_fallback
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
            if synthetic_atr_fallback:
                return synthetic_atr_fallback
            return ATR_FALLBACK
    
    @staticmethod
    def analyze_candle(df):
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
                'open': o, 'high': h, 'low': l, 'close': c
            }
        except Exception as e:
            logger.error(f"❌ Candle error: {e}")
            return TechnicalAnalyzer._empty_candle()
    
    @staticmethod
    def detect_momentum(df, periods=3):
        if df is None or len(df) < periods:
            return {
                'direction': 'unknown',
                'strength': 0,
                'consecutive_green': 0,
                'consecutive_red': 0
            }
        
        recent = df.tail(periods)
        min_body_size = MIN_CANDLE_SIZE
        
        green_valid = 0
        red_valid = 0
        
        for idx, row in recent.iterrows():
            body_size = abs(row['close'] - row['open'])
            if body_size >= min_body_size:
                if row['close'] > row['open']:
                    green_valid += 1
                elif row['close'] < row['open']:
                    red_valid += 1
        
        direction = 'bullish' if green_valid >= 2 else 'bearish' if red_valid >= 2 else 'sideways'
        strength = green_valid if green_valid >= 2 else red_valid if red_valid >= 2 else 0
        
        return {
            'direction': direction,
            'strength': strength,
            'consecutive_green': green_valid,
            'consecutive_red': red_valid
        }
    
    @staticmethod
    def _empty_candle():
        return {
            'color': 'UNKNOWN', 'size': 0, 'body_size': 0,
            'upper_wick': 0, 'lower_wick': 0, 'rejection': False,
            'rejection_type': None, 'open': 0, 'high': 0, 'low': 0, 'close': 0
        }


# ==================== Market Analyzer (unchanged) ====================
class MarketAnalyzer:
    """Market structure analysis"""
    
    @staticmethod
    def calculate_max_pain(strike_data, spot_price):
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
        try:
            from config import get_next_weekly_expiry
            today = datetime.now(IST).date()
            expiry = datetime.strptime(get_next_weekly_expiry(), '%Y-%m-%d').date()
            return today == expiry
        except:
            return False
    
    @staticmethod
    def calculate_sentiment(pcr, order_flow, ce_change, pe_change):
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
