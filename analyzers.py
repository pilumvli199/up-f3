"""
Market Analyzers V3: SIMPLIFIED - Yellow Flash Edition ⚡
- Only 3 indicators: EMA-9, VWAP, ATR
- Price Action patterns (Rejection, Engulfing, Break-Retest)
- S/R Levels (Multi-timeframe)
- OI + Volume analysis
- Clean + Fast + Powerful!
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config import *
from utils import IST, setup_logger

logger = setup_logger("analyzers")


# ==================== OI Analyzer ====================
class OIAnalyzer:
    """Open Interest analysis - UNCHANGED (already perfect!)"""
    
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
        """Detect when OI shows confusion (hedge both sides)"""
        BUILDING_THRESHOLD = 3.0
        
        ce_building_5m = ce_5m > BUILDING_THRESHOLD
        pe_building_5m = pe_5m > BUILDING_THRESHOLD
        
        if ce_building_5m and pe_building_5m:
            return True, f"BOTH CE (+{ce_5m:.1f}%) & PE (+{pe_5m:.1f}%) building = CHOPPY"
        
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
    def validate_atm_strike(new_atm, previous_atm, futures_price, tolerance=200):
        """Validate ATM strike against previous and futures price"""
        if previous_atm is None:
            return True, "First ATM, no validation needed"
        
        atm_diff = abs(new_atm - previous_atm)
        
        if atm_diff > tolerance:
            futures_to_atm = abs(futures_price - new_atm)
            if futures_to_atm > 100:
                return False, f"ATM {new_atm} too far from futures {futures_price:.2f} (diff: {futures_to_atm:.0f})"
        
        return True, f"ATM valid (shift: {new_atm - previous_atm:+.0f} pts)"
    
    @staticmethod
    def get_atm_data(strike_data, atm_strike):
        """Get ATM strike data"""
        if not strike_data or atm_strike not in strike_data:
            return {}
        return strike_data[atm_strike]
    
    @staticmethod
    def get_atm_oi_changes(strike_data, atm_strike, previous_strike_data):
        """
        Calculate ATM OI changes vs previous snapshot
        Used for instant ATM battle detection
        """
        if not strike_data or atm_strike not in strike_data:
            return {
                'has_previous_data': False,
                'ce_change': 0,
                'pe_change': 0,
                'ce_change_pct': 0,
                'pe_change_pct': 0
            }
        
        current = strike_data[atm_strike]
        
        if not previous_strike_data or atm_strike not in previous_strike_data:
            return {
                'has_previous_data': False,
                'ce_change': 0,
                'pe_change': 0,
                'ce_change_pct': 0,
                'pe_change_pct': 0
            }
        
        previous = previous_strike_data[atm_strike]
        
        current_ce = current.get('ce_oi', 0)
        current_pe = current.get('pe_oi', 0)
        prev_ce = previous.get('ce_oi', 0)
        prev_pe = previous.get('pe_oi', 0)
        
        ce_change = current_ce - prev_ce
        pe_change = current_pe - prev_pe
        
        ce_change_pct = (ce_change / prev_ce * 100) if prev_ce > 0 else 0
        pe_change_pct = (pe_change / prev_pe * 100) if prev_pe > 0 else 0
        
        return {
            'has_previous_data': True,
            'ce_change': ce_change,
            'pe_change': pe_change,
            'ce_change_pct': round(ce_change_pct, 1),
            'pe_change_pct': round(pe_change_pct, 1)
        }
    
    @staticmethod
    def validate_atm_data(atm_data):
        """
        Validate ATM strike has valid OI data
        Prevents trading on bad data
        """
        if not atm_data:
            return False, "ATM data is empty"
        
        ce_oi = atm_data.get('ce_oi', 0)
        pe_oi = atm_data.get('pe_oi', 0)
        
        if ce_oi == 0 and pe_oi == 0:
            return False, "ATM has zero OI on both sides"
        
        if ce_oi == 0:
            return False, "ATM CE has zero OI (suspicious!)"
        
        if pe_oi == 0:
            return False, "ATM PE has zero OI (suspicious!)"
        
        # Check if OI is reasonable (not too low)
        MIN_ATM_OI = 10000
        if ce_oi < MIN_ATM_OI or pe_oi < MIN_ATM_OI:
            return False, f"ATM OI too low (CE: {ce_oi:,.0f}, PE: {pe_oi:,.0f})"
        
        return True, f"ATM data valid (CE: {ce_oi:,.0f}, PE: {pe_oi:,.0f})"


# ==================== Volume Analyzer ====================
class VolumeAnalyzer:
    """Volume analysis with DELTA tracking"""
    
    @staticmethod
    def analyze_volume_trend(df, live_volume_delta=None, candle_frozen=False):
        """
        V2: Compare DELTA volume to historical per-candle average
        NOT cumulative to per-candle (that was the bug!)
        """
        if df is None or len(df) == 0:
            return {
                'trend': 'unknown',
                'current_volume': 0,
                'avg_volume': 0,
                'ratio': 0,
                'source': 'NO_DATA'
            }
        
        # Calculate historical average (per-candle volumes)
        avg_volume = df['volume'].tail(5).mean()
        
        # Decide which volume to use
        if live_volume_delta is not None and live_volume_delta > 0:
            # BEST: Use delta (1-minute change)
            current_volume = live_volume_delta
            source = 'DELTA'
            adaptive_threshold = VOL_SPIKE_MULTIPLIER  # 1.8x
            
        elif candle_frozen:
            # Candles frozen, no reliable data
            current_volume = 0
            source = 'CANDLE_FROZEN'
            adaptive_threshold = VOL_SPIKE_MULTIPLIER * 1.2
            
        else:
            # Fallback: Use latest candle volume
            current_volume = df['volume'].iloc[-1]
            source = 'CANDLE'
            adaptive_threshold = VOL_SPIKE_MULTIPLIER
        
        # Calculate ratio (CORRECT comparison!)
        if avg_volume > 0:
            ratio = current_volume / avg_volume
        else:
            ratio = 0
        
        # Determine trend
        if ratio >= adaptive_threshold:
            trend = 'spike'
        elif ratio >= 1.2:
            trend = 'increasing'
        elif ratio <= 0.5:
            trend = 'decreasing'
        else:
            trend = 'normal'
        
        # Log for debugging
        if source == 'DELTA':
            logger.info(f"📊 VOL CALC (DELTA MODE - CORRECT!):")
            logger.info(f"   Historical avg (per-candle): {avg_volume:,.0f}")
            logger.info(f"   Current delta (1-min): {current_volume:,.0f}")
            logger.info(f"   Source: {source}")
            logger.info(f"   Ratio: {ratio:.2f}x")
        
        return {
            'trend': trend,
            'current_volume': int(current_volume),
            'avg_volume': int(avg_volume),
            'ratio': round(ratio, 2),
            'source': source,
            'adaptive_threshold': adaptive_threshold
        }
    
    @staticmethod
    def detect_volume_spike(current, average, adaptive_threshold=None):
        """Detect if volume is spiking"""
        if average == 0:
            return False, 0
        
        threshold = adaptive_threshold if adaptive_threshold else VOL_SPIKE_MULTIPLIER
        ratio = current / average
        
        spike = ratio >= threshold
        
        logger.info(f"📊 VOL SPIKE CHECK: {current:,.0f} / {average:,.0f} = {ratio:.2f}x")
        logger.info(f"   Threshold: {threshold:.1f}x, Result: {'🔥 SPIKE!' if spike else 'Normal'}")
        
        return spike, ratio
    
    @staticmethod
    def calculate_order_flow(strike_data):
        """Calculate order flow imbalance"""
        if not strike_data:
            return 1.0
        
        ce_volume = sum(d.get('ce_volume', 0) for d in strike_data.values())
        pe_volume = sum(d.get('pe_volume', 0) for d in strike_data.values())
        
        if pe_volume == 0:
            return 10.0 if ce_volume > 0 else 1.0
        
        return round(ce_volume / pe_volume, 2)


# ==================== Technical Analyzer - SIMPLIFIED! ====================
class TechnicalAnalyzer:
    """Only 3 indicators: EMA-9, VWAP, ATR - CLEAN!"""
    
    @staticmethod
    def calculate_ema9(df):
        """Fast EMA for trend - ONLY indicator we need!"""
        if df is None or len(df) < 9:
            return None
        
        ema = df['close'].ewm(span=9, adjust=False).mean().iloc[-1]
        return round(ema, 2)
    
    @staticmethod
    def calculate_vwap(df, live_vwap_fallback=None):
        """
        VWAP - Institution level
        V2: Support fallback when candles frozen
        """
        if df is None or len(df) == 0:
            if live_vwap_fallback:
                logger.warning(f"⚠️ Using LIVE VWAP fallback: ₹{live_vwap_fallback:.2f}")
                return live_vwap_fallback
            return None
        
        # Check if candles have timestamp issues
        if 'timestamp' in df.columns:
            unique_times = df['timestamp'].nunique()
            total_candles = len(df)
            if unique_times < total_candles * 0.5:
                logger.warning(f"⚠️ Candle timestamps suspicious, using fallback if available")
                if live_vwap_fallback:
                    return live_vwap_fallback
        
        # Calculate VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        
        return round(vwap, 2)
    
    @staticmethod
    def calculate_atr(df, period=14, synthetic_atr_fallback=None):
        """
        ATR - For SL/Target sizing
        V2: Support synthetic ATR when candles frozen
        """
        if df is None or len(df) < period:
            if synthetic_atr_fallback:
                logger.warning(f"⚠️ Using SYNTHETIC ATR: {synthetic_atr_fallback:.1f}")
                return synthetic_atr_fallback
            return 30
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return round(max(atr, 10), 1)
    
    @staticmethod
    def calculate_vwap_distance(price, vwap):
        """Distance from VWAP - simple but powerful!"""
        if not vwap:
            return 0
        return round(price - vwap, 1)
    
    @staticmethod
    def analyze_candle(df):
        """Basic candle analysis - simple is best!"""
        if df is None or len(df) == 0:
            return {'color': 'unknown', 'body_size': 0, 'pattern': 'none'}
        
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        
        if last['close'] > last['open']:
            color = 'green'
        elif last['close'] < last['open']:
            color = 'red'
        else:
            color = 'doji'
        
        return {
            'color': color,
            'body_size': round(body, 1),
            'high': last['high'],
            'low': last['low'],
            'open': last['open'],
            'close': last['close']
        }
    
    @staticmethod
    def detect_momentum(df, periods=3):
        """Simple momentum - last 3 candles trend"""
        if df is None or len(df) < periods:
            return {'direction': 'unknown', 'strength': 0}
        
        recent = df.tail(periods)
        green_count = sum(recent['close'] > recent['open'])
        red_count = periods - green_count
        
        if green_count >= 2:
            direction = 'bullish'
            strength = green_count
        elif red_count >= 2:
            direction = 'bearish'
            strength = red_count
        else:
            direction = 'neutral'
            strength = 0
        
        return {'direction': direction, 'strength': strength}
    
    @staticmethod
    def validate_signal_with_vwap(signal_type, price, vwap, atr):
        """
        Validate if signal aligns with VWAP position
        Returns: (is_valid, reason, score)
        """
        if vwap is None or atr is None:
            return True, "VWAP/ATR not available", 50
        
        distance = price - vwap
        distance_abs = abs(distance)
        
        # Calculate score (0-100)
        # Perfect score at VWAP, decreases with distance
        max_distance = atr * 3.0  # 3x ATR = max acceptable
        
        if distance_abs > max_distance:
            score = 0
        else:
            score = int(100 * (1 - distance_abs / max_distance))
        
        # CE_BUY validation (bullish - price should be near or above VWAP)
        if signal_type == "CE_BUY":
            if distance < -atr * 2.0:  # Too far below VWAP
                return False, f"Price {distance:.0f} pts below VWAP (bearish zone)", score
            elif distance > atr * 2.5:  # Too far above VWAP
                return False, f"Price {distance:.0f} pts above VWAP (exhausted)", score
            else:
                return True, f"Price position valid ({distance:+.0f} pts from VWAP)", score
        
        # PE_BUY validation (bearish - price should be near or below VWAP)
        elif signal_type == "PE_BUY":
            if distance > atr * 2.0:  # Too far above VWAP
                return False, f"Price {distance:.0f} pts above VWAP (bullish zone)", score
            elif distance < -atr * 2.5:  # Too far below VWAP
                return False, f"Price {distance:.0f} pts below VWAP (exhausted)", score
            else:
                return True, f"Price position valid ({distance:+.0f} pts from VWAP)", score
        
        return True, "Unknown signal type", 50


# ==================== Price Action Analyzer - NEW! ====================
class PriceActionAnalyzer:
    """Simple but powerful price action patterns"""
    
    @staticmethod
    def detect_rejection_candle(candle_dict, sr_level=None, tolerance=30):
        """
        Rejection candle = Long wick
        Works with or without S/R level!
        Most reliable pattern!
        """
        if not candle_dict:
            return None
        
        try:
            body = candle_dict.get('body_size', 0)
            high = candle_dict.get('high', 0)
            low = candle_dict.get('low', 0)
            open_price = candle_dict.get('open', 0)
            close = candle_dict.get('close', 0)
            
            if not all([high, low, open_price, close]):
                return None
            
            total_range = high - low
            if total_range == 0 or body == 0:
                return None
            
            # Calculate wicks
            upper_wick = high - max(open_price, close)
            lower_wick = min(open_price, close) - low
            
            # Upper rejection (bearish)
            if upper_wick > 2 * body and upper_wick > 15:
                # If S/R level provided, check if near it
                if sr_level:
                    if abs(high - sr_level) < tolerance:
                        return {
                            'pattern': 'BEARISH_REJECTION',
                            'confidence': 85,  # Higher with S/R
                            'wick_size': round(upper_wick, 1),
                            'near_level': sr_level,
                            'with_sr': True
                        }
                else:
                    # Still valid without S/R, lower confidence
                    return {
                        'pattern': 'BEARISH_REJECTION',
                        'confidence': 70,
                        'wick_size': round(upper_wick, 1),
                        'near_level': None,
                        'with_sr': False
                    }
            
            # Lower rejection (bullish)
            if lower_wick > 2 * body and lower_wick > 15:
                # If S/R level provided, check if near it
                if sr_level:
                    if abs(low - sr_level) < tolerance:
                        return {
                            'pattern': 'BULLISH_REJECTION',
                            'confidence': 85,  # Higher with S/R
                            'wick_size': round(lower_wick, 1),
                            'near_level': sr_level,
                            'with_sr': True
                        }
                else:
                    # Still valid without S/R, lower confidence
                    return {
                        'pattern': 'BULLISH_REJECTION',
                        'confidence': 70,
                        'wick_size': round(lower_wick, 1),
                        'near_level': None,
                        'with_sr': False
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Rejection detection error: {e}")
            return None
    
    @staticmethod
    def detect_engulfing(df):
        """Engulfing pattern - strong reversal signal"""
        if df is None or len(df) < 2:
            return None
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])
        
        # Need decent body sizes
        if prev_body < 5 or curr_body < 8:
            return None
        
        # Bullish Engulfing
        if (prev['close'] < prev['open'] and  # Prev red
            curr['close'] > curr['open'] and  # Curr green
            curr['close'] > prev['open'] and  # Engulfs fully
            curr['open'] < prev['close']):
            return {
                'pattern': 'BULLISH_ENGULFING',
                'confidence': 75,
                'prev_body': round(prev_body, 1),
                'curr_body': round(curr_body, 1)
            }
        
        # Bearish Engulfing
        if (prev['close'] > prev['open'] and  # Prev green
            curr['close'] < curr['open'] and  # Curr red
            curr['close'] < prev['open'] and  # Engulfs fully
            curr['open'] > prev['close']):
            return {
                'pattern': 'BEARISH_ENGULFING',
                'confidence': 75,
                'prev_body': round(prev_body, 1),
                'curr_body': round(curr_body, 1)
            }
        
        return None
    
    @staticmethod
    def detect_break_retest(price_history, sr_level, tolerance=30):
        """
        Break & Retest - safest pattern!
        1. Break above resistance
        2. Come back to test
        3. Bounce = BUY
        """
        if not price_history or len(price_history) < 10:
            return None
        
        # Check if broke above in recent past
        broke_above = False
        broke_below = False
        
        for price in price_history[-10:-3]:
            if price > sr_level + 20:
                broke_above = True
            if price < sr_level - 20:
                broke_below = True
        
        current_price = price_history[-1]
        
        # Resistance became support (broke above, now retesting)
        if broke_above and abs(current_price - sr_level) < tolerance:
            if current_price >= sr_level - 10:
                return {
                    'pattern': 'RETEST_SUPPORT',
                    'confidence': 85,
                    'level': sr_level,
                    'current': current_price
                }
        
        # Support became resistance (broke below, now retesting)
        if broke_below and abs(current_price - sr_level) < tolerance:
            if current_price <= sr_level + 10:
                return {
                    'pattern': 'RETEST_RESISTANCE',
                    'confidence': 85,
                    'level': sr_level,
                    'current': current_price
                }
        
        return None


# ==================== S/R Analyzer - NEW! ====================
class SRAnalyzer:
    """Support/Resistance levels - Multi-timeframe"""
    
    @staticmethod
    def find_pivot_points(df, lookback=20, distance=5):
        """
        Find swing highs and lows - IMPROVED DETECTION
        These become S/R levels
        """
        if df is None or len(df) < 10:  # Relaxed minimum
            logger.warning(f"⚠️ S/R: Insufficient data ({len(df) if df is not None else 0} candles)")
            return {'support': [], 'resistance': []}
        
        highs = df['high'].values
        lows = df['low'].values
        
        resistance_levels = []
        support_levels = []
        
        # Adaptive distance based on data length
        actual_distance = min(distance, len(highs) // 4)
        if actual_distance < 2:
            actual_distance = 2
        
        # Find local maxima (resistance)
        for i in range(actual_distance, len(highs) - actual_distance):
            is_peak = True
            for j in range(i - actual_distance, i + actual_distance + 1):
                if j != i and highs[j] >= highs[i]:
                    is_peak = False
                    break
            if is_peak:
                resistance_levels.append(round(highs[i], 0))
        
        # Find local minima (support)
        for i in range(actual_distance, len(lows) - actual_distance):
            is_valley = True
            for j in range(i - actual_distance, i + actual_distance + 1):
                if j != i and lows[j] <= lows[i]:
                    is_valley = False
                    break
            if is_valley:
                support_levels.append(round(lows[i], 0))
        
        # Fallback if no levels found (use percentiles)
        if not resistance_levels:
            resistance_levels = [
                round(np.percentile(highs, 95), 0),
                round(np.percentile(highs, 90), 0),
                round(np.max(highs), 0)
            ]
            logger.debug(f"  ⚠️ S/R: Using percentile resistance")
        
        if not support_levels:
            support_levels = [
                round(np.percentile(lows, 5), 0),
                round(np.percentile(lows, 10), 0),
                round(np.min(lows), 0)
            ]
            logger.debug(f"  ⚠️ S/R: Using percentile support")
        
        # Remove duplicates (cluster similar levels)
        resistance_levels = SRAnalyzer._cluster_levels(resistance_levels, cluster_distance=15)
        support_levels = SRAnalyzer._cluster_levels(support_levels, cluster_distance=15)
        
        # Log what was found
        logger.debug(f"  Found {len(support_levels)} support, {len(resistance_levels)} resistance")
        
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels, reverse=True)
        }
    
    @staticmethod
    def _cluster_levels(levels, cluster_distance=20):
        """Merge similar levels into clusters"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level - current_cluster[-1] <= cluster_distance:
                current_cluster.append(level)
            else:
                # Average the cluster
                clustered.append(round(sum(current_cluster) / len(current_cluster), 0))
                current_cluster = [level]
        
        # Don't forget last cluster
        if current_cluster:
            clustered.append(round(sum(current_cluster) / len(current_cluster), 0))
        
        return clustered
    
    @staticmethod
    def find_confluence_zones(daily_sr, m15_sr, m5_sr, tolerance=20):
        """
        Find where multiple timeframes agree
        These are STRONGEST levels!
        """
        confluence_support = []
        confluence_resistance = []
        
        # Check support alignment
        for daily_s in daily_sr.get('support', []):
            for m15_s in m15_sr.get('support', []):
                for m5_s in m5_sr.get('support', []):
                    if (abs(daily_s - m15_s) <= tolerance and 
                        abs(m15_s - m5_s) <= tolerance):
                        avg_level = round((daily_s + m15_s + m5_s) / 3, 0)
                        confluence_support.append({
                            'level': avg_level,
                            'daily': daily_s,
                            'm15': m15_s,
                            'm5': m5_s,
                            'strength': 'VERY_STRONG',
                            'confidence': 90
                        })
        
        # Check resistance alignment
        for daily_r in daily_sr.get('resistance', []):
            for m15_r in m15_sr.get('resistance', []):
                for m5_r in m5_sr.get('resistance', []):
                    if (abs(daily_r - m15_r) <= tolerance and 
                        abs(m15_r - m5_r) <= tolerance):
                        avg_level = round((daily_r + m15_r + m5_r) / 3, 0)
                        confluence_resistance.append({
                            'level': avg_level,
                            'daily': daily_r,
                            'm15': m15_r,
                            'm5': m5_r,
                            'strength': 'VERY_STRONG',
                            'confidence': 90
                        })
        
        # Remove duplicate levels (keep unique only)
        confluence_support = list({d['level']: d for d in confluence_support}.values())
        confluence_resistance = list({d['level']: d for d in confluence_resistance}.values())
        
        return {
            'support': sorted(confluence_support, key=lambda x: x['level']),
            'resistance': sorted(confluence_resistance, key=lambda x: x['level'], reverse=True)
        }
    
    @staticmethod
    def check_price_location(price, sr_levels, tolerance=30):
        """
        Check where price is relative to S/R levels
        Returns: 'at_support', 'at_resistance', 'between', 'above_all', 'below_all'
        """
        supports = [s['level'] if isinstance(s, dict) else s for s in sr_levels.get('support', [])]
        resistances = [r['level'] if isinstance(r, dict) else r for r in sr_levels.get('resistance', [])]
        
        # Check if at support
        for support in supports:
            if abs(price - support) <= tolerance:
                return {
                    'location': 'at_support',
                    'level': support,
                    'distance': round(price - support, 1)
                }
        
        # Check if at resistance
        for resistance in resistances:
            if abs(price - resistance) <= tolerance:
                return {
                    'location': 'at_resistance',
                    'level': resistance,
                    'distance': round(price - resistance, 1)
                }
        
        # Check if above all resistance
        if resistances and price > max(resistances) + tolerance:
            return {
                'location': 'above_all',
                'nearest_level': max(resistances),
                'distance': round(price - max(resistances), 1)
            }
        
        # Check if below all support
        if supports and price < min(supports) - tolerance:
            return {
                'location': 'below_all',
                'nearest_level': min(supports),
                'distance': round(price - min(supports), 1)
            }
        
        # Between levels
        return {
            'location': 'between',
            'nearest_support': min(supports, key=lambda x: abs(price - x)) if supports else None,
            'nearest_resistance': min(resistances, key=lambda x: abs(price - x)) if resistances else None
        }
    
    @staticmethod
    def get_nearest_levels(price, sr_levels):
        """Get nearest support and resistance"""
        supports = [s['level'] if isinstance(s, dict) else s for s in sr_levels.get('support', [])]
        resistances = [r['level'] if isinstance(r, dict) else r for r in sr_levels.get('resistance', [])]
        
        # Find nearest support below price
        supports_below = [s for s in supports if s < price]
        nearest_support = max(supports_below) if supports_below else None
        
        # Find nearest resistance above price
        resistances_above = [r for r in resistances if r > price]
        nearest_resistance = min(resistances_above) if resistances_above else None
        
        return {
            'support': nearest_support,
            'resistance': nearest_resistance,
            'support_distance': round(price - nearest_support, 1) if nearest_support else None,
            'resistance_distance': round(nearest_resistance - price, 1) if nearest_resistance else None
        }


# ==================== Market Analyzer - UNCHANGED ====================
class MarketAnalyzer:
    """Market-level analysis"""
    
    @staticmethod
    def detect_gamma_zone():
        return "Not detected"
    
    @staticmethod
    def calculate_risk_reward(entry, target, sl):
        if sl >= entry:
            return 0
        risk = entry - sl
        reward = target - entry
        return round(reward / risk, 1) if risk > 0 else 0


# ==================== EXPORT ====================
__all__ = [
    'OIAnalyzer',
    'VolumeAnalyzer', 
    'TechnicalAnalyzer',
    'PriceActionAnalyzer',
    'SRAnalyzer',
    'MarketAnalyzer'
]
