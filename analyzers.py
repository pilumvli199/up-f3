"""
Market Analyzers: OI, Volume, Technical, Market Structure
All analysis functions in one place
"""

import pandas as pd
from datetime import datetime
from config import *
from utils import IST, setup_logger

logger = setup_logger("analyzers")


# ==================== OI Analyzer ====================
class OIAnalyzer:
    """Open Interest analysis"""
    
    @staticmethod
    def calculate_total_oi(strike_data):
        """Calculate total CE/PE OI"""
        total_ce = sum(d.get('ce_oi', 0) for d in strike_data.values())
        total_pe = sum(d.get('pe_oi', 0) for d in strike_data.values())
        return total_ce, total_pe
    
    @staticmethod
    def calculate_pcr(total_pe, total_ce):
        """Calculate Put-Call Ratio"""
        return round(total_pe / total_ce, 2) if total_ce > 0 else 0.0
    
    @staticmethod
    def detect_unwinding(ce_5m, ce_15m, pe_5m, pe_15m):
        """Detect CE/PE unwinding patterns"""
        ce_unwinding = ce_15m < -OI_THRESHOLD_MEDIUM or ce_5m < -OI_THRESHOLD_MEDIUM
        ce_strength = 'strong' if ce_15m < -OI_THRESHOLD_STRONG else 'medium' if ce_15m < -OI_THRESHOLD_MEDIUM else 'weak'
        
        pe_unwinding = pe_15m < -OI_THRESHOLD_MEDIUM or pe_5m < -OI_THRESHOLD_MEDIUM
        pe_strength = 'strong' if pe_15m < -OI_THRESHOLD_STRONG else 'medium' if pe_15m < -OI_THRESHOLD_MEDIUM else 'weak'
        
        multi_tf = (ce_5m < -1.0 and ce_15m < -1.0) or (pe_5m < -1.0 and pe_15m < -1.0)
        
        return {
            'ce_unwinding': ce_unwinding,
            'pe_unwinding': pe_unwinding,
            'ce_strength': ce_strength,
            'pe_strength': pe_strength,
            'multi_timeframe': multi_tf
        }
    
    @staticmethod
    def get_atm_data(strike_data, atm_strike):
        """Get ATM strike data"""
        return strike_data.get(atm_strike, {})


# ==================== Volume Analyzer ====================
class VolumeAnalyzer:
    """Volume and order flow analysis"""
    
    @staticmethod
    def calculate_total_volume(strike_data):
        """Calculate total CE/PE volume"""
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
        """Analyze volume trend"""
        if len(df) < periods + 1:
            return {'trend': 'unknown', 'avg_volume': 0, 'current_volume': 0, 'ratio': 1.0}
        
        recent = df['volume'].tail(periods + 1)
        avg = recent.iloc[:-1].mean()
        current = recent.iloc[-1]
        ratio = current / avg if avg > 0 else 1.0
        
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
        """Calculate VWAP"""
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
            logger.error(f"VWAP error: {e}")
            return None
    
    @staticmethod
    def calculate_vwap_distance(price, vwap):
        """Calculate distance from VWAP in points"""
        if not vwap or not price:
            return 0
        return round(price - vwap, 2)
    
    @staticmethod
    def calculate_atr(df, period=ATR_PERIOD):
        """Calculate ATR"""
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
            logger.error(f"ATR error: {e}")
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
            
            if upper_wick > body * 2 and body > 0:
                rejection = True
                rejection_type = 'upper'
            elif lower_wick > body * 2 and body > 0:
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
            logger.error(f"Candle analysis error: {e}")
            return TechnicalAnalyzer._empty_candle()
    
    @staticmethod
    def detect_momentum(df, periods=3):
        """Detect price momentum"""
        if df is None or len(df) < periods:
            return {'direction': 'unknown', 'strength': 0, 'consecutive_green': 0, 'consecutive_red': 0}
        
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
            'color': 'UNKNOWN', 'size': 0, 'body_size': 0,
            'upper_wick': 0, 'lower_wick': 0, 'rejection': False,
            'rejection_type': None, 'open': 0, 'high': 0, 'low': 0, 'close': 0
        }


# ==================== Market Analyzer ====================
class MarketAnalyzer:
    """Market structure analysis"""
    
    @staticmethod
    def calculate_max_pain(strike_data, spot_price):
        """Calculate max pain strike"""
        if not strike_data:
            return 0, 0.0
        
        strikes = sorted(strike_data.keys())
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
        """Check if expiry day"""
        try:
            from config import get_next_tuesday_expiry
            today = datetime.now(IST).date()
            expiry = datetime.strptime(get_next_tuesday_expiry(), '%Y-%m-%d').date()
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
