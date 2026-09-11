"""
Technical Indicators and Feature Engineering Module for CryptoPredict AI.
Implements vectorized calculation of RSI, MACD, SMA, EMA, and Bollinger Bands.
"""
import numpy as np
import pandas as pd

def compute_rsi(prices, period=14):
    """
    Computes Relative Strength Index (RSI) using standard Wilder smoothing.
    Returns array of same length as prices, padded with initial values.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    if n < period + 1:
        return np.full(n, 50.0)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi = np.zeros(n)
    rsi[:period] = 50.0

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return np.clip(rsi, 0.0, 100.0)

def compute_ema(prices, period):
    """
    Computes Exponential Moving Average (EMA).
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    if n == 0:
        return np.array([])
    if n < period:
        return prices.copy()
    
    alpha = 2.0 / (period + 1.0)
    ema = np.zeros(n)
    ema[0] = prices[0]
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    return ema

def compute_sma(prices, period):
    """
    Computes Simple Moving Average (SMA).
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    sma = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - period + 1)
        sma[i] = np.mean(prices[start_idx:i + 1])
    return sma

def compute_macd(prices, fast=12, slow=26, signal=9):
    """
    Computes Moving Average Convergence Divergence (MACD), Signal Line, and Histogram.
    """
    prices = np.asarray(prices, dtype=np.float64)
    fast_ema = compute_ema(prices, fast)
    slow_ema = compute_ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger_bands(prices, period=20, num_std=2.0):
    """
    Computes Bollinger Bands: Middle Band (SMA 20), Upper Band (+2 std), Lower Band (-2 std).
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    middle = compute_sma(prices, period)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - period + 1)
        window = prices[start_idx:i + 1]
        std = np.std(window)
        upper[i] = middle[i] + (num_std * std)
        lower[i] = middle[i] - (num_std * std)
        
    return upper, middle, lower

def extract_all_indicators(close_prices, high_prices=None, low_prices=None, volume=None):
    """
    Extracts all indicators and compiles them into a dictionary of lists ready for JSON/Chart.js.
    """
    close = np.asarray(close_prices, dtype=np.float64)
    n = len(close)

    # 1. RSI (14)
    rsi_14 = compute_rsi(close, period=14)

    # 2. MACD (12, 26, 9)
    macd_line, signal_line, hist = compute_macd(close, fast=12, slow=26, signal=9)

    # 3. Moving Averages
    sma_20 = compute_sma(close, period=20)
    sma_50 = compute_sma(close, period=50)
    ema_12 = compute_ema(close, period=12)
    ema_26 = compute_ema(close, period=26)

    # 4. Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close, period=20, num_std=2.0)

    # Current snapshot
    current_rsi = float(rsi_14[-1]) if n > 0 else 50.0
    current_macd = float(macd_line[-1]) if n > 0 else 0.0
    current_signal = float(signal_line[-1]) if n > 0 else 0.0
    current_hist = float(hist[-1]) if n > 0 else 0.0

    return {
        "rsi": [round(float(v), 2) for v in rsi_14],
        "macd_line": [round(float(v), 4) for v in macd_line],
        "signal_line": [round(float(v), 4) for v in signal_line],
        "macd_histogram": [round(float(v), 4) for v in hist],
        "sma_20": [round(float(v), 2) for v in sma_20],
        "sma_50": [round(float(v), 2) for v in sma_50],
        "ema_12": [round(float(v), 2) for v in ema_12],
        "ema_26": [round(float(v), 2) for v in ema_26],
        "bb_upper": [round(float(v), 2) for v in bb_upper],
        "bb_middle": [round(float(v), 2) for v in bb_middle],
        "bb_lower": [round(float(v), 2) for v in bb_lower],
        "summary": {
            "rsi": round(current_rsi, 2),
            "macd": round(current_macd, 4),
            "signal": round(current_signal, 4),
            "histogram": round(current_hist, 4),
            "sma_20": round(float(sma_20[-1]), 2),
            "sma_50": round(float(sma_50[-1]), 2),
            "ema_12": round(float(ema_12[-1]), 2),
            "ema_26": round(float(ema_26[-1]), 2),
            "bb_upper": round(float(bb_upper[-1]), 2),
            "bb_lower": round(float(bb_lower[-1]), 2),
        }
    }
