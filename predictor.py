from flask import Flask, render_template, request
import requests
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# CoinGecko Demo API Key
API_KEY = os.getenv("COINGECKO_API_KEY", "CG-nioVvZdxB6U7gPTG8KFaVmvm")

# Fallback popular coins in case CoinGecko rate limit triggers on initial load
DEFAULT_COINS = [
    {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC"},
    {"id": "ethereum", "name": "Ethereum", "symbol": "ETH"},
    {"id": "solana", "name": "Solana", "symbol": "SOL"},
    {"id": "binancecoin", "name": "BNB", "symbol": "BNB"},
    {"id": "ripple", "name": "XRP", "symbol": "XRP"},
    {"id": "cardano", "name": "Cardano", "symbol": "ADA"},
    {"id": "dogecoin", "name": "Dogecoin", "symbol": "DOGE"},
    {"id": "avalanche-2", "name": "Avalanche", "symbol": "AVAX"},
    {"id": "polkadot", "name": "Polkadot", "symbol": "DOT"},
    {"id": "chainlink", "name": "Chainlink", "symbol": "LINK"},
]

def get_headers():
    headers = {
        "User-Agent": "CryptoPredictorAI/1.0",
        "Accept": "application/json"
    }
    if API_KEY and API_KEY != "YOUR_API_KEY":
        headers["x-cg-demo-api-key"] = API_KEY
    return headers

def fetch_top_coins():
    """Fetch top coins by market cap from CoinGecko, with fallback."""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=30&page=1"
        response = requests.get(url, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return [{"id": c["id"], "name": c["name"], "symbol": c["symbol"]} for c in data]
    except Exception as e:
        print(f"Warning fetching coin list: {e}")
    return DEFAULT_COINS

@app.route('/', methods=['GET', 'POST'])
def index():
    coins = fetch_top_coins()
    coin_ids = [c["id"] for c in coins]

    if request.method == 'POST':
        coin_id = request.form.get('coin_id', 'bitcoin').strip().lower()
        days = request.form.get('days', '7')

        if coin_id in coin_ids or coin_id:
            try:
                predictions, chart_data, stats = get_market_chart(coin_id, days)
                return render_template(
                    'index.html',
                    predictions=predictions,
                    chart_data=chart_data,
                    stats=stats,
                    coins=coins,
                    coin_id=coin_id,
                    days=days
                )
            except Exception as e:
                return render_template(
                    'index.html',
                    error=f"Prediction error: {str(e)}",
                    coins=coins,
                    coin_id=coin_id,
                    days=days
                )
        else:
            return render_template('index.html', error="Invalid coin selected", coins=coins)

    return render_template('index.html', coins=coins, coin_id='bitcoin', days='7')

def get_market_chart(coin_id, days):
    """
    Fetches OHLC price data, trains a Linear Regression model,
    and returns forecast values, chart data, and summary metrics.
    """
    days = str(days)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(url, headers=get_headers(), timeout=10)
    
    if response.status_code != 200:
        raise ValueError(f"CoinGecko API returned status {response.status_code}: {response.text}")

    ohlc_data = response.json()
    if not isinstance(ohlc_data, list) or len(ohlc_data) < 2:
        raise ValueError("Insufficient market data received from CoinGecko. Please try another coin or time window.")

    ohlc = np.array(ohlc_data)
    timestamps = ohlc[:, 0]
    close_prices = ohlc[:, 4]

    # Convert timestamps to readable date labels
    historical_labels = [datetime.fromtimestamp(ts / 1000.0).strftime('%b %d, %H:%M') for ts in timestamps]
    historical_prices = [float(p) for p in close_prices]

    # Prepare training data (index-based regression)
    n_samples = len(close_prices)
    X = np.arange(n_samples).reshape(-1, 1)
    y = close_prices

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # In-sample R2 score
    y_fitted = model.predict(X)
    r2 = float(max(0.0, r2_score(y, y_fitted)))

    # Forecast next 7 intervals
    forecast_steps = 7
    step_duration_seconds = (timestamps[-1] - timestamps[0]) / max(1, n_samples - 1) / 1000.0
    
    X_future = np.arange(n_samples, n_samples + forecast_steps).reshape(-1, 1)
    y_future = model.predict(X_future)

    future_labels = []
    last_timestamp = timestamps[-1] / 1000.0
    for i in range(1, forecast_steps + 1):
        future_time = datetime.fromtimestamp(last_timestamp + (i * step_duration_seconds))
        future_labels.append(future_time.strftime('%b %d, %H:%M'))

    predictions = {label: float(pred) for label, pred in zip(future_labels, y_future)}

    current_price = float(close_prices[-1])
    target_price = float(y_future[-1])
    change_pct = ((target_price - current_price) / current_price) * 100.0

    stats = {
        "current_price": current_price,
        "target_price": target_price,
        "change_pct": change_pct,
        "r2_score": r2
    }

    # Chart.js dataset structure: Historical padded with None for forecast, and forecast padded with None for historical
    all_labels = historical_labels + future_labels
    padded_hist_prices = historical_prices + [None] * forecast_steps
    padded_forecast_prices = [None] * (n_samples - 1) + [historical_prices[-1]] + [float(p) for p in y_future]

    chart_data = {
        "labels": all_labels,
        "historical_prices": padded_hist_prices,
        "forecast_prices": padded_forecast_prices
    }

    return predictions, chart_data, stats

if __name__ == '__main__':
    print("Starting CryptoPredict Flask server at http://127.0.0.1:5000")
    app.run(debug=True)
