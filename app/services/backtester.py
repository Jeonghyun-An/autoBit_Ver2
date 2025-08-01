# app/services/backtester.py
import pandas as pd
from app.models.model_registry import MODEL_REGISTRY
from app.strategies import strategy_registry

def run_backtest(model_name: str, strategy_name: str, threshold: float, df: pd.DataFrame):
    model_class = MODEL_REGISTRY.get(model_name)
    strategy_fn = strategy_registry.get(strategy_name)
    if not model_class or not strategy_fn:
        return {"error": "Invalid model or strategy"}

    model = model_class()
    model.load()

    krw = 1_000_000
    btc = 0.0
    logs = []

    seq_len = 60
    for i in range(seq_len, len(df)):
        sliced_df = df.iloc[i - seq_len:i]
        result = model.predict(sliced_df)
        predicted = result["predicted_price"]
        real = result["latest_real_price"]
        diff = result["diff"]

        action = strategy_fn(predicted, real, threshold)

        timestamp = df.iloc[i]["timestamp"]
        price = real

        if action == "BUY" and krw > 0:
            btc = krw / price
            krw = 0
            logs.append({"time": timestamp, "action": "BUY", "price": price, "krw": krw, "btc": btc})
        elif action == "SELL" and btc > 0:
            krw = btc * price
            btc = 0
            logs.append({"time": timestamp, "action": "SELL", "price": price, "krw": krw, "btc": btc})
        else:
            logs.append({"time": timestamp, "action": "HOLD", "price": price, "krw": krw, "btc": btc})

    final_price = df.iloc[-1]["close"]
    total_value = krw + btc * final_price
    profit = round(total_value - 1_000_000, 2)
    rate = round((total_value / 1_000_000 - 1) * 100, 2)

    return {
        "start_asset": 1_000_000,
        "end_asset": round(total_value, 2),
        "profit": profit,
        "profit_rate": rate,
        "final_krw": round(krw, 2),
        "final_btc": round(btc, 8),
        "logs": logs,
        "model": model_name,
        "strategy": strategy_name,
        "threshold": threshold
    }
