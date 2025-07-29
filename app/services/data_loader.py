# app/services/data_loader.py
import requests
import pandas as pd

def fetch_upbit_data(count=200):
    url = f"https://api.upbit.com/v1/candles/minutes/1"
    params = {
        "market": "KRW-BTC",
        "count": count
    }
    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        return None

    raw_data = response.json()
    df = pd.DataFrame(raw_data)
    df = df[['candle_date_time_kst', 'trade_price', 'candle_acc_trade_volume']]
    df.columns = ['timestamp', 'close', 'volume']
    df = df.sort_values('timestamp')  # 최신순 → 과거순 정렬

    return df
