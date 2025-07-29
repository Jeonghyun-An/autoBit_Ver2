# train_xgb.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from app.services.data_loader import fetch_upbit_data

df = fetch_upbit_data(count=200)
close = df["close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

def make_xgb_data(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback].flatten())
        y.append(data[i+lookback])
    return np.array(X), np.array(y).flatten()

X, y = make_xgb_data(scaled)

model = xgb.XGBRegressor()
model.fit(X, y)

model.save_model("model_xgb.json")
print("XGBoost 모델 저장 완료")
