# scripts/train_xgb.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from app.services.data_loader import fetch_upbit_data
import joblib

SEQ_LEN = 60

def make_sequences(data, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len].flatten())
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def train():
    df = fetch_upbit_data(count=300)
    if df is None or len(df) < SEQ_LEN:
        print("❌ 데이터 부족")
        return

    close = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    X, y = make_sequences(scaled)
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y)

    joblib.dump(model, "model_xgb.pkl")
    joblib.dump(scaler, "scaler_xgb.pkl")
    print("✅ XGBoost 모델 학습 및 저장 완료")
    
        # 손실 그래프
    preds = model.predict(X)
    plt.figure(figsize=(10, 5))
    plt.plot(y, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.title("XGBoost Prediction vs Real")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("xgb_prediction.png", dpi=150)
    print("📈 예측 그래프 저장 완료: xgb_prediction.png")
if __name__ == "__main__":
    train()
