# scripts/train_xgb.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from app.services.data_loader import fetch_upbit_data
import joblib
import matplotlib.pyplot as plt

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

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X, label=y)

    # Training parameters
    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": 0.1,
        "eval_metric": "rmse"
    }

    evals_result = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train")],
        evals_result=evals_result,
        verbose_eval=False
    )

    # Save model and scaler
    model.save_model("model_xgb.json")
    joblib.dump(scaler, "scaler_xgb.pkl")
    print("✅ XGBoost model and scaler saved")

    # Prediction for plot
    preds = model.predict(dtrain)
    plt.figure(figsize=(10, 5))
    plt.plot(y, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.title("XGBoost Prediction vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("xgb_prediction.png", dpi=150)
    print("📈 Prediction graph saved: xgb_prediction.png")

    # Loss plot
    losses = evals_result["train"]["rmse"]
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training RMSE")
    plt.title("XGBoost Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("xgb_loss.png", dpi=150)
    print("📉 Loss graph saved: xgb_loss.png")

if __name__ == "__main__":
    train()
