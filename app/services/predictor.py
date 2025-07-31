# app/services/predictor.py
import pandas as pd
import torch
import matplotlib.pyplot as plt
from app.models.lstm_model import LSTMModel
from app.models.xgb_model import XGBModel
from app.services.data_loader import fetch_upbit_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

model_dict = {
    "lstm": LSTMModel,
    "xgb": XGBModel,
}

def get_price_prediction(model: str) -> dict:
    if model not in model_dict:
        return {"error": f"지원하지 않는 모델입니다: {model}"}

    try:
        df = fetch_upbit_data()
        if df is None or df.empty:
            return {"error": "데이터 로딩 실패"}

        model_class = model_dict[model]()
        model_class.load()
        result = model_class.predict(df)
        return result

    except Exception as e:
        return {"error": f"예측 중 오류 발생: {str(e)}"}


def get_recent_prediction_plot(model: str = "lstm", count: int = 300, use_experiment: bool = False) -> str:
    if use_experiment:
        experiment_dir = f"results/lstm/long_seq_high_hidden/prediction.png"
        if os.path.exists(experiment_dir):
            return experiment_dir


    # fallback: 예측 직접 계산
    df = fetch_upbit_data(count=count)
    if df is None or len(df) < 100:
        raise ValueError("데이터 부족")

    model_class = model_dict[model]()
    model_class.load()

    close = df['close'].values.reshape(-1, 1)
    scaled = model_class.scaler.transform(close)

    seq_len = 90
    x_data = []
    for i in range(len(scaled) - seq_len):
        x_data.append(scaled[i:i+seq_len])
    x_tensor = torch.tensor(np.array(x_data)).float()

    with torch.no_grad():
        y_pred = model_class.model(x_tensor).numpy()

    y_pred_rescaled = model_class.scaler.inverse_transform(y_pred)
    y_true_rescaled = close[seq_len:]

    mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    r2 = r2_score(y_true_rescaled, y_pred_rescaled)


    print(f"[DEBUG] Prediction MSE: {mse:.4f}, MAE: {mae:.2f}, R2: {r2:.4f}")

    # 시각화 저장
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_rescaled, label="Actual", color="blue")
    plt.plot(y_pred_rescaled, label="Predicted", color="red")
    plt.title(f"Recent {count} Prediction\nMSE: {mse:.4f}, MAE: {mae:.2f}, R2: {r2:.4f}")
    plt.xlabel("Time Step")
    plt.ylabel("Price (KRW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"recent_prediction_{model}_{count}.png"
    plt.savefig(plot_path)
    return plot_path
