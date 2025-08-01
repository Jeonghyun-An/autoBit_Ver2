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

    df = fetch_upbit_data(count=count)
    if df is None or len(df) < 100:
        raise ValueError("데이터 부족")

    model_class = model_dict[model]()
    model_class.load()

    # --- 예상 입력 피처 추출 ---
    try:
        feature_names = model_class.scaler.feature_names_in_.tolist()
    except AttributeError:
        feature_count = model_class.scaler.n_features_in_
        # ['close', 'open', 'high', 'low', 'volume'] 기준 우선 순위로 설정
        all_possible = ["close", "open", "high", "low", "volume"]
        feature_names = all_possible[:feature_count]

    # 실제 df에 존재하는 컬럼만 사용
    feature_names = [f for f in feature_names if f in df.columns]
    if len(feature_names) != model_class.scaler.n_features_in_:
        raise ValueError(
            f"입력 피처 수 불일치: scaler expects {model_class.scaler.n_features_in_}, but got {len(feature_names)}"
        )

    input_data = df[feature_names].values
    scaled = model_class.scaler.transform(input_data)

    seq_len = 90
    x_data = []
    for i in range(len(scaled) - seq_len):
        x_data.append(scaled[i:i+seq_len])
    x_tensor = torch.tensor(np.array(x_data)).float()

    with torch.no_grad():
        y_pred = model_class.model(x_tensor).numpy()

    # ✅ close만 inverse_transform
    dummy = np.zeros((len(y_pred), input_data.shape[1]))
    dummy[:, 0] = y_pred[:, 0]
    y_pred_rescaled = model_class.scaler.inverse_transform(dummy)[:, 0]

    y_true_rescaled = df['close'].values[seq_len:]

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
