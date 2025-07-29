# app/services/predictor.py
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from app.models.lstm_model import LSTMPricePredictor

def get_price_prediction():
    # 1. 데이터 불러오기
    df = pd.read_csv("data/btc_sample.csv")
    close = df['close'].values.reshape(-1, 1)

    # 2. 스케일링
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    # 3. 시퀀스 자르기 (최근 60분만 사용)
    seq_len = 60
    if len(scaled) < seq_len:
        return {"error": "Not enough data"}

    seq_input = scaled[-seq_len:]
    x = torch.tensor(seq_input).float().unsqueeze(0)  # [1, 60, 1]

    # 4. 모델 로딩 및 예측
    model = LSTMPricePredictor()
    model.eval()

    with torch.no_grad():
        y_pred = model(x).item()

    # 5. 역스케일링
    predicted_price = scaler.inverse_transform([[y_pred]])[0][0]

    return {
        "predicted_price": round(predicted_price, 2)
    }
