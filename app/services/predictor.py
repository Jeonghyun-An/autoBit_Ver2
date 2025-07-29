# app/services/predictor.py
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from app.models.lstm_model import LSTMPricePredictor
from app.services.data_loader import fetch_upbit_data

def get_price_prediction():
    df = fetch_upbit_data(count=60)
    if df is None or len(df) < 60:
        return {"error": "Upbit에서 데이터를 가져오지 못했습니다."}

    close = df['close'].values.reshape(-1, 1)

    # 2. 스케일링
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    x = torch.tensor(scaled).float().unsqueeze(0)  # [1, 60, 1]

    # 4. 모델 로딩 및 예측
    model = LSTMPricePredictor()
    model.eval()

    with torch.no_grad():
        y_pred = model(x).item()

    # 5. 역스케일링
    predicted_price = scaler.inverse_transform([[y_pred]])[0][0]

    return {
        "predicted_price": round(predicted_price, 2),
        "latest_real_price": float(close[-1]),
        "diff": round(predicted_price - close[-1][0], 2)
    }
