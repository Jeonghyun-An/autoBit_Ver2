# app/models/xgboost_model.py
import xgboost as xgb
import pandas as pd
import numpy as np
from app.models.base import ModelBase
from app.services.data_loader import fetch_upbit_data
from sklearn.preprocessing import MinMaxScaler

class XGBoostModel(ModelBase):
    def __init__(self):
        self.model = xgb.XGBRegressor()
        self.scaler = MinMaxScaler()

    def load(self):
        try:
            self.model.load_model("model_xgb.json")
        except:
            print("모델 파일 없음")

    def predict(self, df: pd.DataFrame) -> dict:
        close = df["close"].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(close)

        # 최근 10개 시점 → 예측용 피처
        features = scaled[-10:].flatten().reshape(1, -1)
        pred_scaled = self.model.predict(features)[0]
        predicted_price = self.scaler.inverse_transform([[pred_scaled]])[0][0]

        return {
            "predicted_price": round(predicted_price, 2),
            "latest_real_price": float(close[-1]),
            "diff": round(predicted_price - close[-1][0], 2)
        }
