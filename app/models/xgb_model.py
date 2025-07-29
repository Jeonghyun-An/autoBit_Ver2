import pandas as pd
import xgboost as xgb
import joblib
from app.models.base import ModelBase

class XGBModel(ModelBase):
    def __init__(self):
        self.model = None
        self.scaler = None

    def load(self):
        self.model = joblib.load("model_xgb.pkl")
        self.scaler = joblib.load("scaler_xgb.pkl")

    def predict(self, df: pd.DataFrame) -> dict:
        close = df['close'].values.reshape(-1, 1)
        scaled = self.scaler.transform(close)

        X = scaled[-30:]  # 마지막 30개로 예측 (LSTM과 유사하게 시계열 기반)
        X = X.reshape(1, -1)

        y_pred = self.model.predict(X)[0]
        predicted_price = self.scaler.inverse_transform([[y_pred]])[0][0]

        return {
            "predicted_price": round(predicted_price, 2),
            "latest_real_price": float(close[-1]),
            "diff": round(predicted_price - close[-1][0], 2)
        }
