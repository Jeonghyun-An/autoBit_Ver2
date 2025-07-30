import pandas as pd
import xgboost as xgb
import joblib
from app.models.base import ModelBase
from sklearn.preprocessing import MinMaxScaler

class XGBModel(ModelBase):
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()

    def load(self):
        self.model = joblib.load("model_xgb.pkl")
        self.scaler = joblib.load("scaler_xgb.pkl")

    def predict(self, df: pd.DataFrame) -> dict:
        close = df['close'].values.reshape(-1, 1)
        scaled = self.scaler.transform(close)

        seq_len = 60
        X = []
        for i in range(len(scaled) - seq_len):
            X.append(scaled[i:i+seq_len].flatten())
        X = X[-1].reshape(1, -1)  # 마지막 시퀀스만 예측

        y_pred = self.model.predict(X)[0]
        predicted_price = self.scaler.inverse_transform([[y_pred]])[0][0]

        return {
            "predicted_price": round(predicted_price, 2),
            "latest_real_price": float(close[-1]),
            "diff": round(predicted_price - close[-1][0], 2)
        }
