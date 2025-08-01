# app/models/lstm_model.py
import joblib
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from app.models.base import ModelBase
from scripts.config_lstm import SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, INPUT_SIZE, OUTPUT_SIZE

class LSTMPricePredictor(torch.nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(ModelBase):
    def __init__(self):
        self.model = LSTMPricePredictor()
        self.scaler = MinMaxScaler()
        self.features = ["close"]  # 기본값, load()에서 덮어씀

    def load(self):
        checkpoint = torch.load("model_lstm.pt", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        scaler_obj = joblib.load("scaler_lstm.pkl")
        if isinstance(scaler_obj, dict):
            self.scaler = scaler_obj["scaler"]
            self.feature_names = scaler_obj["features"]
        else:
            self.scaler = scaler_obj
            self.feature_names = ["close"]  # 기본값

    def prepare_input(self, df: pd.DataFrame) -> torch.Tensor:
        features = df[self.features].values
        scaled = self.scaler.transform(features)
        x = torch.tensor(scaled).float().unsqueeze(0)
        return x

    def predict(self, df: pd.DataFrame) -> dict:
        features = df[self.feature_names].values
        scaled = self.scaler.transform(features)
        x = torch.tensor(scaled).float().unsqueeze(0)
    
        with torch.no_grad():
            y_pred = self.model(x).item()
    
        # 복원 시 첫 번째 열만 close로 복원
        dummy_zero = [0] * (features.shape[1] - 1)
        recovered = self.scaler.inverse_transform([[y_pred] + dummy_zero])[0][0]
    
        return {
            "predicted_price": round(recovered, 2),
            "latest_real_price": float(df["close"].values[-1]),
            "diff": round(recovered - df["close"].values[-1], 2)
        }
