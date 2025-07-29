# app/models/lstm_model.py
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from app.models.base import ModelBase

class LSTMPricePredictor(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(ModelBase):
    def __init__(self):
        self.model = LSTMPricePredictor()
        self.scaler = MinMaxScaler()

    def load(self):
        checkpoint = torch.load("model_lstm.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler.min_, self.scaler.scale_ = checkpoint["scaler_min"], 1 / (checkpoint["scaler_max"] - checkpoint["scaler_min"])
        self.model.eval()


    def predict(self, df: pd.DataFrame) -> dict:
        close = df['close'].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(close)
        x = torch.tensor(scaled).float().unsqueeze(0)

        with torch.no_grad():
            y_pred = self.model(x).item()

        predicted_price = self.scaler.inverse_transform([[y_pred]])[0][0]

        return {
            "predicted_price": round(predicted_price, 2),
            "latest_real_price": float(close[-1]),
            "diff": round(predicted_price - close[-1][0], 2)
        }
