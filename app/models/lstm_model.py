# app/models/lstm_model.py
import joblib
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
        checkpoint = torch.load("model_lstm.pt", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # self.model.load_state_dict(torch.load("model_lstm.pt"))
        self.model.eval()
    
        # scaler 불러오기
        self.scaler = joblib.load("scaler_lstm.pkl")
       


    def predict(self, df: pd.DataFrame) -> dict:
        close = df['close'].values.reshape(-1, 1)
        # scaled = self.scaler.fit_transform(close) # 이미 학습된 모델이므로 금지, transform만 사용
        scaled = self.scaler.transform(close)
        x = torch.tensor(scaled).float().unsqueeze(0)

        with torch.no_grad():
            y_pred = self.model(x).item()

        predicted_price = self.scaler.inverse_transform([[y_pred]])[0][0]

        return {
            "predicted_price": round(predicted_price, 2),
            "latest_real_price": float(close[-1]),
            "diff": round(predicted_price - close[-1][0], 2)
        }
