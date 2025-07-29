# scripts/train_lstm.py
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from app.models.lstm_model import LSTMPricePredictor
from app.services.data_loader import fetch_upbit_data
import joblib


# NOTE: LSTM 모델 학습 → 저장 → 불러오기

# 데이터 로딩
df = fetch_upbit_data(count=300)
if df is None or len(df) < 60:
    print("데이터 부족")
    exit()
close = df['close'].values.reshape(-1, 1)

# 정규화
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

# 시퀀스 데이터 생성
def make_sequences(data, seq_len=60):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(x), np.array(y)

x_data, y_data = make_sequences(scaled)

x_tensor = torch.tensor(x_data).float()
y_tensor = torch.tensor(y_data).float()

# 모델 학습
model = LSTMPricePredictor()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🔁 LSTM 모델 학습 시작...")
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_tensor)
    loss = loss_fn(output, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# 모델 및 스케일러 저장
torch.save({
    "model_state_dict": model.state_dict(),
}, "model_lstm.pt")

joblib.dump(scaler, "scaler_lstm.pkl")

print("✅ LSTM 모델 저장 완료: model_lstm.pt")