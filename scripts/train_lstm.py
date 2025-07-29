# scripts/train_lstm.py
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from app.models.lstm_model import LSTMPricePredictor

# NOTE: LSTM 모델 학습 → 저장 → 불러오기

# 데이터 로딩
from app.services.data_loader import fetch_upbit_data

df = fetch_upbit_data(count=200)
close = df['close'].values.reshape(-1, 1)

# 스케일링
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

model.train()
for epoch in range(50):
    optimizer.zero_grad()
    output = model(x_tensor)
    loss = loss_fn(output, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# 모델 및 스케일러 저장
torch.save({
    "model_state_dict": model.state_dict(),
    "scaler_min": scaler.data_min_,
    "scaler_max": scaler.data_max_
}, "model_lstm.pt")
