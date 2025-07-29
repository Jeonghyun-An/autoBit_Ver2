import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from app.models.lstm_model import LSTMPricePredictor
from app.services.data_loader import fetch_upbit_data
import joblib
from scripts.config_lstm import SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, LR, EPOCHS, COUNT, INPUT_SIZE, OUTPUT_SIZE, MODEL_PATH, SCALER_PATH


def make_sequences(data, seq_len=60):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(x), np.array(y)


def train_lstm():
    df = fetch_upbit_data(count=COUNT)
    if df is None or len(df) < SEQ_LEN:
        print("📉 데이터 부족. 최소 시퀀스 길이보다 작음.")
        return

    close = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    x_data, y_data = make_sequences(scaled, seq_len=SEQ_LEN)
    x_tensor = torch.tensor(x_data).float()
    y_tensor = torch.tensor(y_data).float()

    model = LSTMPricePredictor(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"🔁 학습 시작: seq_len={SEQ_LEN}, hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, lr={LR}, epochs={EPOCHS}")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    torch.save({
        "model_state_dict": model.state_dict()
    }, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ 모델 및 스케일러 저장 완료")


if __name__ == "__main__":
    train_lstm()
