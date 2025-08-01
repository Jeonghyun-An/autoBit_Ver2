# scripts/train_lstm.py
import json
import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from app.models.lstm_model import LSTMPricePredictor
from app.services.data_loader import fetch_upbit_data
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.config_lstm import SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, LR, EPOCHS, COUNT, INPUT_SIZE, OUTPUT_SIZE, MODEL_PATH, SCALER_PATH

def make_sequences(data, seq_len=60):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  # 종가 예측 기준
    return np.array(x), np.array(y)

def train_lstm():
    df = fetch_upbit_data(count=COUNT)
    features = ["close", "open", "high", "low"]  # 여기를 필요에 따라 수정해도 됨

    if df is None or len(df) < SEQ_LEN:
        print("📉 데이터 부족. 최소 시퀀스 길이보다 작음.")
        return

    feature_data = df[features].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_data)

    x_data, y_data = make_sequences(scaled, seq_len=SEQ_LEN)
    x_tensor = torch.tensor(x_data).float()
    y_tensor = torch.tensor(y_data).float().unsqueeze(1)

    model = LSTMPricePredictor(
        input_size=len(features),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    losses = []

    print(f"🔁 학습 시작: features={features}, seq_len={SEQ_LEN}, hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, lr={LR}, epochs={EPOCHS}")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    torch.save({
        "model_state_dict": model.state_dict()
    }, MODEL_PATH)

    # ✅ 스케일러 + 피처 정보 저장
    joblib.dump({
    "scaler": fitted_scaler,
    "features": ["close", "open", "high", "low", "volume"]  # 학습 시 사용한 feature
}, "scaler_lstm.pkl")


    print("✅ 모델 및 스케일러 저장 완료")

    # 예측 그래프 저장
    model.eval()
    with torch.no_grad():
        predicted = model(x_tensor).numpy()
        predicted_prices = scaler.inverse_transform(
            np.concatenate([predicted, np.zeros((predicted.shape[0], len(features)-1))], axis=1)
        )[:, 0]
        y_np = y_tensor.squeeze(1).numpy()
        real_prices = scaler.inverse_transform(
            np.concatenate([y_np.reshape(-1, 1), np.zeros((y_np.shape[0], len(features)-1))], axis=1)
        )[:, 0]

    mse = mean_squared_error(real_prices, predicted_prices)
    mae = mean_absolute_error(real_prices, predicted_prices)
    r2 = r2_score(real_prices, predicted_prices)

    print(f" MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 손실 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Train Loss")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)                               
    plt.legend()
    plt.tight_layout()                            #  레이아웃 조정 (잘림 방지)
    plt.savefig("lstm_loss.png", dpi=150)         #  해상도 업
    print("📉 Loss 그래프 저장 완료: lstm_loss.png")
    
    #  # 예측 결과 및 지표 계산
    # model.eval()
    # with torch.no_grad():
    #     predicted = model(x_tensor).numpy()
    #     predicted_prices = scaler.inverse_transform(predicted)
    #     y_np = y_tensor.squeeze(1).numpy()
    #     real_prices = scaler.inverse_transform(y_np.reshape(-1, 1))

    # mse = mean_squared_error(real_prices, predicted_prices)
    # mae = mean_absolute_error(real_prices, predicted_prices)
    # r2 = r2_score(real_prices, predicted_prices)

    # print(f" MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    info = {
    "mse": float(mse),
    "mae": float(mae),
    "r2": float(r2),
    "seq_len": SEQ_LEN,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "epochs": EPOCHS,
    "lr": LR,
    "input_size": INPUT_SIZE,
    "output_size": OUTPUT_SIZE,
    "count": COUNT
    }
    with open("model_info.json", "w") as f:
        json.dump(info, f, indent=4)


    # 예측 그래프 저장
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices, label="Actual Price", color="blue")
    plt.plot(predicted_prices, label="Predicted Price", color="red")
    plt.title("LSTM Prediction vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Price (KRW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lstm_prediction.png", dpi=150)

    print("📈 예측 결과 그래프 저장 완료: lstm_prediction.png")
    print("✅ LSTM 모델 학습 완료")



if __name__ == "__main__":
    train_lstm()
