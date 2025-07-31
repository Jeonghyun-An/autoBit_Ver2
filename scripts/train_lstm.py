# train_lstm.py
import json
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from app.models.lstm_model import LSTMPricePredictor
from app.services.data_loader import fetch_upbit_data
import joblib
from scripts.config_lstm import SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, LR, EPOCHS, COUNT,INPUT_SIZE,OUTPUT_SIZE, MODEL_PATH, SCALER_PATH
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def make_sequences(data, seq_len):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])  # 종가만 예측
    return np.array(x), np.array(y)


def train_lstm():
    df = fetch_upbit_data(count=COUNT)
    if df is None or len(df) < SEQ_LEN:
        print("📉 데이터 부족. 최소 시퀀스 길이보다 작음.")
        return
    # 멀티 피처 확장 여부 판단
    if INPUT_SIZE == 4:
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        features = df[['close', 'volume', 'high', 'low']].values
    else:
        df = df[['timestamp', 'close']]
        features = df[['close']].values

    # close = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    # scaled = scaler.fit_transform(close)
    scaled = scaler.fit_transform(features)


    x_data, y_data = make_sequences(scaled, SEQ_LEN)
    x_tensor = torch.tensor(x_data).float()
    y_tensor = torch.tensor(y_data).float().unsqueeze(1)

    model = LSTMPricePredictor(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    losses = []

    print(f"🔁 학습 시작: seq_len={SEQ_LEN}, hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, lr={LR}, epochs={EPOCHS}")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        preds = model(x_tensor).numpy()
        y_true = y_tensor.numpy()

        if INPUT_SIZE == 4:
            preds_inv = scaler.inverse_transform(np.hstack([preds, np.zeros((len(preds), 3))]))[:, 0]
            y_inv = scaler.inverse_transform(np.hstack([y_true, np.zeros((len(y_true), 3))]))[:, 0]
        else:
            preds_inv = scaler.inverse_transform(preds)[:, 0]
            y_inv = scaler.inverse_transform(y_true)[:, 0]

    mse = mean_squared_error(y_inv, preds_inv)
    mae = mean_absolute_error(y_inv, preds_inv)
    r2 = r2_score(y_inv, preds_inv)

    print(f"\n📊 Evaluation - MSE: {mse:.4f}, MAE: {mae:.2f}, R2: {r2:.4f}")

    save_dir = "best_model"
    os.makedirs(save_dir, exist_ok=True)

    torch.save({"model_state_dict": model.state_dict()}, os.path.join(save_dir, "model.pt"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    with open(os.path.join(save_dir, "model_info.json"), "w") as f:
        json.dump({
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
        }, f, indent=4)
    
    print(f" 모델 저장 완료: {save_dir}")
    
    # 손실 그래프 저장
    plt.figure(figsize=(10, 5))                  # 그래프 사이즈 지정 (선택)
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
    plt.plot(y_inv, label="Actual", color="blue")
    plt.plot(preds_inv, label="Predicted", color="red")
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
