# scripts/run_lstm_experiments.py
import os
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from app.models.lstm_model import LSTMPricePredictor
from app.services.data_loader import fetch_upbit_data
from scripts.experiments_lstm import experiments
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def make_sequences(data, seq_len):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])  # 종가만 예측
    return np.array(x), np.array(y)

def run_experiment(config):
    print(f"\n🔬 실험: {config['name']} 시작")
    df = fetch_upbit_data(count=config["COUNT"])
    if df is None or len(df) < config["SEQ_LEN"]:
        print("❌ 데이터 부족")
        return

    # 멀티 피처 확장 여부 판단
    if config["INPUT_SIZE"] == 4:
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        features = df[['close', 'volume', 'high', 'low']].values
    else:
        df = df[['timestamp', 'close']]
        features = df[['close']].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    x_data, y_data = make_sequences(scaled, config["SEQ_LEN"])
    x_tensor = torch.tensor(x_data).float()
    y_tensor = torch.tensor(y_data).float().unsqueeze(1)

    model = LSTMPricePredictor(
        input_size=config["INPUT_SIZE"],
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
        output_size=config["OUTPUT_SIZE"]
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    losses = []

    for epoch in range(config["EPOCHS"]):
        model.train()
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        preds = model(x_tensor).numpy()
        y_true = y_tensor.numpy()

        if config["INPUT_SIZE"] == 4:
            # 첫 열에 종가를 두고 나머지는 0으로 채움
            preds_inv = scaler.inverse_transform(np.hstack([preds, np.zeros((len(preds), 3))]))[:, 0]
            y_inv = scaler.inverse_transform(np.hstack([y_true, np.zeros((len(y_true), 3))]))[:, 0]
        else:
            preds_inv = scaler.inverse_transform(preds)[:, 0]
            y_inv = scaler.inverse_transform(y_true)[:, 0]

        mse = mean_squared_error(y_inv, preds_inv)
        mae = mean_absolute_error(y_inv, preds_inv)
        r2 = r2_score(y_inv, preds_inv)

    save_dir = f"results/lstm/{config['name']}"
    os.makedirs(save_dir, exist_ok=True)

    torch.save({"model_state_dict": model.state_dict()}, os.path.join(save_dir, "model.pt"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    plt.figure()
    plt.plot(losses, label="Loss")
    plt.title(f"Loss: {config['name']}")
    plt.savefig(os.path.join(save_dir, "loss.png"))

    plt.figure()
    plt.plot(y_inv, label="Real", color="blue")
    plt.plot(preds_inv, label="Pred", color="red")
    plt.legend()
    plt.title(f"Prediction: {config['name']}")
    plt.savefig(os.path.join(save_dir, "prediction.png"))

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\n")

    print(f"✅저장 완료: {save_dir}")
    print(f" MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    for exp in experiments:
        run_experiment(exp)
