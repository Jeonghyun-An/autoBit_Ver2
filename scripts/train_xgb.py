import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from app.services.data_loader import fetch_upbit_data

# 1. 데이터 로딩
df = fetch_upbit_data(count=200)
close = df['close'].values.reshape(-1, 1)

# 2. 스케일링
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

# 3. 특징 데이터 생성
def make_xgb_data(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback].flatten())
        y.append(data[i+lookback])
    return np.array(X), np.array(y).flatten()

X, y = make_xgb_data(scaled)

# 4. 모델 학습
model = xgb.XGBRegressor()
model.fit(X, y)

# 5. 저장
model.save_model("model_xgb.json")
print("✅ XGBoost 모델 저장 완료: model_xgb.json")
