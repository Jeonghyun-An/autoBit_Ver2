import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import MinMaxScaler
from app.services.data_loader import fetch_upbit_data

print("🔁 XGBoost 모델 학습 시작...")

# 1. 데이터 불러오기
df = fetch_upbit_data(300)
close = df['close'].values.reshape(-1, 1)

# 2. 정규화
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

# 3. 시계열 학습용 데이터 생성
X, y = [], []
seq_len = 30
for i in range(seq_len, len(scaled)):
    X.append(scaled[i - seq_len:i].flatten())
    y.append(scaled[i][0])

X, y = pd.DataFrame(X), pd.Series(y)

# 4. 모델 학습
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X, y)

# 5. 저장
joblib.dump(model, "model_xgb.pkl")
joblib.dump(scaler, "scaler_xgb.pkl")

print("✅ XGBoost 모델 저장 완료: model_xgb.pkl + scaler_xgb.pkl")
