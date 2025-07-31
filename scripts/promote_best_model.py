# scripts/promote_best_model.py
import os
import shutil
import json
from datetime import datetime

EXPERIMENT_NAME = "long_seq_high_hidden"  # 수동 선택 or sys.argv로 받게 변경 가능
EXPERIMENT_DIR = f"results/lstm/{EXPERIMENT_NAME}"
MODEL_SRC = os.path.join(EXPERIMENT_DIR, "model.pt")
SCALER_SRC = os.path.join(EXPERIMENT_DIR, "scaler.pkl")
METRICS_SRC = os.path.join(EXPERIMENT_DIR, "metrics.txt")

MODEL_DST = "model_lstm.pt"
SCALER_DST = "scaler_lstm.pkl"
INFO_DST = "model_info.json"

if not os.path.exists(MODEL_SRC) or not os.path.exists(SCALER_SRC):
    raise FileNotFoundError("❌ 모델 또는 스케일러 파일이 존재하지 않습니다.")

# 파일 복사
shutil.copyfile(MODEL_SRC, MODEL_DST)
shutil.copyfile(SCALER_SRC, SCALER_DST)

# 지표 추출
metrics = {}
if os.path.exists(METRICS_SRC):
    with open(METRICS_SRC, "r", encoding="utf-8") as f:
        for line in f:
            key, val = line.strip().split(":")
            metrics[key.strip()] = float(val.strip())

# model_info.json 생성
model_info = {
    "source_experiment": EXPERIMENT_NAME,
    "MSE": metrics.get("MSE"),
    "MAE": metrics.get("MAE"),
    "R2": metrics.get("R2"),
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(INFO_DST, "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)

print(f"✅ 모델 및 정보가 프로덕션용으로 복사되었습니다 → {MODEL_DST}, {INFO_DST}")
