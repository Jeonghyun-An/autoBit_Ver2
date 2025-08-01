# app/services/model_manager.py
import os
import shutil
import json
from datetime import datetime

MODEL_DST = "model_lstm.pt"
SCALER_DST = "scaler_lstm.pkl"
INFO_DST = "model_info.json"

MODEL_BACKUP = "model_lstm.pt.bak"
SCALER_BACKUP = "scaler_lstm.pkl.bak"
INFO_BACKUP = "model_info.json.bak"

ROLLBACK_DIR = "rollback"
ROLLBACK_MODEL = os.path.join(ROLLBACK_DIR, "model.pt")
ROLLBACK_SCALER = os.path.join(ROLLBACK_DIR, "scaler.pkl")
ROLLBACK_INFO = os.path.join(ROLLBACK_DIR, "info.json")

def init_rollback_model(experiment_name: str):
    src_model = f"results/lstm/{experiment_name}/model.pt"
    src_scaler = f"results/lstm/{experiment_name}/scaler.pkl"
    src_info = f"results/lstm/{experiment_name}/metrics.txt"

    if not os.path.exists(src_model) or not os.path.exists(src_scaler):
        return {"error": "❌ 실험 모델 또는 스케일러가 존재하지 않습니다."}

    os.makedirs(ROLLBACK_DIR, exist_ok=True)
    shutil.copyfile(src_model, ROLLBACK_MODEL)
    shutil.copyfile(src_scaler, ROLLBACK_SCALER)

    metrics = {}
    if os.path.exists(src_info):
        with open(src_info, encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":")
                    metrics[k.strip()] = float(v.strip())

    rollback_info = {
        "source_experiment": experiment_name,
        "MSE": metrics.get("MSE"),
        "MAE": metrics.get("MAE"),
        "R2": metrics.get("R2"),
        "initialized": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(ROLLBACK_INFO, "w", encoding="utf-8") as f:
        json.dump(rollback_info, f, indent=2, ensure_ascii=False)

    return {"message": f"✅ '{experiment_name}' 롤백용 모델 저장 완료", "info": rollback_info}

def rollback_to_saved_model():
    if not os.path.exists(ROLLBACK_MODEL) or not os.path.exists(ROLLBACK_SCALER):
        return {"error": "❌ 롤백용 모델 파일이 없습니다. 먼저 /model/rollback/init 호출 필요"}

    # 백업
    if os.path.exists(MODEL_DST): shutil.copy2(MODEL_DST, MODEL_BACKUP)
    if os.path.exists(SCALER_DST): shutil.copy2(SCALER_DST, SCALER_BACKUP)
    if os.path.exists(INFO_DST): shutil.copy2(INFO_DST, INFO_BACKUP)

    # 복사
    shutil.copyfile(ROLLBACK_MODEL, MODEL_DST)
    shutil.copyfile(ROLLBACK_SCALER, SCALER_DST)
    if os.path.exists(ROLLBACK_INFO):
        shutil.copyfile(ROLLBACK_INFO, INFO_DST)

    return {"message": "🔁 롤백 모델로 되돌렸습니다."}