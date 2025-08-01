# app/routers/model.py
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse
import os
import json
from app.services.predictor import get_price_prediction ,get_recent_prediction_plot
from app.services.model_manager import init_rollback_model, rollback_to_saved_model

router = APIRouter()

@router.get("/predict")
def predict(model: str = Query("lstm", enum=["lstm", "xgb"], description="모델 이름 (lstm 등)")):
    return get_price_prediction(model)


@router.post("/train/{model}")
def train_model(model: str):
    if model == "lstm":
        from scripts.train_lstm import train_lstm
        train_lstm()
        return {"message": "LSTM trained"}
    elif model == "xgb":
        from scripts.train_xgb import train
        train()
        return {"message": "XGBoost trained"}
    else:
        return {"error": "Unsupported model"}

@router.get("/predict/plot")
def predict_plot(
    model: str = Query("lstm", enum=["lstm", "xgb"], description="모델 이름 (lstm 등)"),
    count: int = Query(300, ge=10, le=500, description="최근 N개 포인트"),
    use_experiment: bool = Query(False, description="실험 결과 그래프를 사용할지 여부 (True 시 고정된 prediction.png 반환)")
):
    plot_path = get_recent_prediction_plot(model=model, count=count, use_experiment=use_experiment)
    return FileResponse(plot_path, media_type="image/png")

@router.get("/model/info")
def get_model_info():
    info_path = "model_info.json"
    if not os.path.exists(info_path):
        return JSONResponse(status_code=404, content={"error": "model_info.json not found"})

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info

@router.post("/model/rollback/init")
def init_rollback(experiment_name: str = Query(..., description="저장할 실험 이름")):
    return init_rollback_model(experiment_name)

@router.post("/model/rollback/apply")
def rollback_apply():
    return rollback_to_saved_model()