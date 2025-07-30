# app/routers/model.py
from fastapi import APIRouter, Query
from app.services.predictor import get_price_prediction

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
