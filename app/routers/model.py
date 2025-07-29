# app/routers/model.py
from fastapi import APIRouter, Query
from app.services.predictor import get_price_prediction

router = APIRouter()

@router.get("/predict")
def predict(model: str = Query("lstm", description="모델 이름 (lstm 등)")):
    return get_price_prediction(model)
