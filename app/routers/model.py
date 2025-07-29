# app/routers/model.py
from fastapi import APIRouter
from app.services.predictor import get_price_prediction

router = APIRouter()

@router.get("/predict")
def predict():
    prediction = get_price_prediction()
    return {"prediction": prediction}
