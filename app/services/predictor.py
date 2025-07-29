# app/services/predictor.py
from app.models.dummy_model import dummy_predict

def get_price_prediction():
    return dummy_predict()
