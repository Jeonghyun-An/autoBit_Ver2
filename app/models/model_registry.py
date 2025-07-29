# app/models/model_registry.py
from app.models.lstm_model import LSTMModel
from app.models.xgb_model import XGBModel

MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "xgb": XGBModel,
}
