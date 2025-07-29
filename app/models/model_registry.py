# app/models/model_registry.py
from app.models.lstm_model import LSTMModel
from app.models.xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "xgboost": XGBoostModel,
}
