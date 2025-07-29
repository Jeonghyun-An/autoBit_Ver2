# app/services/predictor.py
from fastapi import HTTPException
from app.models.model_registry import MODEL_REGISTRY
from app.services.data_loader import fetch_upbit_data

def get_price_prediction(model_name="lstm"):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"모델 '{model_name}' 은 등록되어 있지 않습니다.")

    model_cls = MODEL_REGISTRY[model_name]
    model_instance = model_cls()
    model_instance.load()

    df = fetch_upbit_data(count=60)
    if df is None or len(df) < 60:
        return {"error": "Upbit 데이터 부족"}

    return model_instance.predict(df)
