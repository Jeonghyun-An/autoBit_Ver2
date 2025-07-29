# app/services/predictor.py
from fastapi import HTTPException
from app.models.model_registry import MODEL_REGISTRY
from app.services.data_loader import fetch_upbit_data

def get_price_prediction(model_name="lstm"):
    from app.models.model_registry import MODEL_REGISTRY

    try:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"모델 '{model_name}' 등록 안됨")

        model_cls = MODEL_REGISTRY[model_name]
        model_instance = model_cls()
        model_instance.load()

        df = fetch_upbit_data(count=60)
        if df is None or len(df) < 60:
            return {"error": "데이터 부족"}

        result = model_instance.predict(df)
        print(f"[DEBUG] 예측 결과: {result}")
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"예측 중 오류 발생: {str(e)}"}
