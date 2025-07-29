import pandas as pd
from app.models.lstm_model import LSTMModel
from app.models.xgb_model import XGBModel
from app.services.data_loader import fetch_upbit_data

model_dict = {
    "lstm": LSTMModel,
    "xgb": XGBModel,
}

def get_price_prediction(model: str) -> dict:
    if model not in model_dict:
        return {"error": f"지원하지 않는 모델입니다: {model}"}

    try:
        df = fetch_upbit_data()
        if df is None or df.empty:
            return {"error": "데이터 로딩 실패"}

        model_class = model_dict[model]()
        model_class.load()
        result = model_class.predict(df)
        return result

    except Exception as e:
        return {"error": f"예측 중 오류 발생: {str(e)}"}
