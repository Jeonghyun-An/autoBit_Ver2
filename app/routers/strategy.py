from fastapi import APIRouter, Query
from app.services.predictor import get_price_prediction
from app.services.trading import execute_buy, execute_sell

router = APIRouter()

@router.post("/auto-trade")
def auto_trade(model: str = Query("lstm", description="사용할 모델 이름"), threshold: int = 100000):
    result = get_price_prediction(model)
    
    if "error" in result:
        return result
    
    diff = result["diff"]
    action = None

    if diff > threshold:
        action = "BUY"
        execute_buy()
    elif diff < -threshold:
        action = "SELL"
        execute_sell()
    else:
        action = "HOLD"

    return {
        "action": action,
        "model": model,
        "threshold": threshold,
        "predicted_price": result["predicted_price"],
        "latest_real_price": result["latest_real_price"],
        "diff": diff
    }
