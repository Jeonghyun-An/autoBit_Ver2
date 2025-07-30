# app/routers/strategy.py

from fastapi import APIRouter, Query
from app.services.predictor import get_price_prediction
from app.services.trading import execute_buy, execute_sell

router = APIRouter()

@router.post("/auto-trade", tags=["Strategy"])
def auto_trade(
    model: str = Query("lstm", enum=["lstm", "xgb"], description="Model to use (lstm or xgb)"),
    threshold: float = Query(100000.0, description="매수/매도 임계값 (예: 100000.0)")
):
    """
    Execute automatic trading strategy based on model prediction.

    - If predicted price > real price + threshold → BUY
    - If predicted price < real price - threshold → SELL
    - Otherwise → HOLD
    """
    result = get_price_prediction(model)

    if "error" in result:
        return result

    diff = result.get("diff", 0)
    if diff > threshold:
        execute_buy()
        action = "BUY"
    elif diff < -threshold:
        execute_sell()
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "action": action,
        "model": model,
        "threshold": threshold,
        **result
    }
