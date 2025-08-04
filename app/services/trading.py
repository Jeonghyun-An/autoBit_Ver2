# app/services/trading.py
from app.services.predictor import get_price_prediction
from app.loggers.trade_logger import log_trade, log_trade_to_db



def execute_buy(model: str = "lstm", threshold: float = 100000, strategy: str = "default"):
    result = get_price_prediction(model)
    if "error" in result:
        return result
    
    if result["diff"] > threshold:
        log_trade(
            action="BUY",
            model=model,
            strategy=strategy,
            predicted_price=result["predicted_price"],
            real_price=result["latest_real_price"],
            diff=result["diff"]
        )
        return {"action": "BUY", **result}
    return {"action": "HOLD", **result}

def execute_sell(model: str = "lstm", threshold: float = 100000, strategy: str = "default"):
    result = get_price_prediction(model)
    if "error" in result:
        return result
    
    if result["diff"] < -threshold:
        log_trade(
            action="SELL",
            model=model,
            strategy=strategy,
            predicted_price=result["predicted_price"],
            real_price=result["latest_real_price"],
            diff=result["diff"]
        )
        return {"action": "SELL", **result}
    return {"action": "HOLD", **result}
