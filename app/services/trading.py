# app/services/trading.py
from app.services.predictor import get_price_prediction
from app.loggers.trade_logger import log_trade

def execute_buy(model: str = "lstm", threshold: float = 100000):
    result = get_price_prediction(model)
    if "error" in result:
        return result
    
    if result["diff"] > threshold:
        log_trade("BUY", model, result["predicted_price"], result["latest_real_price"], result["diff"])
        return {"action": "BUY", **result}
    return {"action": "HOLD", **result}

def execute_sell(model: str = "lstm", threshold: float = 100000):
    result = get_price_prediction(model)
    if "error" in result:
        return result
    
    if result["diff"] < -threshold:
        log_trade("SELL", model, result["predicted_price"], result["latest_real_price"], result["diff"])
        return {"action": "SELL", **result}
    return {"action": "HOLD", **result}
