# app/services/strategy.py
from app.services.predictor import get_price_prediction
from app.services.trading import execute_buy, execute_sell
from app.loggers.trade_logger import log_trade

# 새로운 전략 등록을 위한 함수들
def basic_threshold_strategy(predicted, real, threshold):
    diff = predicted - real
    if diff > threshold:
        return "BUY"
    elif diff < -threshold:
        return "SELL"
    else:
        return "HOLD"

# 향후 EMA, MACD, etc 추가 가능
strategy_registry = {
    "basic": basic_threshold_strategy,
    # "ema": ema_strategy,
    # "macd": macd_strategy,
}

def auto_trade_strategy(model: str, threshold: int, strategy_name: str = "basic"):
    prediction = get_price_prediction(model)
    
    if "error" in prediction:
        return prediction

    predicted = prediction["predicted_price"]
    real = prediction["latest_real_price"]

    # 전략 선택
    strategy_fn = strategy_registry.get(strategy_name)
    if not strategy_fn:
        return {"error": f"Unsupported strategy: {strategy_name}"}
    
    action = strategy_fn(predicted, real, threshold)

    # 매매 실행
    if action == "BUY":
        result = execute_buy()
    elif action == "SELL":
        result = execute_sell()
    else:
        result = {"message": "No trade executed."}

    # 로그 저장
    log_trade(
        model=model,
        strategy=strategy_name,
        action=action,
        predicted_price=predicted,
        real_price=real,
        diff=prediction["diff"]
    )

    return {
        "action": action,
        "model": model,
        "strategy": strategy_name,
        "threshold": threshold,
        **prediction,
        "trade_result": result,
    }
