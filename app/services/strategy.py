# app/services/strategy.py
from app.services.predictor import get_price_prediction
from app.services.trading import execute_buy, execute_sell
from app.loggers.trade_logger import log_trade
from app.services.data_loader import fetch_upbit_data
from app.strategies import strategy_registry

def auto_trade_strategy(model: str, threshold: int, strategy_name: str = "basic"):
    prediction = get_price_prediction(model)
    
    if "error" in prediction:
        return prediction
    
    df = fetch_upbit_data()
    if df is None or df.empty:
        return {"error": "Failed to load data"}

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
        action=action,
        model=model,
        strategy=strategy_name,
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
