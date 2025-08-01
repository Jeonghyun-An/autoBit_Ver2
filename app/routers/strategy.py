# app/routers/strategy.py

from fastapi import APIRouter, Query
from app.services.predictor import get_price_prediction
from app.services.strategy import auto_trade_strategy
from app.services.backtester import run_backtest
from app.services.counter import (
    get_trade_counts,
    reset_trade_counts,
    get_trade_log,
    get_trade_summary,
    get_trade_log_summary
)
from app.services.simulation_state import simulate_trade, reset_simulation, get_simulation_status
from app.services.strategy import strategy_registry
from app.services.data_loader import fetch_upbit_data
from app.loggers.trade_logger import log_trade


router = APIRouter()

@router.post("/auto-trade", tags=["Strategy"])
def auto_trade(
    model: str = Query("lstm", enum=["lstm", "xgb"], description="Model to use (lstm or xgb)"),
    threshold: float = Query(100000.0, description="매수/매도 임계값 (예: 100000.0)"),
    strategy_name: str = Query("basic", enum=["basic","ema","macd"],description="사용할 전략 이름 (기본값: basic)")
):
    """
    Run auto trading strategy using specified model and strategy.
    """
    return auto_trade_strategy(model, threshold, strategy_name)


@router.get("/counts", tags=["Strategy"])
def trade_counts():
    return get_trade_counts()

@router.post("/counts/reset", tags=["Strategy"])
def reset_counts():
    return reset_trade_counts()

@router.get("/logs", tags=["Strategy"])
def trade_logs():
    return get_trade_log()

@router.get("/summary", tags=["Strategy"])
def trade_summary():
    return get_trade_summary()

@router.get("/logs/summary", tags=["Strategy"])
def trade_log_summary():
    return get_trade_log_summary()

@router.post("/simulate-trade", tags=["Strategy"])
def simulate_auto_trade(
    model: str = Query("lstm", enum=["lstm", "xgb"]),
    threshold: float = Query(100000.0),
    strategy_name: str = Query("basic", enum=["basic", "ema", "macd"])
):
    prediction = get_price_prediction(model)
    if "error" in prediction:
        return prediction

    predicted = prediction["predicted_price"]
    real = prediction["latest_real_price"]
    diff = prediction["diff"]

    strategy_fn = strategy_registry.get(strategy_name)
    if not strategy_fn:
        return {"error": f"Unsupported strategy: {strategy_name}"}

    action = strategy_fn(predicted, real, threshold)
    simulate_trade(action, real, model, strategy_name)

    return {
        "action": action,
        "predicted_price": predicted,
        "real_price": real,
        "diff": diff,
        "simulated_result": get_simulation_status()
    }

@router.post("/simulate/reset", tags=["Strategy"])
def reset_simulation_status():
    reset_simulation()
    return {"message": "Simulation state reset."}

@router.get("/simulate/status", tags=["Strategy"])
def get_simulation():
    return get_simulation_status()

@router.get("/backtest", tags=["Strategy"])
def backtest_strategy(
    model: str = Query("lstm", enum=["lstm", "xgb"]),
    strategy_name: str = Query("basic", enum=["basic", "ema", "macd"]),
    threshold: float = Query(100000.0),
    count: int = Query(300, ge=100, le=1000)
):
    df = fetch_upbit_data(count=count)
    if df is None or df.empty:
        return {"error": "Failed to fetch data"}

    return run_backtest(model, strategy_name, threshold, df)