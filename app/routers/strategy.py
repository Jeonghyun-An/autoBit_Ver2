# app/routers/strategy.py

from fastapi import APIRouter, Query
from app.services.strategy import auto_trade_strategy
from app.services.counter import (
    get_trade_counts,
    reset_trade_counts,
    get_trade_log,
    get_trade_summary,
    get_trade_log_summary
)

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
