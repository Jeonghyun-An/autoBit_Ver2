from fastapi import APIRouter
from app.services.trading import execute_buy, excute_sell, get_trade_history

router = APIRouter()

@router.post("/buy")
def buy_signal():
    result = execute_buy()
    return {status:"buy", "result": result}

@router.post("/sell")
def sell_signal():
    result = excute_sell()
    return {status:"sell", "result": result}