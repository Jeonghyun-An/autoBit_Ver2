from fastapi import APIRouter
from app.services.trading import excute_buy, excute_sell

router = APIRouter()

@router.post("/buy")
def buy_signal():
    result = excute_buy()
    return {status:"buy", "result": result}

@router.post("/sell")
def sell_signal():
    result = excute_sell()
    return {status:"sell", "result": result}