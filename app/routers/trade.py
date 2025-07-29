from fastapi import APIRouter
from app.services.trading import excute_buy, excute_sell

router = APIRouter()

@router.post("/buy", summary="매수 요청", description="AI 전략 기반으로 비트코인 매수를 실행합니다.")
def buy_signal():
    result = excute_buy()
    return {"status": "buy", "result": result}

@router.post("/sell", summary="매도 요청", description="AI 전략 기반으로 비트코인 매도를 실행합니다.")
def sell_signal():
    result = excute_sell()
    return {"status": "sell", "result": result}