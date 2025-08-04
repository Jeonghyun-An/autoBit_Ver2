# app/routers/trade.py
from fastapi import APIRouter
from app.services.trading import execute_buy, execute_sell
from sqlalchemy.orm import Session
from app.core.database import SessionLocal 
from app.models.trade import Trade         
from sqlalchemy import func

router = APIRouter()

@router.post("/buy", summary="매수 요청", description="AI 전략 기반으로 비트코인 매수를 실행합니다.")
def buy_signal():
    result = execute_buy()
    return {"status": "buy", "result": result}

@router.post("/sell", summary="매도 요청", description="AI 전략 기반으로 비트코인 매도를 실행합니다.")
def sell_signal():
    result = execute_sell()
    return {"status": "sell", "result": result}

@router.get("/trades", summary="트레이드 전체 조회")
def get_all_trades():
    db: Session = SessionLocal()
    trades = db.query(Trade).order_by(Trade.created_at.desc()).all()
    db.close()
    return [t.__dict__ for t in trades]

@router.get("/trades/summary", summary="트레이드 통계 요약")
def get_trade_summary():
    db: Session = SessionLocal()
    counts = db.query(
        Trade.action,
        func.count(Trade.id).label("count"),
        func.avg(Trade.diff).label("avg_diff"),
        func.avg(Trade.price).label("avg_price")
    ).group_by(Trade.action).all()
    db.close()

    return [
        {
            "action": row.action,
            "count": row.count,
            "avg_diff": round(row.avg_diff, 2) if row.avg_diff else 0,
            "avg_price": round(row.avg_price, 2) if row.avg_price else 0
        }
        for row in counts
    ]

@router.delete("/trades/reset", summary="트레이드 기록 초기화")
def reset_trades():
    db: Session = SessionLocal()
    db.query(Trade).delete()
    db.commit()
    db.close()
    return {"message": "✅ 트레이드 기록 초기화 완료"}