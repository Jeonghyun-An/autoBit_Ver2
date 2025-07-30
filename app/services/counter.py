# app/services/counter.py

import pandas as pd
from pathlib import Path

LOG_PATH = Path("logs/trade_log.csv")

def get_trade_counts():
    if not LOG_PATH.exists():
        return {"BUY": 0, "SELL": 0, "HOLD": 0}
    
    df = pd.read_csv(LOG_PATH)
    counts = df["action"].value_counts().to_dict()

    # 모든 항목을 기본값 0으로 포함
    return {
        "BUY": counts.get("BUY", 0),
        "SELL": counts.get("SELL", 0),
        "HOLD": counts.get("HOLD", 0)
    }
    
def reset_trade_counts():
    if LOG_PATH.exists():
        LOG_PATH.unlink()  # 로그 파일 삭제
    return {"message": "Trade counts reset successfully."}  

def get_trade_summary():
    if not LOG_PATH.exists():
        return {"message": "No trades recorded."}
    
    df = pd.read_csv(LOG_PATH)
    total_trades = len(df)
    buy_count = df[df["action"] == "BUY"].shape[0]
    sell_count = df[df["action"] == "SELL"].shape[0]
    hold_count = df[df["action"] == "HOLD"].shape[0]

    return {
        "total_trades": total_trades,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count
    }
    
def get_trade_log():
    if not LOG_PATH.exists():
        return {"message": "No trades recorded."}
    
    df = pd.read_csv(LOG_PATH)
    return df.to_dict(orient="records")  # DataFrame을 딕셔너리 리스트로 변환   

def get_trade_log_summary():
    if not LOG_PATH.exists():
        return {"message": "No trades recorded."}
    
    df = pd.read_csv(LOG_PATH)
    summary = df.groupby("action").agg(
        total_trades=("action", "count"),
        avg_predicted_price=("predicted_price", "mean"),
        avg_real_price=("real_price", "mean"),
        avg_diff=("diff", "mean")
    ).reset_index()
    
    return summary.to_dict(orient="records")  # DataFrame을 딕셔너리 리스트로 변환
