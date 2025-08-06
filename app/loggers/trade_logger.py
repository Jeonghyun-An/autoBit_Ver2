# app/loggers/trade_logger.py
import csv
from datetime import datetime
from pathlib import Path
from app.models.trade import Trade
from app.core.database import SessionLocal
from app.ws.manager import ws_manager
import base64
from app.services.predictor import get_recent_prediction_plot 


LOG_PATH = Path("logs/trade_log.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# app/loggers/trade_logger.py

def log_trade(action: str, model: str, strategy: str, predicted_price: float, real_price: float, diff: float):
    print("로그 기록 시작")
    is_new = not LOG_PATH.exists()

    with open(LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(["timestamp", "action", "model", "strategy", "predicted_price", "real_price", "diff"])
        writer.writerow([
            datetime.now().isoformat(),
            action.upper(),
            model,
            strategy,
            round(predicted_price, 2),
            round(real_price, 2),
            round(diff, 2)
        ])
    print("로그 기록 완료")
    
async def log_trade_to_db(action, model, strategy, predicted_price, real_price, diff):
    db = SessionLocal()
    try:
        diff = float(diff) 
        trade = Trade(
            action=action.upper(),
            price=real_price,
            model=model,
            strategy=strategy,
            diff=diff
        )
        db.add(trade)
        db.commit()
        print("✅ DB에 트레이드 기록됨")
        await log_trade_to_ws(trade.__dict__)
    except Exception as e:
        db.rollback()
        print(f"❌ DB 기록 실패: {e}")
    finally:
        db.close()
        
        
async def log_trade_to_ws(trade_dict: dict):
    import json
    from app.services.predictor import get_recent_prediction_plot

    # ✅ 차트 이미지 읽기
    chart_path = get_recent_prediction_plot(model=trade_dict["model"], count=300, use_experiment=False)
    chart_base64 = None

    if Path(chart_path).exists():
        with open(chart_path, "rb") as f:
            chart_base64 = base64.b64encode(f.read()).decode("utf-8")

    # ✅ payload에 포함해서 전송
    payload = {
        "type": "trade_update",
        "payload": {
            "trade": trade_dict,
            "chart_base64": chart_base64
        }
    }

    await ws_manager.broadcast(json.dumps(payload, default=str))
