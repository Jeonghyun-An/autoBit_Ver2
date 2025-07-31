# app/loggers/trade_logger.py
import csv
from datetime import datetime
from pathlib import Path

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