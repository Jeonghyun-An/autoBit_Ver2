# app/loggers/trade_logger.py
import csv
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("logs/trade_log.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_trade(action: str, model: str, predicted: float, real: float, diff: float, strategy: str = "default"):
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
            round(predicted, 2),
            round(real, 2),
            round(diff, 2)
        ])
