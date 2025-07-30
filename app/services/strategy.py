# app/services/strategy.py
from app.services.predictor import get_price_prediction
from app.services.trading import execute_buy, execute_sell
from app.loggers.trade_logger import log_trade
from app.services.data_loader import fetch_upbit_data

# 새로운 전략 등록을 위한 함수들
def basic_threshold_strategy(predicted, real, threshold):
    diff = predicted - real
    if diff > threshold:
        return "BUY"
    elif diff < -threshold:
        return "SELL"
    else:
        return "HOLD"

# 지수이동평균 기반
# EMA는 Exponential Moving Average의 약자로, 주가의 이동평균을 이용하여 추세의 강도와 방향을 파악하는 지표입니다.
# EMA는 최근 데이터에 더 많은 가중치를 부여하여 계산됩니다.
def ema_strategy(predicted, real, threshold, df=None):
    if df is None or len(df) < 10:
        return "HOLD"

    ema = df["close"].ewm(span=10).mean().iloc[-1]
    diff = predicted - ema

    if diff > threshold:
        return "BUY"
    elif diff < -threshold:
        return "SELL"
    else:
        return "HOLD"

# MACD 기반 전략
# MACD는 Moving Average Convergence Divergence의 약자로, 주가의 이동평균을 이용하여 추세의 강도와 방향을 파악하는 지표입니다.
# MACD는 단기 이동평균과 장기 이동평균의 차이를 이용하여 계산됩니다.
# MACD는 주로 12일과 26일의 지수이동평균(EMA)을 사용하여 계산됩니다.
# MACD의 신호선은 MACD의 9일 EMA로 계산됩니다.
def macd_strategy(predicted, real, threshold, df=None):
    if df is None or len(df) < 26:
        return "HOLD"

    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal
    latest_macd_diff = macd_hist.iloc[-1]

    if latest_macd_diff > threshold:
        return "BUY"
    elif latest_macd_diff < -threshold:
        return "SELL"
    else:
        return "HOLD"


# 향후 EMA, MACD, etc 추가 가능
strategy_registry = {
    "basic": basic_threshold_strategy,
    "ema": ema_strategy,
    "macd": macd_strategy,
}

def auto_trade_strategy(model: str, threshold: int, strategy_name: str = "basic"):
    prediction = get_price_prediction(model)
    
    if "error" in prediction:
        return prediction
    
    df = fetch_upbit_data()
    if df is None or df.empty:
        return {"error": "Failed to load data"}

    predicted = prediction["predicted_price"]
    real = prediction["latest_real_price"]

    # 전략 선택
    strategy_fn = strategy_registry.get(strategy_name)
    if not strategy_fn:
        return {"error": f"Unsupported strategy: {strategy_name}"}
    
    action = strategy_fn(predicted, real, threshold)

    # 매매 실행
    if action == "BUY":
        result = execute_buy()
    elif action == "SELL":
        result = execute_sell()
    else:
        result = {"message": "No trade executed."}

    # 로그 저장
    log_trade(
        model=model,
        strategy=strategy_name,
        action=action,
        predicted_price=predicted,
        real_price=real,
        diff=prediction["diff"]
    )

    return {
        "action": action,
        "model": model,
        "strategy": strategy_name,
        "threshold": threshold,
        **prediction,
        "trade_result": result,
    }
