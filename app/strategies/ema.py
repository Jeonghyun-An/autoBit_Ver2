import pandas as pd

# 지수이동평균 기반
# EMA는 Exponential Moving Average의 약자로, 주가의 이동평균을 이용하여 추세의 강도와 방향을 파악하는 지표입니다.
# EMA는 최근 데이터에 더 많은 가중치를 부여하여 계산됩니다.
def ema_strategy(predicted, real, threshold, df: pd.DataFrame = None):
    if df is None or len(df) < 20:
        return "HOLD"

    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()

    if df['ema_10'].iloc[-1] > df['ema_20'].iloc[-1]:
        return "BUY"
    elif df['ema_10'].iloc[-1] < df['ema_20'].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"
