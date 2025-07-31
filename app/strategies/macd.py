# MACD 기반 전략
# MACD는 Moving Average Convergence Divergence의 약자로, 주가의 이동평균을 이용하여 추세의 강도와 방향을 파악하는 지표입니다.
# MACD는 단기 이동평균과 장기 이동평균의 차이를 이용하여 계산됩니다.
# MACD는 주로 12일과 26일의 지수이동평균(EMA)을 사용하여 계산됩니다.
# MACD의 신호선은 MACD의 9일 EMA로 계산됩니다.
import pandas as pd

def macd_strategy(predicted, real, threshold, df: pd.DataFrame = None):
    if df is None or len(df) < 26:
        return "HOLD"

    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()

    if macd.iloc[-1] > signal.iloc[-1]:
        return "BUY"
    elif macd.iloc[-1] < signal.iloc[-1]:
        return "SELL"
    else:
        return "HOLD"
