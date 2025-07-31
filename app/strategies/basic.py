# 새로운 전략 등록을 위한 함수들
def basic_threshold_strategy(predicted, real, threshold):
    diff = predicted - real
    if diff > threshold:
        return "BUY"
    elif diff < -threshold:
        return "SELL"
    else:
        return "HOLD"
