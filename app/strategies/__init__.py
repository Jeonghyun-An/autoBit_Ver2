from .basic import basic_threshold_strategy
from .ema import ema_strategy
from .macd import macd_strategy

strategy_registry = {
    "basic": basic_threshold_strategy,
    "ema": ema_strategy,
    "macd": macd_strategy,
}
