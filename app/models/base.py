# app/models/base.py
from abc import ABC, abstractmethod
import pandas as pd

class ModelBase(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> dict:
        pass
