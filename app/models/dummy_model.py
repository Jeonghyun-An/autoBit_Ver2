# app/models/dummy_model.py
import random

def dummy_predict():
    trend = random.choice(["상승", "하락", "횡보"])
    confidence = round(random.uniform(0.6, 0.99), 2)
    return {"trend": trend, "confidence": confidence}
