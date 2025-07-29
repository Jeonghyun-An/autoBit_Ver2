from fastapi import FastAPI
from app.routers import trade, model, strategy

app = FastAPI(title="Bitconin AI AutoTrader")

app.include_router(trade.router, prefix="/trade", tags=["Trade"])
app.include_router(model.router, prefix="/model", tags=["Model"])
app.include_router(strategy.router, prefix="/strategy", tags=["Strategy"])