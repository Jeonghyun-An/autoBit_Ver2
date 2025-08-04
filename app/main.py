from fastapi import FastAPI
from app.routers import trade, model, strategy
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Bitconin AI AutoTrader")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Nuxt dev 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(trade.router, prefix="/trade", tags=["Trade"])
app.include_router(model.router, prefix="/model", tags=["Model"])
app.include_router(strategy.router, prefix="/strategy", tags=["Strategy"])



