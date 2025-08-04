from dotenv import load_dotenv
from fastapi import FastAPI
from app.routers import trade, model, strategy
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.init_db import init_db

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ FastAPI 시작: DB 초기화 실행")
    init_db()
    yield

app = FastAPI(title="Bitcoin AI AutoTrader", lifespan=lifespan)

#  CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Nuxt dev 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  라우터 등록
app.include_router(trade.router, prefix="/trade", tags=["Trade"])
app.include_router(model.router, prefix="/model", tags=["Model"])
app.include_router(strategy.router, prefix="/strategy", tags=["Strategy"])
