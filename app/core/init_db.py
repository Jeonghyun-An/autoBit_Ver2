from app.core.database import engine, Base
from app.models.trade import Trade  # ✅ 모든 테이블 모델 import

def init_db():
    Base.metadata.create_all(bind=engine)
