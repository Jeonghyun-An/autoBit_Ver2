from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from app.core.database import Base  # ✅ 여기서 Base 가져옴

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    model = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    diff = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
