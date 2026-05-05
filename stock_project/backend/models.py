from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from database import Base
import datetime

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    predicted_price = Column(Float)
    actual_price = Column(Float)
    model_type = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)