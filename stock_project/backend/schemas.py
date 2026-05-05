from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, List

class PredictionBase(BaseModel):
    symbol: str
    predicted_price: float
    actual_price: float
    model_type: str

class PredictionCreate(PredictionBase):
    pass

class PredictionResponse(PredictionBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True