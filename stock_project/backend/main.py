from fastapi import FastAPI, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import models, schemas, database

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Stock Prediction Postgres API")

@app.post("/predictions/", response_model=schemas.PredictionResponse)
def create_prediction(prediction: schemas.PredictionCreate, db: Session = Depends(database.get_db)):
    db_prediction = models.Prediction(**prediction.dict())
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

@app.get("/predictions/", response_model=List[schemas.PredictionResponse])
def read_predictions(
    symbol: Optional[str] = None, 
    model_type: Optional[str] = None,
    limit: int = 50, 
    db: Session = Depends(database.get_db)
):
    query = db.query(models.Prediction)
    if symbol:
        query = query.filter(models.Prediction.symbol == symbol)
    if model_type:
        query = query.filter(models.Prediction.model_type == model_type)
        
    return query.order_by(models.Prediction.timestamp.desc()).limit(limit).all()