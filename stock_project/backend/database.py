from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Định dạng: postgresql://user:password@host:port/db_name
SQLALCHEMY_DATABASE_URL = "postgresql://user:password123@stock-postgres:5432/stock_db"

# Lưu ý: Nếu bạn chạy FastAPI bên trong Docker cùng mạng với Postgres, 
# hãy đổi 'localhost' thành 'stock-postgres' (tên container)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()