import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, JSON, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, scoped_session
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
import secrets
from passlib.context import CryptContext
from dotenv import load_dotenv
from config import settings

load_dotenv()
DATABASE_URL = settings.DATABASE_URL
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key = True, index = True)
    email = Column(String, unique= True, index = True, nullable = False)
    hashed_password = Column(String,  nullable = False)
    api_key = Column(String, unique=True, index=True, default=lambda: secrets.token_urlsafe(32))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now(datetime.timezone.utc))

    datasets = relationship("Dataset", back_populates="owner")
    models = relationship("MLModel", back_populates="owner")

    def verify_password(self, plain_password):
        return pwd_context.verify(plain_password, self.hashed_password)

    @staticmethod
    def get_password_hash(password):
        return pwd_context.hash(password)
    
class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key = True, index = True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    storage_path = Column(String, nullable=False) 

    schema_map = Column(JSON, nullable=False)

    is_processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now(datetime.timezone.utc))

    owner = relationship("User", back_populates="datasets")
    models = relationship("MLModel", back_populates="dataset")

class MLModel(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key = True, index = True)
    user_id = Column(Integer, ForeignKey("users.id"))

    dataset_id = Column(Integer, ForeignKey("datasets.id"))

    model_type = Column(String, nullable=False) 

    mlflow_run_id = Column(String)

    status = Column(String, default="queued")
    
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="models")
    dataset = relationship("Dataset", back_populates="models")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()