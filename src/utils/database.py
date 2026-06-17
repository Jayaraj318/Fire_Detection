from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import logging

# Logger
logger = logging.getLogger(__name__)

# Connection String provided by User
# Note: In a real prod env, this should be an env var. keeping it hardcoded as per immediate request context or assuming user has set it up?
# User provided: psql 'postgresql://neondb_owner:npg_68qSGaezWxwu@ep-green-star-adughbxb-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
# SQLAlchemy format: postgresql://user:password@host/dbname?sslmode=require

DATABASE_URL = "postgresql://neondb_owner:npg_68qSGaezWxwu@ep-green-star-adughbxb-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"

Base = declarative_base()

class DetectionEvent(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float)
    class_name = Column(String) # 'Fire', 'Smoke'
    image_path = Column(String, nullable=True) # Path to saved image if any (or S3 url)
    location = Column(String, default="Camera 1")
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True) # Bboxes etc
    status = Column(String, default="active") # active, resolved, critical
    severity = Column(String, default="normal") # normal, warning, critical

class SystemConfig(Base):
    __tablename__ = "system_config"

    key = Column(String, primary_key=True, index=True) # e.g. 'camera_source', 'email_settings'
    value = Column(Text) # JSON string or raw value
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def init_db():
    try:
        # Add timeout and pool_pre_ping for Neon stability
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 5}, pool_pre_ping=True)
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified successfully.")
        return sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e:
        logger.error(f"❌ Database initialization failed (Starting in Offline Mode): {e}")
        return None

# Global Session Local
SessionLocal = None

def get_db():
    if SessionLocal is None:
        return None
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
