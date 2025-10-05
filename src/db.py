# src/db.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://ubi:ubi@localhost:5432/ubi")
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

DDL = """
CREATE TABLE IF NOT EXISTS events(
  driver_id    TEXT NOT NULL,
  ts           TIMESTAMPTZ NOT NULL,
  lat          DOUBLE PRECISION,
  lon          DOUBLE PRECISION,
  speed_kph    DOUBLE PRECISION,
  accel_mps2   DOUBLE PRECISION,
  brake_mps2   DOUBLE PRECISION,
  heading_deg  DOUBLE PRECISION,
  odometer_km  DOUBLE PRECISION,
  phone_use    INT,
  road_type    TEXT,
  weather_code TEXT,
  PRIMARY KEY(driver_id, ts)
);
"""
def init_db():
    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)
