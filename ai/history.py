"""
history.py
──────────
Saves and loads per-worker health records using a local SQLite database.
Each record stores all vitals + risk prediction with a timestamp.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = "worker_health.db"


def init_db():
    """Create the records table if it doesn't exist."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS health_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id   TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL,
            age         REAL,
            bmi         REAL,
            glucose     REAL,
            cholesterol REAL,
            oxygen      REAL,
            heart_rate  REAL,
            heat_exposure REAL,
            sleep_hours REAL,
            smoking     INTEGER,
            risk        TEXT,
            confidence  REAL
        )
    """)
    con.commit()
    con.close()


def save_record(worker_id: str, vitals: dict, risk: str, confidence: float):
    """Insert a new health record for a worker."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO health_records
        (worker_id, timestamp, age, bmi, glucose, cholesterol,
         oxygen, heart_rate, heat_exposure, sleep_hours, smoking, risk, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        worker_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        vitals.get("age"),
        vitals.get("bmi"),
        vitals.get("glucose"),
        vitals.get("cholesterol"),
        vitals.get("oxygen"),
        vitals.get("heart_rate"),
        vitals.get("heat_exposure"),
        vitals.get("sleep_hours"),
        vitals.get("smoking"),
        risk,
        confidence,
    ))
    con.commit()
    con.close()


def get_worker_history(worker_id: str) -> pd.DataFrame:
    """Return all records for a given worker as a DataFrame."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM health_records WHERE worker_id = ? ORDER BY timestamp",
        con,
        params=(worker_id,),
    )
    con.close()
    return df


def get_all_workers() -> list[str]:
    """Return list of all unique worker IDs."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT worker_id FROM health_records ORDER BY worker_id")
    workers = [row[0] for row in cur.fetchall()]
    con.close()
    return workers
