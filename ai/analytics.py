"""
analytics.py
────────────
Aggregate analytics across all worker records.
Used for the management dashboard view.
"""

import sqlite3
import pandas as pd
from ai.history import init_db, DB_PATH


def get_risk_distribution() -> pd.DataFrame:
    """Count of Low / Medium / High risk across all workers (latest record per worker)."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT worker_id, risk, timestamp
        FROM health_records
        WHERE timestamp = (
            SELECT MAX(timestamp) FROM health_records h2
            WHERE h2.worker_id = health_records.worker_id
        )
    """, con)
    con.close()
    return df


def get_avg_vitals() -> pd.DataFrame:
    """Average vitals across all workers (latest record per worker)."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT glucose, cholesterol, oxygen, heart_rate, bmi, heat_exposure, sleep_hours
        FROM health_records
        WHERE timestamp = (
            SELECT MAX(timestamp) FROM health_records h2
            WHERE h2.worker_id = health_records.worker_id
        )
    """, con)
    con.close()
    return df


def get_high_risk_workers() -> pd.DataFrame:
    """All workers currently classified as High risk."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT worker_id, timestamp, glucose, oxygen, heart_rate, confidence
        FROM health_records
        WHERE risk = 'High'
        AND timestamp = (
            SELECT MAX(timestamp) FROM health_records h2
            WHERE h2.worker_id = health_records.worker_id
        )
        ORDER BY confidence DESC
    """, con)
    con.close()
    return df
