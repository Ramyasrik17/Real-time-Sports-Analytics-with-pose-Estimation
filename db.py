"""
db.py
Minimal SQLite helper for storing athlete session scores/history.
"""

import sqlite3
from datetime import datetime

DB_PATH = "sessions.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            avg_score REAL NOT NULL,
            flagged_joints TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_session(athlete_name, avg_score, flagged_joints):
    conn = get_connection()
    conn.execute(
        "INSERT INTO sessions (athlete_name, timestamp, avg_score, flagged_joints) VALUES (?, ?, ?, ?)",
        (athlete_name, datetime.now().isoformat(timespec="seconds"), avg_score, flagged_joints),
    )
    conn.commit()
    conn.close()


def get_sessions(athlete_name):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE athlete_name = ? ORDER BY timestamp ASC",
        (athlete_name,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
