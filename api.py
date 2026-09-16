"""
api.py
FastAPI backend: stores and serves athlete session history.
Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import db

app = FastAPI(title="Sports Analytics API")
db.init_db()


class SessionIn(BaseModel):
    athlete_name: str
    avg_score: float
    flagged_joints: str = ""  # comma-separated joint names, e.g. "left_knee,right_elbow"


@app.post("/sessions")
def create_session(session: SessionIn):
    db.save_session(session.athlete_name, session.avg_score, session.flagged_joints)
    return {"status": "saved"}


@app.get("/sessions/{athlete_name}")
def read_sessions(athlete_name: str):
    return db.get_sessions(athlete_name)


@app.get("/")
def root():
    return {"message": "Sports Analytics API is running"}
