# Real-Time Sports Analytics with Pose Estimation

A fully Python-native pipeline (no React/WebRTC): OpenCV + MediaPipe for
pose detection, joint-angle math for biomechanical analysis, and a
Streamlit dashboard as the frontend.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Test basic pose detection

Confirm your webcam + MediaPipe pipeline works before anything else:

```bash
python live_detector.py
```

You should see your webcam feed with a skeleton overlay and live joint
angles printed on screen. Press `Q` to quit.

## 3. Record a personal baseline

Perform the movement you want to analyze (e.g., a squat, arm raise, or
sport-specific motion) **correctly** for ~20 seconds:

```bash
python baseline_recorder.py --name John --seconds 20
```

This saves `baselines/John.json` — the average "correct form" joint
angles for that athlete.

## 4. Re-run the live detector WITH baseline comparison

```bash
python live_detector.py --name John
```

Now you'll see a live form score and red/green flags per joint,
showing deviation from John's own baseline.

## 5. Launch the full dashboard (the actual deliverable)

```bash
streamlit run app.py
```

- Enter the athlete name (must match a recorded baseline to get scoring).
- Toggle "Start camera" in the sidebar.
- Watch the live video + score + joint table + trend chart update.
- Turn the camera off to save the session; history builds up below.

## 6. (Optional) Run the FastAPI backend separately

If you want a REST API layer (e.g., for a mobile app or external
service to pull athlete history later):

```bash
uvicorn api:app --reload --port 8000
```

Endpoints:
- `POST /sessions` — save a session `{athlete_name, avg_score, flagged_joints}`
- `GET /sessions/{athlete_name}` — fetch history

Note: `app.py` currently writes directly to SQLite via `db.py` for
simplicity. Wiring it to call the FastAPI endpoints instead (via
`requests`) is a straightforward next step once the core pipeline is
working end to end.

## Project structure

```
sports_analytics/
├── requirements.txt
├── pose_utils.py        # joint-angle math, shared by all scripts
├── db.py                 # SQLite storage helper
├── api.py                 # optional FastAPI backend
├── baseline_recorder.py    # records an athlete's personal baseline
├── live_detector.py         # quick OpenCV-window test script
├── app.py                    # Streamlit dashboard (the frontend)
└── baselines/                 # saved baseline JSON files (created at runtime)
```

## Suggested build/demo order

1. `live_detector.py` (no baseline) → prove pose detection works.
2. `baseline_recorder.py` → record your own baseline.
3. `live_detector.py --name <you>` → prove deviation scoring works.
4. `app.py` → wrap it all in the dashboard for the final demo.
5. (Stretch goal) Add multi-person support by upgrading the pose model,
   and wire `app.py` to call the FastAPI endpoints for storage.
