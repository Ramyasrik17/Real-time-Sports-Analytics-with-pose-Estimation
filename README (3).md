# Real-Time Sports Analytics with Pose Estimation

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-00A98F?style=flat&logo=google&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)

Real-time computer vision system that uses pose estimation to analyze an athlete's movement, learn their personal baseline, and detect deviations from it — with live feedback through an interactive dashboard.

Unlike systems that grade every athlete against a single fixed "ideal" template, this project **learns each athlete's own correct-form baseline** and flags deviations from *their own* normal movement — making feedback personalized rather than generic.

---

## Problem Statement

Sports performance analysis often depends on manual observation, subjective coaching feedback, and expensive tracking equipment. These approaches make it difficult to continuously measure an athlete's movement quality and identify subtle changes in performance during training.

This project extracts human body landmarks from video, analyzes joint angles and movement consistency, and converts that into a measurable, real-time performance score — using nothing but a standard webcam.

---

## Demo Flow

```
Video Stream → Multi-Person Pose Estimation → Joint Angle Calculation
→ Form Deviation Detection → Real-Time Overlay Dashboard
```

1. Record a short clip of yourself performing a movement **correctly** → saved as your personal baseline.
2. Run live detection → the system compares your current joint angles against that baseline in real time.
3. View everything on a live dashboard: video feed with skeleton overlay, a live Form Score (0–100), per-joint deviation flags, and session history over time.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Computer Vision | OpenCV |
| Pose Estimation | MediaPipe |
| Deep Learning | PyTorch |
| Backend / API | FastAPI |
| Dashboard / Frontend | Streamlit |
| Storage | SQLite |

No React, no WebRTC — the entire pipeline runs in a single Python stack, keeping the project simple to build, run, and deploy locally.

---

## Features

- Live webcam pose detection with skeleton overlay
- Joint-angle calculation (elbows, shoulders, hips, knees) from body landmarks
- Personalized baseline recording — no generic "ideal form" comparison
- Real-time deviation scoring (0–100 Form Score) with per-joint flags
- Session history logging (SQLite) to track progress over time
- Fully interactive Streamlit dashboard — no frontend framework required

---

## Project Structure

```
sports_analytics/
├── requirements.txt
├── pose_utils.py          # joint-angle math, shared across scripts
├── db.py                  # SQLite storage helper
├── api.py                 # optional FastAPI backend
├── baseline_recorder.py   # records an athlete's personal baseline
├── live_detector.py       # quick OpenCV-window test script
├── app.py                 # Streamlit dashboard (the frontend)
└── baselines/             # saved baseline JSON files (created at runtime)
```

---

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

**1. Test basic pose detection**
```bash
python live_detector.py
```

**2. Record your personal baseline** (perform the movement correctly for 20s)
```bash
python baseline_recorder.py --name yourname --seconds 20
```

**3. Re-run detection with live deviation scoring**
```bash
python live_detector.py --name yourname
```

**4. Launch the full dashboard**
```bash
streamlit run app.py
```

**5. (Optional) Run the FastAPI backend**
```bash
uvicorn api:app --reload --port 8000
```

---

## What It Proves

Multi-object tracking, biomechanical (joint-angle) analysis, and real-time visualization — going beyond basic pose detection, while staying within a single, dependency-light Python stack.

---

## Limitations

- Not a medical-grade tool; not intended for injury diagnosis
- Best suited to movements clearly visible from a single 2D camera angle (e.g., side-on squats)
- Baseline quality depends on the accuracy of the initial "correct form" recording

---

## Roadmap

- [ ] Upgrade to a multi-person pose model for group/team training sessions
- [ ] Connect `app.py` to the FastAPI backend instead of writing directly to SQLite
- [ ] Add configurable rep-counting per movement type
- [ ] Export session history as PDF/CSV reports

---

## License

MIT
