"""
live_detector.py
Standalone test script: opens webcam, runs MediaPipe Pose, overlays the
skeleton + joint angles + deviation flags on the video window using OpenCV.

This is the quickest way to confirm your pose pipeline works before
plugging it into the Streamlit dashboard (app.py).

Usage:
    python live_detector.py --name John
(loads baselines/John.json if it exists; otherwise runs with no baseline comparison)
"""

import argparse
import json
import os

import cv2
import mediapipe as mp

from pose_utils import extract_joint_angles, compute_deviation_score

BASELINE_DIR = "baselines"


def load_baseline(athlete_name):
    path = os.path.join(BASELINE_DIR, f"{athlete_name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run(athlete_name=None, camera_index=0, tolerance_deg=15):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    baseline = load_baseline(athlete_name) if athlete_name else None
    if baseline:
        print(f"Loaded baseline for {athlete_name}.")
    else:
        print("No baseline loaded — running in detection-only mode (no deviation scoring).")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera_index or permissions.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            h, w = frame.shape[:2]
            angles = extract_joint_angles(results.pose_landmarks.landmark, w, h)

            y_offset = 30
            if baseline:
                score, deviations = compute_deviation_score(angles, baseline, tolerance_deg)
                cv2.putText(frame, f"Form Score: {score}", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                y_offset += 30
                for joint, info in deviations.items():
                    color = (0, 0, 255) if info["flagged"] else (0, 255, 0)
                    text = f"{joint}: {info['current']} deg"
                    cv2.putText(frame, text, (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 20
            else:
                for joint, value in angles.items():
                    if value is not None:
                        cv2.putText(frame, f"{joint}: {value:.0f} deg", (20, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += 20
        else:
            cv2.putText(frame, "No person detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Real-Time Pose Analysis - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live pose detection test.")
    parser.add_argument("--name", default=None, help="Athlete name (loads their baseline if present).")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    parser.add_argument("--tolerance", type=float, default=15, help="Deviation tolerance in degrees.")
    args = parser.parse_args()

    run(athlete_name=args.name, camera_index=args.camera, tolerance_deg=args.tolerance)
