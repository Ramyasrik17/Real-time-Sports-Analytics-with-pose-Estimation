"""
baseline_recorder.py
Records a short clip of an athlete performing a movement CORRECTLY,
and saves the average joint angles as that athlete's personal baseline.

Usage:
    python baseline_recorder.py --name John --seconds 20
"""

import argparse
import json
import time
import os

import cv2
import mediapipe as mp

from pose_utils import extract_joint_angles, JOINTS

BASELINE_DIR = "baselines"


def record_baseline(athlete_name, seconds=20, camera_index=0):
    os.makedirs(BASELINE_DIR, exist_ok=True)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera_index or permissions.")

    all_angle_samples = {joint: [] for joint in JOINTS}
    start_time = time.time()

    print(f"Recording baseline for '{athlete_name}' — perform the movement correctly for {seconds}s...")

    while time.time() - start_time < seconds:
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
            for joint, value in angles.items():
                if value is not None:
                    all_angle_samples[joint].append(value)

        remaining = int(seconds - (time.time() - start_time))
        cv2.putText(frame, f"Recording baseline... {remaining}s left", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Baseline Recording - press Q to cancel", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    baseline_angles = {}
    for joint, samples in all_angle_samples.items():
        if samples:
            baseline_angles[joint] = sum(samples) / len(samples)
        else:
            baseline_angles[joint] = None

    out_path = os.path.join(BASELINE_DIR, f"{athlete_name}.json")
    with open(out_path, "w") as f:
        json.dump(baseline_angles, f, indent=2)

    print(f"Baseline saved to {out_path}")
    print(json.dumps(baseline_angles, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record an athlete's movement baseline.")
    parser.add_argument("--name", required=True, help="Athlete name (used as file name).")
    parser.add_argument("--seconds", type=int, default=20, help="Recording duration in seconds.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    args = parser.parse_args()

    record_baseline(args.name, seconds=args.seconds, camera_index=args.camera)
