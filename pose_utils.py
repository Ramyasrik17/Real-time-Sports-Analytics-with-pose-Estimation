"""
pose_utils.py
Shared helper functions for joint-angle calculation and landmark handling.
Used by both the live detector script and the Streamlit dashboard.
"""

import numpy as np

# MediaPipe Pose landmark indices we care about, grouped by joint.
# Each joint angle is defined by three points: (start, middle/vertex, end)
JOINTS = {
    "left_elbow":    (11, 13, 15),   # left shoulder - left elbow - left wrist
    "right_elbow":   (12, 14, 16),   # right shoulder - right elbow - right wrist
    "left_shoulder": (23, 11, 13),   # left hip - left shoulder - left elbow
    "right_shoulder": (24, 12, 14),  # right hip - right shoulder - right elbow
    "left_hip":      (11, 23, 25),   # left shoulder - left hip - left knee
    "right_hip":     (12, 24, 26),   # right shoulder - right hip - right knee
    "left_knee":     (23, 25, 27),   # left hip - left knee - left ankle
    "right_knee":    (24, 26, 28),   # right hip - right knee - right ankle
}


def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b, formed by points a-b-c.
    a, b, c are each (x, y) tuples/arrays.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def extract_joint_angles(landmarks, frame_width, frame_height):
    """
    Given MediaPipe pose landmarks (normalized 0-1 coords) for ONE person,
    return a dict of {joint_name: angle_in_degrees}.

    landmarks: list-like of objects with .x, .y (e.g. results.pose_landmarks.landmark)
    """
    angles = {}

    def px(idx):
        lm = landmarks[idx]
        return (lm.x * frame_width, lm.y * frame_height)

    for joint_name, (i1, i2, i3) in JOINTS.items():
        try:
            p1, p2, p3 = px(i1), px(i2), px(i3)
            angles[joint_name] = calculate_angle(p1, p2, p3)
        except (IndexError, AttributeError):
            angles[joint_name] = None

    return angles


def compute_deviation_score(current_angles, baseline_angles, tolerance_deg=15):
    """
    Compare current joint angles against a baseline (mean angle per joint).
    Returns:
        overall_score (0-100, higher = closer to baseline)
        deviations: dict {joint_name: (current, baseline, deviation_deg, is_flagged)}
    """
    deviations = {}
    diffs = []

    for joint, baseline_val in baseline_angles.items():
        current_val = current_angles.get(joint)
        if current_val is None or baseline_val is None:
            continue
        diff = abs(current_val - baseline_val)
        is_flagged = diff > tolerance_deg
        deviations[joint] = {
            "current": round(current_val, 1),
            "baseline": round(baseline_val, 1),
            "deviation": round(diff, 1),
            "flagged": is_flagged,
        }
        diffs.append(diff)

    if not diffs:
        return 0, deviations

    avg_diff = sum(diffs) / len(diffs)
    # Simple scoring: 0 deviation = 100, deviation >= 45 deg = 0
    score = max(0, 100 - (avg_diff / 45) * 100)
    return round(score, 1), deviations
