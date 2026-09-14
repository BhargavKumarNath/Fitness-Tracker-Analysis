"""Pure baseline prediction functions shared by batch exports and the dashboard."""

from __future__ import annotations


def predict_activity_baseline(steps: int, heart_rate: int) -> str:
    if heart_rate >= 150 or steps >= 15000:
        return "running"
    if heart_rate >= 125 or steps >= 8000:
        return "cycling"
    if steps == 0 and heart_rate < 90:
        return "yoga"
    return "walking"


def predict_calories_baseline(steps: int, heart_rate: int, sleep_hours: float, activity_type: str) -> float:
    activity_factor = {
        "walking": 1.0,
        "running": 1.35,
        "cycling": 1.2,
        "swimming": 1.1,
        "yoga": 0.75,
        "hiking": 1.45,
        "gym_workout": 1.15,
    }.get(activity_type, 1.0)
    sleep_adjustment = max(0.85, min(1.1, 1.0 + (7.5 - sleep_hours) * 0.02))
    return max(50.0, (80.0 + steps * 0.035 + max(0, heart_rate - 60) * 1.5) * activity_factor * sleep_adjustment)