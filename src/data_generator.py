# ============================================================
# src/data_generator.py — Simulate realistic traffic sensor data
# Generates 30 days of historical data for 10 Charlotte intersections
# ============================================================

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from config import (INTERSECTIONS, SIMULATION_DAYS, READINGS_PER_HOUR,
                    RANDOM_SEED, MORNING_RUSH, EVENING_RUSH, LUNCH_PEAK,
                    CONGESTION_LOW, CONGESTION_MEDIUM, BASELINE_TRAVEL_TIME,
                    RAW_DATA_PATH, RANDOM_SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


def get_traffic_multiplier(hour: int, is_weekend: bool, intersection_type: str) -> float:
    """
    Returns a traffic volume multiplier based on:
    - Time of day (rush hours, lunch, overnight)
    - Weekend vs weekday patterns
    - Intersection type (downtown sees higher peaks)
    """
    if is_weekend:
        # Weekends: lighter morning, busier midday/afternoon
        if 0 <= hour < 7:    return 0.2
        if 7 <= hour < 10:   return 0.5
        if 10 <= hour < 14:  return 0.9
        if 14 <= hour < 18:  return 0.85
        if 18 <= hour < 21:  return 0.6
        return 0.25
    else:
        # Weekdays: classic double-hump rush hour pattern
        if 0 <= hour < 5:                          return 0.1
        if MORNING_RUSH[0] <= hour < MORNING_RUSH[1]: return 1.0  # Morning rush peak
        if LUNCH_PEAK[0] <= hour < LUNCH_PEAK[1]:    return 0.65  # Lunch peak
        if EVENING_RUSH[0] <= hour < EVENING_RUSH[1]: return 1.0  # Evening rush peak
        if 5 <= hour < MORNING_RUSH[0]:              return 0.35  # Pre-rush
        if MORNING_RUSH[1] <= hour < LUNCH_PEAK[0]:  return 0.55  # Mid-morning
        if LUNCH_PEAK[1] <= hour < EVENING_RUSH[0]:  return 0.50  # Afternoon
        if EVENING_RUSH[1] <= hour < 22:             return 0.40  # Post-rush
        return 0.15

    # Intersection type amplifiers
    type_amp = {"downtown": 1.3, "arterial": 1.0, "highway": 1.2, "suburban": 0.7}
    return multiplier * type_amp.get(intersection_type, 1.0)


def simulate_vehicle_count(hour: int, is_weekend: bool, intersection_type: str) -> int:
    """
    Simulate vehicle count per 15-minute window.
    Base capacity varies by intersection type with realistic noise.
    """
    base_capacity = {"downtown": 55, "arterial": 45, "highway": 70, "suburban": 30}
    base = base_capacity.get(intersection_type, 40)
    multiplier = get_traffic_multiplier(hour, is_weekend, intersection_type)

    # Add Gaussian noise (10% of base) + occasional incident spike
    noise = np.random.normal(0, base * 0.10)
    incident = np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, np.random.randint(15, 40)]) # ~11% chance
    count = max(0, int(base * multiplier + noise + incident))
    return count


def compute_travel_time(vehicle_count: int, baseline: float, intersection_type: str) -> float:
    """
    Estimate travel time using Bureau of Public Roads (BPR) function.
    travel_time = baseline * (1 + 0.15 * (volume/capacity)^4)
    """
    capacity = {"downtown": 55, "arterial": 45, "highway": 70, "suburban": 30}
    cap = capacity.get(intersection_type, 40)
    bpr = baseline * (1 + 0.15 * (vehicle_count / max(cap, 1)) ** 4)
    noise = np.random.normal(0, 0.3)
    return round(max(baseline, bpr + noise), 2)


def get_congestion_label(vehicle_count: int) -> str:
    """Classify congestion level based on vehicle count thresholds."""
    if vehicle_count < CONGESTION_LOW:    return "Low"
    if vehicle_count < CONGESTION_MEDIUM: return "Medium"
    return "High"


def get_weather_condition(hour: int) -> str:
    """Simulate weather conditions with realistic distribution."""
    conditions = ["Clear", "Clear", "Clear", "Clear", "Cloudy", "Cloudy", "Rain", "Fog"]
    # Fog more likely early morning; rain uniform
    if 5 <= hour <= 8:
        conditions += ["Fog", "Fog"]
    return np.random.choice(conditions)


def simulate_incident(vehicle_count: int, hour: int) -> tuple[bool, str]:
    """
    Simulate traffic incidents. Higher probability during rush hours
    and high-volume periods.
    """
    # Base incident probability: 2% per reading, higher during rush
    base_prob = 0.02
    if vehicle_count >= CONGESTION_MEDIUM:
        base_prob = 0.06
    if MORNING_RUSH[0] <= hour < MORNING_RUSH[1] or EVENING_RUSH[0] <= hour < EVENING_RUSH[1]:
        base_prob += 0.03

    has_incident = np.random.random() < base_prob
    if not has_incident:
        return False, "None"

    incident_types = ["Minor Accident", "Stalled Vehicle", "Road Work", "Debris on Road", "Pedestrian Delay"]
    return True, np.random.choice(incident_types)


def generate_sensor_reading(intersection: dict, timestamp: datetime) -> dict:
    """Generate a single sensor reading for one intersection at one timestamp."""
    hour       = timestamp.hour
    is_weekend = timestamp.weekday() >= 5
    itype      = intersection["type"]
    baseline   = BASELINE_TRAVEL_TIME[itype]

    vehicle_count = simulate_vehicle_count(hour, is_weekend, itype)
    travel_time   = compute_travel_time(vehicle_count, baseline, itype)
    congestion    = get_congestion_label(vehicle_count)
    weather       = get_weather_condition(hour)
    has_incident, incident_type = simulate_incident(vehicle_count, hour)

    # Speed estimate (mph): inversely related to congestion
    max_speed   = {"downtown": 30, "arterial": 45, "highway": 65, "suburban": 40}
    base_speed  = max_speed.get(itype, 35)
    speed_ratio = max(0.2, 1 - (vehicle_count / 80))
    avg_speed   = round(base_speed * speed_ratio + np.random.normal(0, 2), 1)

    return {
        "intersection_id":   intersection["id"],
        "intersection_name": intersection["name"],
        "intersection_type": itype,
        "latitude":          intersection["lat"],
        "longitude":         intersection["lon"],
        "timestamp":         timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "date":              timestamp.strftime("%Y-%m-%d"),
        "hour":              hour,
        "day_of_week":       timestamp.strftime("%A"),
        "is_weekend":        int(is_weekend),
        "vehicle_count":     vehicle_count,
        "avg_speed_mph":     avg_speed,
        "travel_time_min":   travel_time,
        "congestion_level":  congestion,
        "weather":           weather,
        "has_incident":      int(has_incident),
        "incident_type":     incident_type,
        "month":             timestamp.month,
        "week_of_year":      timestamp.isocalendar()[1],
    }


def generate_all_data(days: int = SIMULATION_DAYS) -> pd.DataFrame:
    """
    Generate complete historical dataset:
    - 30 days x 10 intersections x 96 readings/day (every 15 min)
    - Total: ~28,800 records
    """
    logger.info(f"🚦 Generating {days} days of traffic sensor data for {len(INTERSECTIONS)} intersections...")

    start_date = datetime.now() - timedelta(days=days)
    # Round to nearest 15 min
    start_date = start_date.replace(minute=0, second=0, microsecond=0)

    all_records = []
    interval_minutes = 60 // READINGS_PER_HOUR  # 15 minutes

    total_timestamps = days * 24 * READINGS_PER_HOUR
    current = start_date

    for t in range(total_timestamps):
        for intersection in INTERSECTIONS:
            record = generate_sensor_reading(intersection, current)
            all_records.append(record)
        current += timedelta(minutes=interval_minutes)

    df = pd.DataFrame(all_records)
    logger.info(f"✅ Generated {len(df):,} records ({days} days x {len(INTERSECTIONS)} intersections x {READINGS_PER_HOUR * 24} readings/day)")
    return df


def save_raw_data(df: pd.DataFrame) -> str:
    """Save raw simulated data to CSV in raw zone."""
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = os.path.join(RAW_DATA_PATH, f"traffic_raw_{timestamp}.csv")
    df.to_csv(filepath, index=False)
    logger.info(f"💾 Raw data saved → {filepath}  ({len(df):,} rows)")
    return filepath


def run_data_generation() -> tuple[pd.DataFrame, str]:
    """Main entry point for data generation stage."""
    df = generate_all_data()
    filepath = save_raw_data(df)
    return df, filepath


if __name__ == "__main__":
    df, path = run_data_generation()
    print(f"\nSample data:\n{df.head(3).to_string()}")
    print(f"\nCongestion distribution:\n{df['congestion_level'].value_counts()}")
    print(f"\nTotal records: {len(df):,}")
