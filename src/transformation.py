# ============================================================
# src/transformation.py — Clean, transform, feature engineer
# ============================================================

import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Load the most recent raw traffic CSV."""
    files = sorted(glob.glob(os.path.join(RAW_DATA_PATH, "traffic_raw_*.csv")))
    if not files:
        raise FileNotFoundError("No raw data files found — run data generation first")
    latest = files[-1]
    df = pd.read_csv(latest, parse_dates=["timestamp"])
    logger.info(f"📂 Loaded raw data: {latest}  ({len(df):,} rows)")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data quality rules."""
    initial = len(df)

    # Drop nulls in critical columns
    df = df.dropna(subset=["intersection_id", "timestamp", "vehicle_count", "congestion_level"])

    # Validate ranges
    df = df[df["vehicle_count"] >= 0]
    df = df[df["avg_speed_mph"].between(0, 120)]
    df = df[df["travel_time_min"] > 0]

    # Fill safe defaults
    df["incident_type"] = df["incident_type"].fillna("None")
    df["weather"]       = df["weather"].fillna("Clear")

    dropped = initial - len(df)
    if dropped:
        logger.warning(f"⚠️  Dropped {dropped} invalid records")
    logger.info(f"✅ Clean data: {len(df):,} valid records")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for ML model and analytics:
    - Time-based features (rush hour flags, time buckets)
    - Rolling averages per intersection
    - Congestion severity score
    - Spatial features
    """
    df = df.copy()

    # Time features
    df["is_morning_rush"] = df["hour"].between(7, 9).astype(int)
    df["is_evening_rush"] = df["hour"].between(16, 19).astype(int)
    df["is_lunch_peak"]   = df["hour"].between(11, 13).astype(int)
    df["is_night"]        = (df["hour"] < 6).astype(int)

    df["time_bucket"] = pd.cut(
        df["hour"],
        bins   = [-1, 5, 9, 13, 17, 21, 24],
        labels = ["Late Night", "Early Morning", "Morning Rush", "Afternoon", "Evening Rush", "Night"]
    ).astype(str)

    # Sort for rolling calculations
    df = df.sort_values(["intersection_id", "timestamp"]).reset_index(drop=True)

    # Rolling 4-period (1 hour) average vehicle count per intersection
    df["rolling_avg_vehicles"] = (
        df.groupby("intersection_id")["vehicle_count"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
        .round(1)
    )

    # Rolling avg speed per intersection
    df["rolling_avg_speed"] = (
        df.groupby("intersection_id")["avg_speed_mph"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
        .round(1)
    )

    # Congestion severity score (0-100)
    congestion_map = {"Low": 0, "Medium": 50, "High": 100}
    df["congestion_score"] = df["congestion_level"].map(congestion_map)

    # Lag features — previous reading's vehicle count
    df["prev_vehicle_count"] = (
        df.groupby("intersection_id")["vehicle_count"].shift(1).fillna(0)
    )

    # Weather severity
    weather_map = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Fog": 3}
    df["weather_severity"] = df["weather"].map(weather_map).fillna(0)

    # Congestion numeric for ML
    df["congestion_numeric"] = df["congestion_level"].map({"Low": 0, "Medium": 1, "High": 2})

    logger.info(f"✅ Feature engineering complete: {df.shape[1]} columns")
    return df


def compute_aggregates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute multiple aggregation levels for dashboards:
    - By intersection (overall)
    - By hour of day
    - By day of week
    - Daily summary
    """
    # By intersection
    by_intersection = df.groupby(["intersection_id", "intersection_name", "intersection_type"]).agg(
        avg_vehicles      = ("vehicle_count",    "mean"),
        max_vehicles      = ("vehicle_count",    "max"),
        avg_travel_time   = ("travel_time_min",  "mean"),
        avg_speed         = ("avg_speed_mph",    "mean"),
        total_incidents   = ("has_incident",     "sum"),
        high_congestion_pct = ("congestion_numeric", lambda x: (x == 2).mean() * 100),
        total_readings    = ("vehicle_count",    "count")
    ).reset_index().round(2)

    # By hour of day
    by_hour = df.groupby("hour").agg(
        avg_vehicles    = ("vehicle_count",   "mean"),
        avg_travel_time = ("travel_time_min", "mean"),
        avg_speed       = ("avg_speed_mph",   "mean"),
        incident_rate   = ("has_incident",    "mean"),
        high_congestion_pct = ("congestion_numeric", lambda x: (x == 2).mean() * 100)
    ).reset_index().round(2)

    # By day of week
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    by_day = df.groupby("day_of_week").agg(
        avg_vehicles    = ("vehicle_count",   "mean"),
        avg_travel_time = ("travel_time_min", "mean"),
        total_incidents = ("has_incident",    "sum")
    ).reindex(day_order).reset_index().round(2)

    # Daily summary
    daily = df.groupby("date").agg(
        total_vehicles  = ("vehicle_count",   "sum"),
        avg_travel_time = ("travel_time_min", "mean"),
        avg_speed       = ("avg_speed_mph",   "mean"),
        total_incidents = ("has_incident",    "sum"),
        high_congestion_pct = ("congestion_numeric", lambda x: (x == 2).mean() * 100)
    ).reset_index().round(2)

    logger.info(f"📊 Aggregates: {len(by_intersection)} intersections, {len(by_hour)} hours, {len(daily)} days")
    return {
        "by_intersection": by_intersection,
        "by_hour":         by_hour,
        "by_day":          by_day,
        "daily":           daily
    }


def save_processed_data(df: pd.DataFrame, aggs: dict) -> dict[str, str]:
    """Save all processed data to processed zone."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}

    # Main records
    main_path = os.path.join(PROCESSED_DATA_PATH, f"traffic_processed_{ts}.csv")
    df.to_csv(main_path, index=False)
    paths["main"] = main_path
    logger.info(f"💾 Processed records → {main_path}")

    # Aggregates
    for name, agg_df in aggs.items():
        agg_path = os.path.join(PROCESSED_DATA_PATH, f"agg_{name}_{ts}.csv")
        agg_df.to_csv(agg_path, index=False)
        paths[name] = agg_path
        logger.info(f"💾 Aggregate [{name}] → {agg_path}")

    return paths


def run_transformation(df_raw: pd.DataFrame = None) -> tuple[pd.DataFrame, dict]:
    """Main transformation entry point."""
    if df_raw is None:
        df_raw = load_raw_data()

    df = clean_data(df_raw)
    df = engineer_features(df)
    aggs = compute_aggregates(df)
    save_processed_data(df, aggs)

    return df, aggs


if __name__ == "__main__":
    df, aggs = run_transformation()
    print(f"\nProcessed shape: {df.shape}")
    print(f"\nBy intersection:\n{aggs['by_intersection'].to_string()}")
    print(f"\nBy hour (rush hours):\n{aggs['by_hour'][aggs['by_hour']['hour'].between(7,19)].to_string()}")
