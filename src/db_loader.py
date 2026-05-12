# ============================================================
# src/db_loader.py — Star schema DB for traffic pipeline
# ============================================================

import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime
from config import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dim_intersection (
            intersection_id   TEXT PRIMARY KEY,
            intersection_name TEXT NOT NULL,
            intersection_type TEXT NOT NULL,
            latitude          REAL,
            longitude         REAL
        );

        CREATE TABLE IF NOT EXISTS dim_time (
            time_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL UNIQUE,
            date          TEXT,
            hour          INTEGER,
            day_of_week   TEXT,
            is_weekend    INTEGER,
            time_bucket   TEXT,
            month         INTEGER,
            week_of_year  INTEGER
        );

        CREATE TABLE IF NOT EXISTS dim_weather (
            weather_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            weather          TEXT NOT NULL UNIQUE,
            weather_severity INTEGER
        );

        CREATE TABLE IF NOT EXISTS fact_traffic_readings (
            reading_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            intersection_id       TEXT REFERENCES dim_intersection(intersection_id),
            time_id               INTEGER REFERENCES dim_time(time_id),
            weather_id            INTEGER REFERENCES dim_weather(weather_id),
            vehicle_count         INTEGER,
            avg_speed_mph         REAL,
            travel_time_min       REAL,
            congestion_level      TEXT,
            congestion_score      INTEGER,
            has_incident          INTEGER,
            incident_type         TEXT,
            rolling_avg_vehicles  REAL,
            rolling_avg_speed     REAL,
            predicted_congestion  TEXT,
            predicted_travel_time REAL,
            prediction_confidence REAL,
            ingested_at           TEXT
        );

        CREATE TABLE IF NOT EXISTS agg_intersection_daily (
            agg_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            intersection_id     TEXT,
            date                TEXT,
            avg_vehicles        REAL,
            avg_travel_time     REAL,
            avg_speed           REAL,
            total_incidents     INTEGER,
            high_congestion_pct REAL,
            created_at          TEXT
        );

        CREATE TABLE IF NOT EXISTS pipeline_run_log (
            run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp     TEXT,
            stage             TEXT,
            status            TEXT,
            records_processed INTEGER DEFAULT 0,
            error_message     TEXT,
            duration_sec      REAL
        );
    """)
    conn.commit()
    conn.close()
    logger.info("✅ Database schema created/verified")


def upsert_dimensions(df: pd.DataFrame) -> dict:
    conn = get_connection()
    cursor = conn.cursor()

    # dim_intersection
    for _, row in df[["intersection_id","intersection_name","intersection_type","latitude","longitude"]].drop_duplicates().iterrows():
        cursor.execute("""
            INSERT OR IGNORE INTO dim_intersection
            (intersection_id, intersection_name, intersection_type, latitude, longitude)
            VALUES (?,?,?,?,?)
        """, (row.intersection_id, row.intersection_name, row.intersection_type, row.latitude, row.longitude))

    # dim_weather
    for _, row in df[["weather","weather_severity"]].drop_duplicates().iterrows():
        cursor.execute("INSERT OR IGNORE INTO dim_weather (weather, weather_severity) VALUES (?,?)",
                       (row.weather, int(row.weather_severity)))

    # dim_time — batch insert unique timestamps
    time_cols = ["timestamp","date","hour","day_of_week","is_weekend","time_bucket","month","week_of_year"]
    for _, row in df[time_cols].drop_duplicates("timestamp").iterrows():
        cursor.execute("""
            INSERT OR IGNORE INTO dim_time
            (timestamp, date, hour, day_of_week, is_weekend, time_bucket, month, week_of_year)
            VALUES (?,?,?,?,?,?,?,?)
        """, (str(row.timestamp), row.date, int(row.hour), row.day_of_week,
              int(row.is_weekend), row.time_bucket, int(row.month), int(row.week_of_year)))

    conn.commit()

    # Build lookup dicts
    weather_map = {r[1]: r[0] for r in cursor.execute("SELECT weather_id, weather FROM dim_weather")}
    time_map    = {r[1]: r[0] for r in cursor.execute("SELECT time_id, timestamp FROM dim_time")}

    conn.close()
    return {"weather": weather_map, "time": time_map}


def load_fact_table(df: pd.DataFrame, lookup: dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    inserted = 0

    for _, row in df.iterrows():
        weather_id = lookup["weather"].get(row["weather"])
        time_id    = lookup["time"].get(str(row["timestamp"]))

        cursor.execute("""
            INSERT INTO fact_traffic_readings (
                intersection_id, time_id, weather_id,
                vehicle_count, avg_speed_mph, travel_time_min,
                congestion_level, congestion_score, has_incident, incident_type,
                rolling_avg_vehicles, rolling_avg_speed,
                predicted_congestion, predicted_travel_time, prediction_confidence,
                ingested_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            row.get("intersection_id"), time_id, weather_id,
            int(row.get("vehicle_count", 0)), float(row.get("avg_speed_mph", 0)),
            float(row.get("travel_time_min", 0)),
            row.get("congestion_level"), int(row.get("congestion_score", 0)),
            int(row.get("has_incident", 0)), row.get("incident_type", "None"),
            float(row.get("rolling_avg_vehicles", 0)), float(row.get("rolling_avg_speed", 0)),
            row.get("predicted_congestion"), row.get("predicted_travel_time"),
            row.get("prediction_confidence"), now
        ))
        inserted += 1

    conn.commit()
    conn.close()
    logger.info(f"✅ Inserted {inserted:,} records into fact_traffic_readings")
    return inserted


def load_daily_aggregates(df: pd.DataFrame):
    conn = get_connection()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    daily = df.groupby(["intersection_id", "date"]).agg(
        avg_vehicles        = ("vehicle_count",    "mean"),
        avg_travel_time     = ("travel_time_min",  "mean"),
        avg_speed           = ("avg_speed_mph",    "mean"),
        total_incidents     = ("has_incident",     "sum"),
        high_congestion_pct = ("congestion_score", lambda x: (x == 100).mean() * 100)
    ).reset_index().round(2)

    for _, row in daily.iterrows():
        conn.execute("""
            INSERT INTO agg_intersection_daily
            (intersection_id, date, avg_vehicles, avg_travel_time, avg_speed,
             total_incidents, high_congestion_pct, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (row.intersection_id, row.date, row.avg_vehicles, row.avg_travel_time,
              row.avg_speed, int(row.total_incidents), row.high_congestion_pct, now))

    conn.commit()
    conn.close()
    logger.info(f"✅ Inserted {len(daily):,} rows into agg_intersection_daily")


def log_run(stage, status, records=0, error=None, duration=None):
    conn = get_connection()
    conn.execute("""
        INSERT INTO pipeline_run_log (run_timestamp, stage, status, records_processed, error_message, duration_sec)
        VALUES (?,?,?,?,?,?)
    """, (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), stage, status, records, error, duration))
    conn.commit()
    conn.close()


def query_summary() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            i.intersection_name,
            i.intersection_type,
            COUNT(f.reading_id)                               AS total_readings,
            ROUND(AVG(f.vehicle_count), 1)                    AS avg_vehicles,
            ROUND(AVG(f.travel_time_min), 2)                  AS avg_travel_min,
            SUM(f.has_incident)                               AS total_incidents,
            ROUND(AVG(CASE WHEN f.congestion_level='High' THEN 1.0 ELSE 0 END)*100, 1) AS high_pct
        FROM fact_traffic_readings f
        JOIN dim_intersection i ON f.intersection_id = i.intersection_id
        GROUP BY i.intersection_name, i.intersection_type
        ORDER BY avg_vehicles DESC
    """, conn)
    conn.close()
    return df


def run_db_load(df: pd.DataFrame):
    import time
    t = time.time()
    try:
        create_tables()
        lookup = upsert_dimensions(df)
        n = load_fact_table(df, lookup)
        load_daily_aggregates(df)
        duration = round(time.time() - t, 2)
        log_run("db_load", "SUCCESS", n, duration=duration)
        logger.info(f"✅ DB load complete in {duration}s")
    except Exception as e:
        log_run("db_load", "FAILED", error=str(e))
        logger.error(f"❌ DB load failed: {e}")
        raise


if __name__ == "__main__":
    print(query_summary().to_string())
