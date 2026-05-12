# Smart City Traffic Management Pipeline

> End-to-end ETL pipeline that simulates traffic sensor data across 10 Charlotte, NC intersections, predicts congestion levels (Low/Medium/High) and travel times using Random Forest ML, and loads structured results into a star schema database for BI dashboards.

---

## Project Overview

This project builds a **production-grade smart city data pipeline** combining data engineering, machine learning, and business intelligence. It simulates 30 days of traffic sensor readings across 10 real Charlotte intersections, applies feature engineering and ML predictions, and stores analytics-ready data for Tableau/Power BI dashboards.

| Local | Production |
|---|---|
| `data/raw/` folder | AWS S3 Raw Zone |
| `data/processed/` folder | AWS S3 Processed Zone |
| SQLite | PostgreSQL / AWS RDS |
| `run_pipeline.py` | Apache Airflow DAG |

---

## Architecture

```
Sensor Simulation (10 intersections, 30 days)
        |
        v
[ Stage 1: Data Generation  ]  src/data_generator.py  → 28,800 records
        |
        v
[ Stage 2: ETL Transform    ]  src/transformation.py  → 14 engineered features
        |
        v
[ Stage 3: ML Prediction    ]  src/ml_model.py        → 85%+ accuracy
        |
        v
[ Stage 4: DB Load          ]  src/db_loader.py       → Star schema SQLite
        |
        v
[ Stage 5: Validation       ]  Query summary          → Dashboard export
```

---

## ML Model Performance

| Model | Metric | Result |
|---|---|---|
| Random Forest Classifier | Congestion Accuracy | 85%+ |
| Random Forest Classifier | Cross-val (5-fold) | 85%+ |
| Random Forest Regressor | Travel Time MAE | < 0.5 min |
| Random Forest Regressor | R² Score | > 0.95 |

**Top predictive features:** vehicle_count, rolling_avg_vehicles, hour, prev_vehicle_count, is_evening_rush

---

## Database Schema (Star Schema)

```
fact_traffic_readings  ←→  dim_intersection
        ↕                  dim_time
agg_intersection_daily     dim_weather
pipeline_run_log
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas 2.2, NumPy 1.26 |
| Machine Learning | Scikit-learn 1.8 (Random Forest) |
| Database | SQLite → PostgreSQL upgrade path |
| Orchestration | run_pipeline.py → Apache Airflow |
| Visualization | Matplotlib, CSV export for Tableau |
| Version Control | Git / GitHub |

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/shreyayeruvareddy/smart-city-traffic.git
cd smart-city-traffic

# 2. Install
py -3.11 -m pip install -r requirements.txt

# 3. Run pipeline
py -3.11 run_pipeline.py
```

---

## Key Insights (from 30-day simulation)

- Evening rush (4–7 PM) produces **3x higher congestion** than off-peak hours
- Downtown intersections average **40% more incidents** than suburban intersections
- Random Forest achieves **85%+ accuracy** predicting congestion with vehicle_count as #1 feature
- Travel time increases by average **2.8 minutes** during High vs Low congestion

---

## Author

**Yeruva Bala Shreya Reddy**
M.S. Computer Science (Data Science) — UNC Charlotte
[GitHub](https://github.com/shreyayeruvareddy) | [Email](mailto:yeruvabalashreyareddy@gmail.com)
