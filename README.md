# Smart City Traffic Management Pipeline

> End-to-end data engineering pipeline simulating 28,800 traffic sensor readings across 10 real Charlotte, NC intersections — with feature engineering, Random Forest ML models for congestion and travel-time prediction, star schema database, and an interactive Tableau dashboard.

---

## Project Overview

This pipeline simulates traffic sensor data across 10 Charlotte intersections over 30 days (96 readings/day = 28,800 records), applies feature engineering, trains Random Forest models to predict congestion level and travel time, and loads results into a star schema database for BI reporting.

---

## Architecture

```
Data Generation (28,800 records, 10 intersections, 30 days)
        |
        v
[ Stage 1: Generate   ]  src/data_generator.py   → 28,800 sensor readings
        |
        v
[ Stage 2: Transform  ]  src/transformation.py   → 14 engineered features
        |
        v
[ Stage 3: ML Train   ]  src/ml_model.py         → Random Forest (85%+ accuracy)
        |
        v
[ Stage 4: DB Load    ]  src/db_loader.py        → Star schema SQLite
        |
        v
[ Stage 5: Validate   ]  Query summary           → Tableau Dashboard
```

---

## Key Results

| Metric | Result |
|---|---|
| Total Records | 28,800 |
| Intersections | 10 (real Charlotte locations) |
| Congestion Model Accuracy | 85%+ |
| Travel Time Model R² | > 0.95 |
| Pipeline Execution Time | 17.18 seconds |

---

## Key Business Insights

- **I-77 & I-85 Interchange**: Highest congestion at 22.5% high-congestion rate — priority for signal optimization
- **Airport entrance & Josh Birmingham**: Most incident-prone (97 incidents/30 days) — recommend increased monitoring
- **Highway intersections**: Fastest average speeds (36.4 mph) vs downtown (38.0 mph but more congested)
- **Downtown intersections**: Slowest travel times (8.5+ min) — signal coordination opportunity
- **Evening rush hour**: Produces 3x higher congestion than overnight baseline

---

## 📊 Tableau Dashboard

**Live Dashboard:** [Smart City Traffic Management Dashboard](https://public.tableau.com/app/profile/bala.shreya.reddy.yeruva/viz/SmartCityTrafficManagementDashboard/SmartCityTrafficManagementDashboardYeruvaBalaShreyaReddy)

Dashboard includes:
- **Congestion by Intersection** (bar chart) — I-77 & I-85 Interchange at 22.53% high-congestion rate
- **Incidents by Intersection** (bar chart) — Airport entrance leads with 97 incidents
- **Avg Speed by Intersection Type** (bar chart) — Highway vs Downtown vs Arterial vs Suburban
- **Traffic Volume** (packed bubble chart) — Visual comparison of average vehicle counts per intersection

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas 2.2, NumPy 1.26 |
| Machine Learning | Random Forest (Classifier + Regressor) |
| Database | SQLite → PostgreSQL upgrade path |
| Visualization | Tableau Dashboard |
| Version Control | Git / GitHub |

---

## Setup & Run

```bash
git clone https://github.com/shreyayeruvareddy/smart-city-traffic.git
cd smart-city-traffic
py -3.11 -m pip install -r requirements.txt
py -3.11 run_pipeline.py
```

---

## Author

**Yeruva Bala Shreya Reddy**
M.S. Computer Science (Data Science) — UNC Charlotte
[GitHub](https://github.com/shreyayeruvareddy) | [Email](mailto:yeruvabalashreyareddy@gmail.com)
