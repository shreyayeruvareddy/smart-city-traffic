# ============================================================
# config.py — Central configuration for Smart City Traffic Pipeline
# ============================================================

# City & Sensor Setup
CITY = "Charlotte, NC"
NUM_INTERSECTIONS = 10

INTERSECTIONS = [
    {"id": "INT_001", "name": "Trade St & College St",       "lat": 35.2271, "lon": -80.8431, "type": "downtown"},
    {"id": "INT_002", "name": "Tryon St & 5th St",           "lat": 35.2290, "lon": -80.8410, "type": "downtown"},
    {"id": "INT_003", "name": "Independence Blvd & Sharon Amity", "lat": 35.2050, "lon": -80.7890, "type": "arterial"},
    {"id": "INT_004", "name": "South Blvd & Woodlawn Rd",    "lat": 35.1780, "lon": -80.8620, "type": "arterial"},
    {"id": "INT_005", "name": "N Tryon St & Sugar Creek Rd", "lat": 35.2680, "lon": -80.8350, "type": "arterial"},
    {"id": "INT_006", "name": "I-77 & I-85 Interchange",     "lat": 35.2410, "lon": -80.8760, "type": "highway"},
    {"id": "INT_007", "name": "Providence Rd & Fairview Rd", "lat": 35.1560, "lon": -80.7980, "type": "suburban"},
    {"id": "INT_008", "name": "Rea Rd & Ballantyne Commons", "lat": 35.0580, "lon": -80.8420, "type": "suburban"},
    {"id": "INT_009", "name": "W Mallard Creek & Harris Blvd","lat": 35.3290, "lon": -80.7420, "type": "suburban"},
    {"id": "INT_010", "name": "Airport entrance & Josh Birmingham", "lat": 35.2140, "lon": -80.9430, "type": "arterial"},
]

# Simulation Settings
SIMULATION_DAYS = 30          # Generate 30 days of historical data
READINGS_PER_HOUR = 4         # One reading every 15 minutes
RANDOM_SEED = 42

# Rush Hour Windows (24h format)
MORNING_RUSH = (7, 9)         # 7:00 AM – 9:00 AM
EVENING_RUSH = (16, 19)       # 4:00 PM – 7:00 PM
LUNCH_PEAK   = (11, 13)       # 11:00 AM – 1:00 PM

# Congestion Thresholds (vehicles per 15-min window)
CONGESTION_LOW    = 30        # 0–29 vehicles → Low
CONGESTION_MEDIUM = 60        # 30–59 vehicles → Medium
CONGESTION_HIGH   = 999       # 60+ vehicles → High

# Travel Time Baseline (minutes, by intersection type)
BASELINE_TRAVEL_TIME = {
    "downtown": 8,
    "arterial": 5,
    "highway":  3,
    "suburban": 4
}

# ML Model Settings
TEST_SIZE       = 0.2
RANDOM_FOREST_ESTIMATORS = 100
MODEL_PATH      = "models/traffic_model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# File Paths
RAW_DATA_PATH       = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
OUTPUT_PATH         = "outputs"
DB_PATH             = "data/traffic_pipeline.db"

# Pipeline Schedule
FETCH_INTERVAL_MINUTES = 15   # Real-time: every 15 minutes
