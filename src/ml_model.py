# ============================================================
# src/ml_model.py — Random Forest for congestion & travel time
# ============================================================

import pandas as pd
import numpy as np
import os
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, mean_absolute_error, r2_score)
from config import TEST_SIZE, RANDOM_FOREST_ESTIMATORS, MODEL_PATH, LABEL_ENCODER_PATH, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Features used for ML
FEATURES = [
    "hour", "is_weekend", "is_morning_rush", "is_evening_rush",
    "is_lunch_peak", "is_night", "vehicle_count", "avg_speed_mph",
    "rolling_avg_vehicles", "rolling_avg_speed", "prev_vehicle_count",
    "weather_severity", "has_incident", "congestion_score"
]

INTERSECTION_TYPE_MAP = {"downtown": 0, "arterial": 1, "highway": 2, "suburban": 3}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix for model training/prediction."""
    df = df.copy()
    df["intersection_type_num"] = df["intersection_type"].map(INTERSECTION_TYPE_MAP).fillna(0)

    feature_cols = FEATURES + ["intersection_type_num"]
    X = df[feature_cols].fillna(0)
    return X


def train_congestion_classifier(df: pd.DataFrame) -> tuple:
    """
    Train Random Forest Classifier for congestion level prediction.
    Target: Low / Medium / High
    """
    logger.info("🌲 Training congestion classifier...")

    X = prepare_features(df)
    y = df["congestion_level"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
    )

    clf = RandomForestClassifier(
        n_estimators = RANDOM_FOREST_ESTIMATORS,
        max_depth    = 15,
        min_samples_split = 5,
        random_state = RANDOM_SEED,
        n_jobs       = -1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X, y_encoded, cv=5, scoring="accuracy")

    logger.info(f"✅ Congestion Classifier:")
    logger.info(f"   Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"   Cross-val (5-fold):    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    # Feature importance
    feat_cols = FEATURES + ["intersection_type_num"]
    importance_df = pd.DataFrame({
        "feature":   feat_cols,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)
    logger.info(f"\nTop 5 features:\n{importance_df.head(5).to_string()}")

    return clf, le, accuracy, cv_scores.mean(), importance_df


def train_travel_time_regressor(df: pd.DataFrame) -> tuple:
    """
    Train Random Forest Regressor for travel time prediction (minutes).
    """
    logger.info("⏱️  Training travel time regressor...")

    X = prepare_features(df)
    y = df["travel_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    reg = RandomForestRegressor(
        n_estimators = RANDOM_FOREST_ESTIMATORS,
        max_depth    = 15,
        min_samples_split = 5,
        random_state = RANDOM_SEED,
        n_jobs       = -1
    )
    reg.fit(X_train, y_train)

    # Evaluate
    y_pred = reg.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    logger.info(f"✅ Travel Time Regressor:")
    logger.info(f"   MAE (Mean Absolute Error): {mae:.4f} minutes")
    logger.info(f"   R² Score:                  {r2:.4f}")

    return reg, mae, r2


def save_models(clf, le, reg):
    """Save trained models to disk."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"classifier": clf, "regressor": reg}, f)

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    logger.info(f"💾 Models saved → {MODEL_PATH}")
    logger.info(f"💾 Label encoder → {LABEL_ENCODER_PATH}")


def load_models():
    """Load trained models from disk."""
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return models["classifier"], models["regressor"], le


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Run batch predictions on new data using saved models."""
    clf, reg, le = load_models()

    X = prepare_features(df)
    df = df.copy()

    # Congestion prediction
    congestion_encoded = clf.predict(X)
    df["predicted_congestion"] = le.inverse_transform(congestion_encoded)
    congestion_proba = clf.predict_proba(X)
    df["prediction_confidence"] = congestion_proba.max(axis=1).round(3)

    # Travel time prediction
    df["predicted_travel_time"] = reg.predict(X).round(2)

    logger.info(f"✅ Batch predictions complete: {len(df):,} records")
    return df


def run_ml_pipeline(df: pd.DataFrame) -> dict:
    """Train both models and return performance metrics."""
    clf, le, accuracy, cv_mean, importance = train_congestion_classifier(df)
    reg, mae, r2 = train_travel_time_regressor(df)
    save_models(clf, le, reg)

    metrics = {
        "congestion_accuracy":     round(accuracy * 100, 2),
        "congestion_cv_accuracy":  round(cv_mean * 100, 2),
        "travel_time_mae":         round(mae, 4),
        "travel_time_r2":          round(r2, 4),
        "feature_importance":      importance
    }
    return metrics


if __name__ == "__main__":
    import glob
    from config import PROCESSED_DATA_PATH
    files = sorted(glob.glob(os.path.join(PROCESSED_DATA_PATH, "traffic_processed_*.csv")))
    if not files:
        print("Run transformation first")
    else:
        df = pd.read_csv(files[-1])
        metrics = run_ml_pipeline(df)
        print(f"\n--- Model Performance ---")
        print(f"Congestion Accuracy:    {metrics['congestion_accuracy']}%")
        print(f"CV Accuracy (5-fold):   {metrics['congestion_cv_accuracy']}%")
        print(f"Travel Time MAE:        {metrics['travel_time_mae']} min")
        print(f"Travel Time R²:         {metrics['travel_time_r2']}")
