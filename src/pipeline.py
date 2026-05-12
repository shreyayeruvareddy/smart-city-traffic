# ============================================================
# src/pipeline.py — End-to-end traffic pipeline orchestrator
# Stages: Generate >> Transform >> ML >> Load >> Validate
# ============================================================

import time, logging, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    start = time.time()
    run_id = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info(f"🚦 TRAFFIC PIPELINE STARTED  |  run_id: {run_id}")
    logger.info("=" * 60)

    # ── STAGE 1: DATA GENERATION ──────────────────────────────
    logger.info("\n📡 STAGE 1: Traffic Sensor Data Generation")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.data_generator import run_data_generation
        df_raw, _ = run_data_generation()
        logger.info(f"✅ Stage 1 complete in {round(time.time()-t,2)}s | {len(df_raw):,} records")
    except Exception as e:
        logger.error(f"❌ Stage 1 FAILED: {e}"); return False

    # ── STAGE 2: TRANSFORMATION ───────────────────────────────
    logger.info("\n🔄 STAGE 2: ETL Transformation & Feature Engineering")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.transformation import run_transformation
        df, aggs = run_transformation(df_raw)
        logger.info(f"✅ Stage 2 complete in {round(time.time()-t,2)}s | {df.shape[1]} features")
    except Exception as e:
        logger.error(f"❌ Stage 2 FAILED: {e}"); return False

    # ── STAGE 3: ML TRAINING & PREDICTION ────────────────────
    logger.info("\n🌲 STAGE 3: ML Model Training & Batch Prediction")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.ml_model import run_ml_pipeline, predict_batch
        metrics = run_ml_pipeline(df)
        df = predict_batch(df)
        logger.info(f"✅ Stage 3 complete in {round(time.time()-t,2)}s")
        logger.info(f"   Congestion Accuracy: {metrics['congestion_accuracy']}%")
        logger.info(f"   Travel Time MAE:     {metrics['travel_time_mae']} min")
        logger.info(f"   Travel Time R²:      {metrics['travel_time_r2']}")
    except Exception as e:
        logger.error(f"❌ Stage 3 FAILED: {e}"); return False

    # ── STAGE 4: DATABASE LOAD ────────────────────────────────
    logger.info("\n🗄️  STAGE 4: Database Load")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.db_loader import run_db_load
        run_db_load(df)
        logger.info(f"✅ Stage 4 complete in {round(time.time()-t,2)}s")
    except Exception as e:
        logger.error(f"❌ Stage 4 FAILED: {e}"); return False

    # ── STAGE 5: VALIDATION ───────────────────────────────────
    logger.info("\n✅ STAGE 5: Validation Summary")
    logger.info("-" * 40)
    try:
        from src.db_loader import query_summary
        print("\n" + query_summary().to_string())
    except Exception as e:
        logger.warning(f"⚠️  Validation warning: {e}")

    total = round(time.time() - start, 2)
    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 PIPELINE COMPLETE  |  Total time: {total}s")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    run_pipeline()
