# ============================================================
# run_pipeline.py — Run the full traffic pipeline
# Usage:
#   py -3.11 run_pipeline.py          # Run full pipeline once
#   py -3.11 run_pipeline.py --loop   # Run every 15 minutes
# ============================================================

import sys, os, time, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from src.pipeline import run_pipeline
from config import FETCH_INTERVAL_MINUTES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop",     action="store_true")
    parser.add_argument("--interval", type=int, default=FETCH_INTERVAL_MINUTES)
    args = parser.parse_args()

    if args.loop:
        logger.info(f"⏰ Running every {args.interval} minutes. Ctrl+C to stop.")
        run_count = 0
        while True:
            run_count += 1
            logger.info(f"🔁 Run #{run_count}")
            run_pipeline()
            logger.info(f"💤 Sleeping {args.interval} min...\n")
            time.sleep(args.interval * 60)
    else:
        run_pipeline()

if __name__ == "__main__":
    main()
