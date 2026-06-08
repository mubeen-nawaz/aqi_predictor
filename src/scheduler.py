"""
Automated Pipeline Scheduler
No need for Apache Airflow - uses a simple scheduler.
The feature pipeline runs every hour, training runs daily.
"""

import schedule
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.features import run_feature_pipeline
from src.train    import train
from datetime     import datetime


def job_feature_pipeline():
    """Fetch AQI + weather data every hour."""
    print(f"\n⏰ [{datetime.now().strftime('%H:%M')}] Feature pipeline is running...")
    try:
        run_feature_pipeline()
    except Exception as e:
        print(f"❌ Feature pipeline error: {e}")


def job_train():
    """Retrain the model daily with new data."""
    print(f"\n🎓 [{datetime.now().strftime('%H:%M')}] Daily training is running...")
    try:
        train()
    except Exception as e:
        print(f"❌ Training error: {e}")


def main():
    print("🚀 AQI Predictor Scheduler started!")
    print("   📡 Feature pipeline: Every hour")
    print("   🤖 Model training:   Daily at 2 AM")
    print("   (Press Ctrl+C to stop)\n")

    # Run once now
    job_feature_pipeline()

    # Schedule setup
    schedule.every(1).hours.do(job_feature_pipeline)
    schedule.every().day.at("02:00").do(job_train)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
