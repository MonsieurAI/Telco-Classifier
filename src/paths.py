from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / 'data' / 'raw-data.csv'
RAW_TRAIN_PATH = BASE_DIR / 'data' / 'raw-train.csv'
RAW_TEST_PATH = BASE_DIR / 'data' / 'raw-test.csv'
TUNING_BEST_PATH = BASE_DIR / 'data' / 'tuning_best.json'
TUNING_RESULTS_PATH = BASE_DIR / 'data' / 'tuning-results.csv'

FINAL_PIPELINE_PATH = BASE_DIR / 'models' / 'final_pipeline.joblib'