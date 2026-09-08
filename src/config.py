"""All tunable parameters for the project live here — no magic numbers in the pipeline code."""
from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

REG_DATA_PATH = RAW_DATA_DIR / "reg_data.csv"
AUTH_DATA_PATH = RAW_DATA_DIR / "auth_data.csv"
AB_TEST_DATA_PATH = RAW_DATA_DIR / "ab_test.csv"

# Source files use ';' as the field separator, not ','.
RAW_CSV_SEPARATOR = ";"

# auth_data.csv is ~9.6M rows / 170MB — read it in chunks to keep peak memory bounded.
INGEST_CHUNKSIZE = 500_000

REG_DATA_COLUMNS = ["reg_ts", "uid"]
AUTH_DATA_COLUMNS = ["auth_ts", "uid"]
AB_TEST_COLUMNS = ["user_id", "revenue", "testgroup"]

# --- Not yet set ---
# Churn threshold, observation window, prediction window, and player eligibility rule
# are Phase 3 decisions (spec Checkpoint 3) and must come from this data's own
# session-gap distribution, not be guessed here. Do not add them until that
# checkpoint is resolved.
