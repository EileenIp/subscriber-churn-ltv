"""All tunable parameters for the project live here — no magic numbers in the pipeline code."""
from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MEMBERS_PATH = RAW_DATA_DIR / "members_v3.csv"
TRANSACTIONS_PATH = RAW_DATA_DIR / "transactions.csv"
USER_LOGS_PATH = RAW_DATA_DIR / "user_logs_v2.csv"
# WSDM Cup reference label — validation check only, not our source of truth (see spec Phase 3).
TRAIN_LABELS_PATH = RAW_DATA_DIR / "train_v2.csv"

MEMBERS_COLUMNS = ["msno", "city", "bd", "gender", "registered_via", "registration_init_time"]
TRANSACTIONS_COLUMNS = [
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
]
USER_LOGS_COLUMNS = [
    "msno",
    "date",
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs",
]
TRAIN_LABELS_COLUMNS = ["msno", "is_churn"]

# transactions.csv is ~21.5M rows / 1.7GB; user_logs_v2.csv is ~18.4M rows / 1.4GB —
# read both in chunks to keep peak memory bounded. members_v3.csv (6.8M rows / 430MB)
# loads in one pass.
INGEST_CHUNKSIZE = 1_000_000

# --- Not yet set ---
# Churn threshold, observation window, prediction window, and member eligibility rule
# are Phase 3 decisions (spec Checkpoint 3) and must come from this data's own
# transaction-gap distribution, not be guessed here. Do not add them until that
# checkpoint is resolved.
