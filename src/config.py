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

# --- Phase 3: churn definition and windowing (Eileen's decisions, Checkpoint 3, 2026-09-08) ---
#
# Churn: a member is "churned" if no new transaction occurs within CHURN_WINDOW_DAYS
# of a subscription's membership_expire_date. 30 days matches the official WSDM
# competition rule and is the point where this data's own renewal-gap distribution
# flattens out — 96.8% of all renewals-that-happen occur within 30 days of expiry;
# the rest is a long, thin tail (see the Phase 3 renewal-gap analysis in the session
# log / conversation history for the full curve).
CHURN_WINDOW_DAYS = 30

# Observation window: a member's features are built from their FULL transaction
# history up to the labeling cutoff, not a fixed rolling window — standard for
# subscription/LTV modeling, since tenure itself is informative.
OBSERVATION_WINDOW = "full_history_to_cutoff"

# Eligibility: a labeled expiry event is only used if the member has at least this
# many PRIOR transactions before it. This operationalizes "exclude mid-window
# registrants" under a full-history observation window — a member evaluated at
# their very first transaction has no real recency/frequency/trajectory history to
# build features from, which is the direct equivalent of a "mid-window registrant"
# in a fixed-window framing.
MIN_PRIOR_TRANSACTIONS = 1

# transactions.csv covers 2015-01-01 to 2017-02-28 (DATA_MAX_DATE). A labeled expiry
# event needs CHURN_WINDOW_DAYS of runway after it to be resolvable within the data,
# so only expiry events on or before 2017-01-29 can be used as training/validation
# labels. user_logs_v2.csv (March 2017) falls entirely after this boundary and is
# used for live scoring only, never for backtesting (Eileen's decision) — see
# spec Phase 0's "Scope decision" and Phase 3.
DATA_MAX_DATE = "2017-02-28"

# Out-of-time validation splits by the calendar date of the expiry event being
# labeled, not randomly (spec guardrail). Exact train/validation cutoff dates are
# set in Phase 4/5 once feature-matrix row volume per candidate split is visible —
# deferring that avoids picking an arbitrary date now that Phase 4 might need to
# revise anyway.
