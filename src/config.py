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

# Out-of-time validation splits by a shared, external reference date, not by each
# member's own last transaction (see identify_labeled_events's docstring in
# src/features.py for why that first design leaked the label: a first attempt
# splitting by each member's own self-selected cutoff date produced 69% churn in
# train vs 7% in validation from the SAME population — the cutoff date itself was
# a function of churn status, since a churned member's history simply stops early.
# Fixed by evaluating every member as of the same fixed date per split, via
# build_train_validation_matrices).
#
# Real Phase 4 run with these cutoffs: TRAIN (every member's state as of
# TRAIN_CUTOFF_DATE) = 1,125,164 rows, 41.2% churn. VALIDATION (every member's
# state as of VALIDATION_CUTOFF_DATE, excluding anyone already used in train) =
# 217,977 rows, 30.6% churn. Zero member overlap. The ~10-point churn-rate gap
# between splits is normal out-of-time distribution drift, not a leakage signal —
# nowhere near the earlier 69%/7% artifact.
TRAIN_CUTOFF_DATE = "2016-06-30"
# The latest date a labeled transaction's expiry can still resolve within the data
# (DATA_MAX_DATE minus CHURN_WINDOW_DAYS) — maximizes validation recency.
VALIDATION_CUTOFF_DATE = "2017-01-29"

# --- Phase 4: feature engineering results (2026-09-08) ---
# Feature matrix (per member, per split): 23 columns, zero nulls in every column.
# Churn base rates (41.2% train / 30.6% validation) are notably higher than the
# well-known WSDM competition label's 9.0%. Not a bug: that label snapshots
# currently-active subscribers at one point in time, while ours evaluates every
# member with enough history as of a reference date — including many who lapsed
# long before that date and never returned. Document this gap explicitly in the
# write-up rather than letting it look like an error.
