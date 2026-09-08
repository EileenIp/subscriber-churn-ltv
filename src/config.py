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
# state as of VALIDATION_CUTOFF_DATE) = 268,828 rows, 39.7% churn. Zero member
# overlap. tenure_days distribution matches closely between splits (train median
# 893 days, validation median 898) confirming the two populations are genuinely
# comparable, not systematically different member cohorts.
#
# Members are split into disjoint TRAIN/VALIDATION pools BEFORE any time-based
# eligibility logic runs (see build_train_validation_matrices) — an earlier
# version guaranteed disjointness by excluding "already used in train" members
# from validation instead, which silently restricted validation to only
# newer/shorter-tenured members (anyone long-tenured enough to be eligible by the
# train cutoff was always claimed by train first). That biased population
# composition, not genuine time-based drift, and it broke LightGBM badly (ROC-AUC
# fell BELOW 0.5) while logistic regression partially masked it through feature
# scaling. Caught via the tenure_days distribution comparison above.
TRAIN_CUTOFF_DATE = "2016-06-30"
# The latest date a labeled transaction's expiry can still resolve within the data
# (DATA_MAX_DATE minus CHURN_WINDOW_DAYS) — maximizes validation recency.
VALIDATION_CUTOFF_DATE = "2017-01-29"

# --- Phase 4: feature engineering results (2026-09-08, corrected 2026-09-09) ---
# Feature matrix (per member, per split): 24 columns (23 original + days_since_cutoff,
# added after a Phase 5 bug — see below), zero nulls in every column. Final churn
# base rates: 41.2% train / 39.7% validation. Both notably higher than the
# well-known WSDM competition label's 9.0%. Not a bug: that label snapshots
# currently-active subscribers at one point in time, while ours evaluates every
# member with enough history as of a reference date — including many who lapsed
# long before that date and never returned. Document this gap explicitly in the
# write-up rather than letting it look like an error.
#
# days_since_cutoff (reference_date - cutoff_date) was added during Phase 5 after
# a tuned LightGBM showed near-perfect PR-AUC (~0.97) on a random in-snapshot
# split but collapsed to 0.54 on real out-of-time validation. Root cause: within
# any snapshot, how long ago a member's labeled transaction was RELATIVE TO THE
# SCORING DATE is a near-deterministic churn signal (non-churned members' labeled
# transaction sits a median 14 days before the reference date; churned members'
# sits a median 151 days before it) — but that signal wasn't given to the model
# directly, so it had to reconstruct it indirectly from other features, which
# memorized the specific snapshot it was tuned on instead of learning something
# that transfers. Adding it directly (fully known at scoring time, not leaky)
# fixed this — see src/features.py's build_member_features docstring.

# --- Phase 5: model results (2026-09-09) ---
# Real validation-set numbers, all three approaches, same split:
#   Recency rule (days_since_last_transaction)  ROC-AUC=0.567  PR-AUC=0.618
#   LightGBM (simple, untuned)                  ROC-AUC=0.676  PR-AUC=0.647
#   Logistic regression                         ROC-AUC=0.936  PR-AUC=0.925
# LightGBM beats the recency baseline (a real, if modest, win). Logistic
# regression is the strongest performer of the three — reported honestly rather
# than downplayed in favour of the "primary" model, consistent with the spec's
# own baseline-first philosophy. Plausible explanation, not yet independently
# verified: days_since_cutoff is a dominant, close-to-monotonic predictor, which
# favours a linear model; LightGBM's tree splits may be less efficient at
# capturing a single smooth dominant relationship than a scaled linear
# coefficient is.
#
# An explicit hyperparameter tuning attempt (random stratified internal split,
# early stopping) reached train_eval PR-AUC ~0.986 but did NOT improve real
# validation performance over the simple untuned model (0.638 vs 0.647) — the
# simple model was kept. A high in-snapshot PR-AUC is expected here (within any
# single snapshot, days_since_cutoff separates churned/not-churned almost
# perfectly by construction) and is not itself a leakage signal anymore now that
# real out-of-time validation shows a consistent, non-collapsed result.

# Retention-economics assumptions (Checkpoint 5). These are ASSUMED, not observed
# — the spec requires stating them explicitly rather than hiding a threshold
# behind an unstated cost model. cost = half of the modal plan price (149,
# the single most common plan_list_price in the data) as a token retention
# discount. value = 894, which both approximates 6 months at the modal price
# (149 x 6 = 894) AND is itself a real observed plan_list_price in the data
# (likely an actual 6-unit bundle), giving it more grounding than a round guess.
# success_rate = 30%, a commonly-cited real-world retention-offer effectiveness
# figure, not derived from this dataset.
ASSUMED_RETENTION_OFFER_COST = 75
ASSUMED_RETAINED_SUBSCRIBER_VALUE = 894
ASSUMED_OFFER_SUCCESS_RATE = 0.30

# Operating threshold: swept 0.05-0.95 on real validation scores, maximizing
# total expected net benefit under the assumed costs above (see
# model.select_operating_threshold). At 0.88: precision=0.459, recall=0.842,
# flags 195,662 of 268,828 validation members (72.8%). The wide net is a direct
# consequence of the assumed costs (a cheap offer against a high potential
# payoff justifies flagging broadly) — sensitive to those assumptions, which is
# exactly why they're stated explicitly rather than buried.
OPERATING_THRESHOLD = 0.88
