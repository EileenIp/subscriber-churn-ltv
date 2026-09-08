"""Feature engineering — transaction-level then member-level, built from the observation window only.

Churn, in plain English: a member has churned if they let a subscription expire
and don't come back with a new transaction within 30 days of that expiry date.
Most renewals happen same-day (auto-renew); the ones that don't mostly resolve
within the first few days, and by 30 days almost everyone who was coming back
already has. A member who's silent past that point is counted as churned.

Each member contributes exactly one labeled row: their LAST eligible transaction
(expiry resolvable within the data, and at least one prior transaction so there's
real history to learn from). Features are built from that member's full
transaction history strictly before that transaction's own date — never from
transactions on or after it. Listening-intensity features (from user_logs_v2,
March 2017) are for scoring current members only — they fall outside the range
the transaction data can independently confirm outcomes for, so they're never
used to train or validate the churn model itself.

See src/config.py for the exact parameters (CHURN_WINDOW_DAYS, MIN_PRIOR_TRANSACTIONS,
DATA_MAX_DATE) and spec Phase 3 for the full reasoning.
"""
import numpy as np
import pandas as pd

from src import config

# Feature columns that are allowed to be genuinely absent-of-signal (documented here,
# not silently null): none — every feature below uses a flag+sentinel pattern
# (e.g. has_prior_cancel + days_since_last_cancel=0) so the feature matrix has zero
# real nulls. This list stays empty on purpose; test_feature_nulls checks against it.
DOCUMENTED_NULLABLE_FEATURE_COLUMNS: list[str] = []


def clean_transactions(tx: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Apply the Phase 2 / Checkpoint 2 data-quality decisions.

    - Drop transactions for members with no matching row in members_v3.csv
      (Eileen: drop from member-joined analysis).
    - Drop the pre-registration transaction rows (Eileen: drop those rows).
    - Collapse same-day repeat transactions per member to the last row
      (billing-correction pattern, not real cadence).
    - Drop the genuinely anomalous negative-coverage rows: membership_expire_date
      before transaction_date AND not a cancellation. The cancellation-linked
      ones are kept — they're a real signal (95.8% of negative-coverage rows are
      is_cancel=1), not a defect.
    """
    tx = tx.merge(members[["msno", "registration_init_time"]], on="msno", how="inner")
    tx = tx[tx["transaction_date"] >= tx["registration_init_time"]]
    tx = tx.drop(columns=["registration_init_time"])

    tx = tx.sort_values(["msno", "transaction_date"])
    tx = tx.drop_duplicates(subset=["msno", "transaction_date"], keep="last")

    covered_days = (tx["membership_expire_date"] - tx["transaction_date"]).dt.days
    anomalous = (covered_days < 0) & (tx["is_cancel"] == 0)
    tx = tx[~anomalous]

    return tx.reset_index(drop=True)


def identify_labeled_events(
    tx: pd.DataFrame,
    max_transaction_date: str | pd.Timestamp | None = None,
    exclude_msno: set | None = None,
) -> pd.DataFrame:
    """Pick each member's one labeled example: their last eligible transaction
    at or before max_transaction_date (a shared, external reference date).

    Eligible = has at least one prior transaction, and its membership_expire_date
    is early enough that the CHURN_WINDOW_DAYS resolution window fits entirely
    within the data (no right-censoring). The label looks only at that member's
    very next transaction after the eligible one — never further ahead, and never
    at anything before it.

    IMPORTANT: selecting "each member's own last eligible transaction" with no
    shared reference date makes the cutoff date itself a function of churn status
    — a member who churned early is permanently pinned to an early cutoff (their
    history simply stops), while a member who kept renewing gets pushed later and
    later. An out-of-time split by that self-selected date leaks the label through
    the date field (verified on the real data: 69% churn in an early split vs 7%
    in a late split, from the exact same population). max_transaction_date fixes
    this — every member in a given split is evaluated as of the same external
    point in time, not their own idiosyncratic last transaction. exclude_msno lets
    a later (e.g. validation) call exclude members already used in an earlier
    (e.g. train) split, guaranteeing disjoint membership without needing a second
    date-based filter.
    """
    tx = tx.sort_values(["msno", "transaction_date"]).reset_index(drop=True)
    tx["has_prior"] = tx.groupby("msno", observed=True).cumcount() > 0
    tx["next_transaction_date"] = tx.groupby("msno", observed=True)["transaction_date"].shift(-1)

    data_max_date = pd.Timestamp(config.DATA_MAX_DATE)
    resolvable_boundary = data_max_date - pd.Timedelta(days=config.CHURN_WINDOW_DAYS)
    eligible_mask = tx["has_prior"] & (tx["membership_expire_date"] <= resolvable_boundary)

    if max_transaction_date is not None:
        eligible_mask &= tx["transaction_date"] <= pd.Timestamp(max_transaction_date)
    if exclude_msno:
        eligible_mask &= ~tx["msno"].isin(exclude_msno)

    eligible_tx = tx[eligible_mask].copy()
    if eligible_tx.empty:
        return eligible_tx.rename(columns={"transaction_date": "cutoff_date"})[
            ["msno", "cutoff_date", "membership_expire_date"]
        ].assign(is_churn=pd.Series(dtype=int))

    last_idx = eligible_tx.groupby("msno", observed=True)["transaction_date"].idxmax()
    labeled = eligible_tx.loc[last_idx].copy()

    gap_days = (labeled["next_transaction_date"] - labeled["membership_expire_date"]).dt.days
    labeled["is_churn"] = (
        labeled["next_transaction_date"].isna() | (gap_days > config.CHURN_WINDOW_DAYS)
    ).astype(int)

    labeled = labeled.rename(columns={"transaction_date": "cutoff_date"})
    return labeled[["msno", "cutoff_date", "membership_expire_date", "is_churn"]].reset_index(drop=True)


def build_member_features(tx: pd.DataFrame, labeled: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Build member-level features from each member's transaction history strictly
    before their own labeled cutoff_date — never on or after it."""
    merged = tx.merge(labeled[["msno", "cutoff_date"]], on="msno", how="inner")
    prior = merged[merged["transaction_date"] < merged["cutoff_date"]].copy()
    prior = prior.sort_values(["msno", "transaction_date"])

    agg = prior.groupby("msno", observed=True).agg(
        n_prior_transactions=("transaction_date", "count"),
        n_active_days=("transaction_date", "nunique"),
        first_transaction_date=("transaction_date", "min"),
        last_transaction_date=("transaction_date", "max"),
        total_amount_paid_prior=("actual_amount_paid", "sum"),
        avg_amount_paid_prior=("actual_amount_paid", "mean"),
        most_recent_plan_price=("plan_list_price", "last"),
        n_prior_cancels=("is_cancel", "sum"),
        auto_renew_rate=("is_auto_renew", "mean"),
    ).reset_index()

    last_cancel = (
        prior[prior["is_cancel"] == 1]
        .groupby("msno", observed=True)["transaction_date"]
        .max()
        .rename("last_cancel_date")
    )
    agg = agg.merge(last_cancel, on="msno", how="left")

    gap_days = prior.groupby("msno", observed=True)["transaction_date"].diff().dt.days
    longest_gap = gap_days.groupby(prior["msno"]).max().rename("longest_gap_days")
    agg = agg.merge(longest_gap, on="msno", how="left")

    counts = prior.groupby("msno", observed=True).size().rename("_count_for_half")
    prior = prior.merge(counts, on="msno")
    prior["_rn"] = prior.groupby("msno", observed=True).cumcount()
    prior["half"] = np.where(prior["_rn"] < prior["_count_for_half"] / 2, "first", "second")

    half_counts = (
        prior.groupby(["msno", "half"], observed=True).size().unstack(fill_value=0)
    )
    half_counts.columns = [f"n_transactions_{c}_half" for c in half_counts.columns]
    agg = agg.merge(half_counts.reset_index(), on="msno", how="left")

    half_amounts = (
        prior.groupby(["msno", "half"], observed=True)["actual_amount_paid"].mean().unstack(fill_value=0)
    )
    half_amounts.columns = [f"avg_amount_{c}_half" for c in half_amounts.columns]
    agg = agg.merge(half_amounts.reset_index(), on="msno", how="left")

    for col in ("n_transactions_first_half", "n_transactions_second_half",
                "avg_amount_first_half", "avg_amount_second_half"):
        if col not in agg.columns:
            agg[col] = 0
        agg[col] = agg[col].fillna(0)

    features = labeled.merge(agg, on="msno", how="inner")
    features = features.merge(
        members[["msno", "registration_init_time", "registered_via"]], on="msno", how="left"
    )

    features["tenure_days"] = (features["cutoff_date"] - features["registration_init_time"]).dt.days
    features["days_since_last_transaction"] = (
        features["cutoff_date"] - features["last_transaction_date"]
    ).dt.days
    features["days_since_first_payment"] = (
        features["cutoff_date"] - features["first_transaction_date"]
    ).dt.days
    features["has_prior_cancel"] = (features["n_prior_cancels"] > 0).astype(int)
    features["days_since_last_cancel"] = (
        (features["cutoff_date"] - features["last_cancel_date"]).dt.days.fillna(0).astype(int)
    )
    features["longest_gap_days"] = features["longest_gap_days"].fillna(0).astype(int)
    features["transaction_count_trend"] = (
        features["n_transactions_second_half"] - features["n_transactions_first_half"]
    )
    features["amount_paid_trend"] = features["avg_amount_second_half"] - features["avg_amount_first_half"]

    feature_columns = [
        "msno", "cutoff_date", "is_churn",
        "n_prior_transactions", "n_active_days", "longest_gap_days",
        "days_since_last_transaction", "days_since_first_payment",
        "total_amount_paid_prior", "avg_amount_paid_prior", "most_recent_plan_price",
        "n_prior_cancels", "has_prior_cancel", "days_since_last_cancel", "auto_renew_rate",
        "tenure_days", "registered_via",
        "n_transactions_first_half", "n_transactions_second_half", "transaction_count_trend",
        "avg_amount_first_half", "avg_amount_second_half", "amount_paid_trend",
    ]
    return features[feature_columns].reset_index(drop=True)


def build_feature_matrix(tx: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """End-to-end, single snapshot: clean, label, and build features using each
    member's own last eligible transaction. One row per eligible member.

    This is the right tool for LIVE SCORING (evaluate current members "as of
    today", where there's no train/validation comparison for a self-selected
    cutoff date to leak into). It is NOT the right tool for building a
    train/validation split — see build_train_validation_matrices for that,
    and the warning in identify_labeled_events for why.
    """
    tx = clean_transactions(tx, members)
    labeled = identify_labeled_events(tx)
    return build_member_features(tx, labeled, members)


def build_train_validation_matrices(
    tx: pd.DataFrame,
    members: pd.DataFrame,
    train_cutoff: str,
    validation_cutoff: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time train/validation feature matrices, built safely.

    Every member in the train matrix is evaluated as of the SAME external
    train_cutoff date (their most recent eligible transaction at or before it)
    — not their own idiosyncratic last transaction. Validation uses the same
    approach at validation_cutoff, explicitly excluding any member already used
    in train. This guarantees disjoint membership by construction and avoids
    leaking churn status through a self-selected cutoff date (see
    identify_labeled_events's docstring for the bug this replaced).
    """
    if pd.Timestamp(train_cutoff) >= pd.Timestamp(validation_cutoff):
        raise ValueError("train_cutoff must be strictly before validation_cutoff")

    tx = clean_transactions(tx, members)

    train_labeled = identify_labeled_events(tx, max_transaction_date=train_cutoff)
    validation_labeled = identify_labeled_events(
        tx, max_transaction_date=validation_cutoff, exclude_msno=set(train_labeled["msno"])
    )

    train_features = build_member_features(tx, train_labeled, members)
    validation_features = build_member_features(tx, validation_labeled, members)
    return train_features, validation_features
