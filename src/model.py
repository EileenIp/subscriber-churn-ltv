"""Baseline (recency rule, logistic regression) and LightGBM churn models.

Evaluated on the out-of-time train/validation split built in Phase 4
(src/config.TRAIN_CUTOFF_DATE / VALIDATION_CUTOFF_DATE). PR-AUC is the headline
metric — not ROC-AUC — because churn here isn't rare (30-41%), but PR-AUC still
better reflects what a retention team cares about: precision among the members
actually flagged, not performance on the (uninteresting) easy majority.

The operating threshold — where on the precision/recall curve to actually flag a
subscriber — is Eileen's decision (Checkpoint 5), not something computed here. It's
a cost question (wasted retention offer vs a lost subscriber), not a statistical one.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src import config

FEATURE_COLUMNS = [
    "n_prior_transactions", "n_active_days", "longest_gap_days",
    "days_since_last_transaction", "days_since_first_payment", "days_since_cutoff",
    "total_amount_paid_prior", "avg_amount_paid_prior", "most_recent_plan_price",
    "n_prior_cancels", "has_prior_cancel", "days_since_last_cancel", "auto_renew_rate",
    "tenure_days", "registered_via",
    "n_transactions_first_half", "n_transactions_second_half", "transaction_count_trend",
    "avg_amount_first_half", "avg_amount_second_half", "amount_paid_trend",
]
CATEGORICAL_COLUMNS = ["registered_via"]


def load_train_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(config.PROCESSED_DATA_DIR / "train_features.parquet")
    validation = pd.read_parquet(config.PROCESSED_DATA_DIR / "validation_features.parquet")
    return train, validation


def recency_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Simplest possible baseline: the longer since a member's prior transaction,
    the more likely they are to churn. Returns days_since_last_transaction itself
    as a continuous score (higher = more likely to churn) so it can be ranked and
    evaluated the same way as a model's predicted probability."""
    return df["days_since_last_transaction"].to_numpy()


def _one_hot(df: pd.DataFrame, reference_categories: dict) -> pd.DataFrame:
    out = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        out[col] = pd.Categorical(out[col], categories=reference_categories[col])
    return pd.get_dummies(out, columns=CATEGORICAL_COLUMNS, drop_first=True)


def fit_logistic_regression(train: pd.DataFrame):
    reference_categories = {c: sorted(train[c].unique()) for c in CATEGORICAL_COLUMNS}
    X_train = _one_hot(train, reference_categories)
    y_train = train["is_churn"].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED
    )
    clf.fit(X_train_scaled, y_train)
    return clf, scaler, reference_categories, X_train.columns.tolist()


def predict_logistic_regression(clf, scaler, reference_categories, columns, df: pd.DataFrame) -> np.ndarray:
    X = _one_hot(df, reference_categories).reindex(columns=columns, fill_value=0)
    X_scaled = scaler.transform(X)
    return clf.predict_proba(X_scaled)[:, 1]


def fit_lightgbm(train: pd.DataFrame):
    """This exact configuration (n_estimators=300, lr=0.05, num_leaves=31,
    is_unbalance=True, no further tuning) is the production choice — a
    hyperparameter sweep (random stratified internal split, early stopping)
    reached higher in-snapshot PR-AUC but did NOT beat this simple config on
    real out-of-time validation (0.638 vs 0.647), so the simple config won on
    the only metric that matters. See src/config.py's Phase 5 notes."""
    X_train = train[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        X_train[col] = X_train[col].astype("category")
    y_train = train["is_churn"].to_numpy()

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=config.RANDOM_SEED,
        is_unbalance=True,
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLUMNS)
    return model


def predict_lightgbm(model, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype("category")
    return model.predict_proba(X)[:, 1]


def select_operating_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Sweep thresholds and pick the one maximizing total expected net benefit,
    under the ASSUMED (not observed) cost model in src/config.py:
      net_benefit_per_flagged = precision * success_rate * retained_value - offer_cost
      total_net_benefit = net_benefit_per_flagged * flagged_count
    A cheap offer against a high potential payoff (as assumed here) means almost
    any positive-precision threshold is profitable, so the optimum tends to favor
    high recall — that's a property of the assumed costs, not a bug. This
    function does not decide anything on its own; Checkpoint 5 in the spec
    requires a human to state and own the cost assumptions."""
    C = config.ASSUMED_RETENTION_OFFER_COST
    V = config.ASSUMED_RETAINED_SUBSCRIBER_VALUE
    S = config.ASSUMED_OFFER_SUCCESS_RATE

    best_threshold, best_net_benefit, best_row = None, -np.inf, None
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_score >= t).astype(int)
        flagged = int(y_pred.sum())
        if flagged == 0:
            continue
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        precision = tp / flagged
        recall = tp / y_true.sum()
        net_benefit_per_flagged = precision * S * V - C
        total_net_benefit = net_benefit_per_flagged * flagged
        if total_net_benefit > best_net_benefit:
            best_net_benefit = total_net_benefit
            best_threshold = round(float(t), 2)
            best_row = {
                "threshold": best_threshold, "precision": precision, "recall": recall,
                "flagged": flagged, "total_net_benefit": total_net_benefit,
            }
    return best_row


def evaluate(y_true: np.ndarray, y_score: np.ndarray, thresholds: list[float]) -> dict:
    """PR-AUC is the headline metric (see module docstring). ROC-AUC is reported
    alongside it since it's the more commonly recognised number. Thresholds are
    reported across a spread for Eileen's Checkpoint 5 decision — none is chosen
    here."""
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    # normalize non-probability scores (e.g. the recency rule) into [0, 1] for the
    # thresholds table, so a threshold like 0.5 is comparable across models
    score_min, score_max = y_score.min(), y_score.max()
    normalized = (y_score - score_min) / (score_max - score_min) if score_max > score_min else y_score

    threshold_rows = []
    for t in thresholds:
        y_pred = (normalized >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        flagged_share = y_pred.mean()
        threshold_rows.append(
            {"threshold": t, "precision": precision, "recall": recall, "f1": f1, "flagged_share": flagged_share}
        )

    y_pred_default = (normalized >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_default)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "thresholds": threshold_rows,
        "confusion_matrix_at_0.5": cm,
    }


def print_evaluation(name: str, result: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print(f"PR-AUC:  {result['pr_auc']:.4f}  <- headline metric")
    print("\nprecision/recall by threshold:")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'flagged %':>10}")
    for row in result["thresholds"]:
        print(
            f"{row['threshold']:>10.2f} {row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['f1']:>10.4f} {row['flagged_share']*100:>9.1f}%"
        )
    print("\nconfusion matrix at threshold=0.5 (illustrative only, not the operating threshold):")
    print(result["confusion_matrix_at_0.5"])
