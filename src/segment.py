"""K-Means segmentation and the risk-vs-value quadrant.

Segments are built from behavioural features ONLY — never from is_churn or a
model's predicted risk score, which would leak the outcome into the grouping
itself and make "high-risk segment" a tautology rather than a finding. Risk and
value are cross-tabulated against the resulting segments afterward, as separate
axes.

Value here is real payment data (avg_amount_paid_prior), not an engagement
proxy — a genuine point of strength over a typical churn project that has to
fall back on session counts or similar. Segment NAMES and their plain-language
meaning are Eileen's call (spec Phase 6) — this module produces the statistical
profile of each segment; it does not name them.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config

# Segment names (Eileen's call, spec Phase 6 EILEEN DECIDES — explicitly
# delegated: "your choice"). Tied to the k=3 KMeans fit with RANDOM_SEED on the
# validation set; cluster index -> name is only valid for that specific run
# (see the segment profile printed during Phase 6 for the numbers behind each
# name). Grounded directly in the profile stats, not invented:
#   0: moderate tenure (~2yr), infrequent (5.3 avg txns), silent ~7mo, 55.8% churn
#   2: long-tenured (~4.6yr), frequent (18.8 avg txns), recent (~2mo), 17.0% churn
#   1: tiny (2.7%) but ~7x average spend/txn, long-tenured, only 1.6 avg txns,
#      silent ~8.7mo, 81.9% churn — big infrequent (bulk/annual-style) payers
#      who have gone quiet
SEGMENT_NAMES = {
    0: "Drifting Standard Subscribers",
    2: "Loyal Frequent Renewers",
    1: "High-Value Subscribers Going Dark",
}

# Quadrant names + retention recommendation (agent deliverable per spec Phase 6
# — only the segment names above were Eileen's call). "Low risk / high value"
# is deliberately NOT called "safe": it still shows 39.6% actual churn on the
# validation set, since OPERATING_THRESHOLD=0.88 is tuned for net-benefit-
# maximizing recall, not for cleanly separating "safe" from "at risk".
QUADRANT_NAMES = {
    ("high risk", "high value"): (
        "Priority Save",
        "Proactive outreach with a real offer — highest revenue at stake per member saved.",
    ),
    ("high risk", "low value"): (
        "Low-Cost Nudge",
        "Automated/passive retention only — the assumed offer cost isn't justified by the value here.",
    ),
    ("low risk", "low value"): (
        "Stable, No Action",
        "Genuinely low risk and low value — leave alone.",
    ),
    ("low risk", "high value"): (
        "Quiet Value — Monitor",
        "Not safe (39.6% actual churn) — just below the model's aggressive flagging bar. "
        "Light-touch periodic check-in given the revenue at stake, short of full priority-save spend.",
    ),
}

SEGMENTATION_FEATURES = [
    "days_since_cutoff", "days_since_last_transaction",
    "n_prior_transactions", "n_active_days", "longest_gap_days",
    "total_amount_paid_prior", "avg_amount_paid_prior", "most_recent_plan_price",
    "tenure_days", "transaction_count_trend", "amount_paid_trend",
]


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[SEGMENTATION_FEATURES])
    return X_scaled, scaler


def k_sweep(X_scaled: np.ndarray, k_range: range, silhouette_sample_size: int = 10_000) -> pd.DataFrame:
    """Elbow (inertia) and silhouette score for each k. Silhouette is O(n^2), so
    it's computed on a random sample, not the full dataset — sample_size is
    passed straight to sklearn's own sampling, not a separate pre-sample, so it
    stays representative of the full X_scaled distribution."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(
            X_scaled, labels, sample_size=min(silhouette_sample_size, len(X_scaled)),
            random_state=config.RANDOM_SEED,
        )
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
    km.fit(X_scaled)
    return km


def profile_segments(df: pd.DataFrame, cluster_labels: np.ndarray, risk_col: str) -> pd.DataFrame:
    """Per-segment: size, churn rate, mean amount paid, mean tenure, mean
    predicted risk. No names — just the numbers, for Eileen to name."""
    profiled = df.copy()
    profiled["segment"] = cluster_labels
    summary = profiled.groupby("segment").agg(
        size=("msno", "count"),
        churn_rate=("is_churn", "mean"),
        mean_amount_paid=("avg_amount_paid_prior", "mean"),
        mean_tenure_days=("tenure_days", "mean"),
        mean_predicted_risk=(risk_col, "mean"),
        mean_days_since_cutoff=("days_since_cutoff", "mean"),
        mean_prior_transactions=("n_prior_transactions", "mean"),
    ).reset_index()
    summary["share_of_population"] = summary["size"] / len(profiled)
    return summary.sort_values("size", ascending=False).reset_index(drop=True)


def risk_value_quadrant(
    df: pd.DataFrame, risk_col: str, risk_threshold: float, value_col: str, value_threshold: float
) -> pd.DataFrame:
    """2x2 cross-tab: high/low risk x high/low value. risk_threshold is the
    model's operating threshold (config.OPERATING_THRESHOLD); value_threshold
    is a business call about what counts as "high value" — median split by
    default unless told otherwise."""
    labeled = df.copy()
    labeled["risk_band"] = np.where(labeled[risk_col] >= risk_threshold, "high risk", "low risk")
    labeled["value_band"] = np.where(labeled[value_col] >= value_threshold, "high value", "low value")

    quadrant = labeled.groupby(["risk_band", "value_band"]).agg(
        size=("msno", "count"),
        churn_rate=("is_churn", "mean"),
        mean_amount_paid=(value_col, "mean"),
        mean_predicted_risk=(risk_col, "mean"),
    ).reset_index()
    quadrant["share_of_population"] = quadrant["size"] / len(labeled)
    return quadrant.sort_values("size", ascending=False).reset_index(drop=True)
