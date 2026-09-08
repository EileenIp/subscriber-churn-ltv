import numpy as np
import pandas as pd

from src import segment


def _synthetic_df(n=200, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "msno": [f"M{i}" for i in range(n)],
        "days_since_cutoff": rng.randint(0, 500, n),
        "days_since_last_transaction": rng.randint(0, 60, n),
        "n_prior_transactions": rng.randint(1, 50, n),
        "n_active_days": rng.randint(1, 50, n),
        "longest_gap_days": rng.randint(0, 100, n),
        "total_amount_paid_prior": rng.randint(0, 5000, n),
        "avg_amount_paid_prior": rng.uniform(0, 200, n),
        "most_recent_plan_price": rng.choice([99, 129, 149, 180], n),
        "tenure_days": rng.randint(1, 3000, n),
        "transaction_count_trend": rng.randint(-10, 10, n),
        "amount_paid_trend": rng.uniform(-50, 50, n),
        "is_churn": rng.randint(0, 2, n),
        "predicted_risk": rng.uniform(0, 1, n),
    })
    return df


def test_scale_features_shape():
    df = _synthetic_df()
    X_scaled, scaler = segment.scale_features(df)
    assert X_scaled.shape == (len(df), len(segment.SEGMENTATION_FEATURES))
    # standardized features should have ~zero mean, unit variance
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-6)


def test_kmeans_and_profile_segments_no_leakage_columns():
    df = _synthetic_df()
    X_scaled, _ = segment.scale_features(df)
    km = segment.fit_kmeans(X_scaled, k=3)
    labels = km.labels_
    assert set(labels) <= {0, 1, 2}

    profile = segment.profile_segments(df, labels, risk_col="predicted_risk")
    assert profile["size"].sum() == len(df)
    assert "segment" in profile.columns
    # is_churn and predicted_risk must never be among the clustering inputs
    assert "is_churn" not in segment.SEGMENTATION_FEATURES
    assert "predicted_risk" not in segment.SEGMENTATION_FEATURES


def test_risk_value_quadrant_covers_whole_population():
    df = _synthetic_df()
    quadrant = segment.risk_value_quadrant(
        df, risk_col="predicted_risk", risk_threshold=0.5,
        value_col="avg_amount_paid_prior", value_threshold=df["avg_amount_paid_prior"].median(),
    )
    assert len(quadrant) <= 4
    assert quadrant["size"].sum() == len(df)
    assert set(quadrant["risk_band"]) <= {"high risk", "low risk"}
    assert set(quadrant["value_band"]) <= {"high value", "low value"}
