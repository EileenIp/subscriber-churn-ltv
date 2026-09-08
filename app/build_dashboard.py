"""Build docs/index.html — the Phase 7a HTML dashboard.

Regenerates the dashboard summary (headline stats, threshold sweep, segment
profiles, risk-value quadrant) and a sample of real subscribers with
plain-language SHAP explanations, then injects both into app/dashboard_template.html
to produce a single self-contained file at docs/index.html — no server, no
external data fetch, opens directly in a browser or deploys to GitHub Pages
from /docs.

Requires data/processed/train_features.parquet, validation_features.parquet,
and final_lgbm_model.pkl to already exist (Phase 4/5 outputs).

Run: python -m app.build_dashboard
"""
import json
import pickle

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config, model as m, segment as sg

REPO_ROOT = config.PROJECT_ROOT
TEMPLATE_PATH = REPO_ROOT / "app" / "dashboard_template.html"
OUTPUT_PATH = REPO_ROOT / "docs" / "index.html"

SHAP_SAMPLE_PER_DECILE = 40
SHAP_RANDOM_SEED = 7


def _load_scored_validation():
    train, validation = m.load_train_validation()
    with open(config.PROCESSED_DATA_DIR / "final_lgbm_model.pkl", "rb") as f:
        saved = pickle.load(f)
    lgbm = saved["model"]

    X_val = validation[m.FEATURE_COLUMNS].copy()
    X_val["registered_via"] = X_val["registered_via"].astype("category")
    validation = validation.copy()
    validation["predicted_risk"] = lgbm.predict_proba(X_val)[:, 1]
    return train, validation, lgbm


def build_dashboard_summary(train: pd.DataFrame, validation: pd.DataFrame, lgbm) -> dict:
    y_val = validation["is_churn"].to_numpy()
    scores = validation["predicted_risk"].to_numpy()

    recency_score = m.recency_rule_score(validation)
    lr_clf, scaler, ref_cats, lr_columns = m.fit_logistic_regression(train)
    lr_score = m.predict_logistic_regression(lr_clf, scaler, ref_cats, lr_columns, validation)

    headline = {
        "validation_population": int(len(validation)),
        "overall_churn_rate": float(y_val.mean()),
        "operating_threshold": config.OPERATING_THRESHOLD,
        "models": {
            "recency_rule": {"roc_auc": float(roc_auc_score(y_val, recency_score)),
                              "pr_auc": float(average_precision_score(y_val, recency_score))},
            "logistic_regression": {"roc_auc": float(roc_auc_score(y_val, lr_score)),
                                     "pr_auc": float(average_precision_score(y_val, lr_score))},
            "lightgbm": {"roc_auc": float(roc_auc_score(y_val, scores)),
                         "pr_auc": float(average_precision_score(y_val, scores))},
        },
    }
    at_thresh = (scores >= config.OPERATING_THRESHOLD).astype(int)
    tp = int(((at_thresh == 1) & (y_val == 1)).sum())
    flagged = int(at_thresh.sum())
    headline["at_operating_threshold"] = {
        "flagged": flagged, "flagged_share": flagged / len(validation),
        "precision": tp / flagged, "recall": tp / y_val.sum(),
    }

    C = config.ASSUMED_RETENTION_OFFER_COST
    V = config.ASSUMED_RETAINED_SUBSCRIBER_VALUE
    S = config.ASSUMED_OFFER_SUCCESS_RATE
    sweep_rows = []
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (scores >= t).astype(int)
        flagged_n = int(y_pred.sum())
        if flagged_n == 0:
            continue
        tp_n = int(((y_pred == 1) & (y_val == 1)).sum())
        precision = tp_n / flagged_n
        recall = tp_n / y_val.sum()
        sweep_rows.append({
            "threshold": round(float(t), 2), "precision": precision, "recall": recall,
            "flagged": flagged_n, "flagged_share": flagged_n / len(validation),
            "mean_value_flagged": float(validation.loc[y_pred == 1, "avg_amount_paid_prior"].mean()),
            "total_value_flagged": float(validation.loc[y_pred == 1, "avg_amount_paid_prior"].sum()),
            "net_benefit": (precision * S * V - C) * flagged_n,
        })

    X_scaled, _ = sg.scale_features(validation)
    km = sg.fit_kmeans(X_scaled, k=3)
    profile = sg.profile_segments(validation, km.labels_, risk_col="predicted_risk")
    segments_out = [
        {"id": int(row["segment"]), "name": sg.SEGMENT_NAMES[int(row["segment"])],
         "size": int(row["size"]), "share": float(row["share_of_population"]),
         "churn_rate": float(row["churn_rate"]), "mean_amount_paid": float(row["mean_amount_paid"]),
         "mean_tenure_days": float(row["mean_tenure_days"]), "mean_predicted_risk": float(row["mean_predicted_risk"])}
        for _, row in profile.iterrows()
    ]

    value_threshold = float(validation["avg_amount_paid_prior"].median())
    quadrant = sg.risk_value_quadrant(
        validation, risk_col="predicted_risk", risk_threshold=config.OPERATING_THRESHOLD,
        value_col="avg_amount_paid_prior", value_threshold=value_threshold,
    )
    quadrant_out = []
    for _, row in quadrant.iterrows():
        name, rec = sg.QUADRANT_NAMES[(row["risk_band"], row["value_band"])]
        quadrant_out.append({
            "risk_band": row["risk_band"], "value_band": row["value_band"], "name": name, "recommendation": rec,
            "size": int(row["size"]), "share": float(row["share_of_population"]),
            "churn_rate": float(row["churn_rate"]), "mean_amount_paid": float(row["mean_amount_paid"]),
        })

    return {
        "headline": headline, "threshold_sweep": sweep_rows,
        "segments": segments_out, "quadrant": quadrant_out, "value_threshold": value_threshold,
    }, km


def _fmt_days(d: float) -> str:
    if d < 30:
        return f"{int(d)} days"
    if d < 365:
        months = round(d / 30)
        return f"about {months} month{'s' if months != 1 else ''}"
    return f"about {round(d / 365, 1)} years"


def _explain_feature(feat: str, value: float, shap_val: float) -> str | None:
    up = shap_val > 0
    if feat == "days_since_cutoff":
        return (f"Hasn't renewed in {_fmt_days(value)}" if up else
                f"Renewed as recently as {_fmt_days(value)} ago")
    if feat == "days_since_last_transaction":
        return (f"Historically leaves long gaps between renewals (last gap was {_fmt_days(value)})" if up else
                f"Historically renews promptly (last gap was only {_fmt_days(value)})")
    if feat == "auto_renew_rate":
        if up:
            return ("Doesn't have auto-renew turned on" if value < 0.5 else
                    "Auto-renew is on, but their other patterns still raise the risk")
        return ("Has auto-renew turned on, which is helping keep risk down" if value >= 0.5 else
                "Doesn't rely on auto-renew, but other signals here are reassuring")
    if feat == "most_recent_plan_price":
        return (f"On a higher-priced plan (paid {int(value)} last time)" if up else
                f"On a lower-priced plan (paid {int(value)} last time)")
    if feat == "total_amount_paid_prior":
        return ("Has spent relatively little with us so far" if up else
                f"Has spent {int(value):,} with us so far — a loyal, established customer")
    if feat == "n_prior_transactions":
        return ("Has only renewed a couple of times before" if up else
                f"Has renewed {int(value)} times before — an established pattern")
    if feat == "tenure_days":
        return ("Is a fairly new subscriber" if up else
                f"Has been a subscriber for {_fmt_days(value)}")
    if feat == "n_prior_cancels":
        return "Has cancelled and come back before" if (up and value > 0) else None
    if feat == "longest_gap_days":
        return f"Has had a gap as long as {_fmt_days(value)} between renewals before" if up else None
    return None


def build_subscriber_sample(validation: pd.DataFrame, km, lgbm) -> list[dict]:
    validation = validation.copy()
    validation["segment_id"] = km.labels_
    value_threshold = float(validation["avg_amount_paid_prior"].median())

    validation["risk_decile"] = pd.qcut(validation["predicted_risk"], 10, labels=False, duplicates="drop")
    sample_idx = (
        validation.groupby("risk_decile", group_keys=False)
        .apply(lambda g: g.sample(min(SHAP_SAMPLE_PER_DECILE, len(g)), random_state=SHAP_RANDOM_SEED))
        .index
    )
    sample = validation.loc[sample_idx].reset_index(drop=True)

    X_sample = sample[m.FEATURE_COLUMNS].copy()
    X_sample["registered_via"] = X_sample["registered_via"].astype("category")
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    records = []
    for i in range(len(sample)):
        row = sample.iloc[i]
        row_shap = pd.Series(shap_values[i], index=m.FEATURE_COLUMNS)
        ranked = row_shap.reindex(row_shap.abs().sort_values(ascending=False).index)

        reasons = []
        for feat in ranked.index:
            if len(reasons) >= 3:
                break
            text = _explain_feature(feat, row[feat], ranked[feat])
            if text:
                reasons.append({"text": text, "direction": "up" if ranked[feat] > 0 else "down"})

        records.append({
            "id": f"SUB-{i + 1:05d}",
            "predicted_risk": round(float(row["predicted_risk"]), 4),
            "risk_band": "high risk" if row["predicted_risk"] >= config.OPERATING_THRESHOLD else "low risk",
            "value_band": "high value" if row["avg_amount_paid_prior"] >= value_threshold else "low value",
            "avg_amount_paid": round(float(row["avg_amount_paid_prior"]), 2),
            "tenure_days": int(row["tenure_days"]), "segment_id": int(row["segment_id"]),
            "actual_churned": bool(row["is_churn"]), "reasons": reasons,
        })
    return records


def main() -> None:
    train, validation, lgbm = _load_scored_validation()
    dashboard_summary, km = build_dashboard_summary(train, validation, lgbm)
    subscriber_sample = build_subscriber_sample(validation, km, lgbm)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__DASHBOARD_DATA__", json.dumps(dashboard_summary, separators=(",", ":")))
    html = html.replace("__SUBSCRIBER_SAMPLE__", json.dumps(subscriber_sample, separators=(",", ":")))

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
