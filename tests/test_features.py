import pandas as pd
import pytest

from src import features


def _tx_row(msno, transaction_date, expire_date, amount=149, is_cancel=0, is_auto_renew=1, plan_price=149):
    return dict(
        msno=msno,
        payment_method_id=1,
        payment_plan_days=30,
        plan_list_price=plan_price,
        actual_amount_paid=amount,
        is_auto_renew=is_auto_renew,
        transaction_date=pd.Timestamp(transaction_date),
        membership_expire_date=pd.Timestamp(expire_date),
        is_cancel=is_cancel,
    )


def _members(msnos, registered="2015-01-01"):
    return pd.DataFrame(
        {
            "msno": msnos,
            "city": [1] * len(msnos),
            "bd": [0] * len(msnos),
            "gender": [None] * len(msnos),
            "registered_via": [7] * len(msnos),
            "registration_init_time": [pd.Timestamp(registered)] * len(msnos),
        }
    )


def test_no_future_events():
    # M1 has three prior transactions, then the labeled (eligible) one, then one
    # more transaction after it — that trailing one must never affect features.
    rows = [
        _tx_row("M1", "2016-01-01", "2016-01-31", amount=100),
        _tx_row("M1", "2016-02-01", "2016-03-02", amount=100),
        _tx_row("M1", "2016-03-02", "2016-04-01", amount=100),
        _tx_row("M1", "2016-04-01", "2016-05-01", amount=999999),  # the labeled cutoff transaction
        _tx_row("M1", "2016-05-05", "2016-06-04", amount=999999),  # after cutoff — must be excluded
    ]
    tx = pd.DataFrame(rows)
    members = _members(["M1"])

    tx_clean = features.clean_transactions(tx, members)
    labeled = features.identify_labeled_events(tx_clean)
    # force the cutoff to the 2016-04-01 transaction regardless of eligibility window
    labeled = labeled.copy()
    labeled.loc[labeled["msno"] == "M1", "cutoff_date"] = pd.Timestamp("2016-04-01")

    result = features.build_member_features(tx_clean, labeled, members)
    row = result[result["msno"] == "M1"].iloc[0]

    assert row["n_prior_transactions"] == 3
    assert row["total_amount_paid_prior"] == 300
    assert row["avg_amount_paid_prior"] == 100
    # the future row's absurd amount must never leak into the aggregate
    assert row["total_amount_paid_prior"] != pytest.approx(300 + 999999)


def test_label_from_prediction_window_only():
    # Each member: txn1 gives the labeled txn2 its required prior history; txn2 is
    # the member's last ELIGIBLE transaction (the thing being labeled); txn3 (where
    # present) supplies the "next transaction" used purely for the gap check — it's
    # given a far-future expire_date so it's itself ineligible (unresolvable within
    # the data) and never displaces txn2 as the selected labeled event.
    far_future = "2025-01-01"
    rows = [
        # M_RENEW_EARLY: next transaction 10 days after expiry -> not churned
        _tx_row("M_RENEW_EARLY", "2015-01-01", "2015-01-31"),
        _tx_row("M_RENEW_EARLY", "2016-01-01", "2016-01-31"),
        _tx_row("M_RENEW_EARLY", "2016-02-10", far_future),
        # M_RENEW_AT_30: next transaction exactly 30 days after expiry -> not churned (boundary inclusive)
        _tx_row("M_RENEW_AT_30", "2015-01-01", "2015-01-31"),
        _tx_row("M_RENEW_AT_30", "2016-01-01", "2016-01-31"),
        _tx_row("M_RENEW_AT_30", "2016-03-01", far_future),
        # M_RENEW_LATE: next transaction 31 days after expiry -> churned
        _tx_row("M_RENEW_LATE", "2015-01-01", "2015-01-31"),
        _tx_row("M_RENEW_LATE", "2016-01-01", "2016-01-31"),
        _tx_row("M_RENEW_LATE", "2016-03-02", far_future),
        # M_NO_RENEW: no further transaction at all -> churned
        _tx_row("M_NO_RENEW", "2015-01-01", "2015-01-31"),
        _tx_row("M_NO_RENEW", "2016-01-01", "2016-01-31"),
    ]
    tx = pd.DataFrame(rows)
    members = _members(["M_RENEW_EARLY", "M_RENEW_AT_30", "M_RENEW_LATE", "M_NO_RENEW"])
    tx_clean = features.clean_transactions(tx, members)
    labeled = features.identify_labeled_events(tx_clean)

    label_by_member = labeled.set_index("msno")["is_churn"].to_dict()
    assert label_by_member["M_RENEW_EARLY"] == 0
    assert label_by_member["M_RENEW_AT_30"] == 0
    assert label_by_member["M_RENEW_LATE"] == 1
    assert label_by_member["M_NO_RENEW"] == 1

    # a member's very first-ever transaction must never be selected as the labeled
    # event (no prior history to build features from)
    only_one_tx = pd.DataFrame([_tx_row("M_SINGLE", "2016-01-01", "2016-01-31")])
    members_single = _members(["M_SINGLE"])
    tx_single_clean = features.clean_transactions(only_one_tx, members_single)
    labeled_single = features.identify_labeled_events(tx_single_clean)
    assert "M_SINGLE" not in set(labeled_single["msno"])


def test_no_member_overlap():
    rows = []
    # M0-M2: eligible well before the train cutoff -> land in train.
    for i in range(3):
        msno = f"M{i}"
        rows.append(_tx_row(msno, "2015-01-01", "2015-01-31"))
        rows.append(_tx_row(msno, "2015-06-01", "2015-06-30"))
    # M3-M5: only become eligible after the train cutoff but before the
    # validation cutoff -> should land in validation, not train.
    for i in range(3, 6):
        msno = f"M{i}"
        rows.append(_tx_row(msno, "2016-08-01", "2016-08-31"))
        rows.append(_tx_row(msno, "2016-09-01", "2016-09-30"))
    # M6: eligible before the train cutoff AND still transacting after it —
    # must end up in train only, never re-appear in validation too.
    rows.append(_tx_row("M6", "2015-01-01", "2015-01-31"))
    rows.append(_tx_row("M6", "2015-06-01", "2015-06-30"))
    rows.append(_tx_row("M6", "2016-09-01", "2016-09-30"))

    tx = pd.DataFrame(rows)
    members = _members([f"M{i}" for i in range(7)])

    train, validation = features.build_train_validation_matrices(
        tx, members, train_cutoff="2016-01-01", validation_cutoff="2016-12-31"
    )
    assert len(train) > 0 and len(validation) > 0
    assert set(train["msno"]).isdisjoint(set(validation["msno"]))
    assert set(train["msno"]) == {"M0", "M1", "M2", "M6"}
    assert set(validation["msno"]) == {"M3", "M4", "M5"}

    with pytest.raises(ValueError):
        features.build_train_validation_matrices(
            tx, members, train_cutoff="2016-12-31", validation_cutoff="2016-01-01"
        )


def test_feature_nulls():
    rows = [
        # a member with a prior cancellation
        _tx_row("M_A", "2015-01-01", "2015-01-31", is_cancel=1),
        _tx_row("M_A", "2015-06-01", "2015-06-30"),
        _tx_row("M_A", "2016-01-01", "2016-01-31"),
        # a member with only the minimum: exactly one prior transaction, no cancels
        _tx_row("M_B", "2015-01-01", "2015-01-31"),
        _tx_row("M_B", "2016-01-01", "2016-01-31"),
    ]
    tx = pd.DataFrame(rows)
    members = _members(["M_A", "M_B"])
    matrix = features.build_feature_matrix(tx, members)

    assert len(matrix) == 2
    null_counts = matrix.isnull().sum()
    offending = null_counts[null_counts > 0]
    allowed = set(features.DOCUMENTED_NULLABLE_FEATURE_COLUMNS)
    assert set(offending.index) <= allowed, f"undocumented nulls in: {dict(offending)}"
