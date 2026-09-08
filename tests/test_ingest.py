from src import config, ingest


def test_load_members_schema():
    df = ingest.load_members()
    assert list(df.columns) == config.MEMBERS_COLUMNS
    assert df["msno"].is_unique


def test_load_train_labels_schema():
    df = ingest.load_train_labels()
    assert list(df.columns) == config.TRAIN_LABELS_COLUMNS
    assert df["msno"].is_unique
    assert set(df["is_churn"].unique()) <= {0, 1}


def test_members_no_nulls_in_id():
    df = ingest.load_members()
    assert df["msno"].isnull().sum() == 0
