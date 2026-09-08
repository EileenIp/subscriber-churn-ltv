from src import config, ingest


def test_load_reg_data_schema():
    df = ingest.load_reg_data()
    assert list(df.columns) == config.REG_DATA_COLUMNS
    assert df["uid"].is_unique


def test_load_ab_test_data_schema():
    df = ingest.load_ab_test_data()
    assert list(df.columns) == config.AB_TEST_COLUMNS
    assert df["user_id"].is_unique


def test_reg_data_no_nulls():
    df = ingest.load_reg_data()
    assert df.isnull().sum().sum() == 0
