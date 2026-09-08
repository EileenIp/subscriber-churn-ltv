"""Load the raw Gamelytics files, validate their schema, and print a data quality summary.

Three raw files, joined on player id:
  reg_data.csv  - one row per player: registration timestamp
  auth_data.csv - one row per login event: repeated per player (~9.6M rows, read in chunks)
  ab_test.csv   - one row per player: lifetime revenue and A/B test group

Run: python -m src.ingest
"""
from dataclasses import dataclass, field

import pandas as pd

from src import config


class SchemaError(ValueError):
    pass


def _validate_columns(actual_columns: list[str], expected_columns: list[str], source_name: str) -> None:
    if list(actual_columns) != expected_columns:
        raise SchemaError(
            f"{source_name}: expected columns {expected_columns}, got {list(actual_columns)}"
        )


@dataclass
class ProfileResult:
    source_name: str
    row_count: int
    null_counts: dict = field(default_factory=dict)
    duplicate_row_count: int = 0
    timestamp_min: pd.Timestamp | None = None
    timestamp_max: pd.Timestamp | None = None
    unique_players: int | None = None

    def print_report(self) -> None:
        print(f"\n--- {self.source_name} ---")
        print(f"rows: {self.row_count:,}")
        if self.unique_players is not None:
            print(f"unique players: {self.unique_players:,}")
        if self.timestamp_min is not None:
            print(f"date span: {self.timestamp_min} to {self.timestamp_max}")
        print(f"duplicate rows: {self.duplicate_row_count:,}")
        print("null counts by column:")
        for col, n in self.null_counts.items():
            print(f"  {col}: {n:,}")


def load_reg_data() -> pd.DataFrame:
    df = pd.read_csv(config.REG_DATA_PATH, sep=config.RAW_CSV_SEPARATOR)
    _validate_columns(df.columns, config.REG_DATA_COLUMNS, "reg_data.csv")
    df["reg_ts"] = pd.to_datetime(df["reg_ts"], unit="s")
    return df


def load_ab_test_data() -> pd.DataFrame:
    df = pd.read_csv(config.AB_TEST_DATA_PATH, sep=config.RAW_CSV_SEPARATOR)
    _validate_columns(df.columns, config.AB_TEST_COLUMNS, "ab_test.csv")
    return df


def profile_reg_data(df: pd.DataFrame) -> ProfileResult:
    return ProfileResult(
        source_name="reg_data.csv",
        row_count=len(df),
        null_counts=df.isnull().sum().to_dict(),
        duplicate_row_count=int(df.duplicated().sum()),
        timestamp_min=df["reg_ts"].min(),
        timestamp_max=df["reg_ts"].max(),
        unique_players=df["uid"].nunique(),
    )


def profile_ab_test_data(df: pd.DataFrame) -> ProfileResult:
    return ProfileResult(
        source_name="ab_test.csv",
        row_count=len(df),
        null_counts=df.isnull().sum().to_dict(),
        duplicate_row_count=int(df.duplicated().sum()),
        unique_players=df["user_id"].nunique(),
    )


def profile_auth_data() -> ProfileResult:
    """Stream auth_data.csv in chunks — it's too large (~9.6M rows) to profile in memory at once.

    Duplicate-row detection is per-chunk only, so a duplicate pair split across a
    chunk boundary is missed; that's an acceptable approximation for this initial
    ingest pass and gets revisited if Phase 2 profiling finds it matters.
    """
    total_rows = 0
    null_counts = {col: 0 for col in config.AUTH_DATA_COLUMNS}
    duplicate_rows = 0
    ts_min = None
    ts_max = None
    seen_uids: set[int] = set()
    first_chunk = True

    reader = pd.read_csv(
        config.AUTH_DATA_PATH,
        sep=config.RAW_CSV_SEPARATOR,
        chunksize=config.INGEST_CHUNKSIZE,
    )
    for chunk in reader:
        if first_chunk:
            _validate_columns(chunk.columns, config.AUTH_DATA_COLUMNS, "auth_data.csv")
            first_chunk = False

        total_rows += len(chunk)
        for col in config.AUTH_DATA_COLUMNS:
            null_counts[col] += int(chunk[col].isnull().sum())
        duplicate_rows += int(chunk.duplicated().sum())

        chunk_min, chunk_max = chunk["auth_ts"].min(), chunk["auth_ts"].max()
        ts_min = chunk_min if ts_min is None else min(ts_min, chunk_min)
        ts_max = chunk_max if ts_max is None else max(ts_max, chunk_max)

        seen_uids.update(chunk["uid"].unique().tolist())

    return ProfileResult(
        source_name="auth_data.csv",
        row_count=total_rows,
        null_counts=null_counts,
        duplicate_row_count=duplicate_rows,
        timestamp_min=pd.to_datetime(ts_min, unit="s"),
        timestamp_max=pd.to_datetime(ts_max, unit="s"),
        unique_players=len(seen_uids),
    )


def main() -> None:
    reg_df = load_reg_data()
    ab_df = load_ab_test_data()

    print("=" * 60)
    print("Data quality summary")
    print("=" * 60)

    profile_reg_data(reg_df).print_report()
    profile_ab_test_data(ab_df).print_report()
    profile_auth_data().print_report()


if __name__ == "__main__":
    main()
