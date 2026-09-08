"""Load the raw KKBox files, validate their schema, and print a data quality summary.

Four raw files, joined on member id (msno):
  members_v3.csv  - one row per member: demographics, registration date
  transactions.csv - one row per subscription transaction (~21.5M rows, read in chunks)
  user_logs_v2.csv - one row per member per day of March 2017 listening activity (~18.4M rows, read in chunks)
  train_v2.csv     - the competition's own is_churn label — reference/validation only,
                      not our source of truth (see spec Phase 3)

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
    date_min: object = None
    date_max: object = None
    unique_members: int | None = None
    extra_notes: dict = field(default_factory=dict)

    def print_report(self) -> None:
        print(f"\n--- {self.source_name} ---")
        print(f"rows: {self.row_count:,}")
        if self.unique_members is not None:
            print(f"unique members: {self.unique_members:,}")
        if self.date_min is not None:
            print(f"date span: {self.date_min} to {self.date_max}")
        print(f"duplicate rows: {self.duplicate_row_count:,}")
        for label, value in self.extra_notes.items():
            print(f"{label}: {value}")
        print("null counts by column:")
        for col, n in self.null_counts.items():
            print(f"  {col}: {n:,}")


def load_members() -> pd.DataFrame:
    df = pd.read_csv(config.MEMBERS_PATH)
    _validate_columns(df.columns, config.MEMBERS_COLUMNS, "members_v3.csv")
    df["registration_init_time"] = pd.to_datetime(df["registration_init_time"], format="%Y%m%d")
    return df


def load_train_labels() -> pd.DataFrame:
    df = pd.read_csv(config.TRAIN_LABELS_PATH)
    _validate_columns(df.columns, config.TRAIN_LABELS_COLUMNS, "train_v2.csv")
    return df


def profile_members(df: pd.DataFrame) -> ProfileResult:
    return ProfileResult(
        source_name="members_v3.csv",
        row_count=len(df),
        null_counts=df.isnull().sum().to_dict(),
        duplicate_row_count=int(df.duplicated().sum()),
        date_min=df["registration_init_time"].min(),
        date_max=df["registration_init_time"].max(),
        unique_members=df["msno"].nunique(),
    )


def profile_train_labels(df: pd.DataFrame) -> ProfileResult:
    return ProfileResult(
        source_name="train_v2.csv (reference label only)",
        row_count=len(df),
        null_counts=df.isnull().sum().to_dict(),
        duplicate_row_count=int(df.duplicated().sum()),
        unique_members=df["msno"].nunique(),
        extra_notes={"is_churn rate": round(df["is_churn"].mean(), 4)},
    )


def profile_transactions() -> ProfileResult:
    """Stream transactions.csv in chunks — it's too large (~21.5M rows) to profile in memory at once.

    Duplicate-row detection is per-chunk only, so a duplicate pair split across a
    chunk boundary is missed; acceptable approximation for this initial ingest pass.
    """
    total_rows = 0
    null_counts = {col: 0 for col in config.TRANSACTIONS_COLUMNS}
    duplicate_rows = 0
    date_min, date_max = None, None
    negative_paid = 0
    cancel_count = 0
    seen_msno: set[str] = set()
    first_chunk = True

    reader = pd.read_csv(config.TRANSACTIONS_PATH, chunksize=config.INGEST_CHUNKSIZE)
    for chunk in reader:
        if first_chunk:
            _validate_columns(chunk.columns, config.TRANSACTIONS_COLUMNS, "transactions.csv")
            first_chunk = False

        total_rows += len(chunk)
        for col in config.TRANSACTIONS_COLUMNS:
            null_counts[col] += int(chunk[col].isnull().sum())
        duplicate_rows += int(chunk.duplicated().sum())

        chunk_min, chunk_max = chunk["transaction_date"].min(), chunk["transaction_date"].max()
        date_min = chunk_min if date_min is None else min(date_min, chunk_min)
        date_max = chunk_max if date_max is None else max(date_max, chunk_max)

        negative_paid += int((chunk["actual_amount_paid"] < 0).sum())
        cancel_count += int(chunk["is_cancel"].sum())
        seen_msno.update(chunk["msno"].unique().tolist())

    return ProfileResult(
        source_name="transactions.csv",
        row_count=total_rows,
        null_counts=null_counts,
        duplicate_row_count=duplicate_rows,
        date_min=pd.to_datetime(str(date_min), format="%Y%m%d"),
        date_max=pd.to_datetime(str(date_max), format="%Y%m%d"),
        unique_members=len(seen_msno),
        extra_notes={
            "negative actual_amount_paid": negative_paid,
            "cancel rate": round(cancel_count / total_rows, 4),
        },
    )


def profile_user_logs() -> ProfileResult:
    """Stream user_logs_v2.csv in chunks — ~18.4M rows, one row per member per day."""
    total_rows = 0
    null_counts = {col: 0 for col in config.USER_LOGS_COLUMNS}
    duplicate_rows = 0
    date_min, date_max = None, None
    negative_secs = 0
    seen_msno: set[str] = set()
    first_chunk = True

    reader = pd.read_csv(config.USER_LOGS_PATH, chunksize=config.INGEST_CHUNKSIZE)
    for chunk in reader:
        if first_chunk:
            _validate_columns(chunk.columns, config.USER_LOGS_COLUMNS, "user_logs_v2.csv")
            first_chunk = False

        total_rows += len(chunk)
        for col in config.USER_LOGS_COLUMNS:
            null_counts[col] += int(chunk[col].isnull().sum())
        duplicate_rows += int(chunk.duplicated().sum())

        chunk_min, chunk_max = chunk["date"].min(), chunk["date"].max()
        date_min = chunk_min if date_min is None else min(date_min, chunk_min)
        date_max = chunk_max if date_max is None else max(date_max, chunk_max)

        negative_secs += int((chunk["total_secs"] < 0).sum())
        seen_msno.update(chunk["msno"].unique().tolist())

    return ProfileResult(
        source_name="user_logs_v2.csv",
        row_count=total_rows,
        null_counts=null_counts,
        duplicate_row_count=duplicate_rows,
        date_min=pd.to_datetime(str(date_min), format="%Y%m%d"),
        date_max=pd.to_datetime(str(date_max), format="%Y%m%d"),
        unique_members=len(seen_msno),
        extra_notes={"negative total_secs": negative_secs},
    )


def main() -> None:
    members_df = load_members()
    labels_df = load_train_labels()

    print("=" * 60)
    print("Data quality summary")
    print("=" * 60)

    profile_members(members_df).print_report()
    profile_train_labels(labels_df).print_report()
    profile_transactions().print_report()
    profile_user_logs().print_report()


if __name__ == "__main__":
    main()
