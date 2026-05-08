# scripts/create_data_splits.py
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")

TIMESTAMP_COLUMNS = (
    "quote_timestamp",
    "timestamp",
    "date",
    "downloaded_at",
    "lastTradeDate",
)

EXPIRATION_COLUMNS = (
    "expiration_date",
    "expiration",
    "expiry",
    "expirationDate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/validation/test splits preserving the Data folder structure."
    )
    parser.add_argument(
        "--data-root",
        default="Data",
        help="Root folder containing the original data. Default: Data",
    )
    parser.add_argument(
        "--output-root",
        default="Splits",
        help="Root folder where split data will be written. Default: Splits",
    )
    parser.add_argument(
        "--strategy",
        choices=("expiration", "timestamp"),
        default="expiration",
        help=(
            "Split criterion. Use 'expiration' for the current one-snapshot options dataset; "
            "use 'timestamp' when multiple option snapshots exist."
        ),
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.60,
        help="Fraction of ordered split keys assigned to train. Default: 0.60",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of ordered split keys assigned to validation. Default: 0.20",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Fraction of ordered split keys assigned to test. Default: 0.20",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the target split folder before writing new splits.",
    )
    parser.add_argument(
        "--copy-non-csv",
        action="store_true",
        help="Copy non-CSV files into shared/ preserving the original structure.",
    )
    return parser.parse_args()


def validate_fractions(train: float, validation: float, test: float) -> None:
    total = train + validation + test
    if any(value < 0 for value in (train, validation, test)):
        raise ValueError("Split fractions must be non-negative.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0. Got {total:.6f}."
        )


def select_split_column(df: pd.DataFrame, strategy: str) -> tuple[str | None, str]:
    columns = set(df.columns)

    if strategy == "expiration":
        for column in EXPIRATION_COLUMNS:
            if column in columns:
                return column, "expiration"

        for column in TIMESTAMP_COLUMNS:
            if column in columns:
                return column, "timestamp_fallback"

    if strategy == "timestamp":
        for column in TIMESTAMP_COLUMNS:
            if column in columns:
                return column, "timestamp"

        for column in EXPIRATION_COLUMNS:
            if column in columns:
                return column, "expiration_fallback"

    return None, "shared"


def build_key_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)

    if parsed.notna().any():
        return parsed

    return series.astype(str)


def ordered_unique_keys(key_series: pd.Series) -> list:
    values = key_series.dropna().unique().tolist()
    return sorted(values)


def allocate_keys(
    keys: list,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, set]:
    n = len(keys)
    allocation = {name: set() for name in SPLIT_NAMES}

    if n == 0:
        return allocation

    if n == 1:
        allocation["train"].add(keys[0])
        return allocation

    if n == 2:
        allocation["train"].add(keys[0])
        allocation["test"].add(keys[1])
        return allocation

    train_count = max(1, int(n * train_fraction))
    validation_count = max(1, int(n * validation_fraction))

    while train_count + validation_count > n - 1:
        if train_count >= validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            break

    test_count = n - train_count - validation_count

    if test_count < 1:
        test_count = 1
        if validation_count > 1:
            validation_count -= 1
        elif train_count > 1:
            train_count -= 1

    allocation["train"] = set(keys[:train_count])
    allocation["validation"] = set(keys[train_count : train_count + validation_count])
    allocation["test"] = set(keys[train_count + validation_count :])

    return allocation


def key_to_text(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def write_dataframe(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)


def copy_to_shared(source: Path, data_root: Path, output_base: Path) -> Path:
    relative_path = Path(data_root.name) / source.relative_to(data_root)
    target = output_base / "shared" / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def iter_files(data_root: Path, copy_non_csv: bool) -> Iterable[Path]:
    pattern = "**/*" if copy_non_csv else "**/*.csv"

    for path in sorted(data_root.glob(pattern)):
        if path.is_file():
            yield path


def split_csv_file(
    source: Path,
    data_root: Path,
    output_base: Path,
    strategy: str,
    train_fraction: float,
    validation_fraction: float,
) -> list[dict]:
    manifest_rows: list[dict] = []

    try:
        df = pd.read_csv(source)
    except Exception as exc:
        target = copy_to_shared(source, data_root, output_base)
        manifest_rows.append(
            {
                "source_path": str(source),
                "target_path": str(target),
                "split": "shared",
                "rows": None,
                "split_column": None,
                "split_mode": "read_error_shared",
                "key_start": None,
                "key_end": None,
                "notes": f"Could not read CSV: {exc}",
            }
        )
        return manifest_rows

    split_column, split_mode = select_split_column(df, strategy)

    if split_column is None:
        target = copy_to_shared(source, data_root, output_base)
        manifest_rows.append(
            {
                "source_path": str(source),
                "target_path": str(target),
                "split": "shared",
                "rows": len(df),
                "split_column": None,
                "split_mode": split_mode,
                "key_start": None,
                "key_end": None,
                "notes": "No recognized split column. Copied to shared.",
            }
        )
        return manifest_rows

    key_series = build_key_series(df[split_column])
    keys = ordered_unique_keys(key_series)
    allocation = allocate_keys(keys, train_fraction, validation_fraction)

    relative_path = Path(data_root.name) / source.relative_to(data_root)

    for split_name in SPLIT_NAMES:
        split_keys = allocation[split_name]

        if not split_keys:
            continue

        mask = key_series.isin(split_keys)
        split_df = df.loc[mask].copy()

        if split_df.empty:
            continue

        target = output_base / split_name / relative_path
        write_dataframe(split_df, target)

        ordered_split_keys = sorted(split_keys)

        manifest_rows.append(
            {
                "source_path": str(source),
                "target_path": str(target),
                "split": split_name,
                "rows": len(split_df),
                "split_column": split_column,
                "split_mode": split_mode,
                "key_start": key_to_text(ordered_split_keys[0]),
                "key_end": key_to_text(ordered_split_keys[-1]),
                "notes": "",
            }
        )

    missing_key_rows = df.loc[key_series.isna()].copy()
    if not missing_key_rows.empty:
        target = output_base / "shared" / relative_path
        write_dataframe(missing_key_rows, target)
        manifest_rows.append(
            {
                "source_path": str(source),
                "target_path": str(target),
                "split": "shared",
                "rows": len(missing_key_rows),
                "split_column": split_column,
                "split_mode": f"{split_mode}_missing_keys",
                "key_start": None,
                "key_end": None,
                "notes": "Rows with missing split key were written to shared.",
            }
        )

    return manifest_rows


def write_manifests(output_base: Path, manifest_rows: list[dict], args: argparse.Namespace) -> None:
    manifest_dir = output_base / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    summary_path = manifest_dir / "split_summary.csv"
    pd.DataFrame(manifest_rows).to_csv(summary_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": args.strategy,
        "data_root": str(Path(args.data_root)),
        "output_root": str(Path(args.output_root)),
        "fractions": {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
        },
        "files": manifest_rows,
    }

    manifest_path = manifest_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    validate_fractions(
        args.train_fraction,
        args.validation_fraction,
        args.test_fraction,
    )

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    output_root = Path(args.output_root)
    output_base = output_root / f"{args.strategy}_split"

    if args.clear_output and output_base.exists():
        shutil.rmtree(output_base)

    for split_name in (*SPLIT_NAMES, "shared", "manifests"):
        (output_base / split_name).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    for source in iter_files(data_root, args.copy_non_csv):
        if source.suffix.lower() != ".csv":
            target = copy_to_shared(source, data_root, output_base)
            manifest_rows.append(
                {
                    "source_path": str(source),
                    "target_path": str(target),
                    "split": "shared",
                    "rows": None,
                    "split_column": None,
                    "split_mode": "non_csv_shared",
                    "key_start": None,
                    "key_end": None,
                    "notes": "Non-CSV file copied to shared.",
                }
            )
            continue

        manifest_rows.extend(
            split_csv_file(
                source=source,
                data_root=data_root,
                output_base=output_base,
                strategy=args.strategy,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
            )
        )

    write_manifests(output_base, manifest_rows, args)

    print(f"[OK] Created split folder: {output_base}")
    print(f"[OK] Manifest: {output_base / 'manifests' / 'split_manifest.json'}")
    print(f"[OK] Summary:  {output_base / 'manifests' / 'split_summary.csv'}")


if __name__ == "__main__":
    main()