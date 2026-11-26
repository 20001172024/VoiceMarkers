"""Load the UCI Parkinson's dataset in one consistent way for every research script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.download import DEFAULT_DEST, EXPECTED_COLUMNS, parse_and_validate, sha256_hex, subject_id

LABEL_COLUMN = "status"  # 1 = Parkinson's, 0 = healthy
FEATURE_COLUMNS: tuple[str, ...] = tuple(
    c for c in EXPECTED_COLUMNS if c not in {"name", LABEL_COLUMN}
)


@dataclass(frozen=True)
class Dataset:
    """Features, labels and subject IDs, aligned row by row."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series  # subject ID per recording, for subject-wise CV (M2)
    sha256: str

    def select(self, columns: tuple[str, ...] | list[str]) -> Dataset:
        """Return a copy restricted to the given feature columns (same rows)."""
        missing = set(columns) - set(self.X.columns)
        if missing:
            raise KeyError(f"Unknown feature columns: {sorted(missing)}")
        return Dataset(self.X[list(columns)], self.y, self.groups, self.sha256)


def load_dataset(path: Path = DEFAULT_DEST) -> Dataset:
    """Read and validate the dataset CSV written by `data/download.py`.

    Validation reuses the download script's schema checks; the row count is not
    enforced here so tests can use small fixture files.

    Raises:
        FileNotFoundError: if the CSV has not been downloaded yet.
        data.download.DatasetError: if the file fails the schema checks.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m data.download` first.")
    content = path.read_bytes()
    parse_and_validate(content.decode("utf-8"), expected_rows=None)

    df = pd.read_csv(path)
    return Dataset(
        X=df[list(FEATURE_COLUMNS)].astype(float),
        y=df[LABEL_COLUMN].astype(int),
        groups=df["name"].map(subject_id),
        sha256=sha256_hex(content),
    )
