"""Download and verify the UCI Parkinson's voice dataset.

Source: Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A. E. and
Moroz, I. M. (2007). Exploiting Nonlinear Recurrence and Fractal Scaling
Properties for Voice Disorder Detection. BioMedical Engineering OnLine 6:23.
Dataset: https://archive.ics.uci.edu/dataset/174/parkinsons (CC BY 4.0).

The file is fetched from UCI, never edited by hand, and checked against a pinned
SHA-256 hash so every experiment can be traced back to the exact same bytes.
Only the standard library is used, so this runs before any dependency is installed.

Usage (from the repo root):
    python -m data.download            # download to data/raw/parkinsons.csv
    python -m data.download --force    # re-download even if the file is present
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
# Hash of the file as served by UCI (checked 2026-11-26). If UCI ever changes the
# file, the download fails loudly instead of silently changing every result.
EXPECTED_SHA256 = "a36116c2deac07895b79ce97a2f99a1b3720ac4892995ff5c3b9ccb397ed2bc5"
DEFAULT_DEST = Path(__file__).resolve().parent / "raw" / "parkinsons.csv"

EXPECTED_COLUMNS: tuple[str, ...] = (
    "name",
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "status",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
)
EXPECTED_ROWS = 195

# e.g. "phon_R01_S01_1" -> subject "S01", recording 1.
_NAME_PATTERN = re.compile(r"^phon_R01_(S\d+)_(\d+)$")


class DatasetError(ValueError):
    """The downloaded file is not the dataset we expect."""


@dataclass(frozen=True)
class DatasetSummary:
    """Counts used to sanity-check the dataset (and later, the evaluation report)."""

    n_recordings: int
    n_subjects: int
    n_pd_recordings: int
    n_healthy_recordings: int
    n_pd_subjects: int
    n_healthy_subjects: int
    min_recordings_per_subject: int
    max_recordings_per_subject: int

    @property
    def majority_class_rate(self) -> float:
        """Accuracy of always predicting the most common class (row level)."""
        return max(self.n_pd_recordings, self.n_healthy_recordings) / self.n_recordings


def subject_id(name: str) -> str:
    """Return the subject ID encoded in a recording name.

    Each subject has ~6 recordings, so this ID is what subject-wise
    cross-validation groups on to keep one person out of both train and test.

    >>> subject_id("phon_R01_S01_1")
    'S01'
    """
    match = _NAME_PATTERN.match(name)
    if match is None:
        raise DatasetError(f"Unexpected recording name: {name!r}")
    return match.group(1)


def sha256_hex(content: bytes) -> str:
    """Return the SHA-256 hex digest of `content`."""
    return hashlib.sha256(content).hexdigest()


def parse_and_validate(
    text: str, expected_rows: int | None = EXPECTED_ROWS
) -> list[dict[str, str]]:
    """Parse the CSV text and check its schema.

    Checks the exact column list, the row count (skipped if `expected_rows` is None),
    that every feature is numeric, that `status` is 0 or 1, and that every
    recording name encodes a subject ID.

    Raises:
        DatasetError: if any check fails.
    """
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
        raise DatasetError(f"Unexpected columns: {reader.fieldnames}")

    rows = list(reader)
    if expected_rows is not None and len(rows) != expected_rows:
        raise DatasetError(f"Expected {expected_rows} rows, got {len(rows)}")

    for i, row in enumerate(rows, start=2):  # line 1 is the header
        subject_id(row["name"])
        if row["status"] not in {"0", "1"}:
            raise DatasetError(f"Line {i}: status must be 0 or 1, got {row['status']!r}")
        for column in EXPECTED_COLUMNS[1:]:
            try:
                float(row[column])
            except ValueError:
                raise DatasetError(f"Line {i}: {column} is not numeric: {row[column]!r}") from None
    return rows


def summarize(rows: list[dict[str, str]]) -> DatasetSummary:
    """Count recordings and subjects per class.

    Raises:
        DatasetError: if a subject has recordings with different labels, which
            would make subject-level grouping meaningless.
    """
    labels_by_subject: dict[str, set[str]] = defaultdict(set)
    recordings_by_subject: dict[str, int] = defaultdict(int)
    for row in rows:
        sid = subject_id(row["name"])
        labels_by_subject[sid].add(row["status"])
        recordings_by_subject[sid] += 1

    mixed = sorted(s for s, labels in labels_by_subject.items() if len(labels) > 1)
    if mixed:
        raise DatasetError(f"Subjects with mixed labels: {mixed}")

    n_pd_recordings = sum(row["status"] == "1" for row in rows)
    n_pd_subjects = sum(labels == {"1"} for labels in labels_by_subject.values())
    counts = recordings_by_subject.values()
    return DatasetSummary(
        n_recordings=len(rows),
        n_subjects=len(labels_by_subject),
        n_pd_recordings=n_pd_recordings,
        n_healthy_recordings=len(rows) - n_pd_recordings,
        n_pd_subjects=n_pd_subjects,
        n_healthy_subjects=len(labels_by_subject) - n_pd_subjects,
        min_recordings_per_subject=min(counts),
        max_recordings_per_subject=max(counts),
    )


def fetch(url: str = DATASET_URL, timeout: float = 30.0) -> bytes:
    """Download `url` and return its raw bytes."""
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (fixed https URL)
        return response.read()


def download(
    dest: Path = DEFAULT_DEST,
    *,
    force: bool = False,
    url: str = DATASET_URL,
    expected_sha256: str = EXPECTED_SHA256,
) -> DatasetSummary:
    """Ensure a verified copy of the dataset exists at `dest` and summarize it.

    An existing file with the right hash is reused unless `force` is set.
    The file is only written after the hash and schema checks pass.

    Raises:
        DatasetError: on a hash mismatch or failed schema check.
    """
    if dest.exists() and not force and sha256_hex(dest.read_bytes()) == expected_sha256:
        content = dest.read_bytes()
    else:
        content = fetch(url)
        actual = sha256_hex(content)
        if actual != expected_sha256:
            raise DatasetError(
                f"SHA-256 mismatch: expected {expected_sha256}, got {actual}. "
                "The upstream file has changed; review it before updating the pinned hash."
            )
        parse_and_validate(content.decode("utf-8"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(dest)  # atomic: no half-written file if interrupted

    return summarize(parse_and_validate(content.decode("utf-8")))


def format_summary(summary: DatasetSummary) -> str:
    """Return a human-readable summary for the CLI."""
    return "\n".join(
        [
            f"Recordings: {summary.n_recordings} "
            f"({summary.n_pd_recordings} Parkinson's, {summary.n_healthy_recordings} healthy)",
            f"Subject IDs: {summary.n_subjects} "
            f"({summary.n_pd_subjects} Parkinson's, {summary.n_healthy_subjects} healthy)",
            f"Recordings per subject: {summary.min_recordings_per_subject}"
            f"-{summary.max_recordings_per_subject}",
            f"Majority-class baseline (row level): {summary.majority_class_rate:.2%}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="output CSV path")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    try:
        summary = download(args.dest, force=args.force)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Dataset ready at {args.dest}")
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
