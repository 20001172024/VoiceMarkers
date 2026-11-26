"""Tests for data/download.py. No network: `fetch` is replaced with a fixture."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from data import download as dl


def _row(name: str, status: str, value: str = "0.5") -> str:
    features = [value] * (len(dl.EXPECTED_COLUMNS) - 2)  # all but name and status
    # status sits at index 17 in the real column order
    cells = [name, *features[:16], status, *features[16:]]
    return ",".join(cells)


def _csv(rows: list[str]) -> str:
    return ",".join(dl.EXPECTED_COLUMNS) + "\n" + "\n".join(rows) + "\n"


# Two subjects: S01 (Parkinson's, 3 recordings) and S07 (healthy, 2 recordings).
SMALL_CSV = _csv(
    [
        _row("phon_R01_S01_1", "1"),
        _row("phon_R01_S01_2", "1"),
        _row("phon_R01_S01_3", "1"),
        _row("phon_R01_S07_1", "0"),
        _row("phon_R01_S07_2", "0"),
    ]
)


def test_subject_id_parses_name() -> None:
    assert dl.subject_id("phon_R01_S01_1") == "S01"
    assert dl.subject_id("phon_R01_S50_6") == "S50"


@pytest.mark.parametrize("bad", ["S01_1", "phon_R01_S01", "phon_R02_S01_1", ""])
def test_subject_id_rejects_unexpected_names(bad: str) -> None:
    with pytest.raises(dl.DatasetError):
        dl.subject_id(bad)


def test_parse_and_summarize_small_fixture() -> None:
    rows = dl.parse_and_validate(SMALL_CSV, expected_rows=5)
    summary = dl.summarize(rows)
    assert summary == dl.DatasetSummary(
        n_recordings=5,
        n_subjects=2,
        n_pd_recordings=3,
        n_healthy_recordings=2,
        n_pd_subjects=1,
        n_healthy_subjects=1,
        min_recordings_per_subject=2,
        max_recordings_per_subject=3,
    )
    assert summary.majority_class_rate == pytest.approx(0.6)


def test_rejects_wrong_row_count() -> None:
    with pytest.raises(dl.DatasetError, match="Expected 195 rows"):
        dl.parse_and_validate(SMALL_CSV)


def test_rejects_wrong_columns() -> None:
    text = SMALL_CSV.replace("MDVP:Fo(Hz)", "Fo", 1)
    with pytest.raises(dl.DatasetError, match="Unexpected columns"):
        dl.parse_and_validate(text, expected_rows=None)


def test_rejects_non_numeric_feature() -> None:
    text = _csv([_row("phon_R01_S01_1", "1", value="abc")])
    with pytest.raises(dl.DatasetError, match="not numeric"):
        dl.parse_and_validate(text, expected_rows=None)


def test_rejects_bad_status() -> None:
    text = _csv([_row("phon_R01_S01_1", "2")])
    with pytest.raises(dl.DatasetError, match="status"):
        dl.parse_and_validate(text, expected_rows=None)


def test_rejects_subject_with_mixed_labels() -> None:
    text = _csv([_row("phon_R01_S01_1", "1"), _row("phon_R01_S01_2", "0")])
    with pytest.raises(dl.DatasetError, match="mixed labels"):
        dl.summarize(dl.parse_and_validate(text, expected_rows=None))


def _fake_dataset(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, str]:
    """Make `fetch` return a 195-row fixture and return (content, sha256)."""
    rows = [_row(f"phon_R01_S{i // 5:02d}_{i % 5 + 1}", "1") for i in range(195)]
    content = _csv(rows).encode()
    monkeypatch.setattr(dl, "fetch", lambda url=dl.DATASET_URL: content)
    return content, hashlib.sha256(content).hexdigest()


def test_download_writes_verified_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content, digest = _fake_dataset(monkeypatch)
    dest = tmp_path / "raw" / "parkinsons.csv"
    summary = dl.download(dest, expected_sha256=digest)
    assert dest.read_bytes() == content
    assert summary.n_recordings == 195


def test_download_refuses_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_dataset(monkeypatch)
    dest = tmp_path / "parkinsons.csv"
    with pytest.raises(dl.DatasetError, match="SHA-256 mismatch"):
        dl.download(dest, expected_sha256="0" * 64)
    assert not dest.exists()  # nothing written on failure


def test_download_reuses_valid_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content, digest = _fake_dataset(monkeypatch)
    dest = tmp_path / "parkinsons.csv"
    dest.write_bytes(content)

    def fail(url: str = dl.DATASET_URL) -> bytes:
        raise AssertionError("should not re-download a valid file")

    monkeypatch.setattr(dl, "fetch", fail)
    assert dl.download(dest, expected_sha256=digest).n_recordings == 195
