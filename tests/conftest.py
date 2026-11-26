"""Shared fixtures: SYNTHETIC datasets in the real UCI CSV schema (no download needed)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.download import EXPECTED_COLUMNS
from research.dataset import FEATURE_COLUMNS


def write_synthetic_csv(path: Path, n_pd: int, n_healthy: int, per_subject: int = 6) -> Path:
    """Write a synthetic dataset with `n_pd + n_healthy` subjects.

    Two features are shifted by class (so models can learn something) and every
    subject gets a personal offset on all features (so, like real voices,
    recordings of one person resemble each other).
    """
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n_pd + n_healthy):
        status = 1 if s < n_pd else 0
        speaker = rng.normal(scale=0.5, size=len(FEATURE_COLUMNS))
        for r in range(per_subject):
            features = speaker + rng.normal(size=len(FEATURE_COLUMNS))
            features[:2] += 1.5 * status
            row = dict(zip(FEATURE_COLUMNS, features, strict=True))
            rows.append({"name": f"phon_R01_S{s + 1:02d}_{r + 1}", "status": status, **row})
    pd.DataFrame(rows)[list(EXPECTED_COLUMNS)].to_csv(path, index=False)
    return path


@pytest.fixture
def make_csv(tmp_path: Path) -> Callable[..., Path]:
    """Factory fixture: make_csv(n_pd, n_healthy) -> path to a synthetic CSV."""

    def _make(n_pd: int, n_healthy: int) -> Path:
        return write_synthetic_csv(tmp_path / f"synthetic_{n_pd}_{n_healthy}.csv", n_pd, n_healthy)

    return _make


@pytest.fixture
def synthetic_csv(make_csv: Callable[..., Path]) -> Path:
    """60 recordings from 10 subjects (7 'Parkinson's', 3 healthy)."""
    return make_csv(7, 3)
