"""Tests for research/ (dataset loading, models, metrics, reproduction script).

They run on a small SYNTHETIC dataset written in the real CSV schema, so no
download is needed in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research import reproduce_paper as rp
from research.dataset import FEATURE_COLUMNS, load_dataset
from research.metrics import compute_metrics, evaluate_fitted
from research.models import BASELINE, MODEL_NAMES, build_model


def test_load_dataset_aligns_features_labels_groups(synthetic_csv: Path) -> None:
    ds = load_dataset(synthetic_csv)
    assert list(ds.X.columns) == list(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 22
    assert len(ds.X) == len(ds.y) == len(ds.groups) == 60
    assert ds.groups.nunique() == 10
    assert ds.groups.iloc[0] == "S01"
    assert ds.y.sum() == 42


def test_load_dataset_missing_file_hints_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data.download"):
        load_dataset(tmp_path / "missing.csv")


def test_compute_metrics_hand_checked_case() -> None:
    # 4 Parkinson's, 2 healthy. Predictions: TP=3, FN=1, TN=1, FP=1.
    y_true = [1, 1, 1, 1, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 1]
    y_score = [0.9, 0.8, 0.7, 0.4, 0.1, 0.6]
    m = compute_metrics(y_true, y_pred, y_score)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(1 / 2)
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 1 / 2) / 2)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(3 / 4)
    # 8 positive/negative pairs, 7 ranked correctly (0.4 < 0.6 is the miss)
    assert m["roc_auc"] == pytest.approx(7 / 8)


@pytest.mark.filterwarnings("ignore:A single label was found")
def test_compute_metrics_single_class_auc_is_nan() -> None:
    m = compute_metrics([1, 1], [1, 1], [0.5, 0.6])
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["specificity"])


def test_scaler_is_fitted_on_training_rows_only(synthetic_csv: Path) -> None:
    ds = load_dataset(synthetic_csv)
    X_train, y_train = ds.X.iloc[::2], ds.y.iloc[::2]  # both classes, half the rows
    model = build_model("SVM").fit(X_train, y_train)
    scaler = model.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
    assert not np.allclose(scaler.mean_, ds.X.mean().to_numpy())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_scores(synthetic_csv: Path, name: str) -> None:
    ds = load_dataset(synthetic_csv)
    m = evaluate_fitted(build_model(name).fit(ds.X, ds.y), ds.X, ds.y)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert not np.isnan(m["roc_auc"])


def test_unknown_model_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("Naive Bayes")


def test_majority_baseline_predicts_majority_class(synthetic_csv: Path) -> None:
    ds = load_dataset(synthetic_csv)
    m = rp.single_split(ds, BASELINE, seed=0)
    assert m["fp"] + m["tp"] == m["tn"] + m["fp"] + m["fn"] + m["tp"]  # all predicted positive
    assert m["recall"] == 1.0 and m["specificity"] == 0.0
    assert m["roc_auc"] == pytest.approx(0.5)


def test_single_split_is_deterministic(synthetic_csv: Path) -> None:
    ds = load_dataset(synthetic_csv)
    assert rp.single_split(ds, "Random Forest", 3) == rp.single_split(ds, "Random Forest", 3)


def test_single_split_uses_20_percent_test_set(synthetic_csv: Path) -> None:
    m = rp.single_split(load_dataset(synthetic_csv), "SVM", 0)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 12  # 20% of 60


def test_paper_accuracy_is_34_of_39() -> None:
    assert f"{rp.PAPER_SVM_ACCURACY:.2%}" == "87.18%"


def test_main_writes_json_and_markdown(synthetic_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    rc = rp.main(
        ["--data", str(synthetic_csv), "--out-dir", str(out), "--n-repeats", "3", "--n-folds", "3"]
    )
    assert rc == 0
    results = json.loads((out / "paper_reproduction.json").read_text())
    assert set(results["single_split"]) == set(MODEL_NAMES)
    assert results["config"]["n_repeats"] == 3
    report = (out / "paper_reproduction.md").read_text()
    assert "row-level" in report
    for name in MODEL_NAMES:
        assert name in report
