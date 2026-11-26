"""Tests for the M3 research scripts: feature subset, validation, final training."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pytest

from app.audio.features import REPRODUCIBLE_FEATURES, extract_features
from app.audio.load import TARGET_SR, Audio
from data.make_samples import SAMPLES_DIR, VoiceSpec, synth_vowel
from research import evaluate as ev
from research import feature_validation as fv
from research import report
from research import train_final as tf
from research.dataset import Dataset, load_dataset
from research.models import BASELINE

MODELS = (BASELINE, "Logistic Regression", "KNN")
SMALL = {"n_splits": 3, "n_repeats": 1, "inner_splits": 2}


@pytest.fixture(autouse=True)
def single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "N_JOBS", 1)


@pytest.fixture
def ds(make_csv: Callable[..., Path]) -> Dataset:
    return load_dataset(make_csv(10, 6)).select(REPRODUCIBLE_FEATURES)


@pytest.fixture
def evaluation(ds: Dataset) -> dict[str, Any]:
    results = ev.run(ds, models=MODELS, feature_set="reproducible", **SMALL)
    return json.loads(json.dumps(results, default=float))  # as read back from disk


def test_select_keeps_rows_and_requested_columns(make_csv: Callable[..., Path]) -> None:
    full = load_dataset(make_csv(10, 6))
    sub = full.select(REPRODUCIBLE_FEATURES)
    assert list(sub.X.columns) == list(REPRODUCIBLE_FEATURES)
    assert len(sub.X) == len(full.X) and sub.y.equals(full.y) and sub.groups.equals(full.groups)
    with pytest.raises(KeyError, match="Unknown"):
        full.select(["not_a_feature"])


def test_evaluation_records_feature_set(evaluation: dict[str, Any]) -> None:
    assert evaluation["config"]["feature_set"] == "reproducible"
    assert evaluation["config"]["features"] == list(REPRODUCIBLE_FEATURES)


# ----------------------------------------------------------- model selection


def _fake_evaluation(lr: float, best: float, std: float = 0.1) -> dict[str, Any]:
    def entry(mean: float) -> dict[str, Any]:
        return {"subject_tuned": {"summary": {"balanced_accuracy": {"mean": mean, "std": std}}}}

    return {
        "config": {"n_splits": 5, "n_repeats": 4},  # 20 outer folds -> SE = std / sqrt(20)
        "cv": {BASELINE: entry(0.5), "Logistic Regression": entry(lr), "KNN": entry(best)},
    }


def test_prefers_logistic_regression_within_one_standard_error() -> None:
    se = 0.1 / 20**0.5  # ~0.022
    name, why = tf.select_model(_fake_evaluation(lr=0.64, best=0.64 + se * 0.9))
    assert name == "Logistic Regression"
    assert why["best_model"] == "KNN"
    assert why["standard_error"] == pytest.approx(se)


def test_switches_to_best_model_when_lr_is_clearly_worse() -> None:
    name, _ = tf.select_model(_fake_evaluation(lr=0.55, best=0.65))
    assert name == "KNN"


def test_baseline_is_never_selected() -> None:
    name, why = tf.select_model(_fake_evaluation(lr=0.4, best=0.45))
    assert why["best_model"] != BASELINE and name != BASELINE


# --------------------------------------------------------------- training


def test_main_saves_a_loadable_model_with_metadata(
    ds: Dataset,
    evaluation: dict[str, Any],
    make_csv: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf, "FINAL_CV_SPLITS", 3)
    eval_path = tmp_path / "evaluation_reproducible.json"
    eval_path.write_text(json.dumps(evaluation))
    out = tmp_path / "models"
    args = ["--data", str(make_csv(10, 6)), "--evaluation", str(eval_path)]
    args += ["--evaluation-full", str(tmp_path / "missing.json"), "--out-dir", str(out)]
    assert tf.main(args) == 0

    model = joblib.load(out / "model.joblib")
    meta = json.loads((out / "model_metadata.json").read_text())
    assert meta["features"] == list(REPRODUCIBLE_FEATURES)
    assert meta["model_type"] in {"Logistic Regression", "KNN"}
    assert meta["evaluation"]["all_22_features"] is None
    assert meta["base_rate"] == pytest.approx(60 / 96)  # 10 of 16 synthetic subjects
    assert len(meta["explanation_background"]) == tf.BACKGROUND_SIZE
    assert all(len(row) == len(REPRODUCIBLE_FEATURES) for row in meta["explanation_background"])
    for feature in REPRODUCIBLE_FEATURES:
        dist = meta["feature_distributions"][feature]
        assert set(dist) == {"all", "parkinsons", "healthy"}
        assert dist["all"]["p5"] <= dist["all"]["p50"] <= dist["all"]["p95"]

    # The saved model accepts features extracted from audio, in metadata order.
    features = extract_features(Audio(synth_vowel(VoiceSpec()), TARGET_SR))
    import pandas as pd

    row = pd.DataFrame([[features[f] for f in meta["features"]]], columns=meta["features"])
    proba = model.predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_main_refuses_an_evaluation_on_the_wrong_features(
    make_csv: Callable[..., Path], tmp_path: Path
) -> None:
    full = ev.run(load_dataset(make_csv(10, 6)), models=(BASELINE, "KNN"), **SMALL)
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(full, default=float))
    with pytest.raises(SystemExit, match="not run on the reproducible features"):
        tf.main(["--evaluation", str(path), "--out-dir", str(tmp_path)])


# ------------------------------------------------------- feature validation


def test_feature_validation_on_committed_samples(synthetic_csv: Path, tmp_path: Path) -> None:
    assert fv.main(["--data", str(synthetic_csv), "--out-dir", str(tmp_path)]) == 0
    results = json.loads((tmp_path / "feature_validation.json").read_text())
    assert set(results["samples"]) == {p.stem for p in SAMPLES_DIR.glob("*.wav")}
    for rows in results["samples"].values():
        assert [r["feature"] for r in rows] == list(REPRODUCIBLE_FEATURES)
    md = (tmp_path / "feature_validation.md").read_text()
    assert "NOT monotonic" not in md  # every sensitivity sweep moves the right way


def test_range_check_flags_out_of_range_values(synthetic_csv: Path) -> None:
    data = load_dataset(synthetic_csv)
    extreme = {f: 1e6 for f in REPRODUCIBLE_FEATURES}
    rows = fv.range_check(extreme, data)
    assert not any(r["in_range"] for r in rows)
    assert all(r["percentile"] == 100 for r in rows)


# ------------------------------------------------------------------ report


def test_report_includes_table_3_and_deployed_model(
    evaluation: dict[str, Any], synthetic_csv: Path, tmp_path: Path
) -> None:
    from research import reproduce_paper as rp

    # The report's discussion is written about SVM, so the full-feature run includes it.
    full = ev.run(load_dataset(synthetic_csv), models=(*MODELS, "SVM"), **SMALL)
    paper = rp.run(load_dataset(synthetic_csv), n_repeats=2, n_folds=3)
    meta = {
        "model_type": "Logistic Regression",
        "model_version": "1.0.0",
        "trained_at": "2026-01-01",
        "params": {"C": 0.1},
        "features": list(REPRODUCIBLE_FEATURES),
        "selection": {
            "rule": "rule",
            "best_model": "KNN",
            "best_mean": 0.6,
            "preferred_mean": 0.59,
            "gap": 0.01,
            "standard_error": 0.02,
        },
        "evaluation": {
            "reproducible_features": evaluation["cv"]["Logistic Regression"]["subject_tuned"][
                "summary"
            ]
        },
    }
    paths = {}
    for name, data in {"paper": paper, "ev": full, "repro": evaluation, "meta": meta}.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(data, default=float))
    args = ["--paper", str(paths["paper"]), "--evaluation", str(paths["ev"])]
    args += [
        "--evaluation-reproducible",
        str(paths["repro"]),
        "--model-metadata",
        str(paths["meta"]),
    ]
    assert (
        report.main(
            [
                *args,
                "--docs-dir",
                str(tmp_path / "docs"),
                "--frontend-summary",
                str(tmp_path / "summary.json"),
            ]
        )
        == 0
    )

    md = (tmp_path / "docs" / "evaluation.md").read_text()
    section5 = md.split("## 5.")[1].split("## 6.")[0]
    assert "Not generated yet" not in section5
    assert "### Deployed model" in section5
    assert "`NHR`" in section5 and "`PPE`" in section5  # dropped features listed

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["deployed"]["model_type"] == "Logistic Regression"
    assert set(summary["reproducible"]["dropped"]) >= {"NHR", "PPE"}
    assert summary["dataset"]["n_recordings"] == 60  # the 10-subject fixture
    assert "SVM" in summary["protocols"]
