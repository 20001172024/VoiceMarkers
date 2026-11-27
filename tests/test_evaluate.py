"""Tests for research/evaluate.py and research/report.py on SYNTHETIC data."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from research import evaluate as ev
from research import report
from research import reproduce_paper as rp
from research.dataset import Dataset, load_dataset
from research.models import BASELINE

SMALL = {"n_splits": 3, "n_repeats": 2, "inner_splits": 2}
MODELS = (BASELINE, "SVM", "KNN")


@pytest.fixture(autouse=True)
def single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "N_JOBS", 1)


@pytest.fixture
def ds(make_csv: Callable[..., Path]) -> Dataset:
    """16 subjects (10 'Parkinson's', 6 healthy): enough healthy subjects for inner folds."""
    return load_dataset(make_csv(10, 6))


def test_subject_wise_folds_never_share_a_subject(ds: Dataset) -> None:
    folds = list(ev.outer_folds(ds, subject_wise=True, n_splits=3, n_repeats=2))
    assert len(folds) == 6
    for f in folds:
        train_subjects = set(ds.groups.iloc[f.train])
        test_subjects = set(ds.groups.iloc[f.test])
        assert train_subjects.isdisjoint(test_subjects)
        assert set(ds.y.iloc[f.test]) == {0, 1}  # stratified: both classes in every test fold


def test_each_repeat_covers_every_recording_once(ds: Dataset) -> None:
    for subject_wise in (True, False):
        folds = [f for f in ev.outer_folds(ds, subject_wise, 3, 2) if f.repeat == 1]
        test_rows = np.concatenate([f.test for f in folds])
        assert sorted(test_rows) == list(range(len(ds.y)))


def test_row_level_folds_do_share_subjects(ds: Dataset) -> None:
    """The leak the subject-wise protocol removes is really there at row level."""
    f = next(ev.outer_folds(ds, subject_wise=False, n_splits=3, n_repeats=1))
    assert set(ds.groups.iloc[f.train]) & set(ds.groups.iloc[f.test])


def test_tuning_uses_subject_wise_inner_cv(ds: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    original_fit = GridSearchCV.fit

    def spy_fit(self: GridSearchCV, X: Any, y: Any = None, **kwargs: Any) -> GridSearchCV:
        seen["cv"] = self.cv
        seen["groups"] = kwargs.get("groups")
        seen["scoring"] = self.scoring
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(GridSearchCV, "fit", spy_fit)
    _, params = ev.fit_model("SVM", ds.X, ds.y, ds.groups, tune=True, seed=0, inner_splits=2)
    assert isinstance(seen["cv"], StratifiedGroupKFold)
    assert seen["groups"] is not None and seen["groups"].equals(ds.groups)
    assert seen["scoring"] == "balanced_accuracy"
    assert set(params) == set(ev.PARAM_GRIDS["SVM"])

    # The inner splits themselves never share a subject either.
    for train, test in seen["cv"].split(ds.X, ds.y, ds.groups):
        assert set(ds.groups.iloc[train]).isdisjoint(ds.groups.iloc[test])


def test_untuned_and_baseline_skip_the_search(ds: Dataset) -> None:
    assert ev.fit_model("SVM", ds.X, ds.y, ds.groups, tune=False, seed=0)[1] == {}
    assert ev.fit_model(BASELINE, ds.X, ds.y, ds.groups, tune=True, seed=0)[1] == {}


def test_models_are_only_fitted_on_training_subjects(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy on fit_model: no test-fold subject ever reaches training or tuning."""
    trained_on: list[set[str]] = []
    original = ev.fit_model

    def spy(name: str, X: Any, y: Any, groups: Any, **kwargs: Any) -> Any:
        trained_on.append(set(groups))
        return original(name, X, y, groups, **kwargs)

    monkeypatch.setattr(ev, "fit_model", spy)
    ev.cross_validate(ds, "SVM", "subject_tuned", **SMALL)
    folds = list(ev.outer_folds(ds, True, SMALL["n_splits"], SMALL["n_repeats"]))
    assert len(trained_on) == len(folds)
    for subjects, fold in zip(trained_on, folds, strict=True):
        assert subjects.isdisjoint(ds.groups.iloc[fold.test])


def test_cross_validate_outputs(ds: Dataset) -> None:
    res = ev.cross_validate(ds, "SVM", "subject_tuned", **SMALL)
    assert len(res["folds"]["balanced_accuracy"]) == 6
    assert -1 not in res["oof_repeat0"]["pred"]  # every recording predicted once
    assert sum(c["folds"] for c in res["chosen_params"]) == 6
    assert 0.0 <= res["summary"]["balanced_accuracy"]["mean"] <= 1.0


def test_loso_pools_predictions_over_all_subjects(ds: Dataset) -> None:
    m = ev.leave_one_subject_out(ds, "KNN", tune=False)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(ds.y)


def test_baseline_scores_chance_balanced_accuracy(ds: Dataset) -> None:
    res = ev.cross_validate(ds, BASELINE, "subject_default", **SMALL)
    assert res["summary"]["balanced_accuracy"]["mean"] == pytest.approx(0.5)
    assert res["summary"]["specificity"]["mean"] == 0.0


@pytest.fixture
def small_results(ds: Dataset) -> dict[str, Any]:
    return ev.run(ds, models=MODELS, **SMALL)


def test_run_picks_best_non_baseline_model(small_results: dict[str, Any]) -> None:
    assert set(small_results["cv"]) == set(MODELS)
    assert small_results["best_model"] in {"SVM", "KNN"}
    assert len(small_results["feature_importance"]) == 22


def test_main_writes_json_and_logs_to_mlflow(
    make_csv: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    real_run = ev.run
    monkeypatch.setattr(ev, "run", lambda ds, **kw: real_run(ds, models=MODELS, **kw))
    out = tmp_path / "evaluation.json"
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    args = ["--data", str(make_csv(10, 6)), "--out", str(out), "--tracking-uri", uri]
    args += ["--n-splits", "3", "--n-repeats", "1", "--inner-splits", "2"]
    assert ev.main(args) == 0

    assert set(json.loads(out.read_text())["cv"]) == set(MODELS)
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[ev.EXPERIMENT_NAME])
    # 3 CV protocols + 2 LOSO settings per model
    assert len(runs) == len(MODELS) * 5
    assert {"metrics.balanced_accuracy_mean", "params.protocol"} <= set(runs.columns)


def test_report_writes_every_section_and_figures(
    small_results: dict[str, Any], synthetic_csv: Path, tmp_path: Path
) -> None:
    paper = rp.run(load_dataset(synthetic_csv), n_repeats=2, n_folds=3, n_tuned_repeats=2)
    (tmp_path / "paper.json").write_text(json.dumps(paper))
    (tmp_path / "evaluation.json").write_text(json.dumps(small_results, default=float))
    docs = tmp_path / "docs"
    args = [
        "--paper",
        str(tmp_path / "paper.json"),
        "--evaluation",
        str(tmp_path / "evaluation.json"),
        # Don't pick up the real M3 result files from the repo.
        "--evaluation-reproducible",
        str(tmp_path / "none.json"),
        "--model-metadata",
        str(tmp_path / "none.json"),
    ]
    assert (
        report.main(
            [
                *args,
                "--docs-dir",
                str(docs),
                "--readme",
                str(tmp_path / "no-readme.md"),
                "--frontend-summary",
                str(tmp_path / "summary.json"),
            ]
        )
        == 0
    )

    md = (docs / "evaluation.md").read_text()
    for heading in [
        "## 1. Dataset",
        "## 2. Majority-class baseline",
        "## 3. Table 1",
        "## 4. Table 2",
        "## 5. Table 3",
        "## 6. Confusion matrices",
        "## 7. Top features",
        "## 8. Discussion",
    ]:
        assert heading in md
    assert "nan" not in md.lower().replace("n/a", "")
    for fig in ("protocol_comparison.png", "roc_subject_tuned.png"):
        assert (docs / "figures" / fig).stat().st_size > 1000
