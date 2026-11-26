"""Place each feature value relative to the training data's two groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureRange:
    """Where one value sits in the training distribution."""

    feature: str
    value: float
    parkinsons: dict[str, float]  # p5/p25/p50/p75/p95 of the Parkinson's group
    healthy: dict[str, float]  # same for the healthy group
    within_training_range: bool  # inside the 5th-95th percentile of all recordings
    closer_to: str  # "parkinsons", "healthy" or "neither" (by distance to group medians)


def compare(features: dict[str, float], metadata: dict[str, Any]) -> list[FeatureRange]:
    """Compare every feature with the per-class percentiles stored at training time."""
    out = []
    for name in metadata["features"]:
        dist = metadata["feature_distributions"][name]
        value = features[name]
        pd_median, h_median = dist["parkinsons"]["p50"], dist["healthy"]["p50"]
        d_pd, d_h = abs(value - pd_median), abs(value - h_median)
        if abs(d_pd - d_h) <= 0.05 * abs(pd_median - h_median):
            closer = "neither"  # roughly halfway between the two medians
        else:
            closer = "parkinsons" if d_pd < d_h else "healthy"
        out.append(
            FeatureRange(
                feature=name,
                value=value,
                parkinsons=dist["parkinsons"],
                healthy=dist["healthy"],
                within_training_range=dist["all"]["p5"] <= value <= dist["all"]["p95"],
                closer_to=closer,
            )
        )
    return out
