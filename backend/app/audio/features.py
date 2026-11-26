"""Extract the audio-reproducible UCI voice features with Praat (via Parselmouth).

The UCI dataset's features were computed with MDVP (Kay Pentax) and custom
nonlinear-dynamics code. Praat implements the same *definitions* for the 15
features below, but not identical algorithms, so values are comparable in
range rather than identical. The other 7 UCI features are not reproduced:

- NHR: Praat has no equivalent of MDVP's noise-to-harmonics ratio.
- RPDE, DFA, D2, PPE, spread1, spread2: computed by Little et al.'s own code,
  whose exact settings are not published.

Unit trap: despite its name, the dataset's ``MDVP:Jitter(%)`` column holds a
fraction (values ~0.002-0.016), so Praat's fractional local jitter is used
as-is and NOT multiplied by 100.
"""

from __future__ import annotations

import math

import parselmouth
from parselmouth.praat import call

from app.audio.errors import AudioError
from app.audio.load import Audio
from app.audio.validate import F0_MAX_HZ, F0_MIN_HZ

EDGE_TRIM_S = 0.25  # skip voice onset/offset, where the voice is not yet steady

REPRODUCIBLE_FEATURES: tuple[str, ...] = (
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
    "HNR",
)

# Praat's standard settings for period-based measures (as in its Voice Report).
_PERIOD_FLOOR, _PERIOD_CEILING, _MAX_PERIOD_FACTOR, _MAX_AMP_FACTOR = 0.0001, 0.02, 1.3, 1.6

# Dataset column -> Praat command on the PointProcess (jitter).
_JITTER = {
    "MDVP:Jitter(%)": "Get jitter (local)",
    "MDVP:Jitter(Abs)": "Get jitter (local, absolute)",
    "MDVP:RAP": "Get jitter (rap)",
    "MDVP:PPQ": "Get jitter (ppq5)",
    "Jitter:DDP": "Get jitter (ddp)",
}
# Dataset column -> Praat command on (Sound, PointProcess) (shimmer).
_SHIMMER = {
    "MDVP:Shimmer": "Get shimmer (local)",
    "MDVP:Shimmer(dB)": "Get shimmer (local_dB)",
    "Shimmer:APQ3": "Get shimmer (apq3)",
    "Shimmer:APQ5": "Get shimmer (apq5)",
    "MDVP:APQ": "Get shimmer (apq11)",  # MDVP's APQ uses an 11-period window
    "Shimmer:DDA": "Get shimmer (dda)",
}


def extract_features(audio: Audio) -> dict[str, float]:
    """Return the 15 reproducible features, keyed by their UCI column names.

    Call `validate_audio` first. The first and last `EDGE_TRIM_S` seconds are
    skipped.

    Raises:
        AudioError: code ``feature_failed`` if Praat cannot measure a feature
            (e.g. too few voice periods).
    """
    sound = parselmouth.Sound(audio.samples, sampling_frequency=audio.sample_rate)
    if sound.duration > 4 * EDGE_TRIM_S:
        sound = sound.extract_part(EDGE_TRIM_S, sound.duration - EDGE_TRIM_S)

    pitch = call(sound, "To Pitch", 0.0, F0_MIN_HZ, F0_MAX_HZ)
    points = call(sound, "To PointProcess (periodic, cc)", F0_MIN_HZ, F0_MAX_HZ)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, F0_MIN_HZ, 0.1, 1.0)

    features = {
        "MDVP:Fo(Hz)": call(pitch, "Get mean", 0, 0, "Hertz"),
        "MDVP:Fhi(Hz)": call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"),
        "MDVP:Flo(Hz)": call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"),
    }
    for column, command in _JITTER.items():
        features[column] = call(
            points, command, 0, 0, _PERIOD_FLOOR, _PERIOD_CEILING, _MAX_PERIOD_FACTOR
        )
    for column, command in _SHIMMER.items():
        features[column] = call(
            [sound, points],
            command,
            0,
            0,
            _PERIOD_FLOOR,
            _PERIOD_CEILING,
            _MAX_PERIOD_FACTOR,
            _MAX_AMP_FACTOR,
        )
    features["HNR"] = call(harmonicity, "Get mean", 0, 0)

    result = {name: float(features[name]) for name in REPRODUCIBLE_FEATURES}
    failed = [name for name, value in result.items() if not math.isfinite(value)]
    if failed:
        raise AudioError(
            "feature_failed",
            "Could not measure the voice reliably (" + ", ".join(failed) + "). "
            "Please record a steady 'aaah' of 3-5 seconds.",
        )
    return result
