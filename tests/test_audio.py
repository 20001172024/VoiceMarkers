"""Tests for backend/app/audio on SYNTHETIC recordings with known properties."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.audio.errors import AudioError
from app.audio.features import REPRODUCIBLE_FEATURES, extract_features
from app.audio.load import TARGET_SR, Audio, load_audio
from app.audio.validate import validate_audio, voiced_fraction
from data.make_samples import SAMPLES, SAMPLES_DIR, VoiceSpec, synth_vowel, to_wav_bytes
from research.dataset import FEATURE_COLUMNS


def vowel(**kwargs: float) -> Audio:
    return Audio(synth_vowel(VoiceSpec(**kwargs)), TARGET_SR)


# ------------------------------------------------------------------ loading


def test_load_wav_round_trip() -> None:
    samples = synth_vowel(VoiceSpec(duration=2.0))
    audio = load_audio(to_wav_bytes(samples))
    assert audio.sample_rate == TARGET_SR
    assert audio.duration == pytest.approx(2.0, abs=1e-3)


def test_load_resamples_and_mixes_to_mono() -> None:
    sr = 16_000
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    buffer = io.BytesIO()
    sf.write(buffer, np.column_stack([tone, tone]), sr, format="WAV")
    audio = load_audio(buffer.getvalue())
    assert audio.samples.ndim == 1
    assert audio.sample_rate == TARGET_SR
    assert audio.duration == pytest.approx(1.0, abs=1e-3)


def test_load_rejects_non_audio_bytes() -> None:
    with pytest.raises(AudioError) as exc:
        load_audio(b"definitely not a wav file")
    assert exc.value.code == "decode_failed"


# --------------------------------------------------------------- validation


def test_valid_vowel_passes() -> None:
    validate_audio(vowel())


@pytest.mark.parametrize(
    ("audio", "code"),
    [
        (Audio(np.zeros(TARGET_SR * 3), TARGET_SR), "silent"),
        (Audio(1e-4 * synth_vowel(VoiceSpec()), TARGET_SR), "silent"),
        (Audio(synth_vowel(VoiceSpec(duration=1.0)), TARGET_SR), "too_short"),
        (Audio(synth_vowel(VoiceSpec(duration=11.0)), TARGET_SR), "too_long"),
        (Audio(np.clip(5 * synth_vowel(VoiceSpec()), -1, 1), TARGET_SR), "clipped"),
        (
            Audio(0.3 * np.random.default_rng(0).standard_normal(TARGET_SR * 3), TARGET_SR),
            "not_voiced",
        ),
    ],
    ids=["digital-silence", "too-quiet", "too-short", "too-long", "clipped", "noise-only"],
)
def test_validation_rejects_bad_recordings(audio: Audio, code: str) -> None:
    with pytest.raises(AudioError) as exc:
        validate_audio(audio)
    assert exc.value.code == code
    assert exc.value.message  # always a user-facing explanation


def test_vowel_buried_in_noise_is_not_voiced() -> None:
    loud = vowel(noise=0.6)
    noisy = Audio(0.9 * loud.samples / np.abs(loud.samples).max(), TARGET_SR)  # avoid clipping
    assert voiced_fraction(noisy) < 0.6
    with pytest.raises(AudioError, match="No steady voice"):
        validate_audio(noisy)


# --------------------------------------------------------------- extraction


def test_extracts_exactly_the_reproducible_features() -> None:
    features = extract_features(vowel())
    assert tuple(features) == REPRODUCIBLE_FEATURES
    assert set(REPRODUCIBLE_FEATURES) <= set(FEATURE_COLUMNS)  # all are real UCI columns
    assert all(np.isfinite(v) for v in features.values())


def test_pitch_matches_the_synthesised_f0() -> None:
    for f0 in (110.0, 200.0):
        features = extract_features(vowel(f0=f0))
        assert features["MDVP:Fo(Hz)"] == pytest.approx(f0, rel=0.02)
        assert features["MDVP:Flo(Hz)"] <= features["MDVP:Fo(Hz)"] <= features["MDVP:Fhi(Hz)"]


def test_jitter_is_a_fraction_not_a_percentage() -> None:
    """The dataset's 'MDVP:Jitter(%)' column holds fractions (~0.002-0.016)."""
    jitter = extract_features(vowel(jitter=0.005))["MDVP:Jitter(%)"]
    assert 0.001 < jitter < 0.02


def test_measured_jitter_increases_with_synthesised_jitter() -> None:
    measured = [extract_features(vowel(jitter=j))["MDVP:Jitter(%)"] for j in (0.002, 0.006, 0.012)]
    assert measured == sorted(measured)
    assert measured[-1] > 3 * measured[0]


def test_measured_shimmer_increases_with_synthesised_shimmer() -> None:
    measured = [extract_features(vowel(shimmer=s))["MDVP:Shimmer"] for s in (0.01, 0.04, 0.08)]
    assert measured == sorted(measured)


def test_hnr_drops_when_noise_is_added() -> None:
    clean = extract_features(vowel(noise=0.001))["HNR"]
    noisy = extract_features(vowel(noise=0.03))["HNR"]
    assert noisy < clean - 5


def test_ddp_is_three_times_rap() -> None:
    """By definition Jitter:DDP = 3 x RAP and Shimmer:DDA = 3 x APQ3 (true in the dataset too)."""
    f = extract_features(vowel())
    assert f["Jitter:DDP"] == pytest.approx(3 * f["MDVP:RAP"], rel=1e-3)
    assert f["Shimmer:DDA"] == pytest.approx(3 * f["Shimmer:APQ3"], rel=1e-3)


# ------------------------------------------------------------ committed WAVs


@pytest.mark.parametrize("name", SAMPLES)
def test_committed_samples_match_the_generator(name: str) -> None:
    path = SAMPLES_DIR / f"{name}.wav"
    assert path.read_bytes() == to_wav_bytes(synth_vowel(SAMPLES[name]))


@pytest.mark.parametrize("name", SAMPLES)
def test_committed_samples_pass_the_full_pipeline(name: str) -> None:
    audio = load_audio((SAMPLES_DIR / f"{name}.wav").read_bytes())
    validate_audio(audio)
    assert len(extract_features(audio)) == len(REPRODUCIBLE_FEATURES)
