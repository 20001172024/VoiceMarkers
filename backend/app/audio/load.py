"""Decode audio bytes into a mono waveform at the analysis sample rate."""

from __future__ import annotations

import io
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

from app.audio.errors import AudioError

# The UCI recordings were made at 44.1 kHz; resampling to the same rate keeps
# Praat's period-level measures (jitter, shimmer) on a comparable time grid.
TARGET_SR = 44_100


@dataclass(frozen=True)
class Audio:
    """A mono waveform in [-1, 1] and its sample rate."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return len(self.samples) / self.sample_rate


def load_audio(data: bytes, target_sr: int = TARGET_SR) -> Audio:
    """Decode WAV/FLAC/OGG bytes, mix to mono and resample to `target_sr`.

    Formats are those libsndfile reads. Browser WebM/Opus needs ffmpeg and is
    handled in the API layer (M4).

    Raises:
        AudioError: code ``decode_failed`` if the bytes are not readable audio.
    """
    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
        raise AudioError(
            "decode_failed", "Could not read this file as audio. Please upload a WAV file."
        ) from exc
    mono = samples.mean(axis=1)
    if sr != target_sr and len(mono):
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    return Audio(samples=mono, sample_rate=target_sr)
