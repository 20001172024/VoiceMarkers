"""Decode audio bytes into a mono waveform at the analysis sample rate."""

from __future__ import annotations

import io
import shutil
import subprocess
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


FFMPEG_TIMEOUT_S = 20


def _ffmpeg_to_wav(data: bytes) -> bytes:
    """Convert any ffmpeg-readable audio (e.g. browser WebM/Opus, MP4/AAC) to WAV.

    Bytes go in through stdin and come out through stdout, so the recording is
    never written to disk.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioError(
            "unsupported_format",
            "This audio format is not supported on this server. Please upload a WAV file.",
        )
    try:
        result = subprocess.run(  # noqa: S603 (fixed argument list, no shell)
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
            input=data,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_S,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AudioError("decode_failed", _DECODE_MESSAGE) from exc
    return result.stdout


_DECODE_MESSAGE = "Could not read this file as audio. Please upload a WAV file."


def load_audio(data: bytes, target_sr: int = TARGET_SR) -> Audio:
    """Decode audio bytes, mix to mono and resample to `target_sr`.

    WAV, FLAC and OGG are read directly (libsndfile). Anything else, such as the
    WebM or MP4 that browsers record, is converted in memory by ffmpeg.

    Raises:
        AudioError: code ``decode_failed`` if the bytes are not readable audio,
            or ``unsupported_format`` if conversion is needed but ffmpeg is missing.
    """
    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError):
        wav = _ffmpeg_to_wav(data)
        try:
            samples, sr = sf.read(io.BytesIO(wav), dtype="float64", always_2d=True)
        except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
            raise AudioError("decode_failed", _DECODE_MESSAGE) from exc
    mono = samples.mean(axis=1)
    if sr != target_sr and len(mono):
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    return Audio(samples=mono, sample_rate=target_sr)
