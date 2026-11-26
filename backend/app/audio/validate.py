"""Reject recordings that cannot give meaningful voice features."""

from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

from app.audio.errors import AudioError
from app.audio.load import Audio

MIN_DURATION_S = 1.5
MAX_DURATION_S = 10.0
SILENCE_RMS_DBFS = -50.0  # quieter than this overall = effectively silent
CLIP_LEVEL = 0.999
MAX_CLIPPED_FRACTION = 0.001  # >0.1% of samples at full scale = distorted
MIN_VOICED_FRACTION = 0.6  # a sustained vowel is voiced almost throughout
F0_MIN_HZ = 75.0
F0_MAX_HZ = 600.0


def voiced_fraction(audio: Audio) -> float:
    """Share of Praat pitch frames in which a pitch was detected."""
    sound = parselmouth.Sound(audio.samples, sampling_frequency=audio.sample_rate)
    pitch = call(sound, "To Pitch", 0.0, F0_MIN_HZ, F0_MAX_HZ)
    frames = pitch.selected_array["frequency"]
    return float(np.mean(frames > 0)) if len(frames) else 0.0


def validate_audio(audio: Audio) -> None:
    """Raise `AudioError` if the recording is unusable; return None if it is fine.

    Checks, in order: duration, silence, clipping, and whether it is a sustained
    voiced sound (rather than speech, noise or whispering).
    """
    if audio.duration < MIN_DURATION_S:
        raise AudioError(
            "too_short",
            f"The recording is {audio.duration:.1f} s long. Please hold the 'aaah' for "
            f"at least {MIN_DURATION_S:g} seconds (3-5 seconds is ideal).",
        )
    if audio.duration > MAX_DURATION_S:
        raise AudioError(
            "too_long",
            f"The recording is {audio.duration:.1f} s long. Please keep it under "
            f"{MAX_DURATION_S:g} seconds.",
        )
    rms = float(np.sqrt(np.mean(audio.samples**2)))
    if rms == 0 or 20 * np.log10(rms) < SILENCE_RMS_DBFS:
        raise AudioError("silent", "The recording is silent or too quiet. Check your microphone.")
    clipped = float(np.mean(np.abs(audio.samples) >= CLIP_LEVEL))
    if clipped > MAX_CLIPPED_FRACTION:
        raise AudioError(
            "clipped",
            "The recording is distorted because it is too loud. Move a little further "
            "from the microphone and try again.",
        )
    if voiced_fraction(audio) < MIN_VOICED_FRACTION:
        raise AudioError(
            "not_voiced",
            "No steady voice was detected. Please record a single, sustained 'aaah' in a "
            "quiet room, without speaking words.",
        )
