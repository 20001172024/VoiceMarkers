"""Generate SYNTHETIC sustained-vowel recordings for tests and the demo.

These are not human voices. Each is a glottal pulse train with controlled
period jitter and amplitude shimmer, shaped by four /a/ formants and mixed with
white noise. Because the perturbations are set directly, they give known inputs
for testing feature extraction. They are released with the repo and need no
licence beyond it.

Usage (from the repo root):
    python -m data.make_samples          # writes data/samples/*.wav
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLE_RATE = 44_100
# /a/ formant frequencies and bandwidths (Hz), typical adult values.
FORMANTS_A = ((700, 130), (1220, 70), (2600, 160), (3300, 250))


@dataclass(frozen=True)
class VoiceSpec:
    """Parameters of a synthetic sustained vowel."""

    f0: float = 150.0  # mean pitch (Hz)
    jitter: float = 0.005  # std of period-to-period variation, as a fraction of the period
    shimmer: float = 0.03  # std of pulse amplitude variation, as a fraction
    noise: float = 0.002  # white-noise amplitude relative to a peak of 0.8
    duration: float = 3.0
    seed: int = 0


def synth_vowel(spec: VoiceSpec, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a mono waveform (peak 0.8) for the given voice parameters."""
    rng = np.random.default_rng(spec.seed)
    n = int(spec.duration * sample_rate)
    pulses = np.zeros(n)
    t = 0.0
    while True:
        t += (1 / spec.f0) * (1 + spec.jitter * rng.standard_normal())
        i = int(t * sample_rate)
        if i >= n:
            break
        pulses[i] = 1 + spec.shimmer * rng.standard_normal()

    sound = parselmouth.Sound(pulses, sampling_frequency=sample_rate)
    for freq, bandwidth in FORMANTS_A:
        call(sound, "Filter with one formant (in-place)", freq, bandwidth)
    voiced = sound.values[0]
    voiced = 0.8 * voiced / np.abs(voiced).max()
    return voiced + spec.noise * rng.standard_normal(n)


def to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode a waveform as 16-bit PCM WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# The committed demo samples. "steady" has low perturbation; "rough" has the
# higher jitter/shimmer and noise typical of a dysphonic voice.
SAMPLES: dict[str, VoiceSpec] = {
    "synthetic_steady": VoiceSpec(f0=140, jitter=0.002, shimmer=0.02, noise=0.001, seed=1),
    "synthetic_rough": VoiceSpec(f0=160, jitter=0.007, shimmer=0.045, noise=0.008, seed=2),
}


def main() -> int:
    """Write every sample in `SAMPLES` to data/samples/."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in SAMPLES.items():
        path = SAMPLES_DIR / f"{name}.wav"
        path.write_bytes(to_wav_bytes(synth_vowel(spec)))
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
