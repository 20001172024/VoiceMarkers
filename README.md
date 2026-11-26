# VoiceMarkers

Voice-based Parkinson's disease detection with machine learning — a full-stack
research and education demo of my ICOIICS 2025 paper.

Record a sustained "aaah" (or upload a WAV), the backend extracts voice features
(jitter, shimmer, HNR, pitch statistics) with Parselmouth, and a classifier trained
on the UCI Parkinson's dataset returns a probability with an explanation of which
features drove it. The project also re-evaluates the paper's model with
subject-wise cross-validation and reports both sets of numbers side by side.

> **Disclaimer:** VoiceMarkers is a research and education demo. It is **not** a
> medical device and does not diagnose, screen for or rule out Parkinson's disease
> or any other condition. Do not use it to make health decisions.

🚧 Work in progress — see the milestones in [`CLAUDE.md`](CLAUDE.md).
