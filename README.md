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

## What this demo can and cannot do

The deployed model is a **demonstration of the method and its limits, not a working
detector**. Evaluated the way that matters for a new person (subject-wise, on the 15
features that can be computed from a recording), it reaches roughly 64% balanced
accuracy, where 50% is chance, with large variation between folds. On recordings made
with a different microphone or in a different room it is expected to do worse. The
[evaluation report](docs/evaluation.md) has the full numbers and explains how they
relate to the paper's 87.18%.

## Status

🚧 Work in progress. Built milestone by milestone (see [`CLAUDE.md`](CLAUDE.md)):

- [x] **M0** — repo skeleton, `/health` API, dataset download, lint + CI
- [x] **M1** — reproduce the paper's row-level results ([report](research/results/paper_reproduction.md))
- [x] **M2** — subject-wise re-evaluation and evaluation report ([report](docs/evaluation.md))
- [x] **M3** — Parselmouth feature extraction and the deployed model ([validation](research/results/feature_validation.md))
- [x] **M4** — prediction API (see [API](#api))
- [x] **M5** — frontend (record, upload or try a sample; results with explanations; About the model)
- [ ] **M6** — deployment
- [ ] **M7** — polish

## Quick start

Requires Python 3.11 and Node 22.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e "backend[dev]" -r research/requirements.txt
python -m data.download          # fetches and verifies the UCI dataset
pytest                           # backend + research + script tests
python -m research.reproduce_paper   # row-level paper reproduction (~1 min)
python -m research.evaluate          # subject-wise nested CV, logged to MLflow
python -m research.evaluate --features reproducible   # same, on the 15 audio features
python -m research.feature_validation # check extracted features against the dataset
python -m research.train_final        # train + save backend/models/model.joblib
python -m research.report            # regenerate docs/evaluation.md
cd backend && uvicorn app.main:app --reload   # http://localhost:8000/health
```

```bash
cd frontend && npm install && npm run dev     # http://localhost:5173 (needs the backend running)
cd frontend && npm test                       # Vitest + Testing Library
```

Or run both with Docker: `cp .env.example .env && docker compose up --build`.

Lint hooks: `pre-commit install` (ruff, black, prettier).

## API

Interactive docs at `http://localhost:8000/docs` when the backend is running.

| Endpoint | Purpose |
|---|---|
| `POST /predict/audio` | Raw audio as the request body (WAV, FLAC, OGG, or browser WebM/MP4 via ffmpeg) → features, probability, label, SHAP contributions, feature ranges |
| `POST /predict/features` | `{"features": {...}}` with the 15 feature values → same response |
| `GET /model/info` | Model type, version, training date, features, nested-CV metrics, limitations |
| `GET /health` | Liveness and whether the model loaded |

```bash
curl -X POST --data-binary @data/samples/synthetic_steady.wav \
  -H "Content-Type: audio/wav" http://localhost:8000/predict/audio
```

Every response, including errors, carries the disclaimer. Audio is read from the
request body into memory (not a multipart upload, which the server would spool to a
temporary file above 1 MB) and is never stored or logged. Prediction endpoints are
limited to 10 requests per minute per client.

## Dataset

UCI Parkinson's dataset — 195 sustained-vowel recordings with 22 precomputed voice
features, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`data/download.py` fetches it from UCI and checks a pinned SHA-256 hash; the CSV is
never committed or edited.

> Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A. E. and Moroz, I. M.
> (2007). Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice
> Disorder Detection. *BioMedical Engineering OnLine* 6:23.
> https://archive.ics.uci.edu/dataset/174/parkinsons
