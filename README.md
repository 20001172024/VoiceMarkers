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

## Status

🚧 Work in progress. Built milestone by milestone (see [`CLAUDE.md`](CLAUDE.md)):

- [x] **M0** — repo skeleton, `/health` API, dataset download, lint + CI
- [x] **M1** — reproduce the paper's row-level results ([report](research/results/paper_reproduction.md))
- [ ] **M2** — subject-wise re-evaluation and evaluation report
- [ ] **M3** — Parselmouth feature extraction and the deployed model
- [ ] **M4** — prediction API
- [ ] **M5** — frontend
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
cd backend && uvicorn app.main:app --reload   # http://localhost:8000/health
```

```bash
cd frontend && npm install && npm run dev     # http://localhost:5173
```

Or run both with Docker: `cp .env.example .env && docker compose up --build`.

Lint hooks: `pre-commit install` (ruff, black, prettier).

## Dataset

UCI Parkinson's dataset — 195 sustained-vowel recordings with 22 precomputed voice
features, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`data/download.py` fetches it from UCI and checks a pinned SHA-256 hash; the CSV is
never committed or edited.

> Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A. E. and Moroz, I. M.
> (2007). Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice
> Disorder Detection. *BioMedical Engineering OnLine* 6:23.
> https://archive.ics.uci.edu/dataset/174/parkinsons
