# VoiceMarkers — Project Brief for Claude Code

## 1. What this project is

VoiceMarkers is a web app built around my published research on **voice-based Parkinson's disease detection using machine learning** (ICOIICS 2025). The original work trained and compared ML classifiers on the UCI Parkinson's voice dataset; the best model (SVM) reached 87.18% accuracy.

This project turns that research into a deployed, full-stack application:

1. A user records a sustained vowel ("aaah") in the browser, or uploads a WAV file.
2. The backend extracts voice features (jitter, shimmer, harmonics-to-noise ratio, pitch statistics and others) from the audio.
3. A trained classifier returns a prediction with a probability, and the UI shows which features drove it and how they compare to the training data.

It also **re-evaluates the original model more rigorously** (subject-wise splits, metrics beyond accuracy) and reports those results honestly alongside the paper's.

**Positioning:** a research and education demo of my paper. **Not a diagnostic or medical tool.** A clear disclaimer must appear in the UI (before and after a prediction), in every API response, and in the README.

**Owner:** Swarnima Arora (B.Tech CSE-AI, IGDTUW). This is a portfolio project for SWE/SDE roles. I must be able to explain every design decision in interviews.

---

## 2. How I want you (Claude Code) to work

- **Work one milestone at a time** (Section 8). Don't jump ahead or generate the whole project at once.
- **Before writing code for a milestone, propose a short plan** (files, approach, trade-offs) and wait for my OK.
- **Explain non-obvious choices** in a few sentences (e.g. why a subject-wise split, why a given feature can or can't be reproduced from raw audio).
- **Never invent results.** Every metric must come from actually running the code. If the rigorous re-evaluation gives a lower number than the paper, report it as it is and explain why.
- **Ask before adding any dependency** not listed in Section 4.
- **Write tests for every module** (pytest for backend, basic tests for frontend).
- **Keep commits small and meaningful** (`feat(features): extract jitter and shimmer with Parselmouth`).
- Type hints and docstrings in Python. Prefer simple, readable code.
- Keep secrets in `.env` (never commit it); provide `.env.example`.

---

## 3. Data, scientific honesty and privacy

**Dataset:** UCI Parkinson's dataset (Little et al.) — 195 sustained-vowel recordings from 31 people (23 with Parkinson's, 8 healthy), 22 precomputed voice features per recording. Licensed CC BY 4.0; cite it in the README. A download script fetches it; the CSV is not hand-edited.

**My original code:** if present, my original notebook lives in `research/original/`. Read it first and reproduce its results before changing anything, so the old and new numbers can be compared fairly.

**Three issues you must handle explicitly:**

1. **Subject leakage.** Each person has about 6 recordings. A random row-level split puts the same person in both train and test sets, which inflates accuracy. The `name` column (e.g. `phon_R01_S01_1`) encodes the subject ID. All new evaluation must use **subject-wise splits** (`GroupKFold` / leave-one-subject-out). Report both the row-level result (to match the paper) and the subject-wise result, side by side.
2. **Class imbalance.** About 75% of recordings are from people with Parkinson's, so always predicting "Parkinson's" scores ~75% accuracy. Report **balanced accuracy, precision, recall, F1, ROC-AUC and a confusion matrix**, and the majority-class baseline for comparison.
3. **Features from raw audio.** The dataset ships precomputed features; the web app has raw audio. Use **Parselmouth** (Python interface to Praat) to extract features from uploaded audio. Some UCI features (e.g. RPDE, D2, PPE, spread1/2) are hard to reproduce exactly. Approach:
   - Identify which features can be reliably reproduced from audio.
   - Train the **deployed model only on reproducible features**, and report how its performance compares to the full-feature model.
   - Validate the extraction: check that features extracted from sample recordings fall in plausible ranges compared with the dataset's distributions.
   - Document clearly that microphone, environment and recording conditions differ from the original dataset, so predictions on new recordings are **not reliable** — this is a limitation, stated in the README and UI.

**Privacy:** uploaded or recorded audio is processed in memory and **never stored**, logged or sent to third parties. State this in the UI.

---

## 4. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| ML | Python 3.11, scikit-learn, pandas, NumPy | SVM, Random Forest, Logistic Regression, KNN, Gradient Boosting compared |
| Explainability | SHAP or permutation importance | Per-prediction feature contributions shown in UI |
| Audio features | Parselmouth (Praat), librosa for loading/resampling | Optionally `nolds` for DFA |
| Experiment tracking | MLflow (local) | Logs every evaluation run |
| Backend | FastAPI, Pydantic, Uvicorn | Model loaded once at startup |
| Frontend | React + Vite + TypeScript, Tailwind CSS, Recharts | In-browser recording via MediaRecorder API |
| Testing | pytest, httpx test client; Vitest for frontend | |
| Containers | Docker, docker-compose | |
| CI | GitHub Actions | Lint, tests, build, deploy |
| Hosting | Backend: Render or Hugging Face Spaces (Docker). Frontend: Vercel or Netlify | Free tiers |

---

## 5. Features

**ML / research**
- Reproduce the paper's original results (row-level split)
- Subject-wise re-evaluation with all metrics from Section 3, plus majority-class baseline
- Model comparison table (SVM, RF, LR, KNN, Gradient Boosting) with hyperparameter tuning inside the cross-validation (nested CV or a proper train/validation split — no tuning on test folds)
- Feature-scaling and pipeline handled with `sklearn.Pipeline` to avoid data leakage
- Reduced-feature model trained only on audio-reproducible features, with comparison to the full model
- Saved model artifact with version and training metadata
- An evaluation report (`docs/evaluation.md`) generated from real runs

**Backend API**
- `POST /predict/audio` — WAV/WebM upload → extracted features, probability, prediction label, top feature contributions, feature-range comparison, disclaimer
- `POST /predict/features` — JSON of feature values → same response (for testing and for users who have precomputed features)
- `GET /model/info` — model type, version, training date, feature list, evaluation metrics
- `GET /health`
- Input validation: file type, size limit, minimum and maximum duration, sample rate handling, clear error messages for silent or too-noisy audio
- Basic rate limiting

**Frontend**
- Landing page: what the project is, link to the paper, disclaimer
- Record panel: microphone permission handling, countdown, record a 3–5 second "aaah", live waveform, playback before submitting
- Upload option for WAV files, plus "try a sample recording" button
- Results page: probability gauge, prediction label (worded carefully, e.g. "voice features more similar to the Parkinson's group in the training data"), feature contribution chart, where each feature falls relative to both groups in the dataset, and the disclaimer
- "About the model" page: evaluation results, the row-level vs subject-wise comparison, limitations
- Mobile-friendly layout; accessible (labels, keyboard navigation, colour contrast)

---

## 6. Repository structure

```
voicemarkers/
├── CLAUDE.md
├── README.md
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml
├── research/
│   ├── original/              # my original notebook (if available)
│   ├── reproduce_paper.py     # row-level reproduction of the paper's results
│   ├── evaluate.py            # subject-wise evaluation, all metrics, model comparison
│   ├── feature_validation.py  # checks audio-extracted features vs dataset ranges
│   └── train_final.py         # trains and saves the deployed model
├── data/
│   ├── download.py            # fetches the UCI dataset
│   └── samples/               # a few sample WAV files that are licensed for reuse
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/               # predict, model_info, health routers
│   │   ├── audio/             # loading, validation, feature extraction (Parselmouth)
│   │   ├── ml/                # model loading, prediction, explanations
│   │   └── config.py
│   ├── models/                # saved model artifact + metadata JSON
│   └── tests/
├── frontend/
│   └── src/ (components, pages, api client, recorder hook)
└── docs/
    ├── evaluation.md          # generated results
    ├── limitations.md
    └── architecture.md
```

---

## 7. Evaluation report contents (`docs/evaluation.md`)

- Dataset summary: recordings, subjects, class balance
- Majority-class baseline
- Table 1: paper reproduction (row-level split) — accuracy and other metrics
- Table 2: subject-wise evaluation for every model — balanced accuracy, precision, recall, F1, ROC-AUC, with mean ± std across folds
- Table 3: full-feature vs audio-reproducible-feature model
- Confusion matrices and ROC curves
- Top features by importance
- A short honest discussion of why row-level and subject-wise results differ, and what that means for the paper's 87.18% figure

All numbers generated by scripts and logged in MLflow — none typed by hand.

---

## 8. Milestones (do these in order)

**M0 — Setup.** Repo skeleton, Python environment, docker-compose, linting (ruff, black, prettier), CI skeleton, `/health` endpoint, dataset download script.

**M1 — Reproduce the paper.** Read my original notebook if present. Reproduce the row-level results, including the SVM's 87.18% if possible. If the number doesn't match, find out why and tell me before moving on.

**M2 — Rigorous re-evaluation.** Subject-wise CV, all metrics, majority baseline, model comparison with tuning inside CV, MLflow logging, generate `docs/evaluation.md`.

**M3 — Audio feature extraction.** Parselmouth pipeline for the reproducible features, validation script against dataset ranges, unit tests with sample audio (including silent, very short and noisy edge cases). Train and save the reduced-feature deployed model, with its evaluation added to the report.

**M4 — Backend API.** Endpoints from Section 5, input validation, explanations, rate limiting, disclaimer in every response, tests with the FastAPI test client.

**M5 — Frontend.** All pages and components from Section 5, in-browser recording, results visualizations, disclaimer and privacy notice, responsive and accessible.

**M6 — Deployment.** Production Dockerfile for the backend, deploy backend and frontend, environment configuration, CORS, GitHub Actions running lint + tests + build on every push and deploying on merge to main.

**M7 — Polish.** README with architecture diagram, live demo link, screenshots or GIF, evaluation summary table, link to the ICOIICS paper, dataset citation, limitations section and disclaimer.

---

## 9. Definition of done

- Live demo URL where someone can record or upload audio, or use a sample, and see a result in one click
- Paper results reproduced, and subject-wise results reported honestly next to them
- Deployed model uses only features reproducible from audio, with its performance documented
- All tests pass in CI; deployment is automatic on merge to main
- Audio is never stored; privacy notice and medical disclaimer visible everywhere a prediction appears
- README is complete, with citation of the dataset and my paper
- I can explain the subject-leakage issue, every metric, the feature extraction and every module without looking at the code
