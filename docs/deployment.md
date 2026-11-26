# Deployment

| Part | Host | URL |
|---|---|---|
| Backend (FastAPI, Docker) | Hugging Face Spaces (free CPU) | `https://arora2006-voicemarkers-api.hf.space` |
| Frontend (static Vite build) | Vercel (free) | `https://<your-project>.vercel.app` |

Why these hosts: the backend loads scikit-learn, SHAP (with numba) and Praat, about 500 MB
in memory. The free Hugging Face CPU Space has 16 GB, while free tiers with 512 MB (such
as Render's) would run out of memory and also sleep when idle. The frontend is plain
static files, which any static host serves well.

## How deploys happen

Everything runs in `.github/workflows/ci.yml`:

1. **On every push and pull request:** Python lint and tests, frontend checks, and a
   **Docker smoke test**. The smoke test builds the production image, starts it, checks
   `/health`, sends the sample WAV and a WebM file to `/predict/audio`, and checks that
   the container runs as a non-root user.
2. **On a push to `main` (after all of the above pass):**
   - **deploy-backend:** `deploy/deploy_space.py`
     1. sets the Space variables;
     2. uploads the Dockerfile, the `app/` package and the model to the Space in one commit;
     3. waits until `/health` reports the new commit as its `build`. The Space rebuilds its
        image after every upload, which takes several minutes.
   - **deploy-frontend:**
     1. builds the site with `VITE_API_BASE_URL` pointing at the Space;
     2. deploys it to Vercel's production URL with the Vercel CLI.

If a secret is missing, its deploy job prints a notice and skips without failing, so CI
stays green before deployment is set up.

## One-time setup

### 1. Hugging Face Space (backend)

1. Create the Space at <https://huggingface.co/new-space>:
   - owner `arora2006`, name `voicemarkers-api`;
   - SDK **Docker**, blank template;
   - hardware **CPU basic (free)**, visibility **public**.
2. Create a token at <https://huggingface.co/settings/tokens>:
   - choose a **fine-grained** token;
   - give it **write** access to this Space only.
3. In GitHub, go to **Settings → Secrets and variables → Actions → New repository
   secret** and add `HF_TOKEN` with that token.

You don't need to edit anything in the Space by hand: each deploy replaces its files
and sets these variables.

| Variable | Value | Purpose |
|---|---|---|
| `BUILD_SHA` | the git commit | shown by `/health`, so the deploy job can tell the new build is live |
| `FORWARDED_HOPS` | `1` | the rate limiter takes the client IP from the entry the Space's proxy appends to `X-Forwarded-For` |
| `CORS_ORIGINS` | from the `FRONTEND_URL` repository variable | the only website allowed to call the API from a browser |

### 2. Vercel (frontend)

1. Install the CLI with `npm i -g vercel` and log in with `vercel login`. Then, inside
   `frontend/`, run `vercel link` and create a new project. That writes
   `frontend/.vercel/project.json`; don't commit it.
   - Keep the project's **Root Directory** empty. The workflow deploys from inside
     `frontend/`.
   - **Don't connect the Git repository** in Vercel. GitHub Actions does the deploys,
     and connecting Git would also make Vercel deploy every push itself, so each push
     would be deployed twice.
2. Create a token at <https://vercel.com/account/tokens>.
3. Add three repository secrets:
   - `VERCEL_TOKEN`: the token;
   - `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID`: the `orgId` and `projectId` from
     `frontend/.vercel/project.json`.
4. Add a repository **variable** (not a secret), `FRONTEND_URL`, set to the production
   URL, for example `https://voicemarkers.vercel.app`. The next backend deploy allows
   that origin through CORS.

Optional repository variables, if the targets change: `HF_SPACE`
(default `arora2006/voicemarkers-api`) and `API_BASE_URL`
(default `https://arora2006-voicemarkers-api.hf.space`).

### 3. First deploy

Merge to `main` (or re-run the latest `main` workflow). Then check that:

- `https://arora2006-voicemarkers-api.hf.space/health` shows `"model_loaded": true` and the
  commit as `build`;
- the Vercel URL loads, and "Try a sample" returns a result;
- recording works in a real browser with a real microphone. CI can't test this.

## Configuration reference (backend environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `7860` in the image | Port uvicorn listens on |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed browser origins |
| `MAX_UPLOAD_MB` | `5` | Upload size limit |
| `RATE_LIMIT_PER_MINUTE` | `10` | Prediction requests per client per minute (0 = off) |
| `FORWARDED_HOPS` | `0` | Trusted proxies in front of the app (1 on Spaces) |
| `MODEL_PATH` | `backend/models/` | Directory with `model.joblib` and `model_metadata.json` |
| `BUILD_SHA` | `dev` | Build identifier shown by `/health` |

## Rolling back

- **Backend:** re-run the deploy workflow for an earlier commit on `main`. Or open the
  Space's **Files → History** and revert the last commit; the Space rebuilds from it.
- **Frontend:** in Vercel, open **Deployments**, choose an earlier production deployment,
  then **Promote to Production**. Or run `vercel rollback`.

## Things to know

- **Cold start:** free Spaces sleep after about 48 hours without traffic. The first
  request after that waits while the Space starts.
- **Rate limits are per process:** if the Space ever runs more than one worker, each
  counts separately.
- **Privacy:** the Space's logs show request lines (method, path, status) but never audio
  bodies. The app doesn't log request content.
