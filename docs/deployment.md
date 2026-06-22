# Deployment — Backend on Render, Frontend on Vercel

Two services:
- **Backend** (FastAPI) → **Render** (web service)
- **Frontend** (Vite/React) → **Vercel** (static site)

> They reference each other's URLs, so the order is: **deploy backend → get its URL
> → deploy frontend with that URL → set the backend's `CORS_ORIGINS` to the
> frontend URL.** A `render.yaml` (repo root) and `frontend/vercel.json` are
> included to streamline this.

Prerequisite: push this repo to **GitHub/GitLab**.

---

## Part A — Backend on Render

### Option 1: Blueprint (uses `render.yaml`)
1. https://dashboard.render.com → **New + → Blueprint** → connect the repo.
2. Render reads `render.yaml` and proposes the `doc-intelligence-api` web service.
3. Set the secret env vars (below) → **Apply**.

### Option 2: Manual web service
1. **New + → Web Service** → connect the repo.
2. Configure:
   - **Root Directory:** `backend`
   - **Runtime:** Python
   - **Build Command:**
     ```
     pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt && python -m scripts.train_ner
     ```
     > CPU-only torch keeps the build within free-tier limits. The final
     > `train_ner` bakes the NER model so the API serves the **hybrid** tagger.
     > Drop it to ship rules-only (lighter, faster build).
   - **Start Command:**
     ```
     uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Health Check Path:** `/health`

### Environment variables (Render → service → Environment)
| Key | Value | Required |
|-----|-------|----------|
| `PYTHON_VERSION` | `3.12.7` | yes |
| `CORS_ORIGINS` | your Vercel URL, e.g. `https://your-app.vercel.app` | yes |
| `AUTH_SECRET` | a random string (`openssl rand -hex 32`) — signs session tokens | yes (else an insecure default is used) |
| `GROQ_API_KEY` | your Groq key | optional (LLM answers/summaries) |
| `DATABASE_URL` | your Neon/Postgres + pgvector URL | optional (persistent search + `users` table) |

> `AUTH_SECRET` is **not** an external credential — it's any random value you
> choose. Losing it just logs everyone out (passwords/accounts are stored hashed
> in the DB, unaffected). In `render.yaml` it's set to `generateValue: true`, so
> Render creates a strong one automatically.

> You won't know the Vercel URL until Part B. Put a placeholder now and **update
> `CORS_ORIGINS` after the frontend is live** (a redeploy/restart applies it).

### Verify
Open `https://<your-service>.onrender.com/health` → JSON with
`"status":"ok"` and `"tagger":"hybrid"` (or `"rule"` if you skipped training).
Interactive docs at `/docs`. **Copy the service URL** — you need it for Vercel.

> ⏰ **Free tier sleeps** after ~15 min idle; the first request then takes
> ~30–50 s to wake. To keep it warm, ping `/health` every ~10 min with a free
> uptime monitor (e.g. UptimeRobot) — optional.

---

## Part B — Frontend on Vercel

1. https://vercel.com → **Add New → Project** → import the repo.
2. Configure:
   - **Root Directory:** `frontend`  ← important
   - Framework preset auto-detects **Vite** (build `npm run build`, output `dist`;
     `frontend/vercel.json` also pins this and adds SPA fallback).
3. **Environment Variables:**
   | Key | Value |
   |-----|-------|
   | `VITE_API_URL` | your Render URL, e.g. `https://doc-intelligence-api.onrender.com` |
   | `VITE_GOOGLE_CLIENT_ID` | *(optional)* your Google OAuth client ID |
4. **Deploy.** Vercel gives you `https://your-app.vercel.app`.

---

## Part C — Connect the two

1. **Render → `CORS_ORIGINS`** = `https://your-app.vercel.app` → save (redeploys).
2. **(If using Google OAuth)** Google Cloud Console → Credentials → your OAuth
   client → **Authorized JavaScript origins** → add `https://your-app.vercel.app`.
3. Open the Vercel URL → sign up / sign in → it talks to the Render API.

> Changed `VITE_API_URL` later? Vite inlines env vars at **build time** — trigger
> a **Vercel redeploy** for changes to take effect.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Browser console **CORS error** | `CORS_ORIGINS` on Render must exactly equal the Vercel origin (`https://…`, no trailing slash). Redeploy after changing. |
| Frontend calls `localhost:8000` in prod | `VITE_API_URL` wasn't set at build → set it in Vercel and **redeploy**. |
| Render build **out of memory / too large** | Ensure CPU torch is installed first (the `--index-url …/cpu` flag). Or drop `python -m scripts.train_ner` to ship rules-only; or upgrade the instance. |
| First request very slow | Free instance was asleep — see the uptime-ping note above. |
| Google sign-in `origin_mismatch` | Add the Vercel origin to the OAuth client's **Authorized JavaScript origins**. |
| Health shows `"tagger":"rule"` | You skipped training, or it OOM'd at build — add/keep `python -m scripts.train_ner` and ensure the instance has enough memory. |
