# AniSync

## 1. Overview

AniSync is a private group anime-decision web app: friends create a room, each member submits a few liked anime (and an optional mood line), the host triggers a compute, and the room votes on a shared shortlist. The interesting part is what's underneath — group recommendation is structurally different from single-user recommendation. There is no single user whose preferences the system can predict, so collapsing the room into one averaged query throws away the structure the rest of the pipeline is supposed to honor.

AniSync's design takes the multiplicity of the group seriously at three pipeline stages:

1. **Per-user retrieval.** Each member runs their own pgvector cosine search against the catalog (top 100 each), and the per-user pools are unioned across the room. We never average the group into one query before searching, because that collapses the structure later stages are trying to preserve. This also functions as a fairness mechanism at the catalog level — users whose taste sits in sparse regions of embedding space get equal pool representation.
2. **Min-aggregation in scoring.** For each candidate, the system takes every user's best-matching liked item (max over their likes), then takes the worst of those across users (min over users). A high score means every user has at least one liked anime that the candidate resembles. A small text-alignment term (α = 0.30) blends in mood-description matching averaged across users; α was chosen by ablation.
3. **Silhouette-guided clustering for diversification.** K-means is implemented from scratch in [`api/app/ml/kmeans.py`](api/app/ml/kmeans.py) — including farthest-point initialization, empty-cluster repair, silhouette scoring, and bounded K selection over k ∈ {2, ..., 6} — with no scikit-learn dependency. It clusters the candidate pool, and the voting list takes top items per cluster, which guarantees it spans the room's taste space rather than collapsing toward whatever scoring direction dominates globally.

An offline benchmark on MyAnimeList user histories was used as a sieve — it ruled out structurally broken approaches (centroid-based retrieval, negative-similarity penalties, asymmetric MS-MARCO embeddings) but does not adjudicate between close-performing algorithms in the surviving family. Final algorithm choice rests on group-level fairness and discovery properties the metric structurally cannot measure. Full methodology and quantitative findings are in [`docs/benchmark_methodology.tex`](docs/benchmark_methodology.tex); the runtime algorithm spec is in [`docs/recommendation_algorithm.tex`](docs/recommendation_algorithm.tex).

## 2. Datasets

- **anime-offline-database** — AniSync's main anime catalog source: titles, metadata, tags, scores, images, and the text inputs we embed offline. Credit: [manami-project/anime-offline-database](https://github.com/manami-project/anime-offline-database).
- **MyAnimeList Dataset** — Used only for the offline benchmark, where real user rating histories are converted into held-out recommendation tests. Credit: [Kaggle — MyAnimeList Dataset by azathoth42](https://www.kaggle.com/datasets/azathoth42/myanimelist).

Both datasets are bundled in compressed form under `data/` so the app and the benchmark can run without manual downloads.

## 3. Algorithm

The runtime recommendation score for a candidate $i$ across users $u$ with liked-item embeddings $\ell_{u,j}$ and text embedding $t_u$ is:

```
score(i) = (1 − α) · min_u  max_j (e_i · ℓ_{u,j})
         +     α  · mean_u (e_i · t_u)
```

with `α = 0.30` (text contributes a light complement to the per-user fairness floor) and `λ = 0` (negative-similarity penalty disabled — proven harmful by the ablation). All embeddings come from `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ℓ₂-normalized). Catalog embeddings are precomputed during preprocessing; only user mood text is embedded at request time. See [`docs/recommendation_algorithm.tex`](docs/recommendation_algorithm.tex) for the full pipeline specification, and [`docs/benchmark_methodology.tex`](docs/benchmark_methodology.tex) for the ablations behind the hyperparameter choices.

## 4. Tech Stack

- **Backend:** Python 3.13, FastAPI, Uvicorn, SQLAlchemy 2, Psycopg 3, Pydantic, Argon2, itsdangerous.
- **Database:** PostgreSQL with pgvector, JSONB metadata storage, Docker Compose for local database setup.
- **ML / Recommendation:** `sentence-transformers/all-MiniLM-L6-v2`, NumPy, pgvector cosine search, manual K-means, silhouette-based K selection, GroupFit pos+text ranking.
- **Preprocessing:** httpx, Pillow, zstd, offline image normalization, deterministic image placeholders.
- **Frontend:** React 19, TypeScript, Vite, React Router, Tailwind CSS 4, Framer Motion, lucide-react.
- **Realtime:** FastAPI WebSockets with room state revisions.
- **Deployment:** Supabase Postgres + Storage, Render backend, Vercel frontend.

## 5. Local Setup

### Prerequisites

Install these tools once:

- Git
- Docker Desktop
- Python 3.13 with `uv`
- Node.js ≥ 20.19 (or ≥ 22.12) with `npm` — Vite 7 will not run under earlier 20.x versions
- `zstd`

### Clone the repo

```bash
git clone <REMOTE_REPO_URL>
cd anisync
```

### Create environment files

```bash
cp .env.example .env
cp web/.env.example web/.env
```

Update `SESSION_SECRET` in `.env` before running seriously. Do not commit `.env` files.

### Start PostgreSQL + pgvector

```bash
docker compose up -d
docker compose ps
```

Expected: `anisync_pg` is healthy.

### Install backend dependencies and initialize schema

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt

cd api
python -m app.scripts.init_db
```

Expected: `Database initialized successfully.`

### Prepare the anime catalog

From the repo root:

```bash
zstd -d -k data/raw/anime-offline-database.jsonl.zst
```

Then run the full preprocessing job from `api/`:

```bash
cd api
source ../.venv/bin/activate

python -m app.scripts.preprocess_catalog \
  --raw ../data/raw/anime-offline-database.jsonl \
  --media-dir ../media \
  --processed-output ../data/processed/catalog_summary.jsonl \
  --reset \
  --batch-size 128 \
  --workers 32
```

This filters the raw catalog, downloads or generates image assets, builds curated embedding text, computes normalized embeddings, and loads `catalog_items` into PostgreSQL.

### Optional local seed users

```bash
cd api
source ../.venv/bin/activate
python -m app.scripts.seed_demo
```

### Optional benchmark cache

The benchmark suite includes a `groupmatch_raw_llm` variant that uses
LLM-written natural-language profile summaries as query vectors (this mirrors
the live web app, where users type a free-text mood line). The 2,000 translated
profiles for the canonical config are bundled in `benchmark/cache/llm.tar.zst`
so the benchmark is reproducible without re-running Claude. Unpack once before
running benchmarks:

```bash
tar --zstd -xf benchmark/cache/llm.tar.zst -C benchmark/cache
```

End-to-end benchmark commands and the option to refresh the cache are
documented in [`benchmark/README.md`](benchmark/README.md).

### Run backend

```bash
cd api
source ../.venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/api/health
```

Expected:

```json
{"ok": true, "service": "anisync-api"}
```

### Run frontend

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

### Run checks

```bash
cd api
source ../.venv/bin/activate
PYTHONPATH=. pytest -q

cd ..
ruff check api

cd web
npm run build
```

## 6. Repo Structure

- **`api/`** — FastAPI backend, database models, authentication, WebSocket room sync, recommendation service, ML utilities, preprocessing scripts, and tests.
  - **`api/app/main.py`** — Main API routes for auth, catalog search, rooms, submissions, constraints, compute, voting, health, and WebSockets.
  - **`api/app/services/recommender.py`** — Production recommendation pipeline using host filters, pgvector retrieval, GroupFit pos+text scoring, clustering, and vote summaries.
  - **`api/app/ml/kmeans.py`** — Manual K-means implementation, empty-cluster repair, silhouette scoring, and bounded K selection.
  - **`api/app/scripts/preprocess_catalog.py`** — Offline catalog preprocessing, metadata normalization, image handling, embedding generation, and catalog import.
  - **`api/app/scripts/sync_media_to_supabase.py`** — Deployment helper for uploading local media assets to Supabase Storage and updating public image URLs.
- **`api/benchmark/`** — Implementations of the recommendation algorithms compared in the offline benchmark. See [`api/benchmark/algorithms/README.md`](api/benchmark/algorithms/README.md) for the algorithm interface and the catalog of variants (centroid, GroupMatch raw/clustered, GroupFit family).
- **`benchmark/`** — Benchmark configuration, cached LLM translations, and ablation result JSONs. See [`benchmark/README.md`](benchmark/README.md) for how to run sweeps and reproduce the published numbers.
- **`docs/`** — LaTeX writeups: `benchmark_methodology.tex` (NDCG@K methodology + ablation findings) and `recommendation_algorithm.tex` (runtime algorithm specification).
- **`web/`** — React/Vite frontend application. See [`web/README.md`](web/README.md) for the Vite/TypeScript/Tailwind toolchain.
  - **`web/src/App.tsx`** — Main UI for login, registration, dashboard, rooms, preferences, recommendations, voting, and final results.
  - **`web/src/api.ts`** — Frontend API helper for authenticated requests, bearer-token fallback, and media URLs.
- **`data/`** — Raw and processed dataset files used for preprocessing and benchmarking. The bundled `.zst` archives keep the repo small while making the app and benchmark runnable from a fresh clone.
- **`media/`** — Generated local posters and thumbnails for development.
- **`docker-compose.yml`** — Local PostgreSQL + pgvector database service.
- **`requirements.txt`** — Backend, ML, preprocessing, benchmark, and test dependencies.

## 7. Deployment Plan

AniSync is deployed as a split frontend/backend system. The preprocessed catalog is prepared locally once, then restored into **Supabase Postgres** with pgvector enabled. Poster and thumbnail assets are uploaded to **Supabase Storage**, and the catalog image paths are updated to public storage URLs. The FastAPI backend is hosted on **Render** with production database, cookie, CORS, embedding-model, and small SQLAlchemy pool environment variables. The React/Vite frontend is hosted on **Vercel** with `VITE_API_BASE_URL` pointing to the Render API. Runtime deployment should not download the anime catalog or recompute catalog embeddings.
