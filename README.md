# AniSync

AniSync is a private group anime recommendation web app. Members of a room each share what they want to watch (liked anime + an optional mood text), the host triggers a compute, and the system clusters candidate anime and produces a final list everyone votes on.

---

## 0. Getting Started (local)

### Prereqs

Install these once:

- **Git**
- **Docker Desktop** — runs the local PostgreSQL + pgvector container
- **Python 3.13** with **[uv](https://github.com/astral-sh/uv)** for the venv
- **Node.js ≥ 20** with `npm` for the frontend (Vite needs a modern Node)
- **zstd** — to decompress the bundled anime dataset (`brew install zstd` on macOS)

### 1. Clone and create env files

```bash
git clone <REMOTE_REPO_URL>
cd anisync
cp .env.example .env
cp web/.env.example web/.env
```

Do not commit the `.env` files.

### 2. Start the database

```bash
docker compose up -d
docker compose ps   # anisync_pg should show "Up ... (healthy)"
```

### 3. Backend venv + schema

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt

cd api && python -m app.scripts.init_db
```

Expected: `Database initialized successfully.`

### 4. Decompress the bundled anime catalog

The anime metadata dump (`anime-offline-database.jsonl`, ~60 MB raw) is stored compressed in the repo as `data/raw/anime-offline-database.jsonl.zst` (~6 MB).
Decompress it once before preprocessing:

```bash
# from the repo root
zstd -d data/raw/anime-offline-database.jsonl.zst
```

If you want a newer release of the metadata dump, grab the latest `anime-offline-database.jsonl` from the upstream project: <https://github.com/manami-project/anime-offline-database/releases>. Drop it into `data/raw/` (replacing the bundled copy) and continue with preprocessing.

### 5. Preprocess the catalog

From inside `api/`:

```bash
# Quick smoke test (1k items, no image downloads — fast)
python -m app.scripts.preprocess_catalog \
  --raw ../data/raw/anime-offline-database.jsonl \
  --media-dir ../media \
  --processed-output ../data/processed/catalog_summary.jsonl \
  --reset --max-items 1000 --skip-images

# Full run (slow: tens of thousands of embeddings + image downloads)
python -m app.scripts.preprocess_catalog \
  --raw ../data/raw/anime-offline-database.jsonl \
  --batch-size 128 --workers 32
```

`--batch-size` controls GPU embedding batching, `--workers` controls parallel image downloads.

### 6. Seed demo users

From inside `api/`:

```bash
python -m app.scripts.seed_demo
```

Creates four demo accounts:

```
host@example.com
kai@example.com
mina@example.com
theo@example.com
```

Password for all of them: `AniSyncDemo123!`

### 7. Run the backend

From inside `api/` (with the venv active):

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Sanity check: `curl http://localhost:8000/api/health` → `{"ok":true,"service":"anisync-api"}`.

### 8. Run the frontend

In a new terminal:

```bash
cd web
npm install
npm run dev
```

Open http://localhost:5173.

### 9. Demo flow

Use two browser sessions (e.g. one normal + one incognito):

1. Login as `host@example.com`, create a room, copy the room code.
2. Login as `kai@example.com` in the second session, join with the code.
3. Both users add a few liked anime via the search bar (and optionally a mood line).
4. Host clicks **Generate Group Recommendations**.
5. Both users vote on the final list.
6. After everyone votes, the group result appears with the winner highlighted.

### 10. Optional — running the offline benchmark

The offline benchmark in `benchmark/` measures recommendation quality (NDCG@K) by replaying real MAL user profiles against AniSync's recommender. Two derived inputs are bundled in `data/processed/`:

- `user_profiles.jsonl.zst` — ~2,000 MAL users with ≥15 ratings, with each rating already mapped to a `catalog_item_id` (decompress with `zstd -d data/processed/user_profiles.jsonl.zst`)
- `mal_to_catalog.json` — MAL ID → catalog ID mapping derived from the bundled anime database

The 2.1 GB raw `animelists_cleaned.csv` is **not** bundled and is only needed if you want to regenerate `user_profiles.jsonl` from scratch. If you do, download it from Kaggle: <https://www.kaggle.com/datasets/azathoth42/myanimelist?resource=download&select=animelists_cleaned.csv>. With the bundled files, you can run benchmarks directly — see `benchmark/README.md` for details.

---

## 1. Tech Stack

### Backend and API

- **Python 3.13** — Main backend and machine learning language for the AniSync API, preprocessing scripts, embeddings, and manual K-means implementation.
- **FastAPI** — Python web framework used to build authentication routes, room APIs, voting APIs, recommendation APIs, and WebSocket endpoints.
- **Uvicorn** — ASGI server used to run the FastAPI backend locally and in production.
- **SQLAlchemy 2** — ORM used to define database models and query PostgreSQL from Python.
- **psycopg 3** — PostgreSQL driver used by SQLAlchemy to connect the backend to the database.
- **Pydantic** — Validation library used for API request bodies such as login, registration, room creation, constraints, submissions, and votes.
- **Argon2 / argon2-cffi** — Password hashing library used to store user passwords securely.

### Database and Data Storage

- **PostgreSQL 18** — Main relational database for users, rooms, room members, votes, anime metadata, embeddings, and computed recommendation results.
- **pgvector** — PostgreSQL extension used to store 384-dimensional anime embeddings and run exact vector similarity search.
- **JSONB columns** — PostgreSQL JSON storage used to preserve rich anime metadata, raw retained dataset records, room results, and structured lists.
- **Local media asset store** — Local folder storage used during development for downloaded or generated anime poster and thumbnail images.
- **Supabase Postgres** — Production PostgreSQL database option for deployed app data and pgvector support.
- **Supabase Storage** — Production media storage option for anime posters and thumbnails after offline preprocessing.

### Machine Learning and Recommendation System

- **sentence-transformers/all-MiniLM-L6-v2** — Text embedding model used to convert anime metadata and user preference text into normalized 384-dimensional vectors.
- **NumPy** — Numerical computing library used for vector normalization, similarity calculations, and the manual K-means algorithm.
- **Manual K-means implementation** — Course-required clustering algorithm used to group the room candidate anime pool into diverse recommendation clusters.
- **Silhouette-based K selection** — Product-aware cluster-count selection method used to choose a small, readable number of clusters for the UI.

### Offline Preprocessing and Media Pipeline

- **anime-offline-database JSONL dataset** — Real anime metadata dataset manually downloaded and processed before runtime.
- **httpx** — HTTP client used during offline preprocessing to download anime image assets safely.
- **Pillow** — Image-processing library used to normalize anime posters, thumbnails, and deterministic placeholder images.

### Frontend and UI

- **React 19** — Frontend UI library used to build the single-page AniSync web app.
- **TypeScript** — Typed JavaScript layer used to make frontend code safer and easier for teammates to maintain.
- **Vite** — Frontend build tool and development server used for fast React development and production builds.
- **React Router** — Client-side routing library used for login, registration, dashboard, room creation, room joining, and room pages.
- **Tailwind CSS 4** — Utility-first CSS framework used to create the modern dark theme, responsive layout, cards, badges, and premium UI styling.
- **Framer Motion** — Animation library used for motion design, page transitions, card animations, and micro-interactions.
- **lucide-react** — Icon library used for clean interface icons throughout the app.

### Real-Time Synchronization

- **FastAPI WebSockets** — Real-time transport used to notify room members when users join, submit preferences, update constraints, compute results, vote, or finish voting.
- **Room state revisions** — Monotonic room version numbers used to keep connected clients synchronized and ignore stale updates.

### Development, Collaboration, and Deployment

- **Docker Compose** — Local development tool used to run PostgreSQL with pgvector in a repeatable container setup.
- **uv** — Python environment and package management tool used to create the virtual environment and install backend dependencies.
- **npm** — JavaScript package manager used to install and run frontend dependencies.
- **Git and GitHub** — Version control and team collaboration tools used for branches, commits, pull requests, and the shared remote repository.
- **Render** — Production hosting option for the FastAPI backend service.
- **Vercel** — Production hosting option for the React/Vite frontend.

## 2. Repo Structure

### Root-Level Files and Folders

- **`README.md`** — Project overview file describing the tech stack and important repository structure.
- **`requirements.txt`** — Pinned Python dependency list used by the backend, preprocessing scripts, tests, and deployment.
- **`docker-compose.yml`** — Local Docker configuration for running PostgreSQL with pgvector.
- **`.env.example`** — Safe example environment file showing required local and deployment environment variables.
- **`.gitignore`** — Git ignore rules that prevent virtual environments, local datasets, generated media, secrets, and build outputs from being committed.
- **`api/`** — Python backend folder containing the FastAPI app, database models, ML code, preprocessing scripts, and backend tests.
- **`web/`** — React/Vite frontend folder containing the user interface code and frontend configuration.
- **`data/`** — Local data folder used for manually downloaded raw dataset files and generated preprocessing summaries.
- **`media/`** — Local media folder used for generated anime posters and thumbnails during development.

### Backend: `api/`

- **`api/app/main.py`** — Main FastAPI application containing authentication, room, constraint, compute, vote, health, and WebSocket routes.
- **`api/app/config.py`** — Central settings module that reads database, cookie, CORS, media, embedding, and deployment configuration from environment variables.
- **`api/app/db.py`** — SQLAlchemy engine, session factory, declarative base, and database dependency setup.
- **`api/app/models.py`** — SQLAlchemy models for users, catalog items, rooms, room members, preference submissions, and votes.
- **`api/app/security.py`** — Password hashing, password verification, and current-user session authentication helpers.
- **`api/app/embeddings.py`** — Embedding model loader and vector normalization utilities for anime text and user queries.
- **`api/app/realtime.py`** — In-memory WebSocket connection manager for room-level live update broadcasts.
- **`api/app/services/recommender.py`** — Recommendation pipeline that retrieves candidates, clusters anime, ranks results, builds final lists, and summarizes votes.
- **`api/app/ml/kmeans.py`** — Manual K-means implementation with initialization, assignment, update, empty-cluster repair, silhouette scoring, and K selection.
- **`api/app/scripts/init_db.py`** — Script that creates the pgvector extension and initializes database tables.
- **`api/app/scripts/preprocess_catalog.py`** — Offline preprocessing script that filters anime, builds metadata, downloads or generates images in parallel (with atomic writes for safe restarts), computes embeddings using the best available device (CUDA/MPS/CPU), and loads the catalog into PostgreSQL.
- **`api/app/scripts/seed_demo.py`** — Demo seeding script that creates local test users with known login credentials.
- **`api/app/scripts/sync_media_to_supabase.py`** — Deployment helper script for uploading local media assets to Supabase Storage and updating catalog image paths.
- **`api/tests/test_kmeans.py`** — Backend unit tests for the manual K-means and silhouette logic.

### Frontend: `web/`

- **`web/package.json`** — Frontend package file defining React, Vite, Tailwind, animation, icon, routing, and build dependencies.
- **`web/package-lock.json`** — Locked npm dependency tree used to keep frontend installs consistent across teammates.
- **`web/vite.config.ts`** — Vite configuration that enables React and Tailwind CSS support.
- **`web/.env.example`** — Example frontend environment file showing the API base URL variable.
- **`web/index.html`** — Vite HTML entry point that loads the React application.
- **`web/src/main.tsx`** — React entry file that mounts the app and enables browser routing.
- **`web/src/App.tsx`** — Main frontend application file containing the login, registration, dashboard, room, recommendation, voting, and result UI.
- **`web/src/api.ts`** — Frontend API helper for authenticated requests and media URL handling.
- **`web/src/index.css`** — Global CSS file that imports Tailwind and defines base styling for the dark premium UI theme.

### Data and Media Folders

- **`data/raw/`** — Local folder where developers manually place `anime-offline-database.jsonl` before preprocessing.
- **`data/processed/`** — Local folder where preprocessing can write catalog summary outputs for debugging and inspection.
- **`media/posters/`** — Local folder for normalized poster images created during offline preprocessing.
- **`media/thumbnails/`** — Local folder for normalized thumbnail images used in anime cards and result lists.
