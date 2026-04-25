"""
Score one algorithm against the benchmark dataset.

Each algorithm in benchmark/algorithms/ is a self-contained module exposing:
  recommend(db, visible_by_user, cfg) -> list[int]

This script handles everything else: loading profiles, sampling groups,
splitting visible/hidden, calling the algorithm, computing NDCG, and
writing the result JSON.

Run:
  cd api && python -m benchmark.run --algorithm centroid --num-groups 5
  cd api && python -m benchmark.run --algorithm groupmatch_clustered
"""
import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from app.db import SessionLocal
from benchmark.config import BenchmarkConfig
from benchmark.methods.base import (
    UserProfile,
    UserRating,
    build_proxy_relevant_set,
    ndcg_at_k,
    split_profile,
)

ALGORITHMS = [
    "centroid",
    "groupmatch_raw",
    "groupmatch_clustered",
    "groupmatch_raw_llm",
    "groupfit",
    "groupfit_pos_text",
]


def load_algorithm(name: str):
    module = importlib.import_module(f"benchmark.algorithms.{name}")
    precompute_fn = getattr(module, "precompute", None)
    return module.recommend, precompute_fn


def load_profiles(profiles_path: Path) -> list[UserProfile]:
    profiles = []
    with profiles_path.open() as f:
        for line in f:
            data = json.loads(line)
            profiles.append(UserProfile(
                username=data["username"],
                ratings=[UserRating(**r) for r in data["ratings"]],
            ))
    return profiles


def sample_groups(
    profiles: list[UserProfile],
    num_groups: int,
    group_size: int,
    group_seed: int,
) -> list[list[UserProfile]]:
    rng = np.random.default_rng(group_seed)
    shuffled = profiles.copy()
    rng.shuffle(shuffled)
    groups = []
    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        if end > len(shuffled):
            break
        groups.append(shuffled[start:end])
    return groups


def score_group(db, group: list[UserProfile], recommend_fn, cfg: BenchmarkConfig) -> dict:
    visible_by_user: dict[str, list[UserRating]] = {}
    hidden_by_user: dict[str, list[UserRating]] = {}

    for profile in group:
        visible, hidden = split_profile(profile, cfg.visible_ratio, cfg.profile_seed)
        visible_by_user[profile.username] = visible
        hidden_by_user[profile.username] = hidden

    ranked_ids = recommend_fn(db, visible_by_user, cfg)

    ndcg_scores = []
    for profile in group:
        proxy_rel = build_proxy_relevant_set(db, hidden_by_user[profile.username])
        ndcg_scores.append(ndcg_at_k(ranked_ids, proxy_rel, cfg.ndcg_k))

    return {
        "users": [p.username for p in group],
        "per_user_ndcg": [round(s, 6) for s in ndcg_scores],
        "mean_ndcg": round(float(np.mean(ndcg_scores)), 6),
        "worst_user_ndcg": round(float(np.min(ndcg_scores)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--config", default="../benchmark/config.yaml")
    parser.add_argument("--visible-ratio", type=float, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--group-seed", type=int, default=None)
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument("--ndcg-k", type=int, default=None)
    parser.add_argument("--profile-seed", type=int, default=None)
    parser.add_argument("--profiles", default="../data/processed/user_profiles.jsonl")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--groupfit-lambda", type=float, default=None)
    parser.add_argument("--groupfit-beta", type=float, default=None)
    parser.add_argument("--groupfit-alpha", type=float, default=None)
    parser.add_argument("--use-msmarco", action="store_true", default=None,
                        help="Use msmarco-MiniLM-L6-cos-v5 for text alignment term")
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_yaml(Path(args.config))
    cfg.apply_args(args)
    if args.results_dir:
        cfg.results_dir = args.results_dir

    profiles_path = Path(args.profiles).resolve()
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Profiles not found: {profiles_path}. Run build_profiles.py first."
        )

    recommend_fn, precompute_fn = load_algorithm(args.algorithm)
    profiles = load_profiles(profiles_path)
    print(f"Loaded {len(profiles)} user profiles.")

    groups = sample_groups(profiles, cfg.num_groups, cfg.group_size, cfg.group_seed)
    print(f"Sampled {len(groups)} groups of {cfg.group_size}. Running [{args.algorithm}]...")

    db = SessionLocal()
    try:
        if precompute_fn is not None:
            precompute_fn(db, groups, cfg)
        group_results = []
        for group_idx, group in enumerate(tqdm(groups, desc="Scoring groups")):
            result = score_group(db, group, recommend_fn, cfg)
            result["group_id"] = group_idx
            group_results.append(result)
    finally:
        db.close()

    all_mean = [g["mean_ndcg"] for g in group_results]
    all_worst = [g["worst_user_ndcg"] for g in group_results]
    summary = {
        "mean_ndcg": round(float(np.mean(all_mean)), 6),
        "std_ndcg": round(float(np.std(all_mean)), 6),
        "mean_worst_user_ndcg": round(float(np.mean(all_worst)), 6),
        "num_groups": len(group_results),
    }

    algo_label = args.algorithm + ("_msmarco" if cfg.use_msmarco else "")
    output = {
        "algorithm": algo_label,
        "config": {
            "visible_ratio": cfg.visible_ratio,
            "group_size": cfg.group_size,
            "num_groups": cfg.num_groups,
            "ndcg_k": cfg.ndcg_k,
            "profile_seed": cfg.profile_seed,
            "group_seed": cfg.group_seed,
            "use_msmarco": cfg.use_msmarco,
        },
        "run_at": datetime.now(timezone.utc).isoformat(),
        "groups": group_results,
        "summary": summary,
    }

    results_dir = Path(cfg.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{algo_label}_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")
    print(
        f"mean_NDCG@{cfg.ndcg_k}={summary['mean_ndcg']:.4f} "
        f"± {summary['std_ndcg']:.4f}  "
        f"mean_worst={summary['mean_worst_user_ndcg']:.4f}"
    )


if __name__ == "__main__":
    main()
