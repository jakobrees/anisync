"""
Overnight ablation study for GroupFit hyperparameters.

groupfit          — grid search over λ (negative weight) × β (text weight)
groupfit_pos_text — sweep over α (text blend, single param)

Run:
  cd api && python -m benchmark.ablation --algorithm groupfit --ndcg-k 200 --num-groups 400
  cd api && python -m benchmark.ablation --algorithm groupfit_pos_text --ndcg-k 200 --num-groups 400
  cd api && python -m benchmark.ablation --ndcg-k 200 --num-groups 400  # both

Results are written to a dedicated subdirectory per ablation run:
  benchmark/results/ablation_groupfit_<timestamp>/
  benchmark/results/ablation_groupfit_pos_text_<timestamp>/

Each parameter combination gets its own JSON file. Run summarize.py on a
specific subdir to avoid mixing ablation results with one-shot runs:
  cd api && python -m benchmark.summarize --results-dir ../benchmark/results/ablation_groupfit_<timestamp>
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
from benchmark.methods.base import UserProfile, UserRating, build_proxy_relevant_set, ndcg_at_k, split_profile

# ── Ablation grids ────────────────────────────────────────────────────────────

GROUPFIT_LAMBDA_VALUES    = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
GROUPFIT_BETA_VALUES      = [0.0, 0.2, 0.5, 0.8, 1.0]
GROUPFIT_POS_TEXT_ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def run_one(
    algorithm: str,
    cfg: BenchmarkConfig,
    groups: list[list[UserProfile]],
    label: str,
    ablation_dir: Path,
) -> dict:
    """Score one parameter combination and write its JSON to ablation_dir."""
    module = importlib.import_module(f"benchmark.algorithms.{algorithm}")
    recommend_fn = module.recommend

    db = SessionLocal()
    try:
        group_results = []
        for group_idx, group in enumerate(tqdm(groups, desc=label, leave=False)):
            result = score_group(db, group, recommend_fn, cfg)
            result["group_id"] = group_idx
            group_results.append(result)
    finally:
        db.close()

    all_mean  = [g["mean_ndcg"] for g in group_results]
    all_worst = [g["worst_user_ndcg"] for g in group_results]
    summary = {
        "mean_ndcg":            round(float(np.mean(all_mean)), 6),
        "std_ndcg":             round(float(np.std(all_mean)), 6),
        "mean_worst_user_ndcg": round(float(np.mean(all_worst)), 6),
        "num_groups":           len(group_results),
    }

    output = {
        "algorithm": algorithm,
        "config": {
            "visible_ratio":   cfg.visible_ratio,
            "group_size":      cfg.group_size,
            "num_groups":      cfg.num_groups,
            "ndcg_k":          cfg.ndcg_k,
            "profile_seed":    cfg.profile_seed,
            "group_seed":      cfg.group_seed,
            "groupfit_lambda": cfg.groupfit_lambda,
            "groupfit_beta":   cfg.groupfit_beta,
            "groupfit_alpha":  cfg.groupfit_alpha,
        },
        "run_at": datetime.now(timezone.utc).isoformat(),
        "groups": group_results,
        "summary": summary,
    }

    # Microsecond timestamp avoids collisions when runs finish within the same second.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_path = ablation_dir / f"{label}_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return summary


# ── Ablation runners ──────────────────────────────────────────────────────────

def ablate_groupfit(base_cfg: BenchmarkConfig, profiles, groups, base_results_dir: Path):
    combos = [(lam, beta) for lam in GROUPFIT_LAMBDA_VALUES for beta in GROUPFIT_BETA_VALUES]
    print(f"\n── groupfit ablation: {len(GROUPFIT_LAMBDA_VALUES)} λ × {len(GROUPFIT_BETA_VALUES)} β "
          f"= {len(combos)} runs ──")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ablation_dir = base_results_dir / f"ablation_groupfit_{ts}"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results → {ablation_dir}")

    # Precompute once — the cache key covers profile_seed, visible_ratio, group_seed,
    # num_groups, so it is reused across all λ/β combinations.
    module = importlib.import_module("benchmark.algorithms.groupfit")
    precompute_fn = getattr(module, "precompute", None)
    if precompute_fn is not None:
        db = SessionLocal()
        try:
            precompute_fn(db, groups, base_cfg)
        finally:
            db.close()

    rows = []
    for lam, beta in tqdm(combos, desc="groupfit grid"):
        cfg = BenchmarkConfig(**{**base_cfg.__dict__, "groupfit_lambda": lam, "groupfit_beta": beta})
        label = f"lam{lam:.2f}_beta{beta:.2f}"
        summary = run_one("groupfit", cfg, groups, label, ablation_dir)
        rows.append((lam, beta, summary["mean_ndcg"], summary["std_ndcg"], summary["mean_worst_user_ndcg"]))

    rows.sort(key=lambda r: -r[2])
    print(f"\n{'λ':>6} {'β':>6} | {'mean_NDCG':>10} {'std':>8} {'mean_worst':>12}")
    print("-" * 52)
    for lam, beta, mean, std, worst in rows:
        print(f"{lam:>6.2f} {beta:>6.2f} | {mean:>10.4f} {std:>8.4f} {worst:>12.4f}")


def ablate_groupfit_pos_text(base_cfg: BenchmarkConfig, profiles, groups, base_results_dir: Path):
    print(f"\n── groupfit_pos_text ablation: {len(GROUPFIT_POS_TEXT_ALPHA_VALUES)} α values ──")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ablation_dir = base_results_dir / f"ablation_groupfit_pos_text_{ts}"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results → {ablation_dir}")

    module = importlib.import_module("benchmark.algorithms.groupfit_pos_text")
    precompute_fn = getattr(module, "precompute", None)
    if precompute_fn is not None:
        db = SessionLocal()
        try:
            precompute_fn(db, groups, base_cfg)
        finally:
            db.close()

    rows = []
    for alpha in tqdm(GROUPFIT_POS_TEXT_ALPHA_VALUES, desc="groupfit_pos_text sweep"):
        cfg = BenchmarkConfig(**{**base_cfg.__dict__, "groupfit_alpha": alpha})
        label = f"alpha{alpha:.2f}"
        summary = run_one("groupfit_pos_text", cfg, groups, label, ablation_dir)
        rows.append((alpha, summary["mean_ndcg"], summary["std_ndcg"], summary["mean_worst_user_ndcg"]))

    rows.sort(key=lambda r: -r[1])
    print(f"\n{'α':>6} | {'mean_NDCG':>10} {'std':>8} {'mean_worst':>12}")
    print("-" * 42)
    for alpha, mean, std, worst in rows:
        print(f"{alpha:>6.2f} | {mean:>10.4f} {std:>8.4f} {worst:>12.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["groupfit", "groupfit_pos_text", "both"], default="both")
    parser.add_argument("--config", default="../benchmark/config.yaml")
    parser.add_argument("--profiles", default="../data/processed/user_profiles.jsonl")
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument("--ndcg-k", type=int, default=None)
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_yaml(Path(args.config))
    if args.num_groups is not None:
        cfg.num_groups = args.num_groups
    if args.ndcg_k is not None:
        cfg.ndcg_k = args.ndcg_k

    profiles_path = Path(args.profiles).resolve()
    if not profiles_path.exists():
        raise FileNotFoundError(f"Profiles not found: {profiles_path}. Run build_profiles.py first.")

    profiles = load_profiles(profiles_path)
    groups = sample_groups(profiles, cfg.num_groups, cfg.group_size, cfg.group_seed)
    print(f"Loaded {len(profiles)} profiles → {len(groups)} groups of {cfg.group_size}")

    base_results_dir = Path(cfg.results_dir).resolve()

    if args.algorithm in ("groupfit", "both"):
        ablate_groupfit(cfg, profiles, groups, base_results_dir)
    if args.algorithm in ("groupfit_pos_text", "both"):
        ablate_groupfit_pos_text(cfg, profiles, groups, base_results_dir)


if __name__ == "__main__":
    main()
