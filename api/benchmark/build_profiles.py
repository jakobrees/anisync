"""
Build user profiles from animelists_cleaned.csv joined against the catalog DB.

Outputs:
  data/processed/mal_to_catalog.json   -- MAL integer ID → catalog_item_id
  data/processed/user_profiles.jsonl   -- sampled user profiles

Run: cd api && python -m benchmark.build_profiles --csv ../data/raw/animelists_cleaned.csv
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy import select
from tqdm import tqdm

from app.db import SessionLocal
from app.models import CatalogItem

MAL_URL_RE = re.compile(r"myanimelist\.net/anime/(\d+)")


def extract_mal_id(sources: list) -> int | None:
    for url in sources:
        m = MAL_URL_RE.search(str(url))
        if m:
            return int(m.group(1))
    return None


def build_mal_mapping(db) -> dict[int, int]:
    rows = db.execute(select(CatalogItem.id, CatalogItem.sources_json)).all()
    mapping: dict[int, int] = {}
    for row in tqdm(rows, desc="Extracting MAL IDs", unit="item"):
        mal_id = extract_mal_id(row.sources_json or [])
        if mal_id is not None:
            mapping[mal_id] = row.id
    return mapping


def load_ratings(csv_path: Path) -> dict[str, list[tuple[int, int]]]:
    ratings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading CSV", unit="rows"):
            try:
                score = int(row["my_score"])
                if score == 0:
                    continue
                ratings[row["username"]].append((int(row["anime_id"]), score))
            except (KeyError, ValueError):
                continue
    return dict(ratings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to animelists_cleaned.csv")
    parser.add_argument("--min-ratings", type=int, default=15)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--profile-seed", type=int, default=123)
    parser.add_argument("--output-dir", default="../data/processed")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        print("Building MAL → catalog_item mapping from DB...")
        mal_mapping = build_mal_mapping(db)
    finally:
        db.close()

    print(f"  Mapped {len(mal_mapping):,} MAL IDs to catalog items.")
    mapping_path = output_dir / "mal_to_catalog.json"
    with mapping_path.open("w") as f:
        json.dump({str(k): v for k, v in mal_mapping.items()}, f)
    print(f"  Saved {mapping_path}")

    print("Loading ratings from CSV...")
    all_ratings = load_ratings(csv_path)
    print(f"  {len(all_ratings):,} users found in CSV.")

    # join and filter
    profiles: list[dict] = []
    for username, raw_ratings in all_ratings.items():
        joined = []
        for mal_id, score in raw_ratings:
            catalog_id = mal_mapping.get(mal_id)
            if catalog_id is not None:
                joined.append({"mal_id": mal_id, "catalog_item_id": catalog_id, "score": score})
        if len(joined) >= args.min_ratings:
            profiles.append({"username": username, "ratings": joined})

    print(f"  {len(profiles):,} users pass min_ratings={args.min_ratings} and join filter.")

    # deterministic subsample
    if len(profiles) > args.max_users:
        profiles.sort(key=lambda p: p["username"])
        rng = np.random.default_rng(args.profile_seed)
        indices = rng.choice(len(profiles), args.max_users, replace=False)
        profiles = [profiles[int(i)] for i in sorted(indices)]

    output_path = output_dir / "user_profiles.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for profile in profiles:
            f.write(json.dumps(profile, ensure_ascii=False) + "\n")

    print(f"  Saved {len(profiles):,} profiles to {output_path}")


if __name__ == "__main__":
    main()
