"""
Generate natural-language preference summaries for benchmark users via Claude Batch API.

Translations are cached by (username, profile_seed, visible_ratio) so subsequent
runs with the same config cost nothing. The batch API gives a 50% discount vs.
real-time calls.

Run: cd api && python -m benchmark.llm_translate --visible-ratio 0.3

To resume after an interrupted run:
  cd api && python -m benchmark.llm_translate --batch-id <id> --visible-ratio 0.3
"""
import argparse
import json
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from sqlalchemy import select

load_dotenv()
from tqdm import tqdm

from app.db import SessionLocal
from app.models import CatalogItem
from benchmark.methods.base import UserProfile, UserRating, split_profile

SYSTEM_PROMPT = (
    "Summarize an anime viewer's tastes from their rated shows into "
    "a concise first-person preference description."
)

USER_PROMPT_TEMPLATE = """\
Here are anime I've rated (title | genres | my score/10):
{entries}

Write 2–3 sentences describing what I enjoy: genres, themes, tone. Be specific. Start with "I enjoy…"\
"""


def cache_file(cache_dir: Path, username: str, profile_seed: int, visible_ratio: float) -> Path:
    key = f"{username}__{profile_seed}__{visible_ratio:.3f}"
    return cache_dir / f"{key}.txt"


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


def build_user_prompt(db, visible: list[UserRating], min_score: int = 7) -> str:
    liked = [r for r in visible if r.score >= min_score] or visible
    ids = [r.catalog_item_id for r in liked]
    items = {
        item.id: item
        for item in db.scalars(select(CatalogItem).where(CatalogItem.id.in_(ids))).all()
    }
    entries = []
    for r in sorted(liked, key=lambda r: -r.score):
        item = items.get(r.catalog_item_id)
        if item is None:
            continue
        tags = (item.tags_json or [])[:5]
        genre_str = ", ".join(str(t) for t in tags) if tags else "anime"
        entries.append(f"- {item.title} | {genre_str} | {r.score}/10")
    return USER_PROMPT_TEMPLATE.format(entries="\n".join(entries))


def submit_and_poll(
    client: anthropic.Anthropic,
    requests: list[dict],
    model: str,
    poll_interval: int,
    batch_id_file: Path,
):
    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": req["custom_id"],
                "params": {
                    "model": model,
                    "max_tokens": 256,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": req["prompt"]}],
                },
            }
            for req in requests
        ]
    )
    print(f"Submitted batch {batch.id}")
    batch_id_file.write_text(batch.id)
    return poll_batch(client, batch.id, poll_interval)


def poll_batch(
    client: anthropic.Anthropic,
    batch_id: str,
    poll_interval: int,
):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  [{batch.processing_status}] "
            f"processing={counts.processing} "
            f"succeeded={counts.succeeded} "
            f"errored={counts.errored}"
        )
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-ratio", type=float, default=0.3)
    parser.add_argument("--profile-seed", type=int, default=123)
    parser.add_argument("--llm-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--cache-dir", default="../benchmark/cache/llm")
    parser.add_argument("--profiles", default="../data/processed/user_profiles.jsonl")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--batch-id", default=None, help="Resume polling an existing batch ID")
    parser.add_argument("--min-score", type=int, default=7, help="Only include anime with score >= this in the prompt (default: 7)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = Path(args.profiles).resolve()

    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Profiles not found: {profiles_path}. Run build_profiles.py first."
        )

    profiles = load_profiles(profiles_path)
    print(f"Loaded {len(profiles)} user profiles.")

    to_translate = [
        p for p in profiles
        if not cache_file(cache_dir, p.username, args.profile_seed, args.visible_ratio).exists()
    ]
    print(f"{len(to_translate)} cache misses.")

    if not to_translate and args.batch_id is None:
        print("All translations cached. Nothing to do.")
        return

    client = anthropic.Anthropic()
    batch_id_file = cache_dir / "pending_batch_id.txt"

    if args.batch_id:
        print(f"Resuming batch {args.batch_id}...")
        idx_to_username = {str(i): p.username for i, p in enumerate(to_translate)}
        poll_batch(client, args.batch_id, args.poll_interval)
        batch_id_for_results = args.batch_id
    else:
        db = SessionLocal()
        try:
            requests = []
            idx_to_username: dict[str, str] = {}
            for i, profile in enumerate(tqdm(to_translate, desc="Preparing prompts")):
                visible, _ = split_profile(profile, args.visible_ratio, args.profile_seed)
                if not visible:
                    continue
                prompt = build_user_prompt(db, visible, min_score=args.min_score)
                custom_id = str(i)
                idx_to_username[custom_id] = profile.username
                requests.append({"custom_id": custom_id, "prompt": prompt})
        finally:
            db.close()

        print(f"Submitting batch of {len(requests)} requests...")
        batch = submit_and_poll(client, requests, args.llm_model, args.poll_interval, batch_id_file)
        batch_id_for_results = batch.id

    # save results to cache
    saved = errors = 0
    for result in client.messages.batches.results(batch_id_for_results):
        username = idx_to_username.get(result.custom_id)
        if username is None:
            continue
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            cf = cache_file(cache_dir, username, args.profile_seed, args.visible_ratio)
            cf.write_text(text, encoding="utf-8")
            saved += 1
        else:
            print(f"  Error for {username}: {result.result}")
            errors += 1

    if batch_id_file.exists():
        batch_id_file.unlink()

    print(f"Done. Saved {saved} translations, {errors} errors.")


if __name__ == "__main__":
    main()
