import argparse
import datetime as dt
import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal
from app.embeddings import embed_texts, get_embedding_model
from app.models import CatalogItem


ALLOWED_TYPES = {"TV", "MOVIE", "OVA", "ONA", "SPECIAL"}
ALLOWED_STATUSES = {"FINISHED", "ONGOING"}


def as_list(value) -> list:
    """Safely normalize a JSON field into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def string_list(value) -> list[str]:
    """
    Normalize list-like fields into clean strings.

    The dataset usually stores these as arrays of strings,
    but this helper prevents crashes if a future row is odd.
    """
    result: list[str] = []
    for item in as_list(value):
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def normalize_title(title: str) -> str:
    """Normalize a title for internal matching/search."""
    text = unicodedata.normalize("NFKC", title)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def source_item_id(record: dict) -> str:
    sources = string_list(record.get("sources"))
    if sources:
        return sha1_text("|".join(sorted(sources)))
    return sha1_text(str(record.get("title", "")).strip())


def source_provider_domains(sources: list[str]) -> list[str]:
    domains: list[str] = []
    for source in sources:
        try:
            hostname = urlparse(source).hostname
            if hostname:
                domains.append(hostname.lower().removeprefix("www."))
        except Exception:
            continue
    return sorted(set(domains))


def get_duration_seconds(duration: dict | None) -> int | None:
    if not isinstance(duration, dict):
        return None
    if duration.get("unit") != "SECONDS":
        return None
    value = duration.get("value")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def get_score_fields(score: dict | None) -> tuple[float | None, float | None, float | None, float | None]:
    if not isinstance(score, dict):
        return None, None, None, None

    agm = score.get("arithmeticGeometricMean")
    mean = score.get("arithmeticMean")
    median = score.get("median")

    def to_float(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    agm_f = to_float(agm)
    mean_f = to_float(mean)
    median_f = to_float(median)

    # Main display score priority required by the design document.
    main_score = mean_f if mean_f is not None else agm_f if agm_f is not None else median_f
    return main_score, agm_f, mean_f, median_f


def build_search_text(title: str, synonyms: list[str]) -> str:
    values = [title] + synonyms
    return " | ".join(text for text in values if text)


def build_text_blob(
    *,
    title: str,
    media_type: str,
    status: str,
    year: int,
    season: str | None,
    episodes: int | None,
    duration_seconds: int | None,
    score: float | None,
    synonyms: list[str],
    studios: list[str],
    producers: list[str],
    tags: list[str],
) -> str:
    """
    Curated embedding text.

    This intentionally avoids dumping every raw field.
    The field order follows the design document's semantic priority.
    """
    parts: list[str] = []

    def add(label: str, value) -> None:
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        parts.append(f"{label}: {value}")

    add("title", title)
    if synonyms:
        add("synonyms", ", ".join(synonyms[:12]))
    if tags:
        add("tags", ", ".join(tags[:20]))
    add("type", media_type)
    add("status", status)
    add("year", year)
    add("season", season)
    add("episodes", episodes)
    add("duration_seconds", duration_seconds)
    if studios:
        add("studios", ", ".join(studios[:5]))
    if producers:
        add("producers", ", ".join(producers[:8]))
    add("score", score)

    return " | ".join(parts)


def deterministic_color(item_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()
    return (
        int(digest[0:2], 16),
        int(digest[2:4], 16),
        int(digest[4:6], 16),
    )


def initials_from_title(title: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", title)
    if not words:
        return "AN"
    return "".join(word[0].upper() for word in words[:2])


def save_placeholder(title: str, item_id: str, output_path: Path, size: tuple[int, int]) -> None:
    """
    Generate deterministic poster/thumbnail placeholder.

    This prevents broken images in the UI.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bg = deterministic_color(item_id)
    image = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(image)

    initials = initials_from_title(title)
    font = ImageFont.load_default(size=48 if size[0] >= 300 else 24)

    bbox = draw.textbbox((0, 0), initials, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    draw.text(
        ((size[0] - text_w) / 2, (size[1] - text_h) / 2),
        initials,
        fill=(255, 255, 255),
        font=font,
    )
    image.save(output_path, "JPEG", quality=90)


def normalize_and_save_image(raw_bytes: bytes, output_path: Path, size: tuple[int, int]) -> tuple[str, int, int, str]:
    """
    Decode, center-crop, resize, and save a runtime JPEG asset.

    Writes to a temp file then renames atomically so an interrupted write
    never leaves a corrupt file that the cache check would treat as valid.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from io import BytesIO

    image = Image.open(BytesIO(raw_bytes))
    image = image.convert("RGB")
    image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)

    tmp_path = output_path.with_suffix(".tmp")
    image.save(tmp_path, "JPEG", quality=88, optimize=True)
    tmp_path.rename(output_path)

    sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return "image/jpeg", size[0], size[1], sha


def try_download_image(client: httpx.Client, url: str | None, max_bytes: int = 5_000_000) -> bytes | None:
    """
    Download an image safely.

    Requirements:
    - HTTP 200
    - image/* content type
    - below maximum size
    - decodable later by Pillow
    """
    if not url:
        return None

    try:
        response = client.get(url)
        if response.status_code != 200:
            return None

        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            return None

        if len(response.content) > max_bytes:
            return None

        return response.content
    except Exception:
        return None


def create_image_assets(
    *,
    client: httpx.Client,
    item_id: str,
    title: str,
    thumbnail_url: str | None,
    picture_url: str | None,
    media_dir: Path,
) -> dict:
    """
    Create local poster and thumbnail assets.

    Primary order:
    1. dataset thumbnail
    2. dataset picture
    3. deterministic placeholder
    """
    poster_path = media_dir / "posters" / f"{item_id}.jpg"
    thumbnail_path = media_dir / "thumbnails" / f"{item_id}.jpg"

    if poster_path.exists() and thumbnail_path.exists():
        return {
            "image_local_path": f"/media/posters/{item_id}.jpg",
            "thumbnail_local_path": f"/media/thumbnails/{item_id}.jpg",
            "image_download_status": "cached",
            "image_mime_type": "image/jpeg",
            "image_width": None,
            "image_height": None,
            "image_sha256": None,
        }

    raw = try_download_image(client, thumbnail_url) or try_download_image(client, picture_url)

    if raw is not None:
        try:
            mime_type, width, height, sha = normalize_and_save_image(raw, poster_path, (320, 450))
            normalize_and_save_image(raw, thumbnail_path, (160, 225))
            status = "downloaded"
        except Exception:
            save_placeholder(title, item_id, poster_path, (320, 450))
            save_placeholder(title, item_id, thumbnail_path, (160, 225))
            mime_type, width, height = "image/jpeg", 320, 450
            sha = hashlib.sha256(poster_path.read_bytes()).hexdigest()
            status = "generated_placeholder"
    else:
        save_placeholder(title, item_id, poster_path, (320, 450))
        save_placeholder(title, item_id, thumbnail_path, (160, 225))
        mime_type, width, height = "image/jpeg", 320, 450
        sha = hashlib.sha256(poster_path.read_bytes()).hexdigest()
        status = "generated_placeholder"

    return {
        "image_local_path": f"/media/posters/{item_id}.jpg",
        "thumbnail_local_path": f"/media/thumbnails/{item_id}.jpg",
        "image_download_status": status,
        "image_mime_type": mime_type,
        "image_width": width,
        "image_height": height,
        "image_sha256": sha,
    }


def parse_record(record: dict, current_year: int) -> dict | None:
    """
    Convert a raw anime-offline-database record into AniSync's normalized fields.

    Returns None when the record should be discarded.
    """
    title = str(record.get("title") or "").strip()
    if not title:
        return None

    anime_season = record.get("animeSeason") or {}
    if not isinstance(anime_season, dict):
        return None

    year = anime_season.get("year")
    try:
        year = int(year)
    except (TypeError, ValueError):
        return None

    if year < 1960 or year > current_year + 1:
        return None

    media_type = str(record.get("type") or "").strip().upper()
    if media_type not in ALLOWED_TYPES:
        return None

    status = str(record.get("status") or "").strip().upper()
    if status not in ALLOWED_STATUSES:
        return None

    item_id = source_item_id(record)
    synonyms = string_list(record.get("synonyms"))
    studios = string_list(record.get("studios"))
    producers = string_list(record.get("producers"))
    sources = string_list(record.get("sources"))
    related_anime = string_list(record.get("relatedAnime"))
    tags = string_list(record.get("tags"))

    episodes = record.get("episodes")
    try:
        episodes = int(episodes) if episodes is not None else None
    except (TypeError, ValueError):
        episodes = None

    duration_seconds = get_duration_seconds(record.get("duration"))
    score, score_agm, score_mean, score_median = get_score_fields(record.get("score"))

    season = anime_season.get("season")
    season = str(season).strip().upper() if season else None

    search_text = build_search_text(title, synonyms)
    text_blob = build_text_blob(
        title=title,
        media_type=media_type,
        status=status,
        year=year,
        season=season,
        episodes=episodes,
        duration_seconds=duration_seconds,
        score=score,
        synonyms=synonyms,
        studios=studios,
        producers=producers,
        tags=tags,
    )

    metadata_json = {
        "season": season,
        "year": year,
        "episodes": episodes,
        "duration": record.get("duration"),
        "synonyms": synonyms,
        "studios": studios,
        "producers": producers,
        "sources": sources,
        "relatedAnime": related_anime,
        "tags": tags,
        "score": record.get("score"),
        "picture": record.get("picture"),
        "thumbnail": record.get("thumbnail"),
    }

    return {
        "source_item_id": item_id,
        "title": title,
        "primary_title_normalized": normalize_title(title),
        "search_text": search_text,
        "text_blob": text_blob,
        "year": year,
        "season": season,
        "media_type": media_type,
        "status": status,
        "episodes": episodes,
        "duration_seconds": duration_seconds,
        "score": score,
        "score_arithmetic_geometric_mean": score_agm,
        "score_arithmetic_mean": score_mean,
        "score_median": score_median,
        "tags": ", ".join(tags[:8]) if tags else None,
        "tags_json": tags,
        "synonyms_json": synonyms,
        "studios_json": studios,
        "producers_json": producers,
        "sources_json": sources,
        "related_anime_json": related_anime,
        "source_provider_domains": source_provider_domains(sources),
        "related_anime_count": len(related_anime),
        "metadata_json": metadata_json,
        "raw_dataset_record_json": record,
        "image_original_url": record.get("picture"),
        "thumbnail_original_url": record.get("thumbnail"),
    }


def upsert_catalog_item(db: Session, row: dict, embedding: np.ndarray) -> None:
    """
    Insert or update one catalog item.
    """
    existing = db.scalar(select(CatalogItem).where(CatalogItem.source_item_id == row["source_item_id"]))

    values = {
        **row,
        "category": "anime",
        "source_name": "anime-offline-database",
        "embedding": embedding.astype(float).tolist(),
    }

    if existing:
        for key, value in values.items():
            setattr(existing, key, value)
    else:
        db.add(CatalogItem(**values))


def load_jsonl_records(raw_path: Path, current_year: int) -> list[dict]:
    rows: list[dict] = []

    with raw_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number}")
                continue

            # Official dataset JSONL uses the first line for metadata.
            # Anime records have title/type/status/animeSeason.
            if "title" not in raw:
                continue

            parsed = parse_record(raw, current_year)
            if parsed:
                rows.append(parsed)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Path to anime-offline-database.jsonl")
    parser.add_argument("--media-dir", default="../media", help="Local media output directory")
    parser.add_argument("--processed-output", default="../data/processed/catalog_summary.jsonl")
    parser.add_argument("--reset", action="store_true", help="Delete existing catalog_items before import")
    parser.add_argument("--skip-images", action="store_true", help="Generate placeholders instead of downloading images")
    parser.add_argument("--max-items", type=int, default=None, help="Optional development-only item limit")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers for image downloading")
    args = parser.parse_args()

    raw_path = Path(args.raw).resolve()
    media_dir = Path(args.media_dir).resolve()
    processed_output = Path(args.processed_output).resolve()
    processed_output.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {raw_path}")

    current_year = dt.datetime.now(dt.UTC).year
    rows = load_jsonl_records(raw_path, current_year)

    if args.max_items:
        rows = rows[: args.max_items]

    print(f"Retained {len(rows)} anime records after filtering.")

    client = httpx.Client(
        follow_redirects=True,
        timeout=10.0,
        limits=httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers),
        headers={"User-Agent": "AniSync educational project"},
    )

    def fetch_assets(row: dict) -> dict:
        assets = create_image_assets(
            client=client,
            item_id=row["source_item_id"],
            title=row["title"],
            thumbnail_url=None if args.skip_images else row.get("thumbnail_original_url"),
            picture_url=None if args.skip_images else row.get("image_original_url"),
            media_dir=media_dir,
        )
        row.update(assets)
        return row

    with processed_output.open("w", encoding="utf-8") as summary_file:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for row in tqdm(executor.map(fetch_assets, rows), total=len(rows), desc="Downloading images", unit="anime"):
                summary_file.write(json.dumps(
                    {
                        "source_item_id": row["source_item_id"],
                        "title": row["title"],
                        "year": row["year"],
                        "media_type": row["media_type"],
                        "status": row["status"],
                        "thumbnail_local_path": row["thumbnail_local_path"],
                    },
                    ensure_ascii=False,
                ) + "\n")

    text_blobs = [row["text_blob"] for row in rows]

    db = SessionLocal()
    try:
        if args.reset:
            print("Deleting existing catalog_items...")
            db.execute(delete(CatalogItem))
            db.commit()

        get_embedding_model()
        batches = range(0, len(rows), args.batch_size)
        for start in tqdm(batches, desc="Embedding & importing", unit="batch"):
            end = min(start + args.batch_size, len(rows))
            batch_rows = rows[start:end]
            batch_texts = text_blobs[start:end]

            embeddings = embed_texts(batch_texts, batch_size=args.batch_size)

            for row, embedding in zip(batch_rows, embeddings, strict=True):
                upsert_catalog_item(db, row, embedding)

            db.commit()

        print("Catalog preprocessing complete.")
    finally:
        db.close()
        client.close()


if __name__ == "__main__":
    main()
