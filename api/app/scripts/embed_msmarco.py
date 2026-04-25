"""
Populate the embedding_msmarco column on catalog_items.

Uses msmarco-MiniLM-L6-cos-v5 — an asymmetric model fine-tuned on MS MARCO
query→passage pairs, better suited for matching natural language preference
queries against anime metadata than the symmetric all-MiniLM-L6-v2.

This script is safe to run multiple times: it skips rows that already have
embedding_msmarco populated.

Run:
  cd api && python -m app.scripts.embed_msmarco
  cd api && python -m app.scripts.embed_msmarco --batch-size 128
  cd api && python -m app.scripts.embed_msmarco --max-items 100   # smoke test
"""
import argparse

from sqlalchemy import select, text, update
from tqdm import tqdm

from app.db import SessionLocal, engine
from app.embeddings import MSMARCO_MODEL_NAME, embed_texts_msmarco, get_msmarco_model
from app.models import CatalogItem


def ensure_column_exists() -> None:
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE catalog_items "
            "ADD COLUMN IF NOT EXISTS embedding_msmarco vector(384)"
        ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=None, help="Development limit")
    args = parser.parse_args()

    ensure_column_exists()

    db = SessionLocal()
    try:
        # Load all (id, text_blob) pairs upfront — no embeddings, no ORM overhead.
        rows = db.execute(
            select(CatalogItem.id, CatalogItem.text_blob)
            .where(CatalogItem.embedding_msmarco.is_(None))
            .order_by(CatalogItem.id)
        ).all()

        if args.max_items:
            rows = rows[: args.max_items]

        total = len(rows)
        print(f"Items needing embedding_msmarco: {total}")
        if total == 0:
            print("Nothing to do.")
            return

        print("Loading msmarco model (may download on first run)...")
        get_msmarco_model()

        batches = range(0, total, args.batch_size)
        for start in tqdm(batches, desc="Embedding & importing", unit="batch"):
            batch = rows[start: start + args.batch_size]
            ids = [r.id for r in batch]
            texts = [r.text_blob for r in batch]

            embeddings = embed_texts_msmarco(texts, batch_size=args.batch_size)

            for item_id, emb in zip(ids, embeddings):
                db.execute(
                    update(CatalogItem)
                    .where(CatalogItem.id == item_id)
                    .values(embedding_msmarco=emb.astype(float).tolist())
                )

            db.commit()

        print(f"Done. Embedded {total} items with {MSMARCO_MODEL_NAME}.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
