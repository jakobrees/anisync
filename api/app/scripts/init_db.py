from sqlalchemy import text

from app.db import Base, engine
from app import models  # noqa: F401


def main() -> None:
    """
    Initialize the database schema.

    Important:
    - pgvector extension must exist before creating vector(384) columns.
    - This script is safe to run multiple times.
    """
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)

    # Idempotent column additions — keeps existing databases in sync with the
    # current model without requiring a full drop/recreate.
    with engine.begin() as connection:
        connection.execute(text(
            "ALTER TABLE room_preference_submissions "
            "ADD COLUMN IF NOT EXISTS liked_catalog_item_ids JSONB NOT NULL DEFAULT '[]'::jsonb"
        ))

    print("Database initialized successfully.")


if __name__ == "__main__":
    main()
