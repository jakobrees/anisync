import datetime as dt

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utcnow() -> dt.datetime:
    """Timezone-aware UTC timestamp helper."""
    return dt.datetime.now(dt.UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CatalogItem(Base):
    """
    Anime catalog table.

    This table intentionally stores both:
    1. structured columns for runtime filtering and display
    2. JSON snapshots for traceability and future extension

    The embedding column is pgvector vector(384).
    """

    __tablename__ = "catalog_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String(40), default="anime", nullable=False)
    source_name: Mapped[str] = mapped_column(String(80), default="anime-offline-database", nullable=False)
    source_item_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)

    title: Mapped[str] = mapped_column(Text, nullable=False)
    primary_title_normalized: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    text_blob: Mapped[str] = mapped_column(Text, nullable=False)

    year: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    season: Mapped[str | None] = mapped_column(String(40))
    media_type: Mapped[str] = mapped_column(String(40), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(40), index=True, nullable=False)

    episodes: Mapped[int | None] = mapped_column(Integer)
    duration_seconds: Mapped[int | None] = mapped_column(Integer)

    score: Mapped[float | None] = mapped_column(Float)
    score_arithmetic_geometric_mean: Mapped[float | None] = mapped_column(Float)
    score_arithmetic_mean: Mapped[float | None] = mapped_column(Float)
    score_median: Mapped[float | None] = mapped_column(Float)

    tags: Mapped[str | None] = mapped_column(Text)
    tags_json: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    synonyms_json: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    studios_json: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    producers_json: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    sources_json: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    related_anime_json: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    source_provider_domains: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    related_anime_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    embedding_msmarco: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)

    metadata_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    raw_dataset_record_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    image_original_url: Mapped[str | None] = mapped_column(Text)
    thumbnail_original_url: Mapped[str | None] = mapped_column(Text)
    image_local_path: Mapped[str | None] = mapped_column(Text)
    thumbnail_local_path: Mapped[str | None] = mapped_column(Text)
    image_download_status: Mapped[str] = mapped_column(String(80), default="missing", nullable=False)
    image_mime_type: Mapped[str | None] = mapped_column(String(80))
    image_width: Mapped[int | None] = mapped_column(Integer)
    image_height: Mapped[int | None] = mapped_column(Integer)
    image_sha256: Mapped[str | None] = mapped_column(String(128))

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(40), default="anime", nullable=False)

    host_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="open", index=True, nullable=False)

    state_revision: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_activity_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    hard_constraint_year_start: Mapped[int] = mapped_column(Integer, nullable=False)
    hard_constraint_year_end: Mapped[int] = mapped_column(Integer, nullable=False)
    hard_constraint_allowed_types_json: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)

    results_json: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class RoomMember(Base):
    __tablename__ = "room_members"
    __table_args__ = (UniqueConstraint("room_id", "user_id", name="uq_room_member"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    joined_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RoomPreferenceSubmission(Base):
    __tablename__ = "room_preference_submissions"
    __table_args__ = (UniqueConstraint("room_id", "user_id", name="uq_room_preference"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    submitted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class RoomVote(Base):
    __tablename__ = "room_votes"
    __table_args__ = (UniqueConstraint("room_id", "user_id", "catalog_item_id", name="uq_room_vote"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    catalog_item_id: Mapped[int] = mapped_column(ForeignKey("catalog_items.id", ondelete="CASCADE"), nullable=False)
    voted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
