from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application settings.

    Values come from the root .env file during local development.
    In deployment, Render/Supabase/Vercel environment variables replace them.
    """

    database_url: str = "postgresql+psycopg://anisync:anisync@localhost:54329/anisync"
    session_secret: str = "dev-only-change-me"
    frontend_origin: str = "http://localhost:5173"

    cookie_secure: bool = False
    cookie_same_site: str = "lax"

    media_root: str = "../media"
    media_base_url: str = "/media"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    catalog_min_year: int = 1960
    catalog_max_year_buffer: int = 1

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_storage_bucket: str = "anisync-media"

    # Keep SQLAlchemy's pool intentionally small for Supabase Session Pooler.
    database_pool_size: int = 1
    database_max_overflow: int = 2
    database_pool_timeout: int = 10
    database_pool_recycle_seconds: int = 300

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        """
        Allow one or more frontend origins.
        Example:
        FRONTEND_ORIGIN=http://localhost:5173,https://your-app.vercel.app
        """
        return [origin.strip() for origin in self.frontend_origin.split(",") if origin.strip()]

    @property
    def resolved_media_root(self) -> Path:
        return Path(self.media_root).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
