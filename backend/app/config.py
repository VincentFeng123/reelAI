import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> str:
    # Vercel serverless functions can only write under /tmp.
    if os.getenv("VERCEL"):
        return "/tmp/studyreels-data"
    return "./data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_origin: str = "http://localhost:3000"
    data_dir: str = Field(default_factory=_default_data_dir)
    database_url: str = ""

    openai_enabled: bool = False
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    youtube_api_key: str = ""
    retrieval_engine_v2_enabled: bool = True
    retrieval_tier2_enabled: bool = False
    retrieval_debug_logging: bool = True

    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_endpoint_url: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    resend_api_key: str = ""
    verification_hmac_key: str = ""
    community_email_verification_required: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
