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

    # Proxy configuration for YouTube scraping.
    # Supports: direct proxy URLs, rotating proxy services (Bright Data, ScrapingBee),
    # and SOCKS5 proxies.
    # Format: "http://user:pass@proxy:port" or "socks5://user:pass@proxy:port"
    # Multiple proxies: comma-separated for rotation.
    proxy_urls: str = ""  # e.g. "http://user:pass@brd.superproxy.io:22225"
    # ScrapingBee API key (if using ScrapingBee as proxy service)
    scrapingbee_api_key: str = ""
    # Enable proxy for youtube_transcript_api calls
    proxy_transcripts: bool = True
    # Enable proxy for YouTube search HTML scraping
    proxy_search: bool = True

    # PO Token provider for yt-dlp. YouTube enforces PO Tokens on flagged
    # (cloud) IPs for video playback and some caption fetches. The plugin
    # `bgutil-ytdlp-pot-provider` (already in requirements.txt) talks to an
    # HTTP bgutil provider service to generate tokens. Leave blank to skip
    # the feature (yt-dlp falls back to non-PO-token flows, which may fail
    # on cloud IPs). To enable: deploy `brainicism/bgutil-ytdlp-pot-provider`
    # as a Railway sidecar on port 4416 and set this to the sidecar URL
    # (e.g. "http://bgutil-provider.railway.internal:4416"). yt_dlp_adapter
    # reads this and wires it into extractor_args automatically.
    ytdlp_pot_provider_url: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
