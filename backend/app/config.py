import os
import tempfile
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export backend/.env into os.environ (anchored to this file, not the cwd).
# The clip engine reads os.environ directly — pydantic's env_file only fills
# the Settings object — so launchers that don't `source backend/.env`
# (host-up.sh, plain uvicorn) would otherwise run the engine key-less.
# override=False keeps explicitly exported values authoritative.
# REELAI_SKIP_DOTENV is set by the test conftest: tests must stay hermetic
# (no real keys in os.environ → offline fallback paths).
if not os.environ.get("REELAI_SKIP_DOTENV"):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def _default_data_dir() -> str:
    # Runtime artifacts must never be written into the source tree. Railway
    # deployments should set DATA_DIR to a mounted volume (for example /data).
    return str(Path(tempfile.gettempdir()) / "reelai-data")


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
    cerebras_api_key: str = ""
    # llama3.1-8b is the universally-available Cerebras free-tier model.
    # llama-3.3-70b requires explicit account provisioning and 404s on
    # most accounts; switch to gpt-oss-120b if you want a larger model.
    cerebras_model: str = "llama3.1-8b"
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

    # Clip engine configuration (Phase 2).
    clip_engine: str = "gemini"
    supadata_api_key: str = ""
    supadata_base: str = "https://api.supadata.ai/v1"
    segment_model: str = ""
    clip_search_max_videos: int = 5
    generation_job_heartbeat_sec: int = 15
    generation_job_lease_sec: int = 90
    generation_job_poll_sec: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
