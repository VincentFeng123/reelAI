"""Hermetic test environment.

backend/app/config.py exports backend/.env into os.environ at import (the clip
engine reads os.environ directly). Tests must never see the real provider keys:
several suites rely on key-absence to take offline fallback paths, and a keyed
test process silently makes paid network calls (Supadata/Gemini/yt-dlp).

This conftest runs before any test module imports the app:
  * REELAI_SKIP_DOTENV disables the .env export in backend/app/config.py
  * any keys already present in the invoking shell are scrubbed
"""
import os

os.environ["REELAI_SKIP_DOTENV"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

for _key in (
    "SUPADATA_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "YOUTUBE_API_KEY",
    "CEREBRAS_API_KEY",
):
    os.environ.pop(_key, None)
