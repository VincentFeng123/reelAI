"""Pydantic request/response models for the HTTP API."""
from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

# Accept watch URLs, youtu.be short links, shorts, embeds, and live links.
YT_RE = re.compile(
    r"(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|"
    r"youtube\.com/embed/|youtube\.com/live/|m\.youtube\.com/watch\?v=)",
    re.IGNORECASE,
)


class CreateJobReq(BaseModel):
    url: str
    topic: str
    settings: dict = {}

    @field_validator("url")
    @classmethod
    def _url(cls, v: str) -> str:
        if not YT_RE.search(v):
            raise ValueError("Not a recognized YouTube URL")
        return v.strip()

    @field_validator("topic")
    @classmethod
    def _topic(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Topic is required")
        return v.strip()


class CreateJobResp(BaseModel):
    job_id: str
