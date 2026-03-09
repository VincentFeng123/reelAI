import os
import re
import uuid
from abc import ABC, abstractmethod

import boto3

from ..config import get_settings


def _safe_filename(filename: str) -> str:
    base = os.path.basename(str(filename or "").strip()) or "upload"
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return clean[:120] or "upload"


class Storage(ABC):
    @abstractmethod
    def save_bytes(self, content: bytes, filename: str) -> str:
        raise NotImplementedError


class LocalStorage(Storage):
    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def save_bytes(self, content: bytes, filename: str) -> str:
        safe_name = f"{uuid.uuid4()}_{_safe_filename(filename)}"
        path = os.path.join(self.root, safe_name)
        with open(path, "wb") as f:
            f.write(content)
        return path


class S3Storage(Storage):
    def __init__(self) -> None:
        settings = get_settings()
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            region_name=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url or None,
            aws_access_key_id=settings.s3_access_key_id or None,
            aws_secret_access_key=settings.s3_secret_access_key or None,
        )

    def save_bytes(self, content: bytes, filename: str) -> str:
        key = f"uploads/{uuid.uuid4()}_{_safe_filename(filename)}"
        self.client.put_object(Bucket=self.bucket, Key=key, Body=content)
        return f"s3://{self.bucket}/{key}"


def get_storage() -> Storage:
    settings = get_settings()
    if settings.s3_bucket:
        return S3Storage()
    return LocalStorage(os.path.join(settings.data_dir, "uploads"))
