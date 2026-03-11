import hashlib
import json
import math
import logging
import os
from typing import Iterable

import numpy as np

from ..config import get_settings
from ..db import dumps_json, fetch_one, now_iso, upsert
from .openai_client import build_openai_client

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_embed_model
        serverless_mode = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE"))
        allow_openai_serverless = os.getenv("ALLOW_OPENAI_IN_SERVERLESS") == "1"
        can_use_openai = (
            bool(settings.openai_enabled)
            and bool(settings.openai_api_key)
            and (not serverless_mode or allow_openai_serverless)
        )
        self.client = build_openai_client(
            api_key=settings.openai_api_key,
            timeout=8.0,
            enabled=can_use_openai,
        )
        self.dim = 1536 if self.client else 256

    def embed_texts(self, conn, texts: Iterable[str]) -> np.ndarray:
        text_list = [t.strip() for t in texts]
        if not text_list:
            return np.empty((0, self.dim), dtype=np.float32)

        hashes = [self._hash_text(t) for t in text_list]
        embeddings: list[np.ndarray | None] = [None] * len(text_list)
        missing_indices: list[int] = []
        persist_generated_indices: set[int] = set()

        for i, h in enumerate(hashes):
            row = fetch_one(conn, "SELECT embedding_json FROM embedding_cache WHERE text_hash = ?", (h,))
            if row:
                cached_vec, should_persist_replacement = self._load_cached_embedding(row.get("embedding_json"))
                if cached_vec is not None:
                    embeddings[i] = cached_vec
                else:
                    missing_indices.append(i)
                    if should_persist_replacement:
                        persist_generated_indices.add(i)
            else:
                missing_indices.append(i)
                persist_generated_indices.add(i)

        if missing_indices:
            missing_texts = [text_list[i] for i in missing_indices]
            if self.client:
                try:
                    fetched = self._embed_openai(missing_texts)
                except Exception as exc:
                    # Keep API endpoints alive even when OpenAI quota/model calls fail.
                    logger.warning("OpenAI embeddings failed; falling back to local embeddings: %s", exc)
                    fetched = self._embed_local(missing_texts)
            else:
                fetched = self._embed_local(missing_texts)

            for local_i, global_i in enumerate(missing_indices):
                vec = self._normalize(fetched[local_i])
                embeddings[global_i] = vec
                if global_i not in persist_generated_indices:
                    continue
                upsert(
                    conn,
                    "embedding_cache",
                    {
                        "text_hash": hashes[global_i],
                        "embedding_json": dumps_json(vec.tolist()),
                        "created_at": now_iso(),
                    },
                    pk="text_hash",
                )

        if any(embedding is None for embedding in embeddings):
            raise RuntimeError("Failed to build embeddings for every input text.")
        result = np.vstack([e for e in embeddings if e is not None]).astype(np.float32)
        return result

    def should_persist_replacement(self, raw_value: object) -> bool:
        _, should_persist_replacement = self._load_cached_embedding(raw_value)
        return should_persist_replacement

    def _load_cached_embedding(self, raw_value: object) -> tuple[np.ndarray | None, bool]:
        if raw_value in (None, ""):
            return None, True
        try:
            vec = np.array(json.loads(str(raw_value)), dtype=np.float32)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, True
        if vec.ndim != 1:
            return None, True
        if vec.size == self.dim:
            return self._normalize(vec), True
        # Dimension mismatch — always allow replacement so the cache stays usable.
        # When OpenAI is re-enabled, 256-dim entries will be replaced with 1536-dim ones.
        return None, True

    def _embed_openai(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return np.array(vectors, dtype=np.float32)

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        vectors = [self._hash_embed(t) for t in texts]
        return np.array(vectors, dtype=np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        return self._normalize(vec)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
