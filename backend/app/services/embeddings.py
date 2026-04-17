import hashlib
import json
import logging
from typing import Iterable

import numpy as np

from ..db import dumps_json, fetch_one, now_iso, upsert

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        # Hosted embedding providers have been removed; we use a pure-Python
        # hashing embedding that is deterministic, free, and fast. The
        # dimension is fixed at 256 so callers never need to re-hash on model
        # changes.
        self.dim = 256

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
            return self._normalize(vec), False
        return None, True

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
