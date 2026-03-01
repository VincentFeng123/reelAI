import hashlib
import json
import math
from typing import Iterable

import numpy as np
from openai import OpenAI

from ..config import get_settings
from ..db import dumps_json, fetch_one, now_iso, upsert


class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_embed_model
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.dim = 1536 if settings.openai_api_key else 256

    def embed_texts(self, conn, texts: Iterable[str]) -> np.ndarray:
        text_list = [t.strip() for t in texts]
        if not text_list:
            return np.empty((0, self.dim), dtype=np.float32)

        hashes = [self._hash_text(t) for t in text_list]
        embeddings: list[np.ndarray | None] = [None] * len(text_list)
        missing_indices: list[int] = []

        for i, h in enumerate(hashes):
            row = fetch_one(conn, "SELECT embedding_json FROM embedding_cache WHERE text_hash = ?", (h,))
            if row:
                vec = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
                embeddings[i] = self._normalize(vec)
            else:
                missing_indices.append(i)

        if missing_indices:
            missing_texts = [text_list[i] for i in missing_indices]
            if self.client:
                fetched = self._embed_openai(missing_texts)
            else:
                fetched = self._embed_local(missing_texts)

            for local_i, global_i in enumerate(missing_indices):
                vec = self._normalize(fetched[local_i])
                embeddings[global_i] = vec
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

        result = np.vstack([e for e in embeddings if e is not None]).astype(np.float32)
        return result

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
