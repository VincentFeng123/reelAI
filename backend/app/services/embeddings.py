import hashlib
import json
import logging
import threading
from typing import Iterable

import numpy as np

from ..db import dumps_json, fetch_one, now_iso, upsert

logger = logging.getLogger(__name__)


_HASH_DIM = 256
_semantic_inference_lock = threading.Lock()


class EmbeddingService:
    def __init__(self) -> None:
        # The production selector's whole-transcript Gemini verdict is the
        # semantic authority. These fixed-size lexical vectors only support
        # inexpensive persistence and deterministic ranking; they are never
        # accepted as semantic proof by callers.
        self._semantic_model = None
        self.dim = _HASH_DIM

    @property
    def semantic_available(self) -> bool:
        return self._semantic_model is not None

    @property
    def backend_name(self) -> str:
        return "hash-lexical-v1"

    def embed_texts(self, conn, texts: Iterable[str]) -> np.ndarray:
        text_list = [t.strip() for t in texts]
        if not text_list:
            return np.empty((0, self.dim), dtype=np.float32)

        if self._semantic_model is None:
            hash_embeddings = [self._hash_embed(text) for text in text_list]
            # Preserve the cache-miss path's second normalization exactly.
            return np.vstack([self._normalize(vec) for vec in hash_embeddings]).astype(np.float32)

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

    def embed_local(self, texts: Iterable[str]) -> np.ndarray:
        """Uncached embedding generation — used by scoring paths that don't
        need DB caching (e.g., the heuristic clip picker scoring hundreds of
        candidate windows per call where caching each would churn the table).
        """
        text_list = [str(t).strip() for t in texts]
        if not text_list:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._embed_local(text_list)

    def embed_semantic(self, texts: Iterable[str]) -> np.ndarray | None:
        """Return normalized sentence-transformer vectors, never hash vectors."""
        text_list = [str(text).strip() for text in texts]
        if not text_list or self._semantic_model is None:
            return None
        try:
            with _semantic_inference_lock:
                vectors = self._semantic_model.encode(  # type: ignore[attr-defined]
                    text_list,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
        except Exception:
            logger.exception("semantic embed raised; semantic proof unavailable")
            return None
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or vectors.shape[1] != self.dim:
            logger.warning("semantic model returned unexpected shape %s", getattr(vectors, "shape", None))
            return None
        return vectors.astype(np.float32)

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        semantic_vectors = self.embed_semantic(texts)
        if semantic_vectors is not None:
            return semantic_vectors
        if self._semantic_model is not None:
            # A loaded semantic backend that fails at inference time must fail
            # closed. Returning same-dimension hash vectors here would let
            # callers mistake them for cosine-semantic evidence.
            raise RuntimeError("Semantic embedding inference failed.")
        vectors = [self._hash_embed(t) for t in texts]
        return np.array(vectors, dtype=np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        # Hash to self.dim so this works both as the primary embedder
        # (self.dim == _HASH_DIM) and as a fallback when the semantic
        # embed raises mid-batch (self.dim == _SEMANTIC_DIM, vector is
        # mostly zeros but same shape — keeps vstack consistent).
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
