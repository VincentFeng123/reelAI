import hashlib
import json
import logging
import os
import threading
from typing import Iterable

import numpy as np

from ..db import dumps_json, fetch_one, now_iso, upsert

logger = logging.getLogger(__name__)


# Sentence-transformers model id. all-MiniLM-L6-v2 is ~90MB, 384-dim, English-
# biased, and near-SOTA for its size. Trade-off vs. hash embeddings: real
# semantic similarity ("ML" ≈ "machine learning") at the cost of ~5s cold
# start and pytorch-in-the-image. Switched on Railway; serverless (Vercel)
# auto-degrades to the hash path because pytorch cold-starts are too slow
# for function timeouts and the model isn't in that requirements.txt.
_SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_SEMANTIC_DIM = 384
_HASH_DIM = 256


def _serverless_mode() -> bool:
    return bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE"))


# Singleton model handle — lazy-loaded on first use, None means "we already
# tried and the model path is unavailable; never retry this process".
_semantic_model_lock = threading.Lock()
_semantic_model: object | None = None
_semantic_model_tried = False


def _get_semantic_model() -> object | None:
    """Lazy-load sentence-transformers. Returns None when unavailable so
    EmbeddingService degrades to hash embeddings without raising.
    """
    global _semantic_model, _semantic_model_tried
    if _semantic_model_tried:
        return _semantic_model
    if _serverless_mode():
        _semantic_model_tried = True
        return None
    with _semantic_model_lock:
        if _semantic_model_tried:
            return _semantic_model
        _semantic_model_tried = True
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            logger.info("sentence-transformers not available, using hash embeddings: %s", exc)
            return None
        try:
            model = SentenceTransformer(_SEMANTIC_MODEL_ID, device="cpu")
        except Exception:
            logger.exception("could not load %s, using hash embeddings", _SEMANTIC_MODEL_ID)
            return None
        _semantic_model = model
        logger.info("loaded semantic embedding model %s (dim=%d)", _SEMANTIC_MODEL_ID, _SEMANTIC_DIM)
        return _semantic_model


class EmbeddingService:
    def __init__(self) -> None:
        # Dimension is decided at init time, keyed on which backend we can
        # actually load. The cache layer (_load_cached_embedding) compares
        # stored vectors against self.dim and invalidates on mismatch — so
        # rolling from hash (256) → semantic (384) simply expires old rows.
        model = _get_semantic_model()
        if model is None:
            self._semantic_model = None
            self.dim = _HASH_DIM
        else:
            self._semantic_model = model
            self.dim = _SEMANTIC_DIM

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
        if self._semantic_model is not None:
            try:
                # encode() returns np.ndarray; normalize for cosine-sim usage
                # downstream. batch_size and convert_to_numpy defaults are fine.
                vecs = self._semantic_model.encode(  # type: ignore[attr-defined]
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                if isinstance(vecs, np.ndarray) and vecs.ndim == 2 and vecs.shape[1] == self.dim:
                    return vecs.astype(np.float32)
                logger.warning(
                    "semantic model returned unexpected shape %s; falling through to hash embed",
                    getattr(vecs, "shape", None),
                )
            except Exception:
                logger.exception("semantic embed raised; falling back to hash embed for this batch")
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
