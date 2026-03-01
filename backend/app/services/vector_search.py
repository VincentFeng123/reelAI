import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def top_k_cosine(query_vec: np.ndarray, candidate_vecs: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
    if len(candidate_vecs) == 0:
        return []

    q = query_vec.astype(np.float32)
    c = candidate_vecs.astype(np.float32)

    if faiss is not None:
        index = faiss.IndexFlatIP(c.shape[1])
        index.add(c)
        scores, indices = index.search(q.reshape(1, -1), min(top_k, len(candidate_vecs)))
        return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0])) if indices[0][i] >= 0]

    sims = c @ q
    sorted_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in sorted_idx]
