from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from backend.app.services import embeddings as embeddings_module


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_clean_process(script: str) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_main_import_uses_no_torch_embedding_backend() -> None:
    result = _run_clean_process(
        "import json, sys; import backend.app.main as main; "
        "print(json.dumps({'backend': main.embedding_service.backend_name, "
        "'dim': main.embedding_service.dim, 'torch': 'torch' in sys.modules, "
        "'sentence_transformers': 'sentence_transformers' in sys.modules}))"
    )

    assert result == {
        "backend": "hash-lexical-v1",
        "dim": 256,
        "torch": False,
        "sentence_transformers": False,
    }


def test_embedding_request_cannot_lazy_load_torch() -> None:
    result = _run_clean_process(
        "import json, sys; from backend.app.services.embeddings import EmbeddingService; "
        "service = EmbeddingService(); vectors = service.embed_local(['chain rule example']); "
        "print(json.dumps({'shape': list(vectors.shape), 'torch': 'torch' in sys.modules, "
        "'sentence_transformers': 'sentence_transformers' in sys.modules, "
        "'semantic_available': service.semantic_available}))"
    )

    assert result == {
        "shape": [1, 256],
        "torch": False,
        "sentence_transformers": False,
        "semantic_available": False,
    }


def test_railway_requirements_do_not_install_torch_embedding_stack() -> None:
    requirements = (REPO_ROOT / "backend" / "requirements.txt").read_text().casefold()

    assert "sentence-transformers" not in requirements
    assert "\ntorch" not in requirements


def test_railway_deployment_is_pinned_to_one_replica() -> None:
    config = json.loads((REPO_ROOT / "railway.json").read_text())

    assert config["deploy"]["numReplicas"] == 1


def test_hash_embed_texts_bypasses_cache_and_matches_prior_miss_path(monkeypatch) -> None:
    service = embeddings_module.EmbeddingService()
    texts = ["  chain rule example  ", "one one two three four five"]
    stripped = [text.strip() for text in texts]
    generated = service._embed_local(stripped)
    expected = np.vstack([service._normalize(vec) for vec in generated]).astype(np.float32)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("hash embeddings must not touch embedding_cache")

    monkeypatch.setattr(embeddings_module, "fetch_one", fail_if_called)
    monkeypatch.setattr(embeddings_module, "upsert", fail_if_called)

    actual = service.embed_texts(object(), texts)

    np.testing.assert_array_equal(actual, expected)


def test_semantic_embed_texts_keeps_cache_reads_and_writes(monkeypatch) -> None:
    class FakeSemanticModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts, **_kwargs):
            self.calls.append(list(texts))
            vectors = np.zeros((len(texts), 256), dtype=np.float32)
            vectors[:, 3] = 2.0
            return vectors

    service = embeddings_module.EmbeddingService()
    model = FakeSemanticModel()
    service._semantic_model = model
    cached = np.zeros(service.dim, dtype=np.float32)
    cached[0] = 5.0
    cached_hash = service._hash_text("cached text")
    fetch_calls: list[str] = []
    upsert_calls: list[tuple[object, str, dict[str, object], str]] = []

    def fake_fetch_one(conn, query, params):
        assert query == "SELECT embedding_json FROM embedding_cache WHERE text_hash = ?"
        fetch_calls.append(params[0])
        if params[0] == cached_hash:
            return {"embedding_json": json.dumps(cached.tolist())}
        return None

    def fake_upsert(conn, table, data, *, pk):
        upsert_calls.append((conn, table, data, pk))

    monkeypatch.setattr(embeddings_module, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(embeddings_module, "upsert", fake_upsert)
    conn = object()

    actual = service.embed_texts(conn, ["cached text", "semantic miss"])

    expected = np.zeros((2, service.dim), dtype=np.float32)
    expected[0, 0] = 1.0
    expected[1, 3] = 1.0
    np.testing.assert_array_equal(actual, expected)
    assert fetch_calls == [cached_hash, service._hash_text("semantic miss")]
    assert model.calls == [["semantic miss"]]
    assert len(upsert_calls) == 1
    assert upsert_calls[0][0] is conn
    assert upsert_calls[0][1] == "embedding_cache"
    assert upsert_calls[0][2]["text_hash"] == service._hash_text("semantic miss")
    assert upsert_calls[0][3] == "text_hash"
