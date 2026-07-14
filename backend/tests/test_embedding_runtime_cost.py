from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


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
