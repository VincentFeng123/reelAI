"""Run-artifact persistence (W25-G): every assembled run leaves an auditable trail.

The qP coverage diagnosis had to RECONSTRUCT the shipped spans, plan and drop ledger
offline because jobs are in-memory (jobs.py) and assemble_clips' ledger was dropped at
the orchestrator boundary. write_run_artifacts persists the four assembly surfaces —
plan.json (extraction-plan proposals), arcs.json (post-verify arc survivors),
shipped.json (final specs) and ledger.json (the Rejection drop ledger) — under
``work/<video_id>/runs/<UTC ts>/`` from BOTH callers (orchestrator._run_full and
eval.run_eval). plan/arcs ride the caller-owned stats dict select_anchors_planned fills
('plan_proposals' / 'arcs_verified'; absent on the priority selector ⇒ empty files).
Defensive by construction: dataclasses/pydantic/sets/tuples all serialize, unknown
leaves degrade to str(), and the writer never raises into a pipeline (artifacts are
telemetry — a persistence problem must not fail a job or an eval run).
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ... import config


def _jsonable(x):
    """Best-effort JSON projection: dataclass → asdict, pydantic → model_dump,
    dict keys → str, set/tuple → list, NaN → null, unknown leaves → str(x). The
    artifact must always write; specs/rejections/proposals are never trusted to be
    JSON-clean (verdict objects, tuples of kinds, frozensets all ride along)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None              # NaN / ±inf → JSON null
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        try:
            return _jsonable(dataclasses.asdict(x))
        except Exception:                                   # non-deepcopyable field
            return str(x)
    if hasattr(x, "model_dump"):                            # pydantic (Unit/verdict/…)
        try:
            return _jsonable(x.model_dump())
        except Exception:
            return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    return str(x)


def write_run_artifacts(video_id: str, specs, rejections, stats,
                        work_dir: Optional[Path] = None) -> Optional[Path]:
    """Persist one assembled run's plan/arcs/shipped/ledger under
    ``<work_dir>/<video_id>/runs/<UTC timestamp>/`` and return the directory — or None
    on any failure (stderr-logged, never raised). Timestamps carry microseconds and a
    collision suffix so back-to-back eval runs (--runs N) never overwrite each other."""
    try:
        base = Path(work_dir) if work_dir is not None else config.WORK_DIR
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        d = base / video_id / "runs" / ts
        n = 0
        while d.exists():                                   # same-microsecond collision
            n += 1
            d = base / video_id / "runs" / f"{ts}-{n}"
        d.mkdir(parents=True)
        stats = stats or {}
        payloads = {
            "plan.json": stats.get("plan_proposals", []),
            "arcs.json": stats.get("arcs_verified", []),
            "shipped.json": specs or [],
            "ledger.json": rejections or [],
        }
        for name, payload in payloads.items():
            (d / name).write_text(json.dumps(_jsonable(payload), indent=2),
                                  encoding="utf-8")
        return d
    except Exception as e:  # noqa: BLE001 — telemetry only, never job-fatal
        print(f"[artifacts] failed to persist run artifacts for {video_id}: {e}",
              file=sys.stderr)
        return None
