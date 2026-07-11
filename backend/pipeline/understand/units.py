"""Stage: atomic discourse units (spec §3, §4 role labeling, part of §5 concept extraction).

One LLM call per topic splits it into the smallest self-contained communicative actions and,
in the same pass, assigns each a role (from the adapter's universal+domain menu) and extracts
its summary, claims, concepts introduced/required, equations, and textual back-references.
Doing role+concepts in the unit pass keeps each unit's label grounded in its local context and
avoids extra whole-video passes. Reference resolution to source units happens in dependencies.py.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ... import config
from ...roles import ALL_UNIVERSAL_ROLES, coerce_role
from ..select import render_sentences
from ..sentences import Sentence
from .models import ContentMap, Reference, Unit, VisualDependency

ProgressCb = Optional[Callable[[float, str], None]]
_UNIVERSAL = frozenset(ALL_UNIVERSAL_ROLES)


class RefLLM(BaseModel):
    text: str = ""
    resolves_to: str = ""


class VisDepLLM(BaseModel):
    kind: str = "slide"                            # slide|equation|diagram|code|chart|demo
    on_screen_text: str = ""                       # verbatim, if the unit points at on-screen text
    description: str = ""                           # short description of the referenced visual


class UnitLLM(BaseModel):
    sentence_start: int = 0
    sentence_end: int = 0
    role: str = "explanation"
    summary: str = ""
    claims: list[str] = Field(default_factory=list)
    concepts_introduced: list[str] = Field(default_factory=list)
    concepts_required: list[str] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    references: list[RefLLM] = Field(default_factory=list)
    visual_dependencies: list[VisDepLLM] = Field(default_factory=list)


class UnitsLLM(BaseModel):
    units: list[UnitLLM] = Field(default_factory=list)


# NOTE (graph nutrition, Q2): prompt-only change — cached structures are unaffected and the
# persisted schema is unchanged, so SCHEMA_VERSION is deliberately NOT bumped; the added
# concepts_introduced instruction only improves future builds.
_SYSTEM_TMPL = """You segment ONE topic of a video into ATOMIC discourse units and label each.

An atomic unit is the SMALLEST self-contained communicative action — a single definition, one \
claim, one worked step, one demonstration, one question, one answer. Never merge two distinct \
actions; never split a single action.

For each unit return:
- sentence_start, sentence_end: the unit's first and last sentence INDEX (from the [index] tags).
- role: exactly ONE role from this menu:
{role_menu}
- summary: one sentence describing what the unit does.
- claims: any factual claims/results asserted (may be empty).
- concepts_introduced: concepts/terms/equations this unit introduces or defines. For units whose \
role is definition, equation_introduction, or variable_definition, ALWAYS name the concept(s) \
being introduced (e.g. a unit defining momentum -> concepts_introduced: ["momentum"]).
- concepts_required: concepts the unit assumes the viewer already knows (its prerequisites).
- equations: any equations/formulas stated (verbatim, may be empty).
- references: back-references like "this", "that equation", "as we saw" with what each resolves to.
{visual_field}{hints}
Units must be in order, non-overlapping, and stay within the given index range. Output only the \
structured result."""


_VISUAL_FIELD = (
    "- visual_dependencies: if the unit points at something shown on screen (an equation, diagram, "
    "code, chart, or slide listed in the ON-SCREEN block), add one giving its kind and its "
    "on_screen_text VERBATIM (or a short description). Omit when nothing on screen is referenced.\n"
)


def _text_sim(a: str, b: str) -> float:
    try:
        from rapidfuzz.fuzz import partial_ratio
        return partial_ratio(a, b) / 100.0
    except Exception:
        return 1.0 if (a and (a in b or b in a)) else 0.0


def _link_visual(dep, unit_start: float, unit_end: float, perception):
    """Best on-screen ``VisualEvent`` a declared visual dependency refers to (or None)."""
    if not perception:
        return None
    dep_text = (getattr(dep, "on_screen_text", "") or getattr(dep, "description", "") or "").strip().lower()
    dep_kind = (getattr(dep, "kind", "") or "").strip().lower()
    best, best_score = None, 0.0
    for ve in perception.visual_events:
        if ve.end < unit_start - 5 or ve.start > unit_end + 5:
            continue
        score = 0.4                                        # any overlapping on-screen event is a candidate
        if dep_kind and ve.kind and dep_kind == ve.kind.strip().lower():
            score += 0.3
        if dep_text and ve.text:
            score += 0.3 * _text_sim(dep_text, ve.text.lower())
        if score > best_score:
            best, best_score = ve, score
    return best if (best is not None and best_score >= 0.4) else None


def _resolve_role(raw: str, valid: frozenset[str]) -> tuple[str, str]:
    r = (raw or "").strip().lower().replace(" ", "_")
    if r in valid:
        return (r, "") if r in _UNIVERSAL else (r, r)   # domain role: role==role_domain
    return coerce_role(r), ""


def extract_units(sentences: list[Sentence], content_map: ContentMap, adapter,
                  settings: dict, progress: ProgressCb = None, perception=None) -> list[Unit]:
    from ...llm import llm_json
    n = len(sentences)
    if n == 0:
        return []
    valid = adapter.valid_roles()
    has_visual = bool(perception and perception.visual_events)
    # a perception with no visual_events (vision degraded / visual-free video) must behave
    # exactly like perception=None: no visual deps, source_confidence stays 1.0 (Phase-1 invariant).
    vperc = perception if has_visual else None
    hints = "\n".join(h for h in (adapter.labeling_hints(), adapter.concept_hints()) if h)
    system = _SYSTEM_TMPL.format(
        role_menu=adapter.role_menu(),
        visual_field=(_VISUAL_FIELD if has_visual else ""),
        hints=(hints + "\n") if hints else "",
    )

    topics = content_map.topics() or [content_map.nodes[0]]

    # ── PASS A: independent per-topic LLM calls, fired CONCURRENTLY ───────────────
    # Each topic's UnitsLLM segmentation is independent of every other topic's, so the
    # network calls (the 40%-of-cold-latency serial bottleneck) run over a thread pool.
    # A worker builds its topic's prompt EXACTLY as the serial path did, and returns the
    # UnitsLLM response TAGGED with its topic index. Workers NEVER assign unit_ids, mutate
    # the shared units list, or touch the counter/cursor — all id/ordering-sensitive work
    # stays in the serial PASS B below, so the built units are output-neutral to worker
    # count. UNDERSTAND_WORKERS=1 (or a single topic) takes the exact serial path.
    def _fetch(node) -> Optional[UnitsLLM]:
        a, b = node.sentence_range
        a, b = max(0, a), min(n - 1, b)
        if b < a:
            return None
        rendered = render_sentences(sentences[a:b + 1])
        vis = ""
        if has_visual:
            span0, span1 = float(sentences[a].start), float(sentences[b].end)
            vis_events = [ve for ve in perception.visual_events if ve.end >= span0 and ve.start <= span1]
            vis = "\n".join(f"[{ve.kind} @ {ve.start:.0f}s] {(ve.text or ve.description)[:160]}"
                            for ve in vis_events)[:1500]
        user = (f"TOPIC: {node.title}\n\n"
                + (f"ON-SCREEN DURING THIS TOPIC (read from the video frames):\n{vis}\n\n" if vis else "")
                + f"TRANSCRIPT (each line: [index] (start-end seconds) text), indices {a}–{b}:\n{rendered}\n\n"
                + f"Segment indices {a}–{b} into atomic units.")
        try:
            return llm_json(system, user, UnitsLLM, temperature=0.1)
        except Exception:
            return UnitsLLM(units=[UnitLLM(sentence_start=a, sentence_end=b, role="explanation",
                                           summary=node.summary or node.title)])

    workers = max(1, min(config.UNDERSTAND_WORKERS, len(topics)))
    responses: dict[int, UnitsLLM] = {}                    # topic-index-keyed → order-independent
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_fetch, node): ti for ti, node in enumerate(topics)}
            for fut in as_completed(futs):                 # completion order is irrelevant:
                res = fut.result()                         # each result is filed by its topic index
                if res is not None:
                    responses[futs[fut]] = res
    else:
        for ti, node in enumerate(topics):                 # WORKERS=1 → exact serial fetch path
            res = _fetch(node)
            if res is not None:
                responses[ti] = res

    # ── PASS B: deterministic post-processing, SERIAL over ORIGINAL topic order ───
    # Unchanged from the serial build: per-topic non-overlap cursor clamp, the global
    # unit_id counter, boundary_clamped warnings, and the MAX_UNITS truncation.
    units: list[Unit] = []
    counter = 0
    for ti, node in enumerate(topics):
        a, b = node.sentence_range
        a, b = max(0, a), min(n - 1, b)
        if b < a:
            continue
        res = responses[ti]
        cursor = a
        for u in res.units:
            s0 = max(a, min(b, int(u.sentence_start)))
            s1 = max(a, min(b, int(u.sentence_end)))
            if s1 < s0:
                s0, s1 = s1, s0
            # W25-B drift telemetry: the non-overlap clamp below silently SHIFTS a unit's
            # start (and cascades into every later unit in the topic) whenever the LLM
            # double-claims a sentence — record it on the unit so build_structure can roll
            # the drift into Structure.degraded instead of shipping mislabeled boundaries
            # silently (Unit.warnings existed but was never written).
            u_warnings: list[str] = ["boundary_clamped"] if cursor > s0 else []
            s0 = max(s0, cursor)                       # keep non-overlapping within topic
            if s0 > b:
                break
            s1 = max(s1, s0)
            role, role_domain = _resolve_role(u.role, valid)
            text = " ".join((sentences[i].text or "") for i in range(s0, s1 + 1)).strip()
            u_start, u_end = float(sentences[s0].start), float(sentences[s1].end)
            vdeps = _build_visual_deps(getattr(u, "visual_dependencies", None), u_start, u_end, vperc)
            units.append(Unit(
                unit_id=f"u{counter:04d}", start=u_start, end=u_end,
                sentence_range=(s0, s1), node_id=node.node_id, topic=node.title,
                role=role, role_domain=role_domain, summary=u.summary or "", transcript=text,
                claims=[c for c in u.claims if c], equations=[e for e in u.equations if e],
                concepts_introduced=[c.strip().lower() for c in u.concepts_introduced if c.strip()],
                concepts_required=[c.strip().lower() for c in u.concepts_required if c.strip()],
                references=[Reference(text=r.text, resolves_to=r.resolves_to) for r in u.references if r.text],
                visual_dependencies=vdeps,
                speaker=_unit_speaker(u_start, u_end, perception),
                source_confidence=_source_confidence(vdeps, vperc),
                warnings=u_warnings,
            ))
            counter += 1
            cursor = s1 + 1
            if counter >= config.MAX_UNITS:
                break
        if progress:
            progress((ti + 1) / len(topics), f"Units {ti + 1}/{len(topics)} topics ({counter})")
        if counter >= config.MAX_UNITS:
            break
    return units


def drift_stats(units: list[Unit], content_map: ContentMap, n_sentences: int) -> tuple[int, int]:
    """W25-B drift telemetry: (n_clamped, n_uncovered) over the extracted units.

    ``n_clamped`` counts units whose start was moved by the non-overlap clamp (the
    'boundary_clamped' warning above); ``n_uncovered`` counts sentence indices inside the
    unit-bearing topic partition that NO unit covers — units are supposed to partition each
    topic, and gaps (23 uncovered indices on qP) are the other face of the same LLM
    double-claim drift. Mirrors extract_units' partition exactly (same topic fallback,
    same range clamping) so the expectation matches what extraction was actually asked for."""
    n_clamped = sum(1 for u in units if "boundary_clamped" in u.warnings)
    covered: set[int] = set()
    for u in units:
        s0, s1 = u.sentence_range
        covered.update(range(s0, s1 + 1))
    expected: set[int] = set()
    for node in (content_map.topics() or content_map.nodes[:1]):
        a, b = node.sentence_range
        a, b = max(0, a), min(n_sentences - 1, b)
        if b >= a:
            expected.update(range(a, b + 1))
    return n_clamped, len(expected - covered)


def _build_visual_deps(llm_deps, unit_start: float, unit_end: float, perception) -> list[VisualDependency]:
    """Turn LLM-declared visual deps into linked ``VisualDependency`` objects (empty w/o perception)."""
    if not perception or not llm_deps:
        return []
    out: list[VisualDependency] = []
    for vd in llm_deps:
        dep = VisualDependency(
            kind=(getattr(vd, "kind", "") or "slide").strip().lower(),
            description=(getattr(vd, "description", "") or getattr(vd, "on_screen_text", "") or "").strip(),
        )
        ev = _link_visual(vd, unit_start, unit_end, perception)
        if ev is not None:
            dep.visual_event_id = ev.event_id
            dep.keyframe_time = ev.start
        out.append(dep)
    return out


def _source_confidence(vdeps: list[VisualDependency], perception) -> float:
    """1.0 for transcript-only units; a declared-but-unconfirmed visual lowers it slightly."""
    if not perception or not vdeps:
        return 1.0
    return 1.0 if any(d.visual_event_id for d in vdeps) else 0.85


def _unit_speaker(unit_start: float, unit_end: float, perception):
    """Dominant diarized speaker over the unit (None when diarization is off/empty)."""
    turns = getattr(perception, "diarization", None) if perception else None
    if not turns:
        return None
    from . import diarize
    return diarize.assign_speaker(unit_start, unit_end, turns)
