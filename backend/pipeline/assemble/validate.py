"""Clip-only judge + repair loop (spec §8, P2 repair rework).

A separate judge sees ONLY a candidate clip's transcript (+ any on-screen summary) and
decides whether a new viewer could understand it in isolation. The candidate is judged at
its MINIMAL post-closure size FIRST (P2a — never inflated before its first verdict); the
repair loop then routes by verdict: missing-content failures (unresolved reference /
missing prerequisite / problem / reasoning / result / visual) pull the specific missing
units from the graph and GROW the span, while off-topic / coherence failures TRIM the
units temporally farthest from the anchor (never the anchor, never contract-required
units) via bisection of the trim lattice between last-known-good and the failing span
(P2b/P2c). The budget is JUDGE_MAX_REPAIR+1 NEW judgments — text-hash verdict-cache hits
are free. If it still fails, the terminal gate is asymmetric: a rejection requires at
least one failure reason whose evidence_quote passes deterministic containment against
the clip text AND survives a fresh-context confirmation call — otherwise the clip ships
flagged ('unverified_judge_concerns'). The verdict schema doubles as the eval
comprehension scorer.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from ... import config
from ..sentences import Sentence
from ..understand.models import Unit
from .contracts import check_contract, choose_contract
from .types import Candidate


class FailureReason(BaseModel):
    kind: str = "other"     # unresolved_reference|missing_prerequisite|missing_visual|
                            # missing_problem_statement|missing_reasoning|missing_result|
                            # not_source_grounded|off_topic|over_inclusion|other
    detail: str = ""
    missing_concept: Optional[str] = None
    reference_text: Optional[str] = None
    approx_time: Optional[float] = None
    evidence_quote: str = ""    # EXACT contiguous clip-transcript words demonstrating the problem


class JudgeVerdict(BaseModel):
    reasoning: str = ""      # 2-3 sentence CoT the judge writes BEFORE the verdict fields
    understandable: bool = False
    score: float = 0.0       # normalized 0-1; derived from score_10 when the judge emits it
    score_10: int = 0        # judge emits an integer 1-10; 0 = not emitted (legacy/fallback)
    topic_identifiable: bool = True
    purpose_identifiable: bool = True
    all_references_resolved: bool = True
    prerequisites_satisfied: bool = True
    visuals_sufficient: bool = True
    problem_statement_complete: bool = True
    reasoning_complete: bool = True
    result_complete: bool = True
    source_grounded: bool = True
    opening_in_context: bool = True   # does the FIRST sentence stand on its own (not mid-thought
                                      # / not at the answer before the question is posed)?
    # W25-E atomicity: ADVISORY this wave — deliberately NOT in _hard_core_ok or
    # required_verdict_fields (no calibration labels yet); False without a VERIFIED
    # over_inclusion-family reason ⇒ ship warning only, never a gate or kill.
    single_idea: bool = True
    failure_reasons: list[FailureReason] = Field(default_factory=list)
    error: bool = False      # judge call failed (set ONLY by the fallback path, never the LLM)
    # deterministic post-hoc verification (never part of the LLM output schema): one bool per
    # failure_reason — did its evidence_quote pass normalized containment against the clip text?
    _reason_verified: Optional[list[bool]] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _normalize_score(self):
        if self.score_10 != 0:     # 0 = sentinel "not emitted"; any other value gets clamped
            self.score = min(max(self.score_10, 1), 10) / 10.0
        return self


JUDGE_SYSTEM = (
    "You are a strict judge of whether a short video clip is SELF-CONTAINED. You see ONLY the "
    "clip's transcript (and any on-screen text), never the surrounding video. Decide whether a "
    "brand-new viewer, watching only this clip, could understand what it is about and follow it "
    "to a complete thought.\n"
    "Evaluate in four steps:\n"
    "1. Identify what the clip is about and why it matters (topic and purpose).\n"
    "2. Hunt for dangling references — 'this', 'that', 'the previous equation' with no antecedent "
    "inside the clip — and for concepts the clip assumes but never introduces (and that are not "
    "common knowledge).\n"
    "3. If the clip is a worked problem, check the question is stated, the reasoning shown, and "
    "the result reached.\n"
    "4. Check the clip is ONE idea: does it contain MORE than one self-contained idea (e.g. a "
    "second, separate problem or topic after the first completes) or material content that "
    "belongs to a different idea?\n"
    "If a CONTEXT CARD is provided, it is shown to the viewer immediately BEFORE the clip; any "
    "concept it introduces or reference it resolves is available to the viewer — such concepts "
    "satisfy prerequisites_satisfied and such references satisfy all_references_resolved.\n"
    "Score bands for score_10 (integer 1-10):\n"
    "1-2: incomprehensible without the source video. 3-4: major gaps — the topic is unclear OR a "
    "key reference/prerequisite is missing. 5-6: partially followable — topic clear but real gaps "
    "remain. 7-8: fully understandable with minor rough edges. 9-10: flawlessly self-contained.\n"
    "First write `reasoning`: 2-3 sentences applying the steps to THIS clip. Then set every "
    "boolean truthfully and score_10 per the bands:\n"
    "- topic_identifiable / purpose_identifiable: is it clear what this is about and why?\n"
    "- all_references_resolved: no dangling 'this/that/the previous equation' without an antecedent.\n"
    "- prerequisites_satisfied: it doesn't assume a concept it never introduces or that isn't common.\n"
    "- visuals_sufficient: you cannot see the frames — set TRUE unless the transcript explicitly leans "
    "on an unshown, undescribed visual that is essential to follow it.\n"
    "- problem_statement_complete / reasoning_complete / result_complete: for a worked problem, is the "
    "question stated, the reasoning shown, and the result reached? (Set true/NA if not a problem.)\n"
    "- source_grounded: the clip stands on its own words (not obviously mid-argument).\n"
    "- opening_in_context: does the FIRST sentence open the thought — introducing its own "
    "subject/problem/equation/question — rather than continuing one ('and then…', 'so the "
    "answer is…') or referring back ('this', 'the answer') to material shown before the clip? "
    "A CONTEXT CARD does NOT satisfy this — the spoken opening itself must stand on its own. "
    "If FALSE, add a failure_reason kind 'not_source_grounded' quoting the mid-thought opener.\n"
    "- single_idea: FALSE when the clip contains more than one self-contained idea or material "
    "off-idea content (step 4); add an over_inclusion failure_reason with the evidence.\n"
    "Give an overall 'understandable' boolean. For each problem add a failure_reason whose 'kind' "
    "is EXACTLY one of: unresolved_reference, missing_prerequisite, missing_visual, "
    "missing_problem_statement, missing_reasoning, missing_result, not_source_grounded, off_topic, "
    "over_inclusion, other — and set missing_concept (for a prerequisite) or reference_text (the "
    "dangling phrase) when relevant. For EVERY failure_reason you MUST set evidence_quote to the "
    "EXACT contiguous words copied verbatim from the clip transcript that demonstrate the problem "
    "— the dangling phrase, the sentence that assumes a concept never introduced, the incomplete "
    "final sentence, or (for over_inclusion) the opening sentence of the SECOND idea. "
    "If you cannot quote transcript text that demonstrates the problem, do NOT emit that "
    "failure_reason. Never set 'error' or 'score' yourself; emit score_10 only. "
    "Output only the structured result."
)


def _clip_text(sentences: list[Sentence], i0: int, i1: int) -> str:
    return " ".join((sentences[i].text or "") for i in range(i0, i1 + 1)).strip()


def judged_text_hash(text: str) -> str:
    """Stable hash of the EXACT text a verdict was issued on (normalized: lowercase +
    collapsed whitespace). Verdicts are only trustworthy for the text the judge actually
    saw — any post-judge mutation (merge, min-duration extension, trim, cap) changes this
    hash, forcing a verdict-cache miss and a post-snap re-judge, while byte-identical
    convergent candidates still share cache entries."""
    norm = " ".join((text or "").lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def judge_clip(clip_text: str, role: str, adapter, visual_summary: str = "",
               topic: str = "", context_card: str = "") -> JudgeVerdict:
    """Judge a clip's self-containedness from ITS OWN text only. Shared by the repair loop
    (Part B) and the eval comprehension metric — same measurement in both places."""
    from ...llm import llm_json
    required = adapter.required_elements(role) if adapter else []
    user = (
        f"TOPIC: {topic or '(general)'}\n"
        f"CLIP ROLE: {role or '(unspecified)'}"
        + (f" (a complete one needs: {', '.join(required)})" if required else "") + "\n"
        + (f"CONTEXT CARD (shown before the clip): {context_card}\n" if context_card else "")
        + f"ON-SCREEN TEXT: {visual_summary or '(none available)'}\n\n"
        f"CLIP TRANSCRIPT:\n{clip_text}\n\n"
        "Judge whether this clip is self-contained."
    )
    # judge on a different model/provider than the authoring passes (self-preference bias)
    jp = config.JUDGE_PROVIDER
    provider = None if jp in ("", "same") else jp
    jm = config.JUDGE_MODEL
    model = jm if (jm and jm != config.GEMINI_MODEL) else None
    try:
        v = llm_json(JUDGE_SYSTEM, user, JudgeVerdict, temperature=0.0, provider=provider, model=model)
        v.error = False                      # the LLM can never self-flag a transport error
        return v
    except Exception:
        if provider is not None or model is not None:  # cross-model judge failed → authoring model
            try:
                v = llm_json(JUDGE_SYSTEM, user, JudgeVerdict, temperature=0.0)
                v.error = False
                return v
            except Exception:
                pass
        # judge unavailable after retries: honest error verdict — never a free pass (was 0.7)
        return JudgeVerdict(error=True, understandable=False, score=0.0,
                            reasoning="judge unavailable",
                            topic_identifiable=False, purpose_identifiable=False,
                            all_references_resolved=False, prerequisites_satisfied=False,
                            source_grounded=False)


def rebind_contract(cand: Candidate, units_by_id: dict[str, Unit], adapter) -> str:
    """P1c: (re)bind the completeness contract from the roles ACTUALLY in the span. Called
    before the first judge pass and again after EVERY span mutation (contract fill, repair
    expansion, trim) BEFORE re-judging, so the judge, the completeness gate, and final
    scoring all run under the same contract. The anchor role stays untouched on cand.role
    (provenance); no binding → fall back to it (pre-P1 behavior)."""
    cand.contract_role = choose_contract(cand.unit_ids, units_by_id, adapter) or cand.role
    return cand.contract_role


def validate_candidate(cand: Candidate, sentences: list[Sentence], adapter,
                       visual_summary: str = "", topic: str = "") -> JudgeVerdict:
    return judge_clip(_clip_text(sentences, cand.i_start, cand.i_end),
                      cand.contract_role or cand.role, adapter,
                      visual_summary=visual_summary, topic=topic)


def is_complete(v: JudgeVerdict, role: str, adapter, min_score: float) -> bool:
    if v.error:
        return False
    required = adapter.required_verdict_fields(role)
    return v.score >= min_score and all(getattr(v, f, True) for f in required)


def _reindex(unit_ids: list[str], units_by_id: dict[str, Unit], sentences: list[Sentence]):
    units = [units_by_id[u] for u in unit_ids if u in units_by_id]
    i0 = min(u.sentence_range[0] for u in units)
    i1 = max(u.sentence_range[1] for u in units)
    i0 = max(0, min(i0, len(sentences) - 1))
    i1 = max(i0, min(i1, len(sentences) - 1))
    return i0, i1


def _norm_kind(kind: str) -> str:
    """Canonicalize a failure_reason kind for fuzzy matching (the judge free-forms them)."""
    return "".join(c if c.isalnum() else "_" for c in (kind or "").lower())


# ── judge-integrity: quote verification + fresh-context kill confirmation (Wave 1) ──────
# LLM validators over-flag (TPR ~96% / TNR <25%): phantom failure reasons must never kill a
# clip. A kill therefore requires strictly more evidence than a pass — every fatal reason
# must quote real span text (deterministic containment, zero LLM calls) AND survive a
# fresh-context confirmation call.

def _normalize_quote(s: str) -> str:
    """Lowercase, strip punctuation (keep letters/digits/spaces), collapse all whitespace."""
    lowered = (s or "").lower()
    kept = "".join(c if (c.isalnum() or c.isspace()) else " " for c in lowered)
    return " ".join(kept.split())


def _verify_failure_reasons(verdict, clip_text: str) -> list[bool]:
    """Deterministic, zero-LLM verification: reason i is verified iff its evidence_quote is
    non-empty AND contained in the normalized clip text; unresolved_reference-kind reasons
    additionally need reference_text (when present) contained. Results are attached to the
    verdict as a private attr (never part of the LLM output schema) so they travel with it
    to the gate decision and into Rejection records."""
    text = _normalize_quote(clip_text)
    flags: list[bool] = []
    for fr in (getattr(verdict, "failure_reasons", None) or []):
        quote = _normalize_quote(getattr(fr, "evidence_quote", "") or "")
        ok = bool(quote) and quote in text
        if ok and "reference" in _norm_kind(getattr(fr, "kind", "")):
            ref = _normalize_quote(getattr(fr, "reference_text", "") or "")
            if ref and ref not in text:
                ok = False
        flags.append(ok)
    if verdict is not None:
        try:
            verdict._reason_verified = list(flags)
        except (AttributeError, ValueError):    # non-JudgeVerdict stand-ins in tests
            pass
    return flags


class KillClaimCheck(BaseModel):
    claim: int = 0           # 1-based claim number (positional fallback when absent)
    confirmed: bool = False
    quote: str = ""          # exact transcript words supporting the answer


class KillConfirmation(BaseModel):
    claims: list[KillClaimCheck] = Field(default_factory=list)


CONFIRM_SYSTEM = (
    "You are a careful, neutral fact-checker. You are given a transcript and a numbered list of "
    "claims about it. Assess each claim independently and literally against ONLY the transcript "
    "text. For each claim output: claim (its number), confirmed (true only if the claim is "
    "actually true of the transcript), and quote — the EXACT contiguous words copied verbatim "
    "from the transcript that support your answer. If you cannot quote transcript text that "
    "supports confirming a claim, set confirmed to false. Output only the structured result."
)


def confirm_kill(clip_text: str, verified_reasons: list, outage: Optional[dict] = None) -> list[bool]:
    """Fresh-context kill confirmation: ONE llm_json call (judge provider/model, temperature 0)
    that sees ONLY the clip transcript plus each verified failure reason restated as a neutral
    claim. A confirmation whose quote fails the same containment check counts as NOT confirmed.
    Called ONLY on the kill path. On llm_json failure: conservative fallback — nothing is
    confirmed (never kill on an unconfirmable verdict) and ``outage`` (when given) is marked."""
    if not verified_reasons:
        return []
    from ...llm import llm_json
    lines = []
    for i, fr in enumerate(verified_reasons, start=1):
        detail = (getattr(fr, "detail", "") or getattr(fr, "missing_concept", None)
                  or getattr(fr, "reference_text", None) or "(no detail)")
        lines.append(
            f"Claim {i}: {getattr(fr, 'kind', 'other')}: {detail}. Is this actually true of the "
            "transcript below? Answer with confirmed true/false AND quote the exact transcript "
            "text supporting your answer.")
    user = "CLAIMS TO ASSESS:\n" + "\n".join(lines) + f"\n\nTRANSCRIPT:\n{clip_text}\n"
    jp = config.JUDGE_PROVIDER
    provider = None if jp in ("", "same") else jp
    jm = config.JUDGE_MODEL
    model = jm if (jm and jm != config.GEMINI_MODEL) else None
    try:
        res = llm_json(CONFIRM_SYSTEM, user, KillConfirmation, temperature=0.0,
                       provider=provider, model=model)
    except Exception:
        if outage is not None:
            outage["confirm_kill"] = True
        return [False] * len(verified_reasons)
    text = _normalize_quote(clip_text)
    out = [False] * len(verified_reasons)
    for i, item in enumerate(res.claims):
        idx = item.claim - 1 if 1 <= item.claim <= len(out) else i
        if not (0 <= idx < len(out)):
            continue
        quote = _normalize_quote(item.quote)
        out[idx] = bool(item.confirmed) and bool(quote) and quote in text
    return out


def _flag_unverified_atomicity(cand: Candidate) -> None:
    """W25-E: single_idea is ADVISORY this wave (deliberately not in _hard_core_ok or
    required_verdict_fields — no calibration labels yet), so single_idea=False whose
    verdict carries no VERIFIED excess-content reason (over_inclusion family — the
    _TRIM_KIND_HINTS vocabulary; bare 'other' does not demonstrate a second idea) must
    never gate or kill: surface it as a ship warning only (eval visibility). Relies on
    the _reason_verified flags a prior _verify_failure_reasons pass attached."""
    v = getattr(cand, "verdict", None)
    if v is None or getattr(v, "single_idea", True):
        return
    reasons = list(getattr(v, "failure_reasons", None) or [])
    flags = list(getattr(v, "_reason_verified", None) or [])
    for fr, ok in zip(reasons, flags):
        k = _norm_kind(getattr(fr, "kind", "")).strip("_")
        if ok and any(h in k for h in _TRIM_KIND_HINTS):
            return                          # a verified excess-content reason backs the bit
    cand.warnings = tuple(set(getattr(cand, "warnings", ()) or ()) | {"single_idea_unverified"})


def _attach_judge_stats(cand: Candidate, sentences: list[Sentence]) -> None:
    """Countable judge-integrity stats on EVERY validate_and_repair outcome: how many failure
    reasons the final verdict emitted and how many survived quote verification (the inputs to
    phantom_verdict_rate). Pure — zero LLM calls."""
    v = getattr(cand, "verdict", None)
    if v is None:
        return
    flags = _verify_failure_reasons(v, _clip_text(sentences, cand.i_start, cand.i_end))
    reasons = list(getattr(v, "failure_reasons", None) or [])
    cand.n_failure_reasons = len(reasons)
    cand.n_verified = sum(1 for x in flags if x)
    # W25-G kind-level mirror (rides into the spec via candidate_to_boundary_input):
    # phantom_quotable_rate needs WHICH kinds shipped unverified — absence kinds are
    # unquotable by construction and must not count against the quotable gate-health rate.
    cand.verified_kinds = tuple(getattr(f, "kind", "") for f, ok in zip(reasons, flags) if ok)
    cand.unverified_kinds = tuple(getattr(f, "kind", "") for f, ok in zip(reasons, flags) if not ok)
    _flag_unverified_atomicity(cand)        # single_idea=False without evidence ⇒ warning only


def expand_candidate(cand: Candidate, verdict: JudgeVerdict, graph, units: list[Unit],
                     units_by_id: dict[str, Unit], introducers: dict[str, list[str]],
                     sentences: list[Sentence], max_span_s: float,
                     adapter=None) -> Optional[Candidate]:
    """Pull the specific missing context for an incomplete verdict. Targeting is driven by the
    verdict's reliable BOOLEAN gates (kind-agnostic), with the free-form failure_reasons[].kind
    used only as an extra hint — the LLM does not emit a fixed kind vocabulary, so booleans are
    the durable signal. Grows the clip greedily nearest-anchor-first within the span cap.

    W25-F: pulls of elements the BOUND contract deliberately excludes are disabled — a
    practice_prompt-bound clip must never grow toward the answer its contract omits
    (base.py: 'deliberately NO solution element'), else pull_result fires off the advisory
    result_complete=False and choose_contract's specificity tie-break then locks the grown
    span in as a solution clip. An element family (problem/reasoning/result) is excluded
    iff a contract for the bound role EXISTS and neither its verdict boolean is
    contract-required (adapter.required_verdict_fields) nor any contract element (any
    necessity) is satisfiable by the family's pull roles; booleans outside that set become
    advisory for expansion. Contract-free spans (no adapter / no contract) keep the
    pre-W25-F kind-agnostic behavior."""
    order = {u.unit_id: i for i, u in enumerate(units)}
    anchor_i = order.get(cand.anchor_id, 0)
    cur = set(cand.unit_ids)
    targets: set[str] = set()

    bound_role = cand.contract_role or cand.role
    contract = adapter.contract_for(bound_role) if adapter is not None else None
    required_fields = set(adapter.required_verdict_fields(bound_role)) \
        if adapter is not None else set()

    def _element_allowed(gate_field: str, family_roles: tuple[str, ...]) -> bool:
        if contract is None:
            return True
        if gate_field in required_fields:
            return True
        fam = set(family_roles)
        return any(fam & set(el.roles) for el in contract.elements)

    # the family roles mirror EXACTLY what pull_problem/pull_reasoning/pull_result pull
    allow_problem = _element_allowed(
        "problem_statement_complete", ("example_setup", "problem_givens", "practice_prompt", "setup"))
    allow_reasoning = _element_allowed(
        "reasoning_complete", ("worked_step", "calculation", "derivation", "procedure"))
    allow_result = _element_allowed("result_complete", ("result", "solution"))

    def prior_introducer(concept: str) -> Optional[str]:
        lst = introducers.get((concept or "").strip().lower())
        if not lst:
            return None
        prior = [x for x in lst if order.get(x, 0) <= anchor_i]
        return prior[-1] if prior else lst[0]

    def pull_graph(*relations: str) -> None:
        for uid in list(cur):
            for e in graph.needs(uid, relations):
                targets.add(e.target)

    def pull_references() -> None:                          # antecedents for dangling 'this/that/…'
        pull_graph("refers_to", "answers", "requires")
        for uid in list(cur):
            u = units_by_id.get(uid)
            for ref in (u.references if u else []):
                if ref.source_unit:
                    targets.add(ref.source_unit)
        for uid, _rel in cand.referential:                  # far prereqs closure set aside
            targets.add(uid)

    def pull_grounding() -> None:                           # prior same-thread context (not mid-argument)
        pull_graph("continues", "requires", "explains", "defines")
        earliest = min((order[u] for u in cur if u in order), default=anchor_i)
        for u in units:
            if order[u.unit_id] == earliest - 1:            # the immediately preceding unit
                targets.add(u.unit_id)

    def pull_problem() -> None:
        prior = [u for u in units if order[u.unit_id] < anchor_i
                 and u.role in ("example_setup", "problem_givens", "practice_prompt", "setup")]
        if prior:
            targets.add(prior[-1].unit_id)

    def pull_reasoning() -> None:
        for u in units:
            if u.role in ("worked_step", "calculation", "derivation", "procedure") \
                    and abs(order[u.unit_id] - anchor_i) <= 3:
                targets.add(u.unit_id)

    def pull_result() -> None:
        after = [u for u in units if order[u.unit_id] > anchor_i and u.role in ("result", "solution")]
        if after:
            targets.add(after[0].unit_id)

    # 1) targeted hints from the judge's failure_reasons (kind fuzzily normalized);
    # problem/reasoning/result-family hints obey the same contract-exclusion gate as the
    # boolean fallbacks below (W25-F) — the branch still CONSUMES the kind either way, so
    # a gated hint never falls through to a wrong family. --------------------------------
    for fr in verdict.failure_reasons:
        k = _norm_kind(fr.kind)
        if "prerequisite" in k and fr.missing_concept:
            t = prior_introducer(fr.missing_concept)
            if t:
                targets.add(t)
        elif any(w in k for w in ("reference", "dangling", "antecedent", "pronoun")):
            pull_references()
        elif any(w in k for w in ("problem", "givens", "prompt")):
            if allow_problem:
                pull_problem()
        elif any(w in k for w in ("reasoning", "step", "derivation", "calculation")):
            if allow_reasoning:
                pull_reasoning()
        elif any(w in k for w in ("result", "outcome", "answer", "conclusion")):
            if allow_result:
                pull_result()
        elif "visual" in k:
            pull_graph("visually_depends_on")
        elif "ground" in k:
            pull_grounding()

    # 2) boolean-driven fallbacks — the reliable structured signal, kind-agnostic; the
    # W25-F contract-exclusion gate applies (a deliberately-omitted element is advisory) --
    if not verdict.all_references_resolved:
        pull_references()
    if not verdict.prerequisites_satisfied:
        for uid, _rel in cand.referential:
            targets.add(uid)
    if not verdict.source_grounded:
        pull_grounding()
    if not verdict.visuals_sufficient:
        pull_graph("visually_depends_on")
    if not verdict.problem_statement_complete and allow_problem:
        pull_problem()
    if not verdict.reasoning_complete and allow_reasoning:
        pull_reasoning()
    if not verdict.result_complete and allow_result:
        pull_result()

    # add targets nearest-to-anchor first, staying within the span cap (partial progress is fine)
    from dataclasses import replace
    ordered_targets = sorted((t for t in targets - cur if t in units_by_id),
                             key=lambda t: abs(order.get(t, 0) - anchor_i))
    if not ordered_targets:
        return None
    chosen = set(cur)
    for t in ordered_targets:
        trial = sorted(chosen | {t}, key=lambda x: (units_by_id[x].start, order.get(x, 0)))
        i0, i1 = _reindex(trial, units_by_id, sentences)
        if sentences[i1].end - sentences[i0].start <= max_span_s:
            chosen.add(t)
    if chosen == cur:
        return None
    new_ids = sorted(chosen, key=lambda x: (units_by_id[x].start, order.get(x, 0)))
    i0, i1 = _reindex(new_ids, units_by_id, sentences)
    from .integrity import true_contents
    new_ids, new_ref = true_contents(new_ids, cand.referential, units, i0, i1)
    return replace(cand, unit_ids=new_ids, referential=new_ref, i_start=i0, i_end=i1,
                   start=sentences[i0].start, end=sentences[i1].end)


def _fill_contract(cand: Candidate, adapter, units: list[Unit], units_by_id: dict[str, Unit],
                   sentences: list[Sentence], max_span_s: float) -> Candidate:
    """Deterministic contract fill (spec §8): pull the units for any missing REQUIRED
    contract element into the candidate. Since P2a this runs only AFTER the judge has
    failed the candidate at its native size — never before the first judge call — as the
    first move of the grow phase. Span-safe and never rejects — if an element can't be
    added within budget the judge still has final say. Keys off the content-bound
    contract_role (P1b), not the anchor role."""
    contract = adapter.contract_for(cand.contract_role or cand.role)
    if not contract:
        return cand
    order = {u.unit_id: i for i, u in enumerate(units)}
    ai = order.get(cand.anchor_id, 0)
    cur = set(cand.unit_ids)
    for el in contract.elements:
        if el.necessity != "required":
            continue
        present = {units_by_id[uid].role for uid in cur if uid in units_by_id}
        if any(r in present for r in el.roles):
            continue
        want = set(el.roles)
        if el.position == "before":
            picks = [u for u in units if order[u.unit_id] < ai and u.role in want]
            pick = picks[-1] if picks else None          # nearest element before the anchor
        elif el.position == "after":
            picks = [u for u in units if order[u.unit_id] > ai and u.role in want]
            pick = picks[0] if picks else None
        else:
            picks = sorted((u for u in units if u.role in want),
                           key=lambda u: abs(order[u.unit_id] - ai))
            pick = picks[0] if picks else None
        if not pick or pick.unit_id in cur:
            continue
        trial = sorted(cur | {pick.unit_id}, key=lambda x: (units_by_id[x].start, order.get(x, 0)))
        i0, i1 = _reindex(trial, units_by_id, sentences)
        if sentences[i1].end - sentences[i0].start <= max_span_s:
            cur.add(pick.unit_id)
    if cur == set(cand.unit_ids):
        return cand
    new_ids = sorted(cur, key=lambda x: (units_by_id[x].start, order.get(x, 0)))
    i0, i1 = _reindex(new_ids, units_by_id, sentences)
    # keep unit_ids truthful for the widened hull (mirrors expand_candidate): units whose
    # sentences the fill swallowed join unit_ids NOW, so the immediate rebind_contract sees
    # every role actually in the judged text (I1 reconcile of a P1/P2 review finding).
    from .integrity import true_contents
    new_ids, new_ref = true_contents(new_ids, cand.referential, units, i0, i1)
    from dataclasses import replace
    return replace(cand, unit_ids=new_ids, referential=new_ref, i_start=i0, i_end=i1,
                   start=sentences[i0].start, end=sentences[i1].end)


# ── P2: repair rework — trim moves + bisection over the trim lattice ────────────────────

def _hard_core_ok(v) -> bool:
    """The ship-without-accept bar (the asymmetric gate's best-partial test): topic +
    purpose + grounded + references all intact. Doubles as the bisection oracle's
    'known-good' predicate (P2c) — a sub-span passing the hard core is shippable."""
    return bool(v is not None and not getattr(v, "error", False)
                and v.topic_identifiable and v.purpose_identifiable
                and v.source_grounded and v.all_references_resolved)


# ── card-as-repair (Wave 3 CARD1-4): a purely ACCEPT-SIDE rescue of a would-be kill whose
# EVERY confirmed reason is prereq/reference-family — a context card shown BEFORE the clip
# can supply the missing prerequisite or resolve the dangling reference WITHOUT growing the
# span. A rescue converts a would-be Rejection into a ship; it NEVER creates a Rejection, so
# the unverified_kill=0 invariant is structurally untouched. ─────────────────────────────

# kinds a shown-before card can actually satisfy (mirrors expand_candidate's prerequisite/
# reference vocabulary at :442/:446, plus the card-family 'missing_context'). A single
# confirmed reason OUTSIDE this set (e.g. missing_result) means the card cannot fix the clip
# → no rescue, the existing verified+confirmed kill stands.
_CARD_RESCUE_KIND_HINTS = ("prerequisite", "reference", "dangling", "antecedent",
                           "pronoun", "context")


def _card_rescuable(reasons) -> bool:
    """Precondition for a card-as-repair attempt: ≥1 confirmed reason AND EVERY confirmed
    reason is prereq/reference-family (a card cannot supply a missing result/reasoning/etc)."""
    reasons = list(reasons or [])
    if not reasons:
        return False
    return all(any(h in _norm_kind(getattr(fr, "kind", "")).strip("_")
                   for h in _CARD_RESCUE_KIND_HINTS)
               for fr in reasons)


def _card_clears(v) -> bool:
    """The carded re-judge's acceptance bar: the hard core (topic+purpose+grounded+refs) AND
    prerequisites_satisfied AND all_references_resolved all hold — i.e. the shown-before card
    actually resolved what killed the clip. (all_references_resolved is already inside the
    hard core; it is restated to mirror the spec's explicit three-way gate.)"""
    return bool(_hard_core_ok(v)
                and getattr(v, "prerequisites_satisfied", False)
                and getattr(v, "all_references_resolved", False))


def _seed_referential_from_introducers(card_spec: dict, reasons, introducers: dict,
                                       units, units_by_id: dict) -> list:
    """CARD5(a) robust rescue seeding: when a card-rescuable kill's missing prerequisite is
    NOT already in the clip's ``referential`` pool, map each reason's missing_concept → its
    introducer unit (reusing expand_candidate's prior_introducer preference — the introducer
    at/before the anchor) and add that unit id to the card's source set. Returns a NEW
    referential list (original + any seeded ``(uid, 'introduces')`` pairs); the card stays
    grounded because generate_context_card drops any seed whose text can't be verified against
    a real unit — no source unit ⇒ no card ⇒ the existing kill stands. No fabrication, ever."""
    seeded = list(card_spec.get("referential", []))
    if not introducers or not reasons:
        return seeded
    order = {u.unit_id: i for i, u in enumerate(units or [])}
    anchor_i = order.get(card_spec.get("anchor_id", ""), 0)
    in_clip = set(card_spec.get("unit_ids", []))
    have = {uid for uid, _rel in seeded}
    ci = {}                                  # case-insensitive concept view (index is raw-keyed;
    for k, v in introducers.items():         # the judge free-cases missing_concept)
        ci.setdefault((k or "").strip().lower(), v)
    for fr in reasons:
        concept = (getattr(fr, "missing_concept", None) or "").strip()
        if not concept:
            continue
        lst = introducers.get(concept) or ci.get(concept.lower())
        if not lst:
            continue
        prior = [x for x in lst if order.get(x, 0) <= anchor_i]   # prefer an introducer ≤ anchor
        uid = prior[-1] if prior else lst[0]
        if uid in units_by_id and uid not in in_clip and uid not in have:
            seeded.append((uid, "introduces"))
            have.add(uid)
    return seeded


def _card_rescue_verdict(card_spec: dict, clip_text: str, role: str, adapter,
                         units_by_id: dict, topic: str, visual_summary: str,
                         reasons=None, introducers: Optional[dict] = None, units=None):
    """Generate a grounded context card from ``card_spec`` (a dict carrying unit_ids /
    referential / anchor_id — a Candidate shim or a spec dict) and RE-JUDGE the SAME
    ``clip_text`` WITH the card as shown-before context (NO span growth/trim). Returns
    (card, verdict); card='' means no groundable card exists → the caller must NOT rescue and
    the existing kill stands. Two FREE post-budget LLM calls, only on the would-be-kill path
    (like confirm_kill); bounded to 1 card + 1 re-judge; never creates a Rejection.

    CARD5(a): if the first card is '' because the missing prerequisite isn't in ``referential``,
    seed the card from the verdict's missing_concept → introducer unit (``reasons`` +
    ``introducers`` + ``units``) and regenerate once — still grounded, so a concept with no
    real source unit yields '' and no rescue happens."""
    from .context_card import generate_context_card
    card = generate_context_card(card_spec, units_by_id, adapter, topic)
    if not card and introducers and reasons:
        seeded = _seed_referential_from_introducers(card_spec, reasons, introducers, units,
                                                    units_by_id)
        if seeded != list(card_spec.get("referential", [])):
            card = generate_context_card({**card_spec, "referential": seeded},
                                         units_by_id, adapter, topic)
    if not card:
        return "", None
    verdict = judge_clip(clip_text, role, adapter, visual_summary=visual_summary,
                         topic=topic, context_card=card)
    return card, verdict


# W25-E: over_inclusion/multiple_ideas/atomicity variants route to trim too — atomicity
# reasons are quote-friendly (the second idea's opening sentence is verbatim in the span),
# so they can survive the asymmetric kill gate, unlike absence-shaped kinds. Substrings
# are matched against _norm_kind output ('over-inclusion'→'over_inclusion'), and none of
# them can collide with the exact-equality 'other' check in _trim_flavored.
_TRIM_KIND_HINTS = ("off_topic", "coheren", "jump", "tangent", "focus", "drift", "unrelated",
                    "inclusion", "multiple_idea", "atomic")


def _trim_flavored(verdict) -> bool:
    """P2b routing: verdicts whose complaints are about EXCESS content (off-topic edges,
    topic jumps, incoherence, over-inclusion of a second idea) rather than MISSING content
    route to trim moves. These are
    exactly the verdicts that produce zero expansion targets and burned the repair budget
    pre-P2. A reason-less verdict failing the topic/purpose core is coherence-flavored
    too; missing-content kinds (references/prerequisites/problem/reasoning/result/visual/
    grounding) keep routing to growth."""
    reasons = list(getattr(verdict, "failure_reasons", None) or [])
    for fr in reasons:
        k = _norm_kind(getattr(fr, "kind", "")).strip("_")
        if k == "other" or any(h in k for h in _TRIM_KIND_HINTS):
            return True
    if not reasons and not (verdict.topic_identifiable and verdict.purpose_identifiable):
        return True
    return False


def _excess_verified(v, clip_text: str) -> bool:
    """Bisection-oracle inversion guard (W25-E fix): atomicity failures leave the hard
    core INTACT by design (topic/purpose/grounded/refs all true on a two-idea clip — the
    same property that makes over_inclusion quote-survivable), so `_hard_core_ok` alone
    cannot mean 'the excess is gone': a still-over-inclusive trial would count as
    known-good, lo would advance past the true boundary, and a judged-COMPLETE atomic
    sub-span would be overwritten by a larger, still-failing one. True iff the verdict
    carries an excess-content reason (_TRIM_KIND_HINTS family; bare 'other' does not
    demonstrate excess — mirrors _flag_unverified_atomicity) whose evidence_quote passes
    deterministic containment against the trial text. UNVERIFIED excess complaints stay
    warning-only and never shrink the ship (asymmetric-gate philosophy: the judge
    over-flags, so phantom over_inclusion must not move the bisection)."""
    reasons = list(getattr(v, "failure_reasons", None) or [])
    if not reasons:
        return False
    flags = _verify_failure_reasons(v, clip_text)        # deterministic, zero-LLM
    for fr, ok in zip(reasons, flags):
        k = _norm_kind(getattr(fr, "kind", "")).strip("_")
        if ok and any(h in k for h in _TRIM_KIND_HINTS):
            return True
    return False


def _protected_unit_ids(cand: Candidate, units_by_id: dict[str, Unit], adapter,
                        onset_uid: Optional[str] = None) -> set[str]:
    """Units a trim may NEVER drop (P2b): the anchor + every in-span unit whose role
    satisfies a REQUIRED element of the CURRENT bound contract + the clip's onset unit.

    ``onset_uid`` is the original leading unit of the candidate BEFORE any grow/expand
    mutations.  Callers inside validate_and_repair compute it once from the original span
    and thread it here so that units GROWN IN after assembly are not mistakenly treated as
    the onset.  When called directly (e.g. tests), ``onset_uid=None`` causes the function
    to derive the onset from the current ``cand.unit_ids`` — correct for any span that has
    not been mutated by grow."""
    protected = {cand.anchor_id}
    contract = adapter.contract_for(cand.contract_role or cand.role)
    if contract:
        required_roles: set[str] = set()
        for el in contract.elements:
            if el.necessity == "required":
                required_roles.update(el.roles)
        for uid in cand.unit_ids:
            u = units_by_id.get(uid)
            if u is not None and u.role in required_roles:
                protected.add(uid)
    # Protect the clip's LEADING (temporally earliest) in-span unit: trimming it would advance
    # the start past the setup/problem-read and re-open the clip mid-thought (the discourse-onset
    # invariant). The end guard already protects completeness; this protects the opening.
    if onset_uid is not None:
        # Caller has already resolved the onset from the pre-grow span — use it directly.
        if onset_uid in units_by_id:
            protected.add(onset_uid)
    else:
        # Fallback: derive from the current span (correct when cand has not been mutated).
        in_span = [units_by_id[uid] for uid in cand.unit_ids if uid in units_by_id]
        if in_span:
            leading = min(in_span, key=lambda u: (u.start, u.sentence_range[0]))
            protected.add(leading.unit_id)
    return protected


def _trim_lattice(cand: Candidate, units: list[Unit], units_by_id: dict[str, Unit],
                  adapter, onset_uid: Optional[str] = None) -> tuple[list[str], int]:
    """(span units ordered protected-first then nearest-anchor-first, protected count).
    Prefix k of this order is the trim-lattice point keeping the k units closest to the
    anchor (protected units always kept): dropping a suffix drops the temporally farthest
    removable units, so their sentences leave the span (P2b)."""
    order = {u.unit_id: i for i, u in enumerate(units)}
    ai = order.get(cand.anchor_id, 0)
    in_span = [uid for uid in cand.unit_ids if uid in units_by_id]
    protected = _protected_unit_ids(cand, units_by_id, adapter, onset_uid=onset_uid)
    prot = [uid for uid in in_span if uid in protected]
    rest = sorted((uid for uid in in_span if uid not in protected),
                  key=lambda uid: (abs(order.get(uid, 0) - ai), order.get(uid, 0)))
    return prot + rest, len(prot)


def _trim_to(cand: Candidate, keep_ids: list[str], units: list[Unit],
             units_by_id: dict[str, Unit], sentences: list[Sentence],
             ref_memory: dict[str, str]) -> Candidate:
    """P2d: build the trim-lattice candidate keeping exactly ``keep_ids`` (plus any unit
    whose sentences still lie inside the kept hull — integrity.true_contents semantics).
    Units leaving the span leave unit_ids; remembered referential relations for now-
    outside units return to referential; time/sentence bounds are recomputed truthfully."""
    from dataclasses import replace

    from .integrity import true_contents
    i0, i1 = _reindex(keep_ids, units_by_id, sentences)
    merged = list(cand.referential)
    have = {uid for uid, _rel in merged}
    for uid, rel in ref_memory.items():
        if uid not in have:
            merged.append((uid, rel))
    new_ids, new_ref = true_contents(list(keep_ids), merged, units, i0, i1)
    return replace(cand, unit_ids=new_ids, referential=new_ref, i_start=i0, i_end=i1,
                   start=sentences[i0].start, end=sentences[i1].end)


def validate_and_repair(cand: Candidate, sentences: list[Sentence], graph, units: list[Unit],
                        units_by_id: dict[str, Unit], introducers: dict[str, list[str]], adapter,
                        settings: dict, visual_summary_fn, topic: str,
                        cache: dict, cache_lock=None) -> tuple[Optional[Candidate], Optional["Rejection"]]:
    min_score = float(settings.get("min_comprehension_score", config.JUDGE_MIN_SCORE))
    # judged text must be shippable: repair expansion cap can never exceed the ship cap
    max_span = min(float(settings.get("closure_max_span_s", config.CLOSURE_MAX_SPAN_S)),
                   float(settings.get("max_clip_duration_s", config.DEFAULTS["max_clip_duration_s"])))
    budget = config.JUDGE_MAX_REPAIR + 1     # total NEW judgments; verdict-cache hits are free
    new_calls = 0
    evals = 0
    trims_taken = 0                          # trim-lattice probes judged (eval: n_trims)
    from dataclasses import replace

    def _stamp_trims(c: Optional[Candidate]) -> Optional[Candidate]:
        """Record the trim moves this repair took on whichever candidate ships — the
        per-video n_trims eval column sums these off the shipped specs."""
        if c is not None:
            c.n_trims = trims_taken
        return c

    def _cached(key):
        if cache_lock is not None:
            with cache_lock:
                return cache.get(key)
        return cache.get(key)

    def _store(key, verdict):
        if cache_lock is not None:
            with cache_lock:
                cache[key] = verdict
        else:
            cache[key] = verdict

    # P2d: remember every referential relation ever seen on this candidate line, so a trim
    # that pushes a previously-absorbed unit back outside the span can truthfully return
    # its entry to referential (true_contents drops any that are still inside the span).
    ref_memory: dict[str, str] = {uid: rel for uid, rel in cand.referential}

    def _judge(c: Candidate):
        """One judge pass over ``c``: P1 rebind first, then the text-hash verdict cache,
        then (on a miss) the LLM. Stamps verdict/judged_text_hash/attempts on ``c``.
        Only cache MISSES count against the repair budget — revisits are free (P2c)."""
        nonlocal new_calls, evals
        # bind the contract by CONTENT (P1) before EVERY judge pass — fills, expansions
        # and trims all mutate the span, and the judge, the completeness gate, and final
        # scoring must share one contract.
        rebind_contract(c, units_by_id, adapter)
        judged_text = _clip_text(sentences, c.i_start, c.i_end)
        text_hash = judged_text_hash(judged_text)
        # cache key = (units, judged-text hash): byte-identical convergent candidates still
        # share a verdict, but ANY text difference over the same units is a cache miss.
        # (contract_role is a pure function of unit_ids, so cached verdicts always match it.)
        key = (frozenset(c.unit_ids), text_hash)
        verdict = _cached(key)
        if verdict is None:
            verdict = judge_clip(judged_text, c.contract_role, adapter,
                                 visual_summary=visual_summary_fn(c.start, c.end),
                                 topic=topic)
            new_calls += 1
            if not verdict.error:            # an outage verdict must not poison the cache
                _store(key, verdict)
        evals += 1
        c.verdict = verdict
        # record the hash of the EXACT text sent to the judge — on the outage path too: the
        # text WAS judged (the call just failed), so the post-snap seam must not spend an
        # extra judge call on an UNCHANGED outage spec; it still re-judges on any text change.
        c.judged_text_hash = text_hash
        c.attempts = evals
        for uid, rel in c.referential:
            ref_memory.setdefault(uid, rel)
        return verdict

    # ── P2a: judge the candidate at its MINIMAL post-closure size FIRST — before any
    # _fill_contract inflation, so 10–60s anchors are scored at native size (audit F2). ──
    verdict = _judge(cand)
    if verdict.error:
        _attach_judge_stats(cand, sentences)
        return cand, None                   # ship-but-flag: keep the clip, skip repair entirely
    if is_complete(verdict, cand.contract_role, adapter, min_score):
        _attach_judge_stats(cand, sentences)
        return cand, None
    best: Optional[Candidate] = replace(cand)     # the native verdict is the recorded baseline
    # last-known-good endpoint for the bisection (P2c): the largest judged span so far
    # whose verdict cleared the hard core.
    last_good: Optional[Candidate] = replace(cand) if _hard_core_ok(verdict) else None

    # Discourse-onset invariant: identify the clip's leading unit from the ORIGINAL span
    # (before any grow/expand mutations) so that units grown in later are never mistakenly
    # treated as the onset and locked from bisection.
    _orig_in_span = [units_by_id[uid] for uid in cand.unit_ids if uid in units_by_id]
    _onset_uid: Optional[str] = (
        min(_orig_in_span, key=lambda u: (u.start, u.sentence_range[0])).unit_id
        if _orig_in_span else None
    )

    def _removable(c: Candidate) -> bool:
        lattice, n_prot = _trim_lattice(c, units, units_by_id, adapter, onset_uid=_onset_uid)
        return n_prot >= 1 and len(lattice) > n_prot

    # ── grow phase: missing-content verdicts (references / prerequisites / problem /
    # reasoning / result / visual / grounding) pull targeted context — fill + the existing
    # expansion targeting, each grown span re-judged under its rebound contract. Trim
    # takes precedence over growth only when a trim is actually possible. ─────────────────
    while new_calls < budget and not (_trim_flavored(verdict) and _removable(cand)):
        grown = cand
        # deterministic contract fill — since P2a it runs only AFTER a failing verdict,
        # as the first grow move, never before the candidate's first judge call.
        if not check_contract(grown.unit_ids, grown.contract_role, units_by_id, adapter).ok:
            filled = _fill_contract(grown, adapter, units, units_by_id, sentences, max_span)
            if set(filled.unit_ids) != set(grown.unit_ids):
                grown = filled
        expanded = expand_candidate(grown, verdict, graph, units, units_by_id, introducers,
                                    sentences, max_span, adapter=adapter)
        if expanded is not None and set(expanded.unit_ids) != set(grown.unit_ids):
            grown = expanded
        if set(grown.unit_ids) == set(cand.unit_ids):
            break                           # zero grow targets (pre-P2 this burned the budget)
        cand = grown
        verdict = _judge(cand)
        if verdict.error:
            _attach_judge_stats(cand, sentences)
            return cand, None
        if is_complete(verdict, cand.contract_role, adapter, min_score):
            _attach_judge_stats(cand, sentences)
            return cand, None
        if verdict.score > getattr(best.verdict, "score", -1):
            best = replace(cand)
        if _hard_core_ok(verdict):
            last_good = replace(cand)       # a later coherence failure bisects back to here

    # ── trim phase (P2b/P2c): off-topic/coherence complaints drop the units temporally
    # farthest from the anchor (never the anchor, never contract-required units). The trim
    # lattice (unit-prefixes by distance from anchor) is searched by BISECTION between
    # last-known-good and the failing span; the text-hash cache makes revisits free. ──────
    if _trim_flavored(verdict):
        lattice, n_protected = _trim_lattice(cand, units, units_by_id, adapter, onset_uid=_onset_uid)
        n = len(lattice)
        if n_protected >= 1 and n > n_protected:
            lo = n_protected - 1            # virtual floor: just below the untrimmable minimum
            best_good: Optional[Candidate] = None
            if last_good is not None:
                k = len(set(last_good.unit_ids))
                if k < n and set(last_good.unit_ids) == set(lattice[:k]):
                    lo, best_good = k, last_good      # e.g. the anchor-native hard-core pass
            hi = n                          # known-bad: the current failing span
            while hi - lo > 1:
                mid = (lo + hi) // 2
                trial = _trim_to(cand, lattice[:mid], units, units_by_id, sentences, ref_memory)
                trial_text = _clip_text(sentences, trial.i_start, trial.i_end)
                probe_key = (frozenset(trial.unit_ids), judged_text_hash(trial_text))
                if _cached(probe_key) is None and new_calls >= budget:
                    break                   # only NEW judgments spend budget; hits are free
                v = _judge(trial)
                trims_taken += 1            # one trim move probed (cache hit or live)
                if v.error:
                    _attach_judge_stats(trial, sentences)
                    return _stamp_trims(trial), None   # ship-but-flag: same outage policy as above
                # W25-E oracle fix: hard-core-good is NOT known-good while a VERIFIED
                # excess-content reason persists — atomicity failures keep the hard core
                # intact, so the old `or _hard_core_ok(v)` predicate advanced lo over
                # still-over-inclusive trials and overwrote judged-COMPLETE sub-spans.
                if is_complete(v, trial.contract_role, adapter, min_score):
                    lo, best_good = mid, trial           # a complete sub-span always wins upward
                elif _hard_core_ok(v) and not _excess_verified(v, trial_text):
                    lo = mid
                    # never overwrite a judged-COMPLETE best_good with a merely
                    # hard-core-good span — completeness outranks size (P2c refined)
                    if best_good is None or not is_complete(
                            best_good.verdict, best_good.contract_role, adapter, min_score):
                        best_good = trial
                else:
                    hi = mid
                    if v.score > getattr(best.verdict, "score", -1):
                        best = replace(trial)
            if best_good is not None:
                if is_complete(best_good.verdict, best_good.contract_role, adapter, min_score):
                    _attach_judge_stats(best_good, sentences)
                    return _stamp_trims(best_good), None  # the largest passing sub-span wins (P2c)
                best = best_good            # hard-core-good sub-span ships via best-partial

    # keep a best-partial only if it clears the hard core (topic + purpose + grounded + refs);
    # prefer the last hard-core-passing span when the score-best one fails the hard core.
    if best is not None and not _hard_core_ok(best.verdict) and last_good is not None:
        best = last_good
    if best is not None and _hard_core_ok(best.verdict):
        _attach_judge_stats(best, sentences)
        return _stamp_trims(best), None

    # budget exhausted: asymmetric 3-outcome gate — the judge over-flags, so a KILL requires
    # strictly more evidence than a pass: every fatal reason must quote real span text AND
    # survive a fresh-context confirmation. Unverifiable concerns ship flagged, never kill.
    ship = best if best is not None else cand
    last = getattr(ship, "verdict", None)
    text = _clip_text(sentences, ship.i_start, ship.i_end)
    reasons = list(getattr(last, "failure_reasons", None) or [])
    flags = _verify_failure_reasons(last, text) if last is not None else []
    verified = [fr for fr, ok in zip(reasons, flags) if ok]
    unverified = [fr for fr, ok in zip(reasons, flags) if not ok]
    ship.n_failure_reasons = len(reasons)
    ship.n_verified = len(verified)
    ship.verified_kinds = tuple(f.kind for f in verified)        # W25-G kind-level mirror
    ship.unverified_kinds = tuple(f.kind for f in unverified)    # (phantom_quotable_rate)

    confirmed: list = []
    outage: dict = {}
    if verified:                             # confirm_kill runs ONLY on the kill path
        conf = confirm_kill(text, verified, outage)
        confirmed = [fr for fr, ok in zip(verified, conf) if ok]

    # ── CARD2: card-as-repair — a purely ACCEPT-SIDE rescue of this would-be kill. When
    # EVERY confirmed reason is prereq/reference-family, a grounded context card (shown
    # before the clip) may supply the missing prerequisite / resolve the dangling reference
    # on a re-judge of the SAME span (no growth/trim). On success the clip ships carded; a
    # groundless card or a still-failing carded verdict falls through to the EXISTING kill
    # below. Converts a would-be Rejection into a ship, never creates one (unverified_kill=0).
    if confirmed and _card_rescuable(confirmed):
        card, cv = _card_rescue_verdict(
            {"unit_ids": list(ship.unit_ids), "referential": list(ship.referential),
             "anchor_id": ship.anchor_id},
            text, ship.contract_role or ship.role, adapter, units_by_id, topic,
            visual_summary_fn(ship.start, ship.end),
            reasons=confirmed, introducers=introducers, units=units)
        if card and _card_clears(cv):
            ship.context_card = card
            ship.verdict = cv                # stats/scoring now reflect the carded pass
            ship.warnings = tuple(set(ship.warnings or ()) | {"card_completed"})
            _attach_judge_stats(ship, sentences)
            return _stamp_trims(ship), None

    if confirmed:                            # verified AND fresh-context confirmed → kill
        from .integrity import Rejection
        return None, Rejection(
            cand_id=cand.cand_id, title=cand.title, role=cand.role, stage="repair",
            reason="judge verdict incomplete after repair budget",
            score=(float(last.score) if last is not None else None),
            failure_kinds=[f.kind for f in confirmed],
            final_quality=None, start=cand.start, end=cand.end,
            verified_kinds=tuple(f.kind for f in verified),
            unverified_kinds=tuple(f.kind for f in unverified),
            kill_confirmed=True)

    # nothing survived quote verification + confirmation → never reject on unverifiable
    # evidence: ship the best candidate flagged for downstream visibility (+ scoring dock).
    warn = {"unverified_judge_concerns"}
    if outage:
        warn.add("kill_confirm_unavailable")  # confirmation outage recorded on the shipped record
    ship.warnings = tuple(set(ship.warnings or ()) | warn)
    ship.ship_flagged = True
    _flag_unverified_atomicity(ship)         # this path bypasses _attach_judge_stats
    return _stamp_trims(ship), None
