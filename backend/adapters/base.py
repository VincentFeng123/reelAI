"""Adapter framework core (spec §4, §11).

An *adapter* parameterizes the structure-first pipeline for a content genre. It exposes:
- the role menu (universal roles + domain extensions) the labeler chooses from,
- which roles are worth anchoring a clip on (and their priority),
- a per-anchor-role *completeness contract* (the ordered narrative a self-contained clip
  of that kind needs), which the assembler/judge use to close context and validate,
- prompt hints for labeling / concept extraction,
- a role → legacy-facet mapping so the existing cut/export/frontend contract is preserved.

``BaseAdapter`` implements everything generically from three small pieces a concrete
adapter supplies (``_domain_role_specs``, ``_contracts``, and optional hints), so a new
domain is one short class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from ..roles import ALL_UNIVERSAL_ROLES, NON_ANCHOR, UniversalRole


# ── data structures Part B consumes ──────────────────────────────────────────
@dataclass(frozen=True)
class RoleSpec:
    name: str
    description: str
    facet: str = "other"          # legacy facet enum for cut/export/frontend
    is_anchor: bool = False       # can this role seed a clip?
    anchor_priority: int = 0      # higher = clip first
    clippable_alone: bool = False # rarely standalone-worthy without closure


@dataclass(frozen=True)
class ContractElement:
    key: str                            # "problem_statement"
    roles: tuple[str, ...]              # unit roles that SATISFY this slot
    necessity: str = "required"         # required | recommended | optional
    position: str = "within"            # before | within | after (relative to anchor)
    repeatable: bool = False            # e.g. worked_step may occur N times
    description: str = ""


@dataclass(frozen=True)
class CompletenessContract:
    anchor_role: str
    elements: tuple[ContractElement, ...]         # ORDERED — the required narrative shape
    close_concept_prerequisites: bool = True      # pull concepts_required via the dep graph
    max_span_units: int | None = None
    notes: str = ""


E = ContractElement  # local shorthand for tables below


# ── the stable universal role table (facets + anchor policy) ─────────────────
def _rs(name, desc, facet="other", anchor=False, prio=0, alone=False) -> RoleSpec:
    return RoleSpec(name, desc, facet=facet, is_anchor=anchor, anchor_priority=prio,
                    clippable_alone=alone)

R = UniversalRole
UNIVERSAL_ROLE_SPECS: dict[str, RoleSpec] = {
    R.SETUP.value:          _rs("setup", "Background, goal, scenario, or initial conditions.", "overview"),
    R.DEFINITION.value:     _rs("definition", "The meaning of a term or concept.", "definition", True, 68),
    R.CLAIM.value:          _rs("claim", "A principle, rule, thesis, or conclusion.", "other", True, 70),
    R.EXPLANATION.value:    _rs("explanation", "Why or how something works.", "other", True, 45),
    R.INTUITION.value:      _rs("intuition", "An analogy or conceptual interpretation.", "intuition", True, 48),
    R.PROCEDURE.value:      _rs("procedure", "Steps for performing something.", "application", True, 55),
    R.DEMONSTRATION.value:  _rs("demonstration", "Showing the procedure in action.", "application", True, 60),
    R.EXAMPLE_SETUP.value:  _rs("example_setup", "A problem statement and the information provided.", "worked_example"),
    R.WORKED_STEP.value:    _rs("worked_step", "One or more steps in a worked example.", "worked_example"),
    R.RESULT.value:         _rs("result", "The final answer, output, or outcome.", "worked_example", True, 80),
    R.PRACTICE_PROMPT.value:_rs("practice_prompt", "A question posed to the viewer to solve.", "worked_example", True, 52),
    R.SOLUTION.value:       _rs("solution", "The resolution of a practice prompt.", "worked_example", True, 60),
    R.EVIDENCE.value:       _rs("evidence", "A supporting observation, datum, or quotation.", "other", True, 42),
    R.MISCONCEPTION.value:  _rs("misconception", "A common incorrect belief.", "other"),
    R.CORRECTION.value:     _rs("correction", "Correcting a misconception.", "other", True, 50),
    R.EXCEPTION.value:      _rs("exception", "A limitation, edge case, or caveat.", "other"),
    R.SUMMARY.value:        _rs("summary", "A condensed recap.", "overview", True, 40),
    R.TRANSITION.value:     _rs("transition", "Moving to a new subject.", "other"),
    R.ADMINISTRATIVE.value: _rs("administrative", "Sponsorship, announcements, housekeeping.", "other"),
    R.IRRELEVANT.value:     _rs("irrelevant", "Silence, repetition, filler, or unusable material.", "other"),
}


# ── the generic completeness contracts every adapter inherits ────────────────
GENERIC_CONTRACTS: dict[str, CompletenessContract] = {
    "definition": CompletenessContract("definition", (
        E("term", ("definition",), "required", "within"),
        E("unpacking", ("explanation", "intuition", "example_setup"), "recommended", "after"),
    )),
    "claim": CompletenessContract("claim", (
        E("statement", ("claim", "result"), "required", "within"),
        E("support", ("evidence", "explanation", "example_setup", "demonstration"), "recommended", "within"),
        E("caveat", ("exception", "correction"), "optional", "after"),
    )),
    "result": CompletenessContract("result", (          # the worked-example payoff anchor
        E("problem_statement", ("example_setup", "practice_prompt", "setup"), "required", "before"),
        # 'calculation'/'derivation' mirror arcs.STEP_ROLES: the audited kinematics example's
        # steps AND answer were labeled 'calculation' — the result contract must satisfy on
        # those roles or P1a binds the span to the elementless 'procedure' contract and the
        # problem_statement/reasoning/result judge gates are bypassed.
        E("solution_steps", ("worked_step", "calculation", "derivation", "procedure", "demonstration"),
          "required", "within", repeatable=True),
        # calculation-as-final (arcs.py): the answer may live in the last 'calculation' unit.
        E("result", ("result", "solution", "calculation", "derivation"), "required", "within"),
        E("interpretation", ("explanation", "intuition", "exception"), "recommended", "after"),
    ), max_span_units=14),
    "solution": CompletenessContract("solution", (
        E("prompt", ("practice_prompt", "example_setup"), "required", "before"),
        E("steps", ("worked_step", "procedure"), "recommended", "within", repeatable=True),
        E("answer", ("solution", "result", "calculation"), "required", "within"),
    )),
    "practice_prompt": CompletenessContract("practice_prompt", (
        E("question", ("practice_prompt",), "required", "within"),
        E("data", ("example_setup", "setup"), "recommended", "before"),
        # deliberately NO solution element — a practice clip stops before the answer.
    )),
    "procedure": CompletenessContract("procedure", (
        E("goal", ("setup", "definition"), "recommended", "before"),
        E("steps", ("procedure", "worked_step", "demonstration"), "required", "within", repeatable=True),
        E("outcome", ("result", "demonstration"), "recommended", "after"),
    )),
    "demonstration": CompletenessContract("demonstration", (
        E("setup", ("setup", "definition", "example_setup"), "recommended", "before"),
        E("demo", ("demonstration", "procedure"), "required", "within"),
        E("outcome", ("result",), "optional", "after"),
    )),
    "explanation": CompletenessContract("explanation", (
        E("subject", ("setup", "definition", "claim"), "recommended", "before"),
        E("body", ("explanation", "intuition"), "required", "within"),
        E("example", ("example_setup", "demonstration"), "optional", "after"),
    )),
    "correction": CompletenessContract("correction", (
        E("misconception", ("misconception", "claim"), "required", "before"),
        E("correction", ("correction", "explanation", "evidence"), "required", "within"),
    )),
}

# Verdict booleans that gate completeness only for problem-shaped contracts. The assembly/
# judging path keys these off the CONTENT-BOUND contract_role (contracts.choose_contract),
# not the anchor role — a claim-anchored span that swallowed a worked problem gets gated
# as a worked example (P1).
_PROBLEM_ROLES = frozenset({"result", "solution", "worked_example", "derivation", "calculation"})
CORE_VERDICT_FIELDS = ("topic_identifiable", "purpose_identifiable",
                       "all_references_resolved", "prerequisites_satisfied", "source_grounded")


class BaseAdapter:
    domain: str = "generic"
    content_types: tuple[str, ...] = ()

    # ── the three pieces a concrete adapter supplies ──
    def _domain_role_specs(self) -> dict[str, RoleSpec]:
        return {}

    def _contracts(self) -> dict[str, CompletenessContract]:
        return {}

    def labeling_hints(self) -> str:
        return ""

    def concept_hints(self) -> str:
        return ""

    # ── merged views (Part A + Part B call these) ──
    def role_specs(self) -> dict[str, RoleSpec]:
        return {**UNIVERSAL_ROLE_SPECS, **self._domain_role_specs()}

    def valid_roles(self) -> frozenset[str]:
        return frozenset(self.role_specs()) | frozenset(ALL_UNIVERSAL_ROLES)

    def role_menu(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self.role_specs().values())

    def completeness_contracts(self) -> dict[str, CompletenessContract]:
        return {**GENERIC_CONTRACTS, **self._contracts()}

    def contract_for(self, role: str):
        return self.completeness_contracts().get(role)

    def facet_for(self, role: str) -> str:
        spec = self.role_specs().get(role)
        return spec.facet if spec else "other"

    # ── anchor policy (derived from RoleSpec priorities) ──
    def anchor_roles(self) -> list[str]:
        specs = [s for s in self.role_specs().values()
                 if s.is_anchor and s.anchor_priority >= config.ANCHOR_MIN_PRIORITY
                 and s.name not in NON_ANCHOR]
        specs.sort(key=lambda s: s.anchor_priority, reverse=True)
        return [s.name for s in specs]

    def is_anchor_role(self, role: str) -> bool:
        spec = self.role_specs().get(role)
        return bool(spec and spec.is_anchor and role not in NON_ANCHOR)

    def anchor_priority(self, role: str) -> float:
        spec = self.role_specs().get(role)
        return (spec.anchor_priority / 100.0) if spec else 0.0

    # ── Part-B completeness helpers ──
    # `role` here is the role whose CONTRACT governs the clip: assembly/judging callers pass
    # the content-bound contract_role (rebound after every span mutation), never the raw
    # anchor role. The anchor role remains on Candidate.role / spec["role"] for provenance.
    def required_elements(self, role: str) -> list[str]:
        c = self.contract_for(role)
        return [e.key for e in c.elements if e.necessity == "required"] if c else []

    def required_verdict_fields(self, role: str) -> list[str]:
        fields = list(CORE_VERDICT_FIELDS)
        if role in _PROBLEM_ROLES:
            fields += ["problem_statement_complete", "reasoning_complete", "result_complete"]
        elif role == "practice_prompt":
            fields += ["problem_statement_complete"]
        return fields
