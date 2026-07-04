"""The universal content-role ontology (spec §4).

Universal roles are a stable, closed enum: every unit gets exactly one, and eval /
metrics key off this layer so they stay comparable across domains. Domain adapters
add *extra* role strings (see ``adapters/``) without ever touching this enum.
"""
from __future__ import annotations

from enum import Enum


class UniversalRole(str, Enum):
    SETUP = "setup"                     # background, goal, scenario, initial conditions
    DEFINITION = "definition"           # meaning of a term or concept
    CLAIM = "claim"                     # principle, rule, thesis, conclusion
    EXPLANATION = "explanation"         # why or how something works
    INTUITION = "intuition"             # analogy or conceptual interpretation
    PROCEDURE = "procedure"             # steps for performing something
    DEMONSTRATION = "demonstration"     # showing the procedure in action
    EXAMPLE_SETUP = "example_setup"     # problem statement + provided information
    WORKED_STEP = "worked_step"         # one or more steps in a worked example
    RESULT = "result"                   # final answer, output, outcome
    PRACTICE_PROMPT = "practice_prompt" # a question posed to the viewer
    SOLUTION = "solution"               # resolution of a practice prompt
    EVIDENCE = "evidence"               # supporting observation, data, quotation
    MISCONCEPTION = "misconception"     # a common incorrect belief
    CORRECTION = "correction"           # correcting the misconception
    EXCEPTION = "exception"             # limitation, edge case, caveat
    SUMMARY = "summary"                 # condensed recap
    TRANSITION = "transition"           # movement to a new subject
    ADMINISTRATIVE = "administrative"   # sponsorship, announcements, housekeeping
    IRRELEVANT = "irrelevant"           # silence, repetition, filler, unusable

    def __str__(self) -> str:  # so f"{role}" / json is the bare value
        return self.value


ALL_UNIVERSAL_ROLES: tuple[str, ...] = tuple(r.value for r in UniversalRole)

# Structural "connective tissue" — never a clip anchor; safe to trim from a clip's
# leading/trailing edge during boundary refinement.
NON_ANCHOR: frozenset[str] = frozenset({
    UniversalRole.SETUP.value,
    UniversalRole.TRANSITION.value,
    UniversalRole.ADMINISTRATIVE.value,
    UniversalRole.IRRELEVANT.value,
})


def coerce_role(value: str) -> str:
    """Map an arbitrary LLM-emitted role string onto a valid universal role.

    Unknown values fall back to EXPLANATION (a safe, non-anchor-neutral default).
    Domain roles are validated by the adapter, not here.
    """
    v = (value or "").strip().lower().replace(" ", "_")
    return v if v in set(ALL_UNIVERSAL_ROLES) else UniversalRole.EXPLANATION.value
