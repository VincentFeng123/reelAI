"""Coding / programming / software adapter (Phase 4)."""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement
S = RoleSpec


class CodingAdapter(BaseAdapter):
    domain = "coding"
    content_types = ("coding", "programming", "software")

    def _domain_role_specs(self):
        return {r.name: r for r in [
            S("requirement", "What we're building / the spec.", facet="overview"),
            S("code_explanation", "Explaining a piece of code.", facet="other",
              is_anchor=True, anchor_priority=58),
            S("implementation", "Writing/showing code that implements something.", facet="application",
              is_anchor=True, anchor_priority=72),
            S("error", "An error/bug is shown.", facet="other"),
            S("debugging_step", "A step diagnosing/fixing a bug.", facet="application",
              is_anchor=True, anchor_priority=66),
            S("output_validation", "Running it / showing the output is correct.", facet="worked_example"),
        ]}

    def labeling_hints(self):
        return ("This is a coding video. Treat function/class/API names as concepts. An 'implementation' "
                "spans from the requirement through the code to the output check.")

    def concept_hints(self):
        return ("Treat library/function/class/API names and language features as concepts; an "
                "implementation REQUIRES the APIs and prior code it builds on.")

    def _contracts(self):
        return {
            "implementation": CompletenessContract("implementation", (
                E("requirement", ("requirement", "setup"), "required", "before"),
                E("code", ("implementation", "code_explanation"), "required", "within", repeatable=True),
                E("output", ("output_validation", "result"), "required", "after"),
            )),
            "debugging_step": CompletenessContract("debugging_step", (
                E("error", ("error",), "required", "before"),
                E("fix", ("debugging_step", "implementation"), "required", "within", repeatable=True),
                E("output", ("output_validation", "result"), "required", "after"),
            )),
            "code_explanation": CompletenessContract("code_explanation", (
                E("code", ("implementation", "code_explanation"), "required", "within"),
                E("why", ("explanation", "requirement"), "recommended", "after"),
            )),
        }
