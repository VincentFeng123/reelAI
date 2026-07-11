"""Debate adapter (Phase 4)."""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement
S = RoleSpec


class DebateAdapter(BaseAdapter):
    domain = "debate"
    content_types = ("debate",)

    def _domain_role_specs(self):
        return {r.name: r for r in [
            S("position", "A stance/thesis a side is arguing for.", facet="overview"),
            S("argument", "A reasoned argument advancing a position.", facet="other",
              is_anchor=True, anchor_priority=68),
            S("evidence", "Data/example/quotation supporting an argument.", facet="other",
              is_anchor=True, anchor_priority=44),
            S("rebuttal", "A rebuttal or opposing view to an argument.", facet="comparison",
              is_anchor=True, anchor_priority=58),
        ]}

    def labeling_hints(self):
        return ("This is a debate. A self-contained clip states the position, gives the argument and "
                "its evidence, and (ideally) the rebuttal it answers or provokes.")

    def _contracts(self):
        return {
            "argument": CompletenessContract("argument", (
                E("position", ("position", "claim", "setup"), "required", "before"),
                E("argument", ("argument", "claim"), "required", "within"),
                E("evidence", ("evidence", "example_setup", "demonstration"), "recommended", "within"),
                E("rebuttal", ("rebuttal", "correction"), "recommended", "after"),
            )),
            "rebuttal": CompletenessContract("rebuttal", (
                E("target", ("argument", "position", "claim"), "required", "before"),
                E("rebuttal", ("rebuttal",), "required", "within"),
                E("support", ("evidence", "explanation"), "recommended", "within"),
            )),
        }
