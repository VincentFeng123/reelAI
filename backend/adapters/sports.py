"""Sports / sports-analysis adapter (Phase 4)."""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement
S = RoleSpec


class SportsAdapter(BaseAdapter):
    domain = "sports"
    content_types = ("sports", "sports_analysis")

    def _domain_role_specs(self):
        return {r.name: r for r in [
            S("situation", "The game situation / setup before a play.", facet="overview"),
            S("play", "The play / action as it happens.", facet="application",
              is_anchor=True, anchor_priority=60),
            S("outcome", "The immediate result of the play.", facet="worked_example",
              is_anchor=True, anchor_priority=50),
            S("analysis", "Breakdown of why it worked / tactical insight.", facet="other",
              is_anchor=True, anchor_priority=64),
        ]}

    def labeling_hints(self):
        return ("This is sports analysis. A clip runs from the situation through the play and its "
                "outcome to the analysis of why it happened.")

    def _contracts(self):
        return {
            "analysis": CompletenessContract("analysis", (
                E("situation", ("situation", "setup"), "required", "before"),
                E("play", ("play", "demonstration"), "required", "within"),
                E("outcome", ("outcome", "result"), "required", "within"),
                E("analysis", ("analysis", "explanation"), "required", "after"),
            )),
            "play": CompletenessContract("play", (
                E("situation", ("situation", "setup"), "recommended", "before"),
                E("play", ("play",), "required", "within"),
                E("outcome", ("outcome", "result"), "required", "after"),
            )),
        }
