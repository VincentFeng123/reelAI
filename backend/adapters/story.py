"""Storytelling / narrative adapter (Phase 4)."""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement
S = RoleSpec


class StoryAdapter(BaseAdapter):
    domain = "story"
    content_types = ("story", "storytelling")

    def _domain_role_specs(self):
        return {r.name: r for r in [
            S("rising_action", "Escalating events building tension.", facet="other",
              is_anchor=True, anchor_priority=55),
            S("climax", "The turning point / peak of the story.", facet="other",
              is_anchor=True, anchor_priority=72),
            S("resolution", "How it resolves / the aftermath.", facet="other",
              is_anchor=True, anchor_priority=58),
        ]}

    def labeling_hints(self):
        return ("This is a narrative. Use setup for the premise, rising_action for escalation, climax "
                "for the turning point, resolution for the aftermath. A clip should reach a beat.")

    def _contracts(self):
        return {
            "climax": CompletenessContract("climax", (
                E("setup", ("setup", "definition"), "required", "before"),
                E("rising_action", ("rising_action", "explanation"), "recommended", "before", repeatable=True),
                E("climax", ("climax",), "required", "within"),
                E("resolution", ("resolution", "result"), "recommended", "after"),
            )),
            "resolution": CompletenessContract("resolution", (
                E("buildup", ("climax", "rising_action"), "required", "before"),
                E("resolution", ("resolution",), "required", "within"),
            )),
        }
