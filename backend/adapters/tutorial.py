"""Tutorial / how-to adapter (Phase 4).

Mostly reuses the universal procedure/demonstration roles; adds a light how-to pack and
tightens the goal→steps→result contract.
"""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement
S = RoleSpec


class TutorialAdapter(BaseAdapter):
    domain = "tutorial"
    content_types = ("tutorial", "howto", "how_to")

    def _domain_role_specs(self):
        return {r.name: r for r in [
            S("objective", "What the viewer will accomplish / the end goal.", facet="overview"),
            S("action_step", "A concrete step the viewer performs.", facet="application",
              is_anchor=True, anchor_priority=58),
            S("result_check", "Showing the step/result worked as intended.", facet="worked_example"),
            S("tip", "A tip, shortcut, or warning.", facet="other"),
        ]}

    def labeling_hints(self):
        return ("This is a how-to tutorial. A complete clip runs from the goal through the concrete "
                "steps to the visible result. Label sponsor/subscribe boilerplate administrative.")

    def _contracts(self):
        return {
            "procedure": CompletenessContract("procedure", (        # goal → prerequisites → steps → result
                E("goal", ("objective", "setup"), "required", "before"),
                E("prerequisites", ("definition", "setup"), "recommended", "before"),
                E("steps", ("procedure", "action_step", "worked_step", "demonstration"), "required",
                  "within", repeatable=True),
                E("result", ("result_check", "result", "demonstration"), "recommended", "after"),
            )),
            "demonstration": CompletenessContract("demonstration", (
                E("goal", ("objective", "setup"), "required", "before"),
                E("demo", ("demonstration", "action_step", "procedure"), "required", "within", repeatable=True),
                E("result", ("result_check", "result"), "recommended", "after"),
            )),
            "action_step": CompletenessContract("action_step", (
                E("goal", ("objective", "setup"), "recommended", "before"),
                E("step", ("action_step", "procedure"), "required", "within", repeatable=True),
                E("result", ("result_check", "result"), "recommended", "after"),
            )),
        }
