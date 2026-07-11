"""Recipe / cooking adapter (Phase 4)."""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement
S = RoleSpec


class RecipeAdapter(BaseAdapter):
    domain = "recipe"
    content_types = ("recipe", "cooking")

    def _domain_role_specs(self):
        return {r.name: r for r in [
            S("ingredients", "The ingredients / quantities needed.", facet="overview"),
            S("prep_step", "A preparation step (chop, mix, marinate).", facet="application"),
            S("cooking_action", "An active cooking step (sear, bake, simmer).", facet="application",
              is_anchor=True, anchor_priority=66),
            S("plating", "Plating / final assembly / presentation.", facet="other",
              is_anchor=True, anchor_priority=50),
            S("tasting", "Tasting / adjusting seasoning / the verdict.", facet="other"),
        ]}

    def labeling_hints(self):
        return ("This is a recipe. Treat ingredients and techniques as concepts. A cooking clip runs "
                "from the ingredients through the active step to plating.")

    def _contracts(self):
        return {
            "cooking_action": CompletenessContract("cooking_action", (
                E("ingredients", ("ingredients", "setup"), "required", "before"),
                E("action", ("cooking_action", "prep_step", "procedure"), "required", "within", repeatable=True),
                E("plating", ("plating", "tasting", "result"), "recommended", "after"),
            )),
            "plating": CompletenessContract("plating", (
                E("dish", ("cooking_action", "prep_step"), "recommended", "before"),
                E("plating", ("plating",), "required", "within"),
                E("taste", ("tasting",), "optional", "after"),
            )),
        }
