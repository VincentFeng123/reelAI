"""Product-review adapter (Phase 4)."""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement
S = RoleSpec


class ReviewAdapter(BaseAdapter):
    domain = "review"
    content_types = ("product_review", "review")

    def _domain_role_specs(self):
        return {r.name: r for r in [
            S("product_intro", "What is being reviewed / its positioning.", facet="overview"),
            S("criterion", "A criterion the product is judged on.", facet="overview"),
            S("test", "A test / trial demonstrating the criterion.", facet="application"),
            S("finding", "A finding / observation from a test.", facet="other",
              is_anchor=True, anchor_priority=62),
            S("verdict", "The overall verdict / recommendation.", facet="other",
              is_anchor=True, anchor_priority=70),
        ]}

    def labeling_hints(self):
        return ("This is a product review. A finding clip runs from the criterion through the test to "
                "the finding; the verdict summarizes the findings into a recommendation.")

    def _contracts(self):
        return {
            "finding": CompletenessContract("finding", (
                E("criterion", ("criterion", "product_intro", "setup"), "required", "before"),
                E("test", ("test", "demonstration"), "recommended", "within"),
                E("finding", ("finding", "evidence", "result"), "required", "within"),
                E("verdict", ("verdict",), "recommended", "after"),
            )),
            "verdict": CompletenessContract("verdict", (
                E("subject", ("product_intro", "criterion"), "recommended", "before"),
                E("findings", ("finding", "evidence"), "required", "before", repeatable=True),
                E("verdict", ("verdict",), "required", "within"),
            )),
        }
