"""News / news-report adapter (Phase 4)."""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement
S = RoleSpec


class NewsAdapter(BaseAdapter):
    domain = "news"
    content_types = ("news", "news_report")

    def _domain_role_specs(self):
        return {r.name: r for r in [
            S("lede", "The core who/what/when/where of the story.", facet="overview",
              is_anchor=True, anchor_priority=66),
            S("detail", "A supporting fact / development.", facet="other",
              is_anchor=True, anchor_priority=48),
            S("context", "Background giving the story meaning.", facet="other"),
            S("consequence", "The impact / what happens next.", facet="other",
              is_anchor=True, anchor_priority=52),
        ]}

    def labeling_hints(self):
        return ("This is a news report. The lede carries the core facts; a self-contained clip pairs "
                "the lede with its key details and the context needed to understand it.")

    def _contracts(self):
        return {
            "lede": CompletenessContract("lede", (                  # event → actors → background → consequence
                E("lede", ("lede", "claim"), "required", "within"),
                E("actors", ("detail", "lede"), "recommended", "within"),
                E("detail", ("detail", "evidence"), "recommended", "within", repeatable=True),
                E("context", ("context", "explanation"), "recommended", "after"),
            )),
            "consequence": CompletenessContract("consequence", (
                E("lede", ("lede", "detail", "setup"), "required", "before"),
                E("consequence", ("consequence",), "required", "within"),
                E("context", ("context", "explanation"), "recommended", "after"),
            )),
        }
