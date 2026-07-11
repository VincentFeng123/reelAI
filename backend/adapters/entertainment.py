"""Entertainment / music / comedy adapter (Wave 4, GEN2).

Routes music videos, parodies, songs, and comedy — content that has no worked-problem or
lecture structure. Its single anchor role ``moment`` (a hook / punchline / quotable line) uses a
deliberately LENIENT completeness contract: ONE required 'within' element and NO before/after
requirement, so a song/parody line is anchorable. ``moment`` is intentionally NOT in
``base._PROBLEM_ROLES``, so ``required_verdict_fields('moment')`` returns only
``CORE_VERDICT_FIELDS`` — none of the problem_statement/reasoning/result gates that killed every
parody candidate under the lecture adapter. This adapter GATES LESS by construction (never adds a
kill path); the judge kill gate is untouched, so ``unverified_kill`` stays 0.
"""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement
S = RoleSpec


class EntertainmentAdapter(BaseAdapter):
    domain = "entertainment"
    content_types = ("entertainment", "music", "song", "comedy", "parody")

    def _domain_role_specs(self):
        return {r.name: r for r in [
            S("moment", "A hook, punchline, quotable line, chorus, or standout moment.",
              facet="other", is_anchor=True, anchor_priority=70),
        ]}

    def labeling_hints(self):
        return ("This is entertainment (music, a parody/song, or comedy) — NOT a lecture. There is "
                "no worked problem to complete. Label a standout line as 'moment' (a hook, "
                "punchline, quotable line, or chorus); a good clip is a single self-contained "
                "moment, not a step-by-step derivation.")

    def _contracts(self):
        # Lenient contract: one required element, satisfiable by the moment itself or common
        # lead-in roles; nothing required before/after → a bare hook/punchline is complete.
        return {
            "moment": CompletenessContract("moment", (
                E("moment", ("moment", "claim", "evidence", "explanation", "setup"),
                  "required", "within"),
            )),
        }
