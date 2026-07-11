"""Interview / podcast / talk-show adapter (Phase 3 — benefits from diarization).

Attributes questions to the interviewer and answers to the guest (using speaker labels when
diarization is on) so a ``direct_answer`` clip can pull in the ``question`` that prompted it.
Anecdotes, opinions, and counterarguments are the other anchor-worthy moments.
"""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement
S = RoleSpec


class InterviewAdapter(BaseAdapter):
    domain = "interview"
    content_types = ("interview", "podcast", "talk_show")

    def _domain_role_specs(self):
        return {r.name: r for r in [
            S("question", "An interviewer asks a question.", facet="other", is_anchor=False),
            S("direct_answer", "The interviewee directly answers a question.", facet="other",
              is_anchor=True, anchor_priority=70),
            S("story_setup", "The speaker sets up a story or anecdote.", facet="other"),
            S("anecdote", "A personal story or example.", facet="other", is_anchor=True, anchor_priority=68),
            S("supporting_detail", "A detail elaborating an answer.", facet="other"),
            S("opinion", "A stated opinion or take.", facet="other", is_anchor=True, anchor_priority=60),
            S("counterargument", "A rebuttal or opposing view.", facet="other", is_anchor=True, anchor_priority=55),
        ]}

    def labeling_hints(self):
        return ("This is an interview/podcast. Attribute questions to the interviewer and answers to "
                "the guest (use the speaker labels when present). A 'direct_answer' unit should be "
                "paired with the 'question' that prompted it.")

    def _contracts(self):
        return {
            "direct_answer": CompletenessContract("direct_answer", (
                E("question", ("question",), "required", "before"),
                E("answer", ("direct_answer",), "required", "within"),
                E("elaboration", ("supporting_detail", "opinion", "anecdote"), "recommended", "after"),
            )),
            "anecdote": CompletenessContract("anecdote", (
                E("setup", ("story_setup", "question"), "required", "before"),
                E("story", ("anecdote",), "required", "within"),
                E("payoff", ("opinion", "supporting_detail", "result"), "recommended", "after"),
            )),
            "opinion": CompletenessContract("opinion", (
                E("prompt", ("question", "story_setup"), "recommended", "before"),
                E("take", ("opinion", "direct_answer"), "required", "within"),
                E("support", ("supporting_detail", "anecdote", "evidence"), "recommended", "after"),
            )),
        }
