"""Guarded Gemini educational clip segmentation.

Production uses one low-thinking Flash call over the whole timestamped transcript,
then applies deterministic quality, context, grounding, filler, and deduplication
guards. Legacy routing and enrichment helpers remain available only for isolated
evaluation compatibility; the public production adapter never dispatches them.

The public contract stays ``segment_clips(...) -> (clips, notes)``.  Model names,
routing decisions, and call telemetry are logged internally and never added to a
clip, note, or API response.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from functools import lru_cache
from threading import BoundedSemaphore, Lock
from typing import Callable, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

from .. import config

ProgressCb = Optional[Callable[[float, str], None]]
CancelledCb = Optional[Callable[[], bool]]

_SELECTOR_CALL_LIMIT = 3
_SELECTOR_SLOT_POLL_S = 0.05
_SELECTOR_OPERATIONS = frozenset({
    "boundary_selection",
    "flash_boundary_selector",
    "flash_single_candidate",
    "pro_authoritative",
    "pro_fallback",
})
_selector_call_slots = BoundedSemaphore(_SELECTOR_CALL_LIMIT)

log = logging.getLogger("clipper.segment")

_WORD_RE = re.compile(r"[^\W_]+(?:['\u2018\u2019\u02bc][^\W_]+)*", re.UNICODE)
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'"})
_CandidateId = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_BoundaryQuote = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_ClipTitle = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_LearningObjective = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_Facet = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_EvidenceQuote = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_IntentConstraintId = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)
]
_OptionalReason = Annotated[
    str, StringConstraints(strip_whitespace=True)
]
_NON_SPEECH_MARKER_PATTERN = (
    r"(?:\[\s*(?:(?:(?:theme|intro|outro|background)\s+)?music|applause|"
    r"laughter|cheering|inaudible)\s*\]|"
    r"\(\s*(?:(?:(?:theme|intro|outro|background)\s+)?music|applause|"
    r"laughter|cheering|inaudible)\s*\)|"
    r"[\u2669-\u266c]+)"
)
_NON_SPEECH_MARKER_RE = re.compile(_NON_SPEECH_MARKER_PATTERN, re.IGNORECASE)
_STRUCTURAL_FILLER_RE = re.compile(
    rf"(?:{_NON_SPEECH_MARKER_PATTERN}|\b(?:thanks? for watching|have a great day|see you next time|"
    r"like and subscribe|please subscribe|"
    r"subscribe to (?:this|the|my|our) channel|today'?s sponsor|"
    r"check out (?:our|the) video|"
    r"administrative (?:note|announcement)|course (?:administration|logistics)|"
    r"(?:a\s+)?(?:quick|brief|short) (?:aside|tangent)|"
    r"look at (?!this\b)(?!(?:the\s+)?(?:animation|chart|diagram|drawing|equation|"
    r"figure|graph|image|map|object|screen|shape|simulation|slide|table)\b)"
    r"(?:the\s+)?"
    r"(?![^.!?]{0,80}\b(?:after|as|because|before|by|how|if|that|the way|when|"
    r"where|whether|which|while|why)\b)"
    r"(?=(?:[^\W_]+(?:['’\-][^\W_]+)?\s*){1,5}[.!?])"
    r"[^.!?]{1,80}(?=[.!?](?:\s|$))|"
    r"(?<!\bare )(?<!\bis )(?<!\bwere )(?<!\byou are )(?<!\byou['’]re )"
    r"\bwelcome(?:\s+back)?(?:\s+to\s+(?:(?:this|the|my|our)\s+)?"
    r"(?:channel|video|lesson|course|show|episode|series)\b|\s+to\b|"
    r"(?=[!,.]|\s*$))|"
    r"in this (?:video|lesson|course) we(?:['’]ll| will)|"
    r"before we (?:begin|get started)|let(?:['’]?s| us) move on|"
    r"next we(?:['’]ll| will)|cover (?:that|this) in (?:this|the) course)\b|"
    r"(?:but\s+)?we(?:['’]ll| will)\s+(?:(?:talk|discuss|learn|say|cover|"
    r"explore|explain)\s+"
    r"(?:more\s+)?about\s+(?:that|this|it)|(?:talk|discuss)\s+about\s+"
    r"(?:that|this|it)\s+more|(?:discuss|cover|revisit|explore|explain)\s+"
    r"(?:that|this|it)(?:\s+more)?|(?:return|come\s+back)\s+to\s+"
    r"(?:that|this|it))\s+(?:next\s+time|later|in\s+(?:a|the)\s+"
    r"(?:next|future)\s+(?:video|lesson|section|episode))\b|"
    r"(?:^|[.!?]\s+)(?:(?:all right|alright|okay|ok|so|now|well|yeah)"
    r"\s*[,;:]?\s+)*(?:"
    r"let(?:['’]?s| us)\s+(?:begin|get started|start|dive in|delve)"
    r"\s*[.!?](?=\s|$)|"
    r"(?:cool|hey|fun fact|brilliant)\s*[!,.](?=\s|$)|"
    r"there(?:['’]s|\s+is)\s+(?:just\s+)?(?:a|one)\s+"
    r"(?:issue|problem|thing)\s*:|"
    r"(?:oh\s*[,;:]?\s*)?(?:yeah\s*[,;:]?\s*)?by the way\b|"
    r"sponsored by\b|"
    r"we (?:made|have) (?:a|an|another|whole) video "
    r"(?:about|explaining|on)\b|my name is\b|"
    r"to (?:recap|summarize)(?:\s*[,;:]|\s*$)|"
    r"in summary(?:\s*[,;:]|\s+(?=(?:we|the|this|these|there|our|a|an)\b))|"
    r"we (?:are|'re) reaching (?:a|the) crossroad now\b|"
    r"we(?:['’]ve| have) already (?:done|covered|finished)\b.{0,80}))",
    re.IGNORECASE,
)
_INTERNAL_INTERRUPTION_MARKER_RE = re.compile(
    r"\b(?:today'?s sponsor|sponsored by|administrative (?:note|announcement)|"
    r"course (?:administration|logistics)|(?:a\s+)?(?:quick|brief|short) "
    r"(?:aside|tangent)|housekeeping)\b",
    re.IGNORECASE,
)
_COURSE_ADMIN_STRONG_RE = re.compile(
    r"\b(?:(?:this|the|our) (?:course|class)\b[^.!?]{0,120}?"
    r"(?:pass[ /-]?fail|registration|enrollment|grading|requirements?|schedule)|"
    r"(?:course|class) (?:administration|logistics|organization|polic(?:y|ies)|"
    r"requirements?|schedule|registration|enrollment|grading)|"
    r"(?:look(?:ed|ing)? at|from) (?:the )?registration list|"
    r"registration list\b[^.!?]{0,100}\b(?:students?|people|disciplines?|majors?)|"
    r"wait\s*list|office hours?|syllabus|late work|due dates?|lab sections?|"
    r"teaching assistants?)\b",
    re.IGNORECASE,
)
_COURSE_ASSESSMENT_ADMIN_RE = re.compile(
    r"\b(?:(?:when|where) (?:the )?(?:midterms?|exams?|quizzes?) (?:are|happen)|"
    r"(?:midterms?|exams?|quizzes?|homework|assignments?) (?:are )?"
    r"(?:due|graded|posted|required|scheduled)|"
    r"(?:course|class) (?:midterms?|exams?|quizzes?|homework|assignments?))\b",
    re.IGNORECASE,
)
_COURSE_AUDIENCE_LOGISTICS_RE = re.compile(
    r"\b(?:(?:small|large) class\b[^.!?]{0,100}\b(?:students?|you)|"
    r"(?:small|large) number of (?:students|you)|"
    r"people from (?:every|many|different) (?:discipline|major|year)s?|"
    r"(?:listen to|get input from|get feedback from|hear from) you(?: all)?|"
    r"(?:capture|get|understand) the sense of you|"
    r"do well in (?:this|the) course|(?:run|organize) (?:this|the) course|"
    r"nitty gritty of the organization|(?:start|do|make) (?:some )?introductions|"
    r"when (?:the )?(?:exams?|midterms?|quizzes?) are)\b",
    re.IGNORECASE,
)
_SPEAKER_OR_STARTUP_FILLER_RE = re.compile(
    r"^\s*(?:[A-Z][A-Z .'-]{2,}:\s*)?(?:"
    r"ok(?:ay)?|all right|alright|we(?:['’]re| are) going to get going|"
    r"let(?:['’]?s| us) get started|get(?:ting)? started)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_AUDIENCE_COMPOSITION_RE = re.compile(
    r"^\s*(?:and\s+)?we have (?:people|students) (?:from|across)\b"
    r"[^.!?]{0,120}\b(?:disciplines?|majors?|years?|departments?|schools?)\b",
    re.IGNORECASE,
)
_COURSE_ADMIN_CONVEY_RE = re.compile(
    r"\b(?:we|i) (?:need|have) to (?:convey|tell|explain) (?:this|that|the )?"
    r"(?:course (?:information|details?|requirements?|logistics|organization)|"
    r"registration details?|grading requirements?|exam schedule)\b",
    re.IGNORECASE,
)
_COURSE_PREVIEW_RE = re.compile(
    r"\b(?:this is what we(?:['’]re| are) going to do today|"
    r"we(?:['’]re| are) going to talk about the (?:organization|logistics)|"
    r"what you might like to see more of)\b",
    re.IGNORECASE,
)
_COURSE_ADMIN_BRIDGE_RE = re.compile(
    r"^\s*(?:we|i) need to tell you this\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_INSTRUCTIONAL_PREVIEW_PREFIX_RE = re.compile(
    r"^\s*(?:(?:and|so)\s+then\s+)?"
    r"(?:i|we)(?:['’]ll|\s+will|['’]m\s+going\s+to|\s+am\s+going\s+to|"
    r"['’]re\s+going\s+to|\s+are\s+going\s+to)\s+"
    r"(?:take|walk|guide)\s+you\s+through\b[^.!?]{0,220}?"
    r"(?:to\s+(?:show|explain|demonstrate)\s+(?:to\s+)?you\s+"
    r"(?:that|how|why)|so\s+(?:that\s+)?you\s+can\s+"
    r"(?:see|understand)\s+(?:that|how|why))\s+",
    re.IGNORECASE,
)
_VISUAL_DEPENDENCY_RE = re.compile(
    r"\b(?:as you can see|as shown (?:here|on (?:the )?screen)|"
    r"on (?:the )?screen|this (?:diagram|figure|chart|graph|image|slide|drawing)|"
    r"these (?:diagrams|figures|charts|graphs|images|pictures|slides)\b|"
    r"the (?:diagram|figure|chart|graph|image|slide) (?:shows|illustrates)|"
    r"what (?:you(?:['’]re| are)|we(?:['’]re| are)) seeing (?:here|on screen)|"
    r"look (?:here(?=\s*[.!?]*\s*$)|at this(?=\s*[.!?]*\s*$)|"
    r"at (?:(?:this|the) )?(?:animation|chart|diagram|drawing|equation|"
    r"figure|graph|image|map|object|screen|shape|simulation|slide|table))|"
    r"over here|right over there|watch (?:this|what happens)(?=\s*[.!?]*\s*$)|"
    r"looks? (?:something )?like (?:that|this(?=\s*[.!?]*\s*$))|"
    r"(?:I(?:['’]m| am)? (?:drawing|writing)|I(?:['’]ll| will) (?:draw|write))"
    r"(?=\s*(?:(?:(?:this|that|the|a)\s+)?(?:chart|diagram|figure|graph|line|"
    r"map|shape)|here|on (?:the )?screen)\b|\s*[.!?]*\s*$))",
    re.IGNORECASE,
)
_DEICTIC_POINT_DEPENDENCY_RE = re.compile(
    r"\b(?:between|pick|take|choose)\s+"
    r"(?P<first>this|that)\s+point\s+(?:and|to)\s+"
    r"(?P<second>this|that)\s+point\b",
    re.IGNORECASE,
)
_SENTENCE_LOCAL_VISUAL_SIGNAL_RE = re.compile(
    r"\b(?:look\s+(?:here|at\s+this)|watch\s+(?:this|what\s+happens)|"
    r"looks?\s+(?:something\s+)?like\s+this|"
    r"i(?:['’]m|\s+am)?\s+(?:drawing|writing)|"
    r"i(?:['’]ll|\s+will)\s+(?:draw|write))\b",
    re.IGNORECASE,
)
_DEICTIC_POINT_DEFINITION_RE = re.compile(
    r"(?:\b(?:let|define|take|suppose|assume)\s+|\band\s+)"
    r"(?P<label>this|that)\s+point\b[^.!?]{0,80}?"
    r"\b(?:at|be|coordinate|equal|x|y)\b",
    re.IGNORECASE,
)
_DANGLING_TAIL_PREFIX_RE = re.compile(
    r"^\s*tail[.!?]\s+"
    r"(?:(?:[Uu]m+|[Uu]h+|[Ww]ell)[, ]+)?"
    r"(?:[Ii]t|[Tt]his|[Tt]hat|[Tt]hey|[Tt]hese|[Tt]hose|[Hh]e|[Ss]he)\b"
)
_OPENING_COMPARATIVE_FRAGMENT_RE = re.compile(
    r"^\s*(?:much\s+)?(?:more|less)\s+[a-z][a-z'-]*\?(?:\s+|$)",
    re.IGNORECASE,
)
_EXISTENTIAL_OPENING_RE = re.compile(
    r"^\s*there\s+(?:is|are|was|were)\s+"
    r"(?:(?:(?:a|an|no|some|many|several|multiple|numerous|few)|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten)|\d+)\s+)?"
    r"(?P<tail>[^\W_][^\n.!?]*?)\s*$",
    re.IGNORECASE,
)
_EXISTENTIAL_UNRESOLVED_RE = re.compile(
    r"\b(?:this|these|those|they|them|their|theirs|he|him|his|she|her|"
    r"hers|it|its|mine|ours|yours|here|there|"
    r"such|same|former|latter|above|below|previous|following|earlier|more|"
    r"other|others|ones)\b",
    re.IGNORECASE,
)
_EXISTENTIAL_CONTEXTUAL_THAT_RE = re.compile(
    r"(?:^|\b(?:of|for|in|on|about|from|by|to|with|under|over)\s+)that\b",
    re.IGNORECASE,
)
_EXISTENTIAL_TERMINAL_REFERENCE_RE = re.compile(
    r"(?:\b(?:that|so)|\bthat\s+[a-z0-9][a-z0-9'-]*)\s*$",
    re.IGNORECASE,
)
_EXISTENTIAL_DEMONSTRATIVE_THAT_RE = re.compile(
    r"(?:\b(?:why|how|when|where|which|who|whose)\s+that\s+[a-z0-9]|"
    r"\bthat\s+(?:[a-z][a-z'-]*(?:tion|sion|ment|ness|ity|ence|ance|"
    r"ship|hood|ism|ure|age|acy|ics)|answer|cell|enzyme|example|gene|idea|"
    r"method|one|pathway|problem|protein|reaction|result|step|subject|theory|"
    r"thing|topic)\b)",
    re.IGNORECASE,
)
_EXISTENTIAL_BACK_REFERENCE_RE = re.compile(
    r"\b(?:mentioned|shown|discussed|described|introduced|seen|noted|defined|"
    r"explained)\s+(?:earlier|before|above|previously)\b",
    re.IGNORECASE,
)
_EXISTENTIAL_SCOPE_RE = re.compile(
    r"(?:\b(?:of|in|within|among|between|during|under|on|for|about|behind|"
    r"from|by|to|across|inside|outside)\s+(?:the\s+|a\s+|an\s+)?[a-z]|"
    r"\b(?:that|which|who|whose|where|when|why|how)\s+[a-z]|"
    r"\b(?:i|we|you)\s+(?:(?:can|could|will|would|should|do|did)\s+)?"
    r"[a-z][a-z'-]*\s+(?:the\s+|a\s+|an\s+)?[a-z0-9][a-z0-9'-]*|"
    r"\b[a-z][a-z'-]*\s+[a-z][a-z'-]*(?:ing|ed)\s+"
    r"(?:the\s+|a\s+|an\s+)?[a-z0-9]|"
    r",\s*(?:(?:namely|specifically|called)\s+)?[a-z0-9][a-z0-9'-]*"
    r"\s+(?:and|or)\s+[a-z0-9][a-z0-9'-]*|[;:]\s*[a-z0-9])",
    re.IGNORECASE,
)
_TERMINAL_CALLBACK_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:look|think|go|turn|refer) back (?:at|to)\b[^.!?]*[.!?]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_TRANSITION_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:all\s+right\s*[,;:]?\s*)?"
    r"let(?:['’]?s|\s+us)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_DISCOURSE_LEADIN_RE = re.compile(
    r"\b(?:but|however|now|so|then)\s+"
    r"(?:occasionally|sometimes|typically|usually)\s*[,.!?]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_EXAMPLE_INTRO_RE = re.compile(
    r"\b(?:or\s+even|and\s+also|or\s+perhaps|but\s+only|including|"
    r"such\s+as|for\s+(?:example|instance))\s*[,;:]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_INCOMPLETE_SUBJECT_RE = re.compile(
    r"\b(?:i|we|you|he|she|it|they|this|that)['’](?:d|ll|m|re|s|ve)"
    r"[.!?]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_BARE_SUBJECT_RE = re.compile(
    r"\b(?:and|as|because|but|if|or|since|so|that|though|unless|until|when|"
    r"where|which|while|who)"
    r"\s+(?:i|we|you|he|she|it|they)\s*$",
    re.IGNORECASE,
)
_TERMINAL_NOMINAL_SUBJECT_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:one|some|many|most|all|each|none)\s+of\b[^.!?]*$",
    re.IGNORECASE,
)
_TERMINAL_PRIME_DIRECTION_RE = re.compile(
    r"\b(?:three|five|3|5)\s+to\s+(?:three|five|3|5)\s*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_ARTICLE_RE = re.compile(
    r"\b(?:a|an|the)\s*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_LINK_RE = re.compile(
    r"\b(?:among|between|how|than|versus|what|when|where|which|who|whose|whom|why)"
    r"\s*[.!?]?[\"')\]]*$",
    re.IGNORECASE,
)
_TERMINAL_HEADLESS_QUANTIFIER_RE = re.compile(
    r"\b(?:among|between|from|into|of|with)\s+"
    r"(?:(?:more|less|fewer)\s+than\s+)?"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"several|many|multiple|\d+)\s*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_MODAL_PREDICATE_RE = re.compile(
    r"\b(?:can|could|may|might|must|shall|should|will|would)"
    r"(?:\s+(?:not|n['’]t))?\s+"
    r"(?:(?:also|generally|just|often|probably|really|still|usually)\s+)?"
    r"(?:appear|be|become|feel|look|remain|seem|sound)\s*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_AUXILIARY_ADVERB_RE = re.compile(
    r"\b(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would)"
    r"(?:\s+(?:not|n['’]t))?\s+"
    r"(?:actually|also|certainly|eventually|generally|hopefully|likely|often|"
    r"possibly|probably|still|usually)\s*$",
    re.IGNORECASE,
)
_TERMINAL_COMPLETE_SHORT_NP_RE = re.compile(
    r"^(?:(?:the|this|that)\s+)?"
    r"(?:first|second|third|final|last|next|other|previous|same)\s+"
    r"(?:case|cases|example|examples|one|ones|step|steps|thing|things|time)"
    r"[.!?]?[\"')\]]*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_DEGREE_RE = re.compile(
    r"\b(?:am|are|be|been|being|feels?|is|looks?|seems?|sounds?|was|were)"
    r"\s+(?:less|more|quite|rather|really|so|too|very)\s*$",
    re.IGNORECASE,
)
_TERMINAL_AMBIGUOUS_DEGREE_RE = re.compile(
    r"\b(?:am|are|be|been|being|feels?|is|looks?|seems?|sounds?|was|were)"
    r"\s+pretty\s*$",
    re.IGNORECASE,
)
_TERMINAL_EXPLICIT_INCOMPLETE_CLAUSE_RE = re.compile(
    r"\b(?:although|because|if|since|unless|until|when|whereas|while)"
    r"\s*[.!?]?[\"')\]]*$",
    re.IGNORECASE,
)
_TERMINAL_COORDINATING_CONJUNCTION_RE = re.compile(
    r"(?<!\bnow\s)\b(?:and|but|or)(?:\s+(?:now|so|then))?"
    r"\s*[.!?]?[\"')\]]*$",
    re.IGNORECASE,
)
_TERMINAL_REQUIRED_COMPLEMENT_RE = re.compile(
    r"(?:\b(?:depends?|relies?)\s+(?:on|upon)|"
    r"\b(?:consists?|results?)\s+(?:of|from|in)|"
    r"\b(?:leads?|refers?|corresponds?|belongs?)\s+to|"
    r"\b(?:is|are|was|were|be)\s+"
    r"(?:based|caused|defined|determined|known|proportional|related)\s+"
    r"(?:as|by|on|to|with))\s*[.!?]?[\"')\]]*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_PREDICATE_HEAD_RE = re.compile(
    r"\b(?:am|are|be|been|being|can|could|did|do|does|had|has|have|is|"
    r"may|might|must|shall|should|was|were|will|would|allows?|causes?|"
    r"contains?|enables?|equals?|includes?|means?|provides?|requires?)"
    r"\s*[.!?]?[\"')\]]*$",
    re.IGNORECASE,
)
_TRAILING_FORWARD_SETUP_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:but\s+)?what happens if\b.*?\?\s*"
    r"(?:now\s*[,]?\s*)?we\s+can(?:not|['’]t)\b[^.!?]*[.!?]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FORWARD_SOLUTION_CONTINUATION_RE = re.compile(
    r"^\s*(?:so\s+)?instead\b",
    re.IGNORECASE,
)
_TOPIC_NAVIGATION_ACTION_PATTERN = (
    r"(?:cover|discuss|move\s+on|switch\s+to|"
    r"talk\s+about|turn\s+to)"
)
_FORWARD_TOPIC_TRANSITION_RE = re.compile(
    r"^\s*(?:(?:all\s+right|okay|ok|so)\s*[,;:]?\s+)*"
    rf"(?:now\s+(?:we\s+(?:got|have|need)\s+to\s+"
    rf"{_TOPIC_NAVIGATION_ACTION_PATTERN}|"
    r"let(?:['’]?s|\s+us)\s+(?:back\s+up|move\s+on|turn\s+to))|"
    r"(?:(?:now|next)\s*[,;:]?\s+)?"
    r"(?:let(?:['’]?s|\s+us)\s+(?:consider|do|go\s+through|look\s+at|take|"
    r"try|work\s+out|work\s+through)|consider|for|try)\s+"
    r"(?:another|one\s+more|the\s+next|a\s+(?:different|new))\s+"
    r"(?:(?:brief|concrete|quick|short|simple|worked)\s+)*"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)|"
    r"(?:now\s+)?our\s+next\s+"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)"
    r"\s+is|"
    rf"next\s+(?:we|i)(?:\s+will|['’]ll)\s+"
    rf"{_TOPIC_NAVIGATION_ACTION_PATTERN}|"
    r"(?:now\s+)?here(?:['’]s|\s+is)\s+"
    r"(?:another|one\s+more|the\s+next|a\s+(?:different|new))\s+"
    r"(?:(?:brief|concrete|quick|short|simple|worked)\s+)*"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)|"
    r"(?:now\s+)?(?:the\s+)?next\s+(?:topic|concept|section|part)|"
    r"let(?:['’]?s|\s+us)\s+(?:back\s+up|move\s+on|turn\s+to))\b",
    re.IGNORECASE,
)
_EXPLICIT_RELATIONAL_OBJECTIVE_RE = re.compile(
    r"\b(?:compare|contrast|distinguish)\b[^.!?]{0,120}\b(?:and|from|with)\b|"
    r"\b(?:connection|interaction|link|relationship)\s+between\b|"
    r"\b(?:connect|link|relate)\b[^.!?]{0,120}\b(?:and|to|with)\b|"
    r"\b(?:in\s+terms\s+of)\b|"
    r"\bderiv(?:e|ed|es|ing)\b[^.!?]{0,100}\bfrom\b|"
    r"\b(?:how|why)\b[^.!?]{1,100}\b(?:affect|cause|define|depend|derive|differ|"
    r"form|impl(?:y|ies)|influence|interact|produce|relate|shape|use|yield)\w*\b|"
    r"\b(?:form|produce|yield)\w*\b|"
    r"\bversus\b|\bvs\.?(?:\s|$)",
    re.IGNORECASE,
)
_EXPLICIT_COMPARISON_OBJECTIVE_RE = re.compile(
    r"\b(?:compare|contrast|distinguish)\b[^.!?]{0,120}\b(?:and|from|with)\b|"
    r"\bversus\b|\bvs\.?(?:\s|$)",
    re.IGNORECASE,
)
_HARD_TOPIC_RESET_RE = re.compile(
    r"(?<!\w)(?:(?:all right|alright|okay|ok|so)\s*[,;:]?\s+)*"
    r"(?:(?:now\s+)?let(?:['’]?s|\s+us)\s+(?:move\s+on\s+to|switch\s+to|"
    r"shift\s+to|turn\s+to|talk\s+about|discuss|cover|look\s+at)|"
    r"(?:now\s+)?we(?:['’]re|\s+are)\s+(?:moving\s+on|switching|shifting|"
    r"turning)\s+to|"
    r"next\s+(?:we|i)(?:['’]ll|\s+will)\s+(?:move\s+on\s+to|switch\s+to|"
    r"shift\s+to|turn\s+to|talk\s+about|discuss|cover|look\s+at)|"
    r"(?:now\s+)?next\s+up\s+is|"
    r"(?:now\s+)?that\s+brings\s+us\s+to|"
    r"(?:now\s+)?moving\s+on\s+to|"
    r"(?:now\s+)?turn\s+(?:our\s+)?attention\s+to|"
    r"(?:now\s+)?(?:let(?:['’]?s|\s+us)\s+(?:consider|do|go\s+through|"
    r"look\s+at|take|try|work\s+out|work\s+through)|consider|for|try)"
    r"(?=\s+(?:another|one\s+more|the\s+next|a\s+(?:different|new))\s+"
    r"(?:(?:brief|concrete|quick|short|simple|worked)\s+)*"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)\b)|"
    r"(?:now\s+)?our\s+next\s+"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)"
    r"\s+is|"
    r"(?:now\s+)?here(?:['’]s|\s+is)"
    r"(?=\s+(?:another|one\s+more|the\s+next|a\s+(?:different|new))\s+"
    r"(?:(?:brief|concrete|quick|short|simple|worked)\s+)*"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)\b)|"
    r"(?:now\s+)?(?:the\s+)?next\s+(?:topic|concept|section)\s+(?:is|covers))"
    r"\s+(?P<subject>[^.!?,;:]{1,100})",
    re.IGNORECASE,
)
_SAME_UNIT_RESET_SUBJECT_RE = re.compile(
    r"^(?:(?:the|a|an|this)\s+)?(?:(?:next|second|third|final)\s+)?"
    r"(?:step|part|case|example|stage|piece|proof|derivation|calculation|"
    r"denominator|numerator)\b",
    re.IGNORECASE,
)
_RELATIONAL_RESET_SUBJECT_RE = re.compile(r"^(?:how|why)\b", re.IGNORECASE)
_SAME_UNIT_NAVIGATION_RE = re.compile(
    r"\b(?:(?:next|second|third|final)\s+)?"
    r"(?:step|part|case|example|stage|piece|proof|derivation|calculation|"
    r"denominator|numerator)\b",
    re.IGNORECASE,
)
_NEXT_SAME_UNIT_CONTINUATION_RE = re.compile(
    r"^\s*(?:(?:all\s+right|alright|okay|ok|so|now)\s*[,;:]?\s+)*"
    r"let(?:['’]?s|\s+us)\s+(?:look\s+at|continue\s+with|go\s+on\s+to|"
    r"discuss|cover|turn\s+to)\b[^.!?]{0,100}"
    r"\b(?:(?:next|second|third|final)\s+)?"
    r"(?:step|part|case|example|stage|piece|proof|derivation|calculation|"
    r"denominator|numerator)\b",
    re.IGNORECASE,
)
_ANAPHORIC_OPENING_RE = re.compile(
    r"^\s*(?:it|this|that|these|those|they|he|she|which|who)\b",
    re.IGNORECASE,
)
_OPENING_DEMONSTRATIVE_REFERENCE_RE = re.compile(
    r"^\s*(?:this|that|these|those)\s+"
    r"(?:answer|approach|assumption|calculation|case|change|condition|difference|"
    r"equation|expression|formula|idea|method|problem|process|quantity|reason|"
    r"relationship|result|rule|solution|step|term|value|variable)\b",
    re.IGNORECASE,
)
_OPENING_BARE_RELATIONAL_PREDICATE_RE = re.compile(
    r"^\s*(?:is|are|was|were)\s+(?:equal|equivalent|proportional|related|"
    r"similar|connected|dependent)\b",
    re.IGNORECASE,
)
_OPENING_DEPENDENT_PREPOSITION_FRAGMENT_RE = re.compile(
    r"^\s*(?:in\s+terms\s+of|relative\s+to|with\s+respect\s+to)\b"
    r"[^,;:.!?]{1,120}\b(?:and|but|so|then)\s+"
    r"(?:i|he|it|she|that|these|they|this|those|we|you)"
    r"(?:['’](?:d|ll|m|re|s|ve)|\s+(?:am|are|can|could|did|do|does|had|"
    r"has|have|is|may|might|must|should|was|were|will|would))\b",
    re.IGNORECASE,
)
_OPENING_SUBJECTLESS_PREDICATE_RE = re.compile(
    r"^\s*(?:(?:is|are|was|were)\b|"
    r"(?:can|could|may|might|must|should|will|would)\s+"
    r"(?:(?:also|just|really|simply|still)\s+)*"
    r"(?:be|cancel|differentiate|divide|multiply|replace|treat|use|view|write)\b)",
    re.IGNORECASE,
)
_OPENING_MATH_CONTINUATION_RE = re.compile(
    r"^\s*(?:times\b|respect\s+to\b|"
    r"[-+]?\d+(?:\.\d+)?[a-z]?\s+(?:but|plus|minus|times|over|squared|cubed)\b|"
    r"[a-z]\s+(?:plus|minus|times|over|squared|cubed|raised\s+to)\b)",
    re.IGNORECASE,
)
_NEXT_DEPENDENT_COMPLEMENT_RE = re.compile(
    r"^\s*(?:of\b(?!\s+course\b)|with\s+respect\s+to\b|respect\s+to\b|"
    r"squared\b|cubed\b|raised\s+to\b|"
    r"which\s+(?:are|equals?|gives?|is|means?|shows?)\b)",
    re.IGNORECASE,
)
_OPENING_INDEPENDENT_OF_FRAME_RE = re.compile(
    r"^\s*of\s+(?:all\s+)?(?:the\s+)?(?:available|following|possible|"
    r"remaining|several|these|those|various)\b",
    re.IGNORECASE,
)
_TERMINAL_OF_COMPLEMENT_HEAD_RE = re.compile(
    r"\b(?:amount|atoms?|average|basis|cause|cells?|coefficient|cosine|"
    r"definition|degree|derivative|determinant|dimension|effect|example|"
    r"function|grams?|group|integral|kind|limit|logarithm|measure|moles?|"
    r"molecules?|number|part|percentage?|probability|product|ratio|rate|"
    r"root|sine|sum|tangent|type|value)\s*$",
    re.IGNORECASE,
)
_NEXT_SYMBOL_LABEL_CONTINUATION_RE = re.compile(
    r"^\s*[a-z][a-z0-9_]*\s+(?:equals?\b|is\b|that(?:['’]?s|\s+is)\b)",
    re.IGNORECASE,
)
_TRAILING_VISUAL_POINTER_RE = re.compile(
    r"\b(?:this|that)\s+right\s+over\s+(?:here|there)\b[^.!?]*[.!?]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_EMBEDDED_IDENTITY_RE = re.compile(
    r"\b(?:that|which)\s+(?:equals?|is)\s+(?:exactly\s+)?what\s+"
    r"[a-z0-9][a-z0-9_+\-^ ]{0,80}\s+(?:equals?|is)\s*$",
    re.IGNORECASE,
)
_TERMINAL_REQUIRED_VISUAL_COMPLEMENT_RE = re.compile(
    r"\b(?:by|through)\s+(?:(?:[a-z][a-z'’-]*ly)\s+){0,2}"
    r"[a-z][a-z'’-]*ing(?:\s+(?:all|both|each|either|just|only|the|"
    r"these|this|those|two))?\s*$",
    re.IGNORECASE,
)
_OPENING_COMPLETE_ORDINAL_SUBJECT_RE = re.compile(
    r"^\s*the\s+(?:earliest|first|initial)\s+"
    r"(?!(?:answer|case|equation|example|part|problem|result|solution|step|thing)\b)"
    r"(?:[a-z][a-z'’-]*\s+){1,4}?"
    r"(?:are|can|did|do|does|had|has|have|is|may|might|must|should|was|were|"
    r"will|would|[a-z][a-z'’-]*(?:ed|es))\b",
    re.IGNORECASE,
)
_NOMINAL_PREDICATE_CONTINUATION_RE = re.compile(
    r"^\s*(?:is|are|was|were|has|have|had|can|could|will|would|should|may|"
    r"might|must|does|do|did)\b",
    re.IGNORECASE,
)
_OPENING_DEPENDENT_LEADIN_RE = re.compile(
    r"^\s*(?:but\s+)?then\b[^.!?]{0,120}?\bwhen\s+",
    re.IGNORECASE,
)
_LEADING_DISCOURSE_MARKER_RE = re.compile(
    r"^\s*(?:but|so)\s*[,;:]?\s+",
    re.IGNORECASE,
)
_OPENING_TOPIC_ANNOUNCEMENT_PREFIX_RE = re.compile(
    r"^\s*(?:(?:now|so)\s*[,;:]?\s+)?what\s+"
    r"(?:i|we)(?:['’]ll|\s+will)\s+(?:discuss|talk)\s+"
    r"(?:to\s+you\s+)?about\s+is\s+",
    re.IGNORECASE,
)
_OPENING_EXAMPLE_FRAMING_SENTENCE_RE = re.compile(
    r"^\s*(?:"
    r"here(?:['’]s|\s+is)\s+|"
    r"let(?:['’]?s|\s+us)\s+(?:consider|do|go\s+through|look\s+at|take|try|"
    r"walk\s+through|work\s+out|work\s+through)\s+|"
    r"(?:now\s*[,;:]?\s+)?(?:consider|try)\s+"
    r"|(?:now\s*[,;:]?\s+)?for\s+"
    r")"
    r"(?:(?:an?|(?:yet\s+)?another|one\s+more|the\s+(?:following|next)|"
    r"a\s+(?:new|second))\s+)"
    r"(?:(?:brief|concrete|quick|short|simple|worked)\s+)*"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)"
    r"(?:\s+(?:for\s+us|here|now))?\s*[.!?]+\s*$",
    re.IGNORECASE,
)
_NEXT_EXAMPLE_FRAMING_RE = re.compile(
    r"^\s*(?:(?:now|next)\s*[,;:]?\s+)?(?:"
    r"let(?:['’]?s|\s+us)\s+(?:consider|do|go\s+through|look\s+at|take|try|"
    r"work\s+out|work\s+through)\s+|"
    r"(?:consider|for|try)\s+|here(?:['’]s|\s+is)\s+)"
    r"(?:another|one\s+more|the\s+next|a\s+(?:different|new))\s+"
    r"(?:(?:brief|concrete|quick|short|simple|worked)\s+)*"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)"
    r"\s*[.!?]*\s*$|"
    r"^\s*(?:(?:now|next)\s*[,;:]?\s+)?our\s+next\s+"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)"
    r"\s+is\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_OPENING_EDGE_META_SENTENCE_RE = re.compile(
    r"^\s*before\s+(?:i|we)\s+(?:continue|go\s+on|move\s+forward)\b"
    r"[^.!?]*[.!?]",
    re.IGNORECASE,
)
_OPENING_CONTEXTUAL_REFORMULATION_RE = re.compile(
    r"^\s*(?:(?:now|so)\s*[,;:]?\s+)?(?:"
    r"there(?:['’]s|\s+(?:is|are|was|were))\s+(?:an?\s+)?other\b|"
    r"one\s+more\s+(?:case|example|function|point|reason|step|thing|way)s?"
    r"\s*[.!?]?\s*$|"
    r"another\s+(?:(?:one|ones|thing|things)\b|"
    r"(?:example|examples|case|cases)\b(?!\s+of\b))"
    r")",
    re.IGNORECASE,
)
_OPENING_CONTEXTUAL_EXAMPLE_RE = re.compile(
    r"^\s*(?:in|for)\s+(?:this|that|the)\s+"
    r"(?:(?:particular|previous|same|specific)\s+)?"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)\b",
    re.IGNORECASE,
)
_LOCAL_EXAMPLE_SETUP_RE = re.compile(
    r"\b(?:assume|calculate|consider|determine|evaluate|find|given|let|prove|"
    r"show|solve|suppose)\b|"
    r"\blimit\b[^.!?]{0,140}\b(?:approaches|tends\s+to)\b|"
    r"\b(?:equation|expression|function|problem)\b[^.!?]{0,140}"
    r"\b(?:equals?|gives?|is)\b",
    re.IGNORECASE,
)
_LOCAL_SETUP_ACTION_RE = re.compile(
    r"\b(?:calculate|consider|determine|evaluate|find|prove|show|solve)\b"
    r"(?P<object>[^.!?]{0,120})",
    re.IGNORECASE,
)
_GENERAL_SETUP_ACTION_RE = re.compile(
    r"^\s*(?:analyze|apply|classify|compare|contrast|describe|explain|identify|"
    r"interpret|name|state|trace)\b(?P<object>[^.!?]{0,180})",
    re.IGNORECASE,
)
_UNGROUNDED_IMPERATIVE_RE = re.compile(
    r"^\s*(?:apply|cancel|differentiate|divide|factor|multiply|plug|replace|"
    r"substitute|simplify)\b",
    re.IGNORECASE,
)
_LOCAL_EXPLICIT_PROBLEM_RE = re.compile(
    r"\blimit\b[^.!?]{0,140}\b(?:approaches|tends\s+to)\b|"
    r"\b(?:equation|expression|function|problem)\b[^.!?]{0,140}"
    r"\b(?:equals?|gives?|is)\b|"
    r"\b(?:given|let)\s+"
    r"(?!(?:it|this|that|these|those|they|here|there|next|previous|same)\b)"
    r"(?:the\s+)?[a-z0-9][a-z0-9'_-]*\b",
    re.IGNORECASE,
)
_OPENING_AGENDA_RE = re.compile(
    r"^\s*in\s+(?:this|the)\s+(?:course|lesson|section|video)\s+"
    r"(?:i|we)(?:['’](?:ll|m|re)|\s+(?:am|are|will))\s+"
    r"(?:just\s+)?(?:going\s+to\s+)?"
    r"(?:cover|discuss|explain|go(?:\s+over)?|introduce|review|show|talk\s+about|teach)\b",
    re.IGNORECASE,
)
_OPENING_AGENDA_CONTINUATION_RE = re.compile(
    r"^\s*(?:"
    r"(?:about|including|over)\s+|"
    r"(?:and|or)\s+(?:algebraically|analytically|conceptually|graphically|"
    r"numerically|(?:how|what|when|where)\s+to)\b)",
    re.IGNORECASE,
)
_PEDAGOGICAL_SETUP_ONSET_RE = re.compile(
    r"^\s*(?:(?:now|so)\s*[,;:]?\s+)?(?:"
    r"let(?:['’]?s|\s+us)\s+(?:assume|calculate|consider|evaluate|find|say|solve|suppose|try)|"
    r"(?:assume|assuming|consider|given|suppose)\b|"
    r"(?:calculate|determine|evaluate|find|prove|show|solve)\b|"
    r"(?:how|what|when|where|which|why)\b)",
    re.IGNORECASE,
)
_WORKED_UNIT_ACTION_ONSET_RE = re.compile(
    r"^\s*(?:(?:now|next)\s*[,;:]?\s+)?(?:"
    r"let(?:['’]?s|\s+us)\s+(?:analyze|calculate|compute|consider|determine|"
    r"differentiate|evaluate|find|identify|prove|show|solve|try)|"
    r"(?:analyze|calculate|compute|determine|differentiate|evaluate|find|"
    r"identify|prove|show|solve)\b|"
    r"(?:how|what|why)\s+"
    r"(?:are|can|could|did|do|does|is|should|was|were|will|would)\b)",
    re.IGNORECASE,
)
_WORKED_UNIT_ACTION_TOKEN_RE = re.compile(
    r"(?<!\w)(?:analyze|calculate|compute|determine|differentiate|evaluate|"
    r"find|identify|prove|show|solve)\b",
    re.IGNORECASE,
)
_WORKED_UNIT_WH_ONSET_RE = re.compile(
    r"^\s*(?:(?:now|next)\s*[,;:]?\s+)?"
    r"(?:how|what|when|where|which|who|why)\b",
    re.IGNORECASE,
)
_WORKED_UNIT_QUESTION_TOKEN_RE = re.compile(
    r"(?<!\w)(?:how|what|when|where|which|who|why)\b",
    re.IGNORECASE,
)
_WORKED_UNIT_DISCOURSE_CONTINUATION_RE = re.compile(
    r"^\s*what\s+is\s+(?:more|also|worse)\b|"
    r"^\s*what\s+is\s+(?:clear|crucial|important|interesting|notable|"
    r"remarkable|significant)(?:\s+here)?\s+is\b",
    re.IGNORECASE,
)
_WORKED_UNIT_ANAPHORIC_CONTINUATION_RE = re.compile(
    r"^\s*(?:(?:calculate|compute|determine|differentiate|evaluate|find|"
    r"identify|solve)\s+(?:it|that|this)\b|"
    r"(?:prove|show)\s+(?:by|how|it|this|why)\b)",
    re.IGNORECASE,
)
_WORKED_UNIT_NONQUESTION_WH_CONTINUATION_RE = re.compile(
    r"^\s*(?:"
    r"which\s+(?:in\s+turn\b|is\s+(?:how|what|when|where|why)\b|means?\b)|"
    r"who\s+in\s+turn\b|"
    r"what\s+(?:this|that)\s+(?:[a-z][\w'’-]*\s+){0,3}is\b|"
    r"(?:when|where)\s+(?:it|that|this|these|those|we|you)\b)",
    re.IGNORECASE,
)
_WORKED_UNIT_UNRESOLVED_METHOD_REFERENCE_RE = re.compile(
    r"\b(?:apply(?:ing)?|follow(?:ing)?|use|using)\s+"
    r"(?:this|that|these|those)\s+"
    r"(?P<head>approach|formula|idea|method|principle|relationship|result|"
    r"rule|step|technique)\b",
    re.IGNORECASE,
)
_WORKED_UNIT_WH_PREVIOUS_LICENSE_RE = re.compile(
    r"\b(?:about|ask(?:ed|ing)?|describ(?:e|ed|es|ing)|discuss(?:ed|es|ing)?|"
    r"explain(?:ed|ing|s)?|know(?:ing)?|learn(?:ed|ing|s)?|recall(?:ed|ing|s)?|"
    r"remember(?:ed|ing|s)?|see(?:ing)?|show(?:ed|ing|s)?|"
    r"understand(?:ing|s)?|wonder(?:ed|ing|s)?)\s*$",
    re.IGNORECASE,
)
_WORKED_UNIT_TARGET_PROMPT_RE = re.compile(
    r"^\s*(?:(?:now|next)\s*[,;:]?\s+)?(?:"
    r"let(?:['’]?s|\s+us)\s+(?:calculate|compute|determine|evaluate|find|"
    r"identify|prove|show|solve|try)|"
    r"(?:calculate|compute|determine|evaluate|find|identify|prove|show|solve)\b|"
    r"(?:how|what|when|where|which|who|why)\b)",
    re.IGNORECASE,
)
_MARKED_WORKED_UNIT_ONSET_RE = re.compile(
    r"(?<!\w)(?P<navigation>"
    r"(?:(?:all\s+right|alright|okay|ok|so)\s*[,;:]?\s+)*"
    r"(?:now|next)\s*[,;:]?\s+)"
    r"(?P<unit>(?:"
    r"let(?:['’]?s|\s+us)\s+(?:analyze|calculate|compute|consider|determine|"
    r"differentiate|evaluate|find|identify|prove|show|solve|try)|"
    r"(?:analyze|calculate|compute|determine|differentiate|evaluate|find|"
    r"identify|prove|show|solve)\b|"
    r"(?:how|what|when|where|which|who|why)\b))",
    re.IGNORECASE,
)
_WORKED_UNIT_PROCEDURAL_STEP_RE = re.compile(
    r"^\s*(?:(?:first|next|now|then)\s*[,;:]?\s+)?"
    r"(?:calculate|compute|determine|differentiate|find|identify)\s+"
    r"(?:(?:our|the|this)\s+)?(?:"
    r"(?:inner|inside|outer|outside)\s+(?:derivative|function|part|piece)|"
    r"(?:common\s+)?(?:denominator|factor|numerator)|"
    r"(?:current|given|remaining|resulting|same)\s+"
    r"(?:equation|expression|factor|function|quantity|term|value)|"
    r"next\s+(?:step|term)|remaining\s+(?:factor|step|term)|"
    r"derivative\s+of\s+(?:(?:our|the)\s+)?(?:inner|inside|outer|outside)\b)",
    re.IGNORECASE,
)
_WORKED_UNIT_PROCEDURAL_QUESTION_RE = re.compile(
    r"^\s*(?:(?:now|so)\s*[,;:]?\s+)?(?:"
    r"what\s+(?:do|does|should)\s+(?:i|we)\s+do\b|"
    r"what\s+(?:is|would\s+be)\s+(?:our|the)\s+next\s+step\b|"
    r"what\s+remains\b|how\s+do\s+(?:i|we)\s+(?:continue|finish|proceed)\b)",
    re.IGNORECASE,
)
_WORKED_UNIT_POSSIBLE_ONSET_RE = re.compile(
    r"(?<!\w)(?:analyze|calculate|compute|determine|differentiate|evaluate|"
    r"find|identify|prove|show|solve)\b|"
    r"(?<!\w)(?:how|what|when|where|which|who|why)\b|"
    r"(?<!\w)let(?:['’]?s|\s+us)\s+(?:analyze|calculate|compute|consider|"
    r"determine|differentiate|evaluate|find|identify|prove|show|solve|try)\b|"
    r"(?<!\w)(?:here(?:['’]s|\s+is)|let(?:['’]?s|\s+us))\b[^.!?]{0,80}"
    r"\b(?:another|more|next|new)\b[^.!?]{0,50}"
    r"\b(?:calculation|case|derivation|example|exercise|problem|proof)s?\b",
    re.IGNORECASE,
)
_SPLIT_CAPTION_COMPLETION_SIGNAL_RE = re.compile(
    r"\b(?:(?:the\s+)?(?:answer|result|solution)\s+(?:is|equals?)|"
    r"(?:final|simplified)\s+(?:answer|result|solution)|"
    r"fully\s+simplified|that(?:['’]s|\s+is)\s+the\s+final\s+answer)\b",
    re.IGNORECASE,
)
_WORKED_UNIT_CLOSING_TAIL_RE = re.compile(
    r"\b(?:fully\s+simplified|"
    r"that(?:['’]s|\s+is)\s+(?:the\s+)?(?:final\s+)?"
    r"(?:answer|result|solution)|"
    r"that(?:['’]s|\s+is)\s+all\s+(?:i|we|you)\s+need\s+to\s+do\s+"
    r"for\s+(?:this|the)\s+(?:calculation|case|derivation|example|exercise|"
    r"problem|proof)|"
    r"(?:which|that)\s+(?:complete|finish)(?:d|es)?\s+(?:this|the)\s+"
    r"(?:calculation|case|derivation|example|exercise|problem|proof)|"
    r"(?:the\s+)?final\s+(?:answer|result|solution)\s+for\s+(?:this|the)\s+"
    r"(?:calculation|case|derivation|example|exercise|problem|proof))\b",
    re.IGNORECASE,
)
_SPLIT_CAPTION_ONSET_MARKER_RE = re.compile(
    r"(?<!\w)(?:(?:all\s+right|alright|okay|ok|so|now|next)"
    r"\s*[,;:]?\s+)+$",
    re.IGNORECASE,
)
_SPLIT_CAPTION_NEW_UNIT_FRAMING_RE = re.compile(
    r"(?<!\w)(?:(?:all\s+right|alright|okay|ok|so)\s*[,;:]?\s+)*"
    r"(?:now\s+)?let(?:['’]?s|\s+us)\s+"
    r"(?:do|go\s+through|look\s+at|try|work\s+on|work\s+through)\s+"
    r"(?:(?:some|a)\s+)?(?:more|another|next|new)\s+"
    r"(?:calculation|case|derivation|example|exercise|problem|proof)s?\b",
    re.IGNORECASE,
)
_OPENING_PREPOSITIONAL_TAG_RE = re.compile(
    r"^\s*(?:for|by|during|from|in|into|of|onto|to|with|without)\b"
    r"[^.!?]{0,160}\b(?:right|correct)\s*\?\s*$",
    re.IGNORECASE,
)
_OPENING_DEPENDENT_QUESTION_TAIL_RE = re.compile(
    r"^\s*(?:as|at|because|by|during|for|from|if|in|into|of|on|onto|"
    r"than|to|when|where|while|with|without)\b",
    re.IGNORECASE,
)
_PREPOSITION_FRONTED_QUESTION_RE = re.compile(
    r"^\s*(?:at|by|during|for|from|in|into|of|on|onto|to|with|without)\s+"
    r"(?:how|what|when|where|which|who|whose|whom)\b[^?]*\?",
    re.IGNORECASE,
)
_FRAMED_QUESTION_ONSET_RE = re.compile(
    r"[,;:]\s*(?:(?:and|but|now|okay|ok|so|then|well)\s+)?"
    r"(?:what|how|why|where|when|which|who|whose|whom|is|are|can|could|"
    r"would|should|does|do|did|will|was|were|has|have|had)\b",
    re.IGNORECASE,
)
_OPENING_NOMINAL_INFINITIVE_RE = re.compile(
    r"^\s*(?:(?:the|a|an)\s+)?"
    r"(?:ability|capacity|capability|process|method|role|way)\s+to\b",
    re.IGNORECASE,
)
_OPENING_LIST_TAIL_RE = re.compile(
    r"^\s*[a-z][a-z'’-]*\s*,\s+(?:and|or)\b",
    re.IGNORECASE,
)
_OPENING_COMPARATIVE_LEADIN_RE = re.compile(
    r"^\s*(?:(?:also|and|but)\s*[,]?\s+)?"
    r"(?:unlike|like|compared\s+(?:with|to)|in\s+contrast\s+to)\s+"
    r"[^,;:.!?]{1,100}[,;:]\s+",
    re.IGNORECASE,
)
_OPENING_RECOVERABLE_SETUP_RE = re.compile(
    r"^\s*(?:if|suppose|assume|assuming|given|consider)\b",
    re.IGNORECASE,
)
_OPENING_SETUP_BACK_REFERENCE_RE = re.compile(
    r"\b(?:it|this|that|these|those|they|here|there|again|above|earlier|"
    r"next|previous|same)\b",
    re.IGNORECASE,
)
_LOCAL_SETUP_EXPLICIT_GROUNDING_RE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:described|defined|given|written)\s+as\b",
    re.IGNORECASE,
)
_LATER_RESTATED_EXAMPLE_REASONING_RE = re.compile(
    r"\b(?:after|because|before|by|cancel(?:ing|led|s)?|derive(?:d|s)?|"
    r"differentiat(?:e|ed|es|ing)|divide(?:d|s)?|factor(?:ed|ing|s)?|first|"
    r"multiply|next|replace|rewrite|simplif(?:y|ied|ies)|substitut(?:e|ed|es|ing)|"
    r"then|therefore|through|thus|using|which\s+means)\b",
    re.IGNORECASE,
)
_LATER_RESTATED_EXAMPLE_CONCLUSION_RE = re.compile(
    r"\b(?:(?:the\s+)?(?:answer|result|solution)\s+(?:is|equals?)|"
    r"approaches?\s+(?:a\s+)?value|finally|gives?|hence|therefore|thus|"
    r"this\s+(?:means|shows))\b",
    re.IGNORECASE,
)
_NON_STANDALONE_MARKER_SUFFIX_RE = re.compile(
    r"^\s*(?:again|although|because|he|if|in\s+order|it|now|once|she|"
    r"that|then|these|though|"
    r"the\s+(?:answer|calculation|case|condition|difference|equation|expression|"
    r"formula|point|problem|reason|relationship|result|solution|step|thing|"
    r"value|variable)|"
    r"they|this|those|to|unless|until|when|which|while)\b",
    re.IGNORECASE,
)
_UNCONDITIONAL_TRAILING_EDGE_NOISE_PATTERN = (
    r"(?:"
    r"(?:(?:and\s+)?so\s*[,;:]?\s+)?that(?:['’]s|\s+is)\s+it\s+for\s+"
    r"(?:this|that|the)\s+(?:worked\s+)?"
    r"(?:calculation|case|derivation|example|exercise|problem|proof)\b|"
    r"trust\s+me(?:\s*[,;:]\s*|\s+)"
    r"(?:we|i)(?:['’]ll|\s+will)\b[^\n]{0,180}?"
    r"\b(?:later|next\s+time|in\s+(?:a|the)\s+(?:later|future)\s+"
    r"(?:chapter|lesson|section|video))\b|"
    r"(?:we|i)(?:['’]ll|\s+will)\s+(?:do|cover|discuss|revisit|explore|"
    r"explain|develop|return\s+to|come\s+back\s+to|"
    r"get\s+(?:more|much\s+more)\s+involved(?:\s+(?:in|with))?)\b"
    r"[^\n]{0,120}?"
    r"\b(?:later|next\s+time|in\s+(?:a|the)\s+(?:later|future)\s+"
    r"(?:chapter|lesson|section|video))\b|"
    r"(?:there\s+(?:are|is)\s+(?:many|some|several)\s+"
    r"(?:(?:other|more)\s+)?examples?\s*[,;:]?\s*(?:so\s+)?)?"
    r"(?:(?:i|we)(?:['’]ll|\s+will)?\s+leave\s+[^.!?\n]{0,80}?\s+as\s+"
    r"(?:an?\s+)?exercise\b|as\s+an?\s+exercise(?:\s*[,;:]|\s+"
    r"(?:(?:to\s+)?(?:calculate|derive|determine|differentiate|find|prove|"
    r"show|solve|try|work|write))\b))|"
    r"for\s+homework\b"
    r")"
)
_TRAILING_VERSION_EDGE_NOISE_PATTERN = (
    r"(?:(?:so|now)\s+)?let\s+(?:me|us)\s+(?:spell|write)\s+"
    r"(?:it|this|that)\s+out\b[^\n]{0,80}?"
    r"(?:(?:so|now)\s+)?this\s+is\s+(?:the\s+)?"
    r"[^.!?\n]{1,100}\bversion\b(?=\s*[.!?]*\s*$)"
)
_UNCONDITIONAL_TRAILING_EDGE_NOISE_RE = re.compile(
    rf"(?:^|\s)(?P<noise>{_UNCONDITIONAL_TRAILING_EDGE_NOISE_PATTERN})",
    re.IGNORECASE,
)
_TRAILING_VERSION_EDGE_NOISE_RE = re.compile(
    rf"(?:^|\s)(?P<noise>{_TRAILING_VERSION_EDGE_NOISE_PATTERN})",
    re.IGNORECASE,
)
_TRAILING_EDGE_NOISE_RE = re.compile(
    r"(?:^|\s)(?P<noise>"
    rf"{_UNCONDITIONAL_TRAILING_EDGE_NOISE_PATTERN}|"
    rf"{_TRAILING_VERSION_EDGE_NOISE_PATTERN}|"
    r"(?:(?:all\s+right|alright|okay|ok|so)\s*[,;:]?\s+)?"
    r"(?:now\s+)?let(?:['’]?s|\s+us)(?:\s+(?:talk|discuss|cover|define|"
    r"introduce|move\s+on|"
    r"head(?:\s+(?:over|on))?|turn|switch|begin(?:\s+our\s+discussion)?)\b|"
    r"(?=\s*[.!?]?\s*$))|"
    r"(?:(?:now|next)\s+)?(?:we|i)(?:['’]ll|\s+will)\s+(?:move\s+on|turn|switch|"
    r"head(?:\s+(?:over|on))?|talk|discuss|cover|define|introduce)\b|"
    r"so\s+let(?:['’]?s|\s+us)\s+begin\s+our\s+discussion\b|"
    r"you\s+(?:might|may|will|can|could)\s+be\s+(?:tested|quizzed)\s+on\s+"
    r"(?:this|that|it)\b|"
    r"(?:thanks?\s+(?:again\s+)?for\s+watching|"
    r"don['’]?t\s+forget\s+to\s+subscribe|please\s+subscribe|"
    r"subscribe\s+for\s+more)\b|"
    r"that['’]?s\s+(?:it\s+for\s+(?:this|the)\s+video|"
    r"a\s+(?:simplified|quick|brief)\s+(?:review|recap))\b|"
    r"back\s+to\s+(?:the\s+)?[^.!?]{1,80}[.!?]"
    r")",
    re.IGNORECASE,
)
_TRAILING_TRANSITION_FRAGMENT_RE = re.compile(
    r"\bthough\s+now\s*[,;:.!?]*\s*$",
    re.IGNORECASE,
)
_TERMINAL_FUTURE_PREVIEW_RE = re.compile(
    r"^\s*as\s+you(?:['’]ll|\s+will)\s+see\s+in\s+future\s+videos\b",
    re.IGNORECASE,
)
_TERMINAL_MASTERY_RECAP_RE = re.compile(
    r"\b(?:(?:and|so)\s+)?now\s+you\s+know\s+how\s+to\b",
    re.IGNORECASE,
)
_TERMINAL_META_RESUMPTION_RE = re.compile(
    r"^\s*(?:(?:now|next|then|finally)\s*[,;:]?\s+)?(?:"
    r"apply|calculate|compute|continue|derive|determine|divide|evaluate|factor|"
    r"multiply|return\s+to|solve|substitute|the\s+(?:answer|result)\s+is|"
    r"therefore|thus)\b",
    re.IGNORECASE,
)
_EDGE_ONLY_FRAMING_RE = re.compile(
    r"^\s*(?:remember\s+)?(?:on|in|for|with)\b.{0,80}\b"
    r"(?:right\s+)?we\s+(?:talked|spoke|learned|covered|went\s+over)\s+"
    r"(?:about\s+)?(?:that|this|it)\s*[.!?]?\s*$|"
    r"^\s*(?:so\s+)?now\s+(?:the\s+)?next\s+"
    r"(?:goal|thing|step)(?:\s+here)?\s+is(?:\s+that)?\s+(?:we|i)\b"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_EXEMPLIFICATION_RE = re.compile(
    r"(?:^|[.!?;,]\s+)(?P<so>so\s*[,]?\s+)?"
    r"(?:for\s+(?:example|instance)|such\s+as)\s*[,]?\s*"
    r"(?P<tail>[^.!?]*)$",
    re.IGNORECASE,
)
_TERMINAL_STRANDED_PREPOSITION_RE = re.compile(
    r"\b(?:a|an|the|this|that|these|those|our|your|their)\s+"
    r"(?:[a-z][a-z'-]*\s+){1,4}(?:(?:that|which|who|whom)\s+)?"
    r"(?:i|we|you|they)\s+(?:can|could|will|would|should|may|might)\s+"
    r"(?:[a-z][a-z'-]*\s+){1,4}(?:from|with|to|for|about|on|at|by)"
    r"[.!?][\"')\]]*$",
    re.IGNORECASE,
)
_STANDALONE_QUESTION_HEADS = frozenset({
    "what", "how", "why", "where", "when", "which", "who", "whose", "whom",
    "is", "are", "can", "could", "would", "should", "does", "do", "did",
    "will", "was", "were", "has", "have", "had",
})
_QUESTION_PREFIXES = frozenset({
    "and", "but", "so", "now", "well", "okay", "ok", "alright",
})
_NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

PRODUCTION_PRO_PROFILE = "production_pro_v0"
CORRECTED_PRO_PROFILE = "corrected_pro_v1"
FLASH_SINGLE_PROFILE = "flash_single_v1"
FLASH_SPLIT_PROFILE = "flash_split_v1"
PRO_BOUNDARY_PROFILE = "pro_boundary_v1"
# Production Flash performs only the compact, quality-critical boundary choice.
PRODUCTION_FLASH_PROFILE = FLASH_SPLIT_PROFILE
# Authoritative and fallback Pro routes use the same compact boundary contract.
# Legacy profiles remain readable for old cache/test compatibility only.
AUTHORITATIVE_PRO_PROFILE = PRO_BOUNDARY_PROFILE
SEGMENT_PROFILES = (
    PRODUCTION_PRO_PROFILE,
    CORRECTED_PRO_PROFILE,
    FLASH_SINGLE_PROFILE,
    FLASH_SPLIT_PROFILE,
    PRO_BOUNDARY_PROFILE,
)

_TOTAL_DEADLINE_S = 36.0
_FLASH_SINGLE_TIMEOUT_S = 45.0
_FLASH_BOUNDARY_TIMEOUT_S = 20.0
_FLASH_REPAIR_TIMEOUT_S = 20.0
_FLASH_ENRICH_TIMEOUT_S = 25.0
_PRO_TIMEOUT_S = 90.0
_SELECTION_OUTPUT_TOKENS = 24_576
# Six thousand compact-schema tokens still cover the exhaustive candidate cap
# while allowing Fast's two and Slow's three 30k-token source analyses to start
# together within their existing hard job-cost ceilings.
_BOUNDARY_OUTPUT_TOKENS = 6_000
_BOUNDARY_REPAIR_OUTPUT_TOKENS = 1_024
_ENRICH_OUTPUT_TOKENS = 2_048
_MAX_CLIPS = 40
_MAX_SELECTOR_CANDIDATES = _MAX_CLIPS
_GREEN_SCORE = 0.75
_DUPLICATE_OVERLAP = 0.8
_MAX_INTERNAL_FILLER_DURATION_S = 12.0
_MAX_INTERNAL_FILLER_WORDS = 32
_SECTION_RESET_GAP_S = 8.0
_BOUNDARY_PAD_S = 0.3
_REPAIR_NEIGHBOR_CUES = 2
_BOUNDARY_REPAIR_PROMPT_VERSION = "boundary_repair_v1"
_CARD_ENRICHMENT_PROMPT_VERSION = "accepted_clip_enrichment_v1"

_PRICING_VERSION = "gemini-standard-2026-07-11"
_PRICING_PER_MILLION = {
    "flash": {"input": 1.50, "output": 9.00},
    "flash_lite": {"input": 0.25, "output": 1.50},
    "flash_preview": {"input": 0.50, "output": 3.00},
    "pro": {"input": 2.00, "output": 12.00},
}

_flash_disable_lock = Lock()
_flash_disabled_reason: str | None = None


# ---------------------------------------------------------------------------
# Strict model schemas

class _StrictModel(BaseModel):
    # Gemini's response_schema endpoint rejects Pydantic's
    # ``additionalProperties: false`` representation for ``extra="forbid"``.
    # Required typed fields still constrain generation; semantic validation below
    # remains authoritative.
    model_config = ConfigDict(extra="forbid")


class _UncertaintyReason(str, Enum):
    BOUNDARY_AMBIGUOUS = "boundary_ambiguous"
    INCOMPLETE_CONTEXT = "incomplete_context"
    TOPIC_AMBIGUOUS = "topic_ambiguous"
    TRANSCRIPT_NOISE = "transcript_noise"
    OVERLAP_RISK = "overlap_risk"
    OTHER = "other"


class _IntentConstraintKind(str, Enum):
    SUBJECT = "subject"
    TASK = "task"
    RELATIONSHIP = "relationship"
    SCOPE = "scope"
    FORMAT = "format"
    OUTCOME = "outcome"


class _IntentRole(str, Enum):
    PRIMARY = "primary"
    SUPPORTING = "supporting"


class _IntentConstraint(_StrictModel):
    constraint_id: _IntentConstraintId
    kind: _IntentConstraintKind
    source_phrase: _NonBlank = Field(
        description=(
            "Exact consecutive words copied from the user's request that introduce "
            "this atomic constraint."
        )
    )
    requirement: _NonBlank


class _RequestIntent(_StrictModel):
    exact_request: _NonBlank
    constraints: list[_IntentConstraint] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _unique_constraint_ids(self):
        ids = [constraint.constraint_id for constraint in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("intent constraint ids must be unique")
        return self


class _IntentEvidence(_StrictModel):
    constraint_id: _IntentConstraintId
    evidence_quote: _EvidenceQuote = Field(
        description=(
            "Five to sixteen consecutive transcript words copied exactly from the "
            "candidate that demonstrate this intent constraint."
        )
    )


class _AssessmentDraft(_StrictModel):
    prompt: _NonBlank
    options: list[_NonBlank] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3, strict=True)
    explanation: _NonBlank
    evidence_quote: _NonBlank = Field(
        description=(
            "Exact consecutive transcript words copied from inside this clip; preserve "
            "the transcript spelling and never paraphrase or stitch text."
        )
    )

    @model_validator(mode="after")
    def _unique_options(self):
        normalized = {" ".join(option.split()).casefold() for option in self.options}
        if len(normalized) != 4:
            raise ValueError("assessment options must be distinct")
        if any("all of the above" in option.casefold() for option in self.options):
            raise ValueError("all-of-the-above options are not allowed")
        return self


class _BoundaryTopic(_StrictModel):
    candidate_id: _CandidateId
    start_line: int = Field(ge=0, strict=True)
    end_line: int = Field(ge=0, strict=True)
    start_quote: _BoundaryQuote = Field(
        description=(
            "Shortest unique one to twelve consecutive words copied exactly from the "
            "cited start line, beginning at the first required teaching word; never pad."
        )
    )
    end_quote: _BoundaryQuote = Field(
        description=(
            "Shortest unique one to twelve consecutive words copied exactly from the "
            "cited end line, ending at the complete teaching conclusion; never pad."
        )
    )
    title: _ClipTitle
    learning_objective: _LearningObjective
    facet: _Facet
    reason: _OptionalReason = ""
    informativeness: float = Field(ge=0.0, le=1.0, strict=True)
    topic_relevance: float = Field(ge=0.0, le=1.0, strict=True)
    educational_importance: float = Field(ge=0.0, le=1.0, strict=True)
    difficulty: float = Field(ge=0.0, le=1.0, strict=True)
    directly_teaches_topic: bool = Field(strict=True)
    substantive: bool = Field(strict=True)
    factually_grounded: bool = Field(strict=True)
    topic_evidence_quote: _EvidenceQuote = Field(
        description=(
            "Shortest five to twelve consecutive transcript words wholly between the "
            "chosen start and end quotes that prove relevance; never paraphrase or stitch."
        )
    )
    self_contained: bool = Field(strict=True)
    is_standalone: bool = Field(strict=True)
    prerequisite_candidate_ids: list[_CandidateId] = Field(default_factory=list, max_length=8)
    uncertainty: Literal["low", "medium", "high"] = "low"
    uncertainty_reasons: list[_UncertaintyReason] = Field(default_factory=list, max_length=6)
    # Compatibility defaults keep isolated conversion fixtures readable. The
    # live selector uses _IntentBoundaryTopic, where both fields are required.
    intent_role: _IntentRole | None = None
    intent_evidence: list[_IntentEvidence] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _uncertainty_has_reason(self):
        if self.uncertainty != "low" and not self.uncertainty_reasons:
            raise ValueError("medium/high uncertainty requires a reason")
        return self


class _Topic(_BoundaryTopic):
    reason: _NonBlank
    summary: _NonBlank
    takeaways: list[_NonBlank] = Field(min_length=2, max_length=4)
    match_reason: _NonBlank
    assessment: _AssessmentDraft | None

    @field_validator("takeaways")
    @classmethod
    def _distinct_takeaways(cls, value: list[str]) -> list[str]:
        if len({" ".join(item.split()).casefold() for item in value}) != len(value):
            raise ValueError("takeaways must be distinct")
        return value


class _Plan(_StrictModel):
    topics: list[_Topic] = Field(max_length=_MAX_CLIPS)


class _BoundaryPlan(_StrictModel):
    topics: list[_BoundaryTopic] = Field(max_length=_MAX_CLIPS)


_CompactCandidateId = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)
]
_CompactBoundaryQuote = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
]
_CompactTitle = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=96)
]
_CompactObjective = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
]
_CompactFacet = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=96)
]
_CompactEvidenceQuote = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)
]


class _CompactIntentEvidence(_StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    constraint_id: _IntentConstraintId = Field(alias="id")
    evidence_quote: _CompactEvidenceQuote = Field(
        alias="q",
        description="Five to sixteen exact consecutive transcript words in this candidate.",
    )


class _CompactBoundaryTopic(_StrictModel):
    """Token-efficient production schema; attributes retain canonical names."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    candidate_id: _CompactCandidateId = Field(alias="id")
    start_line: int = Field(ge=0, strict=True, alias="s")
    end_line: int = Field(ge=0, strict=True, alias="e")
    start_quote: _CompactBoundaryQuote = Field(
        alias="sq", description="Exact opening transcript quote."
    )
    end_quote: _CompactBoundaryQuote = Field(
        alias="eq", description="Exact concluding transcript quote."
    )
    title: _CompactTitle
    learning_objective: _CompactObjective = Field(alias="obj")
    facet: _CompactFacet
    informativeness: float = Field(ge=0.0, le=1.0, strict=True, alias="info")
    topic_relevance: float = Field(ge=0.0, le=1.0, strict=True, alias="rel")
    educational_importance: float = Field(ge=0.0, le=1.0, strict=True, alias="imp")
    difficulty: float = Field(ge=0.0, le=1.0, strict=True, alias="diff")
    directly_teaches_topic: bool = Field(strict=True, alias="direct")
    substantive: bool = Field(strict=True, alias="sub")
    factually_grounded: bool = Field(strict=True, alias="fact")
    self_contained: bool = Field(strict=True, alias="self")
    is_standalone: bool = Field(strict=True, alias="stand")
    intent_evidence: list[_CompactIntentEvidence] = Field(
        alias="ie",
        min_length=1,
        max_length=8,
        description="One grounded item for each exact-request constraint this unit fulfills.",
    )


class _CompactBoundaryPlan(_StrictModel):
    request_intent: _RequestIntent
    topics: list[_CompactBoundaryTopic] = Field(max_length=_MAX_CLIPS)


class _IntentBoundaryTopic(_BoundaryTopic):
    intent_role: _IntentRole
    intent_evidence: list[_IntentEvidence] = Field(min_length=1, max_length=8)


class _IntentBoundaryPlan(_StrictModel):
    request_intent: _RequestIntent
    topics: list[_IntentBoundaryTopic] = Field(max_length=_MAX_CLIPS)


class _BoundaryRepairItem(_StrictModel):
    candidate_id: _NonBlank
    start_line: int = Field(ge=0, strict=True)
    end_line: int = Field(ge=0, strict=True)
    start_quote: _NonBlank
    end_quote: _NonBlank


class _BoundaryRepairPlan(_StrictModel):
    items: list[_BoundaryRepairItem] = Field(max_length=_MAX_CLIPS)


class _EnrichmentItem(_StrictModel):
    clip_id: _NonBlank
    summary: _NonBlank
    takeaways: list[_NonBlank] = Field(min_length=2, max_length=4)
    match_reason: _NonBlank
    assessment: _AssessmentDraft

    @field_validator("takeaways")
    @classmethod
    def _distinct_takeaways(cls, value: list[str]) -> list[str]:
        if len({" ".join(item.split()).casefold() for item in value}) != len(value):
            raise ValueError("takeaways must be distinct")
        return value


class _EnrichmentPlan(_StrictModel):
    items: list[_EnrichmentItem] = Field(max_length=_MAX_CLIPS)


class _CardEnrichmentItem(_StrictModel):
    clip_id: _NonBlank
    summary: _NonBlank
    takeaways: list[_NonBlank] = Field(min_length=2, max_length=4)
    match_reason: _NonBlank


class _CardEnrichmentPlan(_StrictModel):
    items: list[_CardEnrichmentItem] = Field(max_length=3)


# The frozen production prompt remains available as an immutable evaluation
# baseline.  Its schema is deliberately permissive because that is part of the
# profile being measured; strict application validation still guards its output.
class _ProductionTopic(BaseModel):
    title: str
    start_line: int
    end_line: int
    start_quote: str
    end_quote: str
    reason: str = ""
    facet: str = "other"
    kind: Optional[str] = None
    informativeness: Optional[float] = None
    topic_relevance: Optional[float] = None
    self_contained: Optional[bool] = None
    difficulty: float = 0.5


class _LegacyTopic(_ProductionTopic):
    summary: str = ""
    takeaways: list[str] = Field(default_factory=list)
    match_reason: str = ""
    assessment: dict | None = None


class _LegacyPlan(BaseModel):
    topics: list[_LegacyTopic]


class _ProductionPlan(BaseModel):
    topics: list[_ProductionTopic]


# ---------------------------------------------------------------------------
# Prompt construction

_POLICY_AND_EXAMPLES = """Policy:
- First understand the whole transcript and the viewer's exact request. Return only
  related, complete, substantive teaching units that make sense to a cold viewer hearing
  the clip without seeing the original video. Related includes the requested subject and
  clearly useful prerequisite or supporting facets, not merely adjacent material.
- Include every necessary setup or prerequisite through the explanation's natural
  conclusion. For a worked example, include the question or setup, reasoning, and answer.
- Give each candidate exactly one coherent learning objective. When adjacent speech teaches
  independent facets, return separate candidates for those facets instead of bundling them.
- Never combine the ending of one objective with the opening of another. At an explicit
  topic transition, end the old unit before the transition and start the new unit after it.
- A shared broad subject is not permission to bundle sequential lessons. When the
  grammatical subject or teaching claim changes after the chosen objective is complete,
  end there even without a transition phrase. A comparison of two things ends after that
  comparison; do not append later history, a third thing, or a merely related example.
- Omit greetings, credentials, sponsors, administration, promos, transitions, previews,
  recaps, outros, atmospheric hooks, scene-setting, music/applause caption markers,
  colorful flourishes, audience banter, post-conclusion jokes, tangents, repeated
  restatements, and partial explanations.
- Keep starts and ends free of that filler. Never add filler or incomplete material at an
  opening or ending. Nonessential material inside an otherwise valuable complete unit may
  remain when cutting around it would break the teaching arc;
  never discard the whole unit solely for an internal interruption, regardless of that
  interruption's length.
- Course operations such as enrollment, pass/fail or grading rules, assignment and exam
  scheduling, registration, attendance, office hours, and course requirements are not
  educational context for the subject. Exclude them from clip openings and endings.
- Omit teaching that depends on a diagram, screen, gesture, drawing, or other missing
  visual context. Mark self_contained and is_standalone false for such material.
- Exhaustively enumerate every distinct related teaching unit, up to 40 per source. Prefer an empty
  slot to filler or an incomplete idea. Do not shorten a complete idea to fit a target
  length; clip duration is never a selection criterion.
- Keep distinct informational facets from the same source. Do not return two clips that
  teach the same learning objective in different words.
- Return every qualifying related unit, while scoring the densest, most useful, and most
  central units highest so the application can prioritize them within difficulty stages.
- Treat subject relevance and fulfillment of the user's requested operation, relationship,
  scope, format, and outcome as separate facts. A definition or prerequisite can remain a
  useful supporting unit, but it is not primary fulfillment of a requested example,
  comparison, causal explanation, misconception correction, identification, derivation,
  application, or other task. Never label or rank a supporting facet as primary fulfillment.
- Return a candidate only when informativeness, topic_relevance, and educational_importance
  are each at least 0.75 and the spoken unit satisfies every substantive, grounding,
  context, and filler rule.
- Copy exact transcript line IDs and exact opening/closing quotes. start_quote must be the
  first words a cold viewer needs to hear for this one teaching objective, after every
  atmospheric hook, scene-setting flourish, or opening joke. end_quote must be the last
  words of its complete conclusion, before audience banter, a next-topic setup, or a
  post-conclusion joke. Quotes may begin or end inside a coarse transcript line. Copy the
  shortest unique 1-12 consecutive words that land exactly on the semantic edge, preserve
  the transcript spelling, and keep the quote wholly inside its cited edge line. A clean
  one-word edge is better than padding it with a transition, next topic, recap, or outro.
  Never pad to satisfy a preferred quote length. Never paraphrase, correct, stitch, or cross
  transcript lines. topic_evidence_quote must be the
  shortest useful 5-12 consecutive words wholly between those chosen edges, copied with the
  same exactness.
- A worked example cannot end at its question, setup, or first substituted value. Either
  include its reasoning and answer through end_quote or end the candidate before that
  optional example begins.
- Do not provide chain-of-thought or hidden reasoning.

Examples:
KEEP this complete teaching unit:
[12] 01:20 Gradient descent updates model parameters in the direction that reduces loss.
[13] 01:28 Repeating that update moves the model until the loss reaches a minimum.
Use start_line=12, end_line=13,
start_quote="Gradient descent updates model parameters",
end_quote="until the loss reaches a minimum".

OMIT these non-units:
[20] 02:05 Welcome back, and thanks to today's sponsor.
[21] 02:12 The first half of the explanation is that the variable changes because...
The first line is framing/sponsorship; the second ends before the explanation is complete.
"""


def _topic_rule(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        return "No topic filter was supplied; return every substantive educational unit."
    if re.search(
        r"\b(?:versus|vs\.?|compare(?:d|s|ing)?|comparison|contrast|difference between)\b|/",
        topic,
        flags=re.IGNORECASE,
    ):
        compound_rule = (
            "For a comparison request, deeply teaching either requested side is relevant, "
            "as is teaching their relationship. Return separate substantive units for either "
            "side; do not require every candidate to repeat or compare both sides. "
        )
    elif "," in topic or ";" in topic:
        compound_rule = (
            "When the topic lists multiple requested ideas, a span directly matches when "
            "it deeply teaches any one requested component. Require a relationship between "
            "components only when the viewer explicitly asks to compare, connect, relate, "
            "or apply them together. "
        )
    else:
        compound_rule = (
            "When the topic names multiple linked ideas, deeply teaching any one requested "
            "component or a useful prerequisite facet is relevant. The selected speech or "
            "its exact evidence quote must still anchor that facet to the named subject; a "
            "broad word used in an unrelated domain is not enough. "
        )
    return (
        f"The viewer is studying {topic!r}. Return only units that teach that topic or a "
        "clearly useful prerequisite/supporting facet, and make each learning objective name "
        "the relevant idea. Set directly_teaches_topic=true for either a direct unit or an "
        "explicitly topic-linked prerequisite/supporting unit. Set it false when the span "
        "merely names the subject, course, institution, or speaker, or belongs to an adjacent "
        "field without a useful connection to the request. "
        f"{compound_rule}"
        "When the topic requests "
        "identification, recognition, diagnosis, derivation, comparison, or application, "
        "include units that teach or perform that task for the named object as well as "
        "separate, explicitly topic-anchored prerequisite facets. A history or definition "
        "alone is not a direct match to the complete requested task and is not primary "
        "fulfillment; return it only as a "
        "supporting facet when exact evidence anchors it to the named subject. Task "
        "fulfillment raises educational importance and centrality; it does not exclude a "
        "genuinely related, topic-anchored supporting facet. Shared vocabulary, a loose "
        "analogy, or general "
        "systems thinking alone is not a useful prerequisite. Include supporting material "
        "when it has a clear educational connection to the exact requested topic, even if "
        "it is not strictly required background. Exclude only adjacent broad-field material "
        "whose connection is generic or incidental."
        " Exclude fictional, supernatural, pseudoscientific, or invented mechanisms unless "
        "the viewer explicitly requested that fictional subject. Borrowing real academic "
        "terminology does not make an invented claim educational evidence."
    )


def _learner_rule(level: str) -> str:
    normalized = " ".join(str(level or "").split()).casefold()
    if normalized not in {"beginner", "intermediate", "advanced"}:
        return ""
    return (
        f"The viewer's current level is {normalized}. Still return qualifying units at every "
        "difficulty. Difficulty is metadata, not an eligibility filter; the application "
        "organizes accepted units later."
    )


def _selection_fields(*, enriched: bool, compact: bool = False) -> str:
    fields = (
        "candidate_id (a short unique slug), start_line, end_line, start_quote and "
        "end_quote (each the shortest unique 1-12 exact consecutive words wholly inside its "
        "cited line, preserving transcript spelling and marking the first required teaching "
        "words and last complete conclusion, even inside a coarse line; never pad with nearby "
        "speech, paraphrase, stitch, or cross lines), "
        "title (at most 12 words), "
        "learning_objective (at most 24 words), facet (at most 12 words), "
        "informativeness, topic_relevance, "
        "educational_importance, difficulty, directly_teaches_topic, substantive, "
        "factually_grounded"
    )
    if not compact:
        fields += (
            ", topic_evidence_quote (the shortest exact 5-12 consecutive-word quote copied "
            "wholly between the chosen edges that proves the clip teaches the topic; preserve "
            "spelling and never paraphrase or stitch), self_contained, is_standalone"
            ", prerequisite_candidate_ids (omit it or return []), uncertainty "
            "(omit for low), uncertainty_reasons (omit for low)"
        )
    else:
        fields += (
            ", self_contained, is_standalone"
            ". Use the compact schema keys: id=candidate_id, s=start_line, e=end_line, "
            "sq=start_quote, eq=end_quote, obj=learning_objective, "
            "info=informativeness, rel=topic_relevance, imp=educational_importance, "
            "diff=difficulty, direct=directly_teaches_topic, sub=substantive, "
            "fact=factually_grounded, self=self_contained, stand=is_standalone, "
            "ie=intent_evidence"
        )
    if enriched:
        fields += (
            ", summary, takeaways (2-4 distinct grounded points), match_reason, and "
            "assessment {prompt (at most 16 words), exactly four distinct options "
            "(at most 8 words each), correct_index, explanation (one sentence, at most "
            "24 words), evidence_quote copied exactly from the selected clip}"
        )
    return fields


def _intent_selection_fields() -> str:
    return (
        "intent_role (primary only when the clip fulfills every atomic request constraint; "
        "otherwise supporting), and intent_evidence (one item per fulfilled constraint, "
        "each containing constraint_id and a 5-16 word exact consecutive evidence_quote "
        "copied from inside the candidate)"
    )


def _prompts(
    lines: str,
    n: int,
    topic: str = "",
    learner_level: str = "",
) -> tuple[str, str]:
    """Gemini 3.5 single-pass prompt: policy/examples, context, task last."""
    system = (
        "You select self-contained educational clips from timestamped transcripts.\n\n"
        + _POLICY_AND_EXAMPLES
    )
    learner_rule = _learner_rule(learner_level)
    learner_line = f"{learner_rule}\n" if learner_rule else ""
    user = (
        f"Line IDs must be between 0 and {n - 1}.\n\n"
        f"Transcript ({n} lines, formatted `[index] MM:SS text`):\n{lines}\n\n"
        f"Exact user request: {topic.strip() or '(all educational topics)'}\n"
        f"{_topic_rule(topic)}\n{learner_line}"
        "Based on the preceding transcript, return the chronological educational units. "
        f"Every item must contain {_selection_fields(enriched=True)}. Return no item for "
        "material that is incomplete or non-educational."
    )
    return system, user


def _boundary_prompts(
    lines: str,
    n: int,
    topic: str = "",
    *,
    learner_level: str = "",
) -> tuple[str, str]:
    system = (
        "You select self-contained educational clip boundaries from timestamped transcripts.\n\n"
        + _POLICY_AND_EXAMPLES
    )
    del learner_level
    exact_request = topic.strip() or "(all educational topics)"
    user = (
        f"Transcript ({n} lines, formatted `[index] MM:SS text`; valid line IDs are "
        f"0 through {n - 1}):\n{lines}\n\n"
        f"Exact user request: {exact_request}\n"
        f"{_topic_rule(topic)}\n\n"
        "Task:\n"
        "1. Interpret the exact request before selecting anything. Return request_intent with "
        "exact_request copied exactly from the Exact user request above and 1-8 atomic "
        "constraints. Give every constraint a unique constraint_id, its kind, a concise "
        "requirement, and source_phrase copied as exact consecutive words from that request. "
        "Together the source phrases must cover every content-bearing request term. Preserve "
        "named subjects, requested operations or tasks, relationships, scope qualifiers, "
        "formats, and outcomes. Do not substitute retrieval expansions or a broader topic. "
        "Then scan the whole transcript from first to last and understand it before selecting. "
        "Internally distinguish required setup and teaching from administration, promotion, "
        "navigation, repetition, and visual-dependent speech; do not output that section map.\n"
        "2. Map every distinct educational unit related to the exact request, including "
        "niche facts, useful prerequisite facets, examples, mechanisms, comparisons, and "
        f"conclusions. Return every distinct qualifying moment, up to "
        f"{_MAX_SELECTOR_CANDIDATES} for this source; "
        "do not stop after the first few units or at an arbitrary count below that cap.\n"
        "3. For every qualifying unit, verify its timestamps and choose the minimum complete "
        "span containing necessary setup, reasoning, and the natural conclusion, regardless "
        "of its duration. Give it exactly one learning objective. Split independent adjacent "
        "facets into separate candidates even when they share one coarse transcript line. "
        "Never combine the end of one objective with the beginning of another: end before "
        "the transition for the old unit and start after it for the new unit. When a coarse "
        "line contains the old conclusion, transition, and new opening, place the exact edge "
        "quotes inside that line on the correct side of the transition. "
        "Apply a cold-start and cold-stop test using only the spoken audio: never begin at a "
        "dependent tail, tag question, unresolved 'other one' reformulation, or the "
        "consequence of an analogy whose actors and mapping were omitted. Never end inside "
        "a list, next-topic phrase, recap, contextual bridge, or outro. Use a one-word quote "
        "when that is the exact clean edge; never pad an edge quote with nearby speech. "
        "Keep opening and ending edges clean, including generic lead-ins and bracketed "
        "non-speech markers. Split around an internal interruption when separate complete "
        "units remain; otherwise keep it rather than discard a valuable complete arc. "
        "Never omit a substantive grounded unit solely because its boundary is uncertain: "
        "return the best complete cue span and let deterministic post-processing refine it. "
        "Omit only when content, context, or topic meaning is too uncertain to support grounded "
        "teaching; severe transcript noise counts as content uncertainty. Omit material that "
        "requires an unseen visual.\n"
        "4. Score topic relevance, information density, educational value, and difficulty "
        "honestly. Return a unit only when topic_relevance, informativeness, and "
        "educational_importance are each at least 0.75. Difficulty is metadata, not an "
        "eligibility filter; "
        "it records prior knowledge only: 0.00-0.33 means "
        "beginner, 0.34-0.66 means intermediate, and 0.67-1.00 means advanced. Return units "
        "across that entire scale.\n"
        "5. Return every distinct qualifying unit. Set substantive and factually_grounded true "
        "only for academically sound teaching; course logistics and institutional framing are "
        "not teaching units. Each unit must be standalone, use a unique "
        "candidate_id, and include required setup inside its span.\n"
        f"Return only the object {{request_intent, topics}}. Every topic must contain "
        f"{_selection_fields(enriched=False, compact=True)}. The ie list must be nonempty and "
        "contain one {id, q} item per request constraint the unit fulfills, where id is the "
        "constraint_id and q is an exact consecutive 5-16 word transcript quote wholly inside "
        "the candidate. Do not output a role: the backend derives primary only when grounded "
        "evidence covers every request constraint, and supporting otherwise. Learning details and "
        "assessments are generated later. Do not include them, chain-of-thought, or hidden "
        "reasoning."
    )
    return system, user


def _boundary_repair_prompts(
    candidates: list["_BoundaryRepairCandidate"],
    segments: list[dict],
    topic: str,
) -> tuple[str, str, dict[str, tuple[set[int], set[int]]]]:
    """Render only the neighboring cue windows needed to repair dirty edges."""
    system = (
        "You repair transcript-cue boundaries for already selected educational moments. "
        "Use only the displayed neighboring cues. Return at most one item per candidate, "
        "omit a candidate when no clean self-contained boundary exists, and copy each edge "
        "quote exactly from its selected cue. Do not summarize, enrich, or add assessments."
    )
    blocks: list[str] = []
    allowed: dict[str, tuple[set[int], set[int]]] = {}
    n = len(segments)
    for candidate in candidates:
        start_lines = set(range(
            max(0, candidate.start_line - _REPAIR_NEIGHBOR_CUES),
            min(n, candidate.start_line + _REPAIR_NEIGHBOR_CUES + 1),
        ))
        end_lines = set(range(
            max(0, candidate.end_line - _REPAIR_NEIGHBOR_CUES),
            min(n, candidate.end_line + _REPAIR_NEIGHBOR_CUES + 1),
        ))
        allowed[candidate.candidate_id] = (start_lines, end_lines)

        def render(indices: set[int]) -> str:
            return "\n".join(
                f"[{index}] {_mmss(segments[index].get('start', 0.0))} "
                f"{str(segments[index].get('text') or '').strip()}"
                for index in sorted(indices)
            )

        blocks.append(
            f"<candidate id={candidate.candidate_id!r} failed_check={candidate.reason!r}>\n"
            f"title: {candidate.proposal.title}\n"
            "learning objective: "
            f"{getattr(candidate.proposal, 'learning_objective', '') or getattr(candidate.proposal, 'reason', '')}\n"
            f"original cue range: {candidate.proposal.start_line}-{candidate.proposal.end_line}\n"
            f"allowed start_line IDs: {sorted(start_lines)}\n"
            f"<start_neighbors>\n{render(start_lines)}\n</start_neighbors>\n"
            f"allowed end_line IDs: {sorted(end_lines)}\n"
            f"<end_neighbors>\n{render(end_lines)}\n</end_neighbors>\n"
            "</candidate>"
        )
    user = (
        f"Viewer topic: {topic.strip() or '(none)'}.\n\n"
        + "\n\n".join(blocks)
        + "\n\nRepair only the preceding candidates. For each safe repair return "
          "candidate_id, start_line, end_line, start_quote, and end_quote. Each line ID "
          "must come from that candidate's corresponding allowed list."
    )
    return system, user, allowed


def _legacy_prompts(lines: str, n: int, topic: str = "") -> tuple[str, str]:
    """Pre-router production prompt used by ``production_pro_v0``."""
    topic_rule = ""
    if topic.strip():
        topic_rule = (
            f"The viewer is studying: {topic.strip()!r}. Only return clips that TEACH "
            "material relevant to that topic; skip unrelated sections entirely.\n"
        )
    system = (
        "You select self-contained CLIPS from a lecture/talk transcript for a short-form "
        "learning feed. First read and understand the WHOLE transcript. Then pick the "
        "SUBSTANTIVE teaching moments — one coherent idea, concept, worked example, or "
        "section, taught from its introduction through to its natural conclusion. Skip pure "
        "filler (greetings, admin, 'like and subscribe', tangents), course-logistics intros, "
        "and wrap-up outros.\n" + topic_rule +
        "For every clip return: title; start_line (the line where the idea is INTRODUCED); "
        "end_line (the line where it CLOSES); start_quote (the first ~6 words spoken at the "
        "start, copied verbatim from that line); end_quote (the last ~6 words, verbatim); a "
        "short reason; kind — one of content|educational|intro|outro|admin|promo; "
        "informativeness — 0.0 to 1.0, how much a motivated student learns from this clip "
        "ALONE; topic_relevance — 0.0 to 1.0, how directly it teaches the viewer's topic; "
        "self_contained — true only when it makes sense without omitted context; "
        "difficulty — 0.0 to 1.0, the prior knowledge the clip ASSUMES (0.1: no background, "
        "first exposure; 0.5: comfortable with the basics; 0.9: graduate/expert material). "
        "When supported by the clip, also return a grounded summary, two to four distinct "
        "takeaways, a topic-specific match_reason, and an assessment with prompt, exactly four "
        "options, correct_index, explanation, and an exact evidence_quote from the clip. "
        "Rules: (1) a clip must START at the beginning of the idea and END at its end — "
        "never mid-thought; (2) contextual overlap is allowed when two complete ideas share "
        "setup; (3) go in chronological order; (4) select the minimum complete span and never "
        "truncate required context or a conclusion; (5) line indices range from 0 to "
        f"{n - 1} — never exceed {n - 1}."
    )
    user = (
        f"Transcript ({n} lines, each formatted `[index] MM:SS text`):\n\n" + lines +
        "\n\nReturn every substantive teaching clip as {title, start_line, end_line, "
        "start_quote, end_quote, reason, facet, kind, informativeness, topic_relevance, "
        "self_contained, difficulty, summary, takeaways, match_reason, assessment}."
    )
    return system, user


def _enrichment_prompts(clips: list[dict], topic: str) -> tuple[str, str]:
    system = (
        "Ground learning details only in the supplied accepted clip excerpts. Do not add facts "
        "from titles, outside knowledge, or another clip. Each assessment must have exactly one "
        "correct option and an exact evidence quote from its own excerpt. Do not provide "
        "chain-of-thought."
    )
    blocks = []
    for clip in clips:
        blocks.append(
            f"<clip id={clip['_clip_id']!r}>\n{clip['_clip_text']}\n</clip>"
        )
    user = (
        f"Viewer topic: {topic.strip() or '(none)'}.\n\n"
        + "\n\n".join(blocks)
        + "\n\nBased on the preceding accepted clip excerpts, return one item for every clip_id "
          "with a grounded 1-2 sentence summary, 2-4 distinct takeaways, a topic-specific "
          "match_reason, and a four-option assessment whose evidence_quote is copied exactly "
          "from that clip."
    )
    return system, user


def _card_enrichment_prompts(items: list[dict], topic: str) -> tuple[str, str]:
    system = (
        "Enrich already accepted educational clips using only each supplied transcript "
        "excerpt. Do not create quizzes, assessments, outside facts, or chain-of-thought."
    )
    blocks = []
    for item in items[:3]:
        blocks.append(
            f"<clip id={str(item.get('clip_id') or '')!r}>\n"
            f"Title: {str(item.get('title') or '').strip()}\n"
            f"Learning objective: {str(item.get('learning_objective') or '').strip()}\n"
            f"Transcript: {str(item.get('text') or '').strip()}\n"
            "</clip>"
        )
    user = (
        f"Viewer topic: {topic.strip() or '(none)'}.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn one item per clip_id with a grounded 1-2 sentence summary, "
          "2-4 distinct grounded takeaways, and a short topic-specific match_reason."
    )
    return system, user


# ---------------------------------------------------------------------------
# Transcript alignment and validation

def _mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _toks(text: str) -> list[str]:
    return [
        unicodedata.normalize("NFKC", match.group(0))
        .translate(_APOSTROPHES)
        .casefold()
        for match in _WORD_RE.finditer(text or "")
    ]


def _contains_quote(text: str, quote: str) -> bool:
    haystack, needle = _toks(text), _toks(quote)
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def _exact_boundary_quote(text: str, *, want: str) -> str:
    """Return an exact short quote from the retained transcript edge."""
    matches = list(_WORD_RE.finditer(text or ""))
    if not matches:
        return ""
    chosen = matches[:6] if want == "start" else matches[-6:]
    return (text or "")[chosen[0].start():chosen[-1].end()]


def _locate_quote_match(
    words: list[dict], quote: str, lo_t: float, hi_t: float, want: str,
) -> tuple[float, float] | None:
    """Return ``(boundary_time, score)`` for a high-confidence word alignment."""
    quote_tokens = _toks(quote)
    if not quote_tokens or not words:
        return None
    qn = min(len(quote_tokens), 6)
    target_tokens = quote_tokens[:qn] if want == "start" else quote_tokens[-qn:]
    target = " ".join(target_tokens)
    timed: list[tuple[str, float, float]] = []
    for word in words:
        try:
            start = float(word.get("start", 0.0))
            end = float(word.get("end", start))
        except (TypeError, ValueError):
            continue
        if not (lo_t - 1e-6 <= start <= hi_t + 1e-6):
            continue
        tokens = _toks(str(word.get("word") or ""))
        if tokens:
            timed.append((tokens[0], start, end))
    if len(timed) < qn:
        return None

    try:
        from rapidfuzz import fuzz

        score_fn = lambda a, b: float(fuzz.ratio(a, b))
    except Exception:  # pragma: no cover - rapidfuzz is a required dependency
        from difflib import SequenceMatcher

        score_fn = lambda a, b: 100.0 * SequenceMatcher(None, a, b).ratio()

    best: tuple[float, float] | None = None
    indices = range(len(timed) - qn + 1)
    if want == "end":
        indices = reversed(list(indices))
    for i in indices:
        window = " ".join(token for token, _start, _end in timed[i:i + qn])
        score = score_fn(target, window)
        boundary = timed[i][1] if want == "start" else timed[i + qn - 1][2]
        if best is None or score > best[1]:
            best = (boundary, score)
    return best if best is not None and best[1] >= 85.0 else None


def _locate_quote(words: list[dict], quote: str, lo_t: float, hi_t: float,
                  want: str) -> Optional[float]:
    match = _locate_quote_match(words, quote, lo_t, hi_t, want)
    return match[0] if match else None


def _guard_text(text: str, *, ignore_caption_case: bool) -> str:
    """Remove only the unreliable lowercase signal from auto-caption guards."""
    from .discourse import first_lexical_character_index

    normalized = str(text or "").strip()
    lexical_index = first_lexical_character_index(normalized)
    if (
        ignore_caption_case
        and lexical_index is not None
        and normalized[lexical_index].islower()
    ):
        normalized = (
            normalized[:lexical_index]
            + normalized[lexical_index].upper()
            + normalized[lexical_index + 1:]
        )
    return normalized


def _cue_opens_mid_thought(text: str, *, ignore_caption_case: bool) -> bool:
    from .discourse import CONTEXT_DEP_HEADS, opens_mid_thought

    guarded = _guard_text(text, ignore_caption_case=ignore_caption_case)
    opening_clause = re.split(r"[.!?]", guarded, maxsplit=1)[0]
    existential = _EXISTENTIAL_OPENING_RE.fullmatch(opening_clause)
    if existential is not None:
        tail = existential.group("tail")
        tail_words = [word.casefold() for word in _WORD_RE.findall(tail)]
        that_count = sum(word == "that" for word in tail_words)
        has_contextual_definite = any(
            word == "the"
            and index + 1 < len(tail_words)
            and tail_words[index + 1] in CONTEXT_DEP_HEADS
            for index, word in enumerate(tail_words)
        )
        if (
            not _EXISTENTIAL_UNRESOLVED_RE.search(tail)
            and not _EXISTENTIAL_CONTEXTUAL_THAT_RE.search(tail)
            and not _EXISTENTIAL_TERMINAL_REFERENCE_RE.search(tail)
            and not _EXISTENTIAL_DEMONSTRATIVE_THAT_RE.search(tail)
            and that_count <= 1
            and not _EXISTENTIAL_BACK_REFERENCE_RE.search(tail)
            and not has_contextual_definite
            and _EXISTENTIAL_SCOPE_RE.search(tail)
        ):
            return False
    return opens_mid_thought(guarded)


def _opening_contextual_example_needs_context(text: str) -> bool:
    """Distinguish a backward reference from a locally restated example setup."""
    raw_text = str(text or "").strip()
    match = _OPENING_CONTEXTUAL_EXAMPLE_RE.match(raw_text)
    if match is None:
        return False
    tail = raw_text[match.end():]
    return not _local_example_setup_is_complete(tail)


def _opening_has_context_dependent_subject(text: str) -> bool:
    raw_text = str(text or "").strip()
    match = re.match(
        r"^\s*(?:(?:after|before|during|following|using|with|without)\b"
        r"[^,;:]{0,100}[,;:]\s*)?"
        r"(?:the|this|that|these|those)\s+"
        r"(?P<head>answer|calculation|case|concept|equation|example|expression|"
        r"formula|method|problem|process|proof|result|solution|step|term|thing|value)\b",
        raw_text,
        re.IGNORECASE,
    )
    if match is None:
        return False
    if match.group("head").casefold() in {"concept", "method", "process", "proof"}:
        named_complement = re.match(
            r"\s+of\s+(?!(?:it|this|that|these|those|them)\b)[a-z0-9]",
            raw_text[match.end():],
            re.IGNORECASE,
        )
        if named_complement is not None:
            return False
    return True


def _general_local_setup_is_complete(text: str) -> bool:
    """Recognize a grounded non-math statement, imperative, or question."""
    raw_text = str(text or "").strip(" ,;:—-")
    if (
        _OPENING_AGENDA_RE.match(raw_text)
        or _OPENING_CONTEXTUAL_REFORMULATION_RE.match(raw_text)
        or _opening_has_context_dependent_subject(raw_text)
        or len(_toks(raw_text)) < 3
        or _cue_opens_mid_thought(
        raw_text,
        ignore_caption_case=True,
        )
    ):
        return False
    words = _toks(raw_text)
    if words[0] in _STANDALONE_QUESTION_HEADS:
        return len(_content_tokens(raw_text)) >= 2
    action = _GENERAL_SETUP_ACTION_RE.match(raw_text)
    if action is not None:
        generic = {
            "a", "again", "an", "answer", "briefly", "carefully", "exactly",
            "it", "merely", "one", "result", "simply", "step", "that", "the",
            "them", "thing", "this", "value",
        }
        concrete = {
            token for token in _toks(action.group("object"))
            if token not in generic and len(token) >= 3
        }
        return bool(concrete)
    if words[0] in {
        "and", "but", "he", "here", "i", "it", "or", "she", "so", "that",
        "there", "these", "they", "this", "those", "we", "you",
    }:
        return False
    if _UNGROUNDED_IMPERATIVE_RE.match(raw_text):
        return False
    return len(_content_tokens(raw_text)) >= 3


def _definition_has_named_subject(text: str, definition_start: int) -> bool:
    """Require a noun or symbol before ``is defined as``, not a bare pronoun."""
    prefix = str(text or "")[:definition_start]
    framing = list(re.finditer(
        r"\b(?:assume|consider|given|let|suppose)\b",
        prefix,
        re.IGNORECASE,
    ))
    subject = prefix[framing[-1].end():] if framing else prefix
    generic = {
        "a", "an", "it", "one", "that", "the", "these", "they", "thing",
        "this", "those",
    }
    return any(
        (len(token) == 1 and token not in {"a", "i"})
        or (len(token) >= 2 and token not in generic)
        for token in _toks(subject)
    )


def _explicit_problem_resolves_back_reference(
    text: str,
    explicit_problem: re.Match[str],
    back_reference: re.Match[str],
) -> bool:
    """Reject ``the problem is this`` while retaining a stated expression/result."""
    if explicit_problem.start() >= back_reference.start():
        return False
    between = str(text or "")[explicit_problem.end():back_reference.start()]
    generic = {
        "a", "again", "an", "and", "by", "carefully", "exactly", "for",
        "just", "merely", "now", "of", "one", "only", "simply", "so", "the",
        "then", "to", "with",
    }
    tokens = _toks(between)
    has_named_symbol = any(
        len(token) == 1 and token not in {"a", "i"}
        for token in tokens
    )
    concrete_tokens = {
        token for token in tokens
        if token not in generic and len(token) >= 2
    }
    return has_named_symbol or len(concrete_tokens) >= 2


def _explicit_problem_has_substantive_object(
    text: str,
    explicit_problem: re.Match[str],
) -> bool:
    """Require the actual target/expression, not merely ``the problem is``."""
    matched = explicit_problem.group(0).casefold()
    statement = str(text or "")[explicit_problem.start():]
    statement = re.split(r"[.!?]", statement, maxsplit=1)[0]
    generic = {
        "a", "again", "an", "answer", "calculate", "determine", "equation",
        "equals", "evaluate", "expression", "factor", "find", "function", "given",
        "gives", "is", "it", "just", "one", "problem", "result", "same", "so",
        "solve", "that", "the", "this", "to", "unknown",
    }
    if re.match(r"\s*(?:given|let)\b", matched):
        tail = str(text or "")[explicit_problem.end():]
        tail = re.split(r"[.!?]", tail, maxsplit=1)[0]
        relation = re.search(
            r"\b(?:be|denote|equal|equals|represent)\b|=",
            tail,
            re.IGNORECASE,
        )
        if relation is not None:
            object_tokens = {
                token for token in _toks(tail[relation.end():])
                if token not in {
                    "a", "an", "it", "that", "the", "this", "unknown",
                }
            }
            return bool(object_tokens)
        meaningful_tail = {
            token for token in _toks(tail)
            if token not in generic
        }
        return len(meaningful_tail) >= 2
    meaningful = {
        token for token in _toks(statement)
        if token not in generic
    }
    if "limit" in _toks(matched):
        target = re.split(
            r"\b(?:approaches|tends\s+to)\b",
            statement,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        return bool(
            len(target) == 2
            and {
                token for token in _toks(target[1])
                if token not in generic
            }
        )
    return bool(meaningful)


def _local_example_setup_is_complete(text: str) -> bool:
    """Require a stated problem/object, while allowing an explicitly defined referent."""
    raw_text = str(text or "").strip()
    if (
        _OPENING_AGENDA_RE.match(raw_text)
        or _opening_is_dependent_question_tail(raw_text)
    ):
        return False
    contextual_frame = _OPENING_CONTEXTUAL_EXAMPLE_RE.match(raw_text)
    if contextual_frame is not None:
        return _local_example_setup_is_complete(
            raw_text[contextual_frame.end():]
        )
    if not (
        _OPENING_RECOVERABLE_SETUP_RE.match(raw_text)
        or _LOCAL_EXAMPLE_SETUP_RE.search(raw_text)
    ):
        return _general_local_setup_is_complete(raw_text)
    explicit_problem = _LOCAL_EXPLICIT_PROBLEM_RE.search(raw_text)
    if (
        explicit_problem is not None
        and not _explicit_problem_has_substantive_object(
            raw_text,
            explicit_problem,
        )
    ):
        return False
    action = _LOCAL_SETUP_ACTION_RE.search(raw_text)
    if action is not None and explicit_problem is None:
        generic_object_tokens = {
            "a", "again", "an", "answer", "by", "carefully", "for", "it",
            "next", "of", "one", "problem", "result", "step", "that", "the",
            "them", "thing", "this", "to", "value", "with",
        }
        object_tokens = _toks(action.group("object"))
        has_named_symbol = any(
            len(token) == 1 and token not in {"a", "i"}
            for token in object_tokens
        )
        concrete_tokens = {
            token for token in object_tokens
            if token not in generic_object_tokens and len(token) >= 2
        }
        if not has_named_symbol and len(concrete_tokens) < 2:
            return False
    back_reference = next(
        (
            match
            for match in _OPENING_SETUP_BACK_REFERENCE_RE.finditer(raw_text)
            if not (
                match.group(0).casefold() in {"here", "there"}
                and re.match(
                    r"\s+(?:is|are|was|were)\b",
                    raw_text[match.end():],
                    re.IGNORECASE,
                )
            )
        ),
        None,
    )
    if back_reference is None:
        return True
    explicit_definition = _LOCAL_SETUP_EXPLICIT_GROUNDING_RE.search(raw_text)
    return bool(
        (
            explicit_definition is not None
            and _definition_has_named_subject(
                raw_text,
                explicit_definition.start(),
            )
        )
        or (
            explicit_problem is not None
            and _explicit_problem_resolves_back_reference(
                raw_text,
                explicit_problem,
                back_reference,
            )
        )
    )


def _selected_example_restates_complete_target(text: str) -> bool:
    """Keep a complete worked unit when its explicit target is restated later."""
    raw_text = str(text or "").strip()
    frame = _OPENING_CONTEXTUAL_EXAMPLE_RE.match(raw_text)
    if frame is None or len(_content_tokens(raw_text)) < 12:
        return False
    explicit_targets = [
        match
        for match in _LOCAL_EXPLICIT_PROBLEM_RE.finditer(raw_text, frame.end())
        if _explicit_problem_has_substantive_object(raw_text, match)
    ]
    if not explicit_targets or _terminal_content_is_explicitly_incomplete(raw_text):
        return False
    return bool(
        _LATER_RESTATED_EXAMPLE_REASONING_RE.search(raw_text)
        and _LATER_RESTATED_EXAMPLE_CONCLUSION_RE.search(raw_text)
    )


def _cue_opens_mid_thought_at(
    segments: list[dict],
    index: int,
    *,
    ignore_caption_case: bool,
) -> bool:
    """Use the preceding cue to recover reliable lowercase-fragment evidence."""
    from .discourse import first_lexical_character_index

    text = str(segments[index].get("text") or "").strip()
    sentence_spans = _sentence_character_spans(text)
    opening_text = (
        text[:sentence_spans[0][1]].strip() if sentence_spans else text
    )
    if _opening_is_dependent_question_tail(opening_text):
        if index <= 0:
            return True
        from .sentences import classify_terminator

        previous_text = str(segments[index - 1].get("text") or "").strip()
        if not classify_terminator(previous_text):
            return True
    if (
        _DANGLING_TAIL_PREFIX_RE.search(opening_text)
        or _OPENING_DEMONSTRATIVE_REFERENCE_RE.match(opening_text)
        or _OPENING_BARE_RELATIONAL_PREDICATE_RE.match(opening_text)
        or _OPENING_DEPENDENT_PREPOSITION_FRAGMENT_RE.match(opening_text)
        or _OPENING_CONTEXTUAL_REFORMULATION_RE.match(opening_text)
        or _opening_contextual_example_needs_context(opening_text)
        or _OPENING_AGENDA_RE.match(opening_text)
        or _NEXT_EXAMPLE_FRAMING_RE.fullmatch(opening_text)
        or _OPENING_PREPOSITIONAL_TAG_RE.match(opening_text)
    ):
        return True
    if _cue_opens_mid_thought(
        text, ignore_caption_case=ignore_caption_case
    ):
        return True
    if index > 0:
        previous_text = str(segments[index - 1].get("text") or "")
        rolling_context_tokens = _toks(" ".join(
            str(segments[line].get("text") or "")
            for line in range(max(0, index - 2), index)
        ))
        opening_tokens = _toks(opening_text)
        rolling_overlap = any(
            rolling_context_tokens[-width:] == opening_tokens[:width]
            for width in range(
                min(12, len(rolling_context_tokens), len(opening_tokens)),
                2,
                -1,
            )
        )
        if (
            _cue_has_explicit_dangling_end(
                previous_text,
                text,
            )
        ):
            return True
        # A model quote can begin at a grammatical-looking framing clause even
        # though the provider split one continuous thought immediately before
        # its discourse marker (for example, ``... h prime of x / so I want``).
        # Keeping the preceding cue is a safe, cheap context fallback; a real
        # section pause is still enforced by ``_close_cue_context``.
        from .discourse import CONTINUATION_MARKERS
        from .sentences import classify_terminator

        opening_words = _toks(text)
        subjectless_predicate = _OPENING_SUBJECTLESS_PREDICATE_RE.match(
            opening_text
        )
        following_text = (
            str(segments[index + 1].get("text") or "").strip()
            if index + 1 < len(segments)
            else ""
        )
        question_words = _toks(opening_text)
        answer_words = _toks(following_text)
        subject_index = 1
        if (
            len(question_words) > subject_index
            and question_words[subject_index] in {"a", "an", "the"}
        ):
            subject_index += 1
        question_subject = (
            question_words[subject_index]
            if len(question_words) > subject_index
            else ""
        )
        question_predicate_index = subject_index + 1
        answer_subject = ""
        if answer_words:
            answer_subject_index = (
                1 if answer_words[0] in {"a", "an", "the"} else 0
            )
            if len(answer_words) > answer_subject_index:
                answer_subject = answer_words[answer_subject_index]
        grounded_declarative_answer = bool(
            question_subject
            and answer_subject == question_subject
            and len(question_words) > question_predicate_index
            and question_words[question_predicate_index]
            not in {"by", "for", "from", "of", "to", "with"}
        )
        independent_auxiliary_question = bool(
            subjectless_predicate
            and _cue_begins_standalone_question(opening_text)
            and not _cue_has_explicit_dangling_end(opening_text, "")
            and (
                "?" in opening_text
                or re.match(r"^\s*(?:no|yes)\b", following_text, re.IGNORECASE)
                or grounded_declarative_answer
            )
        )
        if (
            not classify_terminator(previous_text)
            and "?" not in opening_text
            and (
                rolling_overlap
                or (
                    subjectless_predicate
                    and not independent_auxiliary_question
                )
                or _OPENING_MATH_CONTINUATION_RE.match(opening_text)
            )
        ):
            return True
        if (
            opening_words
            and opening_words[0] in {
                "by", "during", "for", "from", "into", "of", "onto", "to",
                "while", "with", "without",
            }
            and not _opening_is_independent_preposition_question(opening_text)
            and not classify_terminator(previous_text)
        ):
            return True
        if (
            opening_words
            and opening_words[0] in CONTINUATION_MARKERS
            and not classify_terminator(previous_text)
        ):
            return True
    if _cue_begins_standalone_question(text):
        return False
    opening_terminator = re.search(r"[.!?]", text)
    lexical_index = first_lexical_character_index(text)
    starts_lowercase = bool(
        lexical_index is not None and text[lexical_index].islower()
    )
    if (
        not ignore_caption_case
        or index <= 0
        or not starts_lowercase
        or (
            opening_terminator is not None
            and opening_terminator.group(0) == "?"
        )
    ):
        return False

    previous_text = str(segments[index - 1].get("text") or "").strip()
    if re.search(r"[,;:\-—][\"')\]]*$", previous_text):
        return True
    words = _toks(text)
    if words and words[0] in {
        "by", "during", "from", "into", "of", "onto", "to", "while", "with",
        "without",
    }:
        return True

    from .sentences import classify_terminator

    previous_words = _toks(previous_text)
    explicit_closure_words = {
        "answer", "answered", "complete", "completed", "completely", "conclusion",
        "end", "ended", "final", "finished", "finishes", "result", "solved",
    }
    return bool(
        len(previous_words) >= 5
        and not classify_terminator(previous_text)
        and not explicit_closure_words.intersection(previous_words[-5:])
    )


def _cue_begins_standalone_question(text: str) -> bool:
    words = _toks(text)
    return bool(
        words
        and (
            words[0] in _STANDALONE_QUESTION_HEADS
            or (
                len(words) > 1
                and words[0] in _QUESTION_PREFIXES
                and words[1] in _STANDALONE_QUESTION_HEADS
            )
        )
    )


def _opening_is_independent_preposition_question(text: str) -> bool:
    """Recognize a complete question introduced by a prepositional frame."""
    opening = str(text or "").strip()
    question = re.match(r"^[^?]*\?", opening)
    if question is None:
        return False
    clause = question.group(0)
    return bool(
        _PREPOSITION_FRONTED_QUESTION_RE.match(clause)
        or _FRAMED_QUESTION_ONSET_RE.search(clause)
    )


def _opening_is_dependent_question_tail(text: str) -> bool:
    """Detect a caption fragment that contains only the tail of a question."""
    opening = str(text or "").strip()
    question = re.match(r"^[^?]*\?", opening)
    if question is None:
        return False
    clause = question.group(0)
    if (
        _cue_begins_standalone_question(clause)
        or _opening_is_independent_preposition_question(clause)
        or not _OPENING_DEPENDENT_QUESTION_TAIL_RE.match(clause)
    ):
        return False
    return True


def _cue_has_weak_end(
    text: str,
    next_text: str,
    *,
    ignore_caption_case: bool,
) -> bool:
    """Use the existing weak-end guard and continuation onset as cue evidence."""
    from .refine import _is_weak_end
    from .sentences import Sentence, classify_terminator

    raw_text = str(text or "").strip()
    visual_pointer = _TRAILING_VISUAL_POINTER_RE.search(raw_text)
    if visual_pointer is not None:
        retained = raw_text[:visual_pointer.start()].rstrip(" ,;:—-")
        if (
            len(_toks(retained)) >= 5
            and not _terminal_content_is_explicitly_incomplete(retained)
            and not _cue_has_explicit_dangling_end(retained, "")
            and not _TERMINAL_REQUIRED_VISUAL_COMPLEMENT_RE.search(retained)
        ):
            return False
    if _cue_has_explicit_dangling_end(raw_text, next_text):
        return True
    if _TERMINAL_COORDINATING_CONJUNCTION_RE.search(raw_text):
        return True
    dependent_complement = _NEXT_DEPENDENT_COMPLEMENT_RE.match(
        str(next_text or "")
    )
    if dependent_complement is not None and not classify_terminator(raw_text):
        next_opening = dependent_complement.group(0).strip().casefold()
        if (
            not next_opening.startswith("of")
            or (
                _OPENING_INDEPENDENT_OF_FRAME_RE.match(str(next_text or ""))
                is None
                and _TERMINAL_OF_COMPLEMENT_HEAD_RE.search(raw_text)
            )
        ):
            return True
    if (
        _NEXT_SYMBOL_LABEL_CONTINUATION_RE.match(str(next_text or ""))
        and re.search(
            r"\b(?:answer|derivative|expression|formula|function|result|"
            r"solution|value|variable)\s*$",
            raw_text,
            re.IGNORECASE,
        )
    ):
        return True
    if _NEXT_SAME_UNIT_CONTINUATION_RE.match(str(next_text or "")):
        return True
    question_mark = raw_text.find("?")
    answer_words = _toks(raw_text[question_mark + 1:]) if question_mark >= 0 else []
    has_local_answer = bool(
        len(answer_words) >= 2
        or (answer_words and answer_words[0] in {"no", "yes"})
    )
    if (
        question_mark >= 0
        and _cue_begins_standalone_question(raw_text)
        and not has_local_answer
    ):
        return True
    guarded = _guard_text(raw_text, ignore_caption_case=ignore_caption_case)
    terminator = classify_terminator(guarded)
    if terminator and _TERMINAL_STRANDED_PREPOSITION_RE.search(guarded):
        return False
    sentence = Sentence(
        idx=0,
        text=guarded,
        start=0.0,
        end=1.0,
        terminator=terminator,
        ends_with_period=bool(terminator),
        word_start_idx=0,
        word_end_idx=0,
        align_confidence=1.0,
    )
    if _TERMINAL_COMPLETE_SHORT_NP_RE.fullmatch(guarded.strip()):
        return False
    if _is_weak_end(sentence):
        return True
    if terminator or not next_text:
        return False
    next_noise_start = _trailing_edge_noise_start(next_text)
    if (
        next_noise_start is not None
        and (
            prefix_words := _toks(str(next_text)[:next_noise_start])
        )
        and len(prefix_words) <= 5
        and {"and", "or", "plus"}.intersection(prefix_words)
    ):
        # A short lexical prefix before navigation/outro commonly completes a
        # list or clause split by fixed-size caption cues.
        return True
    # Supadata's fixed-size cues frequently split an ordinary phrase without
    # punctuation or an explicit conjunction. Long bare edges and gerund-led
    # constructions are therefore uncertain even when the next cue starts with
    # a capitalized noun phrase (for example, "by substituting / the numbers").
    raw_words = _toks(guarded)
    explicit_closure_words = {
        "answer", "answered", "complete", "completed", "completely", "conclusion",
        "end", "ended", "final", "finished", "finishes", "result", "solved",
    }
    if explicit_closure_words.intersection(raw_words[-5:]):
        return False
    next_words = _toks(next_text)
    if (
        next_words
        and next_words[0] == "prime"
        and _TERMINAL_PRIME_DIRECTION_RE.search(raw_text)
    ):
        return True
    if (
        next_words
        and len(next_words[0]) > 5
        and next_words[0].endswith("ing")
    ):
        return True
    if (
        len(raw_words) >= 2
        and raw_words[-1].endswith("ing")
        and len(raw_words[-1]) > 5
        and raw_words[-2]
        in {"after", "are", "before", "being", "by", "is", "was", "were", "while"}
    ):
        return True
    if len(next_words) >= 2 and next_words[1] in {
        "that", "when", "where", "which", "whose",
    }:
        return True
    if not _cue_opens_mid_thought(
        next_text, ignore_caption_case=ignore_caption_case
    ):
        return False
    # A bare auto-caption edge needs expansion only when the following cue has
    # an explicit dependency signal. The onset guard's lowercase/short-text
    # fallbacks are intentionally insufficient by themselves here.
    from .discourse import ANAPHORS, CONTEXT_DEP_HEADS, CONTINUATION_MARKERS, _AUX_VERB

    words = next_words
    if not words:
        return False
    if len(words) < 3 and words[0] in _AUX_VERB:
        return True
    return bool(
        words[0] in CONTINUATION_MARKERS
        or words[0] in ANAPHORS
        or (len(words) > 1 and words[0] == "the" and words[1] in CONTEXT_DEP_HEADS)
    )


def _cue_has_explicit_dangling_end(text: str, next_text: str) -> bool:
    """Detect syntax that cannot end a thought without broad caption guesses."""
    from .discourse import first_lexical_character_index

    raw_text = str(text or "").strip()
    next_lexical_index = first_lexical_character_index(next_text)
    ambiguous_degree_continues = bool(
        _TERMINAL_AMBIGUOUS_DEGREE_RE.search(raw_text)
        and next_lexical_index is not None
        and str(next_text)[next_lexical_index].islower()
    )
    nominal_subject_continues = bool(
        _TERMINAL_NOMINAL_SUBJECT_RE.search(raw_text)
        and _NOMINAL_PREDICATE_CONTINUATION_RE.match(str(next_text or ""))
    )
    return bool(
        _TERMINAL_CALLBACK_RE.search(raw_text)
        or _TERMINAL_DANGLING_TRANSITION_RE.search(raw_text)
        or _TERMINAL_DANGLING_EXAMPLE_INTRO_RE.search(raw_text)
        or _TERMINAL_INCOMPLETE_SUBJECT_RE.search(raw_text)
        or _TERMINAL_BARE_SUBJECT_RE.search(raw_text)
        or nominal_subject_continues
        or _TERMINAL_DANGLING_ARTICLE_RE.search(raw_text)
        or _TERMINAL_DANGLING_LINK_RE.search(raw_text)
        or (
            bool(next_text.strip())
            and _TERMINAL_HEADLESS_QUANTIFIER_RE.search(raw_text)
        )
        or _TERMINAL_DANGLING_MODAL_PREDICATE_RE.search(raw_text)
        or _TERMINAL_DANGLING_AUXILIARY_ADVERB_RE.search(raw_text)
        or _TERMINAL_DANGLING_DEGREE_RE.search(raw_text)
        or (
            bool(str(next_text or "").strip())
            and _TERMINAL_DANGLING_DISCOURSE_LEADIN_RE.search(raw_text)
        )
        or ambiguous_degree_continues
        or _has_unfinished_exemplification_tail(raw_text)
        or re.search(r"[,;:\-—][\"')\]]*$", raw_text)
    )


def _has_unfinished_exemplification_tail(text: str) -> bool:
    """Detect a terminal example setup that never reaches its explanatory clause."""
    raw_text = str(text or "").strip()
    match = _TERMINAL_EXEMPLIFICATION_RE.search(raw_text)
    if match is None:
        return False
    tail_words = _toks(match.group("tail"))
    if not tail_words:
        return True
    if match.group("so") and len(tail_words) <= 3:
        return True
    return len(tail_words) <= 2 and not re.search(r"[.!?][\"')\]]*$", raw_text)


def _has_unanswered_terminal_question(text: str) -> bool:
    """Return true only when the final semantic act is a question with no answer."""
    raw_text = str(text or "").strip()
    question_mark = raw_text.rfind("?")
    if question_mark < 0:
        return False
    tail_words = _toks(raw_text[question_mark + 1:])
    return not (
        len(tail_words) >= 2
        or (tail_words and tail_words[0] in {"no", "yes"})
    )


def _terminal_content_is_explicitly_incomplete(text: str) -> bool:
    """Separate semantic incompleteness from an imperfect transcript edge.

    Boundary heuristics are deliberately conservative and may be uncertain on
    unpunctuated captions. Only direct evidence of a missing educational act is
    allowed to fail the candidate; all other edge uncertainty is shipped with a
    diagnostic so a good teaching unit is never lost to caption segmentation.
    """
    raw_text = str(text or "").strip()
    return bool(
        _has_unanswered_terminal_question(raw_text)
        or _has_unfinished_exemplification_tail(raw_text)
        or _TERMINAL_DANGLING_EXAMPLE_INTRO_RE.search(raw_text)
        or _TERMINAL_EXPLICIT_INCOMPLETE_CLAUSE_RE.search(raw_text)
        or _TERMINAL_COORDINATING_CONJUNCTION_RE.search(raw_text)
        or _TERMINAL_REQUIRED_COMPLEMENT_RE.search(raw_text)
    )


def _complete_prefix_end_quote(text: str) -> str:
    """Recover the last complete sentence before an unfinished edge suffix."""
    raw_text = str(text or "")
    for match in reversed(list(re.finditer(r"[.!?]+[\"'’”)]*", raw_text))):
        prefix = raw_text[:match.end()].strip()
        suffix = raw_text[match.end():]
        if not _WORD_RE.search(prefix) or not _WORD_RE.search(suffix):
            continue
        if _terminal_content_is_explicitly_incomplete(prefix):
            continue
        quote = _exact_boundary_quote(prefix, want="end")
        span = _quote_character_span(prefix, quote)
        return prefix[span[0]:].strip() if span is not None else quote
    return ""


def _cue_boundary_confidence(text: str, *, ignore_caption_case: bool) -> float:
    from .sentences import classify_terminator

    guarded = _guard_text(text, ignore_caption_case=ignore_caption_case)
    return 1.0 if classify_terminator(guarded) else 0.90


def _trim_trailing_incomplete_suffix(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    protected_quote: str = "",
    learning_objective: str = "",
) -> int | None:
    """Trim a cue-aligned teaser/transition suffix, or reject if no clean prefix exists."""
    from .discourse import first_lexical_character_index
    from .sentences import classify_terminator

    following_text = (
        str(segments[end_line + 1].get("text") or "")
        if end_line + 1 < len(segments)
        else ""
    )
    solution_continues = bool(
        _FORWARD_SOLUTION_CONTINUATION_RE.search(following_text)
    )
    full_suffix = _cue_clip_text(segments, start_line, end_line)
    final_text = str(segments[end_line].get("text") or "")
    following_lexical_index = first_lexical_character_index(following_text)
    ambiguous_degree = _TERMINAL_AMBIGUOUS_DEGREE_RE.search(full_suffix)
    bare_subject = _TERMINAL_BARE_SUBJECT_RE.search(full_suffix)
    possible_trigger = bool(
        _TERMINAL_DANGLING_TRANSITION_RE.search(full_suffix)
        or _TERMINAL_DANGLING_DEGREE_RE.search(full_suffix)
        or (
            ambiguous_degree
            and following_lexical_index is not None
            and following_text[following_lexical_index].islower()
        )
        or (
            bare_subject
            and re.search(r"[.!?]", full_suffix[:bare_subject.start()])
        )
        or (
            solution_continues
            and _TRAILING_FORWARD_SETUP_RE.search(full_suffix)
        )
        or _FORWARD_TOPIC_TRANSITION_RE.match(final_text)
        or _unconditional_trailing_edge_noise_start(
            final_text,
            require_edge_prefix=True,
        )
        is not None
        or _trailing_version_edge_noise_start(
            final_text,
            require_edge_prefix=True,
        )
        is not None
    )
    if not possible_trigger:
        return end_line
    for line in range(end_line, start_line - 1, -1):
        suffix = _cue_clip_text(segments, line, end_line)
        unconditional_edge_noise = (
            line == end_line
            and _unconditional_trailing_edge_noise_start(
                suffix,
                require_edge_prefix=True,
            )
            is not None
            and not (
                protected_quote
                and _contains_quote(suffix, protected_quote)
            )
        )
        version_edge_noise = (
            line == end_line
            and _trailing_version_edge_noise_start(
                suffix,
                require_edge_prefix=True,
            )
            is not None
            and not (
                protected_quote
                and _contains_quote(suffix, protected_quote)
            )
            and not _objective_bridges_sections(
                learning_objective,
                _cue_clip_text(segments, start_line, end_line - 1),
                suffix,
            )
        )
        dangling_transition = bool(
            _TERMINAL_DANGLING_TRANSITION_RE.search(suffix)
        )
        forward_setup = bool(
            solution_continues
            and _TRAILING_FORWARD_SETUP_RE.search(suffix)
        )
        forward_transition = bool(
            line == end_line and _FORWARD_TOPIC_TRANSITION_RE.match(suffix)
        )
        dangling_degree = bool(_TERMINAL_DANGLING_DEGREE_RE.search(suffix))
        ambiguous_degree = _TERMINAL_AMBIGUOUS_DEGREE_RE.search(suffix)
        dangling_ambiguous_degree = bool(
            ambiguous_degree
            and following_lexical_index is not None
            and following_text[following_lexical_index].islower()
        )
        bare_subject = _TERMINAL_BARE_SUBJECT_RE.search(suffix)
        dangling_bare_subject = bool(
            bare_subject
            and re.search(r"[.!?]", suffix[:bare_subject.start()])
        )
        if not (
            dangling_transition
            or dangling_degree
            or dangling_ambiguous_degree
            or dangling_bare_subject
            or forward_setup
            or forward_transition
            or unconditional_edge_noise
            or version_edge_noise
        ):
            continue
        previous_line = line - 1
        if previous_line < start_line:
            return end_line if forward_setup else None
        previous_text = str(segments[previous_line].get("text") or "").strip()
        if unconditional_edge_noise or version_edge_noise:
            retained_prefix = _cue_clip_text(
                segments,
                start_line,
                previous_line,
            )
            return (
                previous_line
                if _last_safe_complete_prefix(retained_prefix)
                else None
            )
        if (
            forward_transition
            and len(_toks(previous_text)) >= 3
            and not _cue_has_explicit_dangling_end(previous_text, "")
        ):
            return previous_line
        if classify_terminator(previous_text):
            return previous_line
        return end_line if forward_setup else None
    return end_line


def _close_cue_context(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    ignore_caption_case: bool,
    cue_limit: int | None = None,
    start_boundary_verified: bool = False,
    end_boundary_verified: bool = False,
    protected_quote: str = "",
    learning_objective: str = "",
    min_start_line: int | None = None,
    max_end_line: int | None = None,
) -> tuple[int, int, str | None]:
    """Expand dirty edges until discourse closes or a real section edge is reached."""
    expansion_limit = (
        len(segments) if cue_limit is None else max(0, int(cue_limit))
    )

    def crosses_section_reset(left: int, right: int) -> bool:
        return (
            float(segments[right].get("start", 0.0))
            - float(segments[left].get("end", 0.0))
            >= _SECTION_RESET_GAP_S
        )

    following_text = (
        str(segments[end_line + 1].get("text") or "")
        if end_line + 1 < len(segments)
        else ""
    )
    forward_solution_needed = bool(
        _FORWARD_SOLUTION_CONTINUATION_RE.search(following_text)
        and any(
            _TRAILING_FORWARD_SETUP_RE.search(
                _cue_clip_text(segments, line, end_line)
            )
            for line in range(start_line, end_line + 1)
        )
    )
    trimmed_end = _trim_trailing_incomplete_suffix(
        segments,
        start_line,
        end_line,
        protected_quote=protected_quote,
        learning_objective=learning_objective,
    )
    if trimmed_end is None:
        return start_line, end_line, "unresolved_weak_end"
    suffix_was_trimmed = trimmed_end < end_line
    end_line = trimmed_end
    force_end_clause_completion = bool(
        _TERMINAL_INCOMPLETE_SUBJECT_RE.search(
            str(segments[end_line].get("text") or "").strip()
        )
    )
    from .sentences import classify_terminator

    force_start_question_setup = bool(
        start_line > 0
        and _OPENING_COMPARATIVE_FRAGMENT_RE.search(
            str(segments[start_line].get("text") or "").strip()
        )
        and not classify_terminator(
            str(segments[start_line - 1].get("text") or "").strip()
        )
    )
    if _DANGLING_TAIL_PREFIX_RE.search(
        str(segments[start_line].get("text") or "").strip()
    ):
        for candidate in range(start_line + 1, end_line + 1):
            if crosses_section_reset(candidate - 1, candidate) and not _cue_opens_mid_thought_at(
                segments,
                candidate,
                ignore_caption_case=ignore_caption_case,
            ):
                start_line = candidate
                break
    selected_start_line = start_line
    from .discourse import _has_unresolved_opening_back_reference

    selected_start_text = str(segments[selected_start_line].get("text") or "")
    opening_reference_requires_context = bool(
        not start_boundary_verified
        and _has_unresolved_opening_back_reference(selected_start_text)
    )

    def opening_reference_is_unresolved() -> bool:
        if not opening_reference_requires_context:
            return False
        prior_text = (
            _cue_clip_text(segments, start_line, selected_start_line - 1)
            if start_line < selected_start_line
            else ""
        )
        return _has_unresolved_opening_back_reference(
            selected_start_text,
            prior_text=prior_text,
        )

    original_end = end_line
    start_expansions = 0
    while start_expansions < expansion_limit:
        if start_boundary_verified:
            break
        if force_start_question_setup and _cue_begins_standalone_question(
            str(segments[start_line].get("text") or "")
        ):
            force_start_question_setup = False
            break
        if (
            not force_start_question_setup
            and not _cue_opens_mid_thought_at(
                segments,
                start_line,
                ignore_caption_case=ignore_caption_case,
            )
            and not opening_reference_is_unresolved()
        ):
            break
        candidate = start_line - 1
        if candidate < 0:
            break
        if min_start_line is not None and candidate < min_start_line:
            break
        if crosses_section_reset(candidate, start_line):
            break
        start_line = candidate
        start_expansions += 1
    unresolved_start = bool(
        not start_boundary_verified
        and (
            force_start_question_setup
            or _cue_opens_mid_thought_at(
                segments,
                start_line,
                ignore_caption_case=ignore_caption_case,
            )
            or opening_reference_is_unresolved()
        )
    )

    consumed_end_cues = 0
    if forward_solution_needed and not suffix_was_trimmed:
        candidate = end_line + 1
        if candidate >= len(segments) or expansion_limit <= 0:
            return start_line, end_line, "unresolved_weak_end"
        if max_end_line is not None and candidate > max_end_line:
            return start_line, end_line, "unresolved_weak_end"
        if crosses_section_reset(end_line, candidate):
            return start_line, end_line, "unresolved_weak_end"
        end_line = candidate
        consumed_end_cues = 1

    end_cue_limit = (
        0
        if suffix_was_trimmed or end_boundary_verified
        else max(0, expansion_limit - consumed_end_cues)
    )
    end_expansions = 0
    while end_expansions < end_cue_limit:
        current_end_text = str(segments[end_line].get("text") or "")
        if force_end_clause_completion and classify_terminator(current_end_text):
            force_end_clause_completion = False
        next_text = (
            str(segments[end_line + 1].get("text") or "")
            if end_line + 1 < len(segments)
            else ""
        )
        if next_text and _FORWARD_TOPIC_TRANSITION_RE.match(next_text):
            break
        if not force_end_clause_completion and not _cue_has_weak_end(
            current_end_text,
            next_text,
            ignore_caption_case=ignore_caption_case,
        ):
            break
        candidate = end_line + 1
        if candidate >= len(segments):
            break
        if max_end_line is not None and candidate > max_end_line:
            break
        if crosses_section_reset(end_line, candidate):
            break
        end_line = candidate
        end_expansions += 1
    next_text = "" if suffix_was_trimmed else (
        str(segments[end_line + 1].get("text") or "")
        if end_line + 1 < len(segments)
        else ""
    )
    if next_text and _FORWARD_TOPIC_TRANSITION_RE.match(next_text):
        next_text = ""
    final_end_text = (
        _cue_clip_text(segments, original_end, end_line)
        if end_line > original_end
        else str(segments[end_line].get("text") or "")
    )
    if (
        not end_boundary_verified
        and _cue_has_weak_end(
            final_end_text,
            next_text,
            ignore_caption_case=ignore_caption_case,
        )
    ) or force_end_clause_completion:
        if _terminal_content_is_explicitly_incomplete(final_end_text):
            return start_line, end_line, "unresolved_weak_end"
        return start_line, end_line, "unresolved_boundary_end"
    return (
        start_line,
        end_line,
        "unresolved_weak_start" if unresolved_start else None,
    )


def _padded_cue_bounds(
    segments: list[dict], start_line: int, end_line: int,
) -> tuple[float, float]:
    """Add 300 ms room without crossing the midpoint to adjacent speech."""
    start = float(segments[start_line].get("start", 0.0))
    end = float(segments[end_line].get("end", start))
    if start_line > 0:
        previous_end = float(segments[start_line - 1].get("end", start))
        if previous_end <= start:
            start = max(start - _BOUNDARY_PAD_S, (previous_end + start) / 2.0)
    else:
        start = max(0.0, start - _BOUNDARY_PAD_S)
    if end_line + 1 < len(segments):
        next_start = float(segments[end_line + 1].get("start", end))
        if next_start >= end:
            end = min(end + _BOUNDARY_PAD_S, (end + next_start) / 2.0)
    else:
        end = min(end + _BOUNDARY_PAD_S, float(segments[-1].get("end", end)))
    return start, end


def _cue_clip_text(segments: list[dict], start_line: int, end_line: int) -> str:
    return " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments[start_line:end_line + 1]
        if str(segment.get("text") or "").strip()
    ).strip()


_FILLER_REMAINDER_WORDS = frozenset({
    "a", "an", "and", "back", "for", "from", "have", "hope", "i", "ll", "no",
    "our", "that", "the", "thanks", "thank", "to", "today", "we", "we'll", "will", "you",
    "your",
})


def _structural_filler_matches(text: str) -> list[re.Match[str]]:
    return list(_STRUCTURAL_FILLER_RE.finditer(str(text or "")))


def _structural_match_is_edge(
    text: str,
    match: re.Match[str],
    *,
    want: str,
) -> bool:
    before = text[:match.start()]
    after = text[match.end():]
    is_non_speech_marker = bool(
        _NON_SPEECH_MARKER_RE.fullmatch(match.group(0))
    )
    if want == "start":
        # A caption marker is removable only when it actually precedes speech.
        # This keeps a brief marker between teaching clauses inside the clip.
        clean_prefix = (
            not _WORD_RE.search(before)
            if is_non_speech_marker
            else (
                not re.search(r"[.!?]", before)
                and len(_toks(before)) <= 5
            )
        )
        return bool(clean_prefix and _WORD_RE.search(after))
    return bool(_WORD_RE.search(before) and not _WORD_RE.search(after))


def _literal_structural_filler_only(text: str) -> bool:
    if _EDGE_ONLY_FRAMING_RE.fullmatch(str(text or "")):
        return True
    matches = _structural_filler_matches(text)
    if not matches:
        return False
    remainder = list(str(text or ""))
    for match in matches:
        remainder[match.start():match.end()] = " " * (match.end() - match.start())
    return set(_toks("".join(remainder))).issubset(_FILLER_REMAINDER_WORDS)


def _course_admin_sentence(text: str) -> bool:
    """Recognize high-confidence course operations, not subject-matter teaching."""
    raw_text = str(text or "").strip()
    if not _WORD_RE.search(raw_text):
        return False
    return bool(
        _SPEAKER_OR_STARTUP_FILLER_RE.fullmatch(raw_text)
        or _COURSE_ADMIN_STRONG_RE.search(raw_text)
        or _COURSE_ASSESSMENT_ADMIN_RE.search(raw_text)
        or _COURSE_AUDIENCE_LOGISTICS_RE.search(raw_text)
        or _AUDIENCE_COMPOSITION_RE.search(raw_text)
        or _COURSE_ADMIN_CONVEY_RE.search(raw_text)
        or _COURSE_PREVIEW_RE.search(raw_text)
    )


@lru_cache(maxsize=4_096)
def _sentence_character_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return abbreviation-aware sentence spans without rewriting source text."""
    raw_text = str(text or "")
    from .sentences import segment_sentences

    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence, _terminator in segment_sentences(raw_text):
        left = raw_text.find(sentence, cursor)
        if left < 0:
            spans = []
            break
        right = left + len(sentence)
        spans.append((left, right))
        cursor = right
    if spans:
        return tuple(spans)
    return tuple(
        (match.start(), match.end())
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", raw_text)
        if _WORD_RE.search(match.group(0))
    )


def _cue_is_only_structural_filler(text: str) -> bool:
    raw_text = str(text or "")
    if (
        _literal_structural_filler_only(raw_text)
        or _NEXT_EXAMPLE_FRAMING_RE.fullmatch(raw_text)
    ):
        return True
    sentences = [
        raw_text[left:right].strip()
        for left, right in _sentence_character_spans(raw_text)
        if _WORD_RE.search(raw_text[left:right])
    ]
    if not sentences:
        return False
    classifications = [
        _literal_structural_filler_only(sentence)
        or _course_admin_sentence(sentence)
        for sentence in sentences
    ]
    if all(classifications):
        return True
    admin_count = sum(_course_admin_sentence(sentence) for sentence in sentences)
    return admin_count >= 2 and all(
        classified or _COURSE_ADMIN_BRIDGE_RE.fullmatch(sentence)
        for sentence, classified in zip(sentences, classifications)
    )


def _quote_character_spans(text: str, quote: str) -> list[tuple[int, int]]:
    text_matches = list(_WORD_RE.finditer(str(text or "")))
    quote_tokens = _toks(quote)
    if not quote_tokens or len(quote_tokens) > len(text_matches):
        return []
    text_tokens = [
        unicodedata.normalize("NFKC", match.group(0))
        .translate(_APOSTROPHES)
        .casefold()
        for match in text_matches
    ]
    spans: list[tuple[int, int]] = []
    for index in range(len(text_tokens) - len(quote_tokens) + 1):
        if text_tokens[index:index + len(quote_tokens)] == quote_tokens:
            spans.append((
                text_matches[index].start(),
                text_matches[index + len(quote_tokens) - 1].end(),
            ))
    return spans


def _quote_character_span(text: str, quote: str) -> tuple[int, int] | None:
    spans = _quote_character_spans(text, quote)
    return spans[0] if spans else None


def _cross_cue_token_windows(
    segments: list[dict],
    quote: str,
    start_line: int,
    end_line: int,
) -> list[tuple[int, int, int, int, int, int]]:
    """Locate exact full-token quote windows that cross contiguous cues."""
    quote_tokens = _toks(quote)
    if not quote_tokens or start_line < 0 or end_line >= len(segments):
        return []

    flattened: list[tuple[str, int, int, int]] = []
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        for match in _WORD_RE.finditer(text):
            [token] = _toks(match.group(0))
            flattened.append((token, line, match.start(), match.end()))
    if len(flattened) < len(quote_tokens):
        return []

    matches: list[tuple[int, int, int, int, int, int]] = []
    for index in range(len(flattened) - len(quote_tokens) + 1):
        window = flattened[index:index + len(quote_tokens)]
        if [item[0] for item in window] != quote_tokens:
            continue
        first_line = window[0][1]
        last_line = window[-1][1]
        if first_line == last_line:
            continue
        crossed_lines = sorted({item[1] for item in window})
        crosses_reset = False
        for left, right in zip(crossed_lines, crossed_lines[1:]):
            if right != left + 1:
                crosses_reset = True
                break
            try:
                gap = float(segments[right].get("start")) - float(
                    segments[left].get("end")
                )
            except (TypeError, ValueError, OverflowError):
                crosses_reset = True
                break
            if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
                crosses_reset = True
                break
        if crosses_reset:
            continue

        first_tokens = [item for item in window if item[1] == first_line]
        last_tokens = [item for item in window if item[1] == last_line]
        matches.append((
            first_line,
            last_line,
            first_tokens[0][2],
            first_tokens[-1][3],
            last_tokens[0][2],
            last_tokens[-1][3],
        ))
    return matches


def _cross_cue_quote_matches(
    segments: list[dict],
    quote: str,
    start_line: int,
    end_line: int,
) -> list[tuple[int, int, str, str]]:
    """Return unique literal edge projections for exact adjacent-cue quotes."""
    matches: list[tuple[int, int, str, str]] = []
    for (
        first_line,
        last_line,
        first_left,
        first_right,
        last_left,
        last_right,
    ) in _cross_cue_token_windows(segments, quote, start_line, end_line):
        first_text = str(segments[first_line].get("text") or "")
        last_text = str(segments[last_line].get("text") or "")
        start_quote = first_text[first_left:first_right]
        end_quote = last_text[last_left:last_right]
        if (
            len(_quote_character_spans(first_text, start_quote)) != 1
            or len(_quote_character_spans(last_text, end_quote)) != 1
        ):
            continue
        matches.append((first_line, last_line, start_quote, end_quote))
    return matches


def _unique_evidence_location(
    segments: list[dict],
    quote: str,
    start_line: int,
    end_line: int,
) -> tuple[int, int, int, int] | None:
    """Ground one evidence quote either within one cue or across adjacent cues."""
    locations: list[tuple[int, int, int, int]] = []
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        locations.extend(
            (line, left, line, right)
            for left, right in _quote_character_spans(text, quote)
        )
    locations.extend(
        (first_line, first_left, last_line, last_right)
        for (
            first_line,
            last_line,
            first_left,
            _first_right,
            _last_left,
            last_right,
        ) in _cross_cue_token_windows(segments, quote, start_line, end_line)
    )
    return locations[0] if len(locations) == 1 else None


def _proposal_evidence_anchor(
    proposal: object,
    intent_constraints: dict[str, _IntentConstraint],
    segments: list[dict],
    start_line: int,
    end_line: int,
) -> tuple[str, tuple[int, int, int, int] | None]:
    """Choose the first uniquely grounded selector quote for unit trimming."""
    direct_quote = " ".join(
        str(getattr(proposal, "topic_evidence_quote", "") or "").split()
    )
    candidates: list[str] = [direct_quote] if direct_quote else []
    if isinstance(proposal, _CompactBoundaryTopic) and not candidates:
        proposed = list(proposal.intent_evidence)
        specificity = {
            _IntentConstraintKind.RELATIONSHIP: 0,
            _IntentConstraintKind.TASK: 1,
            _IntentConstraintKind.OUTCOME: 1,
            _IntentConstraintKind.FORMAT: 2,
            _IntentConstraintKind.SCOPE: 3,
            _IntentConstraintKind.SUBJECT: 4,
        }
        ordered_constraint_ids = sorted(
            intent_constraints,
            key=lambda constraint_id: (
                specificity.get(intent_constraints[constraint_id].kind, 5),
                list(intent_constraints).index(constraint_id),
            ),
        )
        ordered = [
            evidence
            for constraint_id in ordered_constraint_ids
            for evidence in proposed
            if evidence.constraint_id == constraint_id
        ]
        ordered.extend(item for item in proposed if item not in ordered)
        candidates.extend(
            " ".join(str(item.evidence_quote or "").split())
            for item in ordered
        )
    candidates = list(dict.fromkeys(quote for quote in candidates if quote))
    for quote in candidates:
        location = _unique_evidence_location(
            segments,
            quote,
            start_line,
            end_line,
        )
        if location is not None:
            return quote, location
    return (candidates[0], None) if candidates else ("", None)


def _literal_source_quote(
    text: str,
    quote: str,
    span: tuple[int, int],
) -> str:
    """Return the matched source spelling, including exact requested punctuation."""
    source = str(text or "")
    quote_matches = list(_WORD_RE.finditer(str(quote or "")))
    start, end = span
    if quote_matches:
        prefix = str(quote or "")[:quote_matches[0].start()]
        suffix = str(quote or "")[quote_matches[-1].end():]
        if prefix and start >= len(prefix) and source[start - len(prefix):start] == prefix:
            start -= len(prefix)
        if suffix and source[end:end + len(suffix)] == suffix:
            end += len(suffix)
    return source[start:end]


def _semantic_edge_quote(
    text: str,
    quote: str,
    *,
    want: str,
) -> tuple[tuple[int, int] | None, bool, str | None]:
    """Ground one semantic edge without inventing a timestamp.

    The edge occurrence nearest the physical cue edge is authoritative when the quote
    already begins/ends the cue. A quote that excludes real words requires projection and
    therefore must have exactly one normalized occurrence in that cue.
    """
    word_matches = list(_WORD_RE.finditer(str(text or "")))
    spans = _quote_character_spans(text, quote)
    if not word_matches or not spans:
        return None, False, "ungrounded_boundary_quote"
    span = spans[0] if want == "start" else spans[-1]
    quote_matches = list(_WORD_RE.finditer(str(quote or "")))
    if quote_matches:
        left, right = span
        prefix = str(quote or "")[:quote_matches[0].start()]
        suffix = str(quote or "")[quote_matches[-1].end():]
        if prefix and left >= len(prefix) and text[left - len(prefix):left] == prefix:
            left -= len(prefix)
        if suffix and text[right:right + len(suffix)] == suffix:
            right += len(suffix)
        span = (left, right)
    projected = bool(
        _WORD_RE.search(text[:span[0]])
        if want == "start"
        else _WORD_RE.search(text[span[1]:])
    )
    if projected and len(spans) != 1:
        return None, True, f"ambiguous_{want}_quote"
    return span, projected, None


def _expanded_context_edge_quote(
    text: str,
    *,
    want: str,
) -> tuple[str, str | None]:
    """Choose an exact edge quote after safely removable filler-only sentences."""
    raw_text = str(text or "")
    if want == "start":
        example_replacement = _leading_example_framing_quote(raw_text)
        if example_replacement:
            return example_replacement, None
    sentence_spans = _sentence_character_spans(raw_text)
    if not sentence_spans:
        return "", "empty_expanded_context_edge"

    retained_left = 0
    retained_right = len(raw_text)
    full_text_edge_matches = [
        match
        for match in _structural_filler_matches(raw_text)
        if _structural_match_is_edge(raw_text, match, want=want)
    ]
    ordered_spans = sentence_spans if want == "start" else list(reversed(sentence_spans))
    for left, right in ordered_spans:
        sentence = raw_text[left:right]
        matches = _structural_filler_matches(sentence)
        absolute_edge_match = any(
            match.start() < right and match.end() > left
            for match in full_text_edge_matches
        )
        if _cue_is_only_structural_filler(sentence):
            if want == "start":
                retained_left = right
            else:
                retained_right = left
            continue
        if absolute_edge_match and not matches:
            if want == "start" and right < len(raw_text):
                retained_left = right
                continue
            if want == "end" and left > 0:
                retained_right = left
                continue
        if not matches:
            break

        edge_matches = [
            match
            for match in matches
            if _structural_match_is_edge(sentence, match, want=want)
        ]
        if not edge_matches and absolute_edge_match:
            if want == "start" and right < len(raw_text):
                retained_left = right
                continue
            if want == "end" and left > 0:
                retained_right = left
                continue
        if edge_matches:
            inline_boundary_applied = False
            if want == "start":
                match = max(edge_matches, key=lambda value: value.end())
                separator_pattern = (
                    r"(?:\s+|\s*[,;:—-]\s*)"
                    if _NON_SPEECH_MARKER_RE.fullmatch(match.group(0))
                    else r"\s*[,;:—-]\s*"
                )
                separator = re.match(separator_pattern, sentence[match.end():])
                if separator is not None:
                    retained_left = left + match.end() + separator.end()
                    inline_boundary_applied = True
            else:
                match = min(edge_matches, key=lambda value: value.start())
                separator_pattern = (
                    r"(?:\s+|[,;:—-]\s*)$"
                    if _NON_SPEECH_MARKER_RE.fullmatch(match.group(0))
                    else r"[,;:—-]\s*$"
                )
                separator = re.search(separator_pattern, sentence[:match.start()])
                if separator is not None:
                    retained_right = left + separator.start()
                    inline_boundary_applied = True
            if inline_boundary_applied:
                break
            if want == "start" and right < len(raw_text):
                retained_left = right
                continue
            if want == "end" and left > 0:
                retained_right = left
                continue
            return "", "unresolved_expanded_edge_filler"
        break

    if want == "start":
        preview = _INSTRUCTIONAL_PREVIEW_PREFIX_RE.match(raw_text[retained_left:])
        if preview is not None:
            retained_suffix = raw_text[
                retained_left + preview.end():retained_right
            ]
            if (
                _WORD_RE.search(retained_suffix)
                and not _ANAPHORIC_OPENING_RE.match(retained_suffix)
            ):
                retained_left += preview.end()

    retained_words = [
        match
        for match in _WORD_RE.finditer(raw_text)
        if retained_left <= match.start() and match.end() <= retained_right
    ]
    if not retained_words:
        return "", "empty_expanded_context_edge"
    chosen = retained_words[:6] if want == "start" else retained_words[-6:]
    quote_end = chosen[-1].end()
    if want == "end":
        retained_suffix_end = retained_right
        while (
            retained_suffix_end > quote_end
            and raw_text[retained_suffix_end - 1].isspace()
        ):
            retained_suffix_end -= 1
        suffix = raw_text[quote_end:retained_suffix_end]
        if suffix and not _WORD_RE.search(suffix):
            quote_end = retained_suffix_end
    return raw_text[chosen[0].start():quote_end], None


def _edge_has_unresolved_structural_filler(
    text: str,
    quote_span: tuple[int, int],
    *,
    want: str,
) -> bool:
    """Reject structural filler that the semantic quote still leaves on an edge."""
    for match in _structural_filler_matches(text):
        is_non_speech_marker = bool(
            _NON_SPEECH_MARKER_RE.fullmatch(match.group(0))
        )
        if (
            want == "start"
            and is_non_speech_marker
            and quote_span[0] < match.end()
        ):
            selected_prefix = (
                text[quote_span[0]:match.start()]
                if quote_span[0] < match.start()
                else ""
            )
            if (
                not _WORD_RE.search(selected_prefix)
                or _cue_is_only_structural_filler(selected_prefix)
            ):
                return True
        is_edge_match = _structural_match_is_edge(text, match, want=want)
        if want == "start" and is_edge_match and quote_span[0] < match.end():
            return True
        if want == "end" and is_edge_match and quote_span[1] > match.start():
            return True
    return False


def _replace_structural_edge_quote(
    text: str,
    quote: str,
    *,
    want: str,
) -> tuple[str, bool, str | None]:
    """Replace a model quote that includes removable filler at a cue edge."""
    if _cue_is_only_structural_filler(text):
        # The existing cue-level pass can remove this cue without projecting a
        # boundary inside it. Same-cue repair is only for mixed teaching/filler.
        return quote, False, None
    quote_span, _projected, error = _semantic_edge_quote(text, quote, want=want)
    if error or quote_span is None:
        # Preserve the normal grounding/ambiguity failure emitted downstream.
        return quote, False, None

    if want == "start":
        selected_tail = text[quote_span[0]:]
        example_replacement = _leading_example_framing_quote(selected_tail)
        if example_replacement:
            replacement_span, projected, replacement_error = _semantic_edge_quote(
                text,
                example_replacement,
                want="start",
            )
            if (
                replacement_error is None
                and replacement_span is not None
                and projected
            ):
                return example_replacement, True, None
        removable_prefixes = (
            _OPENING_TOPIC_ANNOUNCEMENT_PREFIX_RE.match(selected_tail),
            _OPENING_DEPENDENT_LEADIN_RE.match(selected_tail),
            _LEADING_DISCOURSE_MARKER_RE.match(selected_tail),
        )
        for prefix in removable_prefixes:
            if prefix is None:
                continue
            retained = selected_tail[prefix.end():].strip()
            if (
                len(_toks(retained)) < 4
                or _ANAPHORIC_OPENING_RE.match(retained)
                or _NON_STANDALONE_MARKER_SUFFIX_RE.match(retained)
            ):
                continue
            replacement = _exact_boundary_quote(retained, want="start")
            replacement_span, projected, replacement_error = _semantic_edge_quote(
                text,
                replacement,
                want="start",
            )
            if (
                replacement_error is None
                and replacement_span is not None
                and projected
            ):
                return replacement, True, None

    if not _edge_has_unresolved_structural_filler(text, quote_span, want=want):
        return quote, False, None

    replacement, error = _expanded_context_edge_quote(text, want=want)
    if error:
        return "", False, "unresolved_edge_filler"
    replacement_span, projected, error = _semantic_edge_quote(
        text, replacement, want=want
    )
    if error or replacement_span is None or not projected:
        return "", False, error or "unresolved_edge_filler"
    if _edge_has_unresolved_structural_filler(
        text, replacement_span, want=want
    ):
        return "", False, "unresolved_edge_filler"
    return replacement, True, None


def _opening_clause_is_standalone(text: str) -> bool:
    selected = str(text or "").strip()
    from .discourse import CONTINUATION_MARKERS

    sentence_spans = _sentence_character_spans(selected)
    opening_clause = (
        selected[:sentence_spans[0][1]].strip() if sentence_spans else selected
    )
    opening_words = _toks(opening_clause)
    finite_clause_signal = re.search(
        r"\b(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
        r"shall|should|was|were|will|would|means?|allows?|enables?|lets?|powers?)\b",
        opening_clause,
        re.IGNORECASE,
    )
    possessive_reference = re.search(
        r"\b(?:its|their)\s+[a-z][a-z'’-]*\b",
        opening_clause,
        re.IGNORECASE,
    )
    if possessive_reference is not None:
        explicit_antecedent = re.search(
            r"(?:^|[,;:]\s*)(?:(?:for|from|in|on|within)\s+)?"
            r"(?:the|a|an)\s+[a-z][a-z'’-]*\b",
            opening_clause[:possessive_reference.start()],
            re.IGNORECASE,
        )
        if explicit_antecedent is None:
            return False
    if (
        _OPENING_EDGE_META_SENTENCE_RE.match(opening_clause)
        or _OPENING_CONTEXTUAL_REFORMULATION_RE.match(opening_clause)
        or _opening_contextual_example_needs_context(opening_clause)
        or _opening_has_context_dependent_subject(opening_clause)
        or _OPENING_AGENDA_RE.match(opening_clause)
        or _NEXT_EXAMPLE_FRAMING_RE.fullmatch(opening_clause)
        or _OPENING_LIST_TAIL_RE.match(opening_clause)
        or _OPENING_PREPOSITIONAL_TAG_RE.match(opening_clause)
        or _OPENING_DEPENDENT_PREPOSITION_FRAGMENT_RE.match(opening_clause)
        or _opening_is_dependent_question_tail(opening_clause)
        or (
            _OPENING_NOMINAL_INFINITIVE_RE.match(opening_clause)
            and finite_clause_signal is None
        )
    ):
        return False
    if (
        re.match(r"^\s*i\s+mentioned\b", opening_clause, re.IGNORECASE)
        and len(_content_tokens(opening_clause)) >= 4
    ):
        return True
    if (
        (opening_words and opening_words[0] in CONTINUATION_MARKERS)
        or _DANGLING_TAIL_PREFIX_RE.search(opening_clause)
        or _OPENING_DEMONSTRATIVE_REFERENCE_RE.match(opening_clause)
        or _OPENING_BARE_RELATIONAL_PREDICATE_RE.match(opening_clause)
        or _ANAPHORIC_OPENING_RE.match(opening_clause)
        or (
            _NON_STANDALONE_MARKER_SUFFIX_RE.match(opening_clause)
            and not _PREPOSITION_FRONTED_QUESTION_RE.match(opening_clause)
        )
    ):
        return False
    if _OPENING_COMPLETE_ORDINAL_SUBJECT_RE.match(opening_clause):
        return True
    from .discourse import _has_unresolved_opening_back_reference

    if _has_unresolved_opening_back_reference(opening_clause):
        return False
    if re.match(
        r"^\s*(?:here|there)\s+(?:is|are|was|were)\b",
        opening_clause,
        re.IGNORECASE,
    ):
        return True
    return not _cue_opens_mid_thought(opening_clause, ignore_caption_case=True)


def _leading_example_framing_quote(text: str) -> str:
    """Skip a sentence-only example label when teaching starts in the same cue."""
    raw_text = str(text or "")
    sentence_spans = _sentence_character_spans(raw_text)
    if len(sentence_spans) < 2:
        return ""
    first_left, first_right = sentence_spans[0]
    if not _OPENING_EXAMPLE_FRAMING_SENTENCE_RE.fullmatch(
        raw_text[first_left:first_right]
    ):
        return ""
    retained = raw_text[first_right:].lstrip()
    if len(_toks(retained)) < 4:
        return ""
    complete_setup = bool(
        _local_example_setup_is_complete(retained)
        and len(_toks(retained)) >= 5
    )
    if not (_opening_clause_is_standalone(retained) or complete_setup):
        return ""
    return _exact_boundary_quote(retained, want="start")


def _projected_start_is_standalone(
    text: str,
    quote_span: tuple[int, int],
) -> bool:
    """Trust a grounded start only across a real discourse boundary."""
    if quote_span[0] < 0:
        return False
    selected = text[quote_span[0]:].strip()
    if not _opening_clause_is_standalone(selected):
        return False
    if quote_span[0] == 0:
        return True
    omitted_prefix = text[:quote_span[0]].rstrip()
    if not _WORD_RE.search(omitted_prefix):
        return True
    from .sentences import classify_terminator

    return bool(
        classify_terminator(omitted_prefix)
        or _cue_is_only_structural_filler(omitted_prefix)
    )


def _opening_agenda_prefix_is_only_preview(
    segments: list[dict],
    start_line: int,
    teaching_line: int,
) -> bool:
    """Allow forward agenda trimming only across preview fragments, never teaching."""
    if teaching_line <= start_line:
        return False
    start_text = str(segments[start_line].get("text") or "").strip()
    agenda = _OPENING_AGENDA_RE.match(start_text)
    if agenda is None:
        return False
    if _LOCAL_EXPLICIT_PROBLEM_RE.search(start_text[agenda.end():]):
        return False
    prefix = _cue_clip_text(segments, start_line, teaching_line - 1)
    for line in range(start_line + 1, teaching_line):
        text = str(segments[line].get("text") or "").strip()
        if (
            _cue_is_only_structural_filler(text)
            or _OPENING_AGENDA_CONTINUATION_RE.match(text)
            or _NEXT_EXAMPLE_FRAMING_RE.fullmatch(text)
        ):
            continue
        return False
    return True


def _recover_agenda_setup_within_cue(
    text: str,
    *,
    evidence_quote: str,
    learning_objective: str,
    following_text: str = "",
) -> str:
    """Recover an explicit setup after an agenda even when captions lack punctuation."""
    raw_text = str(text or "")
    agenda = _OPENING_AGENDA_RE.match(raw_text)
    if agenda is None:
        return ""
    setup = _LOCAL_EXPLICIT_PROBLEM_RE.search(raw_text, agenda.end())
    if setup is None:
        return ""
    left = setup.start()
    article = re.search(r"\b(?:a|an|the)\s+$", raw_text[:left], re.IGNORECASE)
    if article is not None and article.end() == left:
        left = article.start()
    retained = raw_text[left:].strip()
    if not _local_example_setup_is_complete(retained):
        return ""
    grounded_text = " ".join(
        part for part in (retained, following_text) if part
    )
    if evidence_quote and not _contains_quote(grounded_text, evidence_quote):
        return ""
    if (
        not evidence_quote
        and len(
            _content_tokens(grounded_text)
            & _content_tokens(learning_objective)
        ) < 2
    ):
        return ""
    return _exact_boundary_quote(retained, want="start")


def _recover_start_forward_across_cues(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    evidence_quote: str,
    learning_objective: str,
) -> tuple[int, str] | None:
    """Find the first later cold-viewer onset that still contains the grounded unit."""
    from .sentences import classify_terminator

    anchor_tokens = _content_tokens(f"{evidence_quote} {learning_objective}")
    agenda_teaching_line = next(
        (
            line
            for line in range(start_line + 1, end_line + 1)
            if _opening_agenda_prefix_is_only_preview(
                segments,
                start_line,
                line,
            )
            and not _OPENING_AGENDA_CONTINUATION_RE.match(
                str(segments[line].get("text") or "").strip()
            )
        ),
        None,
    )
    for line in range(start_line + 1, end_line + 1):
        if agenda_teaching_line is not None:
            if line < agenda_teaching_line:
                continue
            if line > agenda_teaching_line:
                break
        previous = str(segments[line - 1].get("text") or "").strip()
        current = str(segments[line].get("text") or "").strip()
        gap = (
            float(segments[line].get("start", 0.0))
            - float(segments[line - 1].get("end", 0.0))
        )
        ordinary_boundary = bool(
            classify_terminator(previous)
            or _cue_is_only_structural_filler(previous)
            or gap >= _SECTION_RESET_GAP_S
        )
        agenda_boundary = bool(
            agenda_teaching_line is not None
            and line == agenda_teaching_line
        )
        if not ordinary_boundary and not agenda_boundary:
            continue
        if _cue_is_only_structural_filler(current):
            continue
        retained = _cue_clip_text(segments, line, end_line)
        evidence_retained = bool(
            evidence_quote and _contains_quote(retained, evidence_quote)
        )
        if evidence_quote and not evidence_retained:
            continue
        if not evidence_quote and len(_content_tokens(retained) & anchor_tokens) < 2:
            continue
        opening_candidates = [current]
        marker = _LEADING_DISCOURSE_MARKER_RE.match(current)
        if marker is not None:
            without_marker = current[marker.end():].strip()
            if without_marker:
                opening_candidates.insert(0, without_marker)
        comparative = _OPENING_COMPARATIVE_LEADIN_RE.match(current)
        if comparative is not None:
            suffix = current[comparative.end():].strip()
            if suffix:
                opening_candidates.insert(0, suffix)
        for opening in opening_candidates:
            opening_retained = " ".join(
                part
                for part in (
                    opening,
                    _cue_clip_text(segments, line + 1, end_line)
                    if line < end_line
                    else "",
                )
                if part
            )
            if evidence_quote and not _contains_quote(
                opening_retained,
                evidence_quote,
            ):
                continue
            setup_is_complete = bool(
                _local_example_setup_is_complete(opening)
                and len(_toks(opening)) >= 4
                and len(_toks(retained)) >= 8
            )
            if not (
                _opening_clause_is_standalone(opening) or setup_is_complete
            ):
                continue
            quote = _exact_boundary_quote(opening, want="start")
            if quote:
                return line, quote
    return None


def _recover_start_after_edge_navigation(
    text: str,
    *,
    evidence_quote: str,
    learning_objective: str,
    following_text: str = "",
) -> str:
    """Skip a complete navigation/admin sentence before grounded teaching."""
    raw_text = str(text or "")
    anchor_tokens = _content_tokens(f"{evidence_quote} {learning_objective}")
    sentence_spans = _sentence_character_spans(raw_text)
    for noise in reversed(list(_TRAILING_EDGE_NOISE_RE.finditer(raw_text))):
        noise_start = noise.start("noise")
        noise_end = noise.end("noise")
        back_to = re.match(
            r"^\s*back\s+to\s+(?:the\s+)?(?P<subject>[^.!?,;:]{1,100})",
            noise.group("noise"),
            re.IGNORECASE,
        )
        if (
            back_to is not None
            and _SAME_UNIT_RESET_SUBJECT_RE.match(back_to.group("subject"))
        ):
            continue
        containing_sentence = next(
            (
                (left, right)
                for left, right in sentence_spans
                if left <= noise_start < right and noise_end <= right
            ),
            (noise_start, noise_end),
        )
        sentence_left, sentence_right = containing_sentence
        local_prefix_tokens = set(_toks(raw_text[sentence_left:noise_start]))
        local_prefix_tokens.difference_update({
            "all", "alright", "but", "now", "okay", "ok", "right", "so",
        })
        if local_prefix_tokens and not re.match(
            r"^\s*back\s+to\b",
            noise.group("noise"),
            re.IGNORECASE,
        ):
            continue
        retained = raw_text[sentence_right:].lstrip(" ,;:—-")
        if not retained or not _opening_clause_is_standalone(retained):
            continue
        if (
            len(_content_tokens(raw_text[:noise_start]) & anchor_tokens) >= 2
            or _objective_bridges_sections(
                learning_objective,
                raw_text[:noise_start],
                retained,
            )
        ):
            # The navigation is internal to an already relevant teaching arc;
            # tolerate it instead of deleting the substantive opening.
            return ""
        evidence_retained = bool(
            evidence_quote
            and _contains_quote(
                " ".join(part for part in (retained, following_text) if part),
                evidence_quote,
            )
        )
        if evidence_quote and not evidence_retained:
            continue
        if not evidence_quote and len(_content_tokens(retained) & anchor_tokens) < 2:
            continue
        return _exact_boundary_quote(retained, want="start")
    return ""


def _trailing_edge_noise_start(
    text: str,
    *,
    anchor_text: str = "",
) -> int | None:
    """Locate navigation/admin/outro speech that is removable only at an ending."""
    raw_text = str(text or "")
    for match in _TRAILING_EDGE_NOISE_RE.finditer(raw_text):
        start = match.start("noise")
        connector = re.search(
            r"\b(?:and|but|so)\s*$",
            raw_text[:start],
            re.IGNORECASE,
        )
        cut_start = connector.start() if connector is not None else start
        containing_sentence = next(
            (
                (left, right)
                for left, right in _sentence_character_spans(raw_text)
                if left <= start < right
            ),
            None,
        )
        same_unit_navigation = bool(
            containing_sentence is not None
            and _SAME_UNIT_NAVIGATION_RE.search(
                raw_text[containing_sentence[0]:containing_sentence[1]]
            )
        )
        if containing_sentence is not None and not same_unit_navigation:
            navigation_tail = raw_text[
                match.end("noise"):containing_sentence[1]
            ]
            if re.match(r"^\s*(?:how|why)\b", navigation_tail, re.IGNORECASE):
                same_unit_navigation = bool(
                    _content_tokens(navigation_tail)
                    & _content_tokens(anchor_text)
                )
        if (
            containing_sentence is not None
            and _WORD_RE.search(raw_text[containing_sentence[1]:])
            and same_unit_navigation
        ):
            # Explicit same-unit navigation followed by more speech introduces
            # a required next step, example, proof, or conclusion.
            continue
        prefix = raw_text[:cut_start].rstrip(" ,;:—-")
        if len(_toks(prefix)) < 3:
            continue
        if not (
            _terminal_content_is_explicitly_incomplete(prefix)
            or _cue_has_explicit_dangling_end(prefix, "")
            or _TERMINAL_DANGLING_PREDICATE_HEAD_RE.search(prefix)
        ):
            return cut_start

        # If a caption merged an unfinished teaser into the navigation, keep
        # the last independently complete sentence rather than manufacturing
        # an ending such as "the important point is".
        from .sentences import classify_terminator

        for _left, right in reversed(_sentence_character_spans(prefix)):
            complete_prefix = prefix[:right].strip()
            if (
                classify_terminator(complete_prefix)
                and not _terminal_content_is_explicitly_incomplete(complete_prefix)
                and not _TERMINAL_DANGLING_PREDICATE_HEAD_RE.search(complete_prefix)
            ):
                return right
    visual_pointer = _TRAILING_VISUAL_POINTER_RE.search(raw_text)
    if visual_pointer is not None:
        prefix = raw_text[:visual_pointer.start()].rstrip(" ,;:—-")
        embedded_identity = bool(_TERMINAL_EMBEDDED_IDENTITY_RE.search(prefix))
        if (
            len(_toks(prefix)) >= 5
            and (
                embedded_identity
                or (
                    not _terminal_content_is_explicitly_incomplete(prefix)
                    and not _cue_has_explicit_dangling_end(prefix, "")
                    and not _TERMINAL_DANGLING_PREDICATE_HEAD_RE.search(prefix)
                    and not _TERMINAL_REQUIRED_VISUAL_COMPLEMENT_RE.search(prefix)
                )
            )
        ):
            return visual_pointer.start()
    return None


def _unconditional_trailing_edge_noise_start(
    text: str,
    *,
    require_edge_prefix: bool = False,
) -> int | None:
    """Locate assignment/future-unit speech that cannot complete teaching."""
    raw_text = str(text or "")
    for match in _UNCONDITIONAL_TRAILING_EDGE_NOISE_RE.finditer(raw_text):
        start = match.start("noise")
        connector = re.search(
            r"\b(?:and|but|so)\s*$",
            raw_text[:start],
            re.IGNORECASE,
        )
        cut_start = connector.start() if connector is not None else start
        if require_edge_prefix and _WORD_RE.search(raw_text[:cut_start]):
            continue
        return cut_start
    return None


def _trailing_version_edge_noise_start(
    text: str,
    *,
    require_edge_prefix: bool = False,
) -> int | None:
    """Locate a formula-version transition that may belong to another facet."""
    raw_text = str(text or "")
    for match in _TRAILING_VERSION_EDGE_NOISE_RE.finditer(raw_text):
        start = match.start("noise")
        connector = re.search(
            r"\b(?:and|but|so)\s*$",
            raw_text[:start],
            re.IGNORECASE,
        )
        cut_start = connector.start() if connector is not None else start
        if require_edge_prefix and _WORD_RE.search(raw_text[:cut_start]):
            continue
        return cut_start
    return None


def _last_safe_complete_prefix(text: str) -> str:
    """Keep a whole claim before navigation, never a dangling predicate."""
    from .sentences import classify_terminator

    raw_text = str(text or "").rstrip(" ,;:—-")

    def safe(value: str) -> bool:
        return bool(
            len(_toks(value)) >= 3
            and not _terminal_content_is_explicitly_incomplete(value)
            and not _cue_has_explicit_dangling_end(value, "")
            and not _TERMINAL_DANGLING_PREDICATE_HEAD_RE.search(value)
        )

    if safe(raw_text):
        return raw_text
    for _left, right in reversed(_sentence_character_spans(raw_text)):
        prefix = raw_text[:right].strip()
        if classify_terminator(prefix) and safe(prefix):
            return prefix
    return ""


def _trim_end_quote_before_edge_noise(
    text: str,
    quote: str,
    *,
    evidence_quote: str = "",
    learning_objective: str = "",
) -> tuple[str, bool]:
    """Shorten a grounded end quote before an inline next-topic or outro tail."""
    spans = _quote_character_spans(text, quote)
    if not spans:
        return quote, False
    # End projection is authoritative at the final grounded occurrence. This
    # matters for intentionally short one-word quotes such as "energy".
    span = spans[-1]
    selected = str(text or "")[:span[1]]
    noise_start = _trailing_edge_noise_start(
        selected,
        anchor_text=f"{evidence_quote} {learning_objective}",
    )
    if noise_start is None:
        return quote, False
    evidence_spans = _quote_character_spans(selected, evidence_quote)
    if any(evidence_span[0] >= noise_start for evidence_span in evidence_spans):
        return quote, False
    retained = selected[:noise_start].rstrip(" ,;:—-")
    unconditional_noise_start = _unconditional_trailing_edge_noise_start(selected)
    if unconditional_noise_start != noise_start and _objective_bridges_sections(
        learning_objective,
        retained,
        selected[noise_start:],
    ):
        return quote, False
    replacement = _exact_boundary_quote(retained, want="end")
    return (replacement, True) if replacement else (quote, False)


def _projected_end_is_complete(
    text: str,
    quote_span: tuple[int, int],
    *,
    following_text: str,
) -> bool:
    """Accept a projected stop only when adjacent speech proves a clean boundary."""
    from .sentences import classify_terminator

    selected = str(text or "")[:quote_span[1]].rstrip()
    if _terminal_content_is_explicitly_incomplete(selected):
        return False
    if classify_terminator(selected):
        return True
    omitted_suffix = str(text or "")[quote_span[1]:]
    if _trailing_edge_noise_start(f"{selected} {omitted_suffix}") is not None:
        return True
    next_text = str(following_text or "").strip()
    return bool(
        next_text
        and (
            _FORWARD_TOPIC_TRANSITION_RE.match(next_text)
            or _cue_is_only_structural_filler(next_text)
        )
    )


def _complete_split_caption_tail(
    segments: list[dict],
    end_line: int,
    end_quote: str,
    *,
    proposals: list[object],
    proposal_index: int,
    ignore_caption_case: bool,
    anchor_text: str = "",
) -> tuple[int, str] | None:
    """Project a short answer suffix that precedes the next teaching unit.

    Transcript providers may split a formula or concluding phrase at an
    arbitrary character boundary.  Import only the immediate next-cue prefix,
    and only when a grounded new-unit onset gives that prefix a safe stop.
    """
    next_line = end_line + 1
    if next_line >= len(segments):
        return None
    current_text = str(segments[end_line].get("text") or "")
    next_text = str(segments[next_line].get("text") or "")
    if not current_text.strip() or not next_text.strip():
        return None
    try:
        current_end = float(segments[end_line].get("end"))
        next_start = float(segments[next_line].get("start"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(current_end) or not math.isfinite(next_start):
        return None
    if next_start - current_end >= _SECTION_RESET_GAP_S:
        return None

    current_span, current_projected, current_error = _semantic_edge_quote(
        current_text,
        end_quote,
        want="end",
    )
    if current_error or current_span is None or current_projected:
        return None
    selected_current = current_text[:current_span[1]].rstrip()
    selected_tail = " ".join(selected_current.split()[-32:])
    has_explicit_closure = bool(
        _SPLIT_CAPTION_COMPLETION_SIGNAL_RE.search(selected_tail)
    )
    has_weak_end = _cue_has_weak_end(
        selected_current,
        next_text,
        ignore_caption_case=ignore_caption_case,
    )
    if not has_explicit_closure and not has_weak_end:
        return None

    onset_candidates: list[int] = []
    noise_start = _trailing_edge_noise_start(
        next_text,
        anchor_text=anchor_text,
    )
    if noise_start is not None:
        onset_candidates.append(noise_start)
    onset_candidates.extend(
        match.start() for match in _HARD_TOPIC_RESET_RE.finditer(next_text)
    )
    onset_candidates.extend(
        match.start()
        for match in _SPLIT_CAPTION_NEW_UNIT_FRAMING_RE.finditer(next_text)
    )

    for other_index, proposal in enumerate(proposals):
        if other_index == proposal_index:
            continue
        if getattr(proposal, "start_line", None) != next_line:
            continue
        proposal_quote = str(getattr(proposal, "start_quote", "") or "").strip()
        proposal_spans = _quote_character_spans(next_text, proposal_quote)
        if len(proposal_spans) != 1:
            continue
        onset = proposal_spans[0][0]
        marker = _SPLIT_CAPTION_ONSET_MARKER_RE.search(next_text[:onset])
        onset_candidates.append(marker.start() if marker is not None else onset)

    # Bare questions and imperatives are safe only immediately after a short
    # continuation prefix.  This avoids treating reasoning such as "we find"
    # inside a worked solution as the start of another unit.
    if has_explicit_closure:
        for word in _WORD_RE.finditer(next_text):
            onset = word.start()
            if len(_toks(next_text[:onset])) > 3:
                break
            if _PEDAGOGICAL_SETUP_ONSET_RE.match(next_text[onset:]):
                onset_candidates.append(onset)
                break

    max_prefix_words = 12 if has_explicit_closure else 5
    for onset in sorted(set(onset_candidates)):
        if onset <= 0 or not _WORD_RE.search(next_text[onset:]):
            continue
        prefix = next_text[:onset].rstrip(" ,;:—-")
        prefix_matches = list(_WORD_RE.finditer(prefix))
        if not 1 <= len(prefix_matches) <= max_prefix_words:
            continue
        if _cue_is_only_structural_filler(prefix):
            continue
        completed_text = f"{selected_current} {prefix}"
        if (
            _terminal_content_is_explicitly_incomplete(completed_text)
            or _TERMINAL_DANGLING_PREDICATE_HEAD_RE.search(completed_text)
            or _cue_has_explicit_dangling_end(
                completed_text,
                next_text[onset:],
            )
        ):
            continue

        # Start with the normal six-word edge quote and lengthen only when the
        # normalized occurrence would be ambiguous inside this coarse cue.
        initial_width = min(6, len(prefix_matches))
        for width in range(initial_width, min(12, len(prefix_matches)) + 1):
            quote = prefix[
                prefix_matches[-width].start():prefix_matches[-1].end()
            ]
            spans = _quote_character_spans(next_text, quote)
            if len(spans) != 1:
                continue
            span = spans[0]
            if _WORD_RE.search(next_text[span[1]:]):
                return next_line, quote
    return None


def _recover_projected_start_within_cue(
    text: str,
    quote_span: tuple[int, int],
    *,
    evidence_quote: str,
    learning_objective: str,
    following_text: str = "",
) -> str:
    """Move a fragmentary model start to a complete sentence in the same cue."""
    agenda_setup_quote = _recover_agenda_setup_within_cue(
        text,
        evidence_quote=evidence_quote,
        learning_objective=learning_objective,
        following_text=following_text,
    )
    if agenda_setup_quote:
        return agenda_setup_quote
    sentence_spans = _sentence_character_spans(text)
    if not sentence_spans:
        return ""

    def quote_from_sentence(left: int) -> str:
        retained = text[left:].lstrip()
        marker = _LEADING_DISCOURSE_MARKER_RE.match(retained)
        if marker is not None:
            without_marker = retained[marker.end():].strip()
            if _opening_clause_is_standalone(without_marker):
                retained = without_marker
        local_setup = _local_example_setup_is_complete(retained)
        if not (_opening_clause_is_standalone(retained) or local_setup):
            return ""
        return _exact_boundary_quote(retained, want="start")

    current_index = next(
        (
            index
            for index, (left, right) in enumerate(sentence_spans)
            if left <= quote_span[0] < right
        ),
        None,
    )
    if current_index is None:
        return ""
    anchor_tokens = _content_tokens(
        f"{evidence_quote} {learning_objective}"
    )
    selected_suffix = text[quote_span[0]:].lstrip()
    selected_with_following = " ".join(
        part for part in (selected_suffix, following_text) if part
    )
    suffix_is_grounded = bool(
        (
            evidence_quote
            and _contains_quote(selected_with_following, evidence_quote)
        )
        or (
            not evidence_quote
            and len(_content_tokens(selected_with_following) & anchor_tokens) >= 2
        )
    )
    omitted_prefix = text[:quote_span[0]].strip()
    projected_suffix_is_complete = bool(
        _opening_clause_is_standalone(selected_suffix)
        or (
            omitted_prefix
            and _cue_is_only_structural_filler(omitted_prefix)
            and _general_local_setup_is_complete(selected_with_following)
        )
    )
    if suffix_is_grounded and projected_suffix_is_complete:
        return _exact_boundary_quote(selected_suffix, want="start")
    for index in range(current_index + 1, len(sentence_spans)):
        left, _right = sentence_spans[index]
        retained = text[left:]
        evidence_retained = bool(
            evidence_quote
            and _contains_quote(
                " ".join(part for part in (retained, following_text) if part),
                evidence_quote,
            )
        )
        if evidence_quote and not evidence_retained:
            continue
        if not evidence_quote and len(_content_tokens(retained) & anchor_tokens) < 2:
            continue
        replacement = quote_from_sentence(left)
        if replacement:
            return replacement
    for index in range(current_index, -1, -1):
        left = sentence_spans[index][0]
        retained = text[left:]
        if evidence_quote and not _contains_quote(
            " ".join(part for part in (retained, following_text) if part),
            evidence_quote,
        ):
            continue
        replacement = quote_from_sentence(left)
        if replacement:
            return replacement
    return ""


def _semantic_clip_slice(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    start_span: tuple[int, int] | None,
    end_span: tuple[int, int] | None,
) -> tuple[str, dict[str, tuple[int, int]]]:
    """Return transcript speech between projected edge quotes, including internal asides."""
    parts: list[str] = []
    spans_by_cue: dict[str, tuple[int, int]] = {}
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        left = start_span[0] if line == start_line and start_span is not None else 0
        right = end_span[1] if line == end_line and end_span is not None else len(text)
        if right <= left:
            return "", {}
        cue_id = str(segments[line].get("cue_id") or f"cue-{line}")
        spans_by_cue[cue_id] = (left, right)
        selected = text[left:right].strip()
        if selected:
            parts.append(selected)
    return " ".join(parts).strip(), spans_by_cue


def _grounded_evidence_span_with_minimum_words(
    text: str,
    quote: str,
    *,
    minimum_words: int = 5,
) -> tuple[str, tuple[int, int]] | None:
    """Ground a short model quote and extend it locally instead of losing the clip."""
    span = _quote_character_span(text, quote)
    if span is None:
        marker = _LEADING_DISCOURSE_MARKER_RE.match(str(quote or ""))
        trimmed_quote = (
            str(quote or "")[marker.end():].strip() if marker is not None else ""
        )
        if len(_toks(trimmed_quote)) >= minimum_words:
            span = _quote_character_span(text, trimmed_quote)
            if span is not None:
                quote = trimmed_quote
    if span is None:
        return None
    words = list(_WORD_RE.finditer(str(text or "")))
    selected = [
        index
        for index, word in enumerate(words)
        if word.start() < span[1] and word.end() > span[0]
    ]
    if not selected:
        return None
    left, right = selected[0], selected[-1]
    while right - left + 1 < minimum_words:
        if right + 1 < len(words):
            right += 1
        elif left > 0:
            left -= 1
        else:
            break
    grounded_span = (words[left].start(), words[right].end())
    return str(text or "")[grounded_span[0]:grounded_span[1]], grounded_span


def _trim_structural_filler_edges(
    segments: list[dict], start_line: int, end_line: int,
    *,
    ignore_caption_case: bool,
) -> tuple[int, int] | None:
    """Trim contiguous edge filler while keeping the remaining teaching span.

    Discourse expansion and semantic edge projection refine a weak opening or
    ending later. Edge uncertainty alone must not turn an otherwise useful
    educational candidate into a filler rejection.
    """
    filler_lines = {
        line
        for line in range(start_line, end_line + 1)
        if _cue_is_only_structural_filler(str(segments[line].get("text") or ""))
    }
    if not filler_lines:
        return start_line, end_line

    trimmed_start = start_line
    while trimmed_start <= end_line and trimmed_start in filler_lines:
        trimmed_start += 1
    trimmed_end = end_line
    while trimmed_end >= trimmed_start and trimmed_end in filler_lines:
        trimmed_end -= 1
    if trimmed_start > trimmed_end:
        return None
    return trimmed_start, trimmed_end


def _plain_same_unit_navigation_subject(subject: str) -> bool:
    """Recognize a unit label only while its referent remains unresolved."""
    match = _SAME_UNIT_RESET_SUBJECT_RE.match(str(subject or ""))
    if match is None:
        return False
    tail = str(subject or "")[match.end():]
    named_complement = re.match(r"^\s+of\s+(?P<name>[^.!?,;:]+)", tail, re.IGNORECASE)
    if named_complement is None:
        return True
    return bool(
        re.match(
            r"^(?:this|that|the\s+same)\b",
            named_complement.group("name"),
            re.IGNORECASE,
        )
    )


def _objective_bridges_sections(
    learning_objective: str,
    left_text: str,
    right_text: str,
    *,
    reset_subject: str = "",
) -> bool:
    """Require objective anchors on both sides before preserving a reset."""
    generic = {
        "complete", "concept", "describe", "discuss", "example", "explain",
        "idea", "learn", "lesson", "show", "teach", "understand", "work",
    }
    objective_tokens = _content_tokens(learning_objective) - generic
    if not objective_tokens:
        return False

    def objective_overlap(text: str) -> set[str]:
        side_tokens = _content_tokens(text)
        return {
            objective_token
            for objective_token in objective_tokens
            if any(
                objective_token == side_token
                or (
                    min(len(objective_token), len(side_token)) >= 5
                    and (
                        objective_token.startswith(side_token)
                        or side_token.startswith(objective_token)
                        or objective_token == f"sub{side_token}"
                        or side_token == f"sub{objective_token}"
                    )
                )
                for side_token in side_tokens
            )
        }

    left_overlap = objective_overlap(left_text)
    right_overlap = objective_overlap(right_text)
    if not left_overlap or not right_overlap:
        return False
    subject_tokens = _content_tokens(reset_subject) - {
        "about", "cover", "discuss", "how", "look", "next", "talk", "why",
    }
    subject_is_anchored = bool(objective_overlap(reset_subject))
    if subject_tokens and not subject_is_anchored:
        # A broad relational objective can share one word with an adjacent
        # lesson without actually naming that lesson. The explicit reset
        # subject must itself belong to the objective before the reset is kept.
        return False
    normalized_objective = " ".join(str(learning_objective or "").split())
    if _EXPLICIT_COMPARISON_OBJECTIVE_RE.search(normalized_objective):
        # Comparisons may use one anchor per side, but a shared head noun such
        # as "cost" cannot by itself connect opportunity cost to cost accounting.
        return len(left_overlap | right_overlap) >= 2
    if _objective_explicitly_relates_sections(learning_objective):
        if _RELATIONAL_RESET_SUBJECT_RE.match(str(reset_subject or "")):
            # "why/how it is X" explicitly continues the claim on the old
            # side; its anaphoric subject cannot supply two lexical anchors.
            return True
        if len(right_overlap) >= 2:
            # The new side explicitly restates enough of the relationship to
            # connect back to even a one-concept setup.
            return True
        return False
    # A method change within one objective needs independent multi-token
    # evidence on both sides; one shared word cannot merge adjacent lessons.
    return len(left_overlap) >= 2 and len(right_overlap) >= 2


@dataclass(frozen=True)
class _TopicTransition:
    navigation_line: int
    navigation_left: int
    new_side_line: int
    new_side_left: int
    worked_unit: bool = False


def _worked_unit_onsets_in_cue(
    text: str,
    *,
    evidence_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Locate high-confidence new problem/question starts in one coarse cue."""
    raw_text = str(text or "")
    if _WORKED_UNIT_POSSIBLE_ONSET_RE.search(raw_text) is None:
        return []
    onsets: dict[int, int] = {}

    def add(navigation_left: int, new_side_left: int) -> None:
        fragment = raw_text[new_side_left:]
        if (
            _WORKED_UNIT_DISCOURSE_CONTINUATION_RE.match(fragment)
            or _WORKED_UNIT_ANAPHORIC_CONTINUATION_RE.match(fragment)
            or _WORKED_UNIT_NONQUESTION_WH_CONTINUATION_RE.match(fragment)
            or _WORKED_UNIT_PROCEDURAL_STEP_RE.match(fragment)
            or _WORKED_UNIT_PROCEDURAL_QUESTION_RE.match(fragment)
        ):
            return
        previous = onsets.get(navigation_left)
        if previous is None or new_side_left > previous:
            onsets[navigation_left] = new_side_left

    for marked in _MARKED_WORKED_UNIT_ONSET_RE.finditer(raw_text):
        add(marked.start("navigation"), marked.start("unit"))

    for framing in _SPLIT_CAPTION_NEW_UNIT_FRAMING_RE.finditer(raw_text):
        following_action = _WORKED_UNIT_ACTION_TOKEN_RE.search(
            raw_text,
            framing.end(),
        )
        new_side = (
            following_action.start()
            if following_action is not None
            and len(_toks(raw_text[framing.end():following_action.start()])) <= 12
            else framing.start()
        )
        add(framing.start(), new_side)

    sentence_spans = _sentence_character_spans(raw_text)
    for sentence_left, sentence_right in sentence_spans:
        first_word = _WORD_RE.search(raw_text, sentence_left, sentence_right)
        if first_word is None:
            continue
        onset = first_word.start()
        if (
            _WORKED_UNIT_ACTION_ONSET_RE.match(raw_text[onset:sentence_right])
            or _WORKED_UNIT_WH_ONSET_RE.match(raw_text[onset:sentence_right])
        ):
            marker = _SPLIT_CAPTION_ONSET_MARKER_RE.search(raw_text[:onset])
            navigation_left = marker.start() if marker is not None else onset
            add(navigation_left, onset)

    # Caption providers often omit punctuation inside a long cue. The exact
    # grounded evidence still identifies its prompt without guessing from an
    # unrelated reasoning verb elsewhere in the solution.
    for action in _WORKED_UNIT_ACTION_TOKEN_RE.finditer(raw_text):
        action_left = action.start()
        previous_words = list(_WORD_RE.finditer(raw_text[:action_left]))
        previous_word = (
            previous_words[-1].group(0).casefold()
            if previous_words
            else ""
        )
        if previous_word in {
            "and", "by", "can", "could", "first", "i", "must", "should",
            "then", "to", "we", "will", "would", "you",
        }:
            continue
        evidence_anchored = any(
            action_left <= evidence_left
            and evidence_left - action_left <= 160
            and evidence_right - action_left <= 240
            for evidence_left, evidence_right in evidence_spans
        )
        prefix = raw_text[:action_left]
        sentence_left = max(
            raw_text.rfind(marker, 0, action_left)
            for marker in ".!?"
        ) + 1
        local_prefix = raw_text[sentence_left:action_left]
        if not (
            evidence_anchored
            or len(_toks(local_prefix)) <= 12
            or _SPLIT_CAPTION_COMPLETION_SIGNAL_RE.search(local_prefix)
        ):
            continue
        marker = _SPLIT_CAPTION_ONSET_MARKER_RE.search(raw_text[:action_left])
        navigation_left = marker.start() if marker is not None else action_left
        add(navigation_left, action_left)

    for question in _WORKED_UNIT_QUESTION_TOKEN_RE.finditer(raw_text):
        question_left = question.start()
        prefix = raw_text[:question_left]
        sentence_left = max(
            raw_text.rfind(marker, 0, question_left)
            for marker in ".!?"
        ) + 1
        local_prefix = raw_text[sentence_left:question_left]
        previous_words = list(_WORD_RE.finditer(prefix))
        previous_word = (
            previous_words[-1].group(0).casefold()
            if previous_words
            else ""
        )
        if previous_word in {
            "about", "and", "are", "as", "by", "demonstrates", "explains",
            "explaining", "for", "from", "illustrates", "is", "know",
            "knowing", "learn", "learning", "notice", "noticing", "observe",
            "observing", "of", "on", "reason", "recall", "recalling",
            "remember", "remembering", "see", "seeing", "showing", "shows",
            "that", "the", "to", "understand", "understanding", "was", "were",
            "with", "without",
        }:
            continue
        evidence_anchored = any(
            question_left <= evidence_left
            and evidence_left - question_left <= 160
            and evidence_right - question_left <= 240
            for evidence_left, evidence_right in evidence_spans
        )
        if not (
            len(_toks(local_prefix)) <= 3
            or _SPLIT_CAPTION_COMPLETION_SIGNAL_RE.search(local_prefix)
            or evidence_anchored
        ):
            continue
        marker = _SPLIT_CAPTION_ONSET_MARKER_RE.search(prefix)
        navigation_left = marker.start() if marker is not None else question_left
        add(navigation_left, question_left)

    return sorted(onsets.items())


def _worked_unit_prefix_is_complete(
    segments: list[dict],
    start_line: int,
    transition: _TopicTransition,
    *,
    after: _TopicTransition | None = None,
) -> bool:
    """Distinguish prior solved work from useful rule/setup context."""
    first_line = after.new_side_line if after is not None else start_line
    parts: list[str] = []
    for line in range(first_line, transition.navigation_line + 1):
        text = str(segments[line].get("text") or "")
        left = after.new_side_left if after is not None and line == first_line else 0
        right = (
            transition.navigation_left
            if line == transition.navigation_line
            else len(text)
        )
        if right > left:
            parts.append(text[left:right])
    prefix = " ".join(parts)
    return bool(
        _SPLIT_CAPTION_COMPLETION_SIGNAL_RE.search(prefix)
        or re.search(
            r"\b(?:final\s+answer|fully\s+simplified|"
            r"(?:answer|result|solution)\s+(?:is|equals?)|"
            r"that(?:['’]s|\s+is)\s+(?:all\s+we\s+need\s+to\s+do\s+for\s+"
            r"(?:this|the)\s+(?:calculation|case|derivation|example|exercise|"
            r"problem|proof)|(?:the\s+)?(?:answer|result|solution))|"
            r"(?:complete|finish(?:ed|es)?|solv(?:e|ed|es))\s+(?:"
            r"(?:(?:this|the)\s+)?(?:answer|calculation|case|derivation|"
            r"example|exercise|problem|proof|solution)))\b",
            prefix,
            re.IGNORECASE,
        )
    )


def _worked_unit_target_needs_prior_explanation(
    segments: list[dict],
    target: _TopicTransition,
    raw_transitions: list[_TopicTransition],
    end_line: int,
) -> bool:
    """Detect a target solution that explicitly reuses an unnamed prior method."""
    following = min(
        (
            item
            for item in raw_transitions
            if (item.navigation_line, item.navigation_left)
            > (target.navigation_line, target.navigation_left)
        ),
        default=None,
        key=lambda item: (item.navigation_line, item.navigation_left),
    )
    last_line = following.navigation_line if following is not None else end_line
    parts: list[str] = []
    for line in range(target.new_side_line, last_line + 1):
        text = str(segments[line].get("text") or "")
        left = target.new_side_left if line == target.new_side_line else 0
        right = (
            following.navigation_left
            if following is not None and line == following.navigation_line
            else len(text)
        )
        if right > left:
            parts.append(text[left:right])
    target_text = " ".join(parts)
    generic_modifiers = {
        "a", "an", "our", "that", "the", "these", "this", "those", "your",
    }
    for reference in _WORKED_UNIT_UNRESOLVED_METHOD_REFERENCE_RE.finditer(
        target_text
    ):
        head = reference.group("head").casefold()
        prefix_text = target_text[:reference.start()]
        prefix_tokens = _toks(prefix_text)
        locally_defined = re.search(
            rf"\b(?:the|this|that)\s+{re.escape(head)}\s+"
            r"(?:are|equals?|gives?|is|means?|states?|was|were)\b",
            prefix_text,
            re.IGNORECASE,
        ) is not None
        locally_named = any(
            token == head
            and index > 0
            and prefix_tokens[index - 1] not in generic_modifiers
            for index, token in enumerate(prefix_tokens)
        )
        if not locally_defined and not locally_named:
            return True
    return False


def _which_onset_is_interrogative(
    fragment: str,
    learning_objective: str,
) -> bool:
    """Separate a ``Which + noun`` prompt from a dangling relative clause."""
    raw_fragment = str(fragment or "").lstrip()
    if re.match(r"^[^.!?]{1,200}\?", raw_fragment):
        return True
    if re.match(
        r"^\s*(?:choose|determine|distinguish|identify|name|select)\b",
        learning_objective,
        re.IGNORECASE,
    ):
        return True
    words = _toks(raw_fragment)
    if len(words) < 2 or words[0] != "which":
        return False
    second = words[1]
    if second in {"of", "one", "ones"}:
        return True
    if len(words) >= 3 and second.endswith("s") and words[2] in {
        "act", "actually", "affect", "apply", "are", "best", "can", "cause",
        "commonly", "contribute", "control", "could", "determine", "did",
        "directly", "do", "does", "drive", "encode", "explain", "have",
        "influence", "is", "lead", "least", "mainly", "may", "might", "most",
        "of", "often", "primarily", "produce", "regulate", "result", "should",
        "support", "typically", "was", "were", "will", "would",
    }:
        return True
    finite_predicates = {
        "also", "am", "are", "became", "can", "could", "did", "do", "does",
        "had", "has", "have", "is", "led", "made", "may", "might", "must",
        "shall", "should", "then", "thereby", "was", "were", "will", "would",
        "wrote",
    }
    return bool(
        second not in finite_predicates
        and not second.endswith(("ed", "ing", "s"))
    )


def _worked_unit_transitions(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    evidence_location: tuple[int, int, int, int] | None,
    learning_objective: str,
) -> list[_TopicTransition]:
    """Return distinct worked-unit boundaries around one grounded objective."""
    if evidence_location is None:
        return []
    evidence_start_line, evidence_left, evidence_end_line, evidence_right = (
        evidence_location
    )
    normalized_objective = " ".join(str(learning_objective or "").split())
    transitions: list[_TopicTransition] = []
    for line in range(start_line, end_line + 1):
        cue_text = str(segments[line].get("text") or "")
        evidence_spans: list[tuple[int, int]] = []
        if line == evidence_start_line:
            evidence_spans.append((
                evidence_left,
                evidence_right if evidence_end_line == line else len(cue_text),
            ))
        for navigation_left, new_side_left in _worked_unit_onsets_in_cue(
            cue_text,
            evidence_spans=evidence_spans,
        ):
            fragment = cue_text[new_side_left:]
            wh_onset = _WORKED_UNIT_WH_ONSET_RE.match(fragment)
            if (
                wh_onset is not None
                and new_side_left == 0
                and line > start_line
                and _WORKED_UNIT_WH_PREVIOUS_LICENSE_RE.search(
                    str(segments[line - 1].get("text") or "")
                )
            ):
                continue
            if (
                re.match(r"^\s*which\b", fragment, re.IGNORECASE)
                and (new_side_left > 0 or line > start_line)
                and not _which_onset_is_interrogative(
                    fragment,
                    normalized_objective,
                )
            ):
                # A sentence-initial relative ``Which ...`` needs its prior
                # antecedent. Keep it unless punctuation or the selector's
                # objective establishes a real interrogative target.
                continue
            transitions.append(_TopicTransition(
                navigation_line=line,
                navigation_left=navigation_left,
                new_side_line=line,
                new_side_left=new_side_left,
                worked_unit=True,
            ))

    transitions.sort(
        key=lambda item: (item.navigation_line, item.navigation_left)
    )
    raw_transitions = transitions
    target_onset = max(
        (
            item
            for item in raw_transitions
            if (item.new_side_line, item.new_side_left)
            <= (evidence_start_line, evidence_left)
        ),
        default=None,
        key=lambda item: (item.new_side_line, item.new_side_left),
    )
    transitions = []
    current_unit: _TopicTransition | None = None
    for transition in raw_transitions:
        cue_text = str(segments[transition.navigation_line].get("text") or "")
        navigation_text = cue_text[transition.navigation_left:]
        explicitly_marked = bool(
            _SPLIT_CAPTION_NEW_UNIT_FRAMING_RE.match(navigation_text)
        )
        prior_unit_is_complete = bool(
            current_unit is not None
            and _worked_unit_prefix_is_complete(
                segments,
                start_line,
                transition,
                after=current_unit,
            )
        )
        prefix_is_complete = _worked_unit_prefix_is_complete(
            segments,
            start_line,
            transition,
        )
        begins_at_or_before_evidence = bool(
            (transition.new_side_line, transition.new_side_left)
            <= (evidence_start_line, evidence_left)
        )
        follows_evidence_with_target_prompt = bool(
            (transition.navigation_line, transition.navigation_left)
            > (evidence_end_line, evidence_right)
            and _WORKED_UNIT_TARGET_PROMPT_RE.match(
                cue_text[transition.new_side_left:]
            )
        )
        if current_unit is None:
            accept = bool(
                begins_at_or_before_evidence
                or explicitly_marked
                or prefix_is_complete
            )
        else:
            accept = bool(
                transition == target_onset
                or explicitly_marked
                or prior_unit_is_complete
                or follows_evidence_with_target_prompt
            )
        if accept:
            transitions.append(transition)
            current_unit = transition

    if _EXPLICIT_COMPARISON_OBJECTIVE_RE.search(normalized_objective):
        # The candidate's grounded relational evidence identifies the compared
        # arc. Preserve its earlier prompts, but retain the first later prompt
        # as a hard end so a third unrelated exercise cannot leak in.
        return [
            item
            for item in transitions
            if (item.navigation_line, item.navigation_left)
            > (evidence_end_line, evidence_right)
        ]

    if (
        target_onset is not None
        and target_onset in transitions
        and _worked_unit_target_needs_prior_explanation(
            segments,
            target_onset,
            raw_transitions,
            end_line,
        )
    ):
        transitions.remove(target_onset)

    prior_or_target = [
        item
        for item in transitions
        if (item.navigation_line, item.navigation_left)
        <= (evidence_start_line, evidence_left)
    ]
    if (
        len(prior_or_target) == 1
        and not _worked_unit_prefix_is_complete(
            segments,
            start_line,
            prior_or_target[0],
        )
    ):
        # Keep a rule, definition, or prerequisite explanation attached to the
        # first worked problem. Once another problem or a completed answer is
        # present, the evidence-anchored problem becomes its own unit.
        transitions.remove(prior_or_target[0])
    return transitions


_DESCRIBED_UNIT_SUBJECT_RE = re.compile(
    r"^\s*(?:another|one\s+more|the\s+next|a\s+(?:different|new))\s+"
    r"(?:(?:brief|concrete|quick|short|simple|worked)\s+)*"
    r"(?:calculation|case|demonstration|derivation|example|exercise|problem|proof)"
    r"\s*$",
    re.IGNORECASE,
)


def _clause_after_described_unit(
    subject: str,
    suffix: str,
    *,
    suffix_start: int,
) -> int | None:
    """Skip a generic ``another example:`` label before explicit teaching."""
    if _DESCRIBED_UNIT_SUBJECT_RE.fullmatch(str(subject or "")) is None:
        return None
    clause = re.match(
        r"\s*[,;:—-]\s*(?P<text>[a-z0-9])",
        str(suffix or ""),
        re.IGNORECASE,
    )
    return (
        suffix_start + clause.start("text")
        if clause is not None
        else None
    )


def _candidate_topic_transitions(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    evidence_quote: str,
    learning_objective: str,
) -> list[_TopicTransition]:
    """Return unbridged same-cue and adjacent-cue topic navigation."""
    cue_texts = [
        str(segments[line].get("text") or "")
        for line in range(start_line, end_line + 1)
    ]
    dotted_text = " . ".join(cue_texts)
    joined_text = " ".join(cue_texts)
    has_hard_reset = bool(
        _HARD_TOPIC_RESET_RE.search(dotted_text)
        or _HARD_TOPIC_RESET_RE.search(joined_text)
    )
    has_worked_unit_onset = bool(
        _WORKED_UNIT_POSSIBLE_ONSET_RE.search(dotted_text)
        or _WORKED_UNIT_POSSIBLE_ONSET_RE.search(joined_text)
    )
    if not has_hard_reset and not has_worked_unit_onset:
        return []
    evidence_location = _unique_evidence_location(
        segments,
        evidence_quote,
        start_line,
        end_line,
    )
    evidence_locations = {
        line: _quote_character_spans(
            str(segments[line].get("text") or ""), evidence_quote
        )
        for line in range(start_line, end_line + 1)
    }
    transitions: list[_TopicTransition] = []
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        sentence_spans = _sentence_character_spans(text)
        for reset in _HARD_TOPIC_RESET_RE.finditer(text):
            subject = reset.group("subject")
            if _plain_same_unit_navigation_subject(subject):
                continue

            sentence_left, sentence_right = next(
                (
                    (left, right)
                    for left, right in sentence_spans
                    if left <= reset.start() < right
                ),
                (reset.start(), len(text)),
            )
            navigation_left = reset.start()
            connector = re.search(
                r"\b(?:and|but|so)\s*$",
                text[:navigation_left],
                re.IGNORECASE,
            )
            if connector is not None:
                navigation_left = connector.start()
            subject_left = reset.start("subject")
            left_parts = [
                str(segments[index].get("text") or "")
                for index in range(start_line, line)
            ]
            left_parts.append(text[:navigation_left])
            right_parts = [text[subject_left:]]
            right_parts.extend(
                str(segments[index].get("text") or "")
                for index in range(line + 1, end_line + 1)
            )
            if _objective_bridges_sections(
                learning_objective,
                " ".join(left_parts),
                " ".join(right_parts),
                reset_subject=subject,
            ):
                continue

            subject_tail = text[reset.end("subject"):sentence_right]
            evidence_in_reset_body = any(
                span[0] >= subject_left and span[0] < sentence_right
                for span in evidence_locations[line]
            )
            described_clause = _clause_after_described_unit(
                subject,
                subject_tail,
                suffix_start=reset.end("subject"),
            )
            new_side_start = (
                described_clause
                if described_clause is not None
                else (
                    subject_left
                    if _WORD_RE.search(subject_tail) or evidence_in_reset_body
                    else sentence_right
                )
            )
            transitions.append(_TopicTransition(
                navigation_line=line,
                navigation_left=navigation_left,
                new_side_line=line,
                new_side_left=new_side_start,
            ))

    for line in range(start_line, end_line):
        left_text = str(segments[line].get("text") or "")
        right_text = str(segments[line + 1].get("text") or "")
        joined = f"{left_text} {right_text}"
        split = len(left_text) + 1
        sentence_spans = _sentence_character_spans(joined)
        for reset in _HARD_TOPIC_RESET_RE.finditer(joined):
            if not (reset.start() < split < reset.end("subject")):
                continue
            subject = reset.group("subject")
            if _plain_same_unit_navigation_subject(subject):
                continue
            navigation_left = reset.start()
            connector = re.search(
                r"\b(?:and|but|so)\s*$",
                joined[:navigation_left],
                re.IGNORECASE,
            )
            if connector is not None:
                navigation_left = connector.start()
            subject_left = reset.start("subject")
            left_parts = [
                str(segments[index].get("text") or "")
                for index in range(start_line, line)
            ]
            left_parts.append(joined[:navigation_left])
            right_parts = [joined[subject_left:]]
            right_parts.extend(
                str(segments[index].get("text") or "")
                for index in range(line + 2, end_line + 1)
            )
            if _objective_bridges_sections(
                learning_objective,
                " ".join(left_parts),
                " ".join(right_parts),
                reset_subject=subject,
            ):
                continue
            _sentence_left, sentence_right = next(
                (
                    (left, right)
                    for left, right in sentence_spans
                    if left <= reset.start() < right
                ),
                (reset.start(), len(joined)),
            )
            evidence_in_reset_body = any(
                span[0] >= subject_left and span[0] < sentence_right
                for span in _quote_character_spans(joined, evidence_quote)
            )
            subject_tail = joined[reset.end("subject"):sentence_right]
            described_clause = _clause_after_described_unit(
                subject,
                subject_tail,
                suffix_start=reset.end("subject"),
            )
            new_side_global = (
                described_clause
                if described_clause is not None
                else (
                    subject_left
                    if _WORD_RE.search(subject_tail) or evidence_in_reset_body
                    else sentence_right
                )
            )
            new_side_line, new_side_left = (
                (line, new_side_global)
                if new_side_global < split
                else (line + 1, new_side_global - split)
            )
            transitions.append(_TopicTransition(
                navigation_line=line,
                navigation_left=navigation_left,
                new_side_line=new_side_line,
                new_side_left=new_side_left,
            ))
    if has_worked_unit_onset:
        transitions.extend(_worked_unit_transitions(
            segments,
            start_line,
            end_line,
            evidence_location=evidence_location,
            learning_objective=learning_objective,
        ))

    # A phrase such as "now let's work on more examples" may satisfy both the
    # navigation and worked-unit detectors. Keep one boundary and prefer the
    # later grounded teaching onset over its generic framing words.
    unique: dict[tuple[int, int], _TopicTransition] = {}
    for transition in transitions:
        key = (transition.navigation_line, transition.navigation_left)
        previous = unique.get(key)
        if previous is None or (
            transition.new_side_line,
            transition.new_side_left,
        ) > (previous.new_side_line, previous.new_side_left):
            unique[key] = transition
    return sorted(
        unique.values(),
        key=lambda item: (item.navigation_line, item.navigation_left),
    )


def _contains_bridged_topic_navigation(
    text: str,
    *,
    learning_objective: str,
) -> bool:
    """Keep a same-cue setup only when the objective anchors both reset sides."""
    raw_text = str(text or "")
    for reset in _HARD_TOPIC_RESET_RE.finditer(raw_text):
        navigation_left = reset.start()
        connector = re.search(
            r"\b(?:and|but|so)\s*$",
            raw_text[:navigation_left],
            re.IGNORECASE,
        )
        if connector is not None:
            navigation_left = connector.start()
        if _objective_bridges_sections(
            learning_objective,
            raw_text[:navigation_left],
            raw_text[reset.start("subject"):],
            reset_subject=reset.group("subject"),
        ):
            return True
    return False


def _objective_explicitly_relates_sections(learning_objective: str) -> bool:
    """Recognize an expressed relationship, not isolated keyword collisions."""
    return bool(
        _EXPLICIT_RELATIONAL_OBJECTIVE_RE.search(
            " ".join(str(learning_objective or "").split())
        )
    )


def _single_objective_section_bounds(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    evidence_location: tuple[int, int, int, int] | None,
    transitions: list[_TopicTransition],
) -> tuple[int, int]:
    """Keep the explicit topic section containing the candidate's grounded evidence."""
    if evidence_location is None:
        return start_line, end_line
    evidence_start_line, evidence_left, evidence_end_line, evidence_right = (
        evidence_location
    )

    prior = [
        item
        for item in transitions
        if (item.navigation_line, item.navigation_left)
        <= (evidence_start_line, evidence_left)
    ]
    following = [
        item
        for item in transitions
        if (item.navigation_line, item.navigation_left)
        > (evidence_end_line, evidence_right)
    ]

    selected_start = max(
        prior,
        default=None,
        key=lambda item: (item.navigation_line, item.navigation_left),
    )
    selected_end = min(
        following,
        default=None,
        key=lambda item: (item.navigation_line, item.navigation_left),
    )
    bounded_start = start_line
    if selected_start is not None:
        suffix = str(
            segments[selected_start.new_side_line].get("text") or ""
        )[
            selected_start.new_side_left:
        ]
        bounded_start = (
            selected_start.new_side_line
            if _WORD_RE.search(suffix)
            else selected_start.new_side_line + 1
        )
    bounded_end = end_line
    if selected_end is not None:
        prefix = str(
            segments[selected_end.navigation_line].get("text") or ""
        )[
            :selected_end.navigation_left
        ]
        bounded_end = (
            selected_end.navigation_line
            if _WORD_RE.search(prefix)
            else selected_end.navigation_line - 1
        )
    if bounded_start > bounded_end:
        return start_line, end_line
    return bounded_start, bounded_end


def _evidence_crosses_topic_transition(
    *,
    evidence_location: tuple[int, int, int, int] | None,
    transitions: list[_TopicTransition],
) -> bool:
    """Reject a purported evidence quote that stitches across an unbridged reset."""
    if evidence_location is None:
        # A repeated or missing quote cannot identify which side of a real
        # topic reset the model intended. Keeping both sides would violate the
        # one-learning-objective contract, so fail closed here.
        return bool(transitions)
    evidence_start_line, evidence_left, evidence_end_line, evidence_right = (
        evidence_location
    )
    return any(
        (evidence_start_line, evidence_left)
        < (transition.navigation_line, transition.navigation_left)
        < (evidence_end_line, evidence_right)
        for transition in transitions
    )


def _single_objective_intra_cue_quotes(
    segments: list[dict],
    *,
    evidence_location: tuple[int, int, int, int] | None,
    transitions: list[_TopicTransition],
) -> tuple[str | None, str | None]:
    """Project edges around a topic transition contained inside a coarse cue."""
    if evidence_location is None:
        return None, None
    evidence_start_line, evidence_left, evidence_end_line, evidence_right = (
        evidence_location
    )

    prior = [
        item
        for item in transitions
        if (item.new_side_line, item.new_side_left)
        <= (evidence_start_line, evidence_left)
    ]
    following = [
        item
        for item in transitions
        if (item.navigation_line, item.navigation_left)
        > (evidence_end_line, evidence_right)
    ]
    start_quote: str | None = None
    end_quote: str | None = None
    if prior:
        transition = max(
            prior,
            key=lambda item: (item.new_side_line, item.new_side_left),
        )
        text = str(segments[transition.new_side_line].get("text") or "")
        retained = text[transition.new_side_left:]
        if _WORD_RE.search(retained):
            start_quote = _exact_boundary_quote(retained, want="start")
    if following:
        transition = min(
            following,
            key=lambda item: (item.navigation_line, item.navigation_left),
        )
        text = str(segments[transition.navigation_line].get("text") or "")
        raw_prefix = text[:transition.navigation_left].rstrip(" ,;:—-")
        retained = _last_safe_complete_prefix(raw_prefix)
        if (
            not retained
            and transition.navigation_line > 0
            and 1 <= len(_toks(raw_prefix)) <= 12
        ):
            previous_text = str(
                segments[transition.navigation_line - 1].get("text") or ""
            )
            combined_tail = " ".join(
                f"{previous_text} {raw_prefix}".split()[-48:]
            )
            if (
                _SPLIT_CAPTION_COMPLETION_SIGNAL_RE.search(combined_tail)
                and not _terminal_content_is_explicitly_incomplete(combined_tail)
            ):
                retained = raw_prefix
        if _WORD_RE.search(retained):
            quote = _exact_boundary_quote(retained, want="end")
            if transition.worked_unit:
                end_quote = quote
            else:
                span = _quote_character_span(retained, quote)
                end_quote = (
                    retained[span[0]:].strip() if span is not None else quote
                )
    return start_quote, end_quote


def _completed_unit_end_before_transition(
    segments: list[dict],
    *,
    evidence_location: tuple[int, int, int, int] | None,
    transitions: list[_TopicTransition],
) -> tuple[int, str] | None:
    """Recover an explicit answer ending before later coarse-caption filler."""
    if evidence_location is None:
        return None
    _evidence_start_line, _evidence_left, evidence_end_line, evidence_right = (
        evidence_location
    )
    following = [
        item
        for item in transitions
        if (item.navigation_line, item.navigation_left)
        > (evidence_end_line, evidence_right)
    ]
    if not following:
        return None
    transition = min(
        following,
        key=lambda item: (item.navigation_line, item.navigation_left),
    )
    for line in range(transition.navigation_line, evidence_end_line - 1, -1):
        text = str(segments[line].get("text") or "")
        selected = (
            text[:transition.navigation_left]
            if line == transition.navigation_line
            else text
        )
        matches = list(_WORKED_UNIT_CLOSING_TAIL_RE.finditer(selected))
        for closing in reversed(matches):
            if line == evidence_end_line and closing.end() <= evidence_right:
                continue
            retained = selected[:closing.end()].rstrip(" ,;:—-")
            if len(_toks(retained)) < 5:
                continue
            quote = _exact_boundary_quote(retained, want="end")
            if quote:
                return line, quote
    return None


def _internal_structural_filler_reason(
    segments: list[dict], start_line: int, end_line: int,
) -> str | None:
    """Tolerate brief internal interruptions, but reject substantial filler."""
    filler_lines = [
        line
        for line in range(start_line + 1, end_line)
        if (
            _cue_is_only_structural_filler(
                str(segments[line].get("text") or "")
            )
            or _INTERNAL_INTERRUPTION_MARKER_RE.search(
                str(segments[line].get("text") or "")
            )
        )
    ]
    if not filler_lines:
        return None

    duration = 0.0
    word_count = 0
    for line in filler_lines:
        segment = segments[line]
        try:
            start = float(segment.get("start") or 0.0)
            end = float(segment.get("end") or start)
        except (TypeError, ValueError, OverflowError):
            return "long_internal_filler_block"
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            return "long_internal_filler_block"
        duration += end - start
        word_count += len(_toks(str(segment.get("text") or "")))
    if (
        duration > _MAX_INTERNAL_FILLER_DURATION_S
        or word_count > _MAX_INTERNAL_FILLER_WORDS
    ):
        return "long_internal_filler_block"
    return None


def _same_cue_internal_filler_reason(text: str) -> str | None:
    """Apply the existing filler budget to interruption blocks inside coarse cues."""
    raw_text = str(text or "")
    blocks: set[tuple[int, int]] = set()
    for match in _INTERNAL_INTERRUPTION_MARKER_RE.finditer(raw_text):
        if not _WORD_RE.search(raw_text[:match.start()]):
            continue
        if not _WORD_RE.search(raw_text[match.end():]):
            continue
        previous = max(
            raw_text.rfind(".", 0, match.start()),
            raw_text.rfind("!", 0, match.start()),
            raw_text.rfind("?", 0, match.start()),
        )
        following = re.search(r"[.!?]", raw_text[match.end():])
        right = (
            match.end() + following.end()
            if following is not None
            else len(raw_text)
        )
        blocks.add((previous + 1, right))
    if sum(
        len(_toks(raw_text[left:right]))
        for left, right in blocks
    ) > _MAX_INTERNAL_FILLER_WORDS:
        return "long_internal_filler_block"
    return None


def _trim_before_visual_dependency(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    start_quote: str,
    end_quote: str,
    evidence_quote: str,
) -> tuple[int, str] | None:
    """Keep a self-contained spoken explanation before a terminal visual-only run."""
    records: list[tuple[int, int, str, bool]] = []
    prior_parts: list[str] = []
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        left, right = 0, len(text)
        if line == start_line:
            spans = _quote_character_spans(text, start_quote)
            if len(spans) == 1:
                left = spans[0][0]
        if line == end_line:
            spans = _quote_character_spans(text, end_quote)
            if len(spans) == 1:
                right = spans[0][1]
        selected = text[left:right]
        for sentence_left, sentence_right in _sentence_character_spans(selected):
            sentence = selected[sentence_left:sentence_right].strip()
            if not sentence:
                continue
            requires_visual = _sentence_requires_visual_context(
                sentence,
                prior_text=" ".join(prior_parts),
            )
            records.append((line, left + sentence_left, sentence, requires_visual))
            prior_parts.append(sentence)

    for index, (line, sentence_left, _sentence, requires_visual) in enumerate(records):
        if not requires_visual:
            continue
        later_verbal = " ".join(
            sentence
            for _line, _left, sentence, visual in records[index + 1:]
            if not visual
        )
        if len(_toks(later_verbal)) >= 6:
            # The visual passage is internal and later verbal teaching resumes.
            # Boundary trimming must not erase that later educational content.
            continue
        prefix_text = " ".join(
            sentence for _line, _left, sentence, _visual in records[:index]
        ).strip()
        if (
            len(_toks(prefix_text)) < 8
            or not _contains_quote(prefix_text, evidence_quote)
            or _terminal_content_is_explicitly_incomplete(prefix_text)
        ):
            return None
        text = str(segments[line].get("text") or "")
        if sentence_left > 0:
            trim_line = line
            retained_edge = text[:sentence_left].rstrip()
        else:
            trim_line = line - 1
            if trim_line < start_line:
                return None
            retained_edge = str(segments[trim_line].get("text") or "").rstrip()
        replacement, _ = _expanded_context_edge_quote(
            retained_edge,
            want="end",
        )
        return (trim_line, replacement) if replacement else None
    return None


def _trim_terminal_meta_suffix(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    start_quote: str,
    end_quote: str,
    evidence_quote: str,
    learning_objective: str,
) -> tuple[int, str] | None:
    """Drop a terminal future preview or mastery recap after complete teaching."""
    def selected_part(line: int) -> str:
        text = str(segments[line].get("text") or "")
        left, right = 0, len(text)
        if line == start_line:
            spans = _quote_character_spans(text, start_quote)
            if len(spans) == 1:
                left = spans[0][0]
        if line == end_line:
            spans = _quote_character_spans(text, end_quote)
            if len(spans) == 1:
                right = spans[0][1]
        return text[left:right]

    prefix_parts: list[str] = []
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        selected = selected_part(line)
        future_preview = _TERMINAL_FUTURE_PREVIEW_RE.match(selected)
        mastery_recap = _TERMINAL_MASTERY_RECAP_RE.search(selected)
        noise = future_preview or mastery_recap
        if noise is None:
            if selected.strip():
                prefix_parts.append(selected.strip())
            continue
        retained_here = selected[:noise.start()].rstrip(" ,;:—-")
        prefix_text = " ".join(
            part for part in (*prefix_parts, retained_here) if part
        ).strip()
        later_parts = [
            selected_part(later).strip()
            for later in range(line + 1, end_line + 1)
            if selected_part(later).strip()
        ]
        same_cue_suffix = selected[noise.start():]
        later_sentences = [
            same_cue_suffix[left:right].strip()
            for left, right in _sentence_character_spans(same_cue_suffix)
        ][1:]
        teaching_resumes = any(
            _TERMINAL_META_RESUMPTION_RE.match(part)
            for part in (*later_sentences, *later_parts)
        )
        later_text = " ".join(later_parts)
        if (
            future_preview is not None
            and later_text
            and _objective_bridges_sections(
                learning_objective,
                prefix_text,
                later_text,
            )
        ):
            teaching_resumes = True
        if teaching_resumes:
            prefix_parts.append(selected.strip())
            continue
        if (
            len(_toks(prefix_text)) < 8
            or not _contains_quote(prefix_text, evidence_quote)
            or _terminal_content_is_explicitly_incomplete(prefix_text)
            or _cue_has_explicit_dangling_end(prefix_text, "")
        ):
            if selected.strip():
                prefix_parts.append(selected.strip())
            continue
        if _WORD_RE.search(retained_here):
            replacement = _exact_boundary_quote(retained_here, want="end")
            return (line, replacement) if replacement else None
        trim_line = line - 1
        if trim_line < start_line:
            return None
        retained_edge = str(segments[trim_line].get("text") or "").rstrip()
        replacement = _exact_boundary_quote(retained_edge, want="end")
        return (trim_line, replacement) if replacement else None
    return None


def _sentence_requires_visual_context(
    text: str,
    *,
    prior_text: str = "",
) -> bool:
    raw_text = str(text or "")
    if _VISUAL_DEPENDENCY_RE.search(raw_text):
        return True
    for match in _DEICTIC_POINT_DEPENDENCY_RE.finditer(raw_text):
        grounding_context = " ".join(
            part for part in (prior_text, raw_text[:match.start()]) if part
        )
        definitions = [
            definition.group("label").casefold()
            for definition in _DEICTIC_POINT_DEFINITION_RE.finditer(
                grounding_context
            )
        ]
        first = match.group("first").casefold()
        second = match.group("second").casefold()
        required = {first: 1, second: 1}
        if first == second:
            required[first] = 2
        if any(definitions.count(label) < count for label, count in required.items()):
            return True
    return False


def _clip_requires_visual_context(
    text: str,
    *,
    learning_objective: str = "",
    speech_blocks: list[str] | None = None,
) -> bool:
    raw_text = str(text or "")
    if (
        _VISUAL_DEPENDENCY_RE.search(raw_text) is None
        and _DEICTIC_POINT_DEPENDENCY_RE.search(raw_text) is None
        and _SENTENCE_LOCAL_VISUAL_SIGNAL_RE.search(raw_text) is None
    ):
        return False
    prior_parts: list[str] = []
    records: list[tuple[str, bool]] = []
    raw_sentence_spans = _sentence_character_spans(raw_text)
    max_raw_sentence_words = max(
        (
            len(_toks(raw_text[left:right]))
            for left, right in raw_sentence_spans
        ),
        default=0,
    )
    # Preserve reconstructed logical sentences for ordinary clips. Only fall
    # back to caption-sized speech blocks when missing punctuation would make
    # one runaway sentence hide an internal return to verbal teaching.
    using_speech_blocks = bool(
        speech_blocks and max_raw_sentence_words > 120
    )
    blocks = list(speech_blocks or []) if using_speech_blocks else [raw_text]
    cross_block_visual_indexes: set[int] = set()
    if using_speech_blocks:
        individually_visual = [
            _sentence_requires_visual_context(
                block,
                prior_text=" ".join(blocks[:index]),
            )
            for index, block in enumerate(blocks)
        ]
        # Caption boundaries can split a short signal such as "as you can / see".
        # Scan only two- and three-cue windows whose individual cues were clean;
        # this preserves a narrow internal visual aside instead of making its
        # neighboring verbal teaching visual as well.
        for width in (2, 3):
            for left in range(0, len(blocks) - width + 1):
                indexes = range(left, left + width)
                if any(
                    individually_visual[index]
                    or index in cross_block_visual_indexes
                    for index in indexes
                ):
                    continue
                joined = " ".join(blocks[index] for index in indexes)
                if _sentence_requires_visual_context(
                    joined,
                    prior_text=" ".join(blocks[:left]),
                ):
                    cross_block_visual_indexes.update(indexes)
    for block_index, block in enumerate(blocks):
        for left, right in _sentence_character_spans(block):
            sentence = block[left:right].strip()
            requires_visual = bool(
                block_index in cross_block_visual_indexes
                or _sentence_requires_visual_context(
                    sentence,
                    prior_text=" ".join(prior_parts),
                )
            )
            records.append((sentence, requires_visual))
            if sentence:
                prior_parts.append(sentence)
    visual_indexes = [
        index for index, (_sentence, visual) in enumerate(records) if visual
    ]
    if not visual_indexes:
        return False
    first_visual, last_visual = visual_indexes[0], visual_indexes[-1]
    visual_text = " ".join(records[index][0] for index in visual_indexes)
    if len(
        _content_tokens(learning_objective) & _content_tokens(visual_text)
    ) >= 2:
        return True
    verbal_before = " ".join(
        sentence for sentence, visual in records[:first_visual] if not visual
    )
    verbal_after = " ".join(
        sentence for sentence, visual in records[last_visual + 1:] if not visual
    )
    teaching_resumes = any(
        len(_toks(sentence)) >= 6
        and _opening_clause_is_standalone(sentence)
        for sentence, visual in records[last_visual + 1:]
        if not visual
    )
    teaching_resumes = teaching_resumes or len(_toks(verbal_after)) >= 16
    return not (
        len(_toks(verbal_before)) >= 8
        and len(_toks(verbal_after)) >= 6
        and teaching_resumes
    )


def _near_duplicate(a: dict, b: dict, threshold: float = _DUPLICATE_OVERLAP) -> bool:
    a_cue_ids = {
        str(value).strip()
        for value in (a.get("cue_ids") or [])
        if str(value).strip()
    }
    b_cue_ids = {
        str(value).strip()
        for value in (b.get("cue_ids") or [])
        if str(value).strip()
    }
    if a_cue_ids and b_cue_ids:
        shared = a_cue_ids & b_cue_ids
        if not shared:
            return False
        a_semantic_spans = a.get("_semantic_spans_by_cue")
        b_semantic_spans = b.get("_semantic_spans_by_cue")
        if not isinstance(a_semantic_spans, dict) or not isinstance(
            b_semantic_spans, dict
        ):
            return True
        for cue_id in shared:
            a_span = a_semantic_spans.get(cue_id)
            b_span = b_semantic_spans.get(cue_id)
            if (
                isinstance(a_span, tuple)
                and len(a_span) == 2
                and isinstance(b_span, tuple)
                and len(b_span) == 2
                and max(a_span[0], b_span[0]) < min(a_span[1], b_span[1])
            ):
                return True
        return False

    overlap = min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"]))
    if overlap <= 0:
        return False
    shorter = min(float(a["end"]) - float(a["start"]),
                  float(b["end"]) - float(b["start"]))
    return shorter > 0 and overlap / shorter >= threshold


def _semantic_restatement(
    a: dict,
    b: dict,
    threshold: float = _DUPLICATE_OVERLAP,
) -> bool:
    """Match reworded copies by their normalized objective and facet."""
    generic = {
        "complete", "concept", "example", "explain", "idea", "learn",
        "lesson", "point", "teach", "understand", "work",
    }
    a_tokens = _content_tokens(
        f"{a.get('learning_objective', '')} {a.get('facet', '')}"
    ) - generic
    b_tokens = _content_tokens(
        f"{b.get('learning_objective', '')} {b.get('facet', '')}"
    ) - generic
    smaller = min(len(a_tokens), len(b_tokens))
    if smaller < 2:
        return False
    shared = len(a_tokens & b_tokens)
    return shared >= 2 and shared / smaller >= threshold


def _duplicates(a: dict, b: dict) -> bool:
    return _near_duplicate(a, b) or _semantic_restatement(a, b)


def _content_tokens(text: str) -> set[str]:
    stop = {
        "about", "after", "again", "also", "an", "and", "are", "as", "at", "be",
        "because", "been", "before", "being", "but", "by", "can", "could", "did", "do",
        "does", "doing", "for", "from", "had", "has", "have", "having", "he", "her",
        "here", "him", "his", "how", "if", "in", "into", "is", "it", "its", "just",
        "may", "me", "more", "most", "my", "no", "nor", "not", "now", "of", "off",
        "on", "only", "or", "our", "she", "should", "so", "some", "such", "than",
        "that", "the", "their", "them", "then", "there", "these", "they", "this",
        "those", "through", "to", "too", "us", "very", "was", "we", "were", "what",
        "when", "where", "which", "who", "why", "will", "with", "would", "yes", "you",
        "your",
    }

    def stem(token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 5:
            return token[:-3]
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        if token.endswith("es") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    return {
        stem(token) for token in _toks(text)
        if (len(token) >= 2 or token in {"c", "r"}) and token not in stop
    }


def _text_has_grounding(text: str, transcript_text: str) -> bool:
    source = _content_tokens(transcript_text)
    generated = _content_tokens(text)
    if not source or not generated:
        return False
    shared = source & generated
    return len(shared) >= min(2, len(generated))


def _range_repair_lost_support(
    text: str,
    *,
    original_text: str,
    retained_text: str,
) -> bool:
    generated_tokens = _content_tokens(text)
    original_support = generated_tokens & _content_tokens(original_text)
    retained_support = generated_tokens & _content_tokens(retained_text)
    return bool(original_support - retained_support)


def _objective_after_range_repair(
    objective: str,
    *,
    original_text: str,
    retained_text: str,
    evidence_quote: str,
    require_grounding: bool = False,
) -> str:
    """Remove claims whose only transcript support was cut by range repair."""
    lost_support = _range_repair_lost_support(
        objective,
        original_text=original_text,
        retained_text=retained_text,
    )
    if not lost_support and not (
        require_grounding and not _text_has_grounding(objective, retained_text)
    ):
        return objective
    grounded = " ".join(str(evidence_quote or "").split()).rstrip(" .!?;:")
    if not grounded:
        return objective
    return f"Understand this transcript-grounded point: {grounded}."


def _title_after_range_repair(
    title: str,
    *,
    original_text: str,
    retained_text: str,
    evidence_quote: str,
    require_grounding: bool = False,
) -> str:
    """Keep repaired clip labels within the teaching claim that remains."""
    lost_support = _range_repair_lost_support(
        title,
        original_text=original_text,
        retained_text=retained_text,
    )
    if not lost_support and not (
        require_grounding and not _text_has_grounding(title, retained_text)
    ):
        return title
    grounded_words = " ".join(str(evidence_quote or "").split()).split()
    if not grounded_words:
        return title
    grounded = " ".join(grounded_words[:10]).rstrip(" .!?;:")
    return grounded[:80]


def _validated_assessment(value: object, *, grounding_text: str) -> dict | None:
    """Validate assessment evidence only against the real clip transcript.

    ``evidence_quote`` is consumed as an internal guard and intentionally omitted
    from the stored dictionary to preserve the existing optional assessment shape.
    """
    try:
        draft = value if isinstance(value, _AssessmentDraft) else _AssessmentDraft.model_validate(value)
    except (ValidationError, TypeError, ValueError):
        return None
    if not _contains_quote(grounding_text, draft.evidence_quote):
        return None
    evidence_tokens = _content_tokens(draft.evidence_quote)
    if len(evidence_tokens) < 2:
        return None
    answer_support = _content_tokens(
        f"{draft.options[draft.correct_index]} {draft.explanation}"
    )
    if not evidence_tokens & answer_support:
        return None
    return {
        "prompt": draft.prompt,
        "options": list(draft.options),
        "correct_index": draft.correct_index,
        "explanation": draft.explanation,
    }


@dataclass
class _Conversion:
    clips: list[dict] = field(default_factory=list)
    proposed_count: int = 0
    rejected_reasons: list[str] = field(default_factory=list)
    enrichment_errors: list[str] = field(default_factory=list)
    near_duplicate: bool = False
    non_chronological: bool = False
    medium_uncertainty: bool = False
    score_below_green: bool = False
    long_clip: bool = False
    repair_candidates: list["_BoundaryRepairCandidate"] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.clips)


@dataclass(frozen=True)
class _BoundaryRepairCandidate:
    candidate_id: str
    prefix: str
    proposal: _BoundaryTopic
    start_line: int
    end_line: int
    reason: str


def _strict_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if 0.0 <= score <= 1.0 else None


def _normalized_request_text(value: object) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value or "")).casefold().split()
    )


def _validated_intent_constraints(
    plan: object,
    topic: str,
) -> tuple[dict[str, _IntentConstraint], str | None]:
    """Validate the selector's same-call interpretation against the exact request."""
    if not isinstance(plan, (_CompactBoundaryPlan, _IntentBoundaryPlan)):
        return {}, None
    expected_request = topic.strip() or "(all educational topics)"
    request_intent = plan.request_intent
    if _normalized_request_text(request_intent.exact_request) != _normalized_request_text(
        expected_request
    ):
        return {}, "intent_contract_request_mismatch"
    constraints = {
        constraint.constraint_id: constraint
        for constraint in request_intent.constraints
    }
    if len(constraints) != len(request_intent.constraints) or not constraints:
        return {}, "intent_contract_duplicate_or_empty_ids"
    if any(
        not _contains_quote(expected_request, constraint.source_phrase)
        for constraint in constraints.values()
    ):
        return {}, "intent_contract_ungrounded_source_phrase"
    required_tokens = _content_tokens(expected_request)
    covered_tokens = set().union(*(
        _content_tokens(constraint.source_phrase)
        for constraint in constraints.values()
    ))
    if required_tokens and not required_tokens.issubset(covered_tokens):
        return {}, "intent_contract_incomplete_request_coverage"
    return constraints, None


def _learning_details(topic_obj: object, clip_text: str, topic: str) -> tuple[dict, list[str]]:
    errors: list[str] = []
    details = {"summary": "", "takeaways": [], "match_reason": "", "assessment": None}
    summary = " ".join(str(getattr(topic_obj, "summary", "") or "").split())
    takeaways = [" ".join(str(item).split()) for item in
                 (getattr(topic_obj, "takeaways", None) or []) if str(item).strip()]
    match_reason = " ".join(str(getattr(topic_obj, "match_reason", "") or "").split())
    if not summary or not _text_has_grounding(summary, clip_text):
        errors.append("summary_not_grounded")
    else:
        details["summary"] = summary
    if not 2 <= len(takeaways) <= 4 or len({item.casefold() for item in takeaways}) != len(takeaways):
        errors.append("takeaways_invalid")
    elif any(not _text_has_grounding(item, clip_text) for item in takeaways):
        errors.append("takeaways_not_grounded")
    else:
        details["takeaways"] = takeaways
    if not match_reason or not _text_has_grounding(match_reason, clip_text):
        errors.append("match_reason_not_grounded")
    elif topic.strip() and not (_content_tokens(topic) & _content_tokens(match_reason)):
        errors.append("match_reason_not_topic_specific")
    else:
        details["match_reason"] = match_reason
    assessment = _validated_assessment(
        getattr(topic_obj, "assessment", None), grounding_text=clip_text,
    )
    if assessment is None:
        errors.append("assessment_invalid")
    else:
        details["assessment"] = assessment
    return details, errors


def _configured_clip_limit(settings: dict) -> int | None:
    configured = settings.get("max_clips")
    return None if configured is None else max(0, int(configured))


def _quality_order(clip: dict) -> tuple[float, float, float]:
    scores = (
        float(clip.get("informativeness") or 0.0),
        float(clip.get("topic_relevance") or 0.0),
        float(clip.get("educational_importance") or 0.0),
    )
    return min(scores), sum(scores) / len(scores), scores[1]


def _intent_priority(clip: dict) -> tuple[int, float]:
    role = str(clip.get("intent_role") or "primary").strip().lower()
    try:
        coverage = max(0.0, min(1.0, float(clip.get("intent_coverage", 1.0))))
    except (TypeError, ValueError, OverflowError):
        coverage = 0.0
    return (0 if role == "primary" else 1, -coverage)


def _difficulty_stage(difficulty: object) -> int:
    score = float(difficulty or 0.0)
    return 0 if score < 0.34 else 1 if score < 0.67 else 2


def _finalize_clips(clips: list[dict], settings: dict) -> list[dict]:
    """Keep the strongest restatement and stage every qualifying candidate."""
    quality_order = sorted(
        clips,
        key=lambda clip: (
            -_quality_order(clip)[0],
            -_quality_order(clip)[1],
            -_quality_order(clip)[2],
            float(clip["start"]),
            float(clip["end"]),
            int(clip.get("_proposal_index") or 0),
            str(clip.get("selection_candidate_id") or ""),
        ),
    )
    kept: list[dict] = []
    for candidate in quality_order:
        if not any(_duplicates(candidate, prior) for prior in kept):
            kept.append(candidate)
    limit = _configured_clip_limit(settings)
    by_candidate_id = {
        str(clip.get("selection_candidate_id") or ""): clip
        for clip in kept
        if str(clip.get("selection_candidate_id") or "")
    }

    def prerequisite_closure(
        candidate_id: str,
        trail: set[str],
    ) -> list[dict] | None:
        if candidate_id in trail:
            return None
        clip = by_candidate_id.get(candidate_id)
        if clip is None:
            return None
        closure: list[dict] = []
        for prerequisite in clip.get("prerequisite_ids") or []:
            prerequisite_items = prerequisite_closure(
                str(prerequisite),
                {*trail, candidate_id},
            )
            if prerequisite_items is None:
                return None
            closure.extend(prerequisite_items)
        closure.append(clip)
        deduped: list[dict] = []
        seen_ids: set[str] = set()
        for item in closure:
            item_id = str(item.get("selection_candidate_id") or "")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                deduped.append(item)
        return deduped

    selected: list[dict] = []
    selected_ids: set[str] = set()
    for candidate in kept:
        candidate_id = str(candidate.get("selection_candidate_id") or "")
        bundle = prerequisite_closure(candidate_id, set()) if candidate_id else [candidate]
        if bundle is None:
            continue
        additions = [
            item
            for item in bundle
            if str(item.get("selection_candidate_id") or "") not in selected_ids
        ]
        if limit is not None and len(selected) + len(additions) > limit:
            continue
        for item in additions:
            selected.append(item)
            item_id = str(item.get("selection_candidate_id") or "")
            if item_id:
                selected_ids.add(item_id)
        if limit is not None and len(selected) >= limit:
            break
    selected.sort(
        key=lambda clip: (
            _difficulty_stage(clip.get("difficulty")),
            *_intent_priority(clip),
            -_quality_order(clip)[0],
            -_quality_order(clip)[1],
            -_quality_order(clip)[2],
            float(clip["start"]),
            float(clip["end"]),
            int(clip.get("_proposal_index") or 0),
            str(clip.get("selection_candidate_id") or ""),
        )
    )
    for index, clip in enumerate(selected):
        clip["sequence_index"] = index + 1
    return selected


def _drop_unmet_prerequisite_clips(report: _Conversion) -> None:
    """Fail closed on unknown or cyclic selector dependencies before shipping."""
    by_id = {
        str(clip.get("selection_candidate_id") or ""): clip
        for clip in report.clips
        if str(clip.get("selection_candidate_id") or "")
    }
    resolved = {
        candidate_id
        for candidate_id, clip in by_id.items()
        if bool(clip.get("is_standalone")) and not clip.get("prerequisite_ids")
    }
    changed = True
    while changed:
        changed = False
        for candidate_id, clip in by_id.items():
            if candidate_id in resolved:
                continue
            prerequisites = {
                str(value)
                for value in (clip.get("prerequisite_ids") or [])
                if str(value)
            }
            if prerequisites and prerequisites.issubset(resolved):
                resolved.add(candidate_id)
                changed = True
    if len(resolved) == len(by_id):
        return
    removed = set(by_id) - resolved
    report.clips = [
        clip
        for clip in report.clips
        if str(clip.get("selection_candidate_id") or "") in resolved
    ]
    for candidate_id in sorted(removed):
        report.rejected_reasons.append(
            f"candidate_{candidate_id}:unmet_or_cyclic_prerequisite"
        )
    for index, clip in enumerate(report.clips):
        clip["sequence_index"] = index + 1


def _plan_to_report(
    plan: _Plan | _BoundaryPlan | _CompactBoundaryPlan | _IntentBoundaryPlan |
    _LegacyPlan | _ProductionPlan,
    segments: list[dict],
    words: list[dict],
    settings: dict,
    *,
    topic: str = "",
    require_enrichment: bool = False,
    context_cue_limit: int | None = None,
) -> _Conversion:
    report = _Conversion(proposed_count=len(plan.topics))
    n = len(segments)
    if not n:
        report.rejected_reasons.append("missing_segments")
        return report

    intent_constraints, intent_contract_error = _validated_intent_constraints(
        plan,
        topic,
    )
    if intent_contract_error is not None:
        report.rejected_reasons.append(intent_contract_error)
        return report
    intent_constraint_ids = set(intent_constraints)

    ignore_caption_case = bool(settings.get("_segment_ignore_caption_case", True))
    raw: list[dict] = []
    seen_candidate_ids: set[str] = set()

    for index, proposal in enumerate(plan.topics):
        prefix = f"proposal_{index}"
        if isinstance(proposal, _ProductionTopic):
            kind = str(proposal.kind or "").strip().lower()
            if kind not in {"content", "educational"}:
                report.rejected_reasons.append(f"{prefix}:not_educational")
                continue

        a, b = proposal.start_line, proposal.end_line
        if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int)
                or not isinstance(b, int) or a < 0 or b < 0 or a >= n or b >= n or a > b):
            report.rejected_reasons.append(f"{prefix}:bad_index")
            continue
        proposed_start, proposed_end = a, b
        start_quote = str(proposal.start_quote or "").strip()
        end_quote = str(proposal.end_quote or "").strip()
        start_text = str(segments[a].get("text") or "").strip()
        end_text = str(segments[b].get("text") or "").strip()
        quote_repaired = False
        fallback_start_edge = False
        fallback_end_edge = False
        trimmed_incomplete_end_suffix = False
        trimmed_visual_end_suffix = False
        completed_forward_sentence = False
        completed_split_caption_tail = False
        start_recovered_forward = False
        trimmed_terminal_meta_suffix = False
        boundary_fallback_reasons: list[str] = []
        if not _contains_quote(start_text, start_quote):
            nearby_start = max(0, proposed_start - 2)
            nearby_end = min(n - 1, proposed_end + 2)
            matching_lines = [
                line
                for line in range(nearby_start, nearby_end + 1)
                if _contains_quote(
                    str(segments[line].get("text") or ""), start_quote
                )
            ]
            cross_matches = (
                _cross_cue_quote_matches(
                    segments,
                    start_quote,
                    proposed_start,
                    proposed_end,
                )
                if not matching_lines
                else []
            )
            anchored_line: int | None = None
            anchored_quote = start_quote
            if len(matching_lines) == 1:
                anchored_line = matching_lines[0]
            elif len(cross_matches) == 1:
                anchored_line = cross_matches[0][0]
                anchored_quote = cross_matches[0][2]
            can_reanchor = bool(
                anchored_line is not None
                and all(
                    _cue_is_only_structural_filler(
                        str(segments[line].get("text") or "")
                    )
                    for line in range(proposed_start, anchored_line)
                )
            )
            if can_reanchor and anchored_line is not None:
                a = anchored_line
                start_quote = anchored_quote
                start_text = str(segments[a].get("text") or "").strip()
            else:
                start_quote = _exact_boundary_quote(start_text, want="start")
                fallback_start_edge = True
                boundary_fallback_reasons.append("bad_start_quote")
            quote_repaired = True
        if not _contains_quote(end_text, end_quote):
            nearby_start = max(0, proposed_start - 2)
            nearby_end = min(n - 1, proposed_end + 2)
            matching_lines = [
                line
                for line in range(nearby_start, nearby_end + 1)
                if _contains_quote(
                    str(segments[line].get("text") or ""), end_quote
                )
            ]
            cross_matches = (
                _cross_cue_quote_matches(
                    segments,
                    end_quote,
                    proposed_start,
                    proposed_end,
                )
                if not matching_lines
                else []
            )
            anchored_line = None
            anchored_quote = end_quote
            if len(matching_lines) == 1:
                anchored_line = matching_lines[0]
            elif len(cross_matches) == 1:
                anchored_line = cross_matches[0][1]
                anchored_quote = cross_matches[0][3]
            can_reanchor = bool(
                anchored_line is not None
                and all(
                    _cue_is_only_structural_filler(
                        str(segments[line].get("text") or "")
                    )
                    for line in range(anchored_line + 1, proposed_end + 1)
                )
            )
            if can_reanchor and anchored_line is not None:
                b = anchored_line
                end_quote = anchored_quote
                end_text = str(segments[b].get("text") or "").strip()
            else:
                end_quote = _exact_boundary_quote(end_text, want="end")
                fallback_end_edge = True
                boundary_fallback_reasons.append("bad_end_quote")
            quote_repaired = True
        if a > b:
            a, b = proposed_start, proposed_end
            start_text = str(segments[a].get("text") or "").strip()
            end_text = str(segments[b].get("text") or "").strip()
            start_quote = _exact_boundary_quote(start_text, want="start")
            end_quote = _exact_boundary_quote(end_text, want="end")
            fallback_start_edge = fallback_end_edge = True
            boundary_fallback_reasons.append("reversed_quote_order")
            quote_repaired = True
        evidence_quote_for_section, evidence_location_for_section = (
            _proposal_evidence_anchor(
                proposal,
                intent_constraints,
                segments,
                a,
                b,
            )
        )
        objective_for_section = str(
            getattr(proposal, "learning_objective", "")
            or getattr(proposal, "facet", "")
            or ""
        )
        topic_transitions_for_section = _candidate_topic_transitions(
            segments,
            a,
            b,
            evidence_quote=evidence_quote_for_section,
            learning_objective=objective_for_section,
        )
        if _evidence_crosses_topic_transition(
            evidence_location=evidence_location_for_section,
            transitions=topic_transitions_for_section,
        ):
            report.rejected_reasons.append(
                f"{prefix}:topic_evidence_crosses_topic_reset"
            )
            continue
        intra_start_quote, intra_end_quote = _single_objective_intra_cue_quotes(
            segments,
            evidence_location=evidence_location_for_section,
            transitions=topic_transitions_for_section,
        )
        completed_end_override = (
            None
            if intra_end_quote
            else _completed_unit_end_before_transition(
                segments,
                evidence_location=evidence_location_for_section,
                transitions=topic_transitions_for_section,
            )
        )
        section_start, section_end = _single_objective_section_bounds(
            segments,
            a,
            b,
            evidence_location=evidence_location_for_section,
            transitions=topic_transitions_for_section,
        )
        if completed_end_override is not None:
            section_end, intra_end_quote = completed_end_override
        semantic_min_start: int | None = None
        semantic_max_end: int | None = None
        if evidence_location_for_section is not None:
            evidence_start_line, evidence_left, evidence_end_line, evidence_right = (
                evidence_location_for_section
            )
            if any(
                (item.navigation_line, item.navigation_left)
                <= (evidence_start_line, evidence_left)
                for item in topic_transitions_for_section
            ):
                semantic_min_start = section_start
            if any(
                (item.navigation_line, item.navigation_left)
                > (evidence_end_line, evidence_right)
                for item in topic_transitions_for_section
            ):
                semantic_max_end = section_end
        if section_start != a:
            a = section_start
            start_text = str(segments[a].get("text") or "").strip()
            start_quote, edge_error = _expanded_context_edge_quote(
                start_text,
                want="start",
            )
            if edge_error:
                start_quote = _exact_boundary_quote(start_text, want="start")
            fallback_start_edge = True
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_adjacent_topic_before")
        if section_end != b:
            b = section_end
            end_text = str(segments[b].get("text") or "").strip()
            end_quote, edge_error = _expanded_context_edge_quote(
                end_text,
                want="end",
            )
            if edge_error:
                end_quote = _exact_boundary_quote(end_text, want="end")
            fallback_end_edge = True
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_adjacent_topic_after")
        intra_start_boundary = bool(intra_start_quote)
        intra_end_boundary = bool(intra_end_quote)
        if intra_start_quote:
            start_quote = intra_start_quote
            fallback_start_edge = False
            repaired_start_edge = True
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_same_cue_topic_before")
        if intra_end_quote:
            end_quote = intra_end_quote
            fallback_end_edge = False
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_same_cue_topic_after")
        contextual_example_needs_context = (
            _opening_contextual_example_needs_context(start_text)
        )
        selected_start_before_context = a
        selected_start_quote_before_context = start_quote
        selected_end_before_context = b
        context_repair_source_text = _cue_clip_text(
            segments, a, min(n - 1, b + 1)
        )
        info = _strict_score(proposal.informativeness)
        relevance = _strict_score(proposal.topic_relevance)
        raw_importance = getattr(proposal, "educational_importance", None)
        importance = (
            _strict_score(raw_importance)
            if raw_importance is not None
            else (
                round((float(info) + float(relevance)) / 2.0, 3)
                if info is not None and relevance is not None
                else None
            )
        )
        difficulty = _strict_score(proposal.difficulty)
        if info is None or relevance is None or importance is None or difficulty is None:
            report.rejected_reasons.append(f"{prefix}:score_out_of_range")
            continue
        below_green = next(
            (
                name
                for name, score in (
                    ("informativeness", info),
                    ("topic_relevance", relevance),
                    ("educational_importance", importance),
                )
                if score < _GREEN_SCORE
            ),
            None,
        )
        if below_green is not None:
            report.rejected_reasons.append(f"{prefix}:{below_green}_below_green")
            continue
        if proposal.self_contained is not True:
            report.rejected_reasons.append(f"{prefix}:not_self_contained")
            continue
        if isinstance(proposal, (_BoundaryTopic, _CompactBoundaryTopic)):
            if proposal.directly_teaches_topic is not True:
                report.rejected_reasons.append(f"{prefix}:does_not_directly_teach_topic")
                continue
            if proposal.substantive is not True:
                report.rejected_reasons.append(f"{prefix}:not_substantive")
                continue
            if proposal.factually_grounded is not True:
                report.rejected_reasons.append(f"{prefix}:not_factually_grounded")
                continue
        candidate_id = " ".join(
            str(
                getattr(proposal, "candidate_id", "")
                or f"clip-{index + 1:03d}-{proposal.start_line}-{proposal.end_line}"
            ).split()
        )
        if candidate_id in seen_candidate_ids:
            report.rejected_reasons.append(f"{prefix}:duplicate_candidate_id")
            continue
        seen_candidate_ids.add(candidate_id)
        prerequisites = list(dict.fromkeys(
            " ".join(str(value or "").split())
            for value in (getattr(proposal, "prerequisite_candidate_ids", None) or [])
            if " ".join(str(value or "").split())
        ))
        is_standalone = bool(
            getattr(proposal, "is_standalone", proposal.self_contained)
        )
        if not is_standalone or prerequisites:
            report.rejected_reasons.append(f"{prefix}:not_standalone")
            continue
        uncertainty = str(getattr(proposal, "uncertainty", "low") or "low")
        uncertainty_reasons = [str(getattr(reason, "value", reason))
                               for reason in (getattr(proposal, "uncertainty_reasons", None) or [])]
        boundary_only_uncertainty = bool(
            uncertainty == "high"
            and uncertainty_reasons
            and set(uncertainty_reasons).issubset(
                {"boundary_ambiguous", "overlap_risk"}
            )
        )
        if uncertainty == "high" and not boundary_only_uncertainty:
            report.rejected_reasons.append(f"{prefix}:{uncertainty}_uncertainty")
            continue
        if boundary_only_uncertainty:
            boundary_fallback_reasons.extend(
                f"model_{reason}" for reason in uncertainty_reasons
            )

        start_quote, repaired_start_edge, edge_error = _replace_structural_edge_quote(
            start_text,
            start_quote,
            want="start",
        )
        repaired_start_edge = repaired_start_edge or intra_start_boundary
        if edge_error:
            start_quote = _exact_boundary_quote(start_text, want="start")
            fallback_start_edge = True
            boundary_fallback_reasons.append(edge_error)
            quote_repaired = True
        navigation_recovery = _recover_start_after_edge_navigation(
            start_text,
            evidence_quote=evidence_quote_for_section,
            learning_objective=objective_for_section,
            following_text=(
                _cue_clip_text(segments, a + 1, b) if a < b else ""
            ),
        )
        if navigation_recovery:
            start_quote = navigation_recovery
            repaired_start_edge = True
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_opening_edge_navigation")
        if not repaired_start_edge:
            selected_start_span = _quote_character_span(start_text, start_quote)
            required_setup_bridge = bool(
                selected_start_span is not None
                and selected_start_span[0] == 0
                and (
                    (
                        a < b
                        and _objective_bridges_sections(
                            objective_for_section,
                            start_text,
                            _cue_clip_text(segments, a + 1, b),
                        )
                    )
                    or _contains_bridged_topic_navigation(
                        start_text,
                        learning_objective=objective_for_section,
                    )
                )
            )
            if (
                selected_start_span is not None
                and not required_setup_bridge
                and not _projected_start_is_standalone(
                    start_text,
                    selected_start_span,
                )
            ):
                recovered_start_quote = _recover_projected_start_within_cue(
                    start_text,
                    selected_start_span,
                    evidence_quote=evidence_quote_for_section,
                    learning_objective=objective_for_section,
                    following_text=(
                        _cue_clip_text(segments, a + 1, b) if a < b else ""
                    ),
                )
                if recovered_start_quote:
                    recovered_start_span = _quote_character_span(
                        start_text,
                        recovered_start_quote,
                    )
                    if (
                        recovered_start_span is not None
                        and recovered_start_span[0] == selected_start_span[0]
                    ):
                        # A recovery must move the playback boundary. Treating
                        # the original fragment as repaired bypasses the
                        # cue-by-cue discourse expansion below.
                        recovered_start_quote = ""
                    if (
                        recovered_start_quote
                        and recovered_start_span is not None
                        and recovered_start_span[0] == 0
                        and a > 0
                        and _cue_has_explicit_dangling_end(
                            str(segments[a - 1].get("text") or ""),
                            start_text,
                        )
                    ):
                        recovered_start_quote = ""
                if recovered_start_quote:
                    start_quote = recovered_start_quote
                    repaired_start_edge = True
                    quote_repaired = True
                    boundary_fallback_reasons.append(
                        "recovered_same_cue_sentence_start"
                    )
                else:
                    backward_context_available = False
                    opening_requires_prior_cue = bool(
                        _OPENING_DEPENDENT_PREPOSITION_FRAGMENT_RE.match(start_text)
                        or _OPENING_SUBJECTLESS_PREDICATE_RE.match(start_text)
                        or _OPENING_MATH_CONTINUATION_RE.match(start_text)
                    )
                    if (
                        a > 0
                        and (semantic_min_start is None or a > semantic_min_start)
                        and opening_requires_prior_cue
                    ):
                        try:
                            backward_context_available = (
                                float(segments[a].get("start", 0.0))
                                - float(segments[a - 1].get("end", 0.0))
                                < _SECTION_RESET_GAP_S
                            )
                        except (TypeError, ValueError, OverflowError):
                            backward_context_available = False
                    recovered_forward = (
                        None
                        if backward_context_available
                        else _recover_start_forward_across_cues(
                            segments,
                            a,
                            b,
                            evidence_quote=evidence_quote_for_section,
                            learning_objective=objective_for_section,
                        )
                    )
                    if (
                        recovered_forward is not None
                        and _opening_contextual_example_needs_context(start_text)
                    ):
                        recovered_text = str(
                            segments[recovered_forward[0]].get("text") or ""
                        )
                        if not _local_example_setup_is_complete(recovered_text):
                            recovered_forward = None
                    if recovered_forward is not None:
                        a, start_quote = recovered_forward
                        start_text = str(segments[a].get("text") or "").strip()
                        repaired_start_edge = True
                        start_recovered_forward = True
                        quote_repaired = True
                        boundary_fallback_reasons.append(
                            "recovered_forward_sentence_start"
                        )
        end_quote, repaired_end_edge, edge_error = _replace_structural_edge_quote(
            end_text,
            end_quote,
            want="end",
        )
        if edge_error:
            end_quote = _exact_boundary_quote(end_text, want="end")
            fallback_end_edge = True
            boundary_fallback_reasons.append(edge_error)
            quote_repaired = True
        end_quote, trimmed_edge_noise = _trim_end_quote_before_edge_noise(
            end_text,
            end_quote,
            evidence_quote=evidence_quote_for_section,
            learning_objective=objective_for_section,
        )
        if trimmed_edge_noise:
            repaired_end_edge = True
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_trailing_edge_noise")
        selected_end_span = _quote_character_span(end_text, end_quote)
        following_end_text = (
            str(segments[b + 1].get("text") or "")
            if b + 1 < len(segments)
            else ""
        )
        if (
            selected_end_span is not None
            and (
                _terminal_content_is_explicitly_incomplete(end_text)
                or bool(re.search(r"[,;:\-—][\"'’”)]*\s*$", end_text))
            )
            and _cue_has_weak_end(
                end_text,
                following_end_text,
                ignore_caption_case=ignore_caption_case,
            )
        ):
            recovered_end_quote = _complete_prefix_end_quote(end_text)
            recovered_end_span = _quote_character_span(
                end_text,
                recovered_end_quote,
            )
            if (
                recovered_end_span is not None
                and recovered_end_span[1] < selected_end_span[1]
            ):
                end_quote = recovered_end_quote
                repaired_end_edge = True
                trimmed_incomplete_end_suffix = True
                quote_repaired = True
                boundary_fallback_reasons.append(
                    "trimmed_incomplete_end_suffix"
                )
        visual_trim = _trim_before_visual_dependency(
            segments,
            a,
            b,
            start_quote=start_quote,
            end_quote=end_quote,
            evidence_quote=evidence_quote_for_section,
        )
        if visual_trim is not None:
            b, end_quote = visual_trim
            end_text = str(segments[b].get("text") or "").strip()
            repaired_end_edge = True
            trimmed_visual_end_suffix = True
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_visual_dependent_tail")
        terminal_meta_trim = _trim_terminal_meta_suffix(
            segments,
            a,
            b,
            start_quote=start_quote,
            end_quote=end_quote,
            evidence_quote=evidence_quote_for_section,
            learning_objective=objective_for_section,
        )
        if terminal_meta_trim is not None:
            b, end_quote = terminal_meta_trim
            end_text = str(segments[b].get("text") or "").strip()
            repaired_end_edge = True
            fallback_end_edge = False
            trimmed_terminal_meta_suffix = True
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_terminal_meta_suffix")
        if not trimmed_visual_end_suffix and not trimmed_terminal_meta_suffix:
            split_caption_completion = _complete_split_caption_tail(
                segments,
                b,
                end_quote,
                proposals=list(plan.topics),
                proposal_index=index,
                ignore_caption_case=ignore_caption_case,
                anchor_text=(
                    f"{evidence_quote_for_section} {objective_for_section}"
                ),
            )
            if split_caption_completion is not None:
                b, end_quote = split_caption_completion
                end_text = str(segments[b].get("text") or "").strip()
                repaired_end_edge = True
                fallback_end_edge = False
                completed_split_caption_tail = True
                quote_repaired = True
                boundary_fallback_reasons.append(
                    "completed_split_caption_tail"
                )
        quote_repaired = quote_repaired or repaired_start_edge or repaired_end_edge

        # Run discourse closure against the teaching slice the model selected, not
        # against hook/joke text that is deliberately outside its exact edge quotes.
        closure_segments = segments
        preliminary_start_spans = _quote_character_spans(start_text, start_quote)
        preliminary_end_spans = _quote_character_spans(end_text, end_quote)
        projected_end_needs_continuation = bool(
            len(preliminary_end_spans) == 1
            and b + 1 < len(segments)
            and _TERMINAL_DANGLING_DISCOURSE_LEADIN_RE.search(
                end_text[:preliminary_end_spans[0][1]]
            )
        )
        from .sentences import classify_terminator

        preceding_cue_is_closed = bool(
            a <= 0
            or classify_terminator(
                str(segments[a - 1].get("text") or "")
            )
        )
        projected_start_is_standalone = bool(
            len(preliminary_start_spans) == 1
            and _projected_start_is_standalone(
                start_text,
                preliminary_start_spans[0],
            )
            and (
                preliminary_start_spans[0][0] > 0
                or preceding_cue_is_closed
                or not _cue_opens_mid_thought_at(
                    segments,
                    a,
                    ignore_caption_case=ignore_caption_case,
                )
            )
        )
        projected_end_is_complete = bool(
            trimmed_terminal_meta_suffix
            or completed_split_caption_tail
            or (
                not projected_end_needs_continuation
                and len(preliminary_end_spans) == 1
                and _projected_end_is_complete(
                    end_text,
                    preliminary_end_spans[0],
                    following_text=(
                        str(segments[b + 1].get("text") or "")
                        if b + 1 < len(segments)
                        else ""
                    ),
                )
            )
        )
        if len(preliminary_start_spans) == 1 or len(preliminary_end_spans) == 1:
            closure_segments = [dict(segment) for segment in segments]
            start_left = (
                preliminary_start_spans[0][0]
                if len(preliminary_start_spans) == 1
                else 0
            )
            end_right = (
                preliminary_end_spans[0][1]
                if len(preliminary_end_spans) == 1
                else len(end_text)
            )
            if projected_end_needs_continuation:
                end_right = len(end_text)
            terminal_suffix = re.match(
                r"\s*[,;:.!?]+[\"'’”)]*",
                end_text[end_right:],
            )
            if terminal_suffix is not None:
                end_right += terminal_suffix.end()
            if a == b:
                if start_left < end_right:
                    closure_segments[a]["text"] = start_text[start_left:end_right]
            else:
                closure_segments[a]["text"] = start_text[start_left:]
                closure_segments[b]["text"] = end_text[:end_right]

        closure_selected_end = b
        closure_end_requires_completion = bool(
            not projected_end_is_complete
            and (
                projected_end_needs_continuation
                or re.search(
                    r"[,;:\-—][\"'’”)]*\s*$",
                    str(closure_segments[b].get("text") or ""),
                )
            )
        )
        a, b, closure_error = _close_cue_context(
            closure_segments,
            a,
            b,
            ignore_caption_case=ignore_caption_case,
            cue_limit=context_cue_limit,
            start_boundary_verified=(
                repaired_start_edge or projected_start_is_standalone
            ),
            end_boundary_verified=(
                intra_end_boundary or projected_end_is_complete
            ),
            protected_quote=evidence_quote_for_section,
            learning_objective=objective_for_section,
            min_start_line=semantic_min_start,
            max_end_line=semantic_max_end,
        )
        if closure_error:
            if closure_error == "unresolved_weak_end":
                recovered_end_quote = _complete_prefix_end_quote(
                    str(segments[b].get("text") or "")
                )
                if recovered_end_quote:
                    end_quote = recovered_end_quote
                    fallback_end_edge = False
                    trimmed_incomplete_end_suffix = True
                    quote_repaired = True
                    boundary_fallback_reasons.append(
                        "trimmed_incomplete_end_suffix"
                    )
                    closure_error = None
            if closure_error == "unresolved_weak_end":
                # A setup, question, or example without its answer is not a
                # complete educational unit; this is a content failure rather
                # than a demand for exact acoustic timing.
                report.rejected_reasons.append(f"{prefix}:{closure_error}")
                continue
            if closure_error:
                if closure_error.endswith("_end"):
                    fallback_end_edge = True
                elif not repaired_start_edge:
                    fallback_start_edge = True
                boundary_fallback_reasons.append(closure_error)

        if closure_end_requires_completion and b > closure_selected_end:
            for completion_line in range(closure_selected_end + 1, b + 1):
                completion_quote = _complete_prefix_end_quote(
                    str(segments[completion_line].get("text") or "")
                )
                if not completion_quote:
                    continue
                b = completion_line
                end_text = str(segments[b].get("text") or "").strip()
                end_quote = completion_quote
                repaired_end_edge = True
                trimmed_incomplete_end_suffix = True
                completed_forward_sentence = True
                quote_repaired = True
                boundary_fallback_reasons.append("completed_forward_sentence")
                break

        selected_opening_has_local_setup = False
        if a == selected_start_before_context:
            current_start_text = str(segments[a].get("text") or "")
            current_start_span = _quote_character_span(
                current_start_text,
                start_quote,
            )
            if current_start_span is not None:
                selected_opening_has_local_setup = (
                    _local_example_setup_is_complete(
                        current_start_text[current_start_span[0]:]
                    )
                )
        if (
            contextual_example_needs_context
            and not start_recovered_forward
            and not selected_opening_has_local_setup
        ):
            selected_start_text = str(
                segments[selected_start_before_context].get("text") or ""
            )
            selected_end_text = str(segments[b].get("text") or "")
            selected_example_text, _ = _semantic_clip_slice(
                segments,
                selected_start_before_context,
                b,
                start_span=_quote_character_span(
                    selected_start_text,
                    selected_start_quote_before_context,
                ),
                end_span=_quote_character_span(selected_end_text, end_quote),
            )
            recovered_setup = (
                _cue_clip_text(
                    segments,
                    a,
                    selected_start_before_context - 1,
                )
                if a < selected_start_before_context
                else ""
            )
            recovered_setup_complete = _local_example_setup_is_complete(
                recovered_setup
            )
            later_target_complete = _selected_example_restates_complete_target(
                selected_example_text
            )
            if (
                not recovered_setup_complete
                and not later_target_complete
            ):
                report.rejected_reasons.append(
                    f"{prefix}:unresolved_example_setup"
                )
                continue
            if later_target_complete:
                a = selected_start_before_context
                start_text = str(segments[a].get("text") or "").strip()
                start_quote = selected_start_quote_before_context
                fallback_start_edge = False
                boundary_fallback_reasons.append(
                    "accepted_later_restated_example_target"
                )

        filler_trim = _trim_structural_filler_edges(
            segments,
            a,
            b,
            ignore_caption_case=ignore_caption_case,
        )
        if filler_trim is None:
            report.rejected_reasons.append(f"{prefix}:contains_filler")
            continue
        a, b = filler_trim
        if not trimmed_incomplete_end_suffix and _terminal_content_is_explicitly_incomplete(
            _cue_clip_text(closure_segments, a, b)
        ):
            report.rejected_reasons.append(f"{prefix}:unresolved_weak_end")
            continue
        internal_filler_reason = _internal_structural_filler_reason(segments, a, b)
        if internal_filler_reason:
            boundary_fallback_reasons.append(
                f"retained_{internal_filler_reason}"
            )
        context_was_trimmed = (
            start_recovered_forward or b < selected_end_before_context
        )
        start, end = _padded_cue_bounds(segments, a, b)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            report.rejected_reasons.append(f"{prefix}:reversed_cue_boundary")
            continue
        start, end = round(start, 3), round(end, 3)

        full_clip_text = _cue_clip_text(segments, a, b)
        if not full_clip_text:
            report.rejected_reasons.append(f"{prefix}:empty_cue_transcript")
            continue
        if not _contains_quote(full_clip_text, start_quote):
            start_quote = _exact_boundary_quote(full_clip_text, want="start")
            fallback_start_edge = True
            boundary_fallback_reasons.append("ungrounded_start_quote")
            quote_repaired = True
        if not _contains_quote(full_clip_text, end_quote):
            end_quote = _exact_boundary_quote(full_clip_text, want="end")
            fallback_end_edge = True
            boundary_fallback_reasons.append("ungrounded_end_quote")
            quote_repaired = True
        if not start_quote or not end_quote:
            report.rejected_reasons.append(f"{prefix}:ungrounded_boundary_quote")
            continue

        start_span: tuple[int, int] | None = None
        end_span: tuple[int, int] | None = None
        if a != selected_start_before_context and not start_recovered_forward:
            start_quote, edge_error = _expanded_context_edge_quote(
                str(segments[a].get("text") or ""), want="start"
            )
            if edge_error:
                start_quote = _exact_boundary_quote(
                    str(segments[a].get("text") or ""), want="start"
                )
                fallback_start_edge = True
                boundary_fallback_reasons.append(edge_error)
            quote_repaired = True
        if fallback_start_edge:
            start_text = str(segments[a].get("text") or "")
            trimmed_quote, _ = _expanded_context_edge_quote(
                start_text, want="start"
            )
            start_quote = trimmed_quote or _exact_boundary_quote(
                start_text, want="start"
            )
        start_span, start_projected, edge_error = _semantic_edge_quote(
            str(segments[a].get("text") or ""), start_quote, want="start"
        )
        if edge_error:
            start_quote = _exact_boundary_quote(
                str(segments[a].get("text") or ""), want="start"
            )
            start_span, start_projected, edge_error = _semantic_edge_quote(
                str(segments[a].get("text") or ""), start_quote, want="start"
            )
            fallback_start_edge = True
            boundary_fallback_reasons.append(edge_error or "start_edge_fallback")
        if edge_error or start_span is None:
            report.rejected_reasons.append(f"{prefix}:empty_cue_transcript")
            continue
        assert start_span is not None
        start_text = str(segments[a].get("text") or "")
        start_quote = _literal_source_quote(start_text, start_quote, start_span)
        if _edge_has_unresolved_structural_filler(
            str(segments[a].get("text") or ""), start_span, want="start"
        ):
            trimmed_quote, _ = _expanded_context_edge_quote(
                start_text, want="start"
            )
            start_quote = trimmed_quote or _exact_boundary_quote(
                start_text, want="start"
            )
            start_span, start_projected, _ = _semantic_edge_quote(
                start_text, start_quote, want="start"
            )
            fallback_start_edge = True
            boundary_fallback_reasons.append("unresolved_start_edge_filler")
            if start_span is None:
                report.rejected_reasons.append(f"{prefix}:empty_cue_transcript")
                continue

        if (
            b != selected_end_before_context
            and not trimmed_visual_end_suffix
            and not trimmed_terminal_meta_suffix
            and not completed_forward_sentence
            and not completed_split_caption_tail
        ):
            expanded_end_text = str(segments[b].get("text") or "")
            following_end_text = (
                str(segments[b + 1].get("text") or "")
                if b + 1 < len(segments)
                else ""
            )
            if (
                following_end_text
                and _FORWARD_TOPIC_TRANSITION_RE.match(following_end_text)
            ):
                trimmed_end_text = _TRAILING_TRANSITION_FRAGMENT_RE.sub(
                    "", expanded_end_text
                ).rstrip()
                if _WORD_RE.search(trimmed_end_text):
                    expanded_end_text = trimmed_end_text
            end_quote, edge_error = _expanded_context_edge_quote(
                expanded_end_text, want="end"
            )
            if edge_error:
                end_quote = _exact_boundary_quote(
                    str(segments[b].get("text") or ""), want="end"
                )
                fallback_end_edge = True
                boundary_fallback_reasons.append(edge_error)
            quote_repaired = True
        if fallback_end_edge:
            end_text = str(segments[b].get("text") or "")
            trimmed_quote, _ = _expanded_context_edge_quote(
                end_text, want="end"
            )
            end_quote = trimmed_quote or _exact_boundary_quote(
                end_text, want="end"
            )
        end_quote, final_edge_noise_trimmed = _trim_end_quote_before_edge_noise(
            str(segments[b].get("text") or ""),
            end_quote,
            evidence_quote=evidence_quote_for_section,
            learning_objective=objective_for_section,
        )
        if final_edge_noise_trimmed:
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_trailing_edge_noise")
        end_span, end_projected, edge_error = _semantic_edge_quote(
            str(segments[b].get("text") or ""), end_quote, want="end"
        )
        if edge_error:
            end_quote = _exact_boundary_quote(
                str(segments[b].get("text") or ""), want="end"
            )
            end_span, end_projected, edge_error = _semantic_edge_quote(
                str(segments[b].get("text") or ""), end_quote, want="end"
            )
            fallback_end_edge = True
            boundary_fallback_reasons.append(edge_error or "end_edge_fallback")
        if edge_error or end_span is None:
            report.rejected_reasons.append(f"{prefix}:empty_cue_transcript")
            continue
        assert end_span is not None
        end_text = str(segments[b].get("text") or "")
        end_quote = _literal_source_quote(end_text, end_quote, end_span)
        if _edge_has_unresolved_structural_filler(
            str(segments[b].get("text") or ""), end_span, want="end"
        ):
            trimmed_quote, _ = _expanded_context_edge_quote(
                end_text, want="end"
            )
            end_quote = trimmed_quote or _exact_boundary_quote(
                end_text, want="end"
            )
            end_span, end_projected, _ = _semantic_edge_quote(
                end_text, end_quote, want="end"
            )
            fallback_end_edge = True
            boundary_fallback_reasons.append("unresolved_end_edge_filler")
            if end_span is None:
                report.rejected_reasons.append(f"{prefix}:empty_cue_transcript")
                continue

        clip_text, semantic_spans_by_cue = _semantic_clip_slice(
            segments,
            a,
            b,
            start_span=start_span if start_projected else None,
            end_span=end_span if end_projected else None,
        )
        if not clip_text:
            start_projected = end_projected = False
            clip_text, semantic_spans_by_cue = _semantic_clip_slice(
                segments,
                a,
                b,
                start_span=None,
                end_span=None,
            )
            boundary_fallback_reasons.append("reversed_semantic_boundary")
            if not clip_text:
                report.rejected_reasons.append(f"{prefix}:empty_cue_transcript")
                continue
        same_cue_filler_reason = _same_cue_internal_filler_reason(clip_text)
        if same_cue_filler_reason:
            boundary_fallback_reasons.append(
                f"retained_{same_cue_filler_reason}"
            )
        visual_speech_blocks: list[str] = []
        for segment in segments[a:b + 1]:
            cue_id = str(segment.get("cue_id") or "")
            cue_text = str(segment.get("text") or "")
            semantic_span = semantic_spans_by_cue.get(cue_id)
            if (
                isinstance(semantic_span, tuple)
                and len(semantic_span) == 2
            ):
                cue_text = cue_text[semantic_span[0]:semantic_span[1]]
            if cue_text.strip():
                visual_speech_blocks.append(cue_text.strip())
        if _clip_requires_visual_context(
            clip_text,
            learning_objective=objective_for_section,
            speech_blocks=visual_speech_blocks,
        ):
            report.rejected_reasons.append(f"{prefix}:requires_visual_context")
            continue
        topic_evidence_quote = evidence_quote_for_section
        if isinstance(proposal, (_BoundaryTopic, _CompactBoundaryTopic)):
            grounded_evidence = _grounded_evidence_span_with_minimum_words(
                clip_text,
                topic_evidence_quote,
            )
            if grounded_evidence is None:
                report.rejected_reasons.append(f"{prefix}:ungrounded_topic_evidence_quote")
                continue
            topic_evidence_quote, evidence_span = grounded_evidence
            evidence_word_count = len(_toks(topic_evidence_quote))
            if evidence_word_count < 5 or evidence_word_count > 40:
                report.rejected_reasons.append(f"{prefix}:invalid_topic_evidence_quote_length")
                continue
            topic_evidence_quote = _literal_source_quote(
                clip_text,
                topic_evidence_quote,
                evidence_span,
            )
        intent_role = "primary"
        intent_coverage = 1.0
        grounded_intent_evidence: list[dict[str, str]] = []
        if intent_constraints:
            proposed_intent_evidence = list(
                getattr(proposal, "intent_evidence", None) or []
            )
            evidence_by_constraint: dict[str, str] = {}
            invalid_intent_evidence = False
            for evidence in proposed_intent_evidence:
                constraint_id = " ".join(
                    str(getattr(evidence, "constraint_id", "") or "").split()
                )
                quote = " ".join(
                    str(getattr(evidence, "evidence_quote", "") or "").split()
                )
                if (
                    not constraint_id
                    or constraint_id not in intent_constraint_ids
                    or constraint_id in evidence_by_constraint
                    or not 5 <= len(_toks(quote)) <= 16
                ):
                    invalid_intent_evidence = True
                    break
                evidence_span = _quote_character_span(clip_text, quote)
                if evidence_span is None:
                    invalid_intent_evidence = True
                    break
                evidence_by_constraint[constraint_id] = _literal_source_quote(
                    clip_text,
                    quote,
                    evidence_span,
                )
            if invalid_intent_evidence or not evidence_by_constraint:
                report.rejected_reasons.append(f"{prefix}:invalid_intent_evidence")
                continue
            fulfilled_ids = set(evidence_by_constraint)
            intent_role = (
                "primary"
                if fulfilled_ids == intent_constraint_ids
                else "supporting"
            )
            intent_coverage = len(fulfilled_ids) / max(1, len(intent_constraint_ids))
            grounded_intent_evidence = [
                {
                    "constraint_id": constraint_id,
                    "evidence_quote": evidence_by_constraint[constraint_id],
                }
                for constraint_id in intent_constraints
                if constraint_id in evidence_by_constraint
            ]
        learning_objective = str(
            getattr(proposal, "learning_objective", "")
            or getattr(proposal, "reason", "")
            or proposal.title
        ).strip()
        clip_title = str(proposal.title or "").strip()
        clip_facet = str(proposal.facet or "").strip()
        clip_reason = str(
            getattr(proposal, "reason", "") or learning_objective
        ).strip()
        if context_was_trimmed:
            repair_source_text = context_repair_source_text
            require_grounding = context_was_trimmed
            learning_objective = _objective_after_range_repair(
                learning_objective,
                original_text=repair_source_text,
                retained_text=clip_text,
                evidence_quote=topic_evidence_quote,
                require_grounding=require_grounding,
            )
            clip_title = _title_after_range_repair(
                clip_title,
                original_text=repair_source_text,
                retained_text=clip_text,
                evidence_quote=topic_evidence_quote,
                require_grounding=require_grounding,
            )
            clip_facet = _title_after_range_repair(
                clip_facet,
                original_text=repair_source_text,
                retained_text=clip_text,
                evidence_quote=topic_evidence_quote,
                require_grounding=require_grounding,
            )
            clip_reason = _objective_after_range_repair(
                clip_reason,
                original_text=repair_source_text,
                retained_text=clip_text,
                evidence_quote=topic_evidence_quote,
                require_grounding=require_grounding,
            )
        cue_ids = [
            str(segments[line].get("cue_id") or f"cue-{line}")
            for line in range(a, b + 1)
        ]
        edge_projection: dict[str, dict[str, object]] = {}
        if start_projected:
            edge_projection["start"] = {
                "required": True,
                "cue_id": cue_ids[0],
                "quote": start_quote,
            }
        if end_projected:
            edge_projection["end"] = {
                "required": True,
                "cue_id": cue_ids[-1],
                "quote": end_quote,
            }
        clip_id = f"clip-{index + 1:03d}-{a}-{b}"
        clip = {
            "start": start,
            "end": end,
            "start_quote": start_quote,
            "end_quote": end_quote,
            "title": clip_title,
            "learning_objective": learning_objective,
            "facet": clip_facet,
            "reason": clip_reason,
            "kind": "educational",
            "informativeness": info,
            "topic_relevance": relevance,
            "self_contained": True,
            "difficulty": difficulty,
            "educational_importance": importance,
            "boundary_confidence": _cue_boundary_confidence(
                str(segments[b].get("text") or ""),
                ignore_caption_case=ignore_caption_case,
            ),
            "is_standalone": is_standalone,
            "chain_id": "",
            "chain_position": 0,
            "prerequisite_ids": prerequisites,
            "cue_ids": cue_ids,
            "start_cue_id": cue_ids[0],
            "end_cue_id": cue_ids[-1],
            "selection_candidate_id": candidate_id,
            "uncertainty": uncertainty,
            "uncertainty_reasons": uncertainty_reasons,
            "_start_line": a,
            "_end_line": b,
            "_clip_id": clip_id,
            "_clip_text": clip_text,
            "_proposal_index": index,
            "_semantic_spans_by_cue": semantic_spans_by_cue,
            "_quote_repaired": quote_repaired,
            "_boundary_fallback_reasons": list(
                dict.fromkeys(boundary_fallback_reasons)
            ),
            "directly_teaches_topic": bool(
                getattr(proposal, "directly_teaches_topic", True)
            ),
            "substantive": bool(getattr(proposal, "substantive", True)),
            "factually_grounded": bool(getattr(proposal, "factually_grounded", True)),
            "topic_evidence_quote": topic_evidence_quote,
            "intent_role": intent_role,
            "intent_coverage": round(intent_coverage, 6),
            "intent_evidence": grounded_intent_evidence,
            "summary": "",
            "takeaways": [],
            "match_reason": "",
            "assessment": None,
        }
        if edge_projection:
            clip["edge_projection"] = edge_projection
        if hasattr(proposal, "summary"):
            details, errors = _learning_details(proposal, clip_text, topic)
            clip.update(details)
            if errors:
                report.enrichment_errors.extend(f"{clip_id}:{error}" for error in errors)
        elif require_enrichment:
            report.enrichment_errors.append(f"{clip_id}:missing_enrichment")
        raw.append(clip)

    by_candidate_id = {
        str(clip["selection_candidate_id"]): clip for clip in raw
    }
    depended_on = {
        prerequisite
        for clip in raw
        for prerequisite in (clip.get("prerequisite_ids") or [])
    }

    def chain_root_and_depth(candidate_id: str, trail: set[str]) -> tuple[str, int]:
        if candidate_id in trail:
            return candidate_id, 0
        clip = by_candidate_id.get(candidate_id)
        prerequisites = list((clip or {}).get("prerequisite_ids") or [])
        if not prerequisites:
            return candidate_id, 0
        roots_and_depths = [
            chain_root_and_depth(prerequisite, {*trail, candidate_id})
            for prerequisite in prerequisites
        ]
        root = sorted(value[0] for value in roots_and_depths)[0]
        return root, max(value[1] for value in roots_and_depths) + 1

    for clip in raw:
        candidate_id = str(clip["selection_candidate_id"])
        if clip.get("prerequisite_ids") or candidate_id in depended_on:
            root, depth = chain_root_and_depth(candidate_id, set())
            clip["chain_id"] = f"chain:{root}"
            clip["chain_position"] = depth

    # Detect duplicates before removing them so classification cannot turn green by repair.
    report.medium_uncertainty = any(
        clip.get("uncertainty") == "medium" for clip in raw
    )
    report.score_below_green = any(
        min(
            float(clip["informativeness"]),
            float(clip["topic_relevance"]),
            float(clip["educational_importance"]),
        )
        < _GREEN_SCORE
        for clip in raw
    )
    for i, candidate in enumerate(raw):
        if any(_duplicates(candidate, other) for other in raw[i + 1:]):
            report.near_duplicate = True
            break

    report.clips = _finalize_clips(raw, settings)
    return report


def _public_clips(clips: list[dict]) -> list[dict]:
    return [{key: value for key, value in clip.items() if not key.startswith("_")}
            for clip in clips]


def _plan_to_clips(plan: _Plan | _BoundaryPlan | _IntentBoundaryPlan | _LegacyPlan | _ProductionPlan,
                   segments: list[dict],
                   words: list[dict], settings: dict) -> list[dict]:
    """Compatibility helper used by focused conversion tests."""
    return _public_clips(_plan_to_report(plan, segments, words, settings).clips)


@dataclass(frozen=True)
class _Classification:
    status: Literal["green", "uncertain", "invalid"]
    reasons: tuple[str, ...]


def _transcript_duration(segments: list[dict]) -> float:
    if not segments:
        return 0.0
    starts = [float(segment.get("start", 0.0)) for segment in segments]
    ends = [float(segment.get("end", start)) for segment, start in zip(segments, starts)]
    return max(ends) - min(starts)


def _classify_flash(report: _Conversion, segments: list[dict], topic: str,
                    *, enrichment_required: bool) -> _Classification:
    del segments, topic, enrichment_required
    if report.accepted_count:
        if report.score_below_green:
            return _Classification("invalid", ("quality_score_below_green",))
        return _Classification("green", ())
    reasons = list(report.rejected_reasons) or ["zero_valid_candidates"]
    return _Classification("invalid", tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# Gemini calls, enrichment, routing, and telemetry

@dataclass
class SegmentResult:
    clips: list[dict]
    notes: str
    route: str
    classification: str
    classification_reasons: list[str] = field(default_factory=list)
    fallback_reasons: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    proposed_count: int = 0
    accepted_count: int = 0
    error: str | None = None
    flash_configuration_error: str | None = None
    rejection_reasons: list[str] = field(default_factory=list)


class _SchemaResponseError(RuntimeError):
    def __init__(self, message: str, telemetry: object):
        super().__init__(message)
        self.telemetry = telemetry


class _ModelCallError(RuntimeError):
    def __init__(self, message: str, telemetry: dict):
        super().__init__(message)
        self.telemetry = telemetry


def _telemetry_dict(value: object) -> dict:
    if value is None:
        return {}
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {"value": str(value)}


def _validate_model_response(
    schema: type[BaseModel], text: str,
) -> tuple[BaseModel, list[str]]:
    """Validate one response, salvaging valid boundary candidates independently."""
    if schema not in {_BoundaryPlan, _CompactBoundaryPlan, _IntentBoundaryPlan}:
        return schema.model_validate_json(text), []

    payload = json.loads(text)
    expected_keys = (
        {"request_intent", "topics"}
        if schema in {_CompactBoundaryPlan, _IntentBoundaryPlan}
        else {"topics"}
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or not isinstance(payload.get("topics"), list)
    ):
        return schema.model_validate(payload), []

    request_intent: _RequestIntent | None = None
    topic_schema: type[BaseModel] = (
        _CompactBoundaryTopic if schema is _CompactBoundaryPlan else _BoundaryTopic
    )
    if schema in {_CompactBoundaryPlan, _IntentBoundaryPlan}:
        request_intent = _RequestIntent.model_validate(payload["request_intent"])
    if schema is _IntentBoundaryPlan:
        topic_schema = _IntentBoundaryTopic

    topics: list[_BoundaryTopic] = []
    rejection_reasons: list[str] = []
    for index, raw_topic in enumerate(payload["topics"]):
        try:
            topics.append(topic_schema.model_validate(raw_topic))
        except ValidationError as exc:
            first_error = exc.errors(include_url=False)[0]
            location = ".".join(str(part) for part in first_error.get("loc", ()))
            rejection_reasons.append(
                f"proposal_{index}:schema_invalid:{location or 'item'}:"
                f"{first_error.get('type') or 'validation_error'}"
            )
    if schema is _IntentBoundaryPlan:
        assert request_intent is not None
        return _IntentBoundaryPlan(
            request_intent=request_intent,
            topics=topics,
        ), rejection_reasons
    if schema is _CompactBoundaryPlan:
        assert request_intent is not None
        return _CompactBoundaryPlan(
            request_intent=request_intent,
            topics=topics,
        ), rejection_reasons
    return _BoundaryPlan(topics=topics), rejection_reasons


def _acquire_selector_slot(
    *,
    operation: str,
    model: str,
    thinking_level: str,
    prompt_version: str,
    deadline_monotonic: float,
    cancelled: CancelledCb,
) -> BoundedSemaphore | None:
    """Bound selector dispatches across jobs without hiding cancellation."""
    if str(operation or "").casefold() not in _SELECTOR_OPERATIONS:
        return None

    slot = _selector_call_slots
    while True:
        if _cancel_requested(cancelled):
            raise _ModelCallError(
                "Gemini selector capacity wait cancelled",
                {
                    "model": model,
                    "operation": operation,
                    "prompt_version": prompt_version,
                    "thinking_level": thinking_level,
                    "error_type": "GeminiCancelledError",
                    "dispatched": False,
                },
            )
        remaining_s = float(deadline_monotonic) - time.monotonic()
        if remaining_s <= 0:
            raise _ModelCallError(
                "Gemini selector capacity deadline exceeded",
                {
                    "model": model,
                    "operation": operation,
                    "prompt_version": prompt_version,
                    "thinking_level": thinking_level,
                    "error_type": "GeminiDeadlineExceededError",
                    "dispatched": False,
                },
            )
        if slot.acquire(timeout=min(_SELECTOR_SLOT_POLL_S, remaining_s)):
            return slot


def _telemetry_error_history(telemetry: dict) -> tuple[dict, ...]:
    raw = telemetry.get("error_history")
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(dict(item) for item in raw if isinstance(item, dict))


def _flash_failover_reason(
    telemetry: dict,
    *,
    primary_exception: Exception,
    primary_model: str,
    failover_model: str | None,
    operation: str,
    deadline_monotonic: float,
    cancelled: CancelledCb,
) -> str | None:
    primary = str(primary_model or "").strip().casefold()
    failover = str(failover_model or "").strip().casefold()
    primary_leaf = primary.rsplit("/", 1)[-1]
    failover_leaf = failover.rsplit("/", 1)[-1]
    configured_primary_leaf = (
        str(config.SEGMENT_FLASH_MODEL).strip().casefold().rsplit("/", 1)[-1]
    )
    history = _telemetry_error_history(telemetry)
    try:
        retries = int(telemetry.get("retries") or 0)
    except (TypeError, ValueError, OverflowError):
        retries = 0
    if not (
        operation == "flash_boundary_selector"
        and type(primary_exception).__name__ == "GeminiTransportError"
        and primary_leaf == configured_primary_leaf
        and primary_leaf == "gemini-3.5-flash"
        and primary_leaf != failover_leaf
        and re.fullmatch(
            r"gemini-3(?:\.\d+)?-flash-lite", failover_leaf
        ) is not None
        and telemetry.get("retryable") is True
        and telemetry.get("dispatched", True) is not False
        and not _cancel_requested(cancelled)
        and float(deadline_monotonic) - time.monotonic() >= 5.0
    ):
        return None
    statuses = tuple(item.get("provider_status_code") for item in history)
    if (
        retries == 0
        and statuses in {(500,), (502,), (504,)}
        and telemetry.get("provider_status_code") == statuses[-1]
        and history[0].get("retryable") is True
    ):
        return "primary_transient_5xx_failover"
    if (
        retries == 1
        and len(statuses) == 2
        and statuses[0] == 503
        and statuses[1] in {500, 502, 503, 504}
        and telemetry.get("provider_status_code") == statuses[-1]
        and all(item.get("retryable") is True for item in history)
    ):
        return "primary_503_retry_exhausted"
    return None


def _merge_failover_telemetry(
    primary: dict,
    failover: dict,
    *,
    primary_model: str,
    failover_model: str,
    failover_reason: str,
    started: float,
) -> dict:
    merged = dict(failover)
    try:
        primary_retries = int(primary.get("retries") or 0)
    except (TypeError, ValueError, OverflowError):
        primary_retries = 0
    try:
        failover_retries = int(failover.get("retries") or 0)
    except (TypeError, ValueError, OverflowError):
        failover_retries = 0
    merged.update({
        "model": str(failover.get("model") or failover_model),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "retries": primary_retries + 1 + failover_retries,
        "error_history": (
            _telemetry_error_history(primary)
            + _telemetry_error_history(failover)
        ),
        "failover_from_model": str(primary_model),
        "failover_model": str(failover_model),
        "failover_reason": failover_reason,
        "quality_degraded": True,
        "dispatched": True,
    })
    return merged


def _call_model(
    system: str,
    user: str,
    schema: type[BaseModel],
    *,
    model: str,
    thinking_level: str,
    max_output_tokens: int,
    timeout_s: float,
    deadline_monotonic: float,
    operation: str,
    prompt_version: str,
    cancelled: CancelledCb,
    budget_reserve: Optional[Callable[..., object]] = None,
    budget_reconcile: Optional[Callable[..., object]] = None,
    max_retries: int = 1,
    retry_status_codes: frozenset[int] | set[int] | None = None,
    failover_model: str | None = None,
) -> tuple[BaseModel, dict]:
    from ..gemini_client import generate_json_v3

    prompt_text = f"{system}\n\n{user}"
    selector_slot = _acquire_selector_slot(
        operation=operation,
        model=model,
        thinking_level=thinking_level,
        prompt_version=prompt_version,
        deadline_monotonic=deadline_monotonic,
        cancelled=cancelled,
    )
    try:
        reservation: dict[str, object] = {}
        if callable(budget_reserve):
            reserved = budget_reserve(
                operation=operation,
                model=model,
                max_output_tokens=max_output_tokens,
                prompt_text=prompt_text,
                estimated_input_tokens=max(
                    1,
                    math.ceil(len(prompt_text) / 3) + 1_000,
                ),
                deadline_monotonic=deadline_monotonic,
                cancelled=cancelled,
            )
            if isinstance(reserved, dict):
                reservation = dict(reserved)
        call_started = time.perf_counter()
        successful_telemetry: dict | None = None
        failure_telemetry_override: dict | None = None
        try:
            try:
                result = generate_json_v3(
                    system,
                    user,
                    schema,
                    model=model,
                    thinking_level=thinking_level,
                    max_output_tokens=max_output_tokens,
                    timeout_s=timeout_s,
                    deadline_monotonic=deadline_monotonic,
                    operation=operation,
                    prompt_version=prompt_version,
                    max_retries=max_retries,
                    retry_status_codes=retry_status_codes,
                    cancelled=cancelled,
                )
            except Exception as primary_exc:
                primary_telemetry = _telemetry_dict(
                    getattr(primary_exc, "telemetry", None)
                )
                failover_reason = _flash_failover_reason(
                    primary_telemetry,
                    primary_exception=primary_exc,
                    primary_model=model,
                    failover_model=failover_model,
                    operation=operation,
                    deadline_monotonic=deadline_monotonic,
                    cancelled=cancelled,
                )
                if failover_reason is None:
                    raise
                try:
                    result = generate_json_v3(
                        system,
                        user,
                        schema,
                        model=str(failover_model),
                        thinking_level=thinking_level,
                        max_output_tokens=max_output_tokens,
                        timeout_s=timeout_s,
                        deadline_monotonic=deadline_monotonic,
                        operation=operation,
                        prompt_version=prompt_version,
                        max_retries=0,
                        cancelled=cancelled,
                    )
                except Exception as failover_exc:
                    failure_telemetry_override = _merge_failover_telemetry(
                        primary_telemetry,
                        _telemetry_dict(getattr(failover_exc, "telemetry", None)),
                        primary_model=model,
                        failover_model=str(failover_model),
                        failover_reason=failover_reason,
                        started=call_started,
                    )
                    raise
                successful_telemetry = _merge_failover_telemetry(
                    primary_telemetry,
                    _telemetry_dict(result.telemetry),
                    primary_model=model,
                    failover_model=str(failover_model),
                    failover_reason=failover_reason,
                    started=call_started,
                )
        except Exception as exc:
            provider_telemetry = (
                failure_telemetry_override
                or _telemetry_dict(getattr(exc, "telemetry", None))
            )
            provider_dispatched = provider_telemetry.get("dispatched", True)
            dispatched = (
                provider_dispatched
                if isinstance(provider_dispatched, bool)
                else True
            )
            failure_telemetry = {
                "model": model,
                "operation": operation,
                "prompt_version": prompt_version,
                "thinking_level": thinking_level,
                **provider_telemetry,
                **reservation,
                "error_type": type(exc).__name__,
                "dispatched": dispatched,
            }
            if callable(budget_reconcile):
                try:
                    budget_reconcile(
                        model_used=str(failure_telemetry.get("model") or model),
                        usage=failure_telemetry,
                        dispatched=dispatched,
                    )
                except Exception:
                    log.warning("Gemini budget reconciliation failed", exc_info=True)
            if _cancel_requested(cancelled):
                raise
            raise _ModelCallError(
                f"{type(exc).__name__}: Gemini model call failed",
                failure_telemetry,
            ) from exc
        telemetry = successful_telemetry or _telemetry_dict(result.telemetry)
        for key, value in reservation.items():
            telemetry.setdefault(key, value)
        telemetry.setdefault("dispatched", True)
        if callable(budget_reconcile):
            try:
                budget_reconcile(
                    model_used=str(telemetry.get("model") or model),
                    usage=telemetry,
                    dispatched=True,
                )
            except Exception:
                log.warning("Gemini budget reconciliation failed", exc_info=True)
        try:
            parsed, schema_rejections = _validate_model_response(
                schema, result.text.strip(),
            )
        except (ValidationError, ValueError) as exc:
            raise _SchemaResponseError(
                f"invalid {schema.__name__} response: {exc}", telemetry,
            ) from exc
        if schema_rejections:
            telemetry["schema_rejected_count"] = len(schema_rejections)
            telemetry["schema_rejection_reasons"] = schema_rejections
        return parsed, telemetry
    finally:
        if selector_slot is not None:
            selector_slot.release()


def _exception_telemetry(exc: Exception) -> dict:
    return _telemetry_dict(getattr(exc, "telemetry", None))


def _model_cost(call: dict) -> float:
    model = str(call.get("model") or "").lower()
    if "flash-lite" in model:
        tier = "flash_lite"
    elif "gemini-3-flash" in model and "gemini-3.5-flash" not in model:
        tier = "flash_preview"
    else:
        tier = "flash" if "flash" in model else "pro"
    rates = _PRICING_PER_MILLION[tier]
    prompt = int(call.get("prompt_tokens") or call.get("prompt_token_count") or 0)
    candidate = int(
        call.get("candidate_tokens") or call.get("candidates_token_count") or 0
    )
    thought = int(call.get("thought_tokens") or call.get("thoughts_token_count") or 0)
    return (prompt * rates["input"] + (candidate + thought) * rates["output"]) / 1_000_000.0


def _emit(sink: Optional[Callable[[dict], None]], event: str, **fields) -> None:
    payload = {"event": event, **fields}
    log.info("segment_event %s", json.dumps(payload, sort_keys=True, default=str))
    if sink is not None:
        try:
            sink(payload)
        except Exception:  # telemetry must never fail segmentation
            log.warning("segment telemetry sink failed", exc_info=True)


def _cancel_requested(cancelled: object) -> bool:
    if cancelled is None:
        return False
    if callable(cancelled):
        return bool(cancelled())
    is_set = getattr(cancelled, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancelled)


def _lines(segments: list[dict]) -> str:
    return "\n".join(
        f"[{index}] {_mmss(segment.get('start', 0.0))} "
        f"{str(segment.get('text') or '').strip()}"
        for index, segment in enumerate(segments)
    )


def _repair_failed_boundaries(
    report: _Conversion,
    segments: list[dict],
    words: list[dict],
    topic: str,
    settings: dict,
    *,
    deadline: float,
    cancelled: CancelledCb,
) -> list[dict]:
    """Run one localized Flash batch and merge only independently valid repairs."""
    candidates = report.repair_candidates[:_MAX_CLIPS]
    configured_limit = _configured_clip_limit(settings)
    if (
        not candidates
        or (
            configured_limit is not None
            and report.accepted_count >= configured_limit
        )
    ):
        return []

    system, user, allowed = _boundary_repair_prompts(candidates, segments, topic)
    sink = settings.get("_segment_telemetry")
    try:
        plan, call = _call_model(
            system,
            user,
            _BoundaryRepairPlan,
            model=config.SEGMENT_FLASH_MODEL,
            thinking_level="low",
            max_output_tokens=_BOUNDARY_REPAIR_OUTPUT_TOKENS,
            timeout_s=_FLASH_REPAIR_TIMEOUT_S,
            deadline_monotonic=deadline,
            operation="flash_boundary_repair",
            prompt_version=_BOUNDARY_REPAIR_PROMPT_VERSION,
            cancelled=cancelled,
            budget_reserve=settings.get("_segment_budget_reserve"),
            budget_reconcile=settings.get("_segment_budget_reconcile"),
        )
        calls = [call]
    except Exception as exc:
        telemetry = _exception_telemetry(exc)
        calls = [telemetry] if telemetry else []
        report.rejected_reasons.append(
            f"boundary_repair_request_failure:{type(exc).__name__}"
        )
        _emit(
            sink,
            "boundary_repair",
            attempted_count=len(candidates),
            accepted_count=0,
            reason=f"request_failure:{type(exc).__name__}",
        )
        return calls

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    seen: set[str] = set()
    repaired: list[dict] = []
    for item in plan.items:
        candidate = by_id.get(item.candidate_id)
        if candidate is None:
            report.rejected_reasons.append(
                f"boundary_repair:unknown_candidate_id:{item.candidate_id}"
            )
            continue
        if item.candidate_id in seen:
            report.rejected_reasons.append(f"{candidate.prefix}:duplicate_repair")
            continue
        seen.add(item.candidate_id)
        allowed_starts, allowed_ends = allowed[item.candidate_id]
        if item.start_line not in allowed_starts or item.end_line not in allowed_ends:
            report.rejected_reasons.append(f"{candidate.prefix}:repair_outside_neighbors")
            continue
        if item.end_line < item.start_line:
            report.rejected_reasons.append(f"{candidate.prefix}:repair_reversed_range")
            continue

        repaired_proposal = candidate.proposal.model_copy(update={
            "start_line": item.start_line,
            "end_line": item.end_line,
            "start_quote": item.start_quote,
            "end_quote": item.end_quote,
        })
        repaired_report = _plan_to_report(
            _BoundaryPlan(topics=[repaired_proposal]),
            segments,
            words,
            settings,
            topic=topic,
            context_cue_limit=0,
        )
        if repaired_report.accepted_count != 1:
            reasons = repaired_report.rejected_reasons or ["invalid_boundary"]
            for reason in reasons:
                suffix = reason.split(":", 1)[-1]
                report.rejected_reasons.append(
                    f"{candidate.prefix}:repair_{suffix}"
                )
            continue

        clip = repaired_report.clips[0]
        clip["boundary_confidence"] = 0.85
        clip["selection_candidate_id"] = candidate.candidate_id
        clip["_clip_id"] = candidate.candidate_id
        repaired.append(clip)
        original_rejection = f"{candidate.prefix}:{candidate.reason}"
        if original_rejection in report.rejected_reasons:
            report.rejected_reasons.remove(original_rejection)

    for candidate in candidates:
        if candidate.candidate_id not in seen:
            report.rejected_reasons.append(f"{candidate.prefix}:repair_omitted")

    report.clips = _finalize_clips([*report.clips, *repaired], settings)
    _emit(
        sink,
        "boundary_repair",
        attempted_count=len(candidates),
        accepted_count=len(repaired),
        rejected_count=max(0, len(candidates) - len(repaired)),
    )
    return calls


def _run_selection_profile(
    profile: str,
    transcript: dict,
    topic: str,
    settings: dict,
    *,
    deadline: float,
    cancelled: CancelledCb,
) -> tuple[_Conversion, _Classification, list[dict]]:
    segments = transcript.get("segments") or []
    words = transcript.get("words") or []
    rendered = _lines(segments)
    learner_level = str(
        settings.get("_knowledge_level")
        or settings.get("knowledge_level")
        or settings.get("learner_level")
        or ""
    )
    if profile == PRODUCTION_PRO_PROFILE:
        system, user = _legacy_prompts(rendered, len(segments), topic)
        schema: type[BaseModel] = _LegacyPlan
        model = config.SEGMENT_PRO_MODEL
        level, cap, timeout = "high", _SELECTION_OUTPUT_TOKENS, _PRO_TIMEOUT_S
        operation = "pro_authoritative"
    elif profile == CORRECTED_PRO_PROFILE:
        system, user = _prompts(
            rendered, len(segments), topic, learner_level=learner_level,
        )
        schema = _Plan
        model = config.SEGMENT_PRO_MODEL
        level, cap, timeout = "high", _SELECTION_OUTPUT_TOKENS, _PRO_TIMEOUT_S
        operation = "pro_fallback"
    elif profile == FLASH_SINGLE_PROFILE:
        system, user = _prompts(
            rendered, len(segments), topic, learner_level=learner_level,
        )
        schema = _Plan
        model = config.SEGMENT_FLASH_MODEL
        level, cap, timeout = "medium", _SELECTION_OUTPUT_TOKENS, _FLASH_SINGLE_TIMEOUT_S
        operation = "flash_single_candidate"
    elif profile == FLASH_SPLIT_PROFILE:
        system, user = _boundary_prompts(
            rendered,
            len(segments),
            topic,
            learner_level=learner_level,
        )
        schema = _CompactBoundaryPlan
        model = config.SEGMENT_FLASH_MODEL
        level, cap, timeout = "low", _BOUNDARY_OUTPUT_TOKENS, _FLASH_BOUNDARY_TIMEOUT_S
        operation = "flash_boundary_selector"
    elif profile == PRO_BOUNDARY_PROFILE:
        system, user = _boundary_prompts(
            rendered,
            len(segments),
            topic,
            learner_level=learner_level,
        )
        schema = _CompactBoundaryPlan
        model = config.SEGMENT_PRO_MODEL
        level, cap, timeout = "high", _BOUNDARY_OUTPUT_TOKENS, _PRO_TIMEOUT_S
        operation = "pro_fallback"
    else:
        raise ValueError(f"unknown segmentation profile: {profile}")

    if profile == FLASH_SPLIT_PROFILE:
        requested_level = str(
            settings.get("_segment_thinking_level") or level
        ).strip().lower()
        if requested_level in {"minimal", "low"}:
            level = requested_level
    operation = str(settings.get("_segment_operation") or operation)
    parsed, call = _call_model(
        system,
        user,
        schema,
        model=model,
        thinking_level=level,
        max_output_tokens=cap,
        timeout_s=timeout,
        deadline_monotonic=deadline,
        operation=operation,
        prompt_version=profile,
        cancelled=cancelled,
        budget_reserve=settings.get("_segment_budget_reserve"),
        budget_reconcile=settings.get("_segment_budget_reconcile"),
        # Healthy sources still use one physical request. The active Flash
        # selector gets one deadline-aware retry so a brief provider-capacity
        # failure cannot discard an otherwise usable source.
        max_retries=1 if profile == FLASH_SPLIT_PROFILE else 0,
        retry_status_codes=(
            frozenset({503}) if profile == FLASH_SPLIT_PROFILE else None
        ),
        failover_model=(
            config.SEGMENT_FLASH_FALLBACK_MODEL
            if profile == FLASH_SPLIT_PROFILE
            else None
        ),
    )
    require_enrichment = profile in {CORRECTED_PRO_PROFILE, FLASH_SINGLE_PROFILE}
    conversion_settings = dict(settings)
    conversion_settings.setdefault(
        "_segment_ignore_caption_case",
        str(transcript.get("source") or "").casefold() == "supadata",
    )
    if profile in {PRODUCTION_FLASH_PROFILE, PRO_BOUNDARY_PROFILE}:
        # Cache and persist the complete selector result. Public request ceilings
        # apply only when a feed inventory is surfaced, never during selection.
        conversion_settings.pop("max_clips", None)
    report = _plan_to_report(
        parsed,
        segments,
        words,
        conversion_settings,
        topic=topic,
        require_enrichment=require_enrichment,
    )
    schema_rejections = call.get("schema_rejection_reasons")
    if isinstance(schema_rejections, list):
        clean_rejections = [
            str(reason) for reason in schema_rejections if str(reason).strip()
        ]
        report.proposed_count += len(clean_rejections)
        report.rejected_reasons = clean_rejections + report.rejected_reasons
    calls = [call]
    if profile in {FLASH_SPLIT_PROFILE, PRO_BOUNDARY_PROFILE}:
        _drop_unmet_prerequisite_clips(report)
    if profile.startswith("flash_"):
        classification = _classify_flash(
            report, segments, topic, enrichment_required=(profile == FLASH_SINGLE_PROFILE),
        )
    else:
        classification = _Classification(
            "green" if report.clips else "invalid",
            () if report.clips else tuple(dict.fromkeys(report.rejected_reasons)),
        )
    return report, classification, calls


def _apply_enrichment(clips: list[dict], plan: _EnrichmentPlan, topic: str) -> list[str]:
    by_id: dict[str, _EnrichmentItem] = {}
    duplicate_ids: set[str] = set()
    for item in plan.items:
        if item.clip_id in by_id:
            duplicate_ids.add(item.clip_id)
        by_id[item.clip_id] = item
    errors: list[str] = []
    expected = {clip["_clip_id"] for clip in clips}
    if set(by_id) - expected:
        errors.extend(f"unknown_clip_id:{clip_id}" for clip_id in sorted(set(by_id) - expected))
    errors.extend(f"duplicate_clip_id:{clip_id}" for clip_id in sorted(duplicate_ids))
    for clip in clips:
        item = by_id.get(clip["_clip_id"])
        if item is None:
            errors.append(f"missing_clip_id:{clip['_clip_id']}")
            continue
        details, item_errors = _learning_details(item, clip["_clip_text"], topic)
        clip.update(details)
        if item_errors:
            errors.extend(f"{clip['_clip_id']}:{error}" for error in item_errors)
            continue
    return errors


def _invalid_enrichment_clip_ids(errors: list[str], clips: list[dict]) -> set[str]:
    ids = {clip["_clip_id"] for clip in clips}
    invalid: set[str] = set()
    for error in errors:
        for clip_id in ids:
            if clip_id in error:
                invalid.add(clip_id)
    if any(error.startswith(("unknown_clip_id:", "duplicate_clip_id:")) for error in errors):
        invalid.update(ids)
    return invalid


def enrich_accepted_clips(
    items: list[dict],
    *,
    topic: str,
    settings: dict | None = None,
    deadline_monotonic: float | None = None,
    cancelled: CancelledCb = None,
) -> tuple[dict[str, dict], list[dict]]:
    """Enrich at most three persisted clips without affecting clip validity."""
    batch = [dict(item) for item in items[:3] if str(item.get("clip_id") or "").strip()]
    if not batch:
        return {}, []
    source_by_id = {
        str(item["clip_id"]): str(item.get("text") or "").strip()
        for item in batch
    }
    system, user = _card_enrichment_prompts(batch, topic)
    try:
        plan, call = _call_model(
            system,
            user,
            _CardEnrichmentPlan,
            model=config.SEGMENT_FLASH_MODEL,
            thinking_level="low",
            max_output_tokens=_ENRICH_OUTPUT_TOKENS,
            timeout_s=_FLASH_ENRICH_TIMEOUT_S,
            deadline_monotonic=(
                deadline_monotonic or (time.monotonic() + _FLASH_ENRICH_TIMEOUT_S)
            ),
            operation="flash_grounded_enrichment",
            prompt_version=_CARD_ENRICHMENT_PROMPT_VERSION,
            cancelled=cancelled,
            budget_reserve=(settings or {}).get("_segment_budget_reserve"),
            budget_reconcile=(settings or {}).get("_segment_budget_reconcile"),
        )
        calls = [call]
    except Exception as exc:
        telemetry = _exception_telemetry(exc)
        return {}, [telemetry] if telemetry else []

    enriched: dict[str, dict] = {}
    seen: set[str] = set()
    topic_tokens = _content_tokens(topic)
    for item in plan.items:
        clip_id = str(item.clip_id)
        grounding_text = source_by_id.get(clip_id, "")
        if not grounding_text or clip_id in seen:
            continue
        seen.add(clip_id)
        summary = " ".join(item.summary.split())
        takeaways = [" ".join(value.split()) for value in item.takeaways]
        match_reason = " ".join(item.match_reason.split())
        if not _text_has_grounding(summary, grounding_text):
            continue
        if any(not _text_has_grounding(value, grounding_text) for value in takeaways):
            continue
        if not _text_has_grounding(match_reason, grounding_text):
            continue
        if topic_tokens and not topic_tokens.intersection(_content_tokens(match_reason)):
            continue
        enriched[clip_id] = {
            "summary": summary[:700],
            "takeaways": takeaways[:4],
            "match_reason": match_reason[:700],
        }
    return enriched, calls


def _enrich_split(
    clips: list[dict],
    topic: str,
    settings: dict | None = None,
    *,
    deadline: float,
    cancelled: CancelledCb,
) -> tuple[list[dict], list[dict], list[str], str | None]:
    calls: list[dict] = []
    system, user = _enrichment_prompts(clips, topic)
    try:
        plan, call = _call_model(
            system,
            user,
            _EnrichmentPlan,
            model=config.SEGMENT_FLASH_MODEL,
            thinking_level="low",
            max_output_tokens=_ENRICH_OUTPUT_TOKENS,
            timeout_s=_FLASH_ENRICH_TIMEOUT_S,
            deadline_monotonic=deadline,
            operation="flash_grounded_enrichment",
            prompt_version=FLASH_SPLIT_PROFILE,
            cancelled=cancelled,
            budget_reserve=(settings or {}).get("_segment_budget_reserve"),
            budget_reconcile=(settings or {}).get("_segment_budget_reconcile"),
        )
        calls.append(call)
        errors = _apply_enrichment(clips, plan, topic)
    except Exception as exc:  # schema/transport failure makes every enrichment item invalid
        telemetry = _exception_telemetry(exc)
        if telemetry:
            calls.append(telemetry)
        errors = [f"{clip['_clip_id']}:flash_enrichment_failure" for clip in clips]

    invalid_ids = _invalid_enrichment_clip_ids(errors, clips)
    for clip in clips:
        if clip["_clip_id"] in invalid_ids:
            clip.update({
                "summary": "",
                "takeaways": [],
                "match_reason": "",
                "assessment": None,
            })
    # Learning details are optional. A valid boundary selection never incurs a
    # slower Pro retry merely because enrichment was absent or malformed.
    return clips, calls, [], None


def run_segment_profile(
    transcript: dict,
    settings: dict,
    profile: str,
    *,
    topic: str = "",
    deadline_monotonic: float | None = None,
    cancelled: CancelledCb = None,
) -> SegmentResult:
    """Run one immutable benchmark profile without production routing."""
    segments = transcript.get("segments") or []
    if not segments:
        return SegmentResult([], "No transcript segments to segment.", profile, "invalid",
                             ["missing_segments"], proposed_count=0, accepted_count=0)
    deadline = deadline_monotonic or (time.monotonic() + _TOTAL_DEADLINE_S)
    try:
        report, classification, calls = _run_selection_profile(
            profile, transcript, topic, settings, deadline=deadline, cancelled=cancelled,
        )
        fallback_reasons: list[str] = []
        flash_configuration_error: str | None = None
        clips = _public_clips(report.clips)
        notes = f"{len(clips)} topic clip(s) from {len(segments)} transcript segments."
        return SegmentResult(
            clips,
            notes,
            profile,
            classification.status,
            list(classification.reasons),
            fallback_reasons,
            calls,
            report.proposed_count,
            len(clips),
            flash_configuration_error=flash_configuration_error,
            rejection_reasons=list(report.rejected_reasons),
        )
    except Exception as exc:  # callers decide whether an invalid profile should fall back
        if _cancel_requested(cancelled):
            raise
        call = _exception_telemetry(exc)
        calls = [call] if call else []
        error_type = str(call.get("error_type") or type(exc).__name__)
        failure_reason = f"request_failure:{error_type}"
        status_code = call.get("provider_status_code")
        safe_error = f"{error_type}: Gemini model call failed"
        if isinstance(status_code, int):
            safe_error = f"{safe_error}; status {status_code}"
        return SegmentResult(
            [],
            "Segmentation model call failed.",
            profile,
            "invalid",
            [failure_reason],
            calls=calls,
            error=safe_error,
            rejection_reasons=[failure_reason],
        )


def _hybrid_selected(video_id: str, percent: float) -> bool:
    if not video_id or percent <= 0.0:
        return False
    if percent >= 100.0:
        return True
    bucket = int.from_bytes(hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "big") % 10_000
    return bucket < int(percent * 100)


def _flash_configuration_failure(error: str | None) -> bool:
    text = str(error or "").casefold()
    return bool(text) and any(marker in text for marker in (
        "status 400", "status 401", "status 403", "status 404",
        "400 invalid_argument", "401 unauthenticated", "403 permission_denied",
        "404 not_found", "invalid_argument", "permission_denied", "not_found",
        "api key", "permission", "model not found", "model is not found",
        "unsupported model",
        "requires an explicit gemini 3 model", "invalid model",
    ))


def _disable_flash(reason: str) -> None:
    global _flash_disabled_reason
    with _flash_disable_lock:
        if _flash_disabled_reason is None:
            _flash_disabled_reason = reason


def _flash_disable_reason() -> str | None:
    with _flash_disable_lock:
        return _flash_disabled_reason


def _authoritative_pro(transcript: dict, settings: dict, topic: str, deadline: float,
                       cancelled: CancelledCb, *, fallback: bool = False) -> SegmentResult:
    profile = PRO_BOUNDARY_PROFILE if fallback else AUTHORITATIVE_PRO_PROFILE
    operation = "pro_fallback" if fallback else "pro_authoritative"
    runtime_settings = dict(settings)
    runtime_settings["_segment_operation"] = operation
    result = run_segment_profile(
        transcript, runtime_settings, profile, topic=topic,
        deadline_monotonic=deadline, cancelled=cancelled,
    )
    for call in result.calls:
        call["operation"] = operation
    return result


def pro_boundary_fallback_detailed(
    transcript: dict,
    settings: dict,
    *,
    topic: str = "",
    video_id: str = "",
) -> SegmentResult:
    """Run the one aggregate, boundary-only Pro fallback on an existing transcript."""
    sink = settings.get("_segment_telemetry")
    cancelled = settings.get("_segment_cancelled")
    deadline = time.monotonic() + _TOTAL_DEADLINE_S
    configured_deadline = settings.get("deadline_monotonic")
    if configured_deadline is not None:
        try:
            deadline = min(deadline, float(configured_deadline))
        except (TypeError, ValueError, OverflowError):
            pass
    reasons = ["aggregate_initial_yield_below_three"]
    _emit(sink, "pro_fallback", video_id=video_id or None, reasons=reasons)
    result = _authoritative_pro(
        transcript,
        settings,
        topic,
        deadline,
        cancelled,
        fallback=True,
    )
    result.route = "aggregate_pro_fallback"
    result.fallback_reasons = reasons
    for call in result.calls:
        _emit(sink, "model_call", video_id=video_id or None, **call)
    accepted = len(result.clips)
    total_cost = sum(_model_cost(call) for call in result.calls)
    _emit(
        sink,
        "segment_completed" if not result.error else "segment_error",
        video_id=video_id or None,
        route=result.route,
        classification=result.classification,
        classification_reasons=result.classification_reasons,
        rejection_reasons=result.rejection_reasons,
        fallback_reasons=reasons,
        proposed_count=result.proposed_count,
        accepted_count=accepted,
        zero_output=(accepted == 0),
        fallback_rate=1.0,
        pricing_version=_PRICING_VERSION,
        estimated_cost_usd=round(total_cost, 8),
        cost_per_accepted_clip_usd=(
            round(total_cost / accepted, 8) if accepted else None
        ),
        error=result.error,
    )
    return result


def segment_clips_detailed(
    transcript: dict,
    settings: dict,
    *,
    topic: str = "",
    video_id: str = "",
    progress: ProgressCb = None,
    routing_mode: str | None = None,
) -> SegmentResult:
    segments = transcript.get("segments") or []
    if not segments:
        return SegmentResult([], "No transcript segments to segment.", "none", "invalid",
                             ["missing_segments"])

    # Production defaults to the single Flash selector. Explicit alternate modes
    # remain available only to legacy evaluation callers.
    configured_mode = str(routing_mode or "flash_only").lower()
    flash_only = configured_mode == "flash_only"
    mode = configured_mode
    if mode not in {"pro_only", "shadow", "hybrid", "flash_only"}:
        mode = "pro_only"
    disabled_reason = None if flash_only else _flash_disable_reason()
    if disabled_reason is not None and mode in {"shadow", "hybrid"}:
        mode = "pro_only"
    percent = float(config.SEGMENT_HYBRID_PERCENT)
    generation_context = (
        settings.get("generation_context") or settings.get("provider_context")
        if isinstance(settings, dict)
        else None
    )
    generation_hash_key = str(
        getattr(generation_context, "generation_id", "") or video_id
    )
    selected = flash_only or (
        mode == "hybrid" and _hybrid_selected(generation_hash_key, percent)
    )
    route = "flash_first" if selected else "pro_authoritative"
    sink = settings.get("_segment_telemetry") if isinstance(settings, dict) else None
    cancelled = settings.get("_segment_cancelled") if isinstance(settings, dict) else None
    deadline = time.monotonic() + _TOTAL_DEADLINE_S
    configured_deadline = settings.get("deadline_monotonic")
    if configured_deadline is not None:
        try:
            deadline = min(deadline, float(configured_deadline))
        except (TypeError, ValueError, OverflowError):
            pass
    prompt_version = (PRODUCTION_FLASH_PROFILE
                      if selected or mode == "shadow"
                      else AUTHORITATIVE_PRO_PROFILE)
    _emit(sink, "route_selected", video_id=video_id or None, mode=mode,
          configured_mode=configured_mode, route=route, prompt_version=prompt_version,
          hybrid_percent=percent, flash_disabled_reason=disabled_reason)
    if progress and not _cancel_requested(cancelled):
        progress(0.1, "Understanding the transcript…")

    result: SegmentResult
    if mode == "shadow":
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segment-shadow")
        flash_future = pool.submit(
            run_segment_profile, transcript, settings, PRODUCTION_FLASH_PROFILE,
            topic=topic, deadline_monotonic=deadline, cancelled=cancelled,
        )

        def finish_shadow(future) -> None:
            try:
                shadow = future.result()
            except Exception as exc:  # defensive: run_segment_profile normally captures errors
                shadow = SegmentResult(
                    [], "Shadow segmentation failed.", PRODUCTION_FLASH_PROFILE,
                    "invalid", [f"request_failure:{type(exc).__name__}"],
                    error=f"{type(exc).__name__}: {exc}",
                )
            for call in shadow.calls:
                _emit(sink, "model_call", video_id=video_id or None, shadow=True, **call)
            for reason in shadow.fallback_reasons:
                _emit(sink, "pro_fallback", video_id=video_id or None,
                      shadow=True, reason=reason)
            shadow_cost = sum(_model_cost(call) for call in shadow.calls)
            shadow_accepted = shadow.accepted_count
            _emit(
                sink,
                "shadow_comparison",
                video_id=video_id or None,
                classification=shadow.classification,
                reasons=shadow.classification_reasons,
                rejection_reasons=shadow.rejection_reasons,
                proposed_count=shadow.proposed_count,
                accepted_count=shadow_accepted,
                fallback_reasons=shadow.fallback_reasons,
                fallback_rate=1.0 if shadow.fallback_reasons else 0.0,
                pricing_version=_PRICING_VERSION,
                estimated_cost_usd=round(shadow_cost, 8),
                cost_per_accepted_clip_usd=(
                    round(shadow_cost / shadow_accepted, 8) if shadow_accepted else None
                ),
                error=shadow.error,
            )
            flash_error = shadow.flash_configuration_error or shadow.error
            if _flash_configuration_failure(flash_error):
                _disable_flash(str(flash_error))
                _emit(sink, "route_rollback", video_id=video_id or None,
                      reason="flash_model_access_or_configuration_failure")

        flash_future.add_done_callback(finish_shadow)
        pool.shutdown(wait=False)
        result = _authoritative_pro(
            transcript, settings, topic, deadline, cancelled,
        )
        result.route = "shadow_pro_authoritative"
    elif selected:
        flash = run_segment_profile(
            transcript, settings, PRODUCTION_FLASH_PROFILE, topic=topic,
            deadline_monotonic=deadline, cancelled=cancelled,
        )
        _emit(sink, "flash_classified", video_id=video_id or None,
              classification=flash.classification, reasons=flash.classification_reasons,
              rejection_reasons=flash.rejection_reasons,
              proposed_count=flash.proposed_count, accepted_count=flash.accepted_count)
        flash_error = flash.flash_configuration_error or flash.error
        if _flash_configuration_failure(flash_error):
            _disable_flash(str(flash_error))
            _emit(sink, "route_rollback", video_id=video_id or None,
                  reason="flash_model_access_or_configuration_failure")
        if flash.classification == "green" and not flash.error:
            result = flash
            result.route = "hybrid_flash"
        else:
            # Bootstrap's hard latency contract never dispatches Pro, even if a
            # caller forgets to install the normal generation-level fallback gate.
            fallback_allowed = not flash_only
            fallback_gate = (
                settings.get("_segment_pro_fallback_gate")
                if isinstance(settings, dict)
                else None
            )
            if fallback_allowed and callable(fallback_gate):
                try:
                    fallback_allowed = bool(
                        fallback_gate(
                            accepted_count=flash.accepted_count,
                            video_id=video_id,
                        )
                    )
                except Exception as exc:  # fail closed: a gate bug must not spend Pro
                    fallback_allowed = False
                    _emit(
                        sink,
                        "pro_fallback_deferred",
                        video_id=video_id or None,
                        reason=f"fallback_gate_error:{type(exc).__name__}",
                    )
            fallback_reasons = list(flash.classification_reasons) or ["flash_request_failure"]
            if fallback_allowed:
                _emit(
                    sink,
                    "pro_fallback",
                    video_id=video_id or None,
                    reasons=fallback_reasons,
                )
                pro = _authoritative_pro(
                    transcript, settings, topic, deadline, cancelled, fallback=True,
                )
                # Never expose uncertain/invalid Flash when Pro fails.
                result = pro
                result.route = "hybrid_pro_fallback"
                result.fallback_reasons = fallback_reasons
                result.calls = flash.calls + pro.calls
                result.rejection_reasons = [
                    *flash.rejection_reasons,
                    *pro.rejection_reasons,
                ]
            else:
                _emit(
                    sink,
                    "pro_fallback_deferred",
                    video_id=video_id or None,
                    reasons=fallback_reasons,
                )
                # Non-green output never ships while the aggregate gate waits for
                # the second video or chooses a later backfill for the one fallback.
                flash.clips = []
                flash.accepted_count = 0
                flash.fallback_reasons = []
                result = flash
                result.route = "hybrid_flash_deferred"
    else:
        result = _authoritative_pro(
            transcript, settings, topic, deadline, cancelled,
        )
        result.route = "hybrid_control_pro" if mode == "hybrid" else "pro_only"

    cancelled_now = _cancel_requested(cancelled)
    if progress and not cancelled_now:
        progress(0.85, "Placing clip boundaries…")
    total_cost = sum(_model_cost(call) for call in result.calls)
    accepted = len(result.clips)
    fallback_rate = 1.0 if result.fallback_reasons else 0.0
    for call in result.calls:
        _emit(sink, "model_call", video_id=video_id or None, **call)
    _emit(
        sink,
        "segment_completed" if not result.error else "segment_error",
        video_id=video_id or None,
        route=result.route,
        classification=result.classification,
        classification_reasons=result.classification_reasons,
        rejection_reasons=result.rejection_reasons,
        fallback_reasons=result.fallback_reasons,
        proposed_count=result.proposed_count,
        accepted_count=accepted,
        zero_output=(accepted == 0),
        fallback_rate=fallback_rate,
        pricing_version=_PRICING_VERSION,
        estimated_cost_usd=round(total_cost, 8),
        cost_per_accepted_clip_usd=(round(total_cost / accepted, 8) if accepted else None),
        error=result.error,
    )
    if progress and not _cancel_requested(cancelled):
        progress(1.0, f"{accepted} clip(s) ready")
    return result


def segment_clips(
    transcript: dict,
    settings: dict,
    progress: ProgressCb = None,
    topic: str = "",
    video_id: str = "",
) -> tuple[list[dict], str]:
    """Return guarded educational clips while preserving the existing public tuple."""
    result = segment_clips_detailed(
        transcript,
        settings,
        topic=topic,
        video_id=video_id,
        progress=progress,
        routing_mode="flash_only",
    )
    if result.error:
        # A provider/schema/transport failure is not a valid empty selection.
        # Raising prevents the caller from persisting a poisoned empty cache
        # entry while successful zero-match responses remain cacheable.
        telemetry = dict(result.calls[-1]) if result.calls else {}
        telemetry.setdefault(
            "error_type",
            str(result.error).split(":", 1)[0] or "SegmentationProviderError",
        )
        raise _ModelCallError("segmentation provider call failed", telemetry)
    return result.clips, result.notes
