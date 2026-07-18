"""Guarded Gemini educational clip segmentation.

Production uses one medium-thinking Pro boundary-selection call over the whole
timestamped transcript as text, then applies deterministic quality, context,
grounding, filler, and deduplication guards. Video media is not attached.
Legacy routing and enrichment helpers remain available only for isolated
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
from dataclasses import asdict, dataclass, field, is_dataclass, replace
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
    "pro_boundary_audit",
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
    r"(?:got\s+it\s+)?easy\s+peasy(?:\s+lemon\s+squeezy)?|"
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
_PEDAGOGICAL_META_FRAME_ONLY_RE = re.compile(
    r"(?:(?:all right|alright|okay|ok|so|now)\s+)*(?:"
    r"what\s+(?:i|we)\s+(?:want|would\s+like|(?:am|are)\s+going|intend)\s+"
    r"to\s+do(?:\s+(?:here|now|next))?(?:\s+is)?"
    r"(?:\s+a\s+little\s+bit\s+of)?|"
    r"(?:a\s+little\s+bit\s+of\s+)?(?:a\s+)?thought\s+experiment)",
    re.IGNORECASE,
)
_LEADING_PEDAGOGICAL_META_RE = re.compile(
    r"^\s*(?:(?:all right|alright|okay|ok|so|now)\s*[,;:]?\s+)*(?:"
    r"what\s+(?:i|we)\s+(?:want|would\s+like|(?:am|are)\s+going|intend)\s+"
    r"to\s+do(?:\s+(?:here|now|next))?\s+is\s+"
    r"(?:(?:a\s+little\s+bit\s+of\s+)?(?:a\s+)?thought\s+experiment"
    r"[\s,;:.!?—\-\"'’”\)\]]+)?|"
    r"(?:a\s+little\s+bit\s+of\s+)?(?:a\s+)?thought\s+experiment"
    r"[\s,;:.!?—\-\"'’”\)\]]+)",
    re.IGNORECASE,
)
_LEADING_STEP_META_RE = re.compile(
    r"^\s*(?:(?:and|now|so|then)\s*[,;:]?\s+)?"
    r"(?:the\s+)?next\s+step\s+(?:is|would\s+be)\s+(?:that\s+)?"
    r"(?P<teaching>(?:i|we|you)\b[^.!?]{3,220})",
    re.IGNORECASE,
)
_BARE_STEP_ACTION_ONLY_RE = re.compile(
    r"^\s*(?:(?:and|now|so|then)\s*[,;:]?\s+)?(?:the\s+)?next\s+step\s+"
    r"(?:is|would\s+be)\s+(?:that\s+)?(?:i|we|you)\s+"
    r"(?:calculate|compare|compute|determine|evaluate|find|identify|measure|"
    r"solve)\s+"
    r"(?![^.!?]{0,100}\b(?:because|by|from|so\s+that|using|via|when|where|"
    r"which|with)\b)"
    r"(?:[a-z0-9]+(?:[-'’][a-z0-9]+)?\s*){1,6}[.!?]*\s*$",
    re.IGNORECASE,
)
_SEQUENCE_LABEL_ONLY_RE = re.compile(
    r"^\s*(?:(?:all\s+right|alright|and|now|okay|ok|so|then)\s*[,;:]?\s+)*(?:"
    r"(?:(?:i|we)\s+(?:will\s+)?|let\s+me\s+)?"
    r"(?:call|label)\s+(?:this|that|it)\s+(?:as\s+)?"
    r"(?:part|phase|stage|step)\s+[a-z0-9-]+|"
    r"(?:part|phase|stage|step)\s+[a-z0-9-]+\s*[,;:]?\s+"
    r"(?:there\s+(?:are|is)|we\s+have)\s+"
    r"(?:an?|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:cases?|options?|parts?|situations?|steps?)"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_CLARIFICATION_META_ONLY_RE = re.compile(
    r"^\s*(?:(?:all\s+right|alright|and|now|okay|ok|so|then)\s*[,;:]?\s+)*(?:"
    r"there(?:['’]s|\s+is)\s+(?:(?:one|a)\s+)?"
    r"(?:(?:final|important|last|quick)\s+)?(?:clarification|point)"
    r"(?:\s+(?:of|for)\s+clarification|\s+to\s+clarify)?|"
    r"(?:(?:i|we)\s+)?(?:want|wanna|would\s+like|need)\s+to\s+"
    r"(?:clarify|make)\s+(?:(?:one|a|that|this)\s+)?(?:point\s+)?"
    r"(?:(?:absolutely|perfectly|really|very)\s*[,;:]?\s+)*clear|"
    r"that\s+(?:i|we)\s+(?:want|wanna|would\s+like|need)\s+to\s+make\s+"
    r"(?:(?:absolutely|perfectly|really|very)\s*[,;:]?\s+)*clear"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_CROSS_CONTENT_REFERENCE_ONLY_RE = re.compile(
    r"^\s*(?:(?:and|but|now|so|then)\s*[,;:]?\s+)?(?:"
    r"in\s+(?:an(?:other)?|earlier|other|previous)\s+"
    r"(?:episodes?|lessons?|sections?|videos?)\s*[,;:]?\s+)?"
    r"(?:i|we)\s+(?:already\s+|have\s+|['’]ve\s+)?"
    r"(?:covered|discussed|explained|shown|talked\s+about)\s+"
    r"(?:how\s+to\s+do\s+)?(?:it|that|this)"
    r"(?:\s+(?:before|elsewhere|there))?\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_FORWARD_EXPLANATION_PROMISE_ONLY_RE = re.compile(
    r"^\s*(?:(?:and|but|now|so|then)\s*[,;:]?\s+)?"
    r"(?:(?:as\s+)?(?:i|we|you)\s+(?:will|['’]ll)\s+)?"
    r"(?:see|show)\s+(?:how|why)\s+(?:it|that|this)\s+"
    r"(?:comes?\s+into\s+play|connects?|works?)\s+"
    r"(?:in\s+(?:a\s+)?(?:moment|second)|later|shortly)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_ANNOTATION_EMPHASIS_META_ONLY_RE = re.compile(
    r"^\s*(?:(?:and|now|so|then)\s*[,;:]?\s+)?(?:"
    r"(?:i(?:\s+am|['’]m)|we(?:\s+are|['’]re))\s+"
    r"(?:adding|circling|drawing|putting|underlining|writing)\s+"
    r"(?:an?|the)\s+(?:asterisk|box|circle|exclamation\s+mark|star|underline)"
    r"(?:\s+(?:around\s+it|here))?|"
    r"because\s+(?:it|that|this)(?:['’]s|\s+is)\s+"
    r"(?:(?:especially|extremely|really|so|very)\s+)*"
    r"(?:conceptually\s+)?(?:critical|important|key)(?:\s+here)?"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_PEDAGOGICAL_META_TEACHING_ONSET_RE = re.compile(
    r"^\s*(?:ask|calculate|compare|consider|define|derive|determine|differentiate|"
    r"evaluate|explain|find|if|imagine|let(?:['’]?s|\s+us)|prove|show|solve|"
    r"suppose|take|try|what\s+if)\b",
    re.IGNORECASE,
)
_INTERNAL_INTERRUPTION_MARKER_RE = re.compile(
    r"\b(?:today'?s sponsor|sponsored by|administrative (?:note|announcement)|"
    r"course (?:administration|logistics)|(?:a\s+)?(?:quick|brief|short) "
    r"(?:aside|tangent)|housekeeping)\b",
    re.IGNORECASE,
)
_INTERNAL_CHANNEL_PROMO_RE = re.compile(
    r"\bwelcome\s+to\b[^.!?]{0,80}\b(?:season\s+\d+|tips?)\b",
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
_CROSS_CUE_INSTRUCTIONAL_PREVIEW_RE = re.compile(
    r"^\s*what\s+(?:i|we)(?:['’](?:m|re)|\s+(?:am|are))\s+going\s+to\s+"
    r"(?:cover|discuss|explain|go\s+over|introduce|review|show|teach|"
    r"talk\s+about)\s+in\s+(?:this|the)\s+"
    r"(?:course|lesson|section|video)\s+(?:is|are)\b",
    re.IGNORECASE,
)
_OPENING_ANAPHORIC_SETUP_REFERENCE_RE = re.compile(
    r"\b(?:this|that|these|those)\s+(?:approach|argument|case|change|concept|"
    r"effect|equation|example|expression|function|idea|method|model|movement|"
    r"object|problem|process|proof|quantity|relationship|result|rule|statement|"
    r"step|system|term|value)\b",
    re.IGNORECASE,
)
_OPENING_SETUP_PRONOUN_REFERENCE_RE = re.compile(
    r"^\s*given\s+that\b|\b(?:it|them|these|they|this|those)\b|"
    r"\b(?:apply|calculate|consider|evaluate|solve|use)\s+that\b",
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
_OPENING_UNRESOLVED_EDGE_REFERENCE_RE = re.compile(
    r"^\s*(?:based|depending)\s+(?:on|upon)\s+"
    r"(?:which|this|that|these|those|it|them)\b|"
    r"^\s*(?:[a-z]\s+(?:calculate|compare|conclude|determine|evaluate|find|"
    r"identify|solve|test)\b|[a-z]{1,3}\s+then\s+(?:i|we|you|they)\b)|"
    r"^\s*[$€£]?\d[\d,.]*(?:\s*%)?\s+(?:now|so|then)\b|"
    r"^\s*(?:p\s+)?(?:value|statistic|score|probability)\s+to\s+"
    r"(?:a|an|the|your)\b",
    re.IGNORECASE,
)
_OPENING_CONTEXT_ONLY_TRANSITION_RE = re.compile(
    r"^\s*(?:and\s+)?(?:on\s+the\s+other\s+hand|"
    r"(?:now|then)\s+the\s+next\s+(?:step|thing)|"
    r"so\s+step\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+))",
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
_TERMINAL_ATTRIBUTIVE_MODIFIER_RE = re.compile(
    r"\bof\s+"
    r"[a-z][a-z'’\-]{3,}(?:al|ary|ed|ent|ful|ic|ical|ing|ive|less|ory|ous)\s*$",
    re.IGNORECASE,
)
_NEXT_ATTRIBUTIVE_SUBJECT_PREDICATE_RE = re.compile(
    r"^\s*[a-z][\w'’\-]*(?:\s+[a-z][\w'’\-]*){0,2}\s+"
    r"(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would|equals?|means?|represents?)\b",
    re.IGNORECASE,
)
_TERMINAL_UNPUNCTUATED_SUBJECT_PRONOUN_RE = re.compile(
    r"\b(?P<pronoun>i|we|you|he|she|it|they|this|that)\s*$",
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
_TERMINAL_DANGLING_POSSESSIVE_RE = re.compile(
    r"\b(?:her|his|its|my|our|their|your)\s*[.!?]?[\"')\]]*$",
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
    r"\b(?:although|because|even\s+if\s+(?:not|so)|if|since|unless|until|"
    r"when|whereas|while)"
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
    r"\b(?:evidence|support)\s+(?:against|for)|"
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
_TERMINAL_DANGLING_TRANSITIVE_RE = re.compile(
    r"\b(?:"
    r"(?:give|show|tell)(?:s|ed)?\s+(?:her|him|me|them|us|you)|"
    r"(?:can|could|may|might|will|would)\s+"
    r"(?:demonstrate|describe|explain|give|illustrate|indicate|reveal|show|tell)"
    r")"
    r"\s*[.!?]?[\"')\]]*$",
    re.IGNORECASE,
)
_TERMINAL_AUXILIARY_TRANSITIVE_RE = re.compile(
    r"\b(?:can|could|did|do|does|may|might|must|shall|should|will|would)\s+"
    r"(?:calculate|compare|determine|evaluate|find|identify|specify|write)"
    r"\s*[.!?]?[\"')\]]*$",
    re.IGNORECASE,
)
_TERMINAL_EDGE_DISFLUENCY_RE = re.compile(
    r"(?:^|\s)(?P<noise>(?:(?:okay|ok|great|right)\s+)*"
    r"(?:er+|hmm+|uh+|um+))\s*[.!?]?[\"')\]]*$",
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
    r"let(?:['’]?s|\s+us)\s+(?:solve|work\s+on|work\s+through)\s+"
    r"(?:this|that|the)\s+"
    r"(?:calculation|case|derivation|example|exercise|problem|proof)|"
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
    r"\bdifferences?\s+between\b|"
    r"\b(?:connection|interaction|link|relationship)\s+between\b|"
    r"\b(?:connect|link|relate)\b[^.!?]{0,120}\b(?:and|to|with)\b|"
    r"\b(?:in\s+terms\s+of)\b|"
    r"\bderiv(?:e|ed|es|ing)\b[^.!?]{0,100}\bfrom\b|"
    r"\b(?:how|why)\b[^.!?]{1,100}\b(?:affect|cause|define|depend|derive|differ|"
    r"form|impl(?:y|ies)|influence|interact|produce|relate|shape|use|yield)\w*\b|"
    r"\b(?:form|produce|yield)\w*\b|"
    r"\b(?:in\s+contrast|unlike|whereas)\b|"
    r"\bversus\b|\bvs\.?(?:\s|$)",
    re.IGNORECASE,
)
_EXPLICIT_COMPARISON_OBJECTIVE_RE = re.compile(
    r"\b(?:compare|contrast|distinguish)\b[^.!?]{0,120}\b(?:and|from|with)\b|"
    r"\bdifferences?\s+between\b|"
    r"\bversus\b|\bvs\.?(?:\s|$)",
    re.IGNORECASE,
)
_DIRECT_COMPARISON_CLAUSE_RE = re.compile(
    r"\b(?:differs?\s+from|in\s+contrast|rather\s+than|unlike|whereas)\b",
    re.IGNORECASE,
)
_SPOKEN_COMPARISON_RELATION_RE = re.compile(
    r"\b(?:compare|contrast|differ|distinguish)\w*\b|"
    r"\b(?:greater|higher|larger|less|lower|more|fewer|smaller)\b"
    r"[^.!?]{0,80}\bthan\b|"
    r"\b(?:both|same\s+as|similar\s+to|equal\s+to|equivalent\s+to|"
    r"unlike|whereas|while)\b|"
    r"\bbut\b",
    re.IGNORECASE,
)
_SPOKEN_PATH_RELATION_RE = re.compile(
    r"\b(?:become|change|convert|flow|lead|move|pass|transition|transform|turn)"
    r"\w*\b[^.!?]{0,80}\b(?:from|into|through|to)\b|"
    r"\bderive\w*\b[^.!?]{0,80}\bfrom\b|"
    r"\b(?:is|are|was|were)\s+(?:converted|defined|derived|formed|generated|"
    r"produced|transformed)\s+(?:as|by|from|into|through|using)\b|"
    r"\b(?:affect|cause|connect|create|define|derive|drive|enable|form|generate|"
    r"link|power|produce|use|yield)\w*\b",
    re.IGNORECASE,
)
_COMPARISON_SETUP_REFERENCE_RE = re.compile(
    r"\b(?:answer|calculation|case|example|problem|result|solution)\b",
    re.IGNORECASE,
)
_EXPLICIT_CONJUNCTIVE_REQUEST_RE = re.compile(
    r"\b(?:and|alongside|together\s+with)\b|(?<=\w)\s*/\s*(?=\w)",
    re.IGNORECASE,
)
_EXPLICIT_TRANSITION_REQUEST_RE = re.compile(
    r"\btransition(?:s|ed|ing)?\s+(?:(?:from)\b[^,;.!?]{1,100}\b)?"
    r"(?:to|into)\b",
    re.IGNORECASE,
)
_NON_DIRECTIONAL_TO_HEAD_RE = re.compile(
    r"\b(?:access|answer|approach|attachment|attention|barrier|challenge|"
    r"commitment|connection|contribution|guide|introduction|intro|key|lesson|"
    r"objection|overview|reaction|reference|relationship|response|right|road|"
    r"solution|threat|tribute|tutorial)\s*$",
    re.IGNORECASE,
)
_COMPARISON_DISTINCTION_RE = re.compile(
    r"\b(?:are|were)\s+(?:two\s+)?(?:distinct|different)\b",
    re.IGNORECASE,
)
_COMPARISON_CONNECTOR_RE = re.compile(r"\b(?:and|versus|vs\.?)\b", re.IGNORECASE)
_FOLLOWUP_EXAMPLE_REUSE_RE = re.compile(
    r"(?<!\w)(?:(?:now|so)\s*[,;:]?\s+)*"
    r"let(?:['’]?s|\s+us)\s+(?:(?:again|also|still)\s+)?"
    r"(?:reuse|take|use)\b[^.!?]{0,120}?\bas\s+an?\s+"
    r"(?:case|example)\b",
    re.IGNORECASE,
)
_NEW_HYPOTHETICAL_EXAMPLE_RE = re.compile(
    r"(?<!\w)(?P<navigation>"
    r"(?:(?:all\s+right|alright|okay|ok|so)\s*[,;:]?\s+)*"
    r"now\s*[,;:]?\s+let(?:['’]?s|\s+us)\s+imagine\b)"
    r"(?=[^.!?]{0,180}\b(?:(?:another|different|new|second)\b|"
    r"(?:one|two|three|four|five|\d+)\s+more\b|"
    r"just\s+like\s+before\b))",
    re.IGNORECASE,
)
_EXPLICIT_RECAP_NAVIGATION_RE = re.compile(
    r"(?<!\w)(?P<navigation>"
    r"(?:(?:all\s+right|alright|okay|ok|so|now)\s*[,;:]?\s+)*"
    r"(?:"
    r"(?:to\s+|let(?:['’]?s|\s+us)\s+)"
    r"(?:(?:briefly|quickly)\s+)?(?:recap|(?:sum\s+up|summarize)(?:"
    r"(?=\s*(?:[,;:—-]|[.!?…]+|$))|"
    r"\s+what\s+(?:(?:we|you)(?:['’]?ve|\s+have)?\s+)?"
    r"(?:covered|discussed|learned|seen)\s+"
    r"(?:so\s+far|up\s+to\s+this\s+point)))|"
    r"in\s+summary\b(?!\s+(?:form|measure|report|statistic|table)s?\b)"
    r")"
    r"\s*(?:[,;:—-]|[.!?…]+)?\s*)",
    re.IGNORECASE,
)
_STRONG_RECAP_EXIT_RE = re.compile(
    r"\b(?:"
    r"move\s+on\s+to|switch\s+to|shift\s+to|turn\s+to|talk\s+about|"
    r"we(?:['’]re|\s+are)\s+(?:moving\s+on|switching|shifting|turning)\s+to|"
    r"next\s+(?:we|i)(?:['’]ll|\s+will)\s+"
    r"(?:move\s+on\s+to|switch\s+to|shift\s+to|turn\s+to|talk\s+about|"
    r"discuss|cover|look\s+at)|"
    r"let(?:['’]?s|\s+us)\s+(?:discuss|cover|look\s+at)|"
    r"next\s+up\s+is|that\s+brings\s+us\s+to|moving\s+on\s+to|"
    r"turn\s+(?:our\s+)?attention\s+to|"
    r"(?:the\s+)?next\s+(?:topic|concept|section)"
    r")\b",
    re.IGNORECASE,
)
_HARD_TOPIC_RESET_RE = re.compile(
    r"(?<!\w)(?:(?:all right|alright|okay|ok|so)\s*[,;:]?\s+)*"
    r"(?:(?:now\s*[,;:]?\s+)?let(?:['’]?s|\s+us)\s+(?:move\s+on\s+to|switch\s+to|"
    r"shift\s+to|turn\s+to|talk\s+(?:more\s+)?about|discuss|cover|look\s+at|"
    r"(?:get|go)\s+back\s+to|return\s+to)|"
    r"(?:now\s+)?we(?:['’]re|\s+are)\s+(?:moving\s+on|switching|shifting|"
    r"turning)\s+to|"
    r"next\s+(?:we|i)(?:['’]ll|\s+will)\s+(?:move\s+on\s+to|switch\s+to|"
    r"shift\s+to|turn\s+to|talk\s+about|discuss|cover|look\s+at)|"
    r"(?:now\s+)?next\s+up\s+is|"
    r"(?:now\s+)?that\s+brings\s+us\s+to|"
    r"(?:now\s+)?moving\s+on\s+to|"
    r"(?:now\s+)?turn\s+(?:our\s+)?attention\s+to|"
    r"(?P<conditional_handoff>"
    r"(?:but|however)\s+(?:occasionally|sometimes)\s*[,]?\s+"
    r"[^.!?]{1,120}?\b(?:cannot|can['’]?t|does\s+not|doesn['’]?t|"
    r"will\s+not|won['’]?t)\b[^.!?]{0,100}[.!?]\s+"
    r"(?:and\s+)?when\s+that\s+happens\s*[,]?\s+"
    r"(?:we|you)\s+(?:can|could|should)\s+(?:instead\s+)?"
    r"(?:apply|use|turn\s+to))|"
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
_INDEPENDENT_UNIT_RESET_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?P<navigation>"
    r"(?:(?:all\s+right|alright|okay|ok|so|now)\s*[,;:]?\s+)*"
    r"(?P<subject>(?:"
    r"(?:goal|area|branch|concept|principle|topic)\s+"
    r"(?:(?:number\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten)|"
    r"number\s+\d+)|"
    r"(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|"
    r"ninth|tenth)\s+(?:goal|area|branch|concept|principle|topic)"
    r")))\b(?!\s+(?:at|behind|between|by|for|from|in|of|on|over|through|"
    r"under|with|within)\b)",
    re.IGNORECASE,
)
_NAMED_UNIT_LABEL_RESET_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?P<navigation>"
    r"(?:(?:all\s+right|alright|okay|ok|so|now)\s*[,;:]?\s+)+"
    r"(?P<subject>(?:the|our)\s+"
    r"(?!(?:next|previous|same|this|that)\b)"
    r"(?:[a-z][\w'’-]*\s+){1,6}"
    r"(?:problem|question|topic|concept|goal|area)))\b"
    r"(?=\s+(?:asks?\b|concerns?\b|deals?\s+with\b|focuses?\s+on\b|"
    r"is\s+(?:about|how|to|what|whether|why)\b|"
    r":\s*(?:how|what|when|where|which|why)\b))",
    re.IGNORECASE,
)
_NAMED_METHOD_CONTRAST_RESET_RE = re.compile(
    r"(?<!\w)(?P<navigation>"
    r"(?:(?:does|did)\s+that\s+make\s+sense\s+)?"
    r"(?:(?:and|but|so)\s+)?(?:now|next)\s+)"
    r"(?P<subject>(?:(?:a|an|our|that|the|this)\s+)?"
    r"(?:(?!(?:a|an|our|the|this|that)\b)"
    r"[a-z0-9][\w'’]*(?:\s+|-)){1,5}"
    r"(?:distribution|method|model|procedure|test))\s+"
    r"(?:is|works?|uses?|requires?)\s+"
    r"(?:(?:a|an)\s+)?(?:(?:fundamentally|meaningfully|slightly|very)\s+|"
    r"a\s+little\s+)?(?:different|distinct)\b",
    re.IGNORECASE,
)
_NEXT_DISTINCT_UNIT_RESET_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?P<navigation>"
    r"(?:(?:all\s+right|alright|okay|ok|so|now)\s*[,;:]?\s+)*"
    r"(?:the\s+)?next\s+(?:thing|subject|issue)\s+"
    r"(?:(?:i|we)\s+)?(?:have|got|need|want)\s+to\s+"
    r"(?:address|consider|cover|discuss|explain|look\s+at|talk\s+about)\s+"
    r"(?:is\s+)?(?P<subject>(?:(?:a|an|the)\s+)?"
    r"(?:(?:different|new|other)\s+)?"
    r"(?:problem|question|topic|concept|goal|area)))\b",
    re.IGNORECASE,
)
_ENUMERATED_META_OUTLINE_RE = re.compile(
    r"(?<!\w)(?P<navigation>"
    r"(?:(?:(?:small|double)\s+)?bam\s+)?"
    r"(?:(?:all\s+right|alright|okay|ok|so|now)\s*[,;:]?\s+)*"
    r"(?:before\s+(?:(?:we|i)(?:['’]?re|\s+are)?\s+)?"
    r"(?:done|finish|wrap\s+up|are\s+done|end)\s*[,;:]?\s+)?"
    r"let\s+me\s+(?:add|cover|discuss|explain|mention|say)\s+"
    r"(?:(?:two|three|four|five|six|seven|eight|nine|ten)|\d+|"
    r"a\s+couple(?:\s+of)?)\s+(?:more\s+)?"
    r"(?:ideas|notes|observations|points|things)\b\s*)",
    re.IGNORECASE,
)
_ENUMERATED_META_UNIT_RE = re.compile(
    r"(?<!\w)(?P<navigation>"
    r"(?:(?:(?:small|double)\s+)?bam\s+)?"
    r"(?:(?:all\s+right|alright|okay|ok|so|now)\s*[,;:]?\s+)*"
    r"(?:(?:now\s+)?that\s+(?:i|we)(?:['’]?ve|\s+have)\s+"
    r"[^.!?]{0,120}?\s+out\s+of\s+the\s+way\s+)?"
    r"(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|"
    r"ninth|tenth|final|last)\s+thing\s+"
    r"(?:(?:i|we)\s+)?(?:have|need|want|would\s+like)\s+to\s+"
    r"(?:add|cover|discuss|explain|mention|say|tell\s+you)"
    r"(?:\s+(?:is|concerns?))?"
    r"(?:\s+(?:"
    r"(?:(?:(?:just|only)\s+(?:more\s+)?|more\s+)"
    r"(?:background|context|terminology)"
    r"(?:\s+in\s+(?:fancy|formal|technical)\s+"
    r"(?:[a-z][\w'’-]*\s+){0,2}(?:language|lingo|terms))?)|"
    r"(?:background|context|terminology)\s+in\s+"
    r"(?:fancy|formal|technical)\s+"
    r"(?:[a-z][\w'’-]*\s+){0,2}(?:language|lingo|terms))\s+"
    r"(?=(?:a|an|the|this|that|these|those)\b))?\s*)",
    re.IGNORECASE,
)
_ENUMERATED_OUTLINE_RE = re.compile(
    r"\b(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|several|"
    r"multiple)|\d+)\s+"
    r"(?:(?:basic|central|core|distinct|key|main|primary)\s+)?"
    r"(?:areas|branches|concepts|fields|goals|lessons|principles|topics|units)\b",
    re.IGNORECASE,
)
_ENUMERATED_SUBSTANTIVE_RELATION_RE = re.compile(
    r"\b(?:affect|allow|balance|cause|change|connect|control|convert|define|"
    r"describe|determine|differ|divide|enable|explain|govern|interact|limit|"
    r"measure|produce|regulate|relate|represent|shape|show|work|yield)\w*\b",
    re.IGNORECASE,
)
_CLAIM_PROMISSORY_FRAGMENT_RE = re.compile(
    r"^\s*(?:attempt|cover|discuss|go\s+over|introduce|learn|look\s+at|"
    r"preview|review|talk\s+about|teach|try\s+to\s+cover|understand)\b",
    re.IGNORECASE,
)
_ATOMIC_DECLARATIVE_ONSET_RE = re.compile(
    r"^\s*(?:(?:all\s+right|alright|okay|ok|now|so)\s*[,;:]?\s+)*"
    r"(?P<subject>"
    r"(?!(?:after|although|as|assuming|because|before|but|during|given|he|"
    r"how|i|if|it|once|provided|she|since|suppose|that|these|they|this|"
    r"those|unless|until|we|what|when|where|which|while|why|you)\b)"
    r"(?:[a-z0-9][\w'’-]*\s+){1,6}?"
    r")(?P<predicate>are|is|means?|refers?\s+to|describes?|measures?|"
    r"absorbs?|accumulates?|affects?|approaches?|captures?|carries?|causes?|"
    r"changes?|combines?|converts?|creates?|decreases?|denotes?|defines?|"
    r"demonstrates?|describes?|divides?|emits?|enables?|equals?|explains?|"
    r"forms?|generates?|governs?|illustrates?|increases?|measures?|prevents?|"
    r"begins?|occurs?|produces?|receives?|refers?\s+to|regenerates?|regulates?|"
    r"rejects?|releases?|removes?|represents?|solves?|starts?|stores?|"
    r"summarizes?|transfers?|transforms?|uses?)\b",
    re.IGNORECASE,
)
_ATOMIC_DEFINITIONAL_PREDICATE_RE = re.compile(
    r"^(?:are|is|means?|refers?\s+to|describes?|measures?|represents?|"
    r"denotes?|defines?|equals?)$",
    re.IGNORECASE,
)
_ATOMIC_DEPENDENT_SUBJECTS = {
    "answer", "conclusion", "consequence", "outcome", "result", "solution",
    "step", "value",
}
_ATOMIC_BREADTH_RE = re.compile(
    r"\b(?:areas|basics|branches|concepts|fields|foundations?|fundamentals|"
    r"ideas|overview|principles|survey|topics|units)\b",
    re.IGNORECASE,
)
_ATOMIC_WORKED_SCOPE_RE = re.compile(
    r"\b(?:calculate|calculation|compute|derivation|derive|exercise|problem|"
    r"proof|prove|solution|solve|worked\s+example)\b",
    re.IGNORECASE,
)
_ATOMIC_CAUSAL_SCOPE_RE = re.compile(
    r"\b(?:causes?\s+of|effects?\s+of|reasons?\s+for|lead\w*\s+to|"
    r"result\w*\s+in)\b",
    re.IGNORECASE,
)
_ATOMIC_COHERENT_ARC_SCOPE_RE = re.compile(
    r"\b(?:how|why|causal|chain|cycle|mechanism|pathway|process|sequence|"
    r"stages?|steps?)\b",
    re.IGNORECASE,
)
_ATOMIC_COHERENT_LINK_RE = re.compile(
    r"\b(?:after|before|because|causes?|consequently|converts?|drives?|"
    r"enables?|leads?\s+to|produces?|results?\s+in|so\s+that|then|"
    r"therefore|through|thus|which\s+(?:causes?|means?|produces?|shows?))\b",
    re.IGNORECASE,
)
_PEDAGOGICAL_LENS_HANDOFF_RE = re.compile(
    r"(?<!\w)(?P<navigation>(?:(?:and|but|now|so)\s+)?"
    r"(?:this|that)\s+is\s+(?:where|when)\b"
    r"(?:\s+[a-z][\w'’-]*){0,14}\s+"
    r"(?:caveat|intuition|intuitive|interpretation|meaning|notation|"
    r"perspective|subtlety|warning))\b",
    re.IGNORECASE,
)
_PEDAGOGICAL_UNIT_COMPLETION_RE = re.compile(
    r"\b(?:so\s+there|therefore|thus|hence)\s+"
    r"(?:i|we|you)(?:['’]ve|\s+have)?\s+"
    r"(?:applied|completed|demonstrated|derived|established|evaluated|found|"
    r"proved|shown|solved|worked\s+out)\b",
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
_OPENING_RELATIVE_WHERE_FRAGMENT_RE = re.compile(
    r"^\s*where\s+(?:he|her|him|it|she|they|them|we|you)\b",
    re.IGNORECASE,
)
_OPENING_SUBORDINATE_PRONOUN_REFERENCE_RE = re.compile(
    r"^\s*(?:after|as|before|because|if|once|when|while)\s+"
    r"(?:he|it|she|they|this|that|these|those)\b",
    re.IGNORECASE,
)
_OPENING_DEMONSTRATIVE_REFERENCE_RE = re.compile(
    r"^\s*(?:this|that|these|those)\s+"
    r"(?:answer|approach|assumption|calculation|case|change|condition|difference|"
    r"equation|expression|formula|idea|method|problem|process|quantity|reason|"
    r"relationship|result|rule|solution|step|term|value|variable)\b",
    re.IGNORECASE,
)
_OPENING_CONTEXTUAL_MODIFIER_SUBJECT_RE = re.compile(
    r"^\s*(?:the|this|that)\s+(?:aforementioned|current|following|"
    r"intermediate|preceding|previous|remaining|resulting|same)\s+"
    r"[a-z][a-z'’-]*\b",
    re.IGNORECASE,
)
_OPENING_BARE_RELATIONAL_PREDICATE_RE = re.compile(
    r"^\s*(?:(?:is|are|was|were)\s+"
    r"(?:equal|equivalent|proportional|related|similar|connected|dependent)\b|"
    r"(?:(?:directly|inversely)\s+)?"
    r"(?:equal|equivalent|proportional|related|similar|connected|dependent)\s+"
    r"(?:to|on)\b)",
    re.IGNORECASE,
)
_OPENING_SUBJECTLESS_ADJECTIVE_COMPLEMENT_RE = re.compile(
    r"^\s*(?:(?:extremely|highly|quite|really|so|too|very)\s+)*"
    r"(?:clear|likely|possible|probable|unlikely)\s+that\b",
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
    r"(?:given|assuming|provided)(?:\s+that)?\b"
    r"(?![^,;.!?]{0,100}[,;])|"
    r"(?:conditional\s+on|under\s+(?:the\s+)?(?:assumption|condition))\b|"
    r"which\s+(?:are|equals?|gives?|is|means?|shows?)\b)",
    re.IGNORECASE,
)
_EMBEDDED_CLAUSE_ADJUNCT_RE = re.compile(
    r"\b(?:after|before|if|once|when|where|while)\b",
    re.IGNORECASE,
)
_FINITE_PREDICATE_SIGNAL_RE = re.compile(
    r"\b(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would)\b",
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
_PRONOUN_PREDICATE_CONTINUATION_RE = re.compile(
    r"^\s*(?P<auxiliary>am|is|are|was|were|has|have|had|can|could|will|"
    r"would|should|shall|may|might|must|does|do|did)\b",
    re.IGNORECASE,
)
_MODAL_AUXILIARIES = frozenset({
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
})
_SUBJECT_PRONOUN_AUXILIARIES = {
    "i": _MODAL_AUXILIARIES | {"am", "was", "have", "had", "do", "did"},
    "we": _MODAL_AUXILIARIES | {"are", "were", "have", "had", "do", "did"},
    "you": _MODAL_AUXILIARIES | {"are", "were", "have", "had", "do", "did"},
    "they": _MODAL_AUXILIARIES | {"are", "were", "have", "had", "do", "did"},
    "he": _MODAL_AUXILIARIES | {"is", "was", "has", "had", "does", "did"},
    "she": _MODAL_AUXILIARIES | {"is", "was", "has", "had", "does", "did"},
    "it": _MODAL_AUXILIARIES | {"is", "was", "has", "had", "does", "did"},
    "this": _MODAL_AUXILIARIES | {"is", "was", "has", "had", "does", "did"},
    "that": _MODAL_AUXILIARIES | {"is", "was", "has", "had", "does", "did"},
}
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
_OPENING_RELATIVE_BRIDGE_RE = re.compile(
    r"^\s*which\s+(?:also\s+)?(?:aligns?|corresponds?|fits?|"
    r"illustrates?|matches?|means?|reflects?|shows?|supports?)\b"
    r"[^?]{0,160}:\s*$",
    re.IGNORECASE,
)
_OPENING_DEPENDENT_SENTENCE_RE = re.compile(
    r"^\s*(?:although|because|if|since|unless|until|when|whereas|while)\b",
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
    r"^\s*(?:"
    r"first\s*[,;:]?\s+(?:i|we)\s+"
    r"(?:discuss|review|talk\s+about)\b|"
    r"(?:"
    r"in\s+(?:this|the)\s+(?:chapter|class|course|episode|lecture|lesson|"
    r"module|section|talk|tutorial|video)\s*[,;:]?\s+"
    r"(?:i|we)(?:['’](?:ll|m|re)|\s+(?:am|are|will))?\s+|"
    r"today\s*[,;:]?\s+(?:i|we)"
    r"(?:['’](?:ll|m|re)|\s+(?:am|are|will))?\s+|"
    r"(?:this|the)\s+(?:chapter|class|course|episode|lecture|lesson|module|"
    r"section|talk|tutorial|video)\s*[,;:]?\s+(?:i|we)"
    r"(?:['’](?:ll|m|re)|\s+(?:am|are|will))?\s+|"
    r"(?:this|the)\s+(?:chapter|class|course|episode|lecture|lesson|module|"
    r"section|talk|tutorial|video)\s*[,;:]?\s+"
    r"(?:will\s+|(?:is|was)\s+going\s+to\s+)?"
    r")"
    r"(?:just\s+)?(?:(?:am|are)\s+)?(?:going\s+to\s+)?"
    r"(?:(?:attempt|try)\s+to\s+)?"
    r"(?:cover|discuss|explains?|go(?:\s+over)?|introduce|look\s+at|preview|"
    r"review|show(?!\s+that\b)|talk\s+about|teach)\b|"
    r"(?:our|the)\s+(?:aim|goal|objective)\s+(?:for\s+)?today\s+is\b|"
    r"by\s+the\s+end\s+of\s+(?:this|the)\s+"
    r"(?:chapter|class|course|episode|lecture|lesson|module|section|talk|"
    r"tutorial|video)\s*[,;:]?\s+(?:i|we|you)(?:['’]ll|\s+will)\s+"
    r"(?:know|learn|understand)\b"
    r")",
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
_WORKED_UNIT_FUTURE_ACTION_PREFIX_RE = re.compile(
    r"\b(?:i(?:['’]m|\s+am)?|we(?:['’]re|\s+are)?|"
    r"you(?:['’]re|\s+are)?)\s+(?:now\s+)?"
    r"(?:going|need|want)\s+to\s*$",
    re.IGNORECASE,
)
_WORKED_UNIT_EVIDENCE_PROMPT_PREFIX_RE = re.compile(
    r"\b(?:example|exercise|problem)s?\b"
    r"(?P<glue>[^.!?]{0,120})\s*$",
    re.IGNORECASE,
)
_WORKED_UNIT_STRUCTURAL_PROMPT_TOKENS = frozenset({
    "a", "an", "and", "are", "asked", "as", "can", "could", "do",
    "example", "examples", "exercise", "exercises", "for", "here's", "i",
    "if", "let", "let's", "like", "more", "need", "new", "next", "now",
    "one", "on", "please", "problem", "problems", "say", "so", "some",
    "suppose", "the", "to", "try", "us", "use", "want", "we", "work",
    "would", "you",
})
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
    r"what\s+(?:i|we)\s+(?:want|would\s+like|(?:am|are)\s+going|intend)\s+"
    r"to\s+(?:do|explain|show)\b|"
    r"what\s+(?:i|we|you)(?:['’]re|\s+are)\s+"
    r"(?:[a-z][\w'’-]*ing|left\s+with)\b|"
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
_WORKED_UNIT_UNRESOLVED_ANALOGY_REFERENCE_RE = re.compile(
    r"\b(?:it(?:['’]s|\s+is)\s+)?(?:the\s+)?exact(?:ly)?\s+same\s+"
    r"(?:idea|method|operation|process|steps?|thing|way)\b",
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
    r"what\s+(?:i|we)\s+can\s+do\s+(?:here\s+)?is\b|"
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
    r"(?:which|that)\s+(?:complete(?:d|s)?|finish(?:ed|es)?)\s+(?:this|the)\s+"
    r"(?:(?:first|second|third|next|final)\s+)?"
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
    r"(?:now\s+)?let(?:['’]?s|\s+us)\s+(?:"
    r"(?:use|take)\s+(?:an?|this)\s+"
    r"(?:(?:brief|concrete|quick|short|simple|worked)\s+)*"
    r"(?:calculation|case|derivation|example|exercise|problem|proof)|"
    r"(?:do|go\s+through|look\s+at|try|work\s+on|work\s+through)\s+"
    r"(?:some\s+more|one\s+more|more|the\s+next|another|next|"
    r"a\s+(?:different|new)|new)\s+"
    r"(?:calculation|case|derivation|example|exercise|problem|proof)s?)\b",
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
    r"(?:now\s+)?let(?:['’]?s|\s+us)\s+"
    r"(?:solve|work\s+on|work\s+through)\s+(?:this|that|the)\s+"
    r"(?:calculation|case|derivation|example|exercise|problem|proof)\b|"
    r"(?:(?:all\s+right|alright|okay|ok|so)\s*[,;:]?\s+)?"
    r"(?:now\s+)?before\s+(?:i|we)\s+"
    r"(?:go\s+over|start|work\s+through)\b[^.!?\n]{0,140}?"
    r"(?:i|we)\s+(?:first\s+)?(?:have|need|want)\s+to\s+"
    r"(?:cover|discuss|introduce|talk\s+(?:more\s+)?about)\b|"
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
    r"(?:^\s*as\s+you(?:['’]ll|\s+will)\s+see\s+in\s+future\s+videos\b|"
    r"(?:,\s*)?which\s+(?:i|we|you)(?:['’]ll|\s+will)\s+"
    r"(?:cover|discuss|examine|explain|explore|prove|show)\b[^.!?]{0,140}?"
    r"\bin\s+(?:(?:our|the|your)\s+)?(?:following|future|next)\s+"
    r"(?:course|lesson|section|video)\b[.!?]*|"
    r"(?P<media_promise>\b(?:(?:and|but)\s+)?as\s+(?:that|this|it)\s+"
    r"(?:can|could|may|might)\s+not\s+(?:(?:be|seem)\s+)?"
    r"(?:clear|intuitive|obvious|simple)\b[^.!?]{0,180}?"
    r"\bby\s+the\s+end\s+of\s+(?:this|the)\s+"
    r"(?:course|lesson|section|video)\b"
    r"(?:\s+or\s+(?:the\s+)?next\s+"
    r"(?:course|lesson|one|section|video))?[.!?]*))",
    re.IGNORECASE,
)
_TERMINAL_MEDIA_PROMO_RE = re.compile(
    r"^\s*(?:"
    r"(?:be\s+sure\s+to|please)\s+(?:see|view|watch)\b[^.!?]{0,180}"
    r"\b(?:course|lesson|section|video)s?\b[^.!?]*|"
    r"(?:check\s+out|continue\s+to|view|watch)\s+(?:the\s+)?"
    r"(?:following|future|next)\s+(?:course|lesson|section|video)\b"
    r"(?!\s+(?:frame|signal)\b)[^.!?]*|"
    r"see\s+you\s+in\s+(?:the\s+)?(?:following|future|next)\s+"
    r"(?:course|lesson|section|video)\b[^.!?]*|"
    r"(?:i|we)(?:['’]ll|\s+will)\s+(?:cover|discuss|explain|prove|show)\b"
    r"[^.!?]{0,140}\bin\s+(?:the\s+)?(?:following|future|next)\s+"
    r"(?:course|lesson|section|video)\b[^.!?]*|"
    r"you\s+can\s+(?:find|learn|read|see|watch)\s+more\b[^.!?]{0,120}"
    r"\bin\s+(?:the\s+)?(?:following|future|next)\s+"
    r"(?:course|lesson|section|video)\b[^.!?]*"
    r")[.!?]*\s*$",
    re.IGNORECASE,
)
_TERMINAL_MEDIA_SUBJECT_SENSE_RE = re.compile(
    r"\b(?:course|section)s?\s+of\s+"
    r"(?!(?:(?:a|her|his|its|my|our|the|their|this|your)\s+)?"
    r"(?:course|lesson|section|video)\b)|"
    r"\bvideo\s+(?:clip|file|frame|segment|sequence|signal|stream)s?\b",
    re.IGNORECASE,
)
_TERMINAL_MASTERY_RECAP_RE = re.compile(
    r"\b(?:(?:and|so)\s+)?now\s+you\s+know\s+how\s+to\b",
    re.IGNORECASE,
)
_TERMINAL_META_CONTINUATION_INCOMPLETE_RE = re.compile(
    r"\b(?:a|an|the)\s*$",
    re.IGNORECASE,
)
_TERMINAL_META_REQUIRED_OBJECT_RE = re.compile(
    r"^\s*(?:(?:i|they|we|you)\s+(?:can|could|may|might|should|will|would)\s+)?"
    r"(?:apply|calculate|compute|derive|determine|differentiate|evaluate|"
    r"explain|find|solve|use)\s*$",
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
_GENERIC_MODEL_START_RE = re.compile(
    r"^\s*in\s+this\s+(?:video|lesson|course)\s*[,]?\s+"
    r"we(?:['’]re|\s+are)\s+going\s+to\s*[,.!?;:]*\s*$",
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
FLASH_SPLIT_PROFILE = "flash_split_v3"
PRO_BOUNDARY_PROFILE = "pro_boundary_v13"
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

_TOTAL_DEADLINE_S = 75.0
_FLASH_SINGLE_TIMEOUT_S = 45.0
_FLASH_BOUNDARY_TIMEOUT_S = 45.0
_FLASH_REPAIR_TIMEOUT_S = 20.0
_FLASH_ENRICH_TIMEOUT_S = 25.0
_PRO_TIMEOUT_S = 90.0
_SELECTION_OUTPUT_TOKENS = 24_576
# Six thousand compact-schema tokens cover the exhaustive candidate payload.
_BOUNDARY_OUTPUT_TOKENS = 6_000
# Gemini Pro counts hidden thought tokens against max_output_tokens. Reserve a
# separate thought allowance so medium reasoning cannot consume the candidate
# payload budget; two Fast or three Slow 30k-token text-only selectors still fit
# their existing hard job-cost ceilings.
_PRO_BOUNDARY_OUTPUT_TOKENS = 12_288
_PRO_BOUNDARY_AUDIT_OUTPUT_TOKENS = 6_144
# Calls cheap enough under a byte-per-token upper bound may use the local
# estimate. Anything larger gets the provider's free exact count first, so two
# Fast or three Slow selectors cannot collectively under-reserve past the job
# ceiling even for high-entropy text.
_MAX_UNCOUNTED_SELECTOR_COST_USD = 0.20
# At low media resolution Gemini uses roughly one 66-token video frame plus
# 32 audio tokens per second. The rounded rate also covers timestamp metadata.
_LOW_RESOLUTION_VIDEO_TOKENS_PER_SECOND = 100
_BOUNDARY_REPAIR_OUTPUT_TOKENS = 1_024
_ENRICH_OUTPUT_TOKENS = 2_048
_MAX_CLIPS = 40
_MAX_SELECTOR_CANDIDATES = _MAX_CLIPS
_GREEN_SCORE = 0.75
_DUPLICATE_OVERLAP = 0.8
_SECTION_RESET_GAP_S = 8.0
_BOUNDARY_PAD_S = 0.3
_REPAIR_NEIGHBOR_CUES = 2
_BOUNDARY_REPAIR_PROMPT_VERSION = "boundary_repair_v1"
_PRO_BOUNDARY_AUDIT_PROMPT_VERSION = "pro_boundary_audit_v2"
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
    requirement: _NonBlank = Field(
        description=(
            "A context-complete statement of the requested requirement. A nested term such "
            "as a component, variable, unit, or step must retain its governing named object "
            "or relationship; never turn one shared word into an independent topical anchor."
        )
    )


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
        description=(
            "Five to sixteen exact consecutive transcript words that prove a primary "
            "unit fulfills the named constraint or ground a supporting unit's substantive "
            "educational connection to that constraint. The quote must belong to the same "
            "atomic teaching objective as sq, eq, cq, title, obj, and facet; never cite an "
            "earlier completed prerequisite or a later adjacent lesson. The proposition "
            "containing the quote must actually teach the context-complete requirement; "
            "using one requested word as an operand, symbol, unit, or example label does "
            "not ground it."
        ),
    )


class _CompactBoundaryTopic(_StrictModel):
    """Token-efficient production schema; attributes retain canonical names."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    candidate_id: _CompactCandidateId = Field(alias="id")
    start_line: int = Field(
        strict=True,
        alias="s",
        json_schema_extra={"minimum": 0},
    )
    end_line: int = Field(
        strict=True,
        alias="e",
        json_schema_extra={"minimum": 0},
    )
    start_quote: _CompactBoundaryQuote = Field(
        alias="sq",
        description=(
            "Shortest unique exact transcript quote whose first spoken word is the "
            "first word required by this complete one-topic unit and begins a complete, "
            "independently understandable spoken sentence or independent clause. Never "
            "begin at a prior sentence's trailing clause, complement, list item, or clipped "
            "completion. 'Shortest' controls only this matching quote's word count, never "
            "the semantic span or clip duration. Its first token must be a complete spoken "
            "word, never a caption-window fragment. It must occur uniquely inside the s:e "
            "range and may continue across adjacent caption lines; ignore acoustic silence "
            "and never include earlier speech for a pause."
        ),
    )
    end_quote: _CompactBoundaryQuote = Field(
        alias="eq",
        description=(
            "Shortest unique exact transcript quote whose final spoken word is the "
            "final required word after this unit's whole same-objective teaching arc. A "
            "locally complete sentence or intermediate result is not an endpoint when the "
            "same objective continues with reasoning, qualification, or explanation. Its "
            "last word must finish a complete concluding sentence or independent clause, "
            "never stop at a leading clause, sentence prefix, or unfilled predicate. "
            "'Shortest' controls only this matching quote's word count, never the semantic "
            "span or clip duration. Its final token must be a complete spoken word, never a "
            "caption-window fragment. If the word, sentence, answer, or same-objective "
            "reasoning continues in a later cue, increase e and continue to the complete "
            "conclusion. It must occur uniquely inside the s:e range and may begin across "
            "adjacent caption lines; ignore acoustic silence and never include later speech "
            "for a pause."
        ),
    )
    claim_quote: _CompactEvidenceQuote = Field(
        alias="cq",
        description=(
            "Five to sixteen exact consecutive transcript words containing the "
            "unit's substantive educational claim, explanation, or answer. It cannot "
            "be an agenda, outline, topic mention, promise, unanswered question, or "
            "a claim from an earlier background unit. It must teach the same atomic "
            "unit named by title, obj, and facet."
        ),
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
        description=(
            "For a primary unit, one grounded item proving fulfillment of every required "
            "non-scope constraint. For a supporting unit, at least one item grounding its "
            "substantive educational connection to a required non-scope constraint."
        ),
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
- The TOPIC is the viewer's original exact request copied verbatim. Never replace TOPIC with a
  search expansion, inferred broader subject, normalized paraphrase, or retrieval query.
- First understand the whole transcript and use that original exact request as the topical
  center. Return every distinct complete, substantive teaching unit that either fulfills the
  whole request or is genuinely related to its subject, method, mechanism, relationship,
  prerequisite, application, or worked-example family. Every unit must make sense to a cold
  viewer hearing the clip without seeing the original video.
- Include every indispensable unresolved premise and same-objective setup through the
  explanation's natural conclusion. Never prepend a separately complete prerequisite or
  background lesson merely because it appears earlier; return that as its own candidate when
  it qualifies. For a worked example, include the question or setup, reasoning, and answer.
- For a worked numerical example that solves for a dimensioned physical quantity, a bare
  number is not a complete answer. Continue through the contiguous spoken unit statement
  (for example, "two meters per second squared") and place eq after that unit. A later
  general explanation of the unit may also be its own candidate only when it independently
  contains substantive teaching beyond merely labeling the worked answer.
- Give each candidate exactly one coherent learning objective. Split every independent,
  genuinely related facet or worked example into its own candidate. For a constrained
  request, distinguish exact PRIMARY units from related SUPPORTING units; do not collapse the
  source to one literal match.
- An outline such as "three areas of calculus" or "two goals of this lesson" is navigation,
  not a teaching unit and not valid evidence. Return the substantive atomic units it names as
  separate candidates. Keep two concepts together only when their spoken relationship or
  comparison is itself the single teaching objective requested by the user.
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
- Trim filler from an opening, ending, or internal interruption whenever an exact contiguous
  cut preserves the complete teaching arc. Brief unavoidable filler does not invalidate an
  otherwise substantive, coherent unit. Omit only when filler or transcript noise dominates
  the material (for example, roughly 90 percent) or no coherent teaching remains.
- Course operations such as enrollment, pass/fail or grading rules, assignment and exam
  scheduling, registration, attendance, office hours, and course requirements are not
  educational context for the subject. Exclude them from clip openings and endings.
- A reference to a diagram, screen, gesture, drawing, or other unseen visual is not by itself
  a reason to omit a related substantive unit. Judge only the spoken transcript, never invent
  visual evidence, and mark self_contained and is_standalone false when the words alone do not
  establish the context.
- Exhaustively enumerate every distinct teaching unit genuinely related to the topical center,
  up to 40 per source. Prefer an empty slot to a merely broad-field, filler, or incomplete idea.
  Do not shorten a complete idea to fit a target length; clip duration is never a selection
  criterion.
- Keep distinct informational facets from the same source. Do not return two clips that
  teach the same learning objective in different words.
- Return every qualifying primary and supporting unit. Never stop after one exact match when
  the source contains other complete related units. When at least three distinct qualifying
  units exist, return all of them, not an arbitrary one. Never invent or fragment content to
  reach a count.
- Treat subject relevance and fulfillment of the user's requested operation, relationship,
  object, scope, format, and outcome as separate facts. A PRIMARY unit fulfills every required
  non-scope constraint. A SUPPORTING unit has a substantive educational connection to at least
  one required non-scope constraint and materially prepares, explains, verifies, applies, or
  deepens the requested topic. A
  supporting unit may use a different example object when it accurately teaches the same
  requested method or concept; title and objective must name the actual object, and intent
  evidence must never claim fulfillment of the mismatched object or outcome.
- A supporting unit must contain substantive explanation or a complete worked example. A bare
  formula, definition, topic name, result, generic background statement, or shared vocabulary
  is not enough. Shared broad-field relevance is not a topical connection.
- Every supporting unit must stay anchored to the request by either (a) teaching the same
  named subject, object, or relationship from the request, or (b) applying an explicitly named
  technical method or mechanism within the same subject family. Evidence must refer to the
  same thing and relationship as the request; repeating one variable, symbol, or noun is not
  enough. When a request is centered on a named law, equation, theory, or system, a component
  unit qualifies only when it independently teaches that requested component's meaning or
  explicitly connects it to the governing topic. Using the component only as an operand inside
  a different law or equation does not teach the requested component. A generic task or format
  such as explain, calculate, solve for a variable, show steps, give an example, or state units
  is never an anchor by itself. Sharing only the head word of a more specific phrase is also
  not enough: "force" does not ground "net force", and "units" does not ground the units of a
  named law. A different governing law, equation, theory, system, or domain is not supporting
  material merely because it uses the same algebra, variables, units, or vocabulary.
- Score informativeness, topic_relevance, and educational_importance honestly as metadata,
  never as numeric eligibility gates. Return every related, coherent, substantive teaching
  unit that satisfies the grounding, context, and filler rules above; omit it only under those
  spoken-content rules, not because a subjective score falls below a fixed threshold.
- Copy exact transcript line IDs and exact opening/closing quotes. start_quote must be the
  first words a cold viewer needs to hear for this one teaching objective, after every
  atmospheric hook, scene-setting flourish, or opening joke. end_quote must be the last
  words of its complete conclusion, before audience banter, a next-topic setup, or a
  post-conclusion joke. Quotes may begin or end inside a coarse transcript line. Copy the
  shortest unique 1-16 consecutive words that land exactly on the semantic edge, preserve
  the transcript spelling, and keep the quote wholly inside the selected s:e range. An exact
  phrase may cross directly adjacent caption lines, but never skip, stitch, or reorder words. A clean
  one-word edge is better than padding it with a transition, next topic, recap, or outro.
  Never pad to satisfy a preferred quote length. Never paraphrase or correct transcript words.
  topic_evidence_quote must be the
  shortest useful 5-16 consecutive words wholly between those chosen edges, copied with the
  same exactness.
- The semantic opening may not rely on an unresolved referent. An opening such as "this law",
  "this equation", "that change", "the second law", "it", "they", or "these" is invalid when
  the specific referent is not established inside the selected sq-through-eq span. The user
  request, page title, video title, and any previous clip are not spoken setup. Expand sq
  backward to the shortest complete spoken setup that names the referent. If the coarse
  transcript makes that exact edge uncertain, return the best whole grounded span and mark
  self_contained and is_standalone honestly; boundary uncertainty is not an omission reason.
- A worked example cannot end at its question, setup, or first substituted value. Either
  include its reasoning and answer through end_quote or end the candidate before that
  optional example begins.
- A worked numerical example for a dimensioned quantity cannot end at a unitless result.
  Include the contiguous spoken unit as part of its answer and end eq after the unit.
- A claim ending with an unfilled predicate such as "it can tell you" is incomplete. Continue
  through the spoken answer. A terminal "uh" or "um" is never a conclusion. Trim filler where
  exact boundaries permit, but preserve an otherwise coherent teaching arc when brief filler
  cannot be removed cleanly.
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
            "For an explicit comparison request, a primary unit must teach every named side "
            "and the requested relationship in one self-contained span. Also return complete "
            "supporting units that substantively teach a named side as their own objective in "
            "the requested comparison, or teach its explicitly requested comparison mechanism. "
        )
    elif "," in topic or ";" in topic:
        compound_rule = (
            "When the request lists multiple required ideas, a primary unit must fulfill all "
            "of them unless the wording presents alternatives. The list does not turn a "
            "component word into an independent topical anchor for supporting material. "
        )
    elif _EXPLICIT_TRANSITION_REQUEST_RE.search(topic):
        compound_rule = (
            "For an explicit transition request, a primary unit must name both endpoints and "
            "teach the transition between them. A supporting unit's own objective must teach "
            "a named endpoint's role in that transition or a requested transition mechanism. "
        )
    elif _EXPLICIT_CONJUNCTIVE_REQUEST_RE.search(topic):
        compound_rule = (
            "For an explicit conjunctive request, a primary unit fulfills every named "
            "component and constraint. A supporting unit's own objective must substantively "
            "teach a named component's meaning or role in the governing request. "
        )
    else:
        compound_rule = (
            "When the request names multiple linked ideas, tasks, objects, formats, or "
            "outcomes, a primary candidate fulfills every required part. Supporting candidates "
            "must make one required part or its role in the governing request their own "
            "substantive teaching objective; a directly connected example qualifies when its "
            "own objective applies the exact requested method or relationship. "
        )
    return (
        f"The TOPIC is the user's original prompt, exactly {topic!r}. Never substitute a "
        "retrieval expansion, search query, broader category, or inferred replacement topic. "
        "Return every complete unit that either fulfills this original prompt or has a direct "
        "educational connection to at least one of its actual subjects, tasks, methods, "
        "relationships, formats, or outcomes. Make each learning objective name what that "
        "specific clip truly teaches. Set directly_teaches_topic=true only for a primary unit "
        "that fulfills the full original prompt; use false for an honest supporting unit. Set "
        "it false as well when the span "
        "merely names the subject, course, institution, or speaker, or belongs to an adjacent "
        "field without a useful connection to the request. "
        f"{compound_rule}"
        "A listed component may be SUPPORTING only when the candidate's own atomic objective "
        "teaches that component's meaning or role in the exact request, explicitly connects "
        "it to the request's governing subject or relationship, or applies the exact requested "
        "technical method. Merely mentioning or using the component inside another objective, "
        "law, equation, theory, system, or domain is not support. A component that is the "
        "explicit subject of its own requested definition or explanation may qualify; a "
        "component merely used while teaching a different subject may not. "
        "When the topic requests "
        "identification, recognition, diagnosis, derivation, comparison, or application, "
        "a primary unit must actually perform that task for the named object and reach any "
        "requested result. Also return complete supporting units that teach the requested "
        "method or concept on another example, or materially explain a required mechanism, "
        "prerequisite, application, or verification. Do not return a bare history, definition, "
        "formula recital, or generic concept mention without substantive explanation. "
        "Treat a named mathematical function, equation, expression, chemical formula, code "
        "identifier, or other structured object as atomic. Any added, removed, or changed "
        "term, coefficient, exponent, sign, constant, variable, or condition makes it a "
        "different object. A different object cannot be primary and must never be mislabeled "
        "as the requested object, but a complete example using the same requested method may "
        "be an honestly labeled supporting unit. "
        "Shared vocabulary, a loose analogy, or general systems thinking is not request "
        "fulfillment."
        " Exclude fictional, supernatural, pseudoscientific, or invented mechanisms unless "
        "the viewer explicitly requested that fictional subject. Borrowing real academic "
        "terminology does not make an invented claim educational evidence."
    )


def _learner_rule(level: str) -> str:
    normalized = " ".join(str(level or "").split()).casefold()
    if normalized not in {"beginner", "intermediate", "advanced"}:
        return ""
    level_contracts = {
        "beginner": (
            "0.00-0.40",
            "assume no topic-specific background; the selected same-objective span must "
            "plainly establish every symbol, term, unresolved premise, and local setup it "
            "needs, but never prepend a separately complete prerequisite/background lesson",
        ),
        "intermediate": (
            "0.30-0.70",
            "assume foundational topic knowledge but no advanced specialization when "
            "judging the unit's current fit",
        ),
        "advanced": (
            "0.60-1.00",
            "assume strong foundations when judging the unit's current fit",
        ),
    }
    score_band, accessibility = level_contracts[normalized]
    return (
        f"The viewer's current level is {normalized}, whose current-fit difficulty band is "
        f"{score_band}: {accessibility}. Level is metadata, never selection eligibility. "
        "Return every otherwise qualifying relevant, substantive unit at every difficulty, "
        "score diff accurately after accounting for setup inside the clip, and never omit a "
        "unit merely because it is outside the current-fit band. The backend stores every "
        "returned unit and defers or reuses out-of-level units for a matching learner level."
    )


def _selection_fields(*, enriched: bool, compact: bool = False) -> str:
    fields = (
        "candidate_id (a short unique slug), start_line and end_line (the enclosing cue range), "
        "start_quote and end_quote (each the shortest unique 1-16 exact consecutive words "
        "inside that range, preserving transcript spelling; the first word of start_quote and "
        "last word of end_quote are the authoritative semantic edges, even when an exact phrase "
        "crosses directly adjacent caption lines; never pad, paraphrase, skip, stitch, or reorder), "
        "title (at most 12 words), "
        "learning_objective (at most 24 words), facet (at most 12 words), "
        "informativeness, topic_relevance, "
        "educational_importance, difficulty, directly_teaches_topic, substantive, "
        "factually_grounded"
    )
    if not compact:
        fields += (
            ", topic_evidence_quote (the shortest exact 5-16 consecutive-word quote copied "
            "wholly between the chosen edges that proves the clip teaches the topic; preserve "
            "spelling and never paraphrase or stitch), self_contained, is_standalone"
            ", prerequisite_candidate_ids (omit it or return []), uncertainty "
            "(omit for low), uncertainty_reasons (omit for low)"
        )
    else:
        fields += (
            ", self_contained, is_standalone"
            ". Use the compact schema keys: id=candidate_id, s=start_line, e=end_line, "
            "sq=start_quote, eq=end_quote, cq=claim_quote (the shortest exact 5-16 word "
            "substantive educational assertion, explanation, or answer inside the unit; "
            "never an agenda, outline/list header, promise, mere topic mention, or "
            "unanswered question; a sentence explicitly distinguishing two named sides is "
            "a substantive relationship claim, not an outline), obj=learning_objective, "
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


def _compact_output_guide() -> str:
    """Explain every compact selector key and demonstrate exact edge semantics."""
    return """Compact topic schema — use these exact keys:
- id = candidate_id: a short unique slug for this one educational moment. Never reuse an id.
- s = start_line: the zero-based, inclusive index of the first transcript line enclosing the
  desired start. This is a caption-line index, not seconds and not necessarily the exact start.
- e = end_line: the zero-based, inclusive index of the last transcript line enclosing the
  desired end. The enclosing caption range is every line from s through e.
- sq = start_quote: 1-16 exact consecutive transcript words inside s:e. Its FIRST spoken word
  is the exact first word the viewer should hear and must begin a complete independently
  understandable spoken sentence or independent clause—never a trailing clause, complement,
  list item, or clipped completion from the preceding sentence. sq may cross adjacent lines.
  Do not include earlier greeting, filler, transition, or context solely to make the quote
  longer or unique. "Shortest" describes only the quote used to locate that word; it never
  asks for a shorter semantic span or clip.
- eq = end_quote: 1-16 exact consecutive transcript words inside s:e. Its LAST spoken word is
  the exact final required word after the whole same-objective teaching arc. A grammatical
  sentence or intermediate answer is not an endpoint when its reasoning, qualification, or
  explanation continues. That last word must finish a complete concluding sentence or
  independent clause—never a leading clause, sentence prefix, or unfilled predicate. eq may
  cross adjacent lines. Do not include a later transition, recap, outro, joke, or next topic.
  "Shortest" describes only the quote used to locate that word; it never asks for a shorter
  semantic span or clip.
- cq = claim_quote: 5-16 exact consecutive transcript words between the chosen semantic edges
  containing the core educational claim, explanation, result, or answer. cq proves where the
  teaching is; it does not define the start or end and cannot be mere agenda, topic mention, or
  a claim from an earlier prerequisite/background lesson. sq, cq, title, obj, and facet must all
  describe the same atomic educational unit.
- title = a clear viewer-facing title of at most 12 words.
- obj = learning_objective: one precise sentence, at most 24 words, stating what the viewer
  will understand or be able to do after this clip. Give exactly one objective.
- facet = the narrow subtopic taught here, at most 12 words; be more specific than the broad
  user request whenever possible.
- info = informativeness, 0.0-1.0: how much concrete, useful teaching the selected span contains.
- rel = topic_relevance, 0.0-1.0: how strongly this exact unit serves the exact user request.
- imp = educational_importance, 0.0-1.0: how valuable this unit is for learning that request.
- diff = difficulty, 0.0-1.0: required prior knowledge only; 0.00-0.33 beginner,
  0.34-0.66 intermediate, 0.67-1.00 advanced. Difficulty never disqualifies an otherwise
  qualifying unit; the backend stores it and decides when its learner level should surface.
- direct = directly_teaches_topic: true only for a PRIMARY unit that fulfills every required
  non-scope constraint in the original prompt. Use false for a valid SUPPORTING unit; false is
  honest metadata and does not mean omit the supporting unit.
- sub = substantive: true only for real teaching, reasoning, explanation, demonstration, or
  an answer—not greetings, administration, promotion, navigation, or empty framing.
- fact = factually_grounded: true only when the educational claim is supported by the supplied
  spoken transcript. No video is supplied, so never assume an unseen visual proves the claim.
- self = self_contained: true when all setup, referents, reasoning, and conclusion needed to
  understand the objective are included inside sq-through-eq.
- stand = is_standalone: true when a cold viewer can understand and use the clip independently,
  without first watching another part of the source. self concerns included context; stand
  concerns independence from the surrounding lesson or another prerequisite clip.
- ie = intent_evidence: a nonempty list of {id, q} objects. Each q must be an exact consecutive
  5-16-word quote inside the candidate. For a PRIMARY unit, ie covers every required non-scope
  id and each q proves fulfillment of that constraint. For a SUPPORTING unit, ie contains at
  least one id and each q grounds the unit's substantive educational connection to that
  constraint; it need not claim full fulfillment. At least one supporting ie item must ground
  the same named subject, object, or relationship, or an explicitly named technical method used
  in the same subject family. Every q must ground the SAME referent and relationship named by
  its id, not merely contain one of its words, symbols, variables, dimensions, or units. Apply
  a same-referent test before adding every item: read the constraint's source_phrase and q
  literally; if q teaches a different law, equation, relationship, object, or use of a shared
  variable, it cannot ground that constraint. Every ie q must teach the same atomic objective
  as sq, eq, cq, title, obj, and facet; never use evidence from an earlier completed
  prerequisite or later adjacent lesson. Generic
  task, format, or outcome evidence alone is insufficient. Never use supporting ie to falsely
  claim a mismatched object or outcome. Do not output a role; the
  backend mechanically derives primary
  only from direct=true plus full grounded coverage, otherwise supporting, and never rejects a
  unit based on that role.
  For a named expression or formula, q may claim its object id only when it includes the full
  spoken expression, including trailing terms. A different expression may still support a
  method id, but must not claim or masquerade as the requested object.
  When a fulfilled format constraint spans multiple transformations, q anchors that sequence;
  verify every requested transformation across the entire sq-through-eq span. One q never
  replaces the whole-span completeness check.

Subject-anchor counterexamples — apply the same rule in every domain:
- For a biology request about PCR, a general DNA-replication unit does not qualify merely
  because both mention DNA or polymerase. It qualifies only if the selected span explicitly
  teaches PCR itself or connects the general mechanism to a requested PCR step.
- For a law request about negligence, a contract-breach unit does not qualify merely because
  both mention duties or damages. Its ie quote must teach the same legal test or an explicitly
  requested prerequisite, not a neighboring doctrine with shared words.
- For a software request about quicksort, a merge-sort unit does not qualify merely because
  both use divide-and-conquer. It qualifies only when the span teaches quicksort or explicitly
  compares the algorithms in service of the requested quicksort objective.
The rule is referent identity, not vocabulary overlap: shared symbols, variables, units, names,
or generic methods cannot stand in for the object and relationship in the original request.

Worked format example — understand the boundary logic, but never copy its content or scores:
All examples below assume no learner-level restriction. They demonstrate only schema and
boundary behavior. Their diff values never override accurate scoring for the actual span.
Return an otherwise qualifying unit at any difficulty; the backend stores it and decides when
its learner level should surface.
Example transcript:
[18] 02:10 Welcome back. A small p value
[19] 02:15 provides stronger evidence against the null hypothesis. Next, confidence intervals measure uncertainty.
Example exact user request: Explain p-values
Example output:
{"request_intent":{"exact_request":"Explain p-values","constraints":[{"constraint_id":"subject","kind":"subject","source_phrase":"p-values","requirement":"Explain p-values"}]},"topics":[{"id":"small-p-value-evidence","s":18,"e":19,"sq":"A small p value","eq":"against the null hypothesis.","cq":"small p value provides stronger evidence against the null hypothesis","title":"What a Small P-Value Means","obj":"Explain how a small p-value bears on the null hypothesis","facet":"small p-values","info":0.95,"rel":0.99,"imp":0.93,"diff":0.42,"direct":true,"sub":true,"fact":true,"self":true,"stand":true,"ie":[{"id":"subject","q":"small p value provides stronger evidence against the null hypothesis"}]}]}
Why: s remains 18 even though sq starts after "Welcome back" inside line 18. e remains 19
even though eq ends before "Next, confidence intervals" inside line 19. sq's first word "A"
is the semantic start; eq's last word "hypothesis" is the semantic end; cq anchors the claim.

Context example:
[31] 04:00 To calculate acceleration, first find the change in velocity.
[32] 04:06 Divide that change by elapsed time to obtain acceleration.
Do not start at line 32 with "Divide that change" because "that change" depends on omitted
setup. Include line 31 and begin sq at "To calculate acceleration" so self and stand are true.

Named-referent context example:
[50] 08:00 Newton's second law describes how net force changes an object's motion.
[51] 08:06 This law tells us force equals mass times acceleration.
Do not start at line 51 with "This law tells us". The original request and video title are not
spoken context. Include line 50 and begin sq at "Newton's second law" so the referent is audible
inside the clip.

Split-caption boundary examples:
- If one cue ends "The treaty was signed after the" and the next begins "delegates reached a
  compromise", WRONG sq="delegates reached a compromise". Begin with the complete treaty
  sentence and any audible setup it requires.
- If one cue ends "A dormant seed begins to" and the next begins "germinate after absorbing
  water", WRONG sq="germinate after absorbing water". Begin with the complete subject.
- If one cue ends "Suppose the cache contains an old" and the next begins "version when the
  writer arrives", WRONG sq="version when the writer arrives". Begin sq at the complete
  scenario setup, regardless of how many caption cues split it.
- If a cue begins "answer?" because the preceding cue ends "What is the", include the complete
  spoken question and its concrete scenario; never begin sq at "answer?" or at the answer alone.
Do not treat caption-line starts, acoustic silence, or punctuation inside a rolling caption as
proof of a complete semantic start. Read the neighboring words and choose the first word of the
whole spoken sentence, scenario, question, or explanation.

Background-detour boundary example:
[60] 10:00 DNA stores hereditary information in a sequence of bases.
[61] 10:08 Cells copy DNA before division.
[62] 10:16 With that background, PCR uses repeated temperature cycles to amplify a chosen DNA
region through denaturation, primer binding, and extension.
For a PCR candidate whose obj is "Explain the PCR cycle", lines 60-61 are earlier completed
background units, not permission to make the PCR clip begin with a general DNA lesson. Use s=62,
sq="PCR uses repeated temperature cycles", and a cq about the PCR cycle. WRONG: title/obj/facet
describe PCR while sq or cq begins in the earlier general lesson. Also WRONG: a PCR candidate
uses ie q="Cells copy DNA before division"; ie cannot pull a completed prerequisite into a
different objective. This rule still applies for a beginner viewer. If an earlier prerequisite
is independently relevant, return it as its own supporting candidate; do not fold the whole
completed prerequisite into the later target unit merely because it comes first.

Named worked-example boundary example:
[40] 06:00 This is simply equal to five. So that's the derivative of five x minus four. It's five. Now let's try another example. So let's say if f of x is equal to x squared, what is the first derivative?
[41] 06:08 Use the limit definition of the derivative.
[42] 06:16 Substitute x plus h to get open parenthesis x plus h close parenthesis squared minus x squared over h.
[43] 06:26 Expand the square to x squared plus two x h plus h squared, then cancel x squared.
[44] 06:36 Divide by h and cancel the common h to obtain two x plus h.
[45] 06:44 Taking h to zero gives two x, so the derivative of x squared is two x.
Exact request: Use the limit definition to derive f of x equal x squared, include every
algebra step, and finish at two x.
"This is simply equal to five" completes the earlier five-x-minus-four objective, and "Now
let's try another example" is navigation. Both occur inside line 40 but are outside the
requested semantic unit. Begin at the x-squared setup later inside that same coarse line,
preserving its first "So": s=40 and sq="So let's say if f of x is equal to x squared".
WRONG: s=40 and sq="This is simply equal to five".
WRONG: s=41, because that omits the named x-squared setup.
End at the final two-x result in line 45. Lines 42-44 are required reasoning, so never replace
them with a formula-only clip, a summary, another function, or a general derivative explanation.
An x-squared-minus-three example is also a different function and does not qualify as the
primary x-squared unit even though its derivative also simplifies to two x. It cannot
claim the object id. If it contains a complete limit-definition derivation, it may be a
supporting unit for the method only, accurately titled for x squared minus three.
Example output boundaries: s=40, e=45, sq="So let's say if f of x is equal to x squared",
eq="the derivative of x squared is two x". The topic's ie must evidence the named function,
limit-definition task, algebra-step requirement, and final two-x outcome.
Example compact output:
{"request_intent":{"exact_request":"Use the limit definition to derive f of x equal x squared, include every algebra step, and finish at two x.","constraints":[{"constraint_id":"object","kind":"subject","source_phrase":"f of x equal x squared","requirement":"Derive f of x equal x squared"},{"constraint_id":"method","kind":"task","source_phrase":"Use the limit definition","requirement":"Use the limit definition"},{"constraint_id":"steps","kind":"format","source_phrase":"include every algebra step","requirement":"Include every spoken algebra step"},{"constraint_id":"result","kind":"outcome","source_phrase":"finish at two x","requirement":"Reach the final result two x"}]},"topics":[{"id":"x-squared-limit-derivation","s":40,"e":45,"sq":"So let's say if f of x is equal to x squared","eq":"the derivative of x squared is two x","cq":"Taking h to zero gives two x","title":"Derive x Squared from the Limit Definition","obj":"Derive the derivative of x squared through every algebra step","facet":"x-squared limit derivation","info":0.99,"rel":1.0,"imp":0.99,"diff":0.45,"direct":true,"sub":true,"fact":true,"self":true,"stand":true,"ie":[{"id":"object","q":"f of x is equal to x squared"},{"id":"method","q":"Use the limit definition of the derivative"},{"id":"steps","q":"Expand the square to x squared plus two x h"},{"id":"result","q":"Taking h to zero gives two x"}]}]}

Same-source breadth for that original prompt:
If the same transcript also completely derives five x minus four, one over x, square root of
x, eight over square root of x, or another function by the limit definition, return EACH as
its own SUPPORTING clip. Give every clip its actual function and actual result in title and
objective, set direct=false, and include method/format ie only when grounded. Do not claim the
x-squared object or two-x result for a different function. Also return complete explanatory
units about what h means, substitution, expansion, cancellation, or direct substitution when
they materially teach a required part. Omit formula-only recitals, fragments, generic calculus
statements, and duplicate restatements. Never stop after returning the primary x-squared clip.
"""


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
    video_grounded: bool = False,
) -> tuple[str, str]:
    # Kept in the signature for compatibility with older callers, but selector
    # prompts are permanently transcript-only to prevent video-token charges.
    video_grounded = False
    system = (
        "You select self-contained educational clip boundaries from timestamped transcripts.\n\n"
        "INPUT CONTRACT: You receive transcript text only. No video, image, audio file, "
        "frames, thumbnails, or visual metadata are attached. Judge only the supplied "
        "transcript and never infer that an unseen visual supplies missing context or "
        "evidence.\n\n"
        + _POLICY_AND_EXAMPLES
    )
    if video_grounded:
        system += (
            "\n\nAttached-video grounding for this selector call:\n"
            "- Inspect the audio and visual streams jointly over each candidate span. The "
            "viewer will see and hear the selected video clip, so this attached-video rule "
            "replaces transcript-only assumptions about an unseen original video.\n"
            "- You may resolve formulas, diagrams, on-screen text, gestures, or deictic "
            "speech from visuals only when the required visual is present and legible within "
            "the candidate timestamps. If it is absent or illegible, omit the unit.\n"
            "- Visual evidence never replaces transcript boundary grounding: line IDs plus "
            "sq, eq, cq, and every ie q must remain exact consecutive transcript quotes.\n"
            "- Set factually_grounded=true only when both the transcript and any required "
            "visual evidence consistently support the teaching claim."
        )
    learner_rule = _learner_rule(learner_level)
    learner_line = f"{learner_rule}\n\n" if learner_rule else ""
    cold_start_basis = (
        "the selected span's spoken audio together with its visible, legible video"
        if video_grounded
        else "only the supplied spoken transcript"
    )
    visual_rule = (
        "Use required visuals only when they are present and legible inside the selected "
        "span; otherwise omit the unit.\n"
        if video_grounded
        else (
            "Do not omit a related substantive unit solely because its speech references an "
            "unseen visual; judge only the transcript, never invent visual evidence, and set "
            "self and stand honestly.\n"
        )
    )
    factual_rule = (
        "For video-dependent teaching, factually_grounded covers both streams and may be "
        "true only when the transcript and required visible evidence agree. "
        if video_grounded
        else ""
    )
    exact_request = topic.strip() or "(all educational topics)"
    user = (
        f"Transcript ({n} lines, formatted `[index] MM:SS text`; valid line IDs are "
        f"0 through {n - 1}):\n{lines}\n\n"
        f"Exact user request: {exact_request}\n"
        f"{_topic_rule(topic)}\n\n"
        f"{learner_line}"
        "Task:\n"
        "1. Interpret the exact request before selecting anything. Return request_intent with "
        "exact_request copied exactly from the Exact user request above and 1-8 atomic "
        "constraints. Give every constraint a unique constraint_id, its kind, a concise "
        "requirement, and source_phrase copied as exact consecutive words from that request. "
        "Together the source phrases must cover every content-bearing request term. Preserve "
        "named subjects, requested operations or tasks, relationships, scope qualifiers, "
        "formats, and outcomes. This verbatim exact_request is TOPIC for all selection and "
        "relevance decisions. Do not substitute retrieval expansions or a broader topic. "
        "When a request contains a function, equation, expression, formula, identifier, or "
        "other structured object, keep the complete object in one atomic constraint; never "
        "reduce it to one shared term. "
        "Treat course, exam, grade, learner-level, and curriculum labels as scope constraints, "
        "not spoken subject constraints; the exact request and supplied transcript establish "
        "those qualifiers. Words such as 'every', 'each', 'all', 'step-by-step', 'including "
        "every algebra step', and 'final result' are required FORMAT or OUTCOME constraints, "
        "never optional SCOPE constraints. A PRIMARY candidate fulfills such a request only "
        "when one contiguous span contains the named setup, every spoken transformation, and "
        "the requested final result. A supporting candidate may connect to only the constraints "
        "its grounded ie honestly identifies. "
        "Then scan the whole transcript from first to last and understand it before selecting. "
        "Internally distinguish required setup and teaching from administration, promotion, "
        "navigation, repetition, and visual-dependent speech; do not output that section map.\n"
        "2. Map every distinct educational unit genuinely related to TOPIC across the whole "
        "transcript. A PRIMARY unit fulfills every required non-scope constraint in one "
        "complete span. A SUPPORTING unit has a substantive educational connection to at "
        "least one required non-scope constraint and materially teaches a related example, "
        "method, mechanism, prerequisite, "
        "application, verification, comparison side, or explanatory facet. Return supporting "
        "units even when this source contains no primary unit. A different example object may "
        "support the same requested method, but its title, objective, cq, and ie must describe "
        "what it actually teaches and must not claim the requested object or outcome. Return "
        "every distinct qualifying primary and supporting moment, not merely the closest or "
        "first exact match. If one transcript contains six complete related worked examples, "
        "return six separate candidates. Never invent, duplicate, fragment, or lower the "
        "quality bar merely to produce more clips. Return qualifying moments up to "
        f"{_MAX_SELECTOR_CANDIDATES} for this source; "
        "do not stop after the first few units or at an arbitrary count below that cap.\n"
        "3. For every qualifying unit, verify its timestamps and choose one whole coherent "
        "teaching arc containing all contiguous same-objective setup, reasoning, answer, "
        "qualification, and explanation through its natural conclusion. Context and wholeness "
        "have absolute priority over concision. Only choose a tighter boundary when two spans "
        "are equally complete and no required word, context, reasoning, answer, qualification, "
        "or explanation is lost; if completeness and brevity conflict, choose the longer complete "
        "span. A locally grammatical sentence or intermediate result is not an endpoint when "
        "the same objective continues. There is no numeric duration cap. "
        "Duration must be the consequence of the exact semantic scope; it is not permission "
        "to include an earlier completed example or a later adjacent topic. Preserve all "
        "necessary setup and context, "
        "even when that makes the clip longer. End immediately after the whole teaching arc's "
        "natural conclusion and before a new concept begins. Choose "
        "the exact semantic first and final required spoken words and ignore acoustic silence "
        "when choosing them. Do not widen or shorten the semantic unit to guess a pause: "
        "downstream audio processing may only expand outward from your exact interval to "
        "nearby verified silence and may never remove speech you selected. Adjacent named concepts, "
        "procedures, worked examples, decision rules, misconceptions, and error cases are "
        "new-concept boundaries unless their explicit relationship is the single requested "
        "objective. Give it exactly one learning objective. Split independent adjacent "
        "facets into separate candidates even when they share one coarse transcript line. "
        "Never combine the end of one objective with the beginning of another: end before "
        "the transition for the old unit and start after it for the new unit. When a coarse "
        "line contains the old conclusion, transition, and new opening, place the exact edge "
        "quotes inside the selected range on the correct side of the transition. "
        "When one coarse line contains previous-objective conclusion, navigation, then the "
        "requested setup, keep s as that shared line but sq MUST begin at the first word of "
        "the requested setup inside it. Never begin sq at the previous conclusion or "
        "navigation. A clause such as 'So let's say if f' that states the requested object is "
        "setup, not context-only handoff; preserve its first 'So'. "
        "Keep sq, eq, cq, every ie q, title, obj, and facet on the same atomic educational "
        "unit. An earlier "
        "complete prerequisite or background explanation is a separate unit, not required "
        "context merely because it helps introduce the later target. Return that background "
        "separately when it qualifies, and begin the target candidate at the complete spoken "
        "setup required by that target's own objective. "
        "Never start a candidate with context-only handoff language such as 'on the other "
        "hand', 'now/then the next step', or 'so step five'; include the "
        "missing setup or begin at the first independently understandable teaching sentence. "
        f"Apply a cold-start and cold-stop test using {cold_start_basis}: never begin at a "
        "dependent tail, trailing clause, complement, list item, clipped sentence completion, "
        "tag question, unresolved 'other one' reformulation, or the "
        "consequence of an analogy whose actors and mapping were omitted. Never end inside "
        "a leading clause, sentence prefix, unfilled predicate, list, next-topic phrase, recap, "
        "contextual bridge, or outro. Use a one-word quote "
        "when that is the exact clean edge; never pad an edge quote with nearby speech. "
        "An opening demonstrative or pronoun such as 'this law', 'this equation', 'that "
        "change', 'the second law', 'it', 'they', or 'these' must have its specific referent "
        "named inside sq-through-eq; the request, page title, video title, title, obj, facet, "
        "cq, ie, and a previous clip are metadata and do not supply that spoken context. "
        "Expand backward to the shortest complete setup "
        "that names it. Keep opening and ending edges clean, including generic lead-ins and bracketed "
        "non-speech markers. Split around an internal interruption when separate complete "
        "units remain and return only contiguous complete sides. If an exact clean cut is "
        "uncertain, preserve the whole substantive teaching arc and return the closest exact "
        "grounded boundaries; brief unavoidable filler is not a rejection reason. "
        "Before returning, verify every sq and eq token-for-token inside its selected s:e range, "
        "including the 1-16 word limit, and verify every cq is 5-16 exact words. The backend "
        "treats every topic you return as semantically approved: it will not second-guess or "
        "reject that topic. The first word of sq and last word of eq are final semantic edges: "
        "the backend will not move either edge inward or use punctuation, topic vocabulary, "
        "caption length, or silence to reinterpret it. It may only widen outward to include an "
        "exact cq/ie quote you supplied or fall back to your enclosing s:e cue edges when an "
        "edge quote is missing, repeated, or malformed. Do not rely on downstream code to fix "
        "an incomplete thought, omitted referent, late start, early stop, or extra adjacent unit. "
        "Perform a final cold-start/cold-stop reread from sq through eq yourself and correct "
        "those words before returning. "
        "Do not omit an otherwise good, complete, relevant unit solely because coarse captions "
        "make the exact cut uncertain; return your closest grounded s:e, sq, and eq so the "
        "backend can repair the cut. Boundary uncertainty alone is never an omission reason. "
        "Omit only when the spoken material itself is unrelated or lacks a substantive, "
        "coherent teaching unit, such as material dominated by filler or transcript noise. "
        f"{visual_rule}"
        "4. Score topic relevance, information density, educational value, and difficulty "
        "honestly. These scores are metadata, never numeric eligibility gates. Return every "
        "related, coherent, substantive teaching unit admitted by the spoken-content rules "
        "above. Difficulty records prior knowledge only: 0.00-0.33 means beginner, "
        "0.34-0.66 means intermediate, and 0.67-1.00 means advanced. Difficulty is always "
        "metadata, never an eligibility filter. Qualifying "
        "units may span the entire scale; the backend stores and defers units outside the "
        "current learner's fit.\n"
        "5. Return every distinct qualifying unit. Set substantive and factually_grounded true "
        "only for academically sound teaching; "
        f"{factual_rule}"
        "course logistics and institutional framing are "
        "not teaching units. Each unit must be a coherent educational moment, use a unique "
        "candidate_id, include the best available setup inside its span, and set self and "
        "stand honestly. A false self or stand value alone is not an omission reason.\n"
        f"{_compact_output_guide()}\n"
        "MANDATORY FINAL ADMISSION AUDIT (silent): For each proposed topic, answer from its "
        "obj, cq, and ie: 'What concrete new ability or understanding for the exact request "
        "does this clip teach?' Keep the topic only when the answer names a requested subject, "
        "role, relationship, technical method, step, or application and the cited speech "
        "actually teaches it. Omit the topic when the only honest connection is a shared noun, "
        "variable, symbol, unit, broad field, generic task or format, or use of a requested "
        "component inside a different governing law, equation, theory, system, or domain. A "
        "nested component qualifies only when this candidate's own atomic objective and cq "
        "teach its meaning or role in the requested governing topic, explicitly connect it to "
        "that topic, or apply the exact requested technical method. Score rel from that actual "
        "objective and teaching claim, never from an isolated matching word in ie. Prefer no "
        "topic from a loosely related source over a broad-field match. Do not output this audit.\n"
        "MANDATORY FINAL EDGE AUDIT (silent): Read the literal sq-through-eq span as a cold "
        "listener and inspect its neighboring cues. sq's first token and eq's last token must "
        "be complete spoken words, not cue or window fragments. Caption boundaries are never "
        "semantic boundaries. If a word, sentence, reasoning chain, answer, qualification, or "
        "same-objective explanation continues into a later cue, increase e and continue to the "
        "first complete same-objective conclusion. If the tail starts a new subject with a "
        "coordinator or transition but does not complete its predicate—for example, 'And the "
        "comparison subject...' or 'But the second case...'—either extend through that required "
        "complete thought when it belongs to the same objective or move eq back before that "
        "tail. If the supplied transcript itself ends before completion, keep the otherwise "
        "good candidate at the widest grounded cue range, set self and stand honestly, and "
        "never omit it solely for boundary uncertainty. Do not output this audit.\n"
        f"Return only the object {{request_intent, topics}}. Every topic must contain "
        f"{_selection_fields(enriched=False, compact=True)}. The ie list must be nonempty and "
        "use {id, q} items where id is the constraint_id and q is an exact consecutive 5-16 "
        "word transcript quote wholly inside the candidate. For a primary topic, ground every "
        "required non-scope constraint and prove its fulfillment. For a supporting topic, "
        "include at least one non-scope id whose q grounds the substantive educational "
        "connection; it need not prove full fulfillment, but it must never falsely claim a "
        "mismatched object or outcome. Use direct=true for primary and direct=false for supporting. "
        "Do not output a role. Omit scope constraints from ie unless the selected speech states "
        "them exactly. cq independently proves that the candidate teaches a substantive atomic "
        "claim; ie proves its connection to the original prompt and must never substitute for "
        "cq. Learning details and "
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
        "always return the best grounded boundary you can find for each candidate even when "
        "no perfectly clean self-contained cut exists, and copy each edge quote exactly from "
        "its selected cue. Never omit an already selected candidate because boundary repair "
        "is uncertain. Do not summarize, enrich, or add assessments."
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


def _pro_boundary_audit_prompts(
    plan: _CompactBoundaryPlan,
    segments: list[dict],
    topic: str,
) -> tuple[
    str,
    str,
    dict[str, tuple[int, set[int], set[int]]],
]:
    """Render one text-only, non-dropping audit of Gemini's own word edges."""
    allowed: dict[str, tuple[int, set[int], set[int]]] = {}
    blocks: list[str] = []
    cue_count = len(segments)
    transcript_lines = set(range(cue_count))
    for index, candidate in enumerate(plan.topics):
        audit_id = f"candidate-{index + 1}"
        start_lines = set(transcript_lines)
        end_lines = set(transcript_lines)
        allowed[audit_id] = (index, start_lines, end_lines)
        blocks.append(
            f"<candidate audit_id={audit_id!r}>\n"
            f"original id: {candidate.candidate_id}\n"
            f"title: {candidate.title}\n"
            f"learning objective: {candidate.learning_objective}\n"
            f"facet: {candidate.facet}\n"
            f"current s/e: {candidate.start_line}/{candidate.end_line}\n"
            f"current sq: {candidate.start_quote!r}\n"
            f"current eq: {candidate.end_quote!r}\n"
            f"allowed start/end line ID range: 0-{cue_count - 1}\n"
            "</candidate>"
        )

    rendered_cues = "\n".join(
        f"[{index}] {_mmss(segments[index].get('start', 0.0))} "
        f"{str(segments[index].get('text') or '').strip()}"
        for index in range(cue_count)
    )
    system = (
        "You are the final boundary auditor for educational clips already selected by "
        "Gemini. You receive transcript text only; no video, image, audio, URL, frame, or "
        "visual metadata is attached. Preserve every candidate and its semantic objective. "
        "You may correct only start_line, end_line, start_quote, and end_quote. Never add, "
        "remove, reject, merge, split, rank, or semantically reclassify a candidate. Return "
        "exactly one item for every audit_id, even when the best answer is to repeat its "
        "current boundaries. Do not provide reasoning."
    )
    user = (
        f"Exact user request: {topic.strip() or '(all educational topics)'}\n\n"
        "<full_transcript_cues>\n"
        f"{rendered_cues}\n"
        "</full_transcript_cues>\n\n"
        + "\n\n".join(blocks)
        + "\n\nFor every candidate, reread its literal sq-through-eq speech as a cold "
          "listener. Its first and final tokens must be complete spoken words. The opening "
          "must include the candidate's own necessary setup and resolved referents. The ending "
          "must finish its same-objective sentence, reasoning, answer, qualification, and "
          "conclusion. Caption punctuation, cue edges, and silence are not semantic endings. "
          "If a word or thought continues into a later cue, increase end_line and end only at "
          "the first complete same-objective conclusion. If the current tail begins a new "
          "clause or subject but leaves its predicate unfinished, either include that whole "
          "same-objective thought or move end_quote back before the incomplete tail. Never "
          "stop after a coordinator plus a newly introduced subject, such as 'And the second "
          "subject'; continue through its predicate, or stop before the coordinator. If the "
          "transcript corrupts a final word and then "
          "starts a different objective, do not invent the missing letters and do not append "
          "the different objective; move end_quote back to the last complete conclusion for "
          "this candidate. Context and wholeness outrank brevity, and there is no duration "
          "target. Boundary uncertainty is never a reason to omit a candidate.\n\n"
          "Do not widen an already complete opening merely to include navigation or generic "
          "framing such as 'Now the next law you need to know' or 'Now here is another "
          "example.' When the following speech itself names the law, object, or scenario, "
          "begin at that complete teaching setup and leave the navigation outside.\n\n"
          "Each returned candidate_id must be the audit_id exactly. start_line and end_line "
          "must be valid IDs from the supplied full transcript. "
          "start_quote and end_quote must each be the shortest unique 1-16 exact consecutive "
          "transcript words inside the returned inclusive line range. Their first/last words "
          "are the semantic edges. Quotes may cross adjacent cues but may not skip, stitch, "
          "reorder, paraphrase, or correct transcript text. Return only {items}."
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

    if (
        _OPENING_RELATIVE_WHERE_FRAGMENT_RE.match(str(text or ""))
        or _OPENING_SUBORDINATE_PRONOUN_REFERENCE_RE.match(str(text or ""))
        or _OPENING_RELATIVE_BRIDGE_RE.match(str(text or ""))
        or _OPENING_SUBJECTLESS_ADJECTIVE_COMPLEMENT_RE.match(str(text or ""))
    ):
        return True
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


def _next_cue_completes_embedded_predicate(text: str, next_text: str) -> bool:
    """Recognize a caption split between an embedded subject and its predicate."""
    raw_text = str(text or "").strip()
    if not _NOMINAL_PREDICATE_CONTINUATION_RE.match(str(next_text or "")):
        return False
    complementizers = list(
        re.finditer(r"\b(?:that|whether)\b", raw_text, re.IGNORECASE)
    )
    if not complementizers:
        return False
    tail = raw_text[complementizers[-1].end():].strip(" ,;:—-")
    if not tail:
        return False
    adjunct = _EMBEDDED_CLAUSE_ADJUNCT_RE.search(tail)
    subject = tail[:adjunct.start()].strip() if adjunct is not None else tail
    subject_words = _toks(subject)
    return bool(
        1 <= len(subject_words) <= 12
        and not _FINITE_PREDICATE_SIGNAL_RE.search(subject)
    )


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
    if _terminal_content_is_explicitly_incomplete(raw_text):
        return True
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
    if (
        not classify_terminator(raw_text)
        and _next_cue_completes_embedded_predicate(raw_text, next_text)
    ):
        return True
    dependent_complement = _NEXT_DEPENDENT_COMPLEMENT_RE.match(
        str(next_text or "")
    )
    if dependent_complement is not None:
        next_opening = dependent_complement.group(0).strip().casefold()
        conditional_complement = next_opening.startswith(
            ("assuming", "conditional", "given", "provided", "under")
        )
        if (
            (not classify_terminator(raw_text) or conditional_complement)
            and (
                not next_opening.startswith("of")
                or (
                    _OPENING_INDEPENDENT_OF_FRAME_RE.match(str(next_text or ""))
                    is None
                    and _TERMINAL_OF_COMPLEMENT_HEAD_RE.search(raw_text)
                )
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
    next_words = _toks(next_text)
    if (
        not classify_terminator(raw_text)
        and 1 < len(next_words) <= 6
        and next_words[0] in {"at", "by", "for", "from", "on", "to", "with"}
        and next_words[:2] not in (
            ["for", "example"],
            ["for", "instance"],
            ["for", "now"],
        )
        and next_words[:3] != ["for", "this", "reason"]
    ):
        # Fixed-size captions often put a short required prepositional
        # complement in the next cue ("figure it out / for a point.").
        # Treating the first cue as complete produces exactly the random,
        # mid-sentence endings the production selector must never ship.
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
    raw_next_text = str(next_text or "")
    next_lexical_index = first_lexical_character_index(raw_next_text)
    attributive_subject_continues = bool(
        _TERMINAL_ATTRIBUTIVE_MODIFIER_RE.search(raw_text)
        and next_lexical_index is not None
        and raw_next_text[next_lexical_index].islower()
        and _NEXT_ATTRIBUTIVE_SUBJECT_PREDICATE_RE.match(raw_next_text)
    )
    ambiguous_degree_continues = bool(
        _TERMINAL_AMBIGUOUS_DEGREE_RE.search(raw_text)
        and next_lexical_index is not None
        and str(next_text)[next_lexical_index].islower()
    )
    nominal_subject_continues = bool(
        _TERMINAL_NOMINAL_SUBJECT_RE.search(raw_text)
        and _NOMINAL_PREDICATE_CONTINUATION_RE.match(str(next_text or ""))
    )
    pronoun_subject = _TERMINAL_UNPUNCTUATED_SUBJECT_PRONOUN_RE.search(raw_text)
    pronoun_predicate = _PRONOUN_PREDICATE_CONTINUATION_RE.match(
        str(next_text or "")
    )
    next_is_explicit_question = bool(
        "?" in str(next_text or "")
        and _cue_begins_standalone_question(str(next_text or ""))
    )
    pronoun_subject_continues = bool(
        pronoun_subject
        and pronoun_predicate
        and not next_is_explicit_question
        and pronoun_predicate.group("auxiliary").casefold()
        in _SUBJECT_PRONOUN_AUXILIARIES[
            pronoun_subject.group("pronoun").casefold()
        ]
    )
    return bool(
        _TERMINAL_CALLBACK_RE.search(raw_text)
        or _TERMINAL_DANGLING_TRANSITION_RE.search(raw_text)
        or _TERMINAL_DANGLING_EXAMPLE_INTRO_RE.search(raw_text)
        or _TERMINAL_EXPLICIT_INCOMPLETE_CLAUSE_RE.search(raw_text)
        or _TERMINAL_INCOMPLETE_SUBJECT_RE.search(raw_text)
        or _TERMINAL_BARE_SUBJECT_RE.search(raw_text)
        or nominal_subject_continues
        or pronoun_subject_continues
        or attributive_subject_continues
        or _TERMINAL_DANGLING_ARTICLE_RE.search(raw_text)
        or _TERMINAL_DANGLING_POSSESSIVE_RE.search(raw_text)
        or _TERMINAL_DANGLING_LINK_RE.search(raw_text)
        or (
            bool(next_text.strip())
            and _TERMINAL_HEADLESS_QUANTIFIER_RE.search(raw_text)
        )
        or _TERMINAL_DANGLING_MODAL_PREDICATE_RE.search(raw_text)
        or _TERMINAL_DANGLING_AUXILIARY_ADVERB_RE.search(raw_text)
        or _TERMINAL_DANGLING_DEGREE_RE.search(raw_text)
        or _TERMINAL_DANGLING_TRANSITIVE_RE.search(raw_text)
        or _TERMINAL_AUXILIARY_TRANSITIVE_RE.search(raw_text)
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
    edge_disfluency = _TERMINAL_EDGE_DISFLUENCY_RE.search(raw_text)
    dangling_before_disfluency = False
    if edge_disfluency is not None:
        hesitation = re.search(
            r"(?:er+|hmm+|uh+|um+)\s*[.!?]?[\"')\]]*$",
            raw_text,
            re.IGNORECASE,
        )
        if hesitation is not None:
            prefix = raw_text[:hesitation.start()].rstrip(" ,;:—-")
            dangling_before_disfluency = bool(
                _TERMINAL_DANGLING_PREDICATE_HEAD_RE.search(prefix)
                or _TERMINAL_REQUIRED_COMPLEMENT_RE.search(prefix)
                or _TERMINAL_DANGLING_TRANSITIVE_RE.search(prefix)
            )
    return bool(
        dangling_before_disfluency
        or _has_unanswered_terminal_question(raw_text)
        or _has_unfinished_exemplification_tail(raw_text)
        or _TERMINAL_DANGLING_EXAMPLE_INTRO_RE.search(raw_text)
        or _TERMINAL_EXPLICIT_INCOMPLETE_CLAUSE_RE.search(raw_text)
        or _TERMINAL_INCOMPLETE_SUBJECT_RE.search(raw_text)
        or _TERMINAL_COORDINATING_CONJUNCTION_RE.search(raw_text)
        or _TERMINAL_DANGLING_ARTICLE_RE.search(raw_text)
        or _TERMINAL_DANGLING_POSSESSIVE_RE.search(raw_text)
        or _TERMINAL_REQUIRED_COMPLEMENT_RE.search(raw_text)
        or _TERMINAL_DANGLING_TRANSITIVE_RE.search(raw_text)
        or _TERMINAL_AUXILIARY_TRANSITIVE_RE.search(raw_text)
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
    previous_text = (
        str(segments[end_line - 1].get("text") or "").strip()
        if end_line > start_line
        else ""
    )
    following_lexical_index = first_lexical_character_index(following_text)
    ambiguous_degree = _TERMINAL_AMBIGUOUS_DEGREE_RE.search(full_suffix)
    bare_subject = _TERMINAL_BARE_SUBJECT_RE.search(full_suffix)
    trailing_incomplete_sentence = bool(
        previous_text
        and classify_terminator(previous_text)
        and _OPENING_DEPENDENT_SENTENCE_RE.match(final_text)
        and not (
            protected_quote
            and _contains_quote(final_text, protected_quote)
        )
        and _cue_has_weak_end(
            final_text,
            following_text,
            ignore_caption_case=True,
        )
    )
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
        or trailing_incomplete_sentence
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
        previous_line = line - 1
        trailing_incomplete_sentence = bool(
            line == end_line
            and previous_line >= start_line
            and classify_terminator(
                str(segments[previous_line].get("text") or "").strip()
            )
            and _OPENING_DEPENDENT_SENTENCE_RE.match(suffix)
            and not (
                protected_quote
                and _contains_quote(suffix, protected_quote)
            )
            and _cue_has_weak_end(
                suffix,
                following_text,
                ignore_caption_case=True,
            )
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
            or trailing_incomplete_sentence
        ):
            continue
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
        # Fixed-size captions can split an embedded subject across several
        # cues. Preserve the existing last-cue checks, but retain enough of the
        # selected clause to see a complementizer in the preceding cue.
        selected_end_text = _cue_clip_text(segments, start_line, end_line)
        if force_end_clause_completion and classify_terminator(current_end_text):
            force_end_clause_completion = False
        next_text = (
            str(segments[end_line + 1].get("text") or "")
            if end_line + 1 < len(segments)
            else ""
        )
        if next_text and _FORWARD_TOPIC_TRANSITION_RE.match(next_text):
            break
        if (
            not force_end_clause_completion
            and not _cue_has_weak_end(
                current_end_text,
                next_text,
                ignore_caption_case=ignore_caption_case,
            )
            and not _next_cue_completes_embedded_predicate(
                selected_end_text,
                next_text,
            )
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
    selected_end_text = _cue_clip_text(segments, start_line, end_line)
    if (
        not end_boundary_verified
        and _cue_has_weak_end(
            final_end_text,
            next_text,
            ignore_caption_case=ignore_caption_case,
        )
        or _next_cue_completes_embedded_predicate(
            selected_end_text,
            next_text,
        )
    ) or force_end_clause_completion:
        if (
            force_end_clause_completion
            or _terminal_content_is_explicitly_incomplete(final_end_text)
        ):
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
    normalized_text = " ".join(_toks(raw_text))
    if (
        _literal_structural_filler_only(raw_text)
        or _NEXT_EXAMPLE_FRAMING_RE.fullmatch(raw_text)
        or _PEDAGOGICAL_META_FRAME_ONLY_RE.fullmatch(normalized_text)
        or _BARE_STEP_ACTION_ONLY_RE.fullmatch(raw_text)
        or _SEQUENCE_LABEL_ONLY_RE.fullmatch(raw_text)
        or _CLARIFICATION_META_ONLY_RE.fullmatch(raw_text)
        or _CROSS_CONTENT_REFERENCE_ONLY_RE.fullmatch(raw_text)
        or _FORWARD_EXPLANATION_PROMISE_ONLY_RE.fullmatch(raw_text)
        or _ANNOTATION_EMPHASIS_META_ONLY_RE.fullmatch(raw_text)
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


def _quote_character_span(
    text: str,
    quote: str,
    *,
    preferred_span: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    spans = _quote_character_spans(text, quote)
    if preferred_span is not None and preferred_span in spans:
        return preferred_span
    return spans[0] if spans else None


@dataclass(frozen=True)
class _ModelBoundaryAnchor:
    first_line: int
    last_line: int
    first_span: tuple[int, int]
    last_span: tuple[int, int]

    @property
    def first_word_position(self) -> tuple[int, int]:
        return self.first_line, self.first_span[0]

    @property
    def last_word_position(self) -> tuple[int, int]:
        return self.last_line, self.last_span[1]


def _unique_boundary_anchor(
    segments: list[dict],
    quote: str,
    start_line: int,
    end_line: int,
    *,
    allow_timing_gaps: bool = False,
) -> _ModelBoundaryAnchor | None:
    """Locate one exact model edge inside its proposed contiguous cue range."""
    matches: list[_ModelBoundaryAnchor] = []
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        matches.extend(
            _ModelBoundaryAnchor(line, line, span, span)
            for span in _quote_character_spans(text, quote)
        )
    for cross in _cross_cue_token_windows(
        segments,
        quote,
        start_line,
        end_line,
        allow_timing_gaps=allow_timing_gaps,
    ):
        matches.append(_ModelBoundaryAnchor(
            first_line=cross[0],
            last_line=cross[1],
            first_span=(cross[2], cross[3]),
            last_span=(cross[4], cross[5]),
        ))
    return matches[0] if len(matches) == 1 else None


def _cross_cue_token_windows(
    segments: list[dict],
    quote: str,
    start_line: int,
    end_line: int,
    *,
    allow_timing_gaps: bool = False,
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
            if not allow_timing_gaps:
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
    if len(locations) == 1:
        return locations[0]
    if locations and all(
        start == end == locations[0][0]
        for start, _left, end, _right in locations
    ):
        # Repeated rolling captions can duplicate the same exact claim inside
        # one cue. Its timestamp bounds are still unambiguous, and choosing the
        # first occurrence deterministically avoids rejecting valid teaching.
        return locations[0]
    return None


def _proposal_evidence_location(
    segments: list[dict],
    quote: str,
    start_line: int,
    end_line: int,
    *,
    start_anchor: _ModelBoundaryAnchor | None = None,
    end_anchor: _ModelBoundaryAnchor | None = None,
) -> tuple[int, int, int, int] | None:
    """Ground trusted evidence at the occurrence nearest Gemini's frame."""
    locations: set[tuple[int, int, int, int]] = set()
    for line in range(len(segments)):
        text = str(segments[line].get("text") or "")
        locations.update(
            (line, left, line, right)
            for left, right in _quote_character_spans(text, quote)
        )
    locations.update(
        (first_line, first_left, last_line, last_right)
        for (
            first_line,
            last_line,
            first_left,
            _first_right,
            _last_left,
            last_right,
        ) in _cross_cue_token_windows(
            segments,
            quote,
            0,
            len(segments) - 1,
        )
    )
    line_offsets: list[int] = []
    cursor = 0
    for segment in segments:
        line_offsets.append(cursor)
        cursor += len(str(segment.get("text") or "")) + 1

    def absolute_position(line: int, character: int) -> int:
        return line_offsets[line] + character

    frame_start = absolute_position(
        start_anchor.first_line if start_anchor is not None else start_line,
        start_anchor.first_span[0] if start_anchor is not None else 0,
    )
    frame_end_line = (
        end_anchor.last_line if end_anchor is not None else end_line
    )
    frame_end = absolute_position(
        frame_end_line,
        end_anchor.last_span[1]
        if end_anchor is not None
        else len(str(segments[frame_end_line].get("text") or "")),
    )

    def proximity(location: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        first, left, last, right = location
        location_start = absolute_position(first, left)
        location_end = absolute_position(last, right)
        if location_end < frame_start:
            outside_distance = frame_start - location_end
        elif location_start > frame_end:
            outside_distance = location_start - frame_end
        else:
            outside_distance = 0
        frame_distance = (
            abs(location_start - frame_start)
            + abs(location_end - frame_end)
        )
        return outside_distance, frame_distance, location_start, location_end

    return min(locations, default=None, key=proximity)


def _nearest_boundary_anchor(
    segments: list[dict],
    quote: str,
    start_line: int,
    end_line: int,
    *,
    want: str,
) -> _ModelBoundaryAnchor | None:
    """Locate a model edge nearest its frame without global-uniqueness drift."""
    matches: list[_ModelBoundaryAnchor] = []
    for line, segment in enumerate(segments):
        text = str(segment.get("text") or "")
        matches.extend(
            _ModelBoundaryAnchor(line, line, span, span)
            for span in _quote_character_spans(text, quote)
        )
    matches.extend(
        _ModelBoundaryAnchor(
            first_line=cross[0],
            last_line=cross[1],
            first_span=(cross[2], cross[3]),
            last_span=(cross[4], cross[5]),
        )
        for cross in _cross_cue_token_windows(
            segments,
            quote,
            0,
            len(segments) - 1,
        )
    )

    def proximity(anchor: _ModelBoundaryAnchor) -> tuple[int, int, int, int]:
        if anchor.last_line < start_line:
            outside_distance = start_line - anchor.last_line
        elif anchor.first_line > end_line:
            outside_distance = anchor.first_line - end_line
        else:
            outside_distance = 0
        edge_distance = (
            abs(anchor.first_line - start_line)
            if want == "start"
            else abs(anchor.last_line - end_line)
        )
        transcript_position = (
            (anchor.first_line, anchor.first_span[0])
            if want == "start"
            else (-anchor.last_line, -anchor.last_span[1])
        )
        return (
            outside_distance,
            edge_distance,
            transcript_position[0],
            transcript_position[1],
        )

    return min(matches, default=None, key=proximity)


def _proposal_evidence_anchor(
    proposal: object,
    intent_constraints: dict[str, _IntentConstraint],
    segments: list[dict],
    start_line: int,
    end_line: int,
) -> tuple[str, tuple[int, int, int, int] | None]:
    """Choose the first uniquely grounded selector quote for unit trimming."""
    direct_quote = " ".join(
        str(
            getattr(proposal, "claim_quote", "")
            or getattr(proposal, "topic_evidence_quote", "")
            or ""
        ).split()
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
    candidates = list(dict.fromkeys(
        _trim_structural_evidence_prompt(quote)
        for quote in candidates
        if quote
    ))
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


def _evidence_source_sentence(
    segments: list[dict],
    location: tuple[int, int, int, int],
) -> tuple[str, int]:
    """Return the source sentence and the claim's offset inside that sentence."""
    start_line, start_left, end_line, end_right = location
    if start_line != end_line:
        source = " ".join(
            str(segments[line].get("text") or "")
            for line in range(start_line, end_line + 1)
        )
        return source, 0
    text = str(segments[start_line].get("text") or "")
    for sentence_left, sentence_right in _sentence_character_spans(text):
        if sentence_left <= start_left and end_right <= sentence_right:
            return (
                text[sentence_left:sentence_right].strip(),
                max(0, start_left - sentence_left),
            )
    return text.strip(), start_left


def _enumerated_claim_is_only_structure(text: str) -> bool:
    """Distinguish an outline header from a grounded claim about a set."""
    raw_text = str(text or "")
    outline = _ENUMERATED_OUTLINE_RE.search(raw_text)
    if outline is None:
        return False
    distinction = _COMPARISON_DISTINCTION_RE.search(raw_text)
    if distinction is not None:
        connectors = list(
            _COMPARISON_CONNECTOR_RE.finditer(
                raw_text,
                max(0, distinction.start() - 120),
                distinction.start(),
            )
        )
        if connectors:
            connector = connectors[-1]
            left_tokens = _content_tokens(
                raw_text[max(
                    raw_text.rfind(".", 0, connector.start()),
                    raw_text.rfind("!", 0, connector.start()),
                    raw_text.rfind("?", 0, connector.start()),
                ) + 1:connector.start()]
            )
            right_tokens = _content_tokens(
                raw_text[connector.end():distinction.start()]
            )
            if (
                left_tokens
                and right_tokens
                and left_tokens - right_tokens
                and right_tokens - left_tokens
            ):
                return False
    return _ENUMERATED_SUBSTANTIVE_RELATION_RE.search(
        raw_text,
        outline.end(),
    ) is None


def _compact_claim_is_non_substantive(
    segments: list[dict],
    claim_quote: str,
    location: tuple[int, int, int, int],
) -> bool:
    """Reject a selector claim that is only lesson navigation or a promise."""
    if _CLAIM_PROMISSORY_FRAGMENT_RE.match(claim_quote):
        return True
    source_sentence, claim_left = _evidence_source_sentence(segments, location)
    agenda = _OPENING_AGENDA_RE.match(source_sentence)
    if agenda is None:
        return False
    # A model may omit ``Today we'll`` from its shortest quote. It is still an
    # agenda when the quote begins in the promissory action. A proposition
    # after that prefix (for example ``every differentiable function is
    # continuous``) remains valid teaching evidence and keeps its full setup.
    claim_tail = source_sentence[claim_left:].strip()
    bridge = source_sentence[agenda.end():claim_left]
    if (
        claim_left <= agenda.end()
        or _CLAIM_PROMISSORY_FRAGMENT_RE.match(claim_tail)
        or re.search(r"\bthat\b", bridge, re.IGNORECASE) is None
    ):
        return True
    return _ATOMIC_DECLARATIVE_ONSET_RE.match(claim_tail) is None


def _trim_structural_evidence_prompt(quote: str) -> str:
    """Drop only proven prompt glue before a grounded teaching action."""
    source = " ".join(str(quote or "").split())
    for action in _WORKED_UNIT_ACTION_TOKEN_RE.finditer(source):
        prefix_tokens = _toks(source[:action.start()])
        suffix = source[action.start():].strip()
        suffix_count = len(_toks(suffix))
        if (
            prefix_tokens
            and set(prefix_tokens) <= _WORKED_UNIT_STRUCTURAL_PROMPT_TOKENS
            and 5 <= suffix_count <= 16
        ):
            return suffix
        break
    return source


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


def _lexical_span(
    text: str,
    span: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Return first/last word offsets, excluding display punctuation."""
    if span is None:
        return None
    words = [
        word
        for word in _WORD_RE.finditer(str(text or ""))
        if word.start() < span[1] and word.end() > span[0]
    ]
    return (words[0].start(), words[-1].end()) if words else None


def _semantic_edge_quote(
    text: str,
    quote: str,
    *,
    want: str,
    occurrence: str | None = None,
    preferred_span: tuple[int, int] | None = None,
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
    if preferred_span is not None:
        if preferred_span not in spans:
            return None, False, "ungrounded_boundary_quote"
        span = preferred_span
    elif occurrence == "first":
        span = spans[0]
    elif occurrence == "last":
        span = spans[-1]
    else:
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
    if (
        projected
        and len(spans) != 1
        and occurrence not in {"first", "last"}
        and preferred_span is None
    ):
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
        meta_replacement = _leading_pedagogical_meta_quote(raw_text)
        if meta_replacement:
            return meta_replacement, None
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
    preferred_span: tuple[int, int] | None = None,
) -> tuple[str, bool, str | None]:
    """Replace a model quote that includes removable filler at a cue edge."""
    if _cue_is_only_structural_filler(text):
        # The existing cue-level pass can remove this cue without projecting a
        # boundary inside it. Same-cue repair is only for mixed teaching/filler.
        return quote, False, None
    quote_span, _projected, error = _semantic_edge_quote(
        text,
        quote,
        want=want,
        preferred_span=preferred_span,
    )
    if error or quote_span is None:
        # Preserve the normal grounding/ambiguity failure emitted downstream.
        return quote, False, None

    if want == "start":
        leading_filler_end = 0
        for left, right in _sentence_character_spans(text):
            if left > leading_filler_end and _WORD_RE.search(
                text[leading_filler_end:left]
            ):
                break
            if not _cue_is_only_structural_filler(text[left:right]):
                break
            leading_filler_end = right
        if leading_filler_end and quote_span[0] < leading_filler_end:
            replacement, replacement_error = _expanded_context_edge_quote(
                text,
                want="start",
            )
            replacement_span, projected, grounding_error = _semantic_edge_quote(
                text,
                replacement,
                want="start",
            )
            if (
                replacement_error is None
                and grounding_error is None
                and replacement_span is not None
                and projected
            ):
                return replacement, True, None
        selected_tail = text[quote_span[0]:]
        step_meta = _LEADING_STEP_META_RE.match(selected_tail)
        if step_meta is not None:
            retained = selected_tail[step_meta.start("teaching"):].strip()
            if _opening_clause_is_standalone(retained):
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
        meta_replacement = _leading_pedagogical_meta_quote(selected_tail)
        if meta_replacement:
            replacement_span, projected, replacement_error = _semantic_edge_quote(
                text,
                meta_replacement,
                want="start",
            )
            if (
                replacement_error is None
                and replacement_span is not None
                and projected
            ):
                return meta_replacement, True, None
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
        or _OPENING_CONTEXTUAL_MODIFIER_SUBJECT_RE.match(opening_clause)
        or _OPENING_UNRESOLVED_EDGE_REFERENCE_RE.match(opening_clause)
        or _OPENING_CONTEXT_ONLY_TRANSITION_RE.match(opening_clause)
        or _opening_contextual_example_needs_context(opening_clause)
        or _opening_has_context_dependent_subject(opening_clause)
        or _OPENING_RELATIVE_WHERE_FRAGMENT_RE.match(opening_clause)
        or _OPENING_SUBORDINATE_PRONOUN_REFERENCE_RE.match(opening_clause)
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


def _leading_pedagogical_meta_quote(text: str) -> str:
    """Skip generic lesson framing while retaining its concrete premise."""
    raw_text = str(text or "")
    framing = _LEADING_PEDAGOGICAL_META_RE.match(raw_text)
    if framing is None:
        return ""
    retained = raw_text[framing.end():].lstrip(
        " \t\r\n,;:.!?—-\"'’”)]"
    )
    if not _PEDAGOGICAL_META_TEACHING_ONSET_RE.match(retained):
        return ""
    return _exact_boundary_quote(retained, want="start")


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
            or _OPENING_RELATIVE_BRIDGE_RE.match(previous)
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


def _opening_has_unresolved_setup_reference(opening: str) -> bool:
    """Require a setup to name any object referenced by pronoun or demonstrative."""
    references = [
        match
        for pattern in (
            _OPENING_ANAPHORIC_SETUP_REFERENCE_RE,
            _OPENING_SETUP_PRONOUN_REFERENCE_RE,
        )
        if (match := pattern.search(opening)) is not None
    ]
    if not references:
        return False
    first_reference = min(references, key=lambda match: match.start())
    local_prefix = opening[:first_reference.start()]
    return _LOCAL_EXPLICIT_PROBLEM_RE.search(local_prefix) is None


def _trim_initial_instructional_preview(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    evidence_quote: str,
) -> tuple[int, str] | None:
    """Skip a cross-cue lesson preview when a grounded setup follows it.

    Rolling captions commonly split ``what we're going to cover in this
    video`` across several cues, so cue-local filler checks cannot recognize
    it.  Keep this repair conservative: the later setup must be explicit, the
    grounded evidence must remain after the cut, and none of that evidence may
    occur in the omitted preview.
    """
    if start_line >= end_line:
        return None
    preview_head = _cue_clip_text(
        segments,
        start_line,
        min(end_line, start_line + 3),
    )
    if _CROSS_CUE_INSTRUCTIONAL_PREVIEW_RE.match(preview_head) is None:
        return None

    for line in range(start_line + 1, end_line + 1):
        current = str(segments[line].get("text") or "").strip()
        if _PEDAGOGICAL_SETUP_ONSET_RE.match(current) is None:
            continue
        opening_candidates = [current]
        marker = _LEADING_DISCOURSE_MARKER_RE.match(current)
        if marker is not None:
            without_marker = current[marker.end():].strip()
            if without_marker:
                opening_candidates.insert(0, without_marker)
        if not any(
            _opening_clause_is_standalone(opening)
            and not _opening_has_unresolved_setup_reference(opening)
            for opening in opening_candidates
        ):
            continue
        retained = _cue_clip_text(segments, line, end_line)
        omitted = _cue_clip_text(segments, start_line, line - 1)
        if evidence_quote and (
            not _contains_quote(retained, evidence_quote)
            or _contains_quote(omitted, evidence_quote)
        ):
            continue
        if len(_content_tokens(current)) < 3:
            continue
        quote = _exact_boundary_quote(current, want="start")
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
    agenda = _OPENING_AGENDA_RE.match(raw_text)
    if agenda is not None:
        agenda_sentence = next(
            (
                (left, right)
                for left, right in sentence_spans
                if left <= agenda.start() and agenda.end() <= right
            ),
            None,
        )
        if agenda_sentence is not None:
            _agenda_left, agenda_right = agenda_sentence
            retained = raw_text[agenda_right:].lstrip(" \t\r\n,;:—-")
            grounded_retained = " ".join(
                part for part in (retained, following_text) if part
            )
            if (
                retained
                and _opening_clause_is_standalone(retained)
                and (
                    (
                        evidence_quote
                        and _contains_quote(grounded_retained, evidence_quote)
                        and not _contains_quote(
                            raw_text[:agenda_right],
                            evidence_quote,
                        )
                    )
                    or (
                        not evidence_quote
                        and len(_content_tokens(retained) & anchor_tokens) >= 2
                    )
                )
            ):
                return _exact_boundary_quote(retained, want="start")
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
    edge_disfluency = _TERMINAL_EDGE_DISFLUENCY_RE.search(raw_text)
    if edge_disfluency is not None:
        cut_start = edge_disfluency.start("noise")
        prefix = raw_text[:cut_start].rstrip(" ,;:—-")
        if (
            len(_toks(prefix)) >= 3
            and not _terminal_content_is_explicitly_incomplete(prefix)
            and not _cue_has_explicit_dangling_end(prefix, "")
            and not _TERMINAL_DANGLING_PREDICATE_HEAD_RE.search(prefix)
        ):
            return cut_start
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
                raw_text[start:containing_sentence[1]]
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


def _completed_truncated_caption_end(
    text: str,
    quote_span: tuple[int, int] | None,
    *,
    scope_text: str,
) -> tuple[tuple[int, int], str] | None:
    """Move past Supadata's ``squ.`` fragment when ``squared`` is in the cue."""
    if quote_span is None:
        return None
    raw_text = str(text or "")
    if quote_span[1] >= len(raw_text) or _WORD_RE.search(raw_text[quote_span[1]:]) is None:
        return None
    selected_words = list(_WORD_RE.finditer(raw_text[:quote_span[1]]))
    if not selected_words:
        return None
    fragment = _toks(selected_words[-1].group(0))[0]
    if fragment != "squ" or "squared" not in _toks(raw_text):
        return None
    suffix = raw_text[quote_span[1]:]
    connected_tokens = _content_tokens(scope_text)
    retained_right: int | None = None
    for left, right in _sentence_character_spans(suffix):
        sentence_tokens = _content_tokens(suffix[left:right])
        if not sentence_tokens:
            continue
        if not (sentence_tokens & connected_tokens):
            break
        retained_right = quote_span[1] + right
        connected_tokens.update(sentence_tokens)
    if retained_right is None:
        return None
    complete = _last_safe_complete_prefix(raw_text[:retained_right])
    if len(complete) <= quote_span[1]:
        return None
    words = list(_WORD_RE.finditer(complete))
    if not words:
        return None
    quote_left = words[max(0, len(words) - 6)].start()
    return (quote_left, len(complete)), complete[quote_left:]


def _trusted_joined_unit_end(
    segments: list[dict],
    *,
    end_line: int,
    end_span: tuple[int, int] | None,
) -> tuple[int, tuple[int, int], str] | None:
    """Extend Gemini's word edge through its unfinished spoken sentence."""
    if not (0 <= end_line < len(segments)):
        return None
    end_source = str(segments[end_line].get("text") or "")
    end_right = end_span[1] if end_span is not None else len(end_source)
    if not (0 <= end_right <= len(end_source)):
        return None

    parts: list[str] = []
    line_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for line in range(end_line, len(segments)):
        if parts:
            parts.append(" ")
            cursor += 1
        source = str(segments[line].get("text") or "")
        joined_left = cursor
        parts.append(source)
        cursor += len(source)
        line_ranges.append((line, joined_left, cursor))
    joined = "".join(parts)
    selected_right = end_right
    hard_boundaries = [
        boundary
        for boundary in _trusted_joined_unit_boundaries(joined)
        if boundary.group(0)[0] in ".!?"
    ]
    if any(boundary.end() == selected_right for boundary in hard_boundaries):
        return None
    completion = next(
        (
            boundary
            for boundary in hard_boundaries
            if boundary.end() > selected_right
        ),
        None,
    )
    completion_right = completion.end() if completion is not None else len(joined)
    if (
        completion_right <= selected_right
        or _WORD_RE.search(joined[selected_right:completion_right]) is None
    ):
        return None
    mapped = next(
        (
            (line, completion_right - joined_left)
            for line, joined_left, joined_right in line_ranges
            if joined_left < completion_right <= joined_right
        ),
        None,
    )
    if mapped is None:
        return None
    line, source_right = mapped
    source = str(segments[line].get("text") or "")
    retained = source[:source_right]
    words = list(_WORD_RE.finditer(retained))
    if not words:
        return None
    quote_left = words[max(0, len(words) - 6)].start()
    quote = retained[quote_left:source_right]
    return line, (quote_left, source_right), quote


def _trim_end_quote_before_edge_noise(
    text: str,
    quote: str,
    *,
    evidence_quote: str = "",
    learning_objective: str = "",
    preferred_span: tuple[int, int] | None = None,
) -> tuple[str, bool]:
    """Shorten a grounded end quote before an inline next-topic or outro tail."""
    spans = _quote_character_spans(text, quote)
    if not spans:
        return quote, False
    # End projection is authoritative at the final grounded occurrence. This
    # matters for intentionally short one-word quotes such as "energy".
    span = preferred_span if preferred_span in spans else spans[-1]
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


def _projected_end_continues_same_sentence(
    text: str,
    quote_span: tuple[int, int],
) -> bool:
    """Detect a model edge placed before the remaining words of its sentence."""
    suffix = str(text or "")[quote_span[1]:]
    first_word = _WORD_RE.search(suffix)
    if first_word is None:
        return False
    punctuation = suffix[:first_word.start()]
    if re.search(r"[.!?]", punctuation):
        return False
    continuation = suffix[first_word.start():]
    first_tokens = _toks(continuation)
    return bool(
        re.match(r"^[,;:—-]", suffix.lstrip())
        or (
            first_tokens
            and first_tokens[0]
            in {
                "as", "at", "because", "by", "during", "for", "from", "if",
                "in", "into", "of", "on", "onto", "than", "that", "to",
                "when", "where", "which", "while", "with", "without",
            }
        )
    )


def _trim_repeated_rolling_caption_tail(
    segments: list[dict],
    end_line: int,
    end_quote: str,
) -> tuple[str, str | None] | None:
    """Stop before an unfinished verbatim restatement split across captions."""
    next_line = end_line + 1
    if end_line <= 0 or next_line >= len(segments):
        return None
    current_text = str(segments[end_line].get("text") or "")
    next_text = str(segments[next_line].get("text") or "")
    current_span, projected, error = _semantic_edge_quote(
        current_text,
        end_quote,
        want="end",
    )
    if (
        error
        or current_span is None
        or projected
        or _WORD_RE.search(current_text[current_span[1]:])
        or not next_text.strip()
    ):
        return None
    try:
        current_end = float(segments[end_line].get("end"))
        next_start = float(segments[next_line].get("start"))
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(current_end)
        or not math.isfinite(next_start)
        or next_start - current_end >= _SECTION_RESET_GAP_S
    ):
        return None

    contraction_expansions = {
        "i'm": ("i", "am"),
        "you're": ("you", "are"),
        "we're": ("we", "are"),
        "they're": ("they", "are"),
        "can't": ("can", "not"),
        "won't": ("will", "not"),
        "don't": ("do", "not"),
        "doesn't": ("does", "not"),
        "didn't": ("did", "not"),
        "isn't": ("is", "not"),
        "aren't": ("are", "not"),
        "wasn't": ("was", "not"),
        "weren't": ("were", "not"),
        "haven't": ("have", "not"),
        "hasn't": ("has", "not"),
        "hadn't": ("had", "not"),
    }

    def records(text: str, line: int) -> list[tuple[str, int, int, int]]:
        result: list[tuple[str, int, int, int]] = []
        for match in _WORD_RE.finditer(text):
            [token] = _toks(match.group(0))
            for normalized in contraction_expansions.get(token, (token,)):
                result.append((normalized, line, match.start(), match.end()))
        return result

    context_records: list[tuple[str, int, int, int]] = []
    for line in range(max(0, end_line - 2), end_line + 1):
        context_records.extend(
            records(str(segments[line].get("text") or ""), line)
        )
    next_words = list(_WORD_RE.finditer(next_text))
    clause_signals = {
        "am", "are", "can", "could", "did", "do", "does", "had", "has",
        "have", "is", "may", "might", "must", "shall", "should", "was",
        "were", "will", "would",
    }
    for prefix_width in range(1, min(5, len(next_words)) + 1):
        next_prefix_end = next_words[prefix_width - 1].end()
        combined = [
            *context_records,
            *records(next_text[:next_prefix_end], next_line),
        ]
        combined_tokens = [item[0] for item in combined]
        for width in range(min(18, len(combined) // 2), 5, -1):
            suffix_start = len(combined) - width
            repeated = combined_tokens[suffix_start:]
            if combined[suffix_start][1] != end_line:
                continue
            if not (
                set(repeated).intersection(clause_signals)
                or any(
                    len(token) > 4 and token.endswith(("ed", "ing"))
                    for token in repeated
                )
            ):
                continue
            # Only remove an immediate restart. Matching an older clause across
            # substantive intervening teaching would discard valid middle content.
            earlier_start = suffix_start - width
            if earlier_start < 0:
                continue
            if combined_tokens[earlier_start:suffix_start] != repeated:
                continue
            earlier_last = combined[suffix_start - 1]
            if earlier_last[1] != end_line:
                continue
            boundary_right = earlier_last[3]
            if boundary_right >= current_span[1]:
                continue
            prefix = current_text[:boundary_right].rstrip(" ,;:—-")
            quote = _exact_boundary_quote(prefix, want="end")
            target_span = _quote_character_span(prefix, quote)
            if target_span is None:
                continue
            spans = _quote_character_spans(current_text, quote)
            if target_span not in spans:
                continue
            occurrence = None
            if len(spans) != 1:
                if spans[0] == target_span:
                    occurrence = "first"
                elif spans[-1] == target_span:
                    occurrence = "last"
                else:
                    continue
            return quote, occurrence
    return None


def _complete_split_caption_tail(
    segments: list[dict],
    end_line: int,
    end_quote: str,
    *,
    proposals: list[object],
    proposal_index: int,
    ignore_caption_case: bool,
    anchor_text: str = "",
) -> tuple[int, str, str | None] | None:
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

    has_completion_prefix = any(
        onset > 0
        and 1 <= len(_toks(next_text[:onset])) <= 12
        and _SPLIT_CAPTION_COMPLETION_SIGNAL_RE.search(
            f"{selected_tail} {next_text[:onset]}"
        )
        for onset in onset_candidates
    )
    if not has_explicit_closure and not has_weak_end and not has_completion_prefix:
        return None

    max_prefix_words = 12 if has_explicit_closure or has_completion_prefix else 5
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
            target_span = (
                prefix_matches[-width].start(),
                prefix_matches[-1].end(),
            )
            if target_span not in spans:
                continue
            occurrence = None
            if len(spans) != 1:
                if spans[0] != target_span:
                    continue
                occurrence = "first"
            span = target_span
            if _WORD_RE.search(next_text[span[1]:]):
                return next_line, quote, occurrence
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


def _recover_context_expansion_start_quote(
    segments: list[dict],
    start_line: int,
    original_start_line: int,
    end_line: int,
    *,
    evidence_quote: str,
    anchor_text: str,
) -> str:
    """Trim context expansion to the latest complete, topic-anchored sentence."""
    if not (0 <= start_line < original_start_line <= end_line < len(segments)):
        return ""
    text = str(segments[start_line].get("text") or "")
    sentence_spans = _sentence_character_spans(text)
    if not sentence_spans:
        return ""
    anchors = _content_tokens(f"{evidence_quote} {anchor_text}")
    following = _cue_clip_text(segments, start_line + 1, end_line)
    for left, right in reversed(sentence_spans):
        sentence = text[left:right].strip()
        if (
            not (
                _opening_clause_is_standalone(sentence)
                or _general_local_setup_is_complete(sentence)
            )
            or len(_content_tokens(sentence) & anchors) < 2
        ):
            continue
        retained = " ".join(
            part for part in (text[left:].strip(), following) if part
        )
        if evidence_quote and not _contains_quote(retained, evidence_quote):
            continue
        return _exact_boundary_quote(text[left:], want="start")
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


def _trim_around_internal_structural_filler(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    evidence_location: tuple[int, int, int, int] | None,
    ignore_caption_case: bool,
) -> tuple[int, int, bool] | None:
    """Keep the one contiguous, complete teaching run containing the claim.

    A reel cannot silently jump over an advertisement, tangent, lesson reference,
    or administrative interruption.  When such a full cue splits a proposal, keep
    only the clean contiguous side that contains the grounded teaching claim.  If
    that side is not independently complete, fail closed instead of retaining the
    interruption.
    """
    barriers = {
        line
        for line in range(start_line, end_line + 1)
        if (
            _cue_is_only_structural_filler(
                str(segments[line].get("text") or "")
            )
            or _INTERNAL_INTERRUPTION_MARKER_RE.search(
                str(segments[line].get("text") or "")
            )
        )
    }
    if not barriers:
        return start_line, end_line, False
    if evidence_location is None:
        return None
    evidence_start, _left, evidence_end, _right = evidence_location
    if evidence_start in barriers or evidence_end in barriers:
        return None

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for line in range(start_line, end_line + 2):
        if line <= end_line and line not in barriers:
            if run_start is None:
                run_start = line
            continue
        if run_start is not None:
            runs.append((run_start, line - 1))
            run_start = None
    matching_runs = [
        (left, right)
        for left, right in runs
        if left <= evidence_start <= evidence_end <= right
    ]
    if len(matching_runs) != 1:
        return None
    retained_start, retained_end = matching_runs[0]

    opening = str(segments[retained_start].get("text") or "").strip()
    connector = re.match(
        r"^\s*(?:and|but|so)\s*[,;:]?\s+",
        opening,
        re.IGNORECASE,
    )
    standalone_opening = (
        opening[connector.end():].strip() if connector is not None else opening
    )
    if not _opening_clause_is_standalone(standalone_opening):
        return None

    ending = str(segments[retained_end].get("text") or "").strip()
    following = (
        str(segments[retained_end + 1].get("text") or "")
        if retained_end + 1 < len(segments)
        else ""
    )
    if (
        _terminal_content_is_explicitly_incomplete(ending)
        or _cue_has_weak_end(
            ending,
            following,
            ignore_caption_case=ignore_caption_case,
        )
    ):
        return None
    return retained_start, retained_end, True


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


def _conditional_handoff_bridges_objective(
    reset: re.Match[str],
    learning_objective: str,
) -> bool:
    """Keep a method fallback only when the objective names both alternatives."""
    handoff = str(reset.groupdict().get("conditional_handoff") or "")
    if not handoff or not _objective_explicitly_relates_sections(learning_objective):
        return False
    generic = {
        "complete", "concept", "describe", "discuss", "example", "explain",
        "idea", "learn", "lesson", "show", "teach", "understand", "work",
    }
    objective_tokens = _content_tokens(learning_objective) - generic
    premise_overlap = objective_tokens & _content_tokens(handoff)
    alternative_overlap = objective_tokens & _content_tokens(reset.group("subject"))
    return bool(
        premise_overlap
        and alternative_overlap
        and len(premise_overlap | alternative_overlap) >= 2
    )


@dataclass(frozen=True)
class _TopicTransition:
    navigation_line: int
    navigation_left: int
    new_side_line: int
    new_side_left: int
    worked_unit: bool = False
    recap: bool = False
    clears_recap: bool = False
    reset_subject: str = ""
    named_method_handoff: bool = False


def _named_method_handoff_misgrounds_objective(
    transitions: list[_TopicTransition],
    segments: list[dict],
    start_line: int,
    *,
    evidence_location: tuple[int, int, int, int] | None,
    learning_objective: str,
) -> bool:
    """Reject evidence from an old method when the objective names the new one."""
    if evidence_location is None:
        return False
    evidence_end = evidence_location[2], evidence_location[3]
    generic_subject_tokens = {
        "a", "an", "distribution", "method", "model", "our", "procedure",
        "test", "that", "the", "this",
    }
    objective_tokens = set(_toks(learning_objective)) - generic_subject_tokens
    for transition in transitions:
        if (
            not transition.named_method_handoff
            or evidence_end > (
                transition.navigation_line,
                transition.navigation_left,
            )
        ):
            continue
        distinctive_subject = (
            set(_toks(transition.reset_subject)) - generic_subject_tokens
        )
        objective_subject = objective_tokens & distinctive_subject
        if not objective_subject:
            continue
        prior_parts = [
            str(segments[line].get("text") or "")
            for line in range(start_line, transition.navigation_line)
        ]
        prior_parts.append(
            str(segments[transition.navigation_line].get("text") or "")[
                :transition.navigation_left
            ]
        )
        prior_tokens = set(_toks(" ".join(prior_parts))) - generic_subject_tokens
        if not (objective_subject & prior_tokens):
            return True
    return False


@dataclass(frozen=True)
class _AtomicDeclarative:
    line: int
    left: int
    right: int
    subject_tokens: frozenset[str]
    sentence_tokens: frozenset[str]
    definition_like: bool


def _comparison_arc_transitions(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    evidence_location: tuple[int, int, int, int] | None,
    scope_text: str,
    subject_anchors: tuple[frozenset[str], ...] = (),
) -> list[_TopicTransition]:
    """Bound a comparison to its explicit declaration and exclude later examples."""
    if evidence_location is None or not _EXPLICIT_COMPARISON_OBJECTIVE_RE.search(
        str(scope_text or "")
    ):
        return []
    scope_tokens = _content_tokens(scope_text)
    evidence_start = evidence_location[0], evidence_location[1]
    evidence_end = evidence_location[2], evidence_location[3]
    transitions: list[_TopicTransition] = []

    def anchor_matches(anchor: frozenset[str], tokens: set[str]) -> bool:
        return bool(
            anchor
            and any(
                left == right
                or (
                    len(left) >= 6
                    and len(right) >= 6
                    and left[:6] == right[:6]
                )
                or (
                    min(len(left), len(right)) >= 4
                    and left.rstrip("e") == right.rstrip("e")
                )
                for left in anchor
                for right in tokens
            )
        )

    if len(subject_anchors) >= 2:
        for line in range(start_line, end_line + 1):
            text = str(segments[line].get("text") or "")
            for sentence_left, sentence_right in _sentence_character_spans(text):
                sentence = text[sentence_left:sentence_right].strip()
                sentence_tokens = _content_tokens(sentence)
                if not all(
                    anchor_matches(anchor, sentence_tokens)
                    for anchor in subject_anchors
                ):
                    continue
                explicit_relation = bool(
                    _EXPLICIT_COMPARISON_OBJECTIVE_RE.search(sentence)
                    or _DIRECT_COMPARISON_CLAUSE_RE.search(sentence)
                    or (
                        _COMPARISON_DISTINCTION_RE.search(sentence)
                        and _COMPARISON_CONNECTOR_RE.search(sentence)
                    )
                )
                if (
                    not explicit_relation
                    or _COMPARISON_SETUP_REFERENCE_RE.search(sentence)
                    or not (
                        _opening_clause_is_standalone(sentence)
                        or _general_local_setup_is_complete(sentence)
                    )
                ):
                    continue
                sentence_start = (line, sentence_left)
                sentence_end = (line, sentence_right)
                if sentence_start <= evidence_start < sentence_end:
                    if sentence_start > (start_line, 0):
                        transitions.append(_TopicTransition(
                            navigation_line=line,
                            navigation_left=sentence_left,
                            new_side_line=line,
                            new_side_left=sentence_left,
                        ))
                    if _WORD_RE.search(text[sentence_right:]):
                        transitions.append(_TopicTransition(
                            navigation_line=line,
                            navigation_left=sentence_right,
                            new_side_line=line,
                            new_side_left=sentence_right,
                        ))
                    break

    for line in range(start_line, end_line + 1):
        next_text = (
            str(segments[line + 1].get("text") or "")
            if line < end_line
            else ""
        )
        left_text = str(segments[line].get("text") or "")
        joined = f"{left_text} {next_text}" if next_text else left_text
        split = len(left_text) + 1

        for distinction in _COMPARISON_DISTINCTION_RE.finditer(joined):
            connectors = list(
                _COMPARISON_CONNECTOR_RE.finditer(
                    joined,
                    max(0, distinction.start() - 180),
                    distinction.start(),
                )
            )
            if not connectors:
                continue
            connector = connectors[-1]
            right_tokens = _content_tokens(
                joined[connector.end():distinction.start()]
            )
            if not right_tokens or not right_tokens <= scope_tokens:
                continue
            left_words = list(_WORD_RE.finditer(joined[:connector.start()]))
            selected_left: list[re.Match[str]] = []
            for word in reversed(left_words[-8:]):
                token = _content_tokens(word.group(0))
                if token and token <= scope_tokens:
                    selected_left.append(word)
                    continue
                if selected_left:
                    break
            if not selected_left:
                continue
            left_tokens = _content_tokens(
                joined[selected_left[-1].start():connector.start()]
            )
            if not left_tokens or not (left_tokens - right_tokens) or not (
                right_tokens - left_tokens
            ):
                continue
            onset = selected_left[-1].start()
            onset_line, onset_left = (
                (line, onset) if onset < split else (line + 1, onset - split)
            )
            if (onset_line, onset_left) <= evidence_start:
                transitions.append(_TopicTransition(
                    navigation_line=onset_line,
                    navigation_left=onset_left,
                    new_side_line=onset_line,
                    new_side_left=onset_left,
                ))

        for example in _FOLLOWUP_EXAMPLE_REUSE_RE.finditer(joined):
            onset = example.start()
            onset_line, onset_left = (
                (line, onset) if onset < split else (line + 1, onset - split)
            )
            if (onset_line, onset_left) > evidence_end:
                transitions.append(_TopicTransition(
                    navigation_line=onset_line,
                    navigation_left=onset_left,
                    new_side_line=onset_line,
                    new_side_left=onset_left,
                ))
    return sorted(
        {
            (item.navigation_line, item.navigation_left): item
            for item in transitions
        }.values(),
        key=lambda item: (item.navigation_line, item.navigation_left),
    )


def _claim_anchored_atomic_transitions(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    evidence_location: tuple[int, int, int, int] | None,
    evidence_quote: str,
    scope_text: str,
    relationship_bridge_allowed: bool,
) -> list[_TopicTransition]:
    """Bound a broad compact proposal to the atomic claim cited by Gemini.

    This is deliberately a high-confidence fallback for selector mistakes. It
    recognizes adjacent standalone definitions with different named subjects,
    while leaving comparisons, causal explanations, and worked arcs intact.
    """
    if evidence_location is None:
        return []
    normalized_scope = " ".join(str(scope_text or "").split())
    comparison_scope = bool(
        _EXPLICIT_COMPARISON_OBJECTIVE_RE.search(normalized_scope)
    )
    coherent_arc_scope = bool(
        _ATOMIC_CAUSAL_SCOPE_RE.search(normalized_scope)
        or _ATOMIC_COHERENT_ARC_SCOPE_RE.search(normalized_scope)
        or _objective_explicitly_relates_sections(normalized_scope)
    )
    selected_text = _cue_clip_text(segments, start_line, end_line)
    if (
        not comparison_scope
        and (
            relationship_bridge_allowed
            or (
                coherent_arc_scope
                and _ATOMIC_COHERENT_LINK_RE.search(selected_text)
            )
        )
    ):
        return []
    worked_scope = _ATOMIC_WORKED_SCOPE_RE.search(normalized_scope)
    if worked_scope is not None:
        return []

    declarations: list[_AtomicDeclarative] = []
    claim_sentence: tuple[int, int, int, frozenset[str]] | None = None
    evidence_start_line, evidence_left, evidence_end_line, evidence_right = (
        evidence_location
    )
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        for sentence_left, sentence_right in _sentence_character_spans(text):
            sentence = text[sentence_left:sentence_right]
            sentence_tokens = frozenset(_content_tokens(sentence))
            if (
                line <= evidence_start_line <= evidence_end_line <= line
                and sentence_left <= evidence_left
                and evidence_right <= sentence_right
            ):
                claim_sentence = (
                    line,
                    sentence_left,
                    sentence_right,
                    sentence_tokens,
                )
            onset = _ATOMIC_DECLARATIVE_ONSET_RE.match(sentence)
            if onset is None:
                continue
            subject_tokens = frozenset(_content_tokens(onset.group("subject")))
            raw_subject_tokens = set(_toks(onset.group("subject")))
            named_dependent_tokens = raw_subject_tokens - {
                "a", "an", "current", "final", "next", "our", "resulting",
                "same", "that", "the", "their", "this", "your",
                *_ATOMIC_DEPENDENT_SUBJECTS,
            }
            if not subject_tokens or (
                subject_tokens <= _ATOMIC_DEPENDENT_SUBJECTS
                and not named_dependent_tokens
            ):
                continue
            declarations.append(_AtomicDeclarative(
                line=line,
                left=sentence_left + onset.start(),
                right=sentence_right,
                subject_tokens=subject_tokens,
                sentence_tokens=sentence_tokens,
                definition_like=bool(
                    _ATOMIC_DEFINITIONAL_PREDICATE_RE.fullmatch(
                        onset.group("predicate")
                    )
                ),
            ))
    if claim_sentence is None or len(declarations) < 2:
        return []
    claim_line, claim_left, claim_right, _claim_tokens = claim_sentence
    claim_text = str(segments[claim_line].get("text") or "")[
        claim_left:claim_right
    ].strip()
    claim_requires_prior_context = not _opening_clause_is_standalone(claim_text)

    def related(left: _AtomicDeclarative, right: _AtomicDeclarative) -> bool:
        return bool(
            left.subject_tokens <= right.sentence_tokens
            or right.subject_tokens <= left.sentence_tokens
        )

    subject_runs: list[_AtomicDeclarative] = []
    for declaration in declarations:
        if not subject_runs or not related(subject_runs[-1], declaration):
            subject_runs.append(declaration)
    relational_metadata_without_grounded_relation = bool(
        _objective_explicitly_relates_sections(normalized_scope)
        and not _objective_explicitly_relates_sections(evidence_quote)
    )
    coordinated_breadth = bool(
        normalized_scope.count(",") >= 2
        and re.search(r"\band\b", normalized_scope, re.IGNORECASE)
    )
    adjacent_definitions = bool(
        len(subject_runs) >= 2
        and all(item.definition_like for item in subject_runs)
    )
    if not (
        len(subject_runs) >= 3
        or _ATOMIC_BREADTH_RE.search(normalized_scope)
        or coordinated_breadth
        or relational_metadata_without_grounded_relation
        or adjacent_definitions
    ):
        return []
    if len(subject_runs) < 2:
        return []

    claim_line, claim_left, _claim_right, claim_tokens = claim_sentence
    claim_declaration = next(
        (
            item
            for item in declarations
            if item.line == claim_line and item.left <= evidence_left < item.right
        ),
        None,
    )
    cluster_subject = (
        claim_declaration.subject_tokens
        if claim_declaration is not None
        else frozenset(_content_tokens(evidence_quote))
    )
    cluster_tokens = set(claim_tokens)
    cluster_start_line, cluster_start_left = claim_line, claim_left

    prior_declarations = [
        item
        for item in declarations
        if (item.line, item.left) < (claim_line, claim_left)
    ]
    if claim_requires_prior_context:
        if (
            not prior_declarations
            or claim_line - prior_declarations[-1].line > 1
        ):
            # A dependent grounded sentence without adjacent named setup cannot
            # be projected safely; retain the model span for later context gates.
            return []
        context_declaration = prior_declarations.pop()
        cluster_start_line = context_declaration.line
        cluster_start_left = context_declaration.left
        cluster_tokens.update(context_declaration.sentence_tokens)
        cluster_subject = frozenset(
            set(cluster_subject) | set(context_declaration.subject_tokens)
        )
    prior_boundary_needed = False
    for item in reversed(prior_declarations):
        item_is_related = bool(
            item.subject_tokens <= cluster_tokens
            or cluster_subject <= item.sentence_tokens
        )
        if not item_is_related:
            prior_boundary_needed = True
            break
        cluster_start_line, cluster_start_left = item.line, item.left
        cluster_tokens.update(item.sentence_tokens)
        cluster_subject = frozenset(set(cluster_subject) | set(item.subject_tokens))

    transitions: list[_TopicTransition] = []
    if prior_boundary_needed:
        transitions.append(_TopicTransition(
            navigation_line=cluster_start_line,
            navigation_left=cluster_start_left,
            new_side_line=cluster_start_line,
            new_side_left=cluster_start_left,
        ))

    for item in declarations:
        if (item.line, item.left) <= (evidence_end_line, evidence_right):
            continue
        item_is_related = bool(
            item.subject_tokens <= cluster_tokens
            or cluster_subject <= item.sentence_tokens
        )
        if item_is_related:
            cluster_tokens.update(item.sentence_tokens)
            cluster_subject = frozenset(
                set(cluster_subject) | set(item.subject_tokens)
            )
            continue
        transitions.append(_TopicTransition(
            navigation_line=item.line,
            navigation_left=item.left,
            new_side_line=item.line,
            new_side_left=item.left,
        ))
        break
    return transitions


def _worked_unit_prompt_prefix_is_structural(prefix: str) -> bool:
    match = _WORKED_UNIT_EVIDENCE_PROMPT_PREFIX_RE.search(prefix)
    return bool(
        match is not None
        and set(_toks(match.group("glue")))
        <= _WORKED_UNIT_STRUCTURAL_PROMPT_TOKENS
    )


def _cross_cue_grounded_action_onset(
    text: str,
    next_text: str,
) -> tuple[int, int] | None:
    """Find a question action whose object starts in the next caption cue."""
    raw_text = str(text or "")
    following_text = str(next_text or "")
    if not following_text.strip():
        return None
    explicitly_dangling = _cue_has_explicit_dangling_end(
        raw_text,
        following_text,
    )
    for action in reversed(list(_WORKED_UNIT_ACTION_TOKEN_RE.finditer(raw_text))):
        wrapper = _WORKED_UNIT_FUTURE_ACTION_PREFIX_RE.search(
            raw_text[:action.start()]
        )
        suffix_tokens = _toks(raw_text[action.end():])
        fragment = raw_text[action.start():]
        following_first_word = _WORD_RE.search(following_text)
        guarded_following = (
            following_text[following_first_word.start():]
            if following_first_word is not None
            else following_text
        )
        joined_fragment = f"{fragment} {guarded_following}".strip()
        if (
            wrapper is not None
            and len(suffix_tokens) <= 6
            and set(suffix_tokens) <= _WORKED_UNIT_STRUCTURAL_PROMPT_TOKENS
            and (explicitly_dangling or not suffix_tokens)
            and _WORKED_UNIT_PROCEDURAL_STEP_RE.match(joined_fragment) is None
            and _WORKED_UNIT_ANAPHORIC_CONTINUATION_RE.match(joined_fragment) is None
        ):
            return wrapper.start(), action.start()
    return None


def _hard_topic_reset_crosses_cue_boundary(text: str, next_text: str) -> bool:
    """Recognize a new-unit framing phrase split by a coarse caption cue."""
    raw_text = str(text or "")
    following_text = str(next_text or "")
    if not raw_text.strip() or not following_text.strip():
        return False
    joined = f"{raw_text} {following_text}"
    split = len(raw_text) + 1
    return any(
        reset.start() < split < reset.end()
        for reset in _HARD_TOPIC_RESET_RE.finditer(joined)
    )


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
        normalized_fragment = " ".join(_toks(fragment))
        if (
            _WORKED_UNIT_DISCOURSE_CONTINUATION_RE.match(fragment)
            or _WORKED_UNIT_ANAPHORIC_CONTINUATION_RE.match(fragment)
            or _WORKED_UNIT_NONQUESTION_WH_CONTINUATION_RE.match(fragment)
            or _WORKED_UNIT_NONQUESTION_WH_CONTINUATION_RE.match(
                normalized_fragment
            )
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
        intervening_text = (
            raw_text[framing.end():following_action.start()]
            if following_action is not None
            else ""
        )
        intervening_words = _toks(intervening_text)
        intervening_tokens = set(intervening_words)
        new_side = (
            following_action.start()
            if following_action is not None
            and len(intervening_words) <= 12
            and intervening_tokens <= _WORKED_UNIT_STRUCTURAL_PROMPT_TOKENS
            and _LOCAL_EXPLICIT_PROBLEM_RE.search(intervening_text) is None
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
        sentence_left = max(
            raw_text.rfind(marker, 0, action_left)
            for marker in ".!?"
        ) + 1
        local_prefix = raw_text[sentence_left:action_left]
        completion_signals = list(
            _SPLIT_CAPTION_COMPLETION_SIGNAL_RE.finditer(local_prefix)
        )
        active_prefix = (
            local_prefix[completion_signals[-1].end():]
            if completion_signals
            else local_prefix
        )
        evidence_starts_here = any(
            evidence_left == action_left and evidence_right > action_left
            for evidence_left, evidence_right in evidence_spans
        )
        framed_evidence_prompt = bool(
            evidence_starts_here
            and _worked_unit_prompt_prefix_is_structural(
                raw_text[max(0, action_left - 160):action_left]
            )
            and _LOCAL_EXPLICIT_PROBLEM_RE.search(active_prefix) is None
        )
        if _LOCAL_EXPLICIT_PROBLEM_RE.search(active_prefix) is not None:
            continue
        previous_words = list(_WORD_RE.finditer(raw_text[:action_left]))
        previous_word = (
            previous_words[-1].group(0).casefold()
            if previous_words
            else ""
        )
        if previous_word in {
            "and", "by", "can", "could", "first", "i", "must", "should",
            "then", "to", "we", "will", "would", "you",
        } and not framed_evidence_prompt:
            continue
        evidence_anchored = any(
            action_left <= evidence_left
            and evidence_left - action_left <= 160
            and evidence_right - action_left <= 240
            for evidence_left, evidence_right in evidence_spans
        )
        prefix = raw_text[:action_left]
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
        or _WORKED_UNIT_CLOSING_TAIL_RE.search(prefix)
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
    analogy_reference = _WORKED_UNIT_UNRESOLVED_ANALOGY_REFERENCE_RE.search(
        target_text
    )
    if (
        analogy_reference is not None
        and len(_toks(target_text[:analogy_reference.start()])) <= 40
    ):
        # A new-looking prompt that immediately says it is "the same" depends
        # on the comparison just before it. Keep that premise instead of
        # isolating an anaphoric unit.
        return True
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
    evidence_start_text = str(
        segments[evidence_start_line].get("text") or ""
    )
    first_evidence_word = _WORD_RE.search(evidence_start_text)
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
        cue_onsets = _worked_unit_onsets_in_cue(
            cue_text,
            evidence_spans=evidence_spans,
        )
        if (
            line + 1 == evidence_start_line
            and first_evidence_word is not None
            and evidence_left == first_evidence_word.start()
        ):
            try:
                next_gap = (
                    float(segments[line + 1].get("start", 0.0))
                    - float(segments[line].get("end", 0.0))
                )
            except (TypeError, ValueError, OverflowError):
                next_gap = float("inf")
            if math.isfinite(next_gap) and next_gap < _SECTION_RESET_GAP_S:
                cross_cue_onset = _cross_cue_grounded_action_onset(
                    cue_text,
                    str(segments[line + 1].get("text") or ""),
                )
                if cross_cue_onset is not None:
                    cue_onsets.append(cross_cue_onset)
        for navigation_left, new_side_left in sorted(set(cue_onsets)):
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
    only_target_is_explicit_example = False
    only_target_is_grounded_prompt = False
    if len(prior_or_target) == 1:
        only_target = prior_or_target[0]
        only_target_cue_text = str(
            segments[only_target.navigation_line].get("text") or ""
        )
        only_target_text = only_target_cue_text[only_target.navigation_left:]
        only_target_is_explicit_example = bool(
            _SPLIT_CAPTION_NEW_UNIT_FRAMING_RE.match(only_target_text)
        )
        cross_cue_grounded_prompt = None
        if only_target.new_side_line + 1 == evidence_start_line:
            cross_cue_grounded_prompt = _cross_cue_grounded_action_onset(
                only_target_cue_text,
                str(segments[evidence_start_line].get("text") or ""),
            )
        only_target_is_grounded_prompt = bool(
            (
                (only_target.new_side_line, only_target.new_side_left)
                == (evidence_start_line, evidence_left)
                and _WORKED_UNIT_TARGET_PROMPT_RE.match(
                    str(segments[only_target.new_side_line].get("text") or "")[
                        only_target.new_side_left:
                    ]
                )
                and _worked_unit_prompt_prefix_is_structural(
                    str(segments[only_target.new_side_line].get("text") or "")[
                        max(0, only_target.new_side_left - 160):
                        only_target.new_side_left
                    ]
                )
            )
            or cross_cue_grounded_prompt
            == (only_target.navigation_left, only_target.new_side_left)
        )
    if (
        len(prior_or_target) == 1
        and not only_target_is_explicit_example
        and not only_target_is_grounded_prompt
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


def _pedagogical_lens_transitions(
    segments: list[dict],
    start_line: int,
    end_line: int,
) -> list[_TopicTransition]:
    """Find explicit completed-unit handoffs usable as old-side endings."""
    transitions: list[_TopicTransition] = []
    for navigation_line in range(start_line, end_line + 1):
        window_end = min(end_line, navigation_line + 3)
        parts = [
            str(segments[line].get("text") or "")
            for line in range(navigation_line, window_end + 1)
        ]
        joined = " ".join(parts)
        first_cue_end = len(parts[0])
        for handoff in _PEDAGOGICAL_LENS_HANDOFF_RE.finditer(joined):
            navigation_left = handoff.start("navigation")
            if navigation_left >= first_cue_end:
                continue
            body_word = _WORD_RE.search(joined, handoff.end())
            if (
                body_word is None
                or len(_content_tokens(joined[body_word.start():])) < 4
            ):
                continue
            transition = _TopicTransition(
                navigation_line=navigation_line,
                navigation_left=navigation_left,
                new_side_line=navigation_line,
                new_side_left=navigation_left,
            )
            recent_start = max(start_line, navigation_line - 24)
            recent_parts = [
                str(segments[line].get("text") or "")
                for line in range(recent_start, navigation_line)
            ]
            recent_parts.append(
                str(segments[navigation_line].get("text") or "")[
                    :navigation_left
                ]
            )
            recent_prefix = " ".join(recent_parts)
            if not (
                _worked_unit_prefix_is_complete(
                    segments,
                    recent_start,
                    transition,
                )
                or _PEDAGOGICAL_UNIT_COMPLETION_RE.search(recent_prefix)
            ):
                continue
            transitions.append(transition)
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


def _reset_clears_recap(
    reset_pattern: re.Pattern[str],
    navigation_text: str,
) -> bool:
    """Only an explicit new-topic exit can leave an active recap region."""
    return bool(
        reset_pattern is _NEXT_DISTINCT_UNIT_RESET_RE
        or (
            reset_pattern is _HARD_TOPIC_RESET_RE
            and _STRONG_RECAP_EXIT_RE.search(navigation_text)
        )
    )


def _reset_navigation_prefix(reset: re.Match[str]) -> str:
    return reset.string[reset.start():reset.start("subject")]


def _recap_lookback_start(segments: list[dict], start_line: int) -> int:
    """Find the nearest still-active recap marker before a modelled start."""
    if start_line <= 0:
        return start_line
    reset_patterns = (_HARD_TOPIC_RESET_RE, _NEXT_DISTINCT_UNIT_RESET_RE)
    for line in range(start_line - 1, -1, -1):
        text = str(segments[line].get("text") or "")
        following = str(segments[line + 1].get("text") or "")
        joined = f"{text} {following}"
        if _EXPLICIT_RECAP_NAVIGATION_RE.search(joined):
            return line
        for pattern in reset_patterns:
            exits = [
                reset
                for reset in pattern.finditer(joined)
                if _reset_clears_recap(
                    pattern,
                    _reset_navigation_prefix(reset),
                )
                and not (
                    pattern is _HARD_TOPIC_RESET_RE
                    and _plain_same_unit_navigation_subject(reset.group("subject"))
                )
            ]
            if exits:
                return start_line
    return start_line


def _enumerated_meta_lookback_start(
    segments: list[dict],
    start_line: int,
) -> int:
    """Include a short preceding caption only when meta navigation spans it."""
    if start_line <= 0:
        return start_line
    earliest = max(0, start_line - 2)
    selected = start_line
    patterns = (_ENUMERATED_META_OUTLINE_RE, _ENUMERATED_META_UNIT_RE)
    for window_start in range(earliest, start_line):
        texts = [
            str(segments[index].get("text") or "")
            for index in range(window_start, start_line + 1)
        ]
        starts: list[int] = []
        offset = 0
        for text in texts:
            starts.append(offset)
            offset += len(text) + 1
        current_start = starts[-1]
        joined = " ".join(texts)
        for pattern in patterns:
            for match in pattern.finditer(joined):
                if not (
                    match.start("navigation") < current_start
                    <= match.end("navigation")
                ):
                    continue
                match_line_index = max(
                    index
                    for index, cue_start in enumerate(starts)
                    if cue_start <= match.start("navigation")
                )
                selected = min(selected, window_start + match_line_index)
    return selected


def _candidate_topic_transitions(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    evidence_quote: str,
    learning_objective: str,
    relationship_bridge_allowed: bool | None = None,
    atomic_claim_required: bool = False,
    atomic_scope_text: str = "",
    comparison_subject_anchors: tuple[frozenset[str], ...] = (),
) -> list[_TopicTransition]:
    """Return unbridged same-cue and adjacent-cue topic navigation."""
    cue_texts = [
        str(segments[line].get("text") or "")
        for line in range(start_line, end_line + 1)
    ]
    dotted_text = " . ".join(cue_texts)
    joined_text = " ".join(cue_texts)
    reset_patterns = (
        _HARD_TOPIC_RESET_RE,
        _INDEPENDENT_UNIT_RESET_RE,
        _NAMED_UNIT_LABEL_RESET_RE,
        _NAMED_METHOD_CONTRAST_RESET_RE,
        _NEXT_DISTINCT_UNIT_RESET_RE,
    )
    has_hard_reset = any(
        pattern.search(dotted_text) or pattern.search(joined_text)
        for pattern in reset_patterns
    )
    has_worked_unit_onset = bool(
        _WORKED_UNIT_POSSIBLE_ONSET_RE.search(dotted_text)
        or _WORKED_UNIT_POSSIBLE_ONSET_RE.search(joined_text)
    )
    detected_lens_transitions = _pedagogical_lens_transitions(
        segments,
        start_line,
        end_line,
    )
    explicit_boundary_transitions: list[_TopicTransition] = []
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        for match in _EXPLICIT_RECAP_NAVIGATION_RE.finditer(text):
            new_side_line = line
            new_side_left = match.end("navigation")
            if (
                not _WORD_RE.search(text[new_side_left:])
                and line < end_line
            ):
                next_text = str(segments[line + 1].get("text") or "")
                first_next_word = _WORD_RE.search(next_text)
                if first_next_word is not None:
                    new_side_line = line + 1
                    new_side_left = first_next_word.start()
            explicit_boundary_transitions.append(_TopicTransition(
                navigation_line=line,
                navigation_left=match.start("navigation"),
                new_side_line=new_side_line,
                new_side_left=new_side_left,
                recap=True,
            ))
        if relationship_bridge_allowed is not True:
            explicit_boundary_transitions.extend(
                _TopicTransition(
                    navigation_line=line,
                    navigation_left=match.start("navigation"),
                    new_side_line=line,
                    new_side_left=match.start("navigation"),
                    worked_unit=True,
                )
                for match in _NEW_HYPOTHETICAL_EXAMPLE_RE.finditer(text)
            )
        for pattern in (
            _ENUMERATED_META_OUTLINE_RE,
            _ENUMERATED_META_UNIT_RE,
        ):
            explicit_boundary_transitions.extend(
                _TopicTransition(
                    navigation_line=line,
                    navigation_left=match.start("navigation"),
                    new_side_line=line,
                    new_side_left=match.end("navigation"),
                )
                for match in pattern.finditer(text)
            )
    for line in range(start_line, end_line):
        left_text = str(segments[line].get("text") or "")
        right_text = str(segments[line + 1].get("text") or "")
        joined = f"{left_text} {right_text}"
        split = len(left_text) + 1
        for match in _EXPLICIT_RECAP_NAVIGATION_RE.finditer(joined):
            if not (match.start("navigation") < split < match.end("navigation")):
                continue
            explicit_boundary_transitions.append(_TopicTransition(
                navigation_line=line,
                navigation_left=match.start("navigation"),
                new_side_line=line + 1,
                new_side_left=max(0, match.end("navigation") - split),
                recap=True,
            ))
        if (
            relationship_bridge_allowed is not True
            and _NEW_HYPOTHETICAL_EXAMPLE_RE.search(left_text) is None
        ):
            for match in _NEW_HYPOTHETICAL_EXAMPLE_RE.finditer(joined):
                if match.start("navigation") >= split:
                    continue
                explicit_boundary_transitions.append(_TopicTransition(
                    navigation_line=line,
                    navigation_left=match.start("navigation"),
                    new_side_line=line,
                    new_side_left=match.start("navigation"),
                    worked_unit=True,
                ))
        for pattern in (
            _ENUMERATED_META_OUTLINE_RE,
            _ENUMERATED_META_UNIT_RE,
        ):
            for match in pattern.finditer(joined):
                if not (match.start("navigation") < split < match.end("navigation")):
                    continue
                explicit_boundary_transitions.append(_TopicTransition(
                    navigation_line=line,
                    navigation_left=match.start("navigation"),
                    new_side_line=line + 1,
                    new_side_left=max(0, match.end("navigation") - split),
                ))
    for line in range(start_line, end_line - 1):
        texts = [
            str(segments[index].get("text") or "")
            for index in range(line, line + 3)
        ]
        starts = [0, len(texts[0]) + 1]
        starts.append(starts[1] + len(texts[1]) + 1)
        joined = " ".join(texts)
        for pattern in (
            _ENUMERATED_META_OUTLINE_RE,
            _ENUMERATED_META_UNIT_RE,
        ):
            for match in pattern.finditer(joined):
                if not (
                    match.start("navigation") < starts[1]
                    and match.end("navigation") > starts[2]
                ):
                    continue
                new_side_index = max(
                    index
                    for index, cue_start in enumerate(starts)
                    if cue_start <= match.end("navigation")
                )
                explicit_boundary_transitions.append(_TopicTransition(
                    navigation_line=line,
                    navigation_left=match.start("navigation"),
                    new_side_line=line + new_side_index,
                    new_side_left=max(
                        0,
                        match.end("navigation") - starts[new_side_index],
                    ),
                ))
    evidence_location = _unique_evidence_location(
        segments,
        evidence_quote,
        start_line,
        end_line,
    )
    atomic_transitions = (
        _claim_anchored_atomic_transitions(
            segments,
            start_line,
            end_line,
            evidence_location=evidence_location,
            evidence_quote=evidence_quote,
            scope_text=atomic_scope_text,
            relationship_bridge_allowed=bool(relationship_bridge_allowed),
        )
        if atomic_claim_required
        else []
    )
    comparison_transitions = _comparison_arc_transitions(
        segments,
        start_line,
        end_line,
        evidence_location=evidence_location,
        scope_text=f"{learning_objective} {atomic_scope_text}",
        subject_anchors=comparison_subject_anchors,
    )
    if (
        not has_hard_reset
        and not has_worked_unit_onset
        and not detected_lens_transitions
        and not explicit_boundary_transitions
        and not atomic_transitions
        and not comparison_transitions
    ):
        return []
    evidence_locations = {
        line: _quote_character_spans(
            str(segments[line].get("text") or ""), evidence_quote
        )
        for line in range(start_line, end_line + 1)
    }
    transitions: list[_TopicTransition] = [
        *atomic_transitions,
        *comparison_transitions,
        *explicit_boundary_transitions,
    ]
    if evidence_location is not None:
        evidence_end = evidence_location[2], evidence_location[3]
        transitions.extend(
            transition
            for transition in detected_lens_transitions
            if (transition.navigation_line, transition.navigation_left)
            > evidence_end
        )
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        sentence_spans = _sentence_character_spans(text)
        for reset_pattern in reset_patterns:
            for reset in reset_pattern.finditer(text):
                subject = reset.group("subject")
                if _plain_same_unit_navigation_subject(subject):
                    continue

                reset_left = (
                    reset.start()
                    if reset_pattern is _HARD_TOPIC_RESET_RE
                    else reset.start("navigation")
                )
                sentence_left, sentence_right = next(
                    (
                        (left, right)
                        for left, right in sentence_spans
                        if left <= reset_left < right
                    ),
                    (reset_left, len(text)),
                )
                navigation_left = reset_left
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
                objective_bridges = _objective_bridges_sections(
                    learning_objective,
                    " ".join(left_parts),
                    " ".join(right_parts),
                    reset_subject=subject,
                ) or _conditional_handoff_bridges_objective(
                    reset,
                    learning_objective,
                )
                can_bridge = bool(
                    objective_bridges
                    and (
                        relationship_bridge_allowed is None
                        or relationship_bridge_allowed
                        or (
                            reset_pattern is _HARD_TOPIC_RESET_RE
                            and not _objective_explicitly_relates_sections(
                                learning_objective
                            )
                        )
                    )
                )
                if can_bridge:
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
                    clears_recap=_reset_clears_recap(
                        reset_pattern,
                        _reset_navigation_prefix(reset),
                    ),
                    reset_subject=subject,
                    named_method_handoff=(
                        reset_pattern is _NAMED_METHOD_CONTRAST_RESET_RE
                    ),
                ))

    for line in range(start_line, end_line):
        left_text = str(segments[line].get("text") or "")
        right_text = str(segments[line + 1].get("text") or "")
        joined = f"{left_text} {right_text}"
        split = len(left_text) + 1
        sentence_spans = _sentence_character_spans(joined)
        for reset_pattern in reset_patterns:
            for reset in reset_pattern.finditer(joined):
                reset_left = (
                    reset.start()
                    if reset_pattern is _HARD_TOPIC_RESET_RE
                    else reset.start("navigation")
                )
                if not (reset_left < split < reset.end("subject")):
                    continue
                subject = reset.group("subject")
                if _plain_same_unit_navigation_subject(subject):
                    continue
                navigation_left = reset_left
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
                objective_bridges = _objective_bridges_sections(
                    learning_objective,
                    " ".join(left_parts),
                    " ".join(right_parts),
                    reset_subject=subject,
                ) or _conditional_handoff_bridges_objective(
                    reset,
                    learning_objective,
                )
                can_bridge = bool(
                    objective_bridges
                    and (
                        relationship_bridge_allowed is None
                        or relationship_bridge_allowed
                        or (
                            reset_pattern is _HARD_TOPIC_RESET_RE
                            and not _objective_explicitly_relates_sections(
                                learning_objective
                            )
                        )
                    )
                )
                if can_bridge:
                    continue
                _sentence_left, sentence_right = next(
                    (
                        (left, right)
                        for left, right in sentence_spans
                        if left <= reset_left < right
                    ),
                    (reset_left, len(joined)),
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
                    clears_recap=_reset_clears_recap(
                        reset_pattern,
                        _reset_navigation_prefix(reset),
                    ),
                    reset_subject=subject,
                    named_method_handoff=(
                        reset_pattern is _NAMED_METHOD_CONTRAST_RESET_RE
                    ),
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
        selected = transition
        if previous is not None and (
            previous.new_side_line,
            previous.new_side_left,
        ) >= (transition.new_side_line, transition.new_side_left):
            selected = previous
        unique[key] = replace(
            selected,
            recap=transition.recap or bool(previous and previous.recap),
            clears_recap=(
                transition.clears_recap
                or bool(previous and previous.clears_recap)
            ),
            reset_subject=(
                transition.reset_subject
                or (previous.reset_subject if previous is not None else "")
            ),
            named_method_handoff=(
                transition.named_method_handoff
                or bool(previous and previous.named_method_handoff)
            ),
        )
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
        if (
            _objective_bridges_sections(
                learning_objective,
                raw_text[:navigation_left],
                raw_text[reset.start("subject"):],
                reset_subject=reset.group("subject"),
            )
            or _conditional_handoff_bridges_objective(
                reset,
                learning_objective,
            )
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


def _compact_evidence_explicitly_relates_sections(evidence_quote: str) -> bool:
    """Accept a spoken comparison marker without treating a bare ``but`` as one."""
    normalized = " ".join(str(evidence_quote or "").split())
    if (
        _EXPLICIT_RELATIONAL_OBJECTIVE_RE.search(normalized)
        or _DIRECT_COMPARISON_CLAUSE_RE.search(normalized)
    ):
        return True
    return any(
        match.group(0).strip().casefold() != "but"
        for match in _SPOKEN_COMPARISON_RELATION_RE.finditer(normalized)
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
                (
                    _SPLIT_CAPTION_COMPLETION_SIGNAL_RE.search(combined_tail)
                    or _last_safe_complete_prefix(combined_tail)
                    == combined_tail.rstrip(" ,;:—-")
                )
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
            trailing_parts = [selected[closing.end():]]
            for later_line in range(line + 1, transition.navigation_line + 1):
                later_text = str(segments[later_line].get("text") or "")
                later_right = (
                    transition.navigation_left
                    if later_line == transition.navigation_line
                    else len(later_text)
                )
                trailing_parts.append(later_text[:later_right])
            trailing_text = " ".join(trailing_parts).strip(" ,;:—-")
            if (
                _WORD_RE.search(trailing_text)
                and not _cue_is_only_structural_filler(trailing_text)
            ):
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
    """Reject any structural interruption left after clean-side projection."""
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
    return "internal_structural_filler"


def _same_cue_internal_filler_reason(text: str) -> str | None:
    """Reject an interruption that cannot be removed from one coarse cue."""
    raw_text = str(text or "")
    blocks: set[tuple[int, int]] = set()
    for match in _NON_SPEECH_MARKER_RE.finditer(raw_text):
        if (
            _WORD_RE.search(raw_text[:match.start()])
            and _WORD_RE.search(raw_text[match.end():])
        ):
            blocks.add((match.start(), match.end()))
    for match in _INTERNAL_CHANNEL_PROMO_RE.finditer(raw_text):
        if (
            _WORD_RE.search(raw_text[:match.start()])
            and _WORD_RE.search(raw_text[match.end():])
        ):
            blocks.add((match.start(), match.end()))
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
    for left, right in _sentence_character_spans(raw_text):
        sentence = raw_text[left:right]
        if (
            _cue_is_only_structural_filler(sentence)
            and _WORD_RE.search(raw_text[:left])
            and _WORD_RE.search(raw_text[right:])
        ):
            blocks.add((left, right))
    return "internal_structural_filler" if blocks else None


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


def _terminal_meta_tail_is_only_meta(text: str) -> bool:
    """Return true only for bounded watch/admin copy after a terminal preview."""
    raw_text = str(text or "").strip()
    if not _WORD_RE.search(raw_text):
        return True
    sentences = [
        raw_text[left:right].strip()
        for left, right in _sentence_character_spans(raw_text)
        if _WORD_RE.search(raw_text[left:right])
    ]
    return bool(sentences) and all(
        _cue_is_only_structural_filler(sentence)
        or (
            _TERMINAL_MEDIA_PROMO_RE.fullmatch(sentence) is not None
            and _TERMINAL_MEDIA_SUBJECT_SENSE_RE.search(sentence) is None
        )
        for sentence in sentences
    )


def _framed_terminal_meta_tail_start(
    remaining: str,
    preview_parts: list[str],
    noise: re.Match[str],
) -> int:
    """Skip the one sentence grammatically introduced by a meta prefix."""
    containing_sentence = next(
        (
            (left, right)
            for left, right in _sentence_character_spans(remaining)
            if left <= noise.start() < right and noise.end() <= right
        ),
        (noise.start(), noise.end()),
    )
    _sentence_left, sentence_right = containing_sentence
    if (
        sentence_right < len(remaining)
        or re.search(r"[.!?]", remaining[noise.end():sentence_right])
    ):
        return sentence_right
    # Punctuationless captions still expose cue boundaries.  When the cue
    # already completes the meta claim, protect teaching in later cues.
    joined_parts = [part for part in preview_parts if part]
    cursor = 0
    matched_part_index: int | None = None
    matched_part_end = 0
    matched_part_tail = ""
    part_ends: list[int] = []
    for index, part in enumerate(joined_parts):
        part_start = cursor
        part_end = part_start + len(part)
        part_ends.append(part_end)
        if matched_part_index is None and noise.end() <= part_end:
            matched_part_index = index
            matched_part_end = part_end
            matched_part_tail = part[max(0, noise.end() - part_start):]
        cursor = part_end + 1
    if matched_part_index is not None:
        continuation = matched_part_tail.strip()
        continuation_index = matched_part_index
        continuation_end = matched_part_end
        if not _WORD_RE.search(continuation):
            continuation_index += 1
            if continuation_index >= len(joined_parts):
                return sentence_right
            continuation = joined_parts[continuation_index].strip()
            continuation_end = part_ends[continuation_index]
        while continuation_index + 1 < len(joined_parts):
            next_part = joined_parts[continuation_index + 1].strip()
            requires_object = bool(
                _TERMINAL_META_REQUIRED_OBJECT_RE.fullmatch(continuation)
            )
            continuation_is_incomplete = bool(
                _TERMINAL_META_CONTINUATION_INCOMPLETE_RE.search(continuation)
                or requires_object
                or _cue_has_explicit_dangling_end(continuation, next_part)
            )
            if not continuation_is_incomplete:
                break
            continuation_index += 1
            continuation = " ".join(
                part for part in (continuation, next_part) if part
            )
            continuation_end = part_ends[continuation_index]
        return continuation_end
    return sentence_right


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

    selected_parts = {
        line: selected_part(line)
        for line in range(start_line, end_line + 1)
    }
    substantive_suffix: dict[int, bool] = {end_line + 1: False}
    for line in range(end_line, start_line - 1, -1):
        part = selected_parts[line]
        substantive_suffix[line] = bool(
            substantive_suffix[line + 1]
            or (
                _WORD_RE.search(part)
                and not _terminal_meta_tail_is_only_meta(part)
            )
        )

    prefix_parts: list[str] = []
    for line in range(start_line, end_line + 1):
        selected = selected_parts[line]
        preview_parts = [selected]
        preview_characters = len(selected)
        preview_last_line = line
        for later in range(line + 1, min(end_line + 1, line + 12)):
            part = selected_parts[later].strip()
            if part:
                preview_parts.append(part)
                preview_characters += len(part) + 1
                preview_last_line = later
            if preview_characters >= 320:
                break
        remaining = " ".join(part for part in preview_parts if part)
        cross_cue_future_preview = _TERMINAL_FUTURE_PREVIEW_RE.search(remaining)
        future_preview = (
            cross_cue_future_preview
            if cross_cue_future_preview is not None
            and cross_cue_future_preview.start() < len(selected)
            else None
        )
        if future_preview is not None:
            tail_start = (
                future_preview.end()
                if future_preview.group("media_promise") is not None
                else _framed_terminal_meta_tail_start(
                    remaining,
                    preview_parts,
                    future_preview,
                )
            )
            local_tail = remaining[tail_start:].strip()
            if (
                (
                    _WORD_RE.search(local_tail)
                    and not _terminal_meta_tail_is_only_meta(local_tail)
                )
                or substantive_suffix.get(preview_last_line + 1, False)
            ):
                future_preview = None
        cross_cue_mastery_recap = _TERMINAL_MASTERY_RECAP_RE.search(remaining)
        mastery_recap = (
            cross_cue_mastery_recap
            if cross_cue_mastery_recap is not None
            and cross_cue_mastery_recap.start() < len(selected)
            else None
        )
        if mastery_recap is not None:
            tail_start = _framed_terminal_meta_tail_start(
                remaining,
                preview_parts,
                mastery_recap,
            )
            local_tail = remaining[tail_start:].strip()
            if (
                (
                    _WORD_RE.search(local_tail)
                    and not _terminal_meta_tail_is_only_meta(local_tail)
                )
                or substantive_suffix.get(preview_last_line + 1, False)
            ):
                mastery_recap = None
        noise = future_preview or mastery_recap
        if noise is None:
            if selected.strip():
                prefix_parts.append(selected.strip())
            continue
        retained_here = selected[:noise.start()].rstrip(" ,;:—-")
        prefix_text = " ".join(
            part for part in (*prefix_parts, retained_here) if part
        ).strip()
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


def _has_unresolved_deictic_point_pair(text: str) -> bool:
    """Require spoken identities for paired ``this/that point`` references."""
    raw_text = str(text or "")
    for match in _DEICTIC_POINT_DEPENDENCY_RE.finditer(raw_text):
        definitions = [
            item.group("label").casefold()
            for item in _DEICTIC_POINT_DEFINITION_RE.finditer(
                raw_text[:match.start()]
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


def _contract_has_relational_path(
    topic: str,
    constraints: dict[str, _IntentConstraint],
) -> bool:
    """Recognize model-structured paths without treating every ``noun to noun`` as one."""
    endpoints = [
        constraint
        for constraint in constraints.values()
        if constraint.kind in {
            _IntentConstraintKind.SUBJECT,
            _IntentConstraintKind.OUTCOME,
        }
    ]
    relations = [
        constraint
        for constraint in constraints.values()
        if constraint.kind is _IntentConstraintKind.RELATIONSHIP
    ]
    if len(endpoints) < 2 or not relations:
        return False

    request = " ".join(str(topic or "").split())
    endpoint_spans = [
        (span, constraint)
        for constraint in endpoints
        for span in _quote_character_spans(request, constraint.source_phrase)
    ]
    if len(endpoint_spans) < 2:
        return False
    if re.search(r"\bfrom\b[^,;.!?]{1,160}\b(?:into|to)\b", request, re.I):
        return True

    for relation in relations:
        source_phrase = relation.source_phrase
        if re.search(r"\b(?:into|through|using|via|with)\b", source_phrase, re.I):
            return True
        if re.search(r"\bto\b", source_phrase, re.I) is None:
            continue
        for relation_span in _quote_character_spans(request, source_phrase):
            left = [
                (span, constraint)
                for span, constraint in endpoint_spans
                if span[1] <= relation_span[0]
            ]
            right = [
                (span, constraint)
                for span, constraint in endpoint_spans
                if span[0] >= relation_span[1]
            ]
            if not left or not right:
                continue
            left_constraint = max(left, key=lambda item: item[0][1])[1]
            if _NON_DIRECTIONAL_TO_HEAD_RE.search(left_constraint.source_phrase):
                continue
            return True
    return False


def _request_requires_joint_intent_coverage(
    topic: str,
    constraints: dict[str, _IntentConstraint] | None = None,
) -> bool:
    """Return whether one clip must fulfill the whole multi-part request."""
    request = " ".join(str(topic or "").split())
    comparison_request = re.search(
        r"\b(?:versus|vs\.?|compare(?:d|s|ing)?|comparison|contrast|"
        r"difference\s+between)\b|/",
        request,
        re.IGNORECASE,
    )
    if comparison_request is not None:
        return True
    if _EXPLICIT_TRANSITION_REQUEST_RE.search(request) is not None:
        return True
    if "," in request or ";" in request:
        return False
    if _EXPLICIT_CONJUNCTIVE_REQUEST_RE.search(request) is not None:
        return True
    if not constraints:
        return False
    return _contract_has_relational_path(request, constraints)


def _joint_subject_anchor_tokens(
    constraint: _IntentConstraint,
    constraints: dict[str, _IntentConstraint],
) -> set[str]:
    """Keep the words that distinguish one named side from its peers."""

    def tokens(value: str) -> set[str]:
        raw = set(_toks(value))
        return _content_tokens(value) | {
            token
            for token in raw
            if token.isdecimal() or re.fullmatch(r"[ivxlcdm]+", token)
        }

    own = tokens(constraint.source_phrase)
    peer_tokens = set().union(*(
        tokens(peer.source_phrase)
        for peer in constraints.values()
        if peer.constraint_id != constraint.constraint_id
        and peer.kind in {
            _IntentConstraintKind.SUBJECT,
            _IntentConstraintKind.OUTCOME,
        }
    ))
    return (own - peer_tokens) or own


def _joint_subject_evidence_matches(
    constraint: _IntentConstraint,
    quote: str,
    constraints: dict[str, _IntentConstraint],
) -> bool:
    """Require evidence for a named side to actually name that distinct side."""
    anchors = _joint_subject_anchor_tokens(constraint, constraints)
    evidence = _content_tokens(quote) | {
        token
        for token in _toks(quote)
        if token.isdecimal() or re.fullmatch(r"[ivxlcdm]+", token)
    }
    return bool(
        anchors
        and any(
            anchor == token
            or (
                len(anchor) >= 6
                and len(token) >= 6
                and anchor[:6] == token[:6]
            )
            or (
                min(len(anchor), len(token)) >= 4
                and anchor.rstrip("e") == token.rstrip("e")
            )
            for anchor in anchors
            for token in evidence
        )
    )


def _joint_subject_evidence_window(
    text: str,
    constraint: _IntentConstraint,
    constraints: dict[str, _IntentConstraint],
) -> str:
    """Find the shortest grounded window naming one distinct joint subject."""
    words = list(_WORD_RE.finditer(str(text or "")))
    for width in range(5, min(16, len(words)) + 1):
        for offset in range(0, len(words) - width + 1):
            quote = str(text or "")[
                words[offset].start():words[offset + width - 1].end()
            ]
            if _joint_subject_evidence_matches(
                constraint,
                quote,
                constraints,
            ):
                return quote
    return ""


def _joint_relationship_evidence_window(
    text: str,
    topic: str,
    constraints: dict[str, _IntentConstraint],
    subject_evidence: dict[str, str],
) -> str:
    """Infer only an explicitly spoken comparison/transition or a true conjunction."""
    comparison_request = bool(
        re.search(
            r"\b(?:versus|vs\.?|compare(?:d|s|ing)?|comparison|contrast|"
            r"difference\s+between)\b|/",
            str(topic or ""),
            re.IGNORECASE,
        )
    )
    transition_request = bool(_EXPLICIT_TRANSITION_REQUEST_RE.search(topic))
    path_request = bool(
        not comparison_request
        and not transition_request
        and _contract_has_relational_path(topic, constraints)
    )
    if not comparison_request and not transition_request and not path_request:
        return next(iter(subject_evidence.values()), "")

    words = list(_WORD_RE.finditer(str(text or "")))
    for width in range(5, min(16, len(words)) + 1):
        for offset in range(0, len(words) - width + 1):
            quote = str(text or "")[
                words[offset].start():words[offset + width - 1].end()
            ]
            if _joint_relationship_evidence_matches(
                quote,
                topic,
                constraints,
            ):
                return quote
    return ""


def _joint_relationship_evidence_matches(
    quote: str,
    topic: str,
    constraints: dict[str, _IntentConstraint],
) -> bool:
    """Require a spoken relation between both named endpoints in one sentence."""
    request = str(topic or "")
    comparison_request = bool(
        re.search(
            r"\b(?:versus|vs\.?|compare(?:d|s|ing)?|comparison|contrast|"
            r"difference\s+between)\b|/",
            request,
            re.IGNORECASE,
        )
    )
    transition_request = bool(_EXPLICIT_TRANSITION_REQUEST_RE.search(request))
    path_request = bool(
        not comparison_request
        and not transition_request
        and _contract_has_relational_path(request, constraints)
    )
    if not comparison_request and not transition_request and not path_request:
        return True

    endpoints = [
        constraint
        for constraint in constraints.values()
        if constraint.kind in {
            _IntentConstraintKind.SUBJECT,
            _IntentConstraintKind.OUTCOME,
        }
    ]
    if len(endpoints) < 2:
        return False
    source = str(quote or "")
    sentences = [
        source[left:right]
        for left, right in _sentence_character_spans(source)
        if _WORD_RE.search(source[left:right])
    ] or [source]
    for sentence in sentences:
        endpoint_count = sum(
            _joint_subject_evidence_matches(
                constraint,
                sentence,
                constraints,
            )
            for constraint in endpoints
        )
        if endpoint_count < 2:
            continue
        if comparison_request:
            relation_is_explicit = bool(
                _EXPLICIT_COMPARISON_OBJECTIVE_RE.search(sentence)
                or _DIRECT_COMPARISON_CLAUSE_RE.search(sentence)
                or _SPOKEN_COMPARISON_RELATION_RE.search(sentence)
                or (
                    _COMPARISON_DISTINCTION_RE.search(sentence)
                    and _COMPARISON_CONNECTOR_RE.search(sentence)
                )
            )
        else:
            relation_is_explicit = bool(
                _EXPLICIT_TRANSITION_REQUEST_RE.search(sentence)
                or _SPOKEN_PATH_RELATION_RE.search(sentence)
            )
        if relation_is_explicit:
            return True
    return False


def _canonical_binary_comparison_constraints(
    request: str,
    constraints: dict[str, _IntentConstraint],
) -> dict[str, _IntentConstraint] | None:
    """Repair a malformed model contract for an unambiguous ``X vs Y`` request."""
    match = re.fullmatch(
        r"\s*(?P<left>.+?)\s+(?P<connector>versus|vs\.?)\s+(?P<right>.+?)\s*",
        str(request or ""),
        re.IGNORECASE,
    )
    if match is None or not all(
        constraint.kind in {
            _IntentConstraintKind.SUBJECT,
            _IntentConstraintKind.RELATIONSHIP,
        }
        for constraint in constraints.values()
    ):
        return None
    left = match.group("left").strip()
    connector = match.group("connector").strip()
    right = match.group("right").strip()
    if not _content_tokens(left) or not _content_tokens(right):
        return None
    canonical = [
        _IntentConstraint(
            constraint_id="joint_subject_1",
            kind=_IntentConstraintKind.SUBJECT,
            source_phrase=left,
            requirement=f"Teach {left}",
        ),
        _IntentConstraint(
            constraint_id="joint_relationship",
            kind=_IntentConstraintKind.RELATIONSHIP,
            source_phrase=connector,
            requirement=f"Compare {left} with {right}",
        ),
        _IntentConstraint(
            constraint_id="joint_subject_2",
            kind=_IntentConstraintKind.SUBJECT,
            source_phrase=right,
            requirement=f"Teach {right}",
        ),
    ]
    return {constraint.constraint_id: constraint for constraint in canonical}


def _validated_intent_constraints(
    plan: object,
    topic: str,
) -> tuple[dict[str, _IntentConstraint], str | None]:
    """Validate the selector's same-call interpretation against the exact request."""
    if not isinstance(plan, (_CompactBoundaryPlan, _IntentBoundaryPlan)):
        return {}, None
    # An unfiltered source has no user request to protect.  Do not let a model's
    # harmless rewrite of the synthetic all-topics placeholder reject every
    # otherwise valid educational unit.
    expected_request = topic.strip()
    if not expected_request:
        return {}, None
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
    if _request_requires_joint_intent_coverage(expected_request, constraints):
        comparison_request = bool(
            re.search(
                r"\b(?:versus|vs\.?|compare(?:d|s|ing)?|comparison|contrast|"
                r"difference\s+between)\b|/",
                expected_request,
                re.IGNORECASE,
            )
        )
        relationship_request = bool(
            comparison_request
            or _EXPLICIT_TRANSITION_REQUEST_RE.search(expected_request)
            or any(
                constraint.kind is _IntentConstraintKind.RELATIONSHIP
                for constraint in constraints.values()
            )
        )
        incomplete_relationship_structure = bool(
            len(constraints) < 2
            or (
                relationship_request
                and (
                    not any(
                        constraint.kind is _IntentConstraintKind.RELATIONSHIP
                        for constraint in constraints.values()
                    )
                    or sum(
                        constraint.kind is not _IntentConstraintKind.RELATIONSHIP
                        for constraint in constraints.values()
                    ) < 2
                )
            )
        )
        if incomplete_relationship_structure:
            repaired = _canonical_binary_comparison_constraints(
                expected_request,
                constraints,
            )
            if repaired is None:
                return {}, "intent_contract_incomplete_joint_structure"
            constraints = repaired
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


def _best_effort_evidence_quote(text: str) -> str:
    """Return one literal local quote when model evidence cannot be anchored."""
    words = list(_WORD_RE.finditer(str(text or "")))
    if not words:
        return ""
    chosen = words[:min(16, len(words))]
    return str(text or "")[chosen[0].start():chosen[-1].end()]


_TRUSTED_ORDINAL_OPENING_RE = re.compile(
    r"^\s*(?:(?:and|but|now|so)\s*[,;:]?\s+)?the\s+"
    r"(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|"
    r"ninth|tenth|\d+(?:st|nd|rd|th))\s+"
    r"(?P<head>[a-z][\w'’-]*)\b",
    re.IGNORECASE,
)
_TRUSTED_DEMONSTRATIVE_OPENING_RE = re.compile(
    r"^\s*(?:(?:and|but|now|so)\s*[,;:]?\s+)?"
    r"(?:this|that|these|those)\s+"
    r"(?P<head>[a-z][\w'’-]*)\b",
    re.IGNORECASE,
)
_TRUSTED_POSSESSIVE_OPENING_RE = re.compile(
    r"^\s*(?:(?:and|but|now|so)\s*[,;:]?\s+)?(?:its|their)\s+"
    r"(?P<head>[a-z][\w'’-]*)\b",
    re.IGNORECASE,
)
_TRUSTED_PROJECTED_SETUP_RE = re.compile(
    r"^\s*(?:(?:now|so)\s*[,;:]?\s+)*"
    r"(?:consider|imagine|picture|suppose|take|let(?:['’]?s|\s+us)?)\b",
    re.IGNORECASE,
)
_TRUSTED_CONTEXTUAL_CUE_OPENING_RE = re.compile(
    r"^\s*about\b|"
    r"^\s*(?:after|based\s+on|using|with)\s+"
    r"(?:this|that|these|those|such)\b",
    re.IGNORECASE,
)
_TRUSTED_SUBJECTLESS_PREDICATE_OPENING_RE = re.compile(
    r"^\s*(?:means?|implies?|indicates?|shows?|suggests?)\s+(?:that\b|how\b|why\b)",
    re.IGNORECASE,
)
_TRUSTED_SPLIT_COPULA_COMPLEMENT_OPENING_RE = re.compile(
    r"^\s*(?:(?:helpful|important|necessary|useful)\s+to\b|"
    r"(?:clear|crucial|essential|likely|possible|unlikely)\s+that\b)",
    re.IGNORECASE,
)
_TRUSTED_TRAILING_SUBJECT_COPULA_RE = re.compile(
    r"(?P<setup>\b(?:it|that|this)\s+(?:is|was|"
    r"(?:can|could|may|might|must|should|will|would)\s+be))\s*$",
    re.IGNORECASE,
)
_TRUSTED_SYMBOLIC_EQUATION_RE = re.compile(
    r"(?<!\w)[a-z][a-z0-9_]*\s*=\s*"
    r"[a-z0-9][a-z0-9_]*(?:\s*[+\-*/^]\s*[a-z0-9][a-z0-9_]*)*",
    re.IGNORECASE,
)
_TRUSTED_SCENARIO_HANDOFF_RE = re.compile(
    r"(?<!\w)(?:(?:and|but|so)\s+)?(?:now\s+)?(?P<setup>what\s+if\b)",
    re.IGNORECASE,
)
_TRUSTED_EXPLICIT_DEFINITION_RE = re.compile(
    r"^\s*(?:(?:and|but|so)\s*[,;:]?\s+)?"
    r"(?P<subject>[a-z][\w'’-]*(?:\s+[a-z][\w'’-]*){0,2})\s+"
    r"(?:is|are|means?|refers?\s+to)\b",
    re.IGNORECASE,
)
_TRUSTED_DEFINITION_SCOPE_RE = re.compile(
    r"\b(?:define|defining|definition)\b",
    re.IGNORECASE,
)
_TRUSTED_SPEAKER_FRAMING_SETUP_RE = re.compile(
    r"^\s*(?:professor|doctor|dr\.?)\s+[^,;:!?]{1,60}"
    r"[,;:—-]\s*(?P<setup>(?:i|we)\s+(?:want|would\s+like)\s+to\s+"
    r"(?:explain|show|teach|tell)\b)",
    re.IGNORECASE,
)


def _trusted_opening_reference_is_resolved(
    opening_text: str,
    prior_text: str,
) -> bool:
    """Require a literal spoken antecedent for a trusted dependent opening."""
    opening = str(opening_text or "").strip()
    prior = str(prior_text or "")
    ordinal = _TRUSTED_ORDINAL_OPENING_RE.match(opening)
    if ordinal is not None:
        phrase = re.compile(
            rf"\b{re.escape(ordinal.group('ordinal'))}\s+"
            rf"{re.escape(ordinal.group('head'))}\b",
            re.IGNORECASE,
        )
        for match in phrase.finditer(prior):
            if re.search(r"\bthe\s*$", prior[:match.start()], re.IGNORECASE):
                continue
            return True
        return False

    demonstrative = _TRUSTED_DEMONSTRATIVE_OPENING_RE.match(opening)
    if demonstrative is not None:
        return bool(re.search(
            rf"\b{re.escape(demonstrative.group('head'))}s?\b",
            prior,
            re.IGNORECASE,
        ))

    possessive = _TRUSTED_POSSESSIVE_OPENING_RE.match(opening)
    if possessive is not None:
        return bool(re.search(
            rf"\b{re.escape(possessive.group('head'))}s?\b",
            prior,
            re.IGNORECASE,
        ))

    from .discourse import _has_unresolved_opening_back_reference

    return not _has_unresolved_opening_back_reference(
        opening,
        prior_text=prior,
    )


def _trusted_hard_reset_start_span(
    text: str,
    *,
    before: int | None = None,
) -> tuple[int, int] | None:
    """Keep a named structural handoff while dropping rolling-caption text before it."""
    source = str(text or "")
    resets = [
        reset
        for reset in _HARD_TOPIC_RESET_RE.finditer(source)
        if before is None or reset.start() < before
    ]
    if not resets:
        return None
    reset = resets[-1]
    quote = _exact_boundary_quote(source[reset.start():], want="start")
    return _quote_character_span(source, quote) if quote else None


def _trusted_scenario_start_span(text: str) -> tuple[int, int] | None:
    """Recover a literal fresh what-if setup inside a coarse prior cue."""
    source = str(text or "")
    handoffs = list(_TRUSTED_SCENARIO_HANDOFF_RE.finditer(source))
    if not handoffs:
        return None
    setup_left = handoffs[-1].start()
    suffix = source[setup_left:]
    if len(_toks(suffix)) < 6:
        return None
    quote = _exact_boundary_quote(suffix, want="start")
    return _quote_character_span(source, quote) if quote else None


def _trusted_speaker_setup_start_span(text: str) -> tuple[int, int] | None:
    """Drop a speaker tag while retaining its complete first-person topic setup."""
    source = str(text or "")
    framing = _TRUSTED_SPEAKER_FRAMING_SETUP_RE.match(source)
    if framing is None:
        return None
    setup_left = framing.start("setup")
    quote = _exact_boundary_quote(source[setup_left:], want="start")
    return _quote_character_span(source, quote) if quote else None


def _trusted_weak_prior_start_span(
    prior_text: str,
    selected_text: str,
) -> tuple[int, int] | None:
    """Recover the exact prior-sentence onset of a split caption phrase."""
    source = str(prior_text or "")
    selected = str(selected_text or "").strip()
    if not source.strip() or not selected or source.rstrip().endswith("?"):
        return None
    sentence_spans = _sentence_character_spans(source)
    setup_left = sentence_spans[-1][0] if sentence_spans else 0
    setup = source[setup_left:].strip()
    joined = f"{setup} {selected}"
    lexical_predicate_split = bool(
        not re.search(r"[.!?][\"')\]]*\s*$", source)
        and (
            len(_toks(setup)) >= 3
            or _TRUSTED_SPLIT_SUBJECT_FRAGMENT_RE.fullmatch(setup)
        )
        and _TRUSTED_SPLIT_FINITE_PREDICATE_RE.match(selected)
    )
    if not (
        _cue_has_weak_end(
            source,
            selected,
            ignore_caption_case=True,
        )
        or lexical_predicate_split
    ):
        return None
    if not (
        _opening_clause_is_standalone(joined)
        or _local_example_setup_is_complete(joined)
    ):
        return None
    quote = _exact_boundary_quote(source[setup_left:], want="start")
    return _quote_character_span(source, quote) if quote else None


def _trusted_subjectless_predicate_context(
    segments: list[dict],
    start_line: int,
    selected: str,
) -> tuple[int, tuple[int, int]] | None:
    """Ground a bare predicate in the nearest contiguous spoken equation."""
    if _TRUSTED_SUBJECTLESS_PREDICATE_OPENING_RE.match(selected) is None:
        return None
    reset_patterns = (
        _HARD_TOPIC_RESET_RE,
        _INDEPENDENT_UNIT_RESET_RE,
        _NAMED_UNIT_LABEL_RESET_RE,
        _NAMED_METHOD_CONTRAST_RESET_RE,
        _NEXT_DISTINCT_UNIT_RESET_RE,
        _FORWARD_TOPIC_TRANSITION_RE,
    )
    for candidate in range(start_line - 1, -1, -1):
        try:
            gap = (
                float(segments[candidate + 1].get("start", 0.0))
                - float(segments[candidate].get("end", 0.0))
            )
        except (TypeError, ValueError, OverflowError):
            break
        if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
            break
        candidate_text = str(segments[candidate].get("text") or "")
        following_text = str(segments[candidate + 1].get("text") or "")
        joined = f"{candidate_text} {following_text}"
        split = len(candidate_text) + 1
        if any(
            reset.start() < split < reset.end()
            for pattern in reset_patterns
            for reset in pattern.finditer(joined)
        ):
            break
        resets = [
            reset
            for pattern in reset_patterns
            for reset in pattern.finditer(candidate_text)
        ]
        if resets:
            break
        equations = list(
            _TRUSTED_SYMBOLIC_EQUATION_RE.finditer(candidate_text)
        )
        if not equations:
            continue
        equation = equations[-1]
        quote = _exact_boundary_quote(
            candidate_text[equation.start():],
            want="start",
        )
        span = _quote_character_span(candidate_text, quote) if quote else None
        if span is not None:
            return candidate, span
    return None


def _trusted_split_copula_context(
    segments: list[dict],
    start_line: int,
    selected: str,
) -> tuple[int, tuple[int, int]] | None:
    """Recover a subject and copula stranded at the prior caption edge."""
    if (
        start_line <= 0
        or _TRUSTED_SPLIT_COPULA_COMPLEMENT_OPENING_RE.match(selected) is None
    ):
        return None
    try:
        gap = (
            float(segments[start_line].get("start", 0.0))
            - float(segments[start_line - 1].get("end", 0.0))
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
        return None
    prior_text = str(segments[start_line - 1].get("text") or "")
    setup = _TRUSTED_TRAILING_SUBJECT_COPULA_RE.search(prior_text)
    if setup is None:
        return None
    return start_line - 1, setup.span("setup")


def _trusted_start_context_repair(
    segments: list[dict],
    start_line: int,
    start_span: tuple[int, int] | None,
    *,
    force_clipped_start: bool = False,
    min_start_line: int = 0,
) -> tuple[int, tuple[int, int] | None, list[str]]:
    """Expand a trusted Gemini start to spoken context, but never reject it."""
    text = str(segments[start_line].get("text") or "")
    selected = (
        text[start_span[0]:].strip()
        if start_span is not None
        else text.strip()
    )
    # A contextual opening (for example, a pronoun or conjunction) is
    # structurally dependent on preceding speech regardless of its subject
    # matter.  Conservatively widen to the preceding spoken unit; do not try
    # to infer the antecedent from a domain-specific noun list.
    if _TRUSTED_CONTEXTUAL_START_RE.match(selected):
        joined_repair = _trusted_joined_start_context_repair(
            segments,
            start_line,
            start_span,
            force_clipped_start=force_clipped_start,
            min_start_line=min_start_line,
        )
        if (joined_repair[0], joined_repair[1]) != (start_line, start_span):
            return (
                joined_repair[0],
                joined_repair[1],
                ["expanded_start_context", *joined_repair[2]],
            )
    cue_start_is_clipped = bool(force_clipped_start)
    has_same_cue_prefix = bool(
        start_span is not None
        and _WORD_RE.search(text[:start_span[0]]) is not None
    )
    split_copula_context = (
        _trusted_split_copula_context(
            segments,
            start_line,
            selected,
        )
        if not has_same_cue_prefix
        else None
    )
    if split_copula_context is not None:
        repaired_line, repaired_span = split_copula_context
        return repaired_line, repaired_span, ["expanded_split_copula_context"]
    predicate_context = (
        _trusted_subjectless_predicate_context(segments, start_line, selected)
        if not has_same_cue_prefix
        else None
    )
    if predicate_context is not None:
        repaired_line, repaired_span = predicate_context
        return repaired_line, repaired_span, [
            "expanded_subjectless_predicate_context"
        ]
    projected_complete_setup = bool(
        start_span is not None
        and (
            (
                _TRUSTED_PROJECTED_SETUP_RE.match(selected)
                and _local_example_setup_is_complete(selected)
            )
            or _TRUSTED_JOINED_FRESH_DECLARATIVE_RE.match(selected)
        )
    )
    unsafe_projection = bool(
        (
            force_clipped_start
            or (
                start_span is not None
                and not _projected_start_is_standalone(text, start_span)
            )
        )
        and not projected_complete_setup
    )
    unresolved_opening = bool(
        _TRUSTED_SUBJECTLESS_PREDICATE_OPENING_RE.match(selected)
        or (
            start_span is None
            and not _opening_clause_is_standalone(selected)
        )
    )
    prefix = text[:start_span[0]] if start_span is not None else ""
    scenario_prefix = text[:start_span[1]] if start_span is not None else ""
    scenario_span = _trusted_scenario_start_span(scenario_prefix)
    scenario_overlaps_start = bool(
        scenario_span is not None
        and start_span is not None
        and scenario_span[1] > start_span[0]
    )
    if scenario_span is not None and (
        unsafe_projection or unresolved_opening or scenario_overlaps_start
    ):
        return start_line, scenario_span, ["expanded_projected_start_context"]
    if not unsafe_projection and not unresolved_opening:
        return start_line, start_span, []

    original_line = start_line
    original_span = start_span
    reset_span = (
        _trusted_hard_reset_start_span(text, before=start_span[0])
        if start_span is not None
        else None
    )
    if reset_span is not None:
        return start_line, reset_span, ["expanded_projected_start_context"]
    if (
        not cue_start_is_clipped
        and _opening_clause_is_standalone(text)
        and _TRUSTED_SUBJECTLESS_PREDICATE_OPENING_RE.match(selected) is None
        and _TRUSTED_CONTEXTUAL_CUE_OPENING_RE.match(text) is None
        and _trusted_opening_reference_is_resolved(selected, prefix)
    ):
        return start_line, None, ["expanded_projected_start_context"]

    prior_parts = [prefix] if prefix else []
    for candidate in range(start_line - 1, -1, -1):
        if candidate < max(0, min_start_line):
            break
        try:
            gap = (
                float(segments[candidate + 1].get("start", 0.0))
                - float(segments[candidate].get("end", 0.0))
            )
        except (TypeError, ValueError, OverflowError):
            break
        if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
            break
        candidate_text = str(segments[candidate].get("text") or "")
        prior_parts.insert(0, candidate_text)
        weak_prior_span = (
            _trusted_weak_prior_start_span(candidate_text, selected)
            if cue_start_is_clipped and candidate == start_line - 1
            else None
        )
        if weak_prior_span is not None:
            return candidate, weak_prior_span, [
                "expanded_split_cue_start_context"
            ]
        scenario_span = _trusted_scenario_start_span(candidate_text)
        if scenario_span is not None:
            return candidate, scenario_span, ["expanded_start_context"]
        reset_span = _trusted_hard_reset_start_span(candidate_text)
        if reset_span is not None:
            return candidate, reset_span, ["expanded_start_context"]
        if (
            _opening_clause_is_standalone(candidate_text)
            and _TRUSTED_CONTEXTUAL_CUE_OPENING_RE.match(candidate_text) is None
            and _trusted_opening_reference_is_resolved(
                selected,
                " ".join(prior_parts),
            )
        ):
            speaker_setup_span = _trusted_speaker_setup_start_span(
                candidate_text
            )
            if speaker_setup_span is not None:
                retained_prior = " ".join([
                    candidate_text[speaker_setup_span[0]:],
                    *prior_parts[1:],
                ])
                if _trusted_opening_reference_is_resolved(
                    selected,
                    retained_prior,
                ):
                    return candidate, speaker_setup_span, [
                        "expanded_start_context",
                        "trimmed_speaker_framing_prefix",
                    ]
            return candidate, None, ["expanded_start_context"]

    return original_line, original_span, ["unresolved_start_context"]


_TRUSTED_CONTEXTUAL_START_RE = re.compile(
    r"^\s*(?:and|because|but|by|from|he|her|his|it|its|of|she|so|that|"
    r"their|then|there|therefore|these|they|this|those|thus|to|we|when|"
    r"where|which|while|who|whose|with)\b",
    re.IGNORECASE,
)


def _trusted_joined_start_context_repair(
    segments: list[dict],
    start_line: int,
    start_span: tuple[int, int] | None,
    *,
    force_clipped_start: bool = False,
    min_start_line: int = 0,
) -> tuple[int, tuple[int, int] | None, list[str]]:
    """Close an unfinished spoken unit without using topic vocabulary.

    Caption cues are only character-to-time containers.  The repair joins all
    earlier transcript text, finds the nearest unambiguous sentence boundary,
    and maps that character back to its source cue.  Ambiguity widens the
    start; it never authorizes a later cut.
    """
    del min_start_line
    if not (0 <= start_line < len(segments)):
        return start_line, start_span, []
    source = str(segments[start_line].get("text") or "")
    start_left = start_span[0] if start_span is not None else 0
    if not (0 <= start_left <= len(source)):
        return start_line, start_span, []

    selected = source[start_left:].lstrip()
    same_cue_prefix = source[:start_left]
    prefix_boundaries = [
        boundary
        for boundary in _trusted_joined_unit_boundaries(same_cue_prefix)
        if boundary.group(0)[0] in ".!?"
    ]
    prefix_floor = prefix_boundaries[-1].end() if prefix_boundaries else 0
    clipped_inside_cue = bool(
        _WORD_RE.search(same_cue_prefix[prefix_floor:])
    )

    previous = (
        str(segments[start_line - 1].get("text") or "").rstrip()
        if start_line > 0 and start_left == 0
        else ""
    )
    previous_boundaries = [
        boundary
        for boundary in _trusted_joined_unit_boundaries(previous)
        if boundary.group(0)[0] in ".!?"
    ]
    previous_is_closed = bool(
        previous_boundaries
        and previous_boundaries[-1].end() == len(previous)
    )
    contextual_opening = bool(_TRUSTED_CONTEXTUAL_START_RE.match(selected))
    lowercase_opening = bool(
        selected
        and selected[0].isalpha()
        and selected[0].islower()
    )
    include_previous_closed_unit = bool(
        previous.endswith("?") or contextual_opening
    )
    needs_context = bool(
        force_clipped_start
        or clipped_inside_cue
        or (
            start_line > 0
            and start_left == 0
            and (
                not previous_is_closed
                or include_previous_closed_unit
                or lowercase_opening
            )
        )
    )
    if not needs_context:
        return start_line, start_span, []

    pieces: list[str] = []
    line_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for line in range(0, start_line + 1):
        if pieces:
            pieces.append(" ")
            cursor += 1
        line_source = str(segments[line].get("text") or "")
        right = start_left if line == start_line else len(line_source)
        joined_left = cursor
        pieces.append(line_source[:right])
        cursor += right
        line_ranges.append((line, joined_left, cursor))
    joined = "".join(pieces)
    boundaries = [
        boundary
        for boundary in _trusted_joined_unit_boundaries(joined)
        if boundary.group(0)[0] in ".!?"
    ]
    candidate = boundaries[-1].end() if boundaries else 0
    if include_previous_closed_unit and boundaries:
        between = joined[boundaries[-1].end():]
        if _WORD_RE.search(between) is None:
            candidate = boundaries[-2].end() if len(boundaries) >= 2 else 0
    while candidate < len(joined) and joined[candidate].isspace():
        candidate += 1
    if candidate >= len(joined):
        return start_line, start_span, []

    mapped = next(
        (
            (line, candidate - joined_left)
            for line, joined_left, joined_right in line_ranges
            if joined_left <= candidate < joined_right
        ),
        None,
    )
    if mapped is None or mapped >= (start_line, start_left):
        return start_line, start_span, []
    line, source_left = mapped
    repaired_source = str(segments[line].get("text") or "")
    quote = _exact_boundary_quote(repaired_source[source_left:], want="start")
    relative_span = (
        _quote_character_span(repaired_source[source_left:], quote)
        if quote
        else None
    )
    if relative_span is None:
        return start_line, start_span, []
    repaired_span = (
        source_left + relative_span[0],
        source_left + relative_span[1],
    )
    return line, repaired_span, ["expanded_unfinished_spoken_unit"]


_TRUSTED_CLAIM_SETUP_HANDOFF_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?P<teaching>with\s+that\s+knowledge\s+in\s+hand\b"
    r"[^.!?]{0,60}?(?P<teaching_setup>you(?:['’]re|\s+are)\s+"
    r"(?:now\s+)?ready\s+to\s+understand\b))|"
    r"(?P<conditional>(?:(?:and|but|so)\s*[,;:]?\s+)?"
    r"(?:now\s*[,;:]?\s+)?let(?:['’]?s|\s+us)\s+say\s+if\b)|"
    r"(?P<question>now\s*[,;:]?\s+"
    r"(?:how|what|when|where|which|who|why)\b)"
    r")",
    re.IGNORECASE,
)
_TRUSTED_CLAIM_SENTENCE_ONSET_RE = re.compile(
    r"(?<!\w)(?:(?P<ordinal>"
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
    r")\s+(?:it|this|that))\s+"
    r"(?:means?|shows?|implies?|indicates?|suggests?)\b|"
    r"(?P<note>it\s+is)\s+"
    r"(?:important|critical|crucial|essential)\s+to\s+"
    r"(?:note|notice|remember|observe|emphasize)\b)",
    re.IGNORECASE,
)
_TRUSTED_WORKED_TASK_SCOPE_RE = re.compile(
    r"\b(?:applications?|examples?|find(?:ing)?|determin(?:e|ing)|"
    r"evaluat(?:e|ing)|numerical\s+application)\b",
    re.IGNORECASE,
)
_TRUSTED_CLAIM_SENTENCE_ARC_SCOPE_RE = re.compile(
    r"\b(?:causal|chain|cycle|mechanism|pathway|process|sequence|stages?|steps?)\b",
    re.IGNORECASE,
)
_TRUSTED_RELATIONAL_CLAUSE_RE = re.compile(
    r"\b(?:(?:directly|inversely)\s+)?"
    r"(?:equal|equivalent|proportional|related|similar|connected|dependent)\s+"
    r"(?:to|on)\b",
    re.IGNORECASE,
)
_TRUSTED_NAMED_TEACHING_HANDOFF_RE = re.compile(
    r"(?<!\w)(?P<handoff>(?:now\s+)?the\s+next\s+"
    r"(?:concept|equation|idea|law|method|principle|relationship|rule|topic)\b)",
    re.IGNORECASE,
)
_TRUSTED_NAMED_UNIT_LINK_RE = re.compile(
    r"\b(?:is|are)\s+(?:(?:called|known)\s+(?:as\s+)?)?",
    re.IGNORECASE,
)
_TRUSTED_NAMED_UNIT_GENERIC_TOKENS = frozenset({
    "concept",
    "equation",
    "idea",
    "law",
    "method",
    "motion",
    "principle",
    "relationship",
    "rule",
    "topic",
})
_TRUSTED_WORKED_QUESTION_FRAME_RE = re.compile(
    r"(?<!\w)(?P<frame>here(?:['’]s|\s+is)\s+(?:a|the)\s+question"
    r"(?:\s+for\s+you)?)\b",
    re.IGNORECASE,
)
_TRUSTED_WORKED_CONTINUATION_RE = re.compile(
    r"^[\s.!?,;:—-]*(?:(?:and|but|so)\s*[,;:]?\s+)?"
    r"(?:now\b|our\s+next\s+step\b)",
    re.IGNORECASE,
)
_TRUSTED_WORKED_RESULT_SIGNAL_RE = re.compile(
    r"\b(?:(?:the\s+)?(?:answer|result|solution)\s+(?:is|equals?)|"
    r"complet(?:e|ed|es|ing)\s+(?:the\s+)?(?:calculation|comparison|"
    r"derivation|example|exercise|problem|proof)|"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"-?\d+(?:\.\d+)?)\s+(?:kilograms?|meters?|newtons?|seconds?))\b",
    re.IGNORECASE,
)
_TRUSTED_SPLIT_FINITE_PREDICATE_RE = re.compile(
    r"^\s*(?:(?:am|are|can|could|did|do|does|had|has|have|is|may|might|"
    r"must|shall|should|was|were|will|would)\b|"
    r"[a-z][\w'\u2019-]*(?:ed|es|s)\b)",
    re.IGNORECASE,
)
_TRUSTED_SPLIT_SUBJECT_FRAGMENT_RE = re.compile(
    r"\s*(?:a|an|any|each|every|some|the|this|that|these|those)\s+"
    r"(?:(?!(?:about|at|by|for|from|in|of|on|to|under|with)\b)"
    r"[a-z][\w'\u2019-]*\s*){1,3}",
    re.IGNORECASE,
)
_TRUSTED_FRESH_EXAMPLE_RE = re.compile(
    r"(?<!\w)(?:another|one\s+more|the\s+next|a\s+(?:different|new))\s+"
    r"(?:(?:brief|concrete|quick|short|simple|worked)\s+)*"
    r"(?:calculation|case|derivation|example|exercise|problem|proof)\b",
    re.IGNORECASE,
)
_TRUSTED_FORWARD_WORKED_HANDOFF_RE = re.compile(
    r"(?<!\w)(?P<handoff>(?:now\s*[,;:]?\s+)?"
    r"let(?:['’]?s|\s+us)\s+work\s+(?:on|through)\s+"
    r"(?:this|that|the)\s+"
    r"(?:calculation|case|derivation|example|exercise|problem|proof))\b",
    re.IGNORECASE,
)
_TRUSTED_FORWARD_WORKED_COMPLETION_RE = re.compile(
    r"\bso\s+(?:as\s+you\s+can\s+(?:see|notice)|"
    r"this\s+(?:demonstrates|shows))\b",
    re.IGNORECASE,
)
_TRUSTED_FORWARD_WORKED_RESULT_CLAUSE_RE = re.compile(
    r"\b(?:(?:the\s+)?(?:answer|result|solution)\s+(?:is|equals?)|"
    r"(?:it|that|this)\s+(?:equals?|is|will\s+be|is\s+going\s+to\s+be)"
    r"\s+(?:(?:about|approximately|roughly)\s+)?"
    r"(?:zero|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|-?\d+(?:\.\d+)?)\s+"
    r"(?:kilograms?|meters?|newtons?|seconds?))\b",
    re.IGNORECASE,
)
_TRUSTED_CALCULATED_RESULT_RE = re.compile(
    r"\b(?:calculated|computed|obtained)\s+"
    r"[a-z][\w'’-]*(?:\s+[a-z][\w'’-]*){0,2}\s+"
    r"(?:is|equals?)\s+(?:about\s+|approximately\s+|roughly\s+)?"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"-?\d+(?:\.\d+)?)\s+"
    r"(?:kilograms?|meters?|newtons?|seconds?)\b",
    re.IGNORECASE,
)
_TRUSTED_FRESH_WORKED_SCENARIO_RE = re.compile(
    r"(?:^|[.!?]\s+|[,;:]\s+|\b(?:now|next)\s*[,;:]?\s+)\s*(?:"
    r"(?:for|in)\s+(?:(?:this|that|the)\s+)?"
    r"(?:(?:next|new|other)\s+)?"
    r"(?:block|body|box|cart|case|example|object|particle|problem|scenario|system)\b|"
    r"(?:a|an)\s+(?:(?:next|new|other)\s+)?"
    r"(?:block|body|box|cart|crate|object|particle|person|system)\b|"
    r"the\s+(?:next|new|other)\s+"
    r"(?:block|body|box|cart|crate|object|particle|person|system)\b|"
    r"(?:the|this|that)\s+"
    r"(?:block|body|box|cart|crate|object|particle|person|system)\b"
    r"[^.!?]{0,120}\b(?:has|have|weighs?|experiences?|receives?|"
    r"is\s+(?:acted\s+on|assigned|given))\b|"
    r"(?:block|body|box|cart|crate|object|particle|person|system)\s+"
    r"[a-z0-9]+\b[^.!?]{0,120}\b(?:has|have|weighs?|experiences?|"
    r"receives?|is\s+(?:acted\s+on|assigned|given))\b|"
    r"(?:consider|imagine|suppose|take)\s+(?:that\s+)?(?:a|an|the)\b|"
    r"let(?:['’]?s|\s+us)\s+(?:say|suppose|take)\b)",
    re.IGNORECASE,
)
_TRUSTED_WORKED_SCENARIO_ONSET_RE = re.compile(
    r"(?:^|[.!?]\s+|\b(?:now|next)\s*[,;:]?\s+)\s*"
    r"(?P<setup>(?:(?:now|so)\s*[,;:]?\s+)?(?:"
    r"(?:a|an|the|this|that)\s+(?:(?:new|next|other)\s+)?"
    r"(?:block|body|box|cart|crate|mass|object|particle|person|system)\b|"
    r"(?:for|in)\s+(?:(?:this|that|the)\s+)?"
    r"(?:(?:new|next|other)\s+)?"
    r"(?:block|body|box|cart|crate|object|particle|person|system)\b|"
    r"(?:assume|consider|imagine|suppose)\s+(?:that\s+)?"
    r"(?:a|an|the)\s+(?:block|body|box|cart|crate|mass|object|particle|person|system)\b))",
    re.IGNORECASE,
)
_TRUSTED_EXPLICIT_WORKED_SCENARIO_ONSET_RE = re.compile(
    r"^\s*(?:(?:now|so)\s*[,;:]?\s+)?(?:"
    r"(?:a|an|the|this|that)\s+(?:new|next|other)\s+"
    r"(?:block|body|box|cart|crate|mass|object|particle|person|system)\b|"
    r"(?:for|in)\s+(?:(?:this|that|the)\s+)?"
    r"(?:(?:new|next|other)\s+)?"
    r"(?:block|body|box|cart|crate|object|particle|person|system)\b|"
    r"(?:assume|consider|imagine|suppose)\s+(?:that\s+)?"
    r"(?:a|an|the)\s+(?:block|body|box|cart|crate|mass|object|particle|person|system)\b)",
    re.IGNORECASE,
)
_TRUSTED_GENERIC_LOCAL_SCENARIO_RE = re.compile(
    r"^\s*(?:(?:now|next)\s*[,;:]?\s+)?(?:"
    r"(?:for|in)\s+(?:a|an|the|this|that)\s+[a-z][\w'’-]*\b|"
    r"(?:assume|calculate|compute|consider|determine|evaluate|find|given|"
    r"imagine|solve|suppose)\b)",
    re.IGNORECASE,
)
_TRUSTED_COMPLETE_LOCAL_QUESTION_FRAME_RE = re.compile(
    r"^\s*(?:(?:now|next|so)\s*[,;:]?\s+|"
    r"(?:assume|consider|imagine|suppose)\s+(?:that\s+)?|"
    r"(?:for|in)\s+(?:this|the)\s+(?:case|example)\s*[,;:]?\s+)?(?:"
    r"(?:at|given|in|on|under|using|with)\s+"
    r"(?!(?:this|that|the\s+same)\s+"
    r"(?:case|example|point|problem)\b)[^.!?]{2,160}|"
    r"for\s+(?!(?:this|that|the\s+same)\s+"
    r"(?:case|example|problem)\b)[^.!?]{2,160}|"
    r"(?:if|when)\s+(?:(?:a|an|the|this|that|these|those)\s+)?"
    r"(?!(?:he|it|she|they|we|you)\b)[a-z][\w'’-]*"
    r"[^.!?]{0,100}\b(?:are|contains?|does?|equals?|has|have|is|receives?|uses?|"
    r"vanishes?|was|were|[a-z][\w'’-]*(?:ed|es|s))\b[^.!?]{0,100}|"
    r"(?:a|an|the|this|that|these|those)\s+"
    r"(?!(?:he|it|she|they|we|you)\b)[a-z][\w'’-]*"
    r"[^.!?]{0,100}\b(?:are|contains?|does?|equals?|has|have|is|receives?|uses?|"
    r"vanishes?|was|were|[a-z][\w'’-]*(?:ed|es|s))\b[^.!?]{0,100})"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_TRUSTED_LOCAL_FRAME_BACK_REFERENCE_RE = re.compile(
    r"\b(?:above|below|computed|earlier|identified|previous|previously|"
    r"that|these|this|those)\b",
    re.IGNORECASE,
)
_TRUSTED_LOCAL_FRAME_STRONG_BACK_REFERENCE_RE = re.compile(
    r"\b(?:above|below|computed|earlier|identified|previous|previously)\b",
    re.IGNORECASE,
)
_TRUSTED_LOCAL_FRAME_ATTRIBUTED_REFERENCE_RE = re.compile(
    r"\b(?:aforementioned|assumed|given|indicated|stated|supplied|specified)\s+"
    r"[a-z][\w'’-]*(?:\s+[a-z][\w'’-]*){0,2}\b",
    re.IGNORECASE,
)
_TRUSTED_LOCAL_FRAME_GENERIC_DEICTIC_REFERENCE_RE = re.compile(
    r"\b(?:this|that|these|those)\s+(?!(?:a|an|the)\b)"
    r"[a-z][\w'’-]*(?:\s+[a-z][\w'’-]*){0,2}\b",
    re.IGNORECASE,
)
_TRUSTED_LOCAL_FRAME_REFERENTIAL_ACTION_RE = re.compile(
    r"\b(?:this|that|these|those)\s+(?!(?:a|an|the)\b)"
    r"[a-z][\w'’-]*(?:\s+[a-z][\w'’-]*){0,2}\s+"
    r"(?:(?:has|have)\s+been|are|is|was|were)\s+"
    r"[a-z][\w'’-]*(?:ed|en)\b",
    re.IGNORECASE,
)
_TRUSTED_LOCAL_FRAME_NAMED_SUBJECT_PREDICATE_RE = re.compile(
    r"\b(?:a|an|the|this|that|these|those)\s+"
    r"(?!(?:same|previous|prior)\b)"
    r"[a-z][\w'’-]*(?:\s+[a-z][\w'’-]*){0,3}\s+"
    r"(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would|"
    r"[a-z][\w'’-]*(?:ed|es|s))\b",
    re.IGNORECASE,
)
_TRUSTED_LOCAL_FRAME_PRONOUN_PREDICATE_RE = re.compile(
    r"\b(?:he|it|she|they)\s+(?:am|are|can|could|did|do|does|had|has|"
    r"have|is|may|might|must|shall|should|was|were|will|would|"
    r"[a-z][\w'’-]*(?:ed|es|s))\b",
    re.IGNORECASE,
)
_TRUSTED_LOCAL_IF_WHEN_SUBJECT_RE = re.compile(
    r"^\s*(?:if|when)\s+(?:(?:a|an|the|this|that|these|those)\s+)?"
    r"(?!(?:he|it|she|they|we|you)\b)[a-z][\w'’-]*\b",
    re.IGNORECASE,
)
_TRUSTED_LOCAL_FRAME_STATE_RE = re.compile(
    r"\b(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would|[a-z][\w'’-]*(?:ed|es|s))\b",
    re.IGNORECASE,
)
_TRUSTED_GENERIC_PREMISE_SCENARIO_RE = re.compile(
    r"^\s*(?:(?:now|next)\s*[,;:]?\s+)?(?:"
    r"(?:for|in)\s+(?:a|an|the|this|that)\s+"
    r"(?!(?:earlier|first|old|previous|prior|quick|same)\b)"
    r"[a-z][\w'’-]*\b|"
    r"(?:assume|consider|given|imagine|suppose)\s+"
    r"(?:that\s+)?(?:a|an|the)\s+[a-z][\w'’-]*\b)",
    re.IGNORECASE,
)
_TRUSTED_GENERIC_LOCAL_DECLARATIVE_RE = re.compile(
    r"^\s*(?:a|an|the|this|that)\s+(?P<head>[a-z][\w'’-]*)\b"
    r"[^.!?]{0,100}\b(?:comprises?|consists?\s+of|contains?|has|have|"
    r"includes?|receives?|requires?|uses?)\b",
    re.IGNORECASE,
)
_TRUSTED_PHYSICAL_SCENARIO_HEADS = frozenset({
    "block", "body", "box", "cart", "crate", "mass", "object",
    "particle", "person", "system",
})
_TRUSTED_GENERIC_DECLARATIVE_NONHEADS = frozenset({
    "current", "earlier", "final", "first", "following", "last", "next",
    "old", "previous", "prior", "same", "second", "third",
})
_TRUSTED_CAUTION_DEICTIC_RE = re.compile(
    r"\b(?:(?:here|there)(?!\s+(?:is|are|was|were)\b)|"
    r"this|that|these|those)\b",
    re.IGNORECASE,
)
_TRUSTED_COMPARATIVE_BACK_REFERENCE_RE = re.compile(
    r"\b(?:also|again|accordingly|similarly|likewise|too|"
    r"(?:half|twice|three\s+times|four\s+times)\s+as\b|"
    r"(?:greater|higher|larger|less|lower|smaller)\b|"
    r"(?:more|less)\s+[a-z][\w'’-]*(?:\s+[a-z][\w'’-]*){0,2}\s+than\b|"
    r"as\s+well|correspondingly|follows?\s+suit|in\s+turn|"
    r"in\s+response(?!\s+to\b)|once\s+more|proportionally|respectively|"
    r"in\s+the\s+same\s+way|"
    r"(?:the\s+)?(?:other|former|latter|same)\b|"
    r"than\s+(?:before|earlier|previously|the\s+other))\b",
    re.IGNORECASE,
)
_TRUSTED_CONTEXTUAL_ANSWER_IMPERATIVE_RE = re.compile(
    r"^\s*(?:move|place|put|restore|return|set)\s+"
    r"(?:it|this|that|the\s+(?:body|mass|object|system))\b",
    re.IGNORECASE,
)
_TRUSTED_QUANTIFIED_FORWARD_CONTENT_RE = re.compile(
    r"\d|(?:=|\bequals?\b)|\b(?:centimeters?|degrees?|feet|grams?|hours?|"
    r"inches?|joules?|kilograms?|kilometers?|meters?|minutes?|newtons?|"
    r"ohms?|pounds?|seconds?|volts?|watts?)\b",
    re.IGNORECASE,
)
_TRUSTED_FORWARD_CAUTION_HANDOFF_RE = re.compile(
    r"^\s*(?:(?:and|but|so)\s*[,;:]?\s+)?(?:be\s+careful|"
    r"keep\s+in\s+mind|note|notice|remember)\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_RESULT_CLAUSE_RE = re.compile(
    r"\b(?:the\s+)?(?:answer|result|solution)\s+(?:is|equals?)\b",
    re.IGNORECASE,
)
_TRUSTED_CONTEXT_DEPENDENT_SCOPE_RE = re.compile(
    r"\b(?:because|caus(?:al|e|es|ed|ing)|compar(?:e|ed|ing|ison)|"
    r"contrast|versus|why)\b",
    re.IGNORECASE,
)
_TRUSTED_SPLIT_ANSWER_OPENING_RE = re.compile(
    r"^\s*(?:[a-z0-9][\w'’/-]*\s+){0,3}"
    r"[a-z0-9][\w'’/-]*\s*[?,.:;—-]",
    re.IGNORECASE,
)
_TRUSTED_INCOMPLETE_QUESTION_TAIL_RE = re.compile(
    r"\b(?:(?:how|what|which|who)\s+"
    r"(?:are|can|could|did|do|does|is|should|was|were|will|would)"
    r"(?:\s+be)?|what(?:['’]s))"
    r"(?:\s+(?:an?|her|his|its|my|our|the|their|this|that|your))?"
    r"(?:\s+[a-z][\w'’-]*){0,3}\s*$",
    re.IGNORECASE,
)
_TRUSTED_JOINED_QUESTION_ONSET_RE = re.compile(
    r"(?<!\w)(?:how|what|where|which|who|why)\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_PROMPT_ONSET_RE = re.compile(
    r"(?<!\w)(?P<wh>how|what|where|which|who|why)\b|"
    r"(?<!\w)(?P<yes_no>am|are|can|could|did|do|does|had|has|have|is|"
    r"may|might|must|shall|should|was|were|will|would)\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_QUESTION_AUX_RE = re.compile(
    r"^\s*(?:['\u2019]s\b|(?:am|are|can|could|did|do|does|had|has|have|is|"
    r"may|might|must|shall|should|was|were|will|would)\b)",
    re.IGNORECASE,
)
_TRUSTED_JOINED_UNIT_BOUNDARY_RE = re.compile(
    r"(?:[.!?]+(?=\s|$)|[;\u2014]+)"
)


def _trusted_joined_unit_boundaries(text: str) -> list[re.Match[str]]:
    """Return only unambiguous spoken-unit punctuation boundaries.

    A short token or a dotted token before a period may be an abbreviation.
    Boundary repair must widen across that ambiguity instead of cutting there.
    """
    boundaries: list[re.Match[str]] = []
    for boundary in _TRUSTED_JOINED_UNIT_BOUNDARY_RE.finditer(text):
        mark = boundary.group(0)
        if mark.startswith("."):
            prefix = text[:boundary.start()]
            token_match = re.search(r"([^\s]+)$", prefix)
            token = token_match.group(1).strip("\"'()[]{}") if token_match else ""
            letters = "".join(character for character in token if character.isalpha())
            following = re.search(r"\s+([A-Za-z][\w'\u2019-]*)", text[boundary.end():])
            following_word = following.group(1) if following is not None else ""
            single_symbol_clause_end = bool(
                len(letters) == 1
                and re.search(
                    r"\b(?:be|equal(?:s|led)?|is|then)\s+"
                    + re.escape(token)
                    + r"$",
                    prefix,
                    re.IGNORECASE,
                )
            )
            title_like_abbreviation = bool(
                letters
                and len(letters) <= 5
                and letters[0].isupper()
                and following_word[:1].isupper()
                and not single_symbol_clause_end
                and following_word.casefold()
                not in {
                    "a", "an", "and", "assume", "but", "consider", "for",
                    "how", "however", "if", "imagine", "in", "it", "let",
                    "next", "now", "so", "suppose", "the", "then",
                    "therefore", "this", "that", "what", "when", "where",
                    "which", "who", "why", "with",
                }
            )
            numeric_abbreviation = bool(
                letters
                and len(letters) <= 5
                and re.match(r"\s+\d", text[boundary.end():])
            )
            if (
                "." in token
                and any(character.isalpha() for character in token)
            ) or title_like_abbreviation or numeric_abbreviation:
                continue
        boundaries.append(boundary)
    return boundaries


_TRUSTED_JOINED_COMPLETION_RE = re.compile(
    r"^\s*(?P<completion>[^\n.!?;:\u2014,]*?\b[^\n.!?;:\u2014,]*?)"
    r"\s*(?P<mark>[?,.:;\u2014])",
    re.IGNORECASE,
)
_TRUSTED_JOINED_FRESH_SCENARIO_RE = re.compile(
    r"\b(?:now|next|instead|turning\s+to)\s*[,;:]?\s+"
    r"(?P<onset>(?:a|an|the|this|that)\s+[a-z][\w'\u2019-]*)\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_FRESH_SETUP_RE = re.compile(
    r"^\s*(?:(?:now|so)\s*[,;:]?\s+)?(?:assume|consider|imagine|"
    r"suppose|let(?:'s|\s+us)\s+(?:say|suppose))\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_GENERIC_QUANTITY_RE = re.compile(
    r"\d|(?:=|\bequals?\b)|\b(?:zero|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    r"sixty|seventy|eighty|ninety|hundred|thousand|million|billion|"
    r"trillion)\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_GENERIC_DECLARATIVE_RE = re.compile(
    r"^\s*(?:(?:and|but|consequently|hence|so|therefore|thus)\s*[,;:]?\s+)?"
    r"(?:(?:if|when)\s+)?"
    r"(?:a|an|any|each|every|some|the|this|that|these|those)\s+"
    r"(?:[a-z0-9][\w'\u2019-]*\s+){1,7}?"
    r"(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would|[a-z][\w'\u2019-]*(?:ed|es|s))\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_FINITE_CLAUSE_RE = re.compile(
    r"^\s*(?:(?:and|but|consequently|hence|so|therefore|thus)\s*[,;:]?\s+)?"
    r"(?:(?:if|when)\s+)?(?:a|an|any|each|every|some|the|this|that|"
    r"these|those)?\s*(?:[a-z0-9][\w'\u2019-]*\s+){1,8}?"
    r"(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would|[a-z][\w'\u2019-]*(?:ed|es|s))\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_FRESH_DECLARATIVE_RE = re.compile(
    r"^\s*(?:(?:now|next)\s*[,;:]?\s+)?"
    r"(?:a|an|any|each|every|some)\s+"
    r"(?:[a-z0-9][\w'\u2019-]*\s+){1,7}?"
    r"(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would|[a-z][\w'\u2019-]*(?:ed|es|s))\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_ARTICLE_DECLARATIVE_ONSET_RE = re.compile(
    r"(?<!\w)(?P<onset>"
    r"(?:a|an|any|each|every|some|the|this|that|these|those)\s+"
    r"(?:[a-z0-9][\w'\u2019-]*\s+){1,7}?"
    r"(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would|[a-z][\w'\u2019-]*(?:ed|es|s))\b)",
    re.IGNORECASE,
)
_TRUSTED_JOINED_CONTINUATION_RE = re.compile(
    r"^\s*(?:also|and|but|consequently|furthermore|hence|it|likewise|"
    r"let(?:'s|\s+us)|moreover|next|so|then|therefore|they|thus)\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_LEADING_DEFINITE_RE = re.compile(
    r"^\s*(?:the|this|that|these|those)\s+(?P<head>[a-z][\w'\u2019-]*)\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_DEFINITE_REFERENCE_RE = re.compile(
    r"\b(?:the|this|that|these|those)\s+(?P<head>[a-z][\w'\u2019-]*)\b",
    re.IGNORECASE,
)
_TRUSTED_PARTICIPIAL_DEFINITE_REFERENCE_RE = re.compile(
    r"\bthe\s+(?:[a-z][\w'\u2019-]*(?:ed|en))\s+"
    r"[a-z][\w'\u2019-]*\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_NAMED_REFERENCE_RE = re.compile(
    r"\b(?:its|their)\b|"
    r"\b(?:compared|connected|relative)\s+to\s+(?:the|this|that)\b",
    re.IGNORECASE,
)
_TRUSTED_JOINED_META_BARRIER_RE = re.compile(
    r"\b(?:administrative\s+(?:announcement|note)|channel|random\s+aside|"
    r"sponsor(?:ed|ship)?|thanks?\s+for\s+watching)\b",
    re.IGNORECASE,
)
_TRUSTED_SPLIT_GROUNDING_GENERIC_TOKENS = frozenset({
    "answer", "case", "example", "explanation", "final", "problem",
    "question", "result", "solution", "value", "work", "worked",
})


def _trusted_named_teaching_handoff_start(
    segments: list[dict],
    *,
    search_start_line: int,
    claim_location: tuple[int, int, int, int],
    anchor_text: str,
) -> tuple[int, tuple[int, int]] | None:
    """Find an explicit named-unit introduction before the grounded claim."""
    claim_line, claim_left, claim_end_line, _claim_right = claim_location
    if not (
        0 <= search_start_line <= claim_line <= claim_end_line < len(segments)
    ):
        return None
    parts: list[str] = []
    line_offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for line in range(search_start_line, claim_end_line + 1):
        if line > search_start_line:
            try:
                gap = (
                    float(segments[line].get("start", 0.0))
                    - float(segments[line - 1].get("end", 0.0))
                )
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
                return None
            parts.append(" ")
            cursor += 1
        source = str(segments[line].get("text") or "")
        line_offsets[line] = (cursor, cursor + len(source))
        parts.append(source)
        cursor += len(source)
    joined = "".join(parts)
    claim_start = line_offsets[claim_line][0] + claim_left
    anchor_tokens = _content_tokens(anchor_text)
    sentence_spans = _sentence_character_spans(joined)
    candidates: list[tuple[int, tuple[int, int], int]] = []
    for handoff in _TRUSTED_NAMED_TEACHING_HANDOFF_RE.finditer(
        joined,
        0,
        claim_start,
    ):
        onset = handoff.start("handoff")
        sentence_span = next(
            (
                span
                for span in sentence_spans
                if span[0] <= onset < span[1]
            ),
            None,
        )
        if sentence_span is None:
            continue
        if _WORD_RE.search(joined[sentence_span[0]:onset]) is not None:
            continue
        sentence = joined[sentence_span[0]:sentence_span[1]]
        subject_links = list(_TRUSTED_NAMED_UNIT_LINK_RE.finditer(
            sentence,
            handoff.end("handoff") - sentence_span[0],
        ))
        if len(subject_links) != 1:
            continue
        named_subject = sentence[subject_links[0].end():]
        named_tokens = (
            _content_tokens(named_subject)
            - _TRUSTED_NAMED_UNIT_GENERIC_TOKENS
        )
        if len(named_tokens & anchor_tokens) < 2:
            continue
        onset_line = next(
            (
                line
                for line, (left, right) in line_offsets.items()
                if left <= onset < right
            ),
            None,
        )
        if onset_line is None:
            continue
        source = str(segments[onset_line].get("text") or "")
        source_left = onset - line_offsets[onset_line][0]
        quote = _exact_boundary_quote(source[source_left:], want="start")
        span = _quote_character_span(source, quote) if quote else None
        if span is not None and span[0] == source_left:
            candidates.append((onset_line, span, onset))
    if not candidates:
        return None
    line, span, _onset = max(candidates, key=lambda item: item[2])
    return line, span


def _trusted_prior_worked_question_start(
    segments: list[dict],
    *,
    selected_line: int,
    scope_text: str,
) -> tuple[int, tuple[int, int]] | None:
    """Recover one adjacent cue containing a framed worked-problem setup."""
    if (
        selected_line <= 0
        or (
            _ATOMIC_WORKED_SCOPE_RE.search(scope_text) is None
            and _TRUSTED_WORKED_TASK_SCOPE_RE.search(scope_text) is None
        )
        or not _cue_opens_mid_thought_at(
            segments,
            selected_line,
            ignore_caption_case=True,
        )
    ):
        return None
    prior_line = selected_line - 1
    try:
        gap = (
            float(segments[selected_line].get("start", 0.0))
            - float(segments[prior_line].get("end", 0.0))
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
        return None
    prior = str(segments[prior_line].get("text") or "")
    selected = str(segments[selected_line].get("text") or "")
    if not _cue_has_weak_end(prior, selected, ignore_caption_case=True):
        return None
    scope_tokens = _content_tokens(scope_text)
    frames = list(_TRUSTED_WORKED_QUESTION_FRAME_RE.finditer(prior))
    for index in range(len(frames) - 1, -1, -1):
        frame = frames[index]
        setup_right = (
            frames[index + 1].start("frame")
            if index + 1 < len(frames)
            else len(prior)
        )
        setup = prior[frame.start("frame"):setup_right]
        retained_question = prior[frame.start("frame"):]
        completed_problem = bool(
            "?" in setup
            or (
                _WORKED_UNIT_QUESTION_TOKEN_RE.search(setup)
                and _SPLIT_CAPTION_COMPLETION_SIGNAL_RE.search(setup)
            )
        )
        if setup_right < len(prior) and completed_problem:
            # Do not cross a completed earlier problem merely because a later
            # restated question supplies another question mark.
            continue
        conditional = next(
            (
                handoff
                for handoff in _TRUSTED_CLAIM_SETUP_HANDOFF_RE.finditer(
                    setup,
                    frame.end("frame") - frame.start("frame"),
                )
                if handoff.group("conditional") is not None
            ),
            None,
        )
        if (
            conditional is None
            or "?" not in retained_question
            or _WORKED_UNIT_QUESTION_TOKEN_RE.search(retained_question) is None
            or len(_content_tokens(setup) & scope_tokens) < 2
        ):
            continue
        return prior_line, frame.span("frame")
    return None


def _trusted_split_model_start_context(
    segments: list[dict],
    *,
    selected_line: int,
    selected_span: tuple[int, int] | None,
) -> tuple[int, tuple[int, int]] | None:
    """Recover a fresh scenario whose opening sentence is split across cues."""
    if selected_line <= 0 or selected_span is None:
        return None
    selected = str(segments[selected_line].get("text") or "")
    first_word = _WORD_RE.search(selected)
    if first_word is None or selected_span[0] != first_word.start():
        return None
    prior_line = selected_line - 1
    try:
        gap = (
            float(segments[selected_line].get("start", 0.0))
            - float(segments[prior_line].get("end", 0.0))
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
        return None
    prior = str(segments[prior_line].get("text") or "")
    if _TRUSTED_WORKED_QUESTION_FRAME_RE.search(prior) is not None:
        return None
    conditional = next(
        (
            handoff
            for handoff in reversed(list(
                _TRUSTED_CLAIM_SETUP_HANDOFF_RE.finditer(prior)
            ))
            if handoff.group("conditional") is not None
        ),
        None,
    )
    if (
        conditional is not None
        and re.search(r"[.!?]", prior[conditional.start():]) is not None
    ):
        return None
    scenario_quote = (
        _exact_boundary_quote(prior[conditional.start():], want="start")
        if conditional is not None
        else ""
    )
    scenario_span = (
        _quote_character_span(prior, scenario_quote)
        if scenario_quote
        else None
    )
    selected_suffix = selected[selected_span[0]:]
    if (
        scenario_span is None
        or not _cue_has_weak_end(
            prior,
            selected_suffix,
            ignore_caption_case=True,
        )
    ):
        return None
    joined = f"{prior[scenario_span[0]:]} {selected_suffix}"
    sentence_spans = _sentence_character_spans(joined)
    opening = joined[:sentence_spans[0][1]] if sentence_spans else joined
    if not _local_example_setup_is_complete(opening):
        return None
    return prior_line, scenario_span


def _trusted_explicit_definition_start(
    segments: list[dict],
    *,
    selected_line: int,
    selected_left: int,
    claim_location: tuple[int, int, int, int],
    claim_quote: str,
    scope_text: str,
) -> tuple[int, tuple[int, int]] | None:
    """Skip completed sibling definitions when the target definition is explicit."""
    claim_line, claim_left, claim_end_line, claim_right = claim_location
    if (
        claim_line != claim_end_line
        or claim_line < selected_line
        or _TRUSTED_DEFINITION_SCOPE_RE.search(scope_text) is None
    ):
        return None
    source = str(segments[claim_line].get("text") or "")
    sentence_span = next(
        (
            span
            for span in _sentence_character_spans(source)
            if span[0] <= claim_left < claim_right <= span[1]
        ),
        None,
    )
    if sentence_span is None:
        return None
    first_word = _WORD_RE.search(source, sentence_span[0], sentence_span[1])
    if first_word is None:
        return None
    sentence = source[first_word.start():sentence_span[1]]
    sentence_definition = _TRUSTED_EXPLICIT_DEFINITION_RE.match(sentence)
    claim_definition = _TRUSTED_EXPLICIT_DEFINITION_RE.match(claim_quote)
    if sentence_definition is None or claim_definition is None:
        return None
    definition_clause = sentence.split(",", 1)[0]
    sentence_subject = _content_tokens(sentence_definition.group("subject"))
    claim_subject = _content_tokens(claim_definition.group("subject"))
    if (
        not sentence_subject
        or sentence_subject != claim_subject
        or not sentence_subject.issubset(_content_tokens(scope_text))
        or _opening_has_unresolved_setup_reference(definition_clause)
    ):
        return None

    prefix_parts: list[str] = []
    for line in range(selected_line, claim_line + 1):
        text = str(segments[line].get("text") or "")
        left = selected_left if line == selected_line else 0
        right = first_word.start() if line == claim_line else len(text)
        if right > left:
            prefix_parts.append(text[left:right])
    prefix = " ".join(prefix_parts)
    if len(_toks(prefix)) < 6 or re.search(r"[.!?]", prefix) is None:
        return None
    sibling_subjects: set[frozenset[str]] = set()
    for left, right in _sentence_character_spans(prefix):
        prior_word = _WORD_RE.search(prefix, left, right)
        if prior_word is None:
            continue
        prior_definition = _TRUSTED_EXPLICIT_DEFINITION_RE.match(
            prefix[prior_word.start():right]
        )
        if prior_definition is None:
            continue
        prior_subject = frozenset(_content_tokens(
            prior_definition.group("subject")
        ))
        if prior_subject and prior_subject != frozenset(claim_subject):
            sibling_subjects.add(prior_subject)
    if len(sibling_subjects) < 2:
        return None
    quote = _exact_boundary_quote(source[first_word.start():], want="start")
    span = _quote_character_span(source, quote) if quote else None
    return (claim_line, span) if span is not None else None


def _trusted_same_cue_sentence_start(
    segments: list[dict],
    *,
    selected_line: int,
    selected_left: int,
    claim_location: tuple[int, int, int, int],
    scope_text: str,
) -> tuple[int, tuple[int, int]] | None:
    """Drop a dangling rolling-caption prefix before a grounded sentence."""
    claim_line, claim_left, claim_end_line, _claim_right = claim_location
    if (
        not (0 <= selected_line <= claim_line <= claim_end_line < len(segments))
        or _ATOMIC_WORKED_SCOPE_RE.search(scope_text) is not None
        or _TRUSTED_WORKED_TASK_SCOPE_RE.search(scope_text) is not None
        or _ATOMIC_CAUSAL_SCOPE_RE.search(scope_text) is not None
        or _TRUSTED_CLAIM_SENTENCE_ARC_SCOPE_RE.search(scope_text) is not None
        or _EXPLICIT_COMPARISON_OBJECTIVE_RE.search(scope_text) is not None
        or not _cue_opens_mid_thought_at(
            segments,
            selected_line,
            ignore_caption_case=True,
        )
    ):
        return None
    source = str(segments[selected_line].get("text") or "")
    if re.match(
        r"^\s*(?:the\s+)?answer\s+(?:is|equals?)\b",
        source[selected_left:],
        re.IGNORECASE,
    ):
        return None
    for sentence_left, _sentence_right in _sentence_character_spans(source):
        first_word = _WORD_RE.search(source, sentence_left)
        if first_word is None or first_word.start() <= selected_left:
            continue
        if selected_line == claim_line and first_word.start() > claim_left:
            continue
        prefix = source[selected_left:first_word.start()]
        if re.search(r"[.!?]", prefix) is None:
            continue
        joined_parts = [source[first_word.start():]]
        contiguous = True
        for line in range(selected_line + 1, claim_end_line + 1):
            try:
                gap = (
                    float(segments[line].get("start", 0.0))
                    - float(segments[line - 1].get("end", 0.0))
                )
            except (TypeError, ValueError, OverflowError):
                contiguous = False
                break
            if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
                contiguous = False
                break
            joined_parts.append(str(segments[line].get("text") or ""))
        if not contiguous:
            continue
        retained = " ".join(joined_parts)
        retained_sentences = _sentence_character_spans(retained)
        opening = (
            retained[:retained_sentences[0][1]]
            if retained_sentences
            else retained
        )
        if (
            not _opening_clause_is_standalone(opening)
            or _opening_has_unresolved_setup_reference(opening)
            or not (
                _content_tokens(opening) & _content_tokens(scope_text)
            )
        ):
            continue
        quote = _exact_boundary_quote(source[first_word.start():], want="start")
        span = _quote_character_span(source, quote) if quote else None
        if span is not None and span[0] == first_word.start():
            return selected_line, span
    return None


def _trusted_projected_worked_arc_end(
    segments: list[dict],
    *,
    end_line: int,
    end_span: tuple[int, int] | None,
    scope_text: str,
) -> tuple[int, tuple[int, int], str] | None:
    """Finish an explicit same-problem continuation before a fresh unit."""
    if (
        end_span is None
        or not (0 <= end_line < len(segments))
        or (
            _ATOMIC_WORKED_SCOPE_RE.search(scope_text) is None
            and _TRUSTED_WORKED_TASK_SCOPE_RE.search(scope_text) is None
        )
    ):
        return None
    end_text = str(segments[end_line].get("text") or "")
    suffix = end_text[end_span[1]:]
    if (
        _WORD_RE.search(suffix) is None
        or _TRUSTED_WORKED_CONTINUATION_RE.match(suffix) is None
    ):
        return None
    sentence_spans = _sentence_character_spans(suffix)
    continuation_right = (
        sentence_spans[min(1, len(sentence_spans) - 1)][1]
        if sentence_spans
        else len(suffix)
    )
    if (
        len(
            _content_tokens(suffix[:continuation_right])
            & _content_tokens(scope_text)
        )
        < 2
    ):
        return None

    scope_tokens = _content_tokens(scope_text)
    following_text = (
        str(segments[end_line + 1].get("text") or "")
        if end_line + 1 < len(segments)
        else ""
    )
    following_spans = _sentence_character_spans(following_text)
    following_opening = (
        following_text[:following_spans[0][1]]
        if following_spans
        else following_text
    )
    terminal_split_continues = bool(
        following_text
        and _cue_has_weak_end(
            end_text,
            following_text,
            ignore_caption_case=True,
        )
    )
    for index, (left, right) in enumerate(sentence_spans):
        piece = suffix[left:right]
        if (
            _WORD_RE.search(piece)
            and not (_content_tokens(piece) & scope_tokens)
            and not (
                index == len(sentence_spans) - 1
                and terminal_split_continues
                and len(
                    _content_tokens(f"{piece} {following_opening}")
                    & scope_tokens
                ) >= 2
            )
        ):
            return None

    def same_arc_piece(line: int, piece: str) -> bool:
        if _WORD_RE.search(piece) is None:
            return True
        if line <= end_line:
            return False
        previous = str(segments[line - 1].get("text") or "")
        current = str(segments[line].get("text") or "")
        scope_overlap = _content_tokens(piece) & scope_tokens
        independent_worked_result = bool(
            len(scope_overlap) >= 2
            and _TRUSTED_WORKED_RESULT_SIGNAL_RE.search(piece)
        )
        split_named_subject = bool(
            re.search(
                r"\b(?:[Aa]nd|[Bb]ut|[Ss]o)\s+(?:block\s+)?[A-Z]\s*$",
                previous,
            )
            and re.match(r"^\s*[a-z][\w'’\-]*\b", piece)
        )
        return bool(
            (
                scope_overlap
                and (
                    _cue_has_weak_end(
                        previous,
                        current,
                        ignore_caption_case=True,
                    )
                    or split_named_subject
                )
            )
            or (
                len(_sentence_character_spans(piece)) == 1
                and re.search(
                    r"\b(?:connected|dependent|equal|equivalent|opposite|"
                    r"proportional|related|similar)\s*$",
                    previous,
                    re.IGNORECASE,
                )
                is not None
                and re.match(
                    r"^\s*(?:from|to|with)\b",
                    piece,
                    re.IGNORECASE,
                )
                is not None
            )
            or independent_worked_result
        )

    def completed_boundary(
        line: int,
        retained: str,
    ) -> tuple[int, tuple[int, int], str] | None:
        complete = _last_safe_complete_prefix(retained)
        if not complete:
            return None
        words = list(_WORD_RE.finditer(complete))
        if not words:
            return None
        quote_left = words[max(0, len(words) - 6)].start()
        quote = complete[quote_left:]
        return line, (quote_left, len(complete)), quote

    for line in range(end_line, len(segments)):
        if line > end_line:
            try:
                gap = (
                    float(segments[line].get("start", 0.0))
                    - float(segments[line - 1].get("end", 0.0))
                )
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
                return None
        source = str(segments[line].get("text") or "")
        search_left = end_span[1] if line == end_line else 0
        sentence_spans = _sentence_character_spans(source)
        hard_markers = list(_HARD_TOPIC_RESET_RE.finditer(source, search_left))
        if line == end_line:
            continuation = source[search_left:]
            continuation_spans = _sentence_character_spans(continuation)
            if (
                _TRUSTED_WORKED_CONTINUATION_RE.match(continuation)
                and continuation_spans
            ):
                continuation_right = search_left + continuation_spans[0][1]
                hard_markers = [
                    marker
                    for marker in hard_markers
                    if marker.start() >= continuation_right
                ]
        fresh_markers = [
            marker
            for marker in _TRUSTED_FRESH_EXAMPLE_RE.finditer(
                source,
                search_left,
            )
            if not any(
                span[0] <= marker.start() < span[1]
                and _WORD_RE.search(source[span[0]:marker.start()]) is not None
                for span in sentence_spans
            )
        ]
        marker_matches = [*hard_markers, *fresh_markers]
        if not marker_matches:
            if line > end_line and not same_arc_piece(line, source):
                return None
            continue
        marker = min(marker_matches, key=lambda match: match.start())
        marker_sentence = next(
            (
                span
                for span in sentence_spans
                if span[0] <= marker.start() < span[1]
            ),
            None,
        )
        marker_left = marker_sentence[0] if marker_sentence is not None else marker.start()
        if marker_left <= search_left:
            if line <= end_line:
                return None
            previous_line = line - 1
            previous = str(segments[previous_line].get("text") or "")
            return completed_boundary(previous_line, previous)
        retained = source[:marker_left]
        if line > end_line and not same_arc_piece(line, retained):
            return None
        return completed_boundary(line, retained)
    return None


def _trusted_relational_sentence_before_claim(
    segments: list[dict],
    *,
    selected_line: int,
    selected_left: int,
    claim_location: tuple[int, int, int, int],
) -> bool:
    """Confirm that a clipped opening finishes a relational prior sentence."""
    claim_line, claim_left, claim_end_line, claim_right = claim_location
    if not (
        0 <= selected_line <= claim_line <= claim_end_line < len(segments)
    ):
        return False
    for line in range(selected_line, claim_end_line):
        try:
            gap = (
                float(segments[line + 1].get("start", 0.0))
                - float(segments[line].get("end", 0.0))
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
            return False
    parts: list[str] = []
    for line in range(selected_line, claim_line + 1):
        source = str(segments[line].get("text") or "")
        left = selected_left if line == selected_line else 0
        right = claim_left if line == claim_line else len(source)
        if right > left:
            parts.append(source[left:right])
    prefix = " ".join(parts)
    sentence_spans = _sentence_character_spans(prefix)
    first_sentence = (
        prefix[:sentence_spans[0][1]] if sentence_spans else prefix
    )
    if _TRUSTED_RELATIONAL_CLAUSE_RE.search(first_sentence) is None:
        return False
    remainder = prefix[len(first_sentence):].lstrip()
    note_frame = _TRUSTED_CLAIM_SENTENCE_ONSET_RE.match(remainder)
    if note_frame is None or note_frame.group("note") is None:
        return False
    claim_parts: list[str] = []
    for line in range(claim_line, claim_end_line + 1):
        source = str(segments[line].get("text") or "")
        left = claim_left if line == claim_line else 0
        right = claim_right if line == claim_end_line else len(source)
        if right > left:
            claim_parts.append(source[left:right])
    note_body = " ".join((remainder[note_frame.end():], *claim_parts))
    return not _opening_has_unresolved_setup_reference(note_body)


def _trusted_claim_sentence_start(
    segments: list[dict],
    *,
    selected_line: int,
    claim_location: tuple[int, int, int, int],
    allowed_frame: str,
) -> tuple[int, tuple[int, int]] | None:
    """Recover a nearby explicit sentence frame that contains the claim."""
    claim_line, claim_left, claim_end_line, claim_right = claim_location
    search_start = max(0, selected_line - 1)
    if claim_line < search_start or claim_end_line >= len(segments):
        return None

    parts: list[str] = []
    line_offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for line in range(search_start, claim_end_line + 1):
        if parts:
            parts.append(" ")
            cursor += 1
        source = str(segments[line].get("text") or "")
        line_offsets[line] = (cursor, cursor + len(source))
        parts.append(source)
        cursor += len(source)
    joined = "".join(parts)
    claim_start = line_offsets[claim_line][0] + claim_left
    claim_end = line_offsets[claim_end_line][0] + claim_right

    sentence_spans = _sentence_character_spans(joined)
    candidates: list[tuple[int, tuple[int, int], int]] = []
    for frame in _TRUSTED_CLAIM_SENTENCE_ONSET_RE.finditer(
        joined,
        0,
        claim_start,
    ):
        group = "ordinal" if frame.group("ordinal") is not None else "note"
        if group != allowed_frame:
            continue
        onset_left, onset_right = frame.span(group)
        sentence_span = next(
            (
                span
                for span in sentence_spans
                if span[0] <= onset_left < span[1]
            ),
            None,
        )
        verified_sentence_boundary = bool(sentence_span and sentence_span[0] > 0)
        if (
            sentence_span is not None
            and sentence_span[0] == 0
            and search_start > 0
        ):
            from .sentences import classify_terminator

            verified_sentence_boundary = bool(classify_terminator(str(
                segments[search_start - 1].get("text") or ""
            )))
        elif search_start == 0:
            verified_sentence_boundary = True
        if (
            sentence_span is None
            or not verified_sentence_boundary
            or _WORD_RE.search(joined[sentence_span[0]:onset_left]) is not None
            or not (onset_right <= claim_start < claim_end <= sentence_span[1])
        ):
            continue

        onset_line = next(
            (
                line
                for line, (left, right) in line_offsets.items()
                if left <= onset_left < onset_right <= right
            ),
            None,
        )
        if onset_line is None:
            continue
        contiguous = True
        for line in range(onset_line, claim_end_line):
            try:
                gap = (
                    float(segments[line + 1].get("start", 0.0))
                    - float(segments[line].get("end", 0.0))
                )
            except (TypeError, ValueError, OverflowError):
                contiguous = False
                break
            if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
                contiguous = False
                break
        if not contiguous:
            continue
        line_left = line_offsets[onset_line][0]
        candidates.append((
            onset_line,
            (onset_left - line_left, onset_right - line_left),
            onset_left,
        ))
    if not candidates:
        return None
    line, span, _position = max(candidates, key=lambda item: item[2])
    return line, span


def _trusted_claim_setup_start(
    segments: list[dict],
    *,
    selected_line: int,
    selected_left: int,
    claim_location: tuple[int, int, int, int],
    anchor_text: str,
    teaching_handoff_only: bool = False,
    teaching_subject_anchor_text: str = "",
    allow_validated_single_subject_anchor: bool = False,
) -> tuple[int, tuple[int, int]] | None:
    """Advance a Gemini edge to a later explicit, complete setup for its claim."""
    claim_line, claim_left, _claim_end_line, _claim_right = claim_location
    anchor_tokens = _content_tokens(anchor_text)
    candidates: list[tuple[int, tuple[int, int]]] = []
    for line in range(selected_line, claim_line + 1):
        source = str(segments[line].get("text") or "")
        upper = claim_left if line == claim_line else len(source)
        for handoff in _TRUSTED_CLAIM_SETUP_HANDOFF_RE.finditer(source, 0, upper):
            teaching_subject_matches_claim = False
            if (
                teaching_handoff_only
                and handoff.group("teaching_setup") is None
            ):
                continue
            if (
                teaching_subject_anchor_text
                and handoff.group("teaching_setup") is not None
            ):
                subject_tail = source[handoff.end("teaching_setup"):]
                subject_phrase = re.split(
                    r"[,;:.!?]|\b(?:after|and|are|as|before|but|because|"
                    r"describes?|if|is|means?|once|refers?|so|that|then|"
                    r"when|where|whereas|which|while|who|why)\b",
                    subject_tail,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]
                generic_subject_tokens = {
                    "concept", "idea", "lesson", "material", "principle",
                    "subject", "thing", "topic",
                }
                subject_tokens = (
                    _content_tokens(subject_phrase) - generic_subject_tokens
                )
                claim_tokens = (
                    _content_tokens(teaching_subject_anchor_text)
                    - generic_subject_tokens
                )
                if (
                    not subject_tokens
                    or not subject_tokens.issubset(claim_tokens)
                ):
                    continue
                teaching_subject_matches_claim = True
            setup_left = (
                handoff.start("teaching_setup")
                if handoff.group("teaching_setup") is not None
                else handoff.start()
            )
            if (line, setup_left) <= (selected_line, selected_left):
                continue
            retained = source[setup_left:]
            sentence_spans = _sentence_character_spans(retained)
            opening_right = (
                sentence_spans[0][1] if sentence_spans else len(retained)
            )
            opening = retained[:opening_right]
            required_anchor_overlap = (
                1
                if (
                    allow_validated_single_subject_anchor
                    and teaching_subject_matches_claim
                )
                else 2
            )
            if (
                len(_content_tokens(opening) & anchor_tokens)
                < required_anchor_overlap
            ):
                continue
            standalone_opening = (
                _exact_boundary_quote(retained, want="start")
                if handoff.group("teaching_setup") is not None
                else opening
            )
            discourse_marker = _LEADING_DISCOURSE_MARKER_RE.match(
                standalone_opening
            )
            if discourse_marker is not None:
                standalone_opening = standalone_opening[
                    discourse_marker.end():
                ]
            standalone_opening = re.sub(
                r"^\s*now\s*[,;:]?\s+",
                "",
                standalone_opening,
                count=1,
                flags=re.IGNORECASE,
            )
            if (
                not _opening_clause_is_standalone(standalone_opening)
                or _opening_has_unresolved_setup_reference(
                    standalone_opening
                )
            ):
                continue
            if handoff.group("conditional") is not None:
                condition = source[handoff.end():setup_left + opening_right]
                condition_words = _toks(condition)
                if (
                    not condition_words
                    or condition_words[0] in {
                        "he", "her", "him", "his", "it", "its", "she",
                        "that", "their", "them", "these", "they", "this",
                        "those", "we", "you",
                    }
                    or _WORKED_UNIT_QUESTION_TOKEN_RE.search(condition) is None
                ):
                    continue
            quote = _exact_boundary_quote(retained, want="start")
            span = _quote_character_span(source, quote) if quote else None
            if span is not None:
                candidates.append((line, span))
    return max(
        candidates,
        default=None,
        key=lambda item: (item[0], item[1][0]),
    )


def _trusted_split_answer_scenario_start_legacy(
    segments: list[dict],
    *,
    selected_line: int,
    selected_left: int,
    scope_text: str,
) -> tuple[int, tuple[int, int]] | None:
    """Recover the concrete scenario before a caption-split generic answer."""
    selected_source = str(segments[selected_line].get("text") or "")
    selected = selected_source[selected_left:]
    question_line: int | None = None
    question_tail_left: int | None = None
    answer_line: int | None = None

    if _TRUSTED_INCOMPLETE_QUESTION_TAIL_RE.match(selected):
        answer_line = selected_line + 1
        next_text = str(segments[answer_line].get("text") or "") if (
            answer_line < len(segments)
        ) else ""
        if _TRUSTED_SPLIT_ANSWER_OPENING_RE.match(next_text):
            question_line = selected_line
            question_tail_left = selected_left
    elif (
        selected_line > 0
        and _TRUSTED_SPLIT_ANSWER_OPENING_RE.match(selected)
    ):
        answer_line = selected_line
        question_line = selected_line - 1
        question_source = str(segments[question_line].get("text") or "")
        tail = _TRUSTED_INCOMPLETE_QUESTION_TAIL_RE.search(question_source)
        if tail is not None:
            question_tail_left = tail.start()

    if (
        question_line is None
        or question_tail_left is None
        or answer_line is None
        or answer_line >= len(segments)
    ):
        return None
    try:
        answer_gap = (
            float(segments[answer_line].get("start", 0.0))
            - float(segments[question_line].get("end", 0.0))
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(answer_gap) or answer_gap >= _SECTION_RESET_GAP_S:
        return None

    def completion_floor(
        text: str,
        upper: int,
        *,
        protect_question_prefix: bool = False,
    ) -> tuple[bool, int]:
        local_question_frame = _TRUSTED_COMPLETE_LOCAL_QUESTION_FRAME_RE.match(
            text[:upper]
        )
        matches = []
        for pattern in (
            _SPLIT_CAPTION_COMPLETION_SIGNAL_RE,
            _WORKED_UNIT_CLOSING_TAIL_RE,
            _TRUSTED_CALCULATED_RESULT_RE,
        ):
            for match in pattern.finditer(text, 0, upper):
                if re.match(
                    r"\s*(?:never|not)\b",
                    text[match.end():upper],
                    re.IGNORECASE,
                ) is not None:
                    continue
                # In a local condition, words such as "this solution is
                # acidic" describe the current scenario; "solution is" is
                # not a completed prior answer.
                if (
                    pattern is _SPLIT_CAPTION_COMPLETION_SIGNAL_RE
                    and (
                        protect_question_prefix
                        or (
                            local_question_frame is not None
                            and match.end() <= local_question_frame.end()
                        )
                    )
                ):
                    continue
                matches.append(match)
        if not matches:
            return False, 0
        completion = max(matches, key=lambda match: match.end())
        terminator = re.search(r"[.!?]", text[completion.end():upper])
        fresh_scenario = _TRUSTED_WORKED_SCENARIO_ONSET_RE.search(
            text,
            completion.end(),
            upper,
        )
        if (
            fresh_scenario is not None
            and (
                terminator is None
                or fresh_scenario.start("setup")
                < completion.end() + terminator.end()
            )
        ):
            return True, fresh_scenario.start("setup")
        return (
            True,
            completion.end() + terminator.end()
            if terminator is not None
            else completion.end(),
        )

    question_source = str(segments[question_line].get("text") or "")
    found_completion, question_floor = completion_floor(
        question_source,
        question_tail_left,
        protect_question_prefix=True,
    )
    search_floors = {question_line: question_floor}
    contiguous_lines = [question_line]
    for line in range(question_line - 1, max(-1, question_line - 12), -1):
        if found_completion:
            break
        try:
            gap = (
                float(segments[line + 1].get("start", 0.0))
                - float(segments[line].get("end", 0.0))
            )
        except (TypeError, ValueError, OverflowError):
            break
        if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
            break
        contiguous_lines.append(line)
        source = str(segments[line].get("text") or "")
        found_completion, floor = completion_floor(source, len(source))
        search_floors[line] = floor

    explicit_candidates: list[tuple[int, int]] = []
    generic_premise_candidates: list[tuple[int, int]] = []
    generic_declarative_candidates: list[tuple[int, int, str]] = []
    scenario_candidates: list[tuple[int, int]] = []
    scenario_candidate_heads: dict[tuple[int, int], str] = {}
    for line in contiguous_lines:
        source = str(segments[line].get("text") or "")
        lower = search_floors.get(line, 0)
        upper = question_tail_left if line == question_line else len(source)
        local_fragment = source[lower:upper]
        if _TRUSTED_GENERIC_PREMISE_SCENARIO_RE.match(local_fragment):
            generic_premise_candidates.append((line, lower))
        generic_declarative = _TRUSTED_GENERIC_LOCAL_DECLARATIVE_RE.match(
            local_fragment
        )
        if (
            generic_declarative is not None
            and generic_declarative.group("head").casefold()
            not in _TRUSTED_PHYSICAL_SCENARIO_HEADS
            and generic_declarative.group("head").casefold()
            not in _TRUSTED_GENERIC_DECLARATIVE_NONHEADS
        ):
            generic_declarative_candidates.append((
                line,
                lower,
                generic_declarative.group("head").casefold(),
            ))
        explicit_candidates.extend(
            (line, handoff.start("conditional"))
            for handoff in _TRUSTED_CLAIM_SETUP_HANDOFF_RE.finditer(
                source,
                lower,
                upper,
            )
            if handoff.group("conditional") is not None
        )
        for onset in _TRUSTED_WORKED_SCENARIO_ONSET_RE.finditer(
            source,
            lower,
            upper,
        ):
            candidate = (line, onset.start("setup"))
            scenario_candidates.append(candidate)
            scenario_head = re.search(
                r"\b(?:block|body|box|cart|crate|mass|object|particle|"
                r"person|system)\b",
                source[onset.start("setup"):onset.end("setup")],
                re.IGNORECASE,
            )
            if scenario_head is not None:
                scenario_candidate_heads[candidate] = (
                    scenario_head.group(0).casefold()
                )
            if _TRUSTED_EXPLICIT_WORKED_SCENARIO_ONSET_RE.match(
                source[onset.start("setup"):]
            ):
                explicit_candidates.append(candidate)
    local_question_text = question_source[
        question_floor:question_tail_left
    ]
    local_subject = re.match(
        r"^\s*(?:a|an|the|this|that)\s+[a-z][\w'’-]*\b",
        local_question_text,
        re.IGNORECASE,
    )
    local_reference_tail = (
        local_question_text[local_subject.end():]
        if local_subject is not None
        else local_question_text
    )
    local_has_embedded_reference = re.search(
        r"\b(?:the|this|that|these|those)\s+[a-z][\w'’-]*\b",
        local_reference_tail,
        re.IGNORECASE,
    ) is not None
    local_generic_declarative = _TRUSTED_GENERIC_LOCAL_DECLARATIVE_RE.match(
        local_question_text
    )
    local_unknown_declarative = bool(
        local_generic_declarative is not None
        and local_generic_declarative.group("head").casefold()
        not in _TRUSTED_PHYSICAL_SCENARIO_HEADS
        and local_generic_declarative.group("head").casefold()
        not in _TRUSTED_GENERIC_DECLARATIVE_NONHEADS
        and not local_has_embedded_reference
    )
    local_question_frame = _TRUSTED_COMPLETE_LOCAL_QUESTION_FRAME_RE.match(
        local_question_text
    )
    local_deictic_is_self_grounded = bool(
        _TRUSTED_LOCAL_FRAME_BACK_REFERENCE_RE.search(local_question_text)
        and (
            _TRUSTED_QUANTIFIED_FORWARD_CONTENT_RE.search(
                local_question_text
            )
            or (
                len(_content_tokens(local_question_text)) >= 2
                and (
                    _TRUSTED_LOCAL_IF_WHEN_SUBJECT_RE.match(
                        local_question_text
                    )
                    or (
                        _TRUSTED_LOCAL_FRAME_STATE_RE.search(
                            local_question_text
                        )
                        and _TRUSTED_LOCAL_FRAME_PRONOUN_PREDICATE_RE.search(
                            local_question_text
                        ) is None
                    )
                )
            )
        )
    )
    local_frame_has_back_reference = bool(
        (
            _TRUSTED_LOCAL_FRAME_STRONG_BACK_REFERENCE_RE.search(
                local_question_text
            )
            or _TRUSTED_LOCAL_FRAME_ATTRIBUTED_REFERENCE_RE.search(
                local_question_text
            )
            or _TRUSTED_LOCAL_FRAME_REFERENTIAL_ACTION_RE.search(
                local_question_text
            )
            or (
                _TRUSTED_LOCAL_FRAME_GENERIC_DEICTIC_REFERENCE_RE.search(
                    local_question_text
                )
                and _TRUSTED_QUANTIFIED_FORWARD_CONTENT_RE.search(
                    local_question_text
                ) is None
                and _TRUSTED_LOCAL_FRAME_NAMED_SUBJECT_PREDICATE_RE.search(
                    local_question_text
                ) is None
            )
            or (
                _TRUSTED_LOCAL_FRAME_BACK_REFERENCE_RE.search(
                    local_question_text
                )
                and not local_deictic_is_self_grounded
            )
        )
    )
    if local_frame_has_back_reference:
        explicit_candidates = [
            candidate
            for candidate in explicit_candidates
            if candidate[0] != question_line
        ]
        generic_premise_candidates = [
            candidate
            for candidate in generic_premise_candidates
            if candidate[0] != question_line
        ]
        generic_declarative_candidates = [
            candidate
            for candidate in generic_declarative_candidates
            if candidate[0] != question_line
        ]
        scenario_candidates = [
            candidate
            for candidate in scenario_candidates
            if candidate[0] != question_line
        ]
    local_complete_question_frame = bool(
        local_question_frame
        and not local_frame_has_back_reference
        and _TRUSTED_COMPARATIVE_BACK_REFERENCE_RE.search(
            local_question_text
        ) is None
    )
    local_has_later_scenario_onset = any(
        line == question_line and left > question_floor
        for line, left in [
            *explicit_candidates,
            *generic_premise_candidates,
            *scenario_candidates,
        ]
    )
    local_question_tokens = _content_tokens(local_question_text)
    local_frame_has_explicit_onset = re.match(
        r"^\s*(?:at|for|given|if|in|on|under|using|when|with)\b",
        local_question_text,
        re.IGNORECASE,
    ) is not None
    local_frame_is_detached = bool(
        question_line <= 0
        or not _cue_has_weak_end(
            str(segments[question_line - 1].get("text") or ""),
            question_source,
            ignore_caption_case=True,
        )
    )
    local_question_candidate = bool(
        not local_frame_has_back_reference
        and not local_has_later_scenario_onset
        and (
            (
                local_complete_question_frame
                and len(local_question_tokens) >= 1
                and (
                    local_frame_has_explicit_onset
                    or local_frame_is_detached
                )
            )
            or (
                len(local_question_tokens) >= 3
                and (
                    local_unknown_declarative
                    or (
                        _opening_clause_is_standalone(local_question_text)
                        and not _cue_opens_mid_thought_at(
                            segments,
                            question_line,
                            ignore_caption_case=True,
                        )
                        and (
                            _TRUSTED_GENERIC_LOCAL_SCENARIO_RE.match(
                                local_question_text
                            )
                            or not local_has_embedded_reference
                        )
                    )
                )
            )
        )
    )

    def weakly_connected(start_line: int, end_line: int) -> bool:
        return all(
            _cue_has_weak_end(
                str(segments[line].get("text") or ""),
                str(segments[line + 1].get("text") or ""),
                ignore_caption_case=True,
            )
            for line in range(start_line, end_line)
        )

    scenario_chain_start: tuple[int, int] | None = None
    if scenario_candidates:
        scenario_chain_start = max(scenario_candidates)
        for candidate in sorted(scenario_candidates, reverse=True)[1:]:
            current_context = " ".join(
                str(segments[line].get("text") or "")[
                    scenario_chain_start[1]
                    if line == scenario_chain_start[0]
                    else 0:(
                        question_tail_left
                        if line == question_line
                        else None
                    )
                ]
                for line in range(
                    scenario_chain_start[0],
                    question_line + 1,
                )
            )
            candidate_head = scenario_candidate_heads.get(candidate, "")
            refers_to_candidate = bool(
                candidate_head
                and re.search(
                    rf"\b(?:the|this|that)\s+"
                    rf"{re.escape(candidate_head)}\b",
                    current_context,
                    re.IGNORECASE,
                )
            )
            if (
                weakly_connected(candidate[0], scenario_chain_start[0])
                or refers_to_candidate
            ):
                scenario_chain_start = candidate
                continue
            break

    generic_declarative_chain_start: tuple[int, int, str] | None = None
    if generic_declarative_candidates:
        latest_generic = max(generic_declarative_candidates)
        generic_declarative_chain_start = latest_generic
        for candidate in sorted(
            (
                item
                for item in generic_declarative_candidates
                if item[2] == latest_generic[2]
            ),
            reverse=True,
        )[1:]:
            if not weakly_connected(
                candidate[0],
                generic_declarative_chain_start[0],
            ):
                break
            generic_declarative_chain_start = candidate

    structural_candidate = max(explicit_candidates, default=None)
    if generic_premise_candidates:
        latest_premise = max(generic_premise_candidates)
        premise_chain_start = latest_premise
        for premise in sorted(generic_premise_candidates, reverse=True)[1:]:
            if not weakly_connected(premise[0], premise_chain_start[0]):
                break
            premise_chain_start = premise
        later_explicit = [
            candidate
            for candidate in explicit_candidates
            if candidate > latest_premise
        ]
        if later_explicit:
            structural_candidate = max(later_explicit)
        else:
            structural_candidate = premise_chain_start
            prior_explicit = [
                candidate
                for candidate in explicit_candidates
                if candidate < premise_chain_start
            ]
            if prior_explicit:
                nearest_explicit = max(prior_explicit)
                if weakly_connected(
                    nearest_explicit[0],
                    premise_chain_start[0],
                ):
                    structural_candidate = nearest_explicit

    competing_chain_starts = [
        candidate
        for candidate in (
            structural_candidate,
            generic_declarative_chain_start[:2]
            if generic_declarative_chain_start is not None
            else None,
        )
        if candidate is not None
    ]
    scenario_resets_prior_chains = bool(
        scenario_chain_start is not None
        and competing_chain_starts
        and all(
            scenario_chain_start > candidate
            and not weakly_connected(
                candidate[0],
                scenario_chain_start[0],
            )
            for candidate in competing_chain_starts
        )
    )

    if local_question_candidate:
        source_line, source_left = question_line, question_floor
    elif scenario_resets_prior_chains:
        assert scenario_chain_start is not None
        source_line, source_left = scenario_chain_start
    elif (
        structural_candidate is not None
        or generic_declarative_candidates
    ):
        latest_generic = generic_declarative_chain_start
        if (
            latest_generic is not None
            and (
                structural_candidate is None
                or latest_generic[:2] > structural_candidate
            )
        ):
            source_line, source_left, _head = latest_generic
        else:
            assert structural_candidate is not None
            source_line, source_left = structural_candidate
    elif scenario_chain_start is not None:
        source_line, source_left = scenario_chain_start
    else:
        fallback_line = (
            question_line - 1
            if (
                local_frame_has_back_reference
                and question_line - 1 in set(contiguous_lines)
            )
            else question_line
        )
        contiguous_set = set(contiguous_lines)
        for line in range(question_line - 1, min(contiguous_lines) - 1, -1):
            candidate_text = " ".join(
                str(segments[candidate_line].get("text") or "")[
                    search_floors.get(candidate_line, 0)
                    if candidate_line == line
                    else 0:(
                        question_tail_left
                        if candidate_line == question_line
                        else None
                    )
                ]
                for candidate_line in range(line, question_line + 1)
            )
            candidate_is_complete_local_frame = (
                _TRUSTED_COMPLETE_LOCAL_QUESTION_FRAME_RE.match(
                    candidate_text
                ) is not None
            )
            if (
                line not in contiguous_set
                or (
                    not candidate_is_complete_local_frame
                    and not weakly_connected(line, fallback_line)
                )
            ):
                break
            fallback_line = line
            if candidate_is_complete_local_frame:
                break
        fallback_left = search_floors.get(fallback_line, 0)
        fallback_text = " ".join(
            str(segments[line].get("text") or "")[
                fallback_left if line == fallback_line else 0:(
                    question_tail_left
                    if line == question_line
                    else None
                )
            ]
            for line in range(fallback_line, question_line + 1)
        )
        if (
            len(_content_tokens(fallback_text)) < 3
            and not (
                local_frame_has_back_reference
                and fallback_line < question_line
            )
            and _TRUSTED_COMPLETE_LOCAL_QUESTION_FRAME_RE.match(
                fallback_text
            ) is None
        ):
            return None
        source_line, source_left = fallback_line, fallback_left
    scenario_parts = [
        str(segments[line].get("text") or "")[
            source_left if line == source_line else 0:
        ]
        for line in range(source_line, question_line + 1)
    ]
    scenario = " ".join(scenario_parts)
    answer_text = str(segments[answer_line].get("text") or "")
    answer_opening = _TRUSTED_SPLIT_ANSWER_OPENING_RE.match(answer_text)
    if answer_opening is None:
        return None
    question_context = " ".join([
        scenario,
        answer_text[:answer_opening.end()],
    ])
    minimum_scenario_tokens = (
        1
        if (
            (
                source_line == question_line
                and source_left == question_floor
                and local_complete_question_frame
            )
            or _TRUSTED_COMPLETE_LOCAL_QUESTION_FRAME_RE.match(scenario)
            is not None
            or (
                local_frame_has_back_reference
                and source_line < question_line
            )
        )
        else 3
    )
    if (
        _WORKED_UNIT_QUESTION_TOKEN_RE.search(question_context) is None
        or len(_content_tokens(scenario)) < minimum_scenario_tokens
    ):
        return None
    source = str(segments[source_line].get("text") or "")
    quote = _exact_boundary_quote(source[source_left:], want="start")
    relative_span = (
        _quote_character_span(source[source_left:], quote)
        if quote
        else None
    )
    span = (
        (source_left + relative_span[0], source_left + relative_span[1])
        if relative_span is not None
        else None
    )
    return (source_line, span) if span is not None else None


def _trusted_joined_split_question_start(
    segments: list[dict],
    *,
    selected_line: int,
    selected_left: int,
    scope_text: str,
) -> tuple[tuple[int, tuple[int, int]], str] | None:
    """Recover a complete question setup without depending on caption cuts."""
    if not (0 <= selected_line < len(segments)):
        return None
    selected_source = str(segments[selected_line].get("text") or "")
    if not (0 <= selected_left <= len(selected_source)):
        return None

    first_line = selected_line
    for line in range(selected_line - 1, -1, -1):
        try:
            gap = (
                float(segments[line + 1].get("start", 0.0))
                - float(segments[line].get("end", 0.0))
            )
        except (TypeError, ValueError, OverflowError):
            break
        if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
            break
        first_line = line

    parts: list[str] = []
    line_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for line in range(first_line, selected_line + 1):
        source = str(segments[line].get("text") or "")
        right = selected_left if line == selected_line else len(source)
        piece = source[:right]
        if parts:
            parts.append(" ")
            cursor += 1
        joined_left = cursor
        parts.append(piece)
        cursor += len(piece)
        line_ranges.append((line, joined_left, cursor))
    joined = "".join(parts)
    if not joined.strip():
        return None

    completion_parts = [selected_source[selected_left:]]
    completion = _TRUSTED_JOINED_COMPLETION_RE.match(completion_parts[0])
    completion_line = selected_line
    while completion is None and completion_line + 1 < len(segments):
        try:
            gap = (
                float(segments[completion_line + 1].get("start", 0.0))
                - float(segments[completion_line].get("end", 0.0))
            )
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
            return None
        completion_line += 1
        completion_parts.append(
            str(segments[completion_line].get("text") or "")
        )
        completion = _TRUSTED_JOINED_COMPLETION_RE.match(
            " ".join(completion_parts)
        )
    if completion is None or not _WORD_RE.search(completion.group("completion")):
        return None
    completion_text = " ".join(completion_parts)
    answer_tail = completion_text[completion.end():]
    explicit_question_completion = bool(
        completion.group("mark") == "?"
        or re.search(
            r"\b(?:answer|result|solution)\b",
            completion.group("completion"),
            re.IGNORECASE,
        )
    )
    completion_is_question_edge = bool(
        explicit_question_completion or completion.group("mark") == "."
    )
    if not completion_is_question_edge:
        return None

    unit_boundaries = _trusted_joined_unit_boundaries(joined)
    prompt_floor = unit_boundaries[-1].end() if unit_boundaries else 0
    prompt_context = joined[prompt_floor:]
    wh_matches = []
    other_prompt_matches = []
    for match in _TRUSTED_JOINED_PROMPT_ONSET_RE.finditer(prompt_context):
        if match.group("wh") is not None:
            preceding = prompt_context[:match.start()]
            if (
                _TRUSTED_JOINED_QUESTION_AUX_RE.match(
                    prompt_context[match.end():]
                )
                or not _WORD_RE.search(preceding)
                or re.search(r"[,;:]\s*$", preceding)
            ):
                wh_matches.append(match)
            continue
        preceding = prompt_context[:match.start()]
        if not _WORD_RE.search(preceding) or re.search(
            r"\b(?:now|next|please|then)\s*$",
            preceding,
            re.IGNORECASE,
        ):
            other_prompt_matches.append(match)
    prompt_matches = wh_matches or other_prompt_matches
    if prompt_matches:
        question_start = prompt_floor + prompt_matches[-1].start()
    elif (
        _WORD_RE.search(prompt_context)
        and (
            explicit_question_completion
            or len(_toks(prompt_context)) == 1
        )
        and completion.group("mark") in {"?", "."}
    ):
        # An arbitrary imperative cannot be recognized from a universal verb
        # list.  The final unclosed unit immediately before the selected
        # completion is the conservative structural frame.
        question_start = prompt_floor
    else:
        return None

    grounding_tokens = (
        _content_tokens(" ".join((scope_text, answer_tail)))
        - _TRUSTED_SPLIT_GROUNDING_GENERIC_TOKENS
    )

    def context_tokens(text: str) -> set[str]:
        return (
            _content_tokens(text)
            - _TRUSTED_SPLIT_GROUNDING_GENERIC_TOKENS
        )

    def is_meta_barrier(text: str) -> bool:
        return _WORD_RE.search(text) is None

    unit_spans: list[tuple[int, int]] = []
    cursor = 0
    for boundary in _trusted_joined_unit_boundaries(joined[:question_start]):
        if _WORD_RE.search(joined[cursor:boundary.end()]):
            unit_spans.append((cursor, boundary.end()))
        cursor = boundary.end()
    if _WORD_RE.search(joined[cursor:question_start]):
        unit_spans.append((cursor, question_start))

    local_start = prompt_floor
    local_prefix = joined[local_start:question_start]
    fresh_onsets = list(
        _TRUSTED_JOINED_FRESH_SCENARIO_RE.finditer(local_prefix)
    )
    if fresh_onsets:
        local_start += fresh_onsets[-1].start("onset")
        local_prefix = joined[local_start:question_start]
    else:
        declarative_onsets = list(
            _TRUSTED_JOINED_ARTICLE_DECLARATIVE_ONSET_RE.finditer(
                local_prefix
            )
        )
        fresh_declarative = next(
            (
                onset
                for onset in reversed(declarative_onsets)
                if not re.match(
                    r"^\s*(?:at|for|from|given|if|in|on|under|using|"
                    r"when|with)\b",
                    local_prefix[:onset.start("onset")],
                    re.IGNORECASE,
                )
            ),
            None,
        )
        if fresh_declarative is not None:
            local_start += fresh_declarative.start("onset")
            local_prefix = joined[local_start:question_start]

    if context_tokens(local_prefix):
        start = local_start
        seed_index = next(
            (
                index
                for index, (left, right) in enumerate(unit_spans)
                if left <= start < right
            ),
            len(unit_spans) - 1,
        )
    else:
        prior_units = [
            (index, left, right)
            for index, (left, right) in enumerate(unit_spans)
            if right <= question_start
            and context_tokens(joined[left:right])
            and not is_meta_barrier(joined[left:right])
        ]
        if not prior_units:
            return None

        grounded_prior_units = [
            item
            for item in prior_units
            if context_tokens(joined[item[1]:item[2]]) & grounding_tokens
        ]
        seed_index, start, _seed_right = max(
            grounded_prior_units or prior_units,
            key=lambda item: item[0],
        )

    def locally_grounded(text: str) -> bool:
        stripped = text.strip(" \t\r\n,;:\u2014-")
        if not stripped:
            return False
        if _TRUSTED_JOINED_GENERIC_QUANTITY_RE.search(stripped):
            return True
        framed = re.match(
            r"^(?:at|for|in|on|under|with)\s+"
            r"(?:a|an|the|this|that|these|those)\s+"
            r"[a-z][\w'\u2019-]*\b(?P<detail>.*)$",
            stripped,
            re.IGNORECASE,
        )
        if (
            framed
            and len(context_tokens(framed.group("detail"))) >= 2
        ):
            return True
        if re.match(
            r"^(?:at|for|from|given|in|on|under|using|with)\s+"
            r"(?:this|that|these|those)\b",
            stripped,
            re.IGNORECASE,
        ):
            return False
        if _TRUSTED_LOCAL_FRAME_NAMED_SUBJECT_PREDICATE_RE.search(stripped):
            return True
        return bool(
            _TRUSTED_LOCAL_IF_WHEN_SUBJECT_RE.match(stripped)
            and (
                _TRUSTED_LOCAL_FRAME_STATE_RE.search(stripped)
                or _TRUSTED_LOCAL_FRAME_NAMED_SUBJECT_PREDICATE_RE.search(
                    stripped
                )
            )
        )

    def depends_on_previous(text: str, previous: str) -> bool:
        stripped = text.strip(" \t\r\n,;:\u2014-")
        previous_reference_tokens = _content_tokens(previous)
        if not stripped or not previous_reference_tokens:
            return False
        if _TRUSTED_JOINED_FRESH_SETUP_RE.match(stripped):
            return False
        referential_action = bool(
            _TRUSTED_LOCAL_FRAME_REFERENTIAL_ACTION_RE.search(stripped)
        )
        if _TRUSTED_JOINED_CONTINUATION_RE.match(stripped):
            return True
        predicate = _TRUSTED_LOCAL_FRAME_NAMED_SUBJECT_PREDICATE_RE.search(
            stripped
        )
        if predicate is not None:
            for reference in _TRUSTED_JOINED_DEFINITE_REFERENCE_RE.finditer(
                stripped,
                predicate.end(),
            ):
                if (
                    _content_tokens(reference.group("head"))
                    & previous_reference_tokens
                    or _TRUSTED_JOINED_CONTINUATION_RE.match(previous)
                    or _TRUSTED_LOCAL_FRAME_PRONOUN_PREDICATE_RE.search(
                        previous
                    )
                ):
                    return True
        deictic_heads = {
            token
            for match in _TRUSTED_JOINED_DEFINITE_REFERENCE_RE.finditer(
                stripped
            )
            if match.group(0).split(maxsplit=1)[0].casefold()
            in {"this", "that", "these", "those"}
            for token in _content_tokens(match.group("head"))
        }
        if (
            deictic_heads & previous_reference_tokens
            or (
                deictic_heads
                and _TRUSTED_JOINED_CONTINUATION_RE.match(previous)
            )
        ):
            return True
        if referential_action:
            return True
        if (
            _TRUSTED_LOCAL_FRAME_ATTRIBUTED_REFERENCE_RE.search(stripped)
            or _TRUSTED_LOCAL_FRAME_STRONG_BACK_REFERENCE_RE.search(stripped)
            or _TRUSTED_JOINED_NAMED_REFERENCE_RE.search(stripped)
            or _TRUSTED_LOCAL_FRAME_PRONOUN_PREDICATE_RE.search(stripped)
        ):
            return True
        leading = _TRUSTED_JOINED_LEADING_DEFINITE_RE.match(stripped)
        if (
            leading is not None
            and bool(
                _content_tokens(leading.group("head"))
                & previous_reference_tokens
            )
        ):
            return True
        if locally_grounded(stripped):
            return False
        if re.match(
            r"^(?:at|for|from|given|in|on|under|using|with)\s+"
            r"(?:the|this|that|these|those)\b",
            stripped,
            re.IGNORECASE,
        ):
            return True
        return False

    while seed_index > 0:
        current = joined[start:unit_spans[seed_index][1]]
        prior_left, prior_right = unit_spans[seed_index - 1]
        previous = joined[prior_left:prior_right]
        current_grounding = context_tokens(current) & grounding_tokens
        previous_grounding = context_tokens(previous) & grounding_tokens
        quantified_pair = bool(
            locally_grounded(current)
            and locally_grounded(previous)
            and _TRUSTED_JOINED_GENERIC_QUANTITY_RE.search(current)
            and _TRUSTED_JOINED_GENERIC_QUANTITY_RE.search(previous)
        )
        conditional_pair = bool(
            re.match(r"^\s*(?:if|when)\b", current, re.IGNORECASE)
            and re.match(
                r"^\s*(?:if|when)\b",
                previous,
                re.IGNORECASE,
            )
        )
        declarative_pair = bool(current_grounding and previous_grounding)
        co_givens = bool(
            (quantified_pair or conditional_pair or declarative_pair)
            and (
                conditional_pair
                or not current_grounding
                or bool(previous_grounding)
            )
            and _TRUSTED_JOINED_RESULT_CLAUSE_RE.search(previous) is None
        )
        if is_meta_barrier(previous) or not (
            depends_on_previous(current, previous) or co_givens
        ):
            break
        start = prior_left
        seed_index -= 1

    grounded_resets: list[int] = []
    structural_unit_starts = [0, *(
        boundary.end() for boundary in unit_boundaries
    )]
    for unit_start in structural_unit_starts:
        if not (start < unit_start < question_start):
            continue
        candidate = joined[unit_start:question_start].lstrip()
        leading_space = len(joined[unit_start:question_start]) - len(candidate)
        if (
            _TRUSTED_JOINED_FRESH_SETUP_RE.match(candidate)
            and (
                context_tokens(candidate) & grounding_tokens
                or _TRUSTED_JOINED_GENERIC_QUANTITY_RE.search(candidate)
            )
        ):
            grounded_resets.append(unit_start + leading_space)
    for framed in re.finditer(
        r"(?<!\w)(?:at|for|from|given|in|on|under|using|when|with)\s+"
        r"(?:a|an)\s+",
        joined[start:question_start],
        re.IGNORECASE,
    ):
        framed_start = start + framed.start()
        preceding_boundaries = [
            boundary.end()
            for boundary in unit_boundaries
            if boundary.end() <= framed_start
        ]
        structural_floor = (
            preceding_boundaries[-1] if preceding_boundaries else start
        )
        if _WORD_RE.search(joined[structural_floor:framed_start]):
            continue
        framed_text = joined[framed_start:question_start]
        if (
            context_tokens(framed_text) & grounding_tokens
            or _TRUSTED_JOINED_GENERIC_QUANTITY_RE.search(framed_text)
        ):
            grounded_resets.append(framed_start)
    for onset in _TRUSTED_JOINED_ARTICLE_DECLARATIVE_ONSET_RE.finditer(
        joined,
        start,
        question_start,
    ):
        onset_start = onset.start("onset")
        reset_start = onset_start
        preceding_boundaries = [
            boundary.end()
            for boundary in unit_boundaries
            if boundary.end() <= onset_start
        ]
        structural_floor = (
            preceding_boundaries[-1] if preceding_boundaries else 0
        )
        structural_prefix = joined[structural_floor:onset_start]
        prior_significant = joined[:onset_start].rstrip()
        structural_onset = bool(
            not prior_significant
            or prior_significant.endswith((".", "!", "?", ";", ":", "\u2014"))
        )
        determiner_match = re.match(
            r"(?P<determiner>a|an|any|each|every|some|the|this|that|"
            r"these|those)\b",
            onset.group("onset"),
            re.IGNORECASE,
        )
        if (
            determiner_match is not None
            and determiner_match.group("determiner").casefold()
            in {"this", "that", "these", "those"}
        ):
            continue
        if re.fullmatch(
            r"\s*(?:at|for|from|given|if|in|on|under|using|when|with)\s+",
            structural_prefix,
            re.IGNORECASE,
        ):
            structural_onset = True
            reset_start = structural_floor
        elif re.fullmatch(
            r"\s*(?:(?:and|but|so|therefore|thus)\s*[,;:]?\s*)",
            structural_prefix,
            re.IGNORECASE,
        ):
            structural_onset = True
            reset_start = structural_floor
        if not structural_onset:
            continue
        candidate_right = next(
            (
                boundary.end()
                for boundary in unit_boundaries
                if boundary.end() > reset_start
            ),
            question_start,
        )
        candidate_text = joined[reset_start:candidate_right]
        prior_text = joined[start:reset_start]
        prior_ends = [
            boundary.end()
            for boundary in unit_boundaries
            if boundary.end() <= reset_start
        ]
        prior_boundary = prior_ends[-2] if len(prior_ends) >= 2 else start
        immediate_prior = joined[prior_boundary:reset_start]
        head_match = re.match(
            r"(?:a|an|any|each|every|some|the|this|that|these|those)\s+"
            r"(?P<head>[a-z][\w'\u2019-]*)",
            onset.group("onset"),
            re.IGNORECASE,
        )
        repeated_head = bool(
            head_match
            and _content_tokens(head_match.group("head"))
            & _content_tokens(prior_text)
        )
        head_in_immediate_prior = bool(
            head_match
            and _content_tokens(head_match.group("head"))
            & _content_tokens(immediate_prior)
        )
        references_prior = any(
            _content_tokens(reference.group("head"))
            & _content_tokens(prior_text)
            for reference in _TRUSTED_JOINED_DEFINITE_REFERENCE_RE.finditer(
                candidate_text,
                max(0, onset.end("onset") - reset_start),
            )
        )
        if references_prior:
            continue
        if (
            determiner_match is not None
            and determiner_match.group("determiner").casefold() == "the"
            and head_in_immediate_prior
        ):
            continue
        candidate_grounding = context_tokens(candidate_text) & grounding_tokens
        prior_grounding = context_tokens(prior_text) & grounding_tokens
        framed_indefinite_reset = bool(
            determiner_match is not None
            and determiner_match.group("determiner").casefold() in {"a", "an"}
            and re.fullmatch(
                r"\s*(?:at|for|from|given|if|in|on|under|using|when|with)\s+",
                structural_prefix,
                re.IGNORECASE,
            )
            and (
                candidate_grounding
                or _TRUSTED_JOINED_GENERIC_QUANTITY_RE.search(candidate_text)
            )
        )
        prior_head_match = re.search(
            r"(?:^|[.!?;\u2014]\s+)(?:a|an|any|each|every|some|the)\s+"
            r"(?P<head>[a-z][\w'\u2019-]*)",
            prior_text,
            re.IGNORECASE,
        )
        distinct_grounded_co_givens = bool(
            not repeated_head
            and head_match
            and prior_head_match
            and _content_tokens(head_match.group("head")) & grounding_tokens
            and _content_tokens(prior_head_match.group("head"))
            & grounding_tokens
        )
        if (
            (candidate_grounding or repeated_head or framed_indefinite_reset)
            and not distinct_grounded_co_givens
        ):
            grounded_resets.append(reset_start)
    if grounded_resets:
        start = max(start, grounded_resets[-1])

    while start < len(joined) and joined[start].isspace():
        start += 1
    if start >= len(joined):
        return None
    source_line: int | None = None
    source_left: int | None = None
    for line, joined_left, joined_right in line_ranges:
        if joined_left <= start < joined_right:
            source_line = line
            source_left = start - joined_left
            break
    if source_line is None or source_left is None:
        return None
    source = str(segments[source_line].get("text") or "")
    quote = _exact_boundary_quote(source[source_left:], want="start")
    relative_span = (
        _quote_character_span(source[source_left:], quote)
        if quote
        else None
    )
    if relative_span is None:
        return None
    span = (
        source_left + relative_span[0],
        source_left + relative_span[1],
    )
    return (source_line, span), joined[start:question_start]


def _trusted_split_answer_scenario_start(
    segments: list[dict],
    *,
    selected_line: int,
    selected_left: int,
    scope_text: str,
) -> tuple[int, tuple[int, int]] | None:
    """Use only the cue-invariant repair; uncertainty keeps Gemini's edge."""
    joined = _trusted_joined_split_question_start(
        segments,
        selected_line=selected_line,
        selected_left=selected_left,
        scope_text=scope_text,
    )
    return joined[0] if joined is not None else None


def _trusted_grounded_forward_unit_start_legacy(
    segments: list[dict],
    *,
    selected_line: int,
    selected_left: int,
    claim_location: tuple[int, int, int, int],
    intent_locations: list[tuple[int, int, int, int]],
    scope_text: str,
) -> tuple[int, tuple[int, int], str] | None:
    """Trim a completed prior unit only when Gemini's evidence starts later."""
    grounded_positions = [
        (location[0], location[1])
        for location in [claim_location, *intent_locations]
    ]
    earliest_grounded = min(grounded_positions)
    if earliest_grounded <= (selected_line, selected_left):
        return None

    scope = str(scope_text or "")
    selected_fragment = str(segments[selected_line].get("text") or "")[
        selected_left:
    ]
    if (
        selected_left == 0
        and _TRUSTED_QUANTIFIED_FORWARD_CONTENT_RE.search(selected_fragment)
        is None
        and _ATOMIC_WORKED_SCOPE_RE.search(scope) is None
        and _TRUSTED_WORKED_TASK_SCOPE_RE.search(scope) is None
        and _ATOMIC_CAUSAL_SCOPE_RE.search(scope) is None
        and _ATOMIC_COHERENT_ARC_SCOPE_RE.search(scope) is None
        and _TRUSTED_CLAIM_SENTENCE_ARC_SCOPE_RE.search(scope) is None
        and _EXPLICIT_COMPARISON_OBJECTIVE_RE.search(scope) is None
        and _cue_opens_mid_thought_at(
            segments,
            selected_line,
            ignore_caption_case=True,
        )
    ):
        for line in range(selected_line + 1, earliest_grounded[0] + 1):
            try:
                gap = (
                    float(segments[line].get("start", 0.0))
                    - float(segments[line - 1].get("end", 0.0))
                )
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
                return None
            source = str(segments[line].get("text") or "")
            caution = _TRUSTED_FORWARD_CAUTION_HANDOFF_RE.match(source)
            if caution is None:
                continue
            through_grounding = " ".join(
                str(segments[index].get("text") or "")
                for index in range(line, earliest_grounded[0] + 1)
            )
            retained = through_grounding[caution.end():].lstrip(" ,;:—-")
            retained_sentences = _sentence_character_spans(retained)
            retained_opening = (
                retained[:retained_sentences[0][1]]
                if retained_sentences
                else retained
            )
            if (
                not _opening_clause_is_standalone(retained)
                or _opening_has_unresolved_setup_reference(retained_opening)
                or _TRUSTED_CAUTION_DEICTIC_RE.search(retained_opening)
                or _TRUSTED_COMPARATIVE_BACK_REFERENCE_RE.search(
                    retained_opening
                )
            ):
                continue
            quote = _exact_boundary_quote(source, want="start")
            span = _quote_character_span(source, quote) if quote else None
            if span is not None:
                return line, span, quote

    if (
        (
            _ATOMIC_WORKED_SCOPE_RE.search(scope)
            or _TRUSTED_WORKED_TASK_SCOPE_RE.search(scope)
        )
        and _EXPLICIT_COMPARISON_OBJECTIVE_RE.search(scope) is None
        and _EXPLICIT_RELATIONAL_OBJECTIVE_RE.search(scope) is None
    ):
        prefix_parts: list[str] = []
        worked_candidates: list[tuple[int, tuple[int, int], str]] = []
        for line in range(selected_line, earliest_grounded[0] + 1):
            source = str(segments[line].get("text") or "")
            left = selected_left if line == selected_line else 0
            right = earliest_grounded[1] if line == earliest_grounded[0] else len(source)
            if right <= left:
                continue
            for handoff in _TRUSTED_FORWARD_WORKED_HANDOFF_RE.finditer(
                source,
                left,
                right,
            ):
                before = " ".join([
                    *prefix_parts,
                    source[left:handoff.start("handoff")],
                ])
                completion_matches = [
                    (match, pattern)
                    for pattern in (
                        _SPLIT_CAPTION_COMPLETION_SIGNAL_RE,
                        _WORKED_UNIT_CLOSING_TAIL_RE,
                        _TRUSTED_FORWARD_WORKED_COMPLETION_RE,
                    )
                    for match in pattern.finditer(before)
                ]
                if not completion_matches:
                    continue
                last_completion, _completion_pattern = max(
                    completion_matches,
                    key=lambda item: item[0].end(),
                )
                after_completion = before[last_completion.end():]
                generic_completion_is_unclosed = bool(
                    _completion_pattern
                    is _SPLIT_CAPTION_COMPLETION_SIGNAL_RE
                    and re.search(
                        r"\b(?:final|fully|simplified)\b",
                        last_completion.group(0),
                        re.IGNORECASE,
                    ) is None
                    and re.search(r"[.!?]", after_completion) is None
                )
                if (
                    (
                        _completion_pattern
                        is _TRUSTED_FORWARD_WORKED_COMPLETION_RE
                        and _TRUSTED_FORWARD_WORKED_RESULT_CLAUSE_RE.search(
                            before[
                                max(0, last_completion.start() - 180):
                                last_completion.start()
                            ]
                        ) is None
                    )
                    or
                    generic_completion_is_unclosed
                    or
                    re.match(r"\s*(?:never|not)\b", after_completion, re.I)
                    or re.search(r"[.!?]\s+\S", after_completion)
                    or _TRUSTED_FRESH_WORKED_SCENARIO_RE.search(
                        after_completion
                    )
                ):
                    continue
                quote = _exact_boundary_quote(
                    source[handoff.start("handoff"):],
                    want="start",
                )
                relative_span = (
                    _quote_character_span(
                        source[handoff.start("handoff"):],
                        quote,
                    )
                    if quote
                    else None
                )
                span = (
                    (
                        handoff.start("handoff") + relative_span[0],
                        handoff.start("handoff") + relative_span[1],
                    )
                    if relative_span is not None
                    else None
                )
                if span is not None:
                    worked_candidates.append((line, span, quote))
            prefix_parts.append(source[left:right])
        if worked_candidates:
            return max(
                worked_candidates,
                key=lambda item: (item[0], item[1][0]),
            )

    if selected_line <= 0:
        return None
    selected_text = str(segments[selected_line].get("text") or "")
    selected_sentences = _sentence_character_spans(selected_text[selected_left:])
    selected_opening = (
        selected_text[selected_left:selected_left + selected_sentences[0][1]]
        if selected_sentences
        else selected_text[selected_left:]
    )
    previous_text = str(segments[selected_line - 1].get("text") or "").rstrip()
    if (
        not previous_text.endswith("?")
        or _TRUSTED_CONTEXTUAL_ANSWER_IMPERATIVE_RE.match(selected_opening) is None
        or _TRUSTED_QUANTIFIED_FORWARD_CONTENT_RE.search(selected_opening)
        is not None
    ):
        return None

    scope_tokens = _content_tokens(scope)
    selected_opening_tokens = _content_tokens(selected_opening)
    required_selected_tokens = selected_opening_tokens & scope_tokens
    for line in range(selected_line + 1, earliest_grounded[0] + 1):
        try:
            gap = (
                float(segments[line].get("start", 0.0))
                - float(segments[line - 1].get("end", 0.0))
            )
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
            return None
        source = str(segments[line].get("text") or "")
        check = source
        marker = _LEADING_DISCOURSE_MARKER_RE.match(check)
        if marker is not None:
            check = check[marker.end():]
        check = re.sub(
            r"^\s*now\s*[,;:]?\s+",
            "",
            check,
            count=1,
            flags=re.IGNORECASE,
        )
        retained_through_grounding = " ".join(
            str(segments[index].get("text") or "")
            for index in range(line, earliest_grounded[0] + 1)
        )
        retained_tokens = _content_tokens(retained_through_grounding)
        check_sentences = _sentence_character_spans(check)
        check_opening = (
            check[:check_sentences[0][1]]
            if check_sentences
            else check
        )
        if (
            not _opening_clause_is_standalone(check)
            or _TRUSTED_CAUTION_DEICTIC_RE.search(check_opening)
            or _TRUSTED_COMPARATIVE_BACK_REFERENCE_RE.search(check_opening)
            or len(_content_tokens(check) & scope_tokens) < 2
            or len(retained_tokens & selected_opening_tokens) < 2
            or not required_selected_tokens.issubset(retained_tokens)
        ):
            continue
        quote = _exact_boundary_quote(source, want="start")
        span = _quote_character_span(source, quote) if quote else None
        if span is not None:
            return line, span, quote
    return None


def _trusted_grounded_forward_unit_start(
    segments: list[dict],
    *,
    selected_line: int,
    selected_left: int,
    claim_location: tuple[int, int, int, int],
    intent_locations: list[tuple[int, int, int, int]],
    scope_text: str,
) -> tuple[int, tuple[int, int], str] | None:
    """Advance only across a joined, structurally closed prior unit."""
    if not (0 <= selected_line < len(segments)):
        return None
    selected_source = str(segments[selected_line].get("text") or "")
    if not (0 <= selected_left <= len(selected_source)):
        return None

    locations = [claim_location, *intent_locations]
    grounded = [
        location
        for location in locations
        if 0 <= location[0] <= location[2] < len(segments)
    ]
    if not grounded:
        return None
    earliest = min(grounded, key=lambda item: (item[0], item[1]))
    anchor_line, anchor_left, anchor_end_line, anchor_right = earliest
    if (anchor_line, anchor_left) <= (selected_line, selected_left):
        return None

    for line in range(selected_line, anchor_end_line):
        try:
            gap = (
                float(segments[line + 1].get("start", 0.0))
                - float(segments[line].get("end", 0.0))
            )
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(gap) or gap >= _SECTION_RESET_GAP_S:
            return None

    parts: list[str] = []
    line_ranges: list[tuple[int, int, int, int]] = []
    cursor = 0
    for line in range(selected_line, anchor_end_line + 1):
        source = str(segments[line].get("text") or "")
        source_left = selected_left if line == selected_line else 0
        source_right = anchor_right if line == anchor_end_line else len(source)
        if source_right < source_left:
            return None
        if parts:
            parts.append(" ")
            cursor += 1
        joined_left = cursor
        piece = source[source_left:source_right]
        parts.append(piece)
        cursor += len(piece)
        line_ranges.append((line, joined_left, cursor, source_left))
    joined = "".join(parts)
    if not joined.strip():
        return None

    anchor_offset = next(
        (
            joined_left + anchor_left - source_left
            for line, joined_left, _joined_right, source_left in line_ranges
            if line == anchor_line
        ),
        None,
    )
    if anchor_offset is None or anchor_offset <= 0:
        return None

    def location_text(location: tuple[int, int, int, int]) -> str:
        first, left, last, right = location
        if not (selected_line <= first <= last <= anchor_end_line):
            return ""
        pieces = []
        for line in range(first, last + 1):
            source = str(segments[line].get("text") or "")
            piece_left = left if line == first else 0
            piece_right = right if line == last else len(source)
            if piece_right > piece_left:
                pieces.append(source[piece_left:piece_right])
        return " ".join(pieces)

    anchor_tokens = _content_tokens(" ".join(
        location_text(location) for location in grounded
    )) - _TRUSTED_SPLIT_GROUNDING_GENERIC_TOKENS
    scope_tokens = (
        _content_tokens(scope_text) - _TRUSTED_SPLIT_GROUNDING_GENERIC_TOKENS
    )
    all_boundaries = _trusted_joined_unit_boundaries(joined)
    boundaries = [
        boundary
        for boundary in all_boundaries
        if boundary.end() <= anchor_offset
    ]
    selected_end = boundaries[0].end() if boundaries else anchor_offset
    selected_unit = joined[:selected_end]
    selected_scope_tokens = _content_tokens(selected_unit) & scope_tokens
    selected_reference_tokens = _content_tokens(selected_unit)
    selected_prompt = re.sub(
        r"^\s*(?:(?:and|but|now|so|then)\s*[,;:]?\s+)+",
        "",
        selected_unit,
        flags=re.IGNORECASE,
    )
    selected_is_prompt = bool(
        _TRUSTED_JOINED_PROMPT_ONSET_RE.match(selected_prompt)
        or _TRUSTED_JOINED_FRESH_SETUP_RE.match(selected_prompt)
    )

    previous_source = (
        str(segments[selected_line - 1].get("text") or "").rstrip()
        if selected_line > 0 and selected_left == 0
        else ""
    )
    selected_first_word = _WORD_RE.search(selected_unit)
    selected_action_tokens = (
        _content_tokens(selected_first_word.group(0))
        if selected_first_word is not None
        else set()
    )
    if (
        previous_source.endswith("?")
        and selected_action_tokens & scope_tokens
    ):
        return None
    if (
        (
            _ATOMIC_CAUSAL_SCOPE_RE.search(scope_text)
            or _EXPLICIT_COMPARISON_OBJECTIVE_RE.search(scope_text)
            or _EXPLICIT_RELATIONAL_OBJECTIVE_RE.search(scope_text)
            or _TRUSTED_CONTEXT_DEPENDENT_SCOPE_RE.search(scope_text)
        )
        and (
            _TRUSTED_COMPARATIVE_BACK_REFERENCE_RE.search(selected_unit)
            or (
                re.search(
                    r"\b(?:because|since|whereas|while)\b",
                    f"{previous_source} {selected_unit}",
                    re.IGNORECASE,
                )
                and _TRUSTED_CONTEXT_DEPENDENT_SCOPE_RE.search(scope_text)
            )
        )
    ):
        return None

    def next_boundary(position: int) -> int:
        return next(
            (
                boundary.end()
                for boundary in boundaries
                if boundary.end() > position
            ),
            anchor_offset,
        )

    def normalized_opening(text: str) -> str:
        opening = text.strip(" \t\r\n,;:\u2014-")
        marker = _LEADING_DISCOURSE_MARKER_RE.match(opening)
        if marker is not None:
            opening = opening[marker.end():].lstrip(" ,;:\u2014-")
        return re.sub(
            r"^now\s*[,;:]?\s+",
            "",
            opening,
            count=1,
            flags=re.IGNORECASE,
        )

    def reuses_selected_reference(text: str) -> bool:
        return any(
            _content_tokens(reference.group("head")) & selected_reference_tokens
            for reference in _TRUSTED_JOINED_DEFINITE_REFERENCE_RE.finditer(text)
        )

    candidates: list[int] = []
    explicit_handoff_candidates: set[int] = set()

    worked_handoffs = list(_TRUSTED_FORWARD_WORKED_HANDOFF_RE.finditer(
        joined,
        0,
        anchor_offset,
    ))
    for handoff in worked_handoffs:
        if handoff.start("handoff") <= 0:
            continue
        prefix = joined[:handoff.start("handoff")]
        explicit_results = list(
            _TRUSTED_JOINED_RESULT_CLAUSE_RE.finditer(prefix)
        )
        demonstrated_results = list(
            _TRUSTED_FORWARD_WORKED_COMPLETION_RE.finditer(prefix)
        )
        demonstrated_result_is_calculated = bool(
            len(list(_TRUSTED_JOINED_GENERIC_QUANTITY_RE.finditer(prefix))) >= 2
            and re.search(
                r"(?:=|\bequals?\b|\b(?:is|be)\s+(?:going\s+to\s+)?"
                r"(?:about\s+|approximately\s+|roughly\s+)?"
                r"(?:-?\d+(?:\.\d+)?|zero|one|two|three|four|five|six|"
                r"seven|eight|nine|ten)\b)",
                prefix,
                re.IGNORECASE,
            )
            and re.search(
                r"(?:[+*/=]|\b(?:divided|minus|multiplied|plus|times)\b)",
                prefix,
                re.IGNORECASE,
            )
        )
        if (
            demonstrated_results
            and demonstrated_result_is_calculated
        ):
            demonstrated = demonstrated_results[-1]
            demonstrated_boundary = next(
                (
                    boundary
                    for boundary in _trusted_joined_unit_boundaries(prefix)
                    if boundary.end() > demonstrated.end()
                ),
                None,
            )
            if (
                demonstrated_boundary is not None
                and _WORD_RE.search(prefix[demonstrated_boundary.end():])
            ):
                return None
            candidates.append(handoff.start("handoff"))
            explicit_handoff_candidates.add(handoff.start("handoff"))
            continue
        result_markers = list(explicit_results)
        closed_results: list[int] = []
        for result in result_markers:
            closing = next(
                (
                    boundary.end()
                    for boundary in _trusted_joined_unit_boundaries(prefix)
                    if boundary.end() > result.end()
                ),
                None,
            )
            if closing is not None:
                closed_results.append(closing)
        if not closed_results:
            continue
        last_closed_result = max(closed_results)
        if _WORD_RE.search(prefix[last_closed_result:]):
            # Any intervening speech may be a premise, condition, or setup.
            # Its vocabulary is irrelevant: ambiguity preserves Gemini's edge.
            return None
        candidates.append(handoff.start("handoff"))
        explicit_handoff_candidates.add(handoff.start("handoff"))

    if selected_is_prompt and not candidates:
        return None

    unit_starts = [
        boundary.end()
        for boundary in boundaries
        if 0 < boundary.end() < anchor_offset
    ]
    for start in unit_starts:
        while start < anchor_offset and joined[start].isspace():
            start += 1
        if start >= anchor_offset:
            continue
        unit_right = next(
            (
                boundary.end()
                for boundary in all_boundaries
                if boundary.end() > start
            ),
            len(joined),
        )
        unit = joined[start:unit_right]
        caution = _TRUSTED_FORWARD_CAUTION_HANDOFF_RE.match(unit)
        if caution is None:
            continue
        body = unit[caution.end():].lstrip(" ,;:\u2014-")
        body_tokens = _content_tokens(body)
        retained_before_anchor = _content_tokens(
            joined[start + caution.end():anchor_offset]
        )
        caution_continues_into_anchor = unit_right > anchor_offset
        if (
            (not body_tokens and not caution_continues_into_anchor)
            or not (
                len(retained_before_anchor & anchor_tokens) >= 2
                or len(retained_before_anchor & scope_tokens) >= 2
                or caution_continues_into_anchor
            )
            or not _opening_clause_is_standalone(body)
            or _opening_has_unresolved_setup_reference(body)
            or _TRUSTED_PARTICIPIAL_DEFINITE_REFERENCE_RE.search(body)
            or reuses_selected_reference(body)
            or _TRUSTED_CAUTION_DEICTIC_RE.search(body)
            or _TRUSTED_COMPARATIVE_BACK_REFERENCE_RE.search(body)
        ):
            continue
        candidates.append(start)

    for start in unit_starts:
        while start < anchor_offset and joined[start].isspace():
            start += 1
        if start >= anchor_offset:
            continue
        unit_right = next_boundary(start)
        unit = joined[start:unit_right]
        opening = normalized_opening(unit)
        opening_tokens = _content_tokens(opening)
        if (
            not opening_tokens
            or _TRUSTED_FORWARD_CAUTION_HANDOFF_RE.match(unit)
            or _TRUSTED_JOINED_PROMPT_ONSET_RE.match(opening)
            or _TRUSTED_JOINED_FRESH_SETUP_RE.match(opening)
            or re.match(
                r"^let(?:'s|\s+us)\s+",
                opening,
                re.IGNORECASE,
            )
            or not _opening_clause_is_standalone(opening)
            or _opening_has_unresolved_setup_reference(opening)
            or _TRUSTED_JOINED_DEFINITE_REFERENCE_RE.match(opening)
            or reuses_selected_reference(opening)
            or _TRUSTED_COMPARATIVE_BACK_REFERENCE_RE.search(opening)
            or not (
                opening_tokens & anchor_tokens
                or len(opening_tokens & scope_tokens) >= 2
            )
        ):
            continue
        candidates.append(start)

    if not candidates:
        return None
    candidate = min(candidates)
    candidate_is_explicit_handoff = candidate in explicit_handoff_candidates
    retained_tokens = _content_tokens(joined[candidate:])
    if (
        not candidate_is_explicit_handoff
        and len(selected_scope_tokens) >= 2
        and not selected_scope_tokens.issubset(retained_tokens)
    ):
        return None
    if (
        not candidate_is_explicit_handoff
        and _TRUSTED_JOINED_GENERIC_QUANTITY_RE.search(selected_unit)
        and _TRUSTED_JOINED_RESULT_CLAUSE_RE.search(selected_unit) is None
        and _TRUSTED_FORWARD_WORKED_COMPLETION_RE.search(selected_unit) is None
    ):
        return None

    while candidate < len(joined) and joined[candidate].isspace():
        candidate += 1
    mapped = next(
        (
            (line, source_left + candidate - joined_left)
            for line, joined_left, joined_right, source_left in line_ranges
            if joined_left <= candidate < joined_right
        ),
        None,
    )
    if mapped is None:
        return None
    line, source_left = mapped
    source = str(segments[line].get("text") or "")
    quote = _exact_boundary_quote(source[source_left:], want="start")
    relative_span = (
        _quote_character_span(source[source_left:], quote) if quote else None
    )
    if relative_span is None:
        return None
    span = (
        source_left + relative_span[0],
        source_left + relative_span[1],
    )
    return line, span, quote


def _trusted_authoritative_model_edges(
    segments: list[dict],
    *,
    proposal_start: int,
    proposal_end: int,
    start_anchor: _ModelBoundaryAnchor | None,
    end_anchor: _ModelBoundaryAnchor | None,
    evidence_quotes: list[str],
) -> tuple[
    int,
    tuple[int, int] | None,
    int,
    tuple[int, int] | None,
    list[str],
]:
    """Preserve Gemini's semantic words; uncertainty widens but never rejects.

    Only exact, unique Gemini-authored quotes inside the model's own ``s:e``
    frame may affect the semantic interval.  Local punctuation, topic words,
    cue casing, and timing gaps are deliberately irrelevant.  A claim or
    required intent quote may widen the model edge outward, but no local rule
    may advance the start or retract the end.
    """
    diagnostics: list[str] = []
    n = len(segments)
    start_line = min(max(proposal_start, 0), n - 1)
    end_line = min(max(proposal_end, 0), n - 1)
    if start_line > end_line:
        start_line, end_line = end_line, start_line
        diagnostics.append("reversed_model_range")

    start_span: tuple[int, int] | None = None
    end_span: tuple[int, int] | None = None
    if start_anchor is None:
        diagnostics.append("bad_or_ambiguous_start_quote")
    else:
        start_line = start_anchor.first_line
        start_span = start_anchor.first_span
    if end_anchor is None:
        diagnostics.append("bad_or_ambiguous_end_quote")
    else:
        end_line = end_anchor.last_line
        end_span = end_anchor.last_span

    if (
        start_anchor is not None
        and end_anchor is not None
        and start_anchor.first_word_position >= end_anchor.last_word_position
    ):
        start_line = min(max(proposal_start, 0), n - 1)
        end_line = min(max(proposal_end, 0), n - 1)
        if start_line > end_line:
            start_line, end_line = end_line, start_line
        start_span = end_span = None
        diagnostics.append("reversed_model_boundary")

    for quote in dict.fromkeys(
        normalized
        for value in evidence_quotes
        if (normalized := " ".join(str(value or "").split()))
    ):
        evidence_anchor = _unique_boundary_anchor(
            segments,
            quote,
            min(max(proposal_start, 0), n - 1),
            min(max(proposal_end, 0), n - 1),
            allow_timing_gaps=True,
        )
        if evidence_anchor is None:
            diagnostics.append("ambiguous_model_evidence")
            continue
        first_line = evidence_anchor.first_line
        first_left = evidence_anchor.first_span[0]
        last_line = evidence_anchor.last_line
        last_right = evidence_anchor.last_span[1]
        current_start = (
            start_line,
            start_span[0] if start_span is not None else 0,
        )
        current_end = (
            end_line,
            end_span[1]
            if end_span is not None
            else len(str(segments[end_line].get("text") or "")),
        )
        if (first_line, first_left) < current_start:
            source = str(segments[first_line].get("text") or "")
            words = list(_WORD_RE.finditer(source, first_left))
            start_line = first_line
            start_span = (
                (
                    words[0].start(),
                    words[min(5, len(words) - 1)].end(),
                )
                if words
                else None
            )
            diagnostics.append("start_expanded_to_model_evidence")
        if (last_line, last_right) > current_end:
            source = str(segments[last_line].get("text") or "")
            words = [
                word
                for word in _WORD_RE.finditer(source)
                if word.start() < last_right
            ]
            end_line = last_line
            end_span = (
                (words[max(0, len(words) - 6)].start(), last_right)
                if words
                else None
            )
            diagnostics.append("end_expanded_to_model_evidence")

    return (
        start_line,
        start_span,
        end_line,
        end_span,
        list(dict.fromkeys(diagnostics)),
    )


def _trusted_universal_compact_plan_to_report(
    plan: _CompactBoundaryPlan,
    segments: list[dict],
) -> _Conversion:
    """Convert Gemini topics without executing local semantic repair code."""
    report = _Conversion(proposed_count=len(plan.topics))
    if not segments:
        report.rejected_reasons.append("missing_segments")
        return report

    n = len(segments)
    all_constraint_ids = [
        str(constraint.constraint_id)
        for constraint in plan.request_intent.constraints
    ]
    constraint_kinds = {
        str(constraint.constraint_id): constraint.kind
        for constraint in plan.request_intent.constraints
    }
    required_constraint_ids = {
        str(constraint.constraint_id)
        for constraint in plan.request_intent.constraints
        if constraint.kind is not _IntentConstraintKind.SCOPE
    }
    used_candidate_ids: set[str] = set()

    for index, proposal in enumerate(plan.topics):
        diagnostics: list[str] = []
        range_valid = 0 <= proposal.start_line <= proposal.end_line < n
        proposal_start = min(max(proposal.start_line, 0), n - 1)
        proposal_end = min(max(proposal.end_line, 0), n - 1)
        if proposal_start > proposal_end:
            proposal_start, proposal_end = proposal_end, proposal_start
        if not range_valid:
            diagnostics.append("bad_index")

        model_start_quote = str(proposal.start_quote or "").strip()
        model_end_quote = str(proposal.end_quote or "").strip()
        raw_model_claim_quote = " ".join(
            str(proposal.claim_quote or "").split()
        )
        start_anchor = _unique_boundary_anchor(
            segments,
            model_start_quote,
            proposal_start,
            proposal_end,
            allow_timing_gaps=True,
        )
        end_anchor = _unique_boundary_anchor(
            segments,
            model_end_quote,
            proposal_start,
            proposal_end,
            allow_timing_gaps=True,
        )
        evidence_quotes = [raw_model_claim_quote]
        evidence_quotes.extend(
            str(item.evidence_quote)
            for item in proposal.intent_evidence
            if constraint_kinds.get(str(item.constraint_id))
            is not _IntentConstraintKind.SCOPE
        )
        a, start_span, b, end_span, edge_diagnostics = (
            _trusted_authoritative_model_edges(
                segments,
                proposal_start=proposal_start,
                proposal_end=proposal_end,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
                evidence_quotes=evidence_quotes,
            )
        )
        diagnostics.extend(edge_diagnostics)

        if (
            a == b
            and start_span is not None
            and end_span is not None
            and start_span[0] >= end_span[1]
        ):
            start_span = end_span = None
            diagnostics.append("full_cue_boundary_fallback")

        clip_text, semantic_spans_by_cue = _semantic_clip_slice(
            segments,
            a,
            b,
            start_span=start_span,
            end_span=end_span,
        )
        if not clip_text:
            start_span = end_span = None
            clip_text, semantic_spans_by_cue = _semantic_clip_slice(
                segments,
                a,
                b,
                start_span=None,
                end_span=None,
            )
            diagnostics.append("full_cue_boundary_fallback")

        start_text = str(segments[a].get("text") or "")
        end_text = str(segments[b].get("text") or "")
        start_quote = (
            _literal_source_quote(start_text, "", start_span)
            if start_span is not None
            else _exact_boundary_quote(start_text, want="start")
        )
        end_quote = (
            _literal_source_quote(end_text, "", end_span)
            if end_span is not None
            else _exact_boundary_quote(end_text, want="end")
        )
        start_projected = bool(
            start_span is not None
            and _WORD_RE.search(start_text[:start_span[0]])
        )
        end_projected = bool(
            end_span is not None
            and _WORD_RE.search(end_text[end_span[1]:])
        )

        grounded_constraint_ids = {
            str(item.constraint_id)
            for item in proposal.intent_evidence
            if str(item.constraint_id) in all_constraint_ids
            and _contains_quote(clip_text, str(item.evidence_quote))
        }
        topic_evidence_quote = (
            raw_model_claim_quote
            if _contains_quote(clip_text, raw_model_claim_quote)
            else next(
                (
                    str(item.evidence_quote)
                    for item in proposal.intent_evidence
                    if _contains_quote(clip_text, str(item.evidence_quote))
                ),
                _best_effort_evidence_quote(clip_text),
            )
        )
        intent_coverage = (
            len(grounded_constraint_ids & required_constraint_ids)
            / len(required_constraint_ids)
            if required_constraint_ids
            else 1.0
        )
        intent_role = (
            "primary"
            if (
                proposal.directly_teaches_topic is True
                and required_constraint_ids.issubset(grounded_constraint_ids)
            )
            else "supporting"
        )

        base_id = str(proposal.candidate_id or f"candidate-{index + 1}")
        candidate_id = base_id
        suffix = 2
        while candidate_id in used_candidate_ids:
            candidate_id = f"{base_id}-{suffix}"
            suffix += 1
        used_candidate_ids.add(candidate_id)

        cue_ids = [
            str(segments[line].get("cue_id") or f"cue-{line}")
            for line in range(a, b + 1)
        ]
        start, end = _padded_cue_bounds(segments, a, b)
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

        diagnostics = list(dict.fromkeys(diagnostics))
        clip = {
            "start": round(start, 3),
            "end": round(end, 3),
            "start_quote": start_quote,
            "end_quote": end_quote,
            "title": str(proposal.title or "").strip(),
            "learning_objective": str(proposal.learning_objective or "").strip(),
            "facet": str(proposal.facet or "").strip(),
            "reason": str(proposal.learning_objective or "").strip(),
            "kind": "educational",
            "informativeness": float(proposal.informativeness),
            "topic_relevance": float(proposal.topic_relevance),
            "educational_importance": float(proposal.educational_importance),
            "difficulty": float(proposal.difficulty),
            "directly_teaches_topic": bool(proposal.directly_teaches_topic),
            "substantive": bool(proposal.substantive),
            "factually_grounded": bool(proposal.factually_grounded),
            "self_contained": bool(proposal.self_contained),
            "is_standalone": bool(proposal.is_standalone),
            "topic_evidence_quote": topic_evidence_quote,
            "model_claim_quote": raw_model_claim_quote,
            "boundary_confidence": 1.0 if not diagnostics else 0.75,
            "boundary_repair_mode": "exact" if not diagnostics else "best_effort",
            "selection_authority": "gemini",
            "surface_eligible": True,
            "selection_candidate_id": candidate_id,
            "cue_ids": cue_ids,
            "start_cue_id": cue_ids[0],
            "end_cue_id": cue_ids[-1],
            "chain_id": "",
            "chain_position": 0,
            "prerequisite_ids": [],
            "uncertainty": "low",
            "uncertainty_reasons": [],
            "intent_role": intent_role,
            "intent_coverage": round(intent_coverage, 6),
            "intent_evidence": [
                {
                    "constraint_id": str(item.constraint_id),
                    "evidence_quote": str(item.evidence_quote),
                }
                for item in proposal.intent_evidence
            ],
            "summary": "",
            "takeaways": [],
            "match_reason": "",
            "assessment": None,
            "sequence_index": index + 1,
            "_start_line": a,
            "_end_line": b,
            "_clip_id": f"clip-{index + 1:03d}-{a}-{b}",
            "_clip_text": clip_text,
            "_proposal_index": index,
            "_semantic_spans_by_cue": semantic_spans_by_cue,
            "_quote_repaired": bool(diagnostics),
            "_boundary_fallback_reasons": diagnostics,
        }
        if edge_projection:
            clip["edge_projection"] = edge_projection
        report.clips.append(clip)

    return report


def _trusted_compact_plan_to_report(
    plan: _CompactBoundaryPlan,
    segments: list[dict],
    settings: dict,
) -> _Conversion:
    """Pass every schema-valid Gemini topic; repair only structural boundaries."""
    report = _Conversion(proposed_count=len(plan.topics))
    n = len(segments)
    if not n:
        report.rejected_reasons.append("missing_segments")
        return report

    all_constraint_ids = [
        str(constraint.constraint_id)
        for constraint in plan.request_intent.constraints
    ]
    constraint_kinds = {
        str(constraint.constraint_id): constraint.kind
        for constraint in plan.request_intent.constraints
    }
    required_constraint_ids = {
        str(constraint.constraint_id)
        for constraint in plan.request_intent.constraints
        if constraint.kind is not _IntentConstraintKind.SCOPE
    }
    used_candidate_ids: set[str] = set()

    def boundary_anchor(
        quote: str,
        start_line: int,
        end_line: int,
        *,
        want: str,
    ) -> _ModelBoundaryAnchor | None:
        return _nearest_boundary_anchor(
            segments,
            quote,
            start_line,
            end_line,
            want=want,
        )

    for index, proposal in enumerate(plan.topics):
        diagnostics: list[str] = []
        proposed_range_valid = 0 <= proposal.start_line <= proposal.end_line < n
        a = min(max(proposal.start_line, 0), n - 1)
        b = min(max(proposal.end_line, 0), n - 1)
        if a > b:
            a, b = b, a
        proposal_start, proposal_end = a, b
        if not proposed_range_valid:
            diagnostics.append("bad_index")

        model_start_quote = str(proposal.start_quote or "").strip()
        effective_start_quote = model_start_quote
        model_end_quote = str(proposal.end_quote or "").strip()
        effective_end_quote = model_end_quote
        start_anchor = boundary_anchor(
            model_start_quote,
            proposal_start,
            proposal_end,
            want="start",
        )
        end_anchor = boundary_anchor(
            model_end_quote,
            proposal_start,
            proposal_end,
            want="end",
        )
        raw_model_claim_quote = " ".join(
            str(proposal.claim_quote or "").split()
        )
        model_claim_quote = raw_model_claim_quote
        claim_location = _proposal_evidence_location(
            segments,
            model_claim_quote,
            proposal_start,
            proposal_end,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
        )
        grounded_intent_locations: list[
            tuple[str, str, tuple[int, int, int, int]]
        ] = []
        for item in proposal.intent_evidence:
            constraint_id = str(item.constraint_id)
            if constraint_id not in constraint_kinds:
                continue
            location = _proposal_evidence_location(
                segments,
                str(item.evidence_quote),
                proposal_start,
                proposal_end,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
            )
            if location is None:
                diagnostics.append(
                    f"unanchored_intent_evidence:{constraint_id}"
                )
                continue
            grounded_intent_locations.append(
                (constraint_id, str(item.evidence_quote), location)
            )
        if claim_location is None:
            grounded_claim_anchors = [
                (evidence_quote, location)
                for constraint_id, evidence_quote, location
                in grounded_intent_locations
                if constraint_kinds[constraint_id]
                is not _IntentConstraintKind.SCOPE
            ]
            if grounded_claim_anchors:
                model_claim_quote, claim_location = min(
                    grounded_claim_anchors,
                    key=lambda item: (item[1][0], item[1][1]),
                )
                diagnostics.append(
                    "claim_anchor_recovered_from_intent_evidence"
                )
        start_span: tuple[int, int] | None = None
        end_span: tuple[int, int] | None = None

        if start_anchor is None:
            diagnostics.append("bad_start_quote")
        if end_anchor is None:
            diagnostics.append("bad_end_quote")
        if start_anchor is not None and end_anchor is not None:
            if start_anchor.first_word_position < end_anchor.last_word_position:
                recovered_a = start_anchor.first_line
                recovered_b = end_anchor.last_line
                if recovered_a != a or recovered_b != b:
                    diagnostics.append("range_recovered_from_edges")
                a, b = recovered_a, recovered_b
                start_span = start_anchor.first_span
                end_span = end_anchor.last_span
            else:
                diagnostics.append("reversed_model_boundary")
        else:
            if start_anchor is not None:
                a = start_anchor.first_line
                start_span = start_anchor.first_span
            if end_anchor is not None:
                b = end_anchor.last_line
                end_span = end_anchor.last_span

        projected_model_start = (
            (a, start_span)
            if start_span is not None
            else None
        )
        structural_start_trimmed = False
        trusted_structural_onset_applied = False
        if claim_location is not None:
            claim_start_line, claim_left, claim_end_line, claim_right = claim_location
            if claim_start_line < a:
                a = claim_start_line
                start_span = None
                diagnostics.append("start_expanded_to_claim")
            elif (
                claim_start_line == a
                and start_span is not None
                and claim_left < start_span[0]
            ):
                start_span = None
                diagnostics.append("start_expanded_to_claim")
            if claim_end_line > b:
                b = claim_end_line
                end_span = None
                diagnostics.append("end_expanded_to_claim")
            elif (
                claim_end_line == b
                and end_span is not None
                and claim_right > end_span[1]
            ):
                end_span = None
                diagnostics.append("end_expanded_to_claim")

        for constraint_id, _evidence_quote, evidence_location in (
            grounded_intent_locations
        ):
            if constraint_kinds[constraint_id] is _IntentConstraintKind.SCOPE:
                continue
            evidence_start, evidence_left, evidence_end, evidence_right = (
                evidence_location
            )
            if evidence_start < a:
                a = evidence_start
                start_span = None
                diagnostics.append("start_expanded_to_intent_evidence")
            elif (
                evidence_start == a
                and start_span is not None
                and evidence_left < start_span[0]
            ):
                start_span = None
                diagnostics.append("start_expanded_to_intent_evidence")
            if evidence_end > b:
                b = evidence_end
                end_span = None
                diagnostics.append("end_expanded_to_intent_evidence")
            elif (
                evidence_end == b
                and end_span is not None
                and evidence_right > end_span[1]
            ):
                end_span = None
                diagnostics.append("end_expanded_to_intent_evidence")

        required_intent_locations = [
            (evidence_quote, location)
            for constraint_id, evidence_quote, location in grounded_intent_locations
            if constraint_kinds[constraint_id] is not _IntentConstraintKind.SCOPE
        ]
        if required_intent_locations:
            evidence_quote, grounded_location = min(
                required_intent_locations,
                key=lambda item: (item[1][0], item[1][1]),
            )
            grounded_start, grounded_left, grounded_end, grounded_right = (
                grounded_location
            )
            claim_starts_before_grounded_intent = bool(
                claim_location is not None
                and (claim_location[0], claim_location[1])
                < (grounded_start, grounded_left)
            )
            current_start = (
                a,
                start_span[0] if start_span is not None else 0,
            )
            intervening_parts: list[str] = []
            if current_start[0] <= grounded_start:
                for line in range(current_start[0], grounded_start + 1):
                    text = str(segments[line].get("text") or "")
                    left = current_start[1] if line == current_start[0] else 0
                    right = grounded_left if line == grounded_start else len(text)
                    if right > left:
                        intervening_parts.append(text[left:right])
            explicit_new_example_handoff = bool(
                _SPLIT_CAPTION_NEW_UNIT_FRAMING_RE.search(
                    " ".join(intervening_parts)
                )
            )
            generic_model_start = bool(
                _GENERIC_MODEL_START_RE.fullmatch(model_start_quote)
            )
            if (
                (grounded_start, grounded_left) > current_start
                and not claim_starts_before_grounded_intent
                and (generic_model_start or explicit_new_example_handoff)
            ):
                a = grounded_start
                start_text_for_evidence = str(segments[a].get("text") or "")
                start_span = (
                    grounded_left,
                    grounded_right
                    if grounded_start == grounded_end
                    else len(start_text_for_evidence),
                )
                effective_start_quote = evidence_quote
                diagnostics.append(
                    "new_example_start_advanced_to_intent_evidence"
                    if explicit_new_example_handoff
                    else "generic_start_advanced_to_intent_evidence"
                )

        if a > b:
            if claim_location is not None:
                a, b = claim_location[0], claim_location[2]
            else:
                a, b = sorted((a, b))
            start_span = end_span = None
            diagnostics.append("reversed_range_repaired")

        if claim_location is not None:
            # Gemini owns semantic topic selection.  The trusted conversion may
            # repair spoken boundaries, but it must not re-segment a valid
            # proposal with downstream topic/verb vocabularies.
            section_start = a
            claim_anchor_text = " ".join(
                str(value or "")
                for value in (
                    proposal.title,
                    proposal.learning_objective,
                    proposal.facet,
                    model_claim_quote,
                )
            )
            model_start_misses_claim_anchor = (
                len(
                    _content_tokens(model_start_quote)
                    & _content_tokens(claim_anchor_text)
                )
                < 2
            )
            model_start_misses_claim_quote = (
                len(
                    _content_tokens(model_start_quote)
                    & _content_tokens(model_claim_quote)
                )
                < 2
            )
            claim_setup_start: tuple[int, tuple[int, int]] | None = None
            named_handoff_start: tuple[int, tuple[int, int]] | None = None
            split_model_start: tuple[int, tuple[int, int]] | None = None
            explicit_definition_start: tuple[int, tuple[int, int]] | None = None
            current_start_text = str(segments[a].get("text") or "")
            current_start_left = start_span[0] if start_span is not None else 0
            current_start_is_clipped = bool(
                (
                    start_span is not None
                    and start_span[0] > 0
                    and not _projected_start_is_standalone(
                        current_start_text,
                        start_span,
                    )
                )
                or (
                    (start_span is None or start_span[0] == 0)
                    and _cue_opens_mid_thought_at(
                        segments,
                        a,
                        ignore_caption_case=True,
                    )
                )
                or (
                    start_span is None
                    and not _opening_clause_is_standalone(current_start_text)
                )
            )
            claim_mismatch_only = bool(
                model_start_misses_claim_quote
                and not current_start_is_clipped
                and not model_start_misses_claim_anchor
            )
            current_selected = current_start_text[current_start_left:]
            current_sentences = _sentence_character_spans(current_selected)
            current_opening = (
                current_selected[:current_sentences[0][1]]
                if current_sentences
                else current_selected
            )
            preserve_complete_local_setup = bool(
                claim_mismatch_only
                and _TRUSTED_PROJECTED_SETUP_RE.match(current_opening)
                and _local_example_setup_is_complete(current_opening)
            )
            if (
                current_start_is_clipped
                and start_anchor is not None
                and start_anchor.first_line == a
            ):
                split_model_start = _trusted_split_model_start_context(
                    segments,
                    selected_line=a,
                    selected_span=start_anchor.first_span,
                )
            if (
                current_start_is_clipped
                or model_start_misses_claim_anchor
                or model_start_misses_claim_quote
            ):
                named_handoff_start = _trusted_named_teaching_handoff_start(
                    segments,
                    search_start_line=min(
                        a,
                        start_anchor.first_line
                        if start_anchor is not None
                        else a,
                    ),
                    claim_location=claim_location,
                    anchor_text=claim_anchor_text,
                )
            if (
                not preserve_complete_local_setup
                and (
                    current_start_is_clipped
                    or model_start_misses_claim_anchor
                    or model_start_misses_claim_quote
                )
            ):
                claim_setup_start = _trusted_claim_setup_start(
                    segments,
                    selected_line=a,
                    selected_left=current_start_left,
                    claim_location=claim_location,
                    anchor_text=claim_anchor_text,
                    teaching_handoff_only=not current_start_is_clipped,
                    teaching_subject_anchor_text=(
                        model_claim_quote if claim_mismatch_only else ""
                    ),
                )
                if (
                    claim_setup_start is not None
                    and claim_setup_start[0] < section_start
                ):
                    claim_setup_start = None
            if (
                not preserve_complete_local_setup
                and (
                    model_start_misses_claim_anchor
                    or model_start_misses_claim_quote
                )
            ):
                explicit_definition_start = _trusted_explicit_definition_start(
                    segments,
                    selected_line=a,
                    selected_left=current_start_left,
                    claim_location=claim_location,
                    claim_quote=model_claim_quote,
                    scope_text=claim_anchor_text,
                )
                if (
                    explicit_definition_start is not None
                    and explicit_definition_start[0] < section_start
                ):
                    explicit_definition_start = None
            scenario_start: tuple[int, tuple[int, int]] | None = None
            if (
                start_anchor is not None
                and a <= start_anchor.first_line <= claim_location[0]
            ):
                anchor_text = str(
                    segments[start_anchor.first_line].get("text") or ""
                )
                scenario_text = anchor_text[:start_anchor.first_span[1]]
                scenario_span = _trusted_scenario_start_span(scenario_text)
                anchor_selected = anchor_text[start_anchor.first_span[0]:]
                anchor_projected_setup = bool(
                    _TRUSTED_PROJECTED_SETUP_RE.match(anchor_selected)
                    and _local_example_setup_is_complete(anchor_selected)
                )
                anchor_is_unsafe = bool(
                    not _projected_start_is_standalone(
                        anchor_text,
                        start_anchor.first_span,
                    )
                    and not anchor_projected_setup
                )
                scenario_overlaps_start = bool(
                    scenario_span is not None
                    and scenario_span[1] > start_anchor.first_span[0]
                )
                if scenario_span is not None and (
                    scenario_overlaps_start
                    or (
                        start_anchor.first_line >= section_start
                        and anchor_is_unsafe
                    )
                ):
                    scenario_start = start_anchor.first_line, scenario_span
            if (
                section_start <= claim_location[0]
                and claim_location[2] <= b
            ):
                if named_handoff_start is not None:
                    a, start_span = named_handoff_start
                    effective_start_quote = _literal_source_quote(
                        str(segments[a].get("text") or ""),
                        "",
                        start_span,
                    )
                    structural_start_trimmed = True
                    trusted_structural_onset_applied = True
                    diagnostics.append(
                        "trimmed_clipped_start_to_named_handoff"
                    )
                elif split_model_start is not None:
                    a, start_span = split_model_start
                    effective_start_quote = _literal_source_quote(
                        str(segments[a].get("text") or ""),
                        "",
                        start_span,
                    )
                    structural_start_trimmed = True
                    trusted_structural_onset_applied = True
                    diagnostics.append(
                        "expanded_split_model_start_context"
                    )
                elif claim_setup_start is not None:
                    a, start_span = claim_setup_start
                    effective_start_quote = _literal_source_quote(
                        str(segments[a].get("text") or ""),
                        "",
                        start_span,
                    )
                    structural_start_trimmed = True
                    trusted_structural_onset_applied = True
                    diagnostics.append(
                        "trimmed_clipped_start_to_claim_setup"
                    )
                elif explicit_definition_start is not None:
                    a, start_span = explicit_definition_start
                    effective_start_quote = _literal_source_quote(
                        str(segments[a].get("text") or ""),
                        "",
                        start_span,
                    )
                    structural_start_trimmed = True
                    trusted_structural_onset_applied = True
                    diagnostics.append(
                        "trimmed_completed_definitions_before_claim"
                    )
                elif scenario_start is not None:
                    a, start_span = scenario_start
                    structural_start_trimmed = True
                    trusted_structural_onset_applied = True
                    diagnostics.append("trimmed_scenario_before_claim")
                elif section_start != a:
                    a = section_start
                    start_span = None
                    structural_start_trimmed = True
                    diagnostics.append("trimmed_adjacent_unit_before")

        if structural_start_trimmed:
            structural_start_text = str(segments[a].get("text") or "")
            scenario_span = _trusted_scenario_start_span(structural_start_text)
            reset_span = _trusted_hard_reset_start_span(
                structural_start_text
            )
            structural_left = start_span[0] if start_span is not None else 0
            structural_right = (
                claim_location[1]
                if claim_location is not None and a == claim_location[0]
                else len(structural_start_text)
            )
            structural_span = max(
                (
                    span
                    for span in (scenario_span, reset_span)
                    if span is not None
                    and structural_left <= span[0] <= structural_right
                ),
                default=None,
                key=lambda span: span[0],
            )
            if structural_span is not None:
                start_span = structural_span
                effective_start_quote = _literal_source_quote(
                    structural_start_text,
                    "",
                    structural_span,
                )
                diagnostics.append("trimmed_same_cue_unit_before")
                trusted_structural_onset_applied = True

        final_start_text = str(segments[a].get("text") or "")
        final_start_left = start_span[0] if start_span is not None else 0
        if structural_start_trimmed:
            # A non-zero projected span is the explicit claim/scenario handoff
            # selected above. Do not walk backward and undo that repair. Only
            # recover a structural trim that still lands at a clipped cue edge.
            final_start_is_clipped = bool(
                final_start_left == 0
                and (
                    _cue_opens_mid_thought_at(
                        segments,
                        a,
                        ignore_caption_case=True,
                    )
                    or (
                        start_span is None
                        and not _opening_clause_is_standalone(final_start_text)
                    )
                )
            )
        else:
            final_start_is_clipped = bool(
                (
                    start_span is not None
                    and start_span[0] > 0
                    and not _projected_start_is_standalone(
                        final_start_text,
                        start_span,
                    )
                )
                or (
                    final_start_left == 0
                    and _cue_opens_mid_thought_at(
                        segments,
                        a,
                        ignore_caption_case=True,
                    )
                )
                or (
                    start_span is None
                    and not _opening_clause_is_standalone(final_start_text)
                )
            )
        if (
            final_start_is_clipped
            and not trusted_structural_onset_applied
            and claim_location is not None
            and a <= claim_location[0] <= claim_location[2] <= b
        ):
            final_claim_setup = _trusted_claim_setup_start(
                segments,
                selected_line=a,
                selected_left=final_start_left,
                claim_location=claim_location,
                anchor_text=model_claim_quote,
                teaching_handoff_only=True,
                teaching_subject_anchor_text=model_claim_quote,
                allow_validated_single_subject_anchor=True,
            )
            if (
                final_claim_setup is not None
                and (
                    final_claim_setup[0],
                    final_claim_setup[1][0],
                ) > (a, final_start_left)
            ):
                a, start_span = final_claim_setup
                effective_start_quote = _literal_source_quote(
                    str(segments[a].get("text") or ""),
                    "",
                    start_span,
                )
                structural_start_trimmed = True
                trusted_structural_onset_applied = True
                final_start_is_clipped = False
                diagnostics.append(
                    "trimmed_clipped_start_to_claim_setup"
                )
        final_start_text = str(segments[a].get("text") or "")
        final_start_left = start_span[0] if start_span is not None else 0
        if not trusted_structural_onset_applied:
            final_start_is_clipped = bool(
                (
                    start_span is not None
                    and start_span[0] > 0
                    and not _projected_start_is_standalone(
                        final_start_text,
                        start_span,
                    )
                )
                or (
                    final_start_left == 0
                    and _cue_opens_mid_thought_at(
                        segments,
                        a,
                        ignore_caption_case=True,
                    )
                )
                or (
                    start_span is None
                    and not _opening_clause_is_standalone(final_start_text)
                )
            )
        final_scope_text = " ".join(
            str(value or "")
            for value in (
                proposal.title,
                proposal.learning_objective,
                proposal.facet,
                model_claim_quote,
            )
        )
        prior_worked_setup = (
            _trusted_prior_worked_question_start(
                segments,
                selected_line=a,
                scope_text=final_scope_text,
            )
            if final_start_is_clipped
            and not trusted_structural_onset_applied
            else None
        )
        if prior_worked_setup is not None:
            a, start_span = prior_worked_setup
            effective_start_quote = _literal_source_quote(
                str(segments[a].get("text") or ""),
                "",
                start_span,
            )
            structural_start_trimmed = True
            trusted_structural_onset_applied = True
            final_start_is_clipped = False
            diagnostics.append("expanded_prior_worked_question_setup")
            final_start_text = str(segments[a].get("text") or "")
            final_start_left = start_span[0]
        split_claim_subject_start: tuple[int, tuple[int, int]] | None = None
        if (
            not trusted_structural_onset_applied
            and claim_location is not None
            and a > 0
            and claim_location[0] == a == claim_location[2]
            and claim_location[1] == final_start_left
        ):
            opening_spans = _sentence_character_spans(
                final_start_text[final_start_left:]
            )
            opening_right = (
                final_start_left + opening_spans[0][1]
                if opening_spans
                else len(final_start_text)
            )
            try:
                prior_gap = (
                    float(segments[a].get("start", 0.0))
                    - float(segments[a - 1].get("end", 0.0))
                )
            except (TypeError, ValueError, OverflowError):
                prior_gap = math.inf
            if (
                claim_location[3] <= opening_right
                and math.isfinite(prior_gap)
                and prior_gap < _SECTION_RESET_GAP_S
            ):
                prior_text = str(segments[a - 1].get("text") or "")
                prior_span = _trusted_weak_prior_start_span(
                    prior_text,
                    final_start_text[final_start_left:],
                )
                prior_scope_overlap = (
                    _content_tokens(
                        prior_text[prior_span[0]:]
                        if prior_span is not None
                        else ""
                    )
                    & _content_tokens(final_scope_text)
                )
                structural_subject = bool(
                    prior_span is not None
                    and _TRUSTED_SPLIT_SUBJECT_FRAGMENT_RE.fullmatch(
                        prior_text[prior_span[0]:]
                    )
                )
                if prior_span is not None and (
                    len(prior_scope_overlap) >= 2 or structural_subject
                ):
                    split_claim_subject_start = a - 1, prior_span
        if split_claim_subject_start is not None:
            a, start_span = split_claim_subject_start
            effective_start_quote = _literal_source_quote(
                str(segments[a].get("text") or ""),
                "",
                start_span,
            )
            structural_start_trimmed = True
            trusted_structural_onset_applied = True
            final_start_is_clipped = False
            diagnostics.append("expanded_split_claim_subject_context")
            final_start_text = str(segments[a].get("text") or "")
            final_start_left = start_span[0]
        same_cue_sentence_start = (
            _trusted_same_cue_sentence_start(
                segments,
                selected_line=a,
                selected_left=final_start_left,
                claim_location=claim_location,
                scope_text=final_scope_text,
            )
            if final_start_is_clipped
            and not trusted_structural_onset_applied
            and claim_location is not None
            else None
        )
        if same_cue_sentence_start is not None:
            a, start_span = same_cue_sentence_start
            effective_start_quote = _literal_source_quote(
                str(segments[a].get("text") or ""),
                "",
                start_span,
            )
            structural_start_trimmed = True
            trusted_structural_onset_applied = True
            final_start_is_clipped = False
            diagnostics.append("trimmed_dangling_same_cue_prefix")
            final_start_text = str(segments[a].get("text") or "")
            final_start_left = start_span[0]
        claim_sentence_selected = final_start_text[final_start_left:]
        relational_sentence_before_claim = bool(
            claim_location is not None
            and model_start_misses_claim_quote
            and _trusted_relational_sentence_before_claim(
                segments,
                selected_line=a,
                selected_left=final_start_left,
                claim_location=claim_location,
            )
        )
        claim_sentence_frame = (
            "note"
            if (
                _OPENING_BARE_RELATIONAL_PREDICATE_RE.match(
                    claim_sentence_selected
                )
                or relational_sentence_before_claim
            )
            else (
                "ordinal"
                if _TRUSTED_SUBJECTLESS_PREDICATE_OPENING_RE.match(
                    claim_sentence_selected
                )
                else ""
            )
        )
        claim_sentence_scope = final_scope_text
        if (
            (final_start_is_clipped or relational_sentence_before_claim)
            and not trusted_structural_onset_applied
            and claim_location is not None
            and a <= claim_location[0] <= claim_location[2] <= b
            and bool(claim_sentence_frame)
            and _TRUSTED_SPLIT_COPULA_COMPLEMENT_OPENING_RE.match(
                claim_sentence_selected
            ) is None
            and not (
                start_span is not None
                and final_start_left > 0
                and _WORD_RE.search(final_start_text[:final_start_left])
                is not None
            )
            and _ATOMIC_WORKED_SCOPE_RE.search(claim_sentence_scope) is None
            and (
                _TRUSTED_WORKED_TASK_SCOPE_RE.search(claim_sentence_scope)
                is None
                or relational_sentence_before_claim
            )
            and _ATOMIC_CAUSAL_SCOPE_RE.search(claim_sentence_scope) is None
            and _TRUSTED_CLAIM_SENTENCE_ARC_SCOPE_RE.search(
                claim_sentence_scope
            ) is None
        ):
            claim_sentence_start = _trusted_claim_sentence_start(
                segments,
                selected_line=a,
                claim_location=claim_location,
                allowed_frame=claim_sentence_frame,
            )
            if (
                claim_sentence_start is not None
                and (
                    claim_sentence_start[0],
                    claim_sentence_start[1][0],
                ) != (a, final_start_left)
            ):
                a, start_span = claim_sentence_start
                effective_start_quote = _literal_source_quote(
                    str(segments[a].get("text") or ""),
                    "",
                    start_span,
                )
                structural_start_trimmed = True
                trusted_structural_onset_applied = True
                final_start_is_clipped = False
                diagnostics.append(
                    "trimmed_clipped_start_to_claim_sentence"
                )
        should_repair_start_context = bool(
            not structural_start_trimmed
            or (
                final_start_is_clipped
                and not trusted_structural_onset_applied
            )
        )
        if should_repair_start_context:
            a, start_span, start_context_diagnostics = (
                _trusted_start_context_repair(
                    segments,
                    a,
                    start_span,
                    force_clipped_start=(
                        final_start_is_clipped and structural_start_trimmed
                    ),
                    min_start_line=(
                        max(0, a - 1) if structural_start_trimmed else 0
                    ),
                )
            )
            diagnostics.extend(start_context_diagnostics)

        split_answer_start = _trusted_split_answer_scenario_start(
            segments,
            selected_line=a,
            selected_left=(start_span[0] if start_span is not None else 0),
            scope_text=final_scope_text,
        )
        if split_answer_start is not None:
            answer_line, answer_span = split_answer_start
            if (answer_line, answer_span[0]) < (
                a,
                start_span[0] if start_span is not None else 0,
            ):
                a, start_span = answer_line, answer_span
                effective_start_quote = _literal_source_quote(
                    str(segments[a].get("text") or ""),
                    "",
                    start_span,
                )
                structural_start_trimmed = True
                trusted_structural_onset_applied = True
                diagnostics.append("expanded_split_answer_scenario")

        forward_origins = [(
            a,
            start_span[0] if start_span is not None else 0,
        )]
        grounded_forward_candidates = [
            candidate
            for forward_line, forward_left in forward_origins
            if claim_location is not None
            for candidate in [
                _trusted_grounded_forward_unit_start(
                    segments,
                    selected_line=forward_line,
                    selected_left=forward_left,
                    claim_location=claim_location,
                    intent_locations=[
                        location
                        for constraint_id, _quote, location
                        in grounded_intent_locations
                        if constraint_kinds[constraint_id]
                        is not _IntentConstraintKind.SCOPE
                    ],
                    scope_text=final_scope_text,
                )
            ]
            if candidate is not None
        ]
        grounded_forward_start = max(
            grounded_forward_candidates,
            default=None,
            key=lambda item: (item[0], item[1][0]),
        )
        if grounded_forward_start is not None:
            forward_line, forward_span, forward_quote = grounded_forward_start
            if (forward_line, forward_span[0]) > (
                a,
                start_span[0] if start_span is not None else 0,
            ):
                a, start_span = forward_line, forward_span
                effective_start_quote = forward_quote
                structural_start_trimmed = True
                trusted_structural_onset_applied = True
                diagnostics.append("advanced_to_grounded_unit_handoff")

        # A prior branch can move a raw Gemini word edge and mark that move as
        # trusted, yet still land on another caption fragment (for example,
        # ``that heavier...`` -> ``means that heavier...``).  The trust bit
        # records where the edge came from; it is not proof that the final
        # spoken opening is complete.  Re-evaluate the effective edge once,
        # and recover only the adjacent split cue after a structural trim so
        # this guard cannot walk back into a completed unit we intentionally
        # removed.  Boundary uncertainty never rejects the selected clip.
        residual_start_text = str(segments[a].get("text") or "")
        residual_start_left = start_span[0] if start_span is not None else 0
        residual_start_needs_context = bool(
            (
                start_span is not None
                and start_span[0] > 0
                and not _projected_start_is_standalone(
                    residual_start_text,
                    start_span,
                )
            )
            or (
                residual_start_left == 0
                and _cue_opens_mid_thought_at(
                    segments,
                    a,
                    ignore_caption_case=True,
                )
            )
            or (
                start_span is None
                and not _opening_clause_is_standalone(residual_start_text)
            )
        )
        if (
            residual_start_needs_context
            and "expanded_projected_start_context" in diagnostics
        ):
            residual_claim_start = (
                _trusted_claim_sentence_start(
                    segments,
                    selected_line=a,
                    claim_location=claim_location,
                    allowed_frame="note",
                )
                if (
                    claim_location is not None
                    and _trusted_relational_sentence_before_claim(
                        segments,
                        selected_line=a,
                        selected_left=residual_start_left,
                        claim_location=claim_location,
                    )
                )
                else None
            )
            if residual_claim_start is not None:
                residual_line, residual_span = residual_claim_start
                residual_diagnostics = [
                    "trimmed_clipped_start_to_claim_sentence"
                ]
            else:
                residual_line, residual_span, residual_diagnostics = (
                    _trusted_start_context_repair(
                        segments,
                        a,
                        start_span,
                        force_clipped_start=True,
                        min_start_line=(
                            max(0, a - 1) if structural_start_trimmed else 0
                        ),
                    )
                )
            if (residual_line, residual_span) != (a, start_span):
                a, start_span = residual_line, residual_span
                effective_start_quote = (
                    _literal_source_quote(
                        str(segments[a].get("text") or ""),
                        "",
                        start_span,
                    )
                    if start_span is not None
                    else ""
                )
                diagnostics.extend(residual_diagnostics)
                diagnostics.append("finalized_incomplete_start_context")

        completed_unit_end = _trusted_joined_unit_end(
            segments,
            end_line=b,
            end_span=end_span,
        )
        if completed_unit_end is not None:
            b, end_span, effective_end_quote = completed_unit_end
            diagnostics.append("completed_unfinished_spoken_unit")

        start_text = str(segments[a].get("text") or "")
        end_text = str(segments[b].get("text") or "")
        start_quote = (
            _literal_source_quote(start_text, effective_start_quote, start_span)
            if start_span is not None
            else _exact_boundary_quote(start_text, want="start")
        )
        end_quote = (
            _literal_source_quote(end_text, effective_end_quote, end_span)
            if end_span is not None
            else _exact_boundary_quote(end_text, want="end")
        )
        if (
            a == b
            and start_span is not None
            and end_span is not None
            and start_span[0] >= end_span[1]
        ):
            start_span = end_span = None
            diagnostics.append("full_cue_boundary_fallback")

        clip_text, semantic_spans_by_cue = _semantic_clip_slice(
            segments,
            a,
            b,
            start_span=start_span,
            end_span=end_span,
        )
        if not clip_text:
            start_span = end_span = None
            clip_text, semantic_spans_by_cue = _semantic_clip_slice(
                segments, a, b, start_span=None, end_span=None,
            )
            diagnostics.append("full_cue_boundary_fallback")

        start_projected = bool(
            start_span is not None and _WORD_RE.search(start_text[:start_span[0]])
        )
        end_projected = bool(
            end_span is not None and _WORD_RE.search(end_text[end_span[1]:])
        )
        topic_evidence_quote = (
            raw_model_claim_quote
            if _contains_quote(clip_text, raw_model_claim_quote)
            else (
                model_claim_quote
                if _contains_quote(clip_text, model_claim_quote)
                else _best_effort_evidence_quote(clip_text)
            )
        )
        if topic_evidence_quote != raw_model_claim_quote:
            diagnostics.append("claim_quote_reanchored")

        base_id = str(proposal.candidate_id or f"candidate-{index + 1}")
        candidate_id = base_id
        suffix = 2
        while candidate_id in used_candidate_ids:
            candidate_id = f"{base_id}-{suffix}"
            suffix += 1
        used_candidate_ids.add(candidate_id)

        cue_ids = [
            str(segments[line].get("cue_id") or f"cue-{line}")
            for line in range(a, b + 1)
        ]
        start, end = _padded_cue_bounds(segments, a, b)
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

        grounded_constraint_ids = {
            str(item.constraint_id)
            for item in proposal.intent_evidence
            if str(item.constraint_id) in all_constraint_ids
            and _contains_quote(clip_text, str(item.evidence_quote))
        }
        intent_coverage = (
            len(grounded_constraint_ids & required_constraint_ids)
            / len(required_constraint_ids)
            if required_constraint_ids
            else 1.0
        )
        intent_role = (
            "primary"
            if (
                proposal.directly_teaches_topic is True
                and required_constraint_ids.issubset(grounded_constraint_ids)
            )
            else "supporting"
        )

        diagnostics = list(dict.fromkeys(diagnostics))
        clip = {
            "start": round(start, 3),
            "end": round(end, 3),
            "start_quote": start_quote,
            "end_quote": end_quote,
            "title": str(proposal.title or "").strip(),
            "learning_objective": str(proposal.learning_objective or "").strip(),
            "facet": str(proposal.facet or "").strip(),
            "reason": str(proposal.learning_objective or "").strip(),
            "kind": "educational",
            "informativeness": float(proposal.informativeness),
            "topic_relevance": float(proposal.topic_relevance),
            "educational_importance": float(proposal.educational_importance),
            "difficulty": float(proposal.difficulty),
            "directly_teaches_topic": bool(proposal.directly_teaches_topic),
            "substantive": bool(proposal.substantive),
            "factually_grounded": bool(proposal.factually_grounded),
            "self_contained": bool(proposal.self_contained),
            "is_standalone": bool(proposal.is_standalone),
            "topic_evidence_quote": topic_evidence_quote,
            "model_claim_quote": raw_model_claim_quote,
            "boundary_confidence": 1.0 if not diagnostics else 0.75,
            "boundary_repair_mode": "exact" if not diagnostics else "best_effort",
            "selection_authority": "gemini",
            "surface_eligible": True,
            "selection_candidate_id": candidate_id,
            "cue_ids": cue_ids,
            "start_cue_id": cue_ids[0],
            "end_cue_id": cue_ids[-1],
            "chain_id": "",
            "chain_position": 0,
            "prerequisite_ids": [],
            "uncertainty": "low",
            "uncertainty_reasons": [],
            "intent_role": intent_role,
            "intent_coverage": round(intent_coverage, 6),
            "intent_evidence": [
                {
                    "constraint_id": str(item.constraint_id),
                    "evidence_quote": str(item.evidence_quote),
                }
                for item in proposal.intent_evidence
            ],
            "summary": "",
            "takeaways": [],
            "match_reason": "",
            "assessment": None,
            "sequence_index": index + 1,
            "_start_line": a,
            "_end_line": b,
            "_clip_id": f"clip-{index + 1:03d}-{a}-{b}",
            "_clip_text": clip_text,
            "_proposal_index": index,
            "_semantic_spans_by_cue": semantic_spans_by_cue,
            "_quote_repaired": bool(diagnostics),
            "_boundary_fallback_reasons": diagnostics,
        }
        if edge_projection:
            clip["edge_projection"] = edge_projection
        report.clips.append(clip)

    return report


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
    if (
        isinstance(plan, _CompactBoundaryPlan)
        and settings.get("_segment_trust_gemini_semantics") is True
    ):
        if settings.get("_segment_universal_boundaries") is True:
            return _trusted_universal_compact_plan_to_report(plan, segments)
        return _trusted_compact_plan_to_report(plan, segments, settings)

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
    joint_intent_required = bool(
        intent_constraints
        and _request_requires_joint_intent_coverage(topic, intent_constraints)
    )
    comparison_subject_anchors: tuple[frozenset[str], ...] = ()
    if joint_intent_required and re.search(
        r"\b(?:versus|vs\.?)\b",
        str(topic or ""),
        re.IGNORECASE,
    ):
        comparison_subject_anchors = tuple(
            frozenset(_joint_subject_anchor_tokens(constraint, intent_constraints))
            for constraint in intent_constraints.values()
            if constraint.kind is _IntentConstraintKind.SUBJECT
        )

    ignore_caption_case = bool(settings.get("_segment_ignore_caption_case", True))
    video_grounded = (
        bool(settings.get("_segment_video_grounded"))
        if "_segment_video_grounded" in settings
        else bool(settings.get("_segment_video_grounding_required"))
    )
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
        model_boundary_is_authoritative = bool(
            isinstance(proposal, _CompactBoundaryTopic)
            and settings.get("_segment_model_boundary_authoritative") is True
        )
        model_start_line = a
        model_end_line = b
        model_start_span: tuple[int, int] | None = None
        model_end_span: tuple[int, int] | None = None
        if model_boundary_is_authoritative:
            if not 1 <= len(_toks(start_quote)) <= 16:
                report.rejected_reasons.append(
                    f"{prefix}:invalid_model_start_quote_length"
                )
                continue
            if not 1 <= len(_toks(end_quote)) <= 16:
                report.rejected_reasons.append(
                    f"{prefix}:invalid_model_end_quote_length"
                )
                continue
            model_start_anchor = _unique_boundary_anchor(
                segments, start_quote, a, b,
            )
            if model_start_anchor is None:
                report.rejected_reasons.append(
                    f"{prefix}:ungrounded_model_start_quote"
                )
                continue
            model_end_anchor = _unique_boundary_anchor(
                segments, end_quote, a, b,
            )
            if model_end_anchor is None:
                report.rejected_reasons.append(
                    f"{prefix}:ungrounded_model_end_quote"
                )
                continue
            if (
                model_start_anchor.first_word_position
                > model_end_anchor.first_word_position
                or model_start_anchor.last_word_position
                > model_end_anchor.last_word_position
            ):
                report.rejected_reasons.append(
                    f"{prefix}:reversed_model_boundary"
                )
                continue
            model_start_line = model_start_anchor.first_line
            model_start_span = model_start_anchor.first_span
            model_end_line = model_end_anchor.last_line
            model_end_span = model_end_anchor.last_span
            a, b = model_start_line, model_end_line
            start_text = str(segments[a].get("text") or "").strip()
            end_text = str(segments[b].get("text") or "").strip()
            # A caption provider may split one exact model anchor across adjacent
            # cues. Keep the model's first/last word and project only the literal
            # portion inside the cited edge cue; no semantic speech is moved.
            start_quote = _literal_source_quote(
                str(segments[a].get("text") or ""),
                start_quote,
                model_start_span,
            )
            end_quote = _literal_source_quote(
                str(segments[b].get("text") or ""),
                end_quote,
                model_end_span,
            )
        quote_repaired = False
        fallback_start_edge = False
        fallback_end_edge = False
        trimmed_incomplete_end_suffix = False
        trimmed_visual_end_suffix = False
        completed_forward_sentence = False
        completed_split_caption_tail = False
        trimmed_repeated_caption_tail = False
        trimmed_instructional_preview = False
        atomic_claim_trimmed = False
        end_quote_occurrence: str | None = None
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
        if isinstance(proposal, _CompactBoundaryTopic):
            claim_word_count = len(_toks(evidence_quote_for_section))
            if not 5 <= claim_word_count <= 16:
                report.rejected_reasons.append(
                    f"{prefix}:invalid_claim_quote_length"
                )
                continue
            if evidence_location_for_section is None:
                report.rejected_reasons.append(
                    f"{prefix}:invalid_claim_quote_grounding"
                )
                continue
            if _enumerated_claim_is_only_structure(evidence_quote_for_section):
                report.rejected_reasons.append(
                    f"{prefix}:multi_objective_overview"
                )
                continue
            if (
                _compact_claim_is_non_substantive(
                    segments,
                    evidence_quote_for_section,
                    evidence_location_for_section,
                )
                or _has_unanswered_terminal_question(evidence_quote_for_section)
            ):
                report.rejected_reasons.append(
                    f"{prefix}:non_substantive_claim_quote"
                )
                continue
        relationship_bridge_allowed: bool | None = (
            bool(
                _compact_evidence_explicitly_relates_sections(
                    evidence_quote_for_section
                )
                and (
                    _objective_explicitly_relates_sections(topic)
                    or _objective_explicitly_relates_sections(
                        objective_for_section
                    )
                )
            )
            if isinstance(proposal, _CompactBoundaryTopic)
            else None
        )
        atomic_scope_text = " ".join(
            str(value or "")
            for value in (
                topic,
                getattr(proposal, "title", ""),
                objective_for_section,
                getattr(proposal, "facet", ""),
            )
        )
        pre_atomic_source_text = _cue_clip_text(segments, a, b)
        atomic_transitions_for_section = (
            _claim_anchored_atomic_transitions(
                segments,
                a,
                b,
                evidence_location=evidence_location_for_section,
                evidence_quote=evidence_quote_for_section,
                scope_text=atomic_scope_text,
                relationship_bridge_allowed=bool(relationship_bridge_allowed),
            )
            if isinstance(proposal, _CompactBoundaryTopic)
            else []
        )
        instructional_preview_trim = _trim_initial_instructional_preview(
            segments,
            a,
            b,
            evidence_quote=evidence_quote_for_section,
        )
        if instructional_preview_trim is not None:
            a, start_quote = instructional_preview_trim
            start_text = str(segments[a].get("text") or "").strip()
            trimmed_instructional_preview = True
            start_recovered_forward = True
            quote_repaired = True
            boundary_fallback_reasons.append(
                "trimmed_opening_instructional_preview"
            )
        transition_scan_start = min(
            _recap_lookback_start(segments, a),
            _enumerated_meta_lookback_start(segments, a),
        )
        if evidence_location_for_section is not None and a > 0:
            evidence_start_line, evidence_left, _evidence_end_line, _evidence_right = (
                evidence_location_for_section
            )
            evidence_text = str(segments[evidence_start_line].get("text") or "")
            first_evidence_word = _WORD_RE.search(evidence_text)
            evidence_leadin_tokens = _toks(evidence_text[:evidence_left])
            evidence_is_at_early_structural_onset = bool(
                first_evidence_word is not None
                and (
                    evidence_left == first_evidence_word.start()
                    or (
                        1 <= len(evidence_leadin_tokens) <= 3
                        and set(evidence_leadin_tokens)
                        <= _WORKED_UNIT_STRUCTURAL_PROMPT_TOKENS
                    )
                )
            )
            previous_text = str(segments[a - 1].get("text") or "")
            previous_cross_cue_onset = _cross_cue_grounded_action_onset(
                previous_text,
                evidence_text,
            )
            previous_cross_cue_reset = _hard_topic_reset_crosses_cue_boundary(
                previous_text,
                evidence_text,
            )
            previous_unit_is_complete = bool(
                _SPLIT_CAPTION_COMPLETION_SIGNAL_RE.search(previous_text)
                or _WORKED_UNIT_CLOSING_TAIL_RE.search(previous_text)
            )
            try:
                previous_gap = (
                    float(segments[a].get("start", 0.0))
                    - float(segments[a - 1].get("end", 0.0))
                )
            except (TypeError, ValueError, OverflowError):
                previous_gap = float("inf")
            if (
                evidence_start_line == a
                and evidence_is_at_early_structural_onset
                and math.isfinite(previous_gap)
                and previous_gap < _SECTION_RESET_GAP_S
                and (
                    _cue_has_explicit_dangling_end(previous_text, evidence_text)
                    or previous_cross_cue_onset is not None
                    or previous_cross_cue_reset
                    or previous_unit_is_complete
                )
            ):
                # Gemini often cites the first cue containing the grounded
                # subject while a fixed-size caption leaves its question or
                # worked-example prompt in the immediately preceding cue.
                # Inspect that one cue for a semantic onset so context repair
                # cannot walk backward through an unrelated prior lesson.
                transition_scan_start = min(transition_scan_start, a - 1)
        topic_transitions_for_section = _candidate_topic_transitions(
            segments,
            transition_scan_start,
            b,
            evidence_quote=evidence_quote_for_section,
            learning_objective=objective_for_section,
            relationship_bridge_allowed=relationship_bridge_allowed,
            atomic_claim_required=isinstance(proposal, _CompactBoundaryTopic),
            atomic_scope_text=atomic_scope_text,
            comparison_subject_anchors=comparison_subject_anchors,
        )
        if _named_method_handoff_misgrounds_objective(
            topic_transitions_for_section,
            segments,
            transition_scan_start,
            evidence_location=evidence_location_for_section,
            learning_objective=objective_for_section,
        ):
            report.rejected_reasons.append(
                f"{prefix}:topic_evidence_precedes_named_method"
            )
            continue
        if evidence_location_for_section is not None:
            evidence_start = (
                evidence_location_for_section[0],
                evidence_location_for_section[1],
            )
            evidence_end = (
                evidence_location_for_section[2],
                evidence_location_for_section[3],
            )
            recap_marker_overlap = any(
                item.recap
                and (item.navigation_line, item.navigation_left) < evidence_end
                and evidence_start < (item.new_side_line, item.new_side_left)
                for item in topic_transitions_for_section
            )
            recap_active = False
            for item in sorted(
                topic_transitions_for_section,
                key=lambda transition: (
                    transition.new_side_line,
                    transition.new_side_left,
                ),
            ):
                if (item.new_side_line, item.new_side_left) > evidence_start:
                    break
                if item.recap:
                    recap_active = True
                if item.clears_recap:
                    recap_active = False
            if recap_marker_overlap or recap_active:
                report.rejected_reasons.append(f"{prefix}:recap_evidence")
                continue
        completion_transitions = topic_transitions_for_section
        if b + 1 < n:
            try:
                adjacent_completion_cue = (
                    float(segments[b + 1].get("start", 0.0))
                    - float(segments[b].get("end", 0.0))
                    < _SECTION_RESET_GAP_S
                )
            except (TypeError, ValueError, OverflowError):
                adjacent_completion_cue = False
            if adjacent_completion_cue:
                completion_transitions = _candidate_topic_transitions(
                    segments,
                    transition_scan_start,
                    b + 1,
                    evidence_quote=evidence_quote_for_section,
                    learning_objective=objective_for_section,
                    relationship_bridge_allowed=relationship_bridge_allowed,
                    atomic_claim_required=isinstance(
                        proposal, _CompactBoundaryTopic
                    ),
                    atomic_scope_text=atomic_scope_text,
                    comparison_subject_anchors=comparison_subject_anchors,
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
                transitions=completion_transitions,
            )
        )
        section_start, section_end = _single_objective_section_bounds(
            segments,
            a,
            b,
            evidence_location=evidence_location_for_section,
            transitions=topic_transitions_for_section,
        )
        atomic_claim_trimmed = bool(
            atomic_transitions_for_section
            and (
                section_start != a
                or section_end != b
                or intra_start_quote
                or intra_end_quote
            )
        )
        if atomic_claim_trimmed:
            boundary_fallback_reasons.append(
                "trimmed_claim_anchored_atomic_unit"
            )
        if completed_end_override is not None:
            section_end, intra_end_quote = completed_end_override
            if section_end > b:
                completed_split_caption_tail = True
                boundary_fallback_reasons.append(
                    "completed_split_caption_tail"
                )
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
        context_repair_source_text = (
            pre_atomic_source_text
            if atomic_claim_trimmed
            else _cue_clip_text(segments, a, min(n - 1, b + 1))
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
            preferred_span=(
                model_start_span
                if model_boundary_is_authoritative and a == model_start_line
                else None
            ),
        )
        repaired_start_edge = bool(
            repaired_start_edge
            or intra_start_boundary
            or trimmed_instructional_preview
        )
        if edge_error:
            start_quote = _exact_boundary_quote(start_text, want="start")
            fallback_start_edge = True
            boundary_fallback_reasons.append(edge_error)
            quote_repaired = True
        navigation_recovery = (
            ""
            if model_boundary_is_authoritative
            else _recover_start_after_edge_navigation(
                start_text,
                evidence_quote=evidence_quote_for_section,
                learning_objective=objective_for_section,
                following_text=(
                    _cue_clip_text(segments, a + 1, b) if a < b else ""
                ),
            )
        )
        if navigation_recovery:
            start_quote = navigation_recovery
            repaired_start_edge = True
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_opening_edge_navigation")
        if not repaired_start_edge:
            selected_start_span = _quote_character_span(
                start_text,
                start_quote,
                preferred_span=(
                    model_start_span
                    if model_boundary_is_authoritative and a == model_start_line
                    else None
                ),
            )
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
                    or (
                        a < b
                        and evidence_location_for_section is not None
                        and evidence_location_for_section[0] > a
                        and _ATOMIC_WORKED_SCOPE_RE.search(atomic_scope_text)
                        and _local_example_setup_is_complete(start_text)
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
            preferred_span=(
                model_end_span
                if model_boundary_is_authoritative and b == model_end_line
                else None
            ),
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
            preferred_span=(
                model_end_span
                if model_boundary_is_authoritative and b == model_end_line
                else None
            ),
        )
        if trimmed_edge_noise:
            repaired_end_edge = True
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_trailing_edge_noise")
        selected_end_span = _quote_character_span(
            end_text,
            end_quote,
            preferred_span=(
                model_end_span
                if model_boundary_is_authoritative and b == model_end_line
                else None
            ),
        )
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
        visual_trim = None
        if not video_grounded:
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
        if not trimmed_visual_end_suffix and not trimmed_terminal_meta_suffix:
            repeated_caption_trim = _trim_repeated_rolling_caption_tail(
                segments,
                b,
                end_quote,
            )
            if repeated_caption_trim is not None:
                end_quote, end_quote_occurrence = repeated_caption_trim
                repaired_end_edge = True
                fallback_end_edge = False
                intra_end_boundary = True
                trimmed_repeated_caption_tail = True
                quote_repaired = True
                boundary_fallback_reasons.append(
                    "trimmed_repeated_caption_tail"
                )
        if (
            not trimmed_visual_end_suffix
            and not trimmed_terminal_meta_suffix
            and not completed_split_caption_tail
            and not trimmed_repeated_caption_tail
        ):
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
                b, end_quote, end_quote_occurrence = split_caption_completion
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
        if (
            model_boundary_is_authoritative
            and a == model_start_line
            and model_start_span in preliminary_start_spans
        ):
            preliminary_start_spans = [model_start_span]
        if (
            model_boundary_is_authoritative
            and b == model_end_line
            and model_end_span in preliminary_end_spans
        ):
            preliminary_end_spans = [model_end_span]
        elif end_quote_occurrence == "first" and preliminary_end_spans:
            preliminary_end_spans = [preliminary_end_spans[0]]
        elif end_quote_occurrence == "last" and preliminary_end_spans:
            preliminary_end_spans = [preliminary_end_spans[-1]]
        projected_end_context = ""
        if len(preliminary_end_spans) == 1:
            projected_end_context = " ".join(
                part
                for part in (
                    _cue_clip_text(segments, a, b - 1) if a < b else "",
                    end_text[:preliminary_end_spans[0][1]].strip(),
                )
                if part
            )
        projected_end_needs_continuation = bool(
            len(preliminary_end_spans) == 1
            and b + 1 < len(segments)
            and (
                _TERMINAL_DANGLING_DISCOURSE_LEADIN_RE.search(
                    end_text[:preliminary_end_spans[0][1]]
                )
                or _terminal_content_is_explicitly_incomplete(
                    end_text[:preliminary_end_spans[0][1]]
                )
                or _projected_end_continues_same_sentence(
                    end_text,
                    preliminary_end_spans[0],
                )
                or _next_cue_completes_embedded_predicate(
                    projected_end_context,
                    str(segments[b + 1].get("text") or ""),
                )
            )
        )
        from .sentences import classify_terminator

        preceding_cue_is_closed = bool(
            a <= 0
            or classify_terminator(
                str(segments[a - 1].get("text") or "")
            )
        )
        projected_start_has_lexical_prefix = bool(
            len(preliminary_start_spans) == 1
            and _WORD_RE.search(
                start_text[:preliminary_start_spans[0][0]]
            )
        )
        projected_start_is_standalone = bool(
            len(preliminary_start_spans) == 1
            and _projected_start_is_standalone(
                start_text,
                preliminary_start_spans[0],
            )
            and (
                projected_start_has_lexical_prefix
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
            or trimmed_repeated_caption_tail
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
                (
                    intra_end_boundary
                    and not projected_end_needs_continuation
                )
                or projected_end_is_complete
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
                preferred_span=(
                    model_start_span
                    if model_boundary_is_authoritative
                    and a == model_start_line
                    else None
                ),
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
                    preferred_span=(
                        model_start_span
                        if model_boundary_is_authoritative
                        and selected_start_before_context == model_start_line
                        else None
                    ),
                ),
                end_span=_quote_character_span(
                    selected_end_text,
                    end_quote,
                    preferred_span=(
                        model_end_span
                        if model_boundary_is_authoritative
                        and b == model_end_line
                        else None
                    ),
                ),
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

        # Context closure can expand beyond the model's original edges. Run
        # semantic edge cleanup against that final span so it cannot reimport
        # an opening agenda or a terminal promise that the earlier pass never
        # saw.
        final_instructional_preview_trim = _trim_initial_instructional_preview(
            segments,
            a,
            b,
            evidence_quote=evidence_quote_for_section,
        )
        if final_instructional_preview_trim is not None:
            final_start_line, final_start_quote = final_instructional_preview_trim
            if final_start_line > a:
                a, start_quote = final_start_line, final_start_quote
                start_text = str(segments[a].get("text") or "").strip()
                repaired_start_edge = True
                trimmed_instructional_preview = True
                start_recovered_forward = True
                quote_repaired = True
                if (
                    "trimmed_opening_instructional_preview"
                    not in boundary_fallback_reasons
                ):
                    boundary_fallback_reasons.append(
                        "trimmed_opening_instructional_preview"
                    )
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
            end_quote_occurrence = None
            end_text = str(segments[b].get("text") or "").strip()
            repaired_end_edge = True
            fallback_end_edge = False
            trimmed_terminal_meta_suffix = True
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_terminal_meta_suffix")

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
        internal_trim = _trim_around_internal_structural_filler(
            segments,
            a,
            b,
            evidence_location=evidence_location_for_section,
            ignore_caption_case=ignore_caption_case,
        )
        if internal_trim is None:
            report.rejected_reasons.append(
                f"{prefix}:internal_structural_filler"
            )
            continue
        trimmed_start, trimmed_end, removed_internal_filler = internal_trim
        if removed_internal_filler:
            previous_start, previous_end = a, b
            a, b = trimmed_start, trimmed_end
            start_quote = _exact_boundary_quote(
                str(segments[a].get("text") or ""), want="start"
            )
            end_quote = _exact_boundary_quote(
                str(segments[b].get("text") or ""), want="end"
            )
            end_quote_occurrence = None
            if a > previous_start:
                repaired_start_edge = True
                fallback_start_edge = False
                start_recovered_forward = True
            if b < previous_end:
                repaired_end_edge = True
                fallback_end_edge = False
            quote_repaired = True
            boundary_fallback_reasons.append(
                "trimmed_around_internal_structural_filler"
            )
        if not trimmed_incomplete_end_suffix and _terminal_content_is_explicitly_incomplete(
            _cue_clip_text(closure_segments, a, b)
        ):
            report.rejected_reasons.append(f"{prefix}:unresolved_weak_end")
            continue
        internal_filler_reason = _internal_structural_filler_reason(segments, a, b)
        if internal_filler_reason:
            report.rejected_reasons.append(
                f"{prefix}:internal_structural_filler"
            )
            continue
        context_was_trimmed = (
            atomic_claim_trimmed
            or start_recovered_forward
            or b < selected_end_before_context
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
            recovered_context_quote = _recover_context_expansion_start_quote(
                segments,
                a,
                selected_start_before_context,
                b,
                evidence_quote=evidence_quote_for_section,
                anchor_text=f"{objective_for_section} {atomic_scope_text}",
            )
            if recovered_context_quote:
                start_quote = recovered_context_quote
                repaired_start_edge = True
                fallback_start_edge = False
                start_recovered_forward = True
                context_was_trimmed = True
                boundary_fallback_reasons.append(
                    "trimmed_context_expansion_to_topic_sentence"
                )
            else:
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
        start_text = str(segments[a].get("text") or "")
        meta_start_quote = _leading_pedagogical_meta_quote(start_text)
        if meta_start_quote:
            start_quote = meta_start_quote
            quote_repaired = True
            boundary_fallback_reasons.append(
                "trimmed_opening_pedagogical_meta"
            )
        start_span, start_projected, edge_error = _semantic_edge_quote(
            start_text,
            start_quote,
            want="start",
            preferred_span=(
                model_start_span
                if model_boundary_is_authoritative and a == model_start_line
                else None
            ),
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
        if end_quote_occurrence is None:
            end_quote, final_edge_noise_trimmed = _trim_end_quote_before_edge_noise(
                str(segments[b].get("text") or ""),
                end_quote,
                evidence_quote=evidence_quote_for_section,
                learning_objective=objective_for_section,
                preferred_span=(
                    model_end_span
                    if model_boundary_is_authoritative and b == model_end_line
                    else None
                ),
            )
        else:
            final_edge_noise_trimmed = False
        if final_edge_noise_trimmed:
            quote_repaired = True
            boundary_fallback_reasons.append("trimmed_trailing_edge_noise")
        end_span, end_projected, edge_error = _semantic_edge_quote(
            str(segments[b].get("text") or ""),
            end_quote,
            want="end",
            occurrence=end_quote_occurrence,
            preferred_span=(
                model_end_span
                if model_boundary_is_authoritative and b == model_end_line
                else None
            ),
        )
        if edge_error:
            end_quote_occurrence = None
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

        if (
            end_projected
            and _projected_end_continues_same_sentence(end_text, end_span)
            and not _opening_clause_is_standalone(
                end_text[:end_span[1]].strip()
            )
        ):
            evidence_end_line = (
                evidence_location_for_section[2]
                if evidence_location_for_section is not None
                else b
            )
            previous_line = b - 1
            previous_text = (
                str(segments[previous_line].get("text") or "").strip()
                if previous_line >= a
                else ""
            )
            if (
                previous_line < a
                or evidence_end_line > previous_line
                or not previous_text
                or _terminal_content_is_explicitly_incomplete(previous_text)
                or _cue_has_weak_end(
                    previous_text,
                    end_text,
                    ignore_caption_case=ignore_caption_case,
                )
            ):
                report.rejected_reasons.append(f"{prefix}:unresolved_weak_end")
                continue
            b = previous_line
            end_text = previous_text
            end_quote = _exact_boundary_quote(end_text, want="end")
            end_span, end_projected, edge_error = _semantic_edge_quote(
                end_text,
                end_quote,
                want="end",
            )
            if edge_error or end_span is None:
                report.rejected_reasons.append(f"{prefix}:unresolved_weak_end")
                continue
            repaired_end_edge = True
            fallback_end_edge = False
            quote_repaired = True
            boundary_fallback_reasons.append(
                "trimmed_projected_sentence_fragment"
            )
            start, end = _padded_cue_bounds(segments, a, b)
            start, end = round(start, 3), round(end, 3)

        if model_boundary_is_authoritative:
            assert model_start_span is not None and model_end_span is not None
            start_lexical_span = _lexical_span(
                str(segments[a].get("text") or ""),
                start_span,
            )
            end_lexical_span = _lexical_span(
                str(segments[b].get("text") or ""),
                end_span,
            )
            if (
                a != model_start_line
                or b != model_end_line
                or start_lexical_span is None
                or end_lexical_span is None
                or start_lexical_span[0] != model_start_span[0]
                or end_lexical_span[1] != model_end_span[1]
            ):
                report.rejected_reasons.append(
                    f"{prefix}:model_boundary_rewrite_forbidden"
                )
                continue
            a, b = model_start_line, model_end_line
            start_text = str(segments[a].get("text") or "")
            end_text = str(segments[b].get("text") or "")
            start_span = model_start_span
            end_span = model_end_span
            start_quote = _literal_source_quote(
                start_text,
                str(proposal.start_quote or "").strip(),
                start_span,
            )
            end_quote = _literal_source_quote(
                end_text,
                str(proposal.end_quote or "").strip(),
                end_span,
            )
            start_projected = bool(
                _WORD_RE.search(start_text[:start_span[0]])
            )
            end_projected = bool(
                _WORD_RE.search(end_text[end_span[1]:])
            )
            end_quote_occurrence = None
            start, end = _padded_cue_bounds(segments, a, b)
            start, end = round(start, 3), round(end, 3)
            quote_repaired = False
            atomic_claim_trimmed = False
            context_was_trimmed = False
            boundary_fallback_reasons = [
                reason
                for reason in boundary_fallback_reasons
                if reason.startswith("model_")
            ]

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
        semantic_following_text = (
            end_text[end_span[1]:]
            if end_projected and _WORD_RE.search(end_text[end_span[1]:])
            else (
                str(segments[b + 1].get("text") or "")
                if b + 1 < len(segments)
                else ""
            )
        )
        if model_boundary_is_authoritative:
            opening_is_complete = bool(
                _opening_clause_is_standalone(clip_text)
                or (
                    _OPENING_RECOVERABLE_SETUP_RE.match(clip_text)
                    and _local_example_setup_is_complete(clip_text)
                )
            )
            if not opening_is_complete:
                report.rejected_reasons.append(
                    f"{prefix}:model_boundary_start_incomplete"
                )
                continue
            if _cue_has_weak_end(
                clip_text,
                semantic_following_text,
                ignore_caption_case=ignore_caption_case,
            ):
                report.rejected_reasons.append(
                    f"{prefix}:model_boundary_end_incomplete"
                )
                continue
        if _next_cue_completes_embedded_predicate(
            clip_text,
            semantic_following_text,
        ):
            report.rejected_reasons.append(f"{prefix}:unresolved_weak_end")
            continue
        same_cue_filler_reason = _same_cue_internal_filler_reason(clip_text)
        if same_cue_filler_reason:
            report.rejected_reasons.append(
                f"{prefix}:internal_structural_filler"
            )
            continue
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
        unresolved_spoken_points = _has_unresolved_deictic_point_pair(clip_text)
        if unresolved_spoken_points or (
            not video_grounded
            and _clip_requires_visual_context(
                clip_text,
                learning_objective=objective_for_section,
                speech_blocks=visual_speech_blocks,
            )
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
            required_intent_constraint_ids = {
                constraint_id
                for constraint_id, constraint in intent_constraints.items()
                if constraint.kind is not _IntentConstraintKind.SCOPE
            } or set(intent_constraint_ids)
            proposed_intent_evidence = list(
                getattr(proposal, "intent_evidence", None) or []
            )
            evidence_by_constraint: dict[str, str] = {}
            seen_intent_constraint_ids: set[str] = set()
            invalid_intent_evidence = False
            for evidence in proposed_intent_evidence:
                constraint_id = " ".join(
                    str(getattr(evidence, "constraint_id", "") or "").split()
                )
                quote = " ".join(
                    str(getattr(evidence, "evidence_quote", "") or "").split()
                )
                quote = _trim_structural_evidence_prompt(quote)
                if (
                    not constraint_id
                    or constraint_id not in intent_constraint_ids
                    or constraint_id in seen_intent_constraint_ids
                    or not 5 <= len(_toks(quote)) <= 16
                ):
                    if joint_intent_required:
                        continue
                    invalid_intent_evidence = True
                    break
                seen_intent_constraint_ids.add(constraint_id)
                evidence_span = _quote_character_span(clip_text, quote)
                if evidence_span is None:
                    if (
                        atomic_claim_trimmed
                        and _contains_quote(pre_atomic_source_text, quote)
                    ):
                        continue
                    if joint_intent_required:
                        continue
                    invalid_intent_evidence = True
                    break
                literal_evidence = _literal_source_quote(
                    clip_text,
                    quote,
                    evidence_span,
                )
                constraint = intent_constraints[constraint_id]
                if (
                    joint_intent_required
                    and constraint.kind in {
                        _IntentConstraintKind.SUBJECT,
                        _IntentConstraintKind.OUTCOME,
                    }
                    and not _joint_subject_evidence_matches(
                        constraint,
                        literal_evidence,
                        intent_constraints,
                    )
                ):
                    continue
                if (
                    joint_intent_required
                    and constraint.kind is _IntentConstraintKind.RELATIONSHIP
                    and not _joint_relationship_evidence_matches(
                        literal_evidence,
                        topic,
                        intent_constraints,
                    )
                ):
                    continue
                evidence_by_constraint[constraint_id] = literal_evidence
            if joint_intent_required:
                for constraint_id in required_intent_constraint_ids:
                    constraint = intent_constraints[constraint_id]
                    if (
                        constraint_id in evidence_by_constraint
                        or constraint.kind not in {
                            _IntentConstraintKind.SUBJECT,
                            _IntentConstraintKind.OUTCOME,
                        }
                    ):
                        continue
                    inferred = _joint_subject_evidence_window(
                        clip_text,
                        constraint,
                        intent_constraints,
                    )
                    if inferred:
                        evidence_by_constraint[constraint_id] = inferred
                subject_evidence = {
                    constraint_id: quote
                    for constraint_id, quote in evidence_by_constraint.items()
                    if intent_constraints[constraint_id].kind
                    in {
                        _IntentConstraintKind.SUBJECT,
                        _IntentConstraintKind.OUTCOME,
                    }
                }
                for constraint_id in required_intent_constraint_ids:
                    constraint = intent_constraints[constraint_id]
                    if (
                        constraint_id in evidence_by_constraint
                        or constraint.kind is not _IntentConstraintKind.RELATIONSHIP
                    ):
                        continue
                    inferred = _joint_relationship_evidence_window(
                        clip_text,
                        topic,
                        intent_constraints,
                        subject_evidence,
                    )
                    if inferred:
                        evidence_by_constraint[constraint_id] = inferred
            if invalid_intent_evidence or not evidence_by_constraint:
                report.rejected_reasons.append(f"{prefix}:invalid_intent_evidence")
                continue
            fulfilled_ids = set(evidence_by_constraint)
            if (
                joint_intent_required
                and not required_intent_constraint_ids.issubset(fulfilled_ids)
            ):
                report.rejected_reasons.append(
                    f"{prefix}:incomplete_joint_request_coverage"
                )
                continue
            intent_role = (
                "primary"
                if required_intent_constraint_ids.issubset(fulfilled_ids)
                else "supporting"
            )
            intent_coverage = len(
                fulfilled_ids & required_intent_constraint_ids
            ) / max(1, len(required_intent_constraint_ids))
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
            if end_quote_occurrence is not None:
                edge_projection["end"]["occurrence"] = end_quote_occurrence
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


def _video_grounding_duration_seconds(
    transcript: dict,
    segments: list[dict],
) -> float:
    """Return the bounded media span that contains every timestamped cue."""
    candidates: list[float] = []
    for value in (
        transcript.get("video_duration_sec"),
        transcript.get("duration_sec"),
        transcript.get("duration"),
    ):
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(parsed) and parsed > 0.0:
            candidates.append(parsed)
    for segment in segments:
        try:
            end = float(segment.get("end", 0.0))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(end) and end > 0.0:
            candidates.append(end)
    return max(candidates, default=0.0)


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


class GeminiTokenPreflightError(RuntimeError):
    """A long request could not be priced safely before generation dispatch."""

    def __init__(
        self,
        cause: Exception,
        *,
        model: str,
        operation: str,
        prompt_version: str,
        thinking_level: str,
        retryable: bool,
        status_code: int | None,
    ):
        super().__init__("Gemini token preflight was unavailable")
        self.telemetry = {
            "error_type": type(self).__name__,
            "provider_error_type": type(cause).__name__,
            "model": model,
            "operation": operation,
            "prompt_version": prompt_version,
            "thinking_level": thinking_level,
            "retryable": bool(retryable),
            "dispatched": False,
            "token_preflight_failed": True,
        }
        if status_code is not None:
            self.telemetry["provider_status_code"] = int(status_code)


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
        and primary_leaf in {"gemini-3-flash-preview", "gemini-3.5-flash"}
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
        and statuses in {(500,), (502,), (503,), (504,)}
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
    user: str | list,
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
    media_resolution=None,
    estimated_media_tokens: int = 0,
) -> tuple[BaseModel, dict]:
    from ..gemini_client import (
        _gemini_status_code,
        _transient_gemini_error,
        count_request_tokens,
        generate_json_v3,
    )

    if isinstance(user, str):
        prompt_user_text = user
    else:
        prompt_user_text = "\n".join(
            text
            for part in user
            if isinstance((text := getattr(part, "text", None)), str) and text
        )
    prompt_text = f"{system}\n\n{prompt_user_text}"
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
            schema_bytes = len(json.dumps(
                schema.model_json_schema(),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"))
            request_bytes = len(prompt_text.encode("utf-8")) + schema_bytes
            estimated_text_tokens = max(
                1,
                math.ceil((len(prompt_text) + schema_bytes) / 3),
            )
            estimate_buffer_tokens = 1_000
            conservative_uncomputed_cost = _model_cost({
                "model": model,
                "prompt_tokens": (
                    request_bytes + max(0, int(estimated_media_tokens))
                ),
                "candidate_tokens": max_output_tokens,
            })
            if conservative_uncomputed_cost > _MAX_UNCOUNTED_SELECTOR_COST_USD:
                remaining_s = max(0.0, deadline_monotonic - time.monotonic())
                if remaining_s >= 1.0 and not _cancel_requested(cancelled):
                    try:
                        estimated_text_tokens = count_request_tokens(
                            system,
                            prompt_user_text,
                            schema,
                            model=model,
                            timeout_s=min(10.0, remaining_s),
                            thinking_level=thinking_level,
                            max_output_tokens=max_output_tokens,
                        )
                    except Exception as exc:
                        # Raw UTF-8 bytes are not tokens. Treating them as such
                        # can fabricate a >200k long-context price tier for an
                        # ordinary English transcript. Fail before generation
                        # with an explicit, retryable preflight error instead.
                        raise GeminiTokenPreflightError(
                            exc,
                            model=model,
                            operation=operation,
                            prompt_version=prompt_version,
                            thinking_level=thinking_level,
                            retryable=_transient_gemini_error(exc),
                            status_code=_gemini_status_code(exc),
                        ) from exc
                    # CountTokens receives the complete GenerateContentRequest,
                    # including its system instruction and response schema. Its
                    # server-side count chooses the price tier without a local
                    # byte/token guess or an artificial post-count buffer.
                    estimate_buffer_tokens = 0
                else:
                    raise GeminiTokenPreflightError(
                        TimeoutError("token preflight deadline unavailable"),
                        model=model,
                        operation=operation,
                        prompt_version=prompt_version,
                        thinking_level=thinking_level,
                        retryable=True,
                        status_code=None,
                    )
            reserved = budget_reserve(
                operation=operation,
                model=model,
                max_output_tokens=max_output_tokens,
                prompt_text=prompt_text,
                estimated_input_tokens=(
                    estimated_text_tokens
                    + estimate_buffer_tokens
                    + max(0, int(estimated_media_tokens))
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
                    media_resolution=media_resolution,
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
                        media_resolution=media_resolution,
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
    if tier == "pro" and prompt > 200_000:
        rates = {"input": 4.0, "output": 18.0}
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


def _boundary_selector_content(
    transcript_prompt: str,
    settings: dict,
    *,
    media_end_sec: float,
) -> tuple[str | list, object | None, bool]:
    """Return transcript-only selector content; production never uploads video."""
    del settings, media_end_sec
    return transcript_prompt, None, False


def _audit_pro_boundaries(
    plan: _CompactBoundaryPlan,
    segments: list[dict],
    topic: str,
    settings: dict,
    *,
    deadline: float,
    cancelled: CancelledCb,
) -> tuple[_CompactBoundaryPlan, list[dict]]:
    """Let Pro correct its word edges without ever dropping a selected topic."""
    if not plan.topics:
        return plan, []
    system, user, allowed = _pro_boundary_audit_prompts(plan, segments, topic)
    sink = settings.get("_segment_telemetry")
    try:
        audit, call = _call_model(
            system,
            user,
            _BoundaryRepairPlan,
            model=config.SEGMENT_PRO_MODEL,
            thinking_level="low",
            max_output_tokens=_PRO_BOUNDARY_AUDIT_OUTPUT_TOKENS,
            timeout_s=_PRO_TIMEOUT_S,
            deadline_monotonic=deadline,
            operation="pro_boundary_audit",
            prompt_version=_PRO_BOUNDARY_AUDIT_PROMPT_VERSION,
            cancelled=cancelled,
            budget_reserve=settings.get("_segment_budget_reserve"),
            budget_reconcile=settings.get("_segment_budget_reconcile"),
            max_retries=0,
        )
        call["video_grounded"] = False
        calls = [call]
    except Exception as exc:
        telemetry = _exception_telemetry(exc)
        if telemetry:
            telemetry.setdefault("error_type", type(exc).__name__)
            telemetry["video_grounded"] = False
        calls = [telemetry] if telemetry else []
        _emit(
            sink,
            "boundary_audit",
            attempted_count=len(plan.topics),
            applied_count=0,
            reason=f"request_failure:{type(exc).__name__}",
        )
        return plan, calls

    if not isinstance(audit, _BoundaryRepairPlan):
        return plan, calls

    replacements: dict[int, _CompactBoundaryTopic] = {}
    seen: set[str] = set()
    for item in audit.items:
        audit_id = str(item.candidate_id)
        permitted = allowed.get(audit_id)
        if permitted is None or audit_id in seen:
            continue
        seen.add(audit_id)
        index, start_lines, end_lines = permitted
        if (
            item.start_line not in start_lines
            or item.end_line not in end_lines
            or item.end_line < item.start_line
            or not 1 <= len(_toks(item.start_quote)) <= 16
            or not 1 <= len(_toks(item.end_quote)) <= 16
        ):
            continue
        start_anchor = _unique_boundary_anchor(
            segments,
            item.start_quote,
            item.start_line,
            item.end_line,
        )
        end_anchor = _unique_boundary_anchor(
            segments,
            item.end_quote,
            item.start_line,
            item.end_line,
        )
        if (
            start_anchor is None
            or end_anchor is None
            or start_anchor.first_word_position > end_anchor.last_word_position
        ):
            continue
        replacements[index] = plan.topics[index].model_copy(update={
            "start_line": item.start_line,
            "end_line": item.end_line,
            "start_quote": item.start_quote,
            "end_quote": item.end_quote,
        })

    audited_topics = [
        replacements.get(index, topic_item)
        for index, topic_item in enumerate(plan.topics)
    ]
    _emit(
        sink,
        "boundary_audit",
        attempted_count=len(plan.topics),
        returned_count=len(seen),
        applied_count=len(replacements),
        retained_count=len(plan.topics) - len(replacements),
    )
    return plan.model_copy(update={"topics": audited_topics}), calls


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
    boundary_profile = profile in {FLASH_SPLIT_PROFILE, PRO_BOUNDARY_PROFILE}
    # Video input is intentionally disabled for selection. Gemini receives only
    # the indexed transcript and exact user request, regardless of stale caller
    # flags, so an old setting cannot silently consume video-token quota.
    video_grounding_requested = False
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
            video_grounded=video_grounding_requested,
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
            video_grounded=video_grounding_requested,
        )
        schema = _CompactBoundaryPlan
        model = config.SEGMENT_PRO_MODEL
        # Medium was both faster and more complete than high across the clean
        # cross-domain boundary corpus, while materially reducing billed thought
        # tokens. It remains configurable for explicit evaluation callers.
        level, cap, timeout = (
            "medium",
            _PRO_BOUNDARY_OUTPUT_TOKENS,
            _PRO_TIMEOUT_S,
        )
        operation = "pro_fallback"
    else:
        raise ValueError(f"unknown segmentation profile: {profile}")

    if profile in {FLASH_SPLIT_PROFILE, PRO_BOUNDARY_PROFILE}:
        requested_level = str(
            settings.get("_segment_thinking_level") or level
        ).strip().lower()
        supported_levels = (
            {"minimal", "low", "medium", "high"}
            if profile == FLASH_SPLIT_PROFILE
            else {"low", "medium", "high"}
        )
        if requested_level in supported_levels:
            level = requested_level
    selector_user: str | list = user
    media_resolution = None
    video_grounded = False
    media_end_sec = _video_grounding_duration_seconds(transcript, segments)
    if boundary_profile:
        selector_user, media_resolution, video_grounded = _boundary_selector_content(
            user,
            settings,
            media_end_sec=media_end_sec,
        )
    estimated_media_tokens = (
        math.ceil(media_end_sec) * _LOW_RESOLUTION_VIDEO_TOKENS_PER_SECOND
        if video_grounded
        else 0
    )
    retry_flash_capacity_once = bool(
        profile == FLASH_SPLIT_PROFILE
        and not video_grounded
        and settings.get("_segment_allow_flash_lite_failover") is not True
    )
    retry_pro_capacity_once = bool(
        video_grounded and profile == PRO_BOUNDARY_PROFILE
    )
    retry_capacity_once = retry_flash_capacity_once or retry_pro_capacity_once
    operation = str(settings.get("_segment_operation") or operation)
    def invoke_selector() -> tuple[BaseModel, dict]:
        return _call_model(
            system,
            selector_user,
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
            # A confirmed 503 may receive one bounded retry on the same model.
            # This follows Gemini's transient-capacity guidance without changing
            # model tiers or allowing an open-ended wait.
            max_retries=1 if retry_capacity_once else 0,
            retry_status_codes=(
                frozenset({503}) if retry_capacity_once else None
            ),
            failover_model=(
                config.SEGMENT_FLASH_FALLBACK_MODEL
                if (
                    profile == FLASH_SPLIT_PROFILE
                    and not video_grounded
                    and settings.get("_segment_allow_flash_lite_failover") is True
                )
                else None
            ),
            media_resolution=media_resolution,
            estimated_media_tokens=estimated_media_tokens,
        )

    calls: list[dict] = []
    try:
        parsed, call = invoke_selector()
    except _SchemaResponseError as first_exc:
        # A provider-successful response can occasionally be malformed JSON. Give
        # only the authoritative text-only Pro selector one separately budgeted
        # retry with the identical transcript prompt; semantic results are never
        # retried merely because they contain few or unexpected clips.
        first_call = _exception_telemetry(first_exc)
        first_call.update({
            "error_type": type(first_exc).__name__,
            "video_grounded": video_grounded,
        })
        first_exc.telemetry = first_call
        if profile != PRO_BOUNDARY_PROFILE:
            raise
        first_call.update({
            "schema_retry_attempt": 1,
            "schema_retry_reason": "invalid_structured_response",
        })
        if _cancel_requested(cancelled) or deadline <= time.monotonic():
            raise
        calls.append(first_call)
        try:
            parsed, call = invoke_selector()
        except Exception as retry_exc:
            retry_call = _exception_telemetry(retry_exc)
            retry_call.setdefault("error_type", type(retry_exc).__name__)
            retry_call.update({
                "video_grounded": video_grounded,
                "schema_retry_attempt": 2,
                "schema_retry_exhausted": True,
            })
            # Preserve both billable attempts if the bounded retry also fails.
            retry_exc.selection_attempt_calls = [first_call, retry_call]
            raise
        call.update({
            "schema_retry_attempt": 2,
            "schema_retry_recovered": True,
        })
    call["video_grounded"] = video_grounded
    calls.append(call)
    if profile == PRO_BOUNDARY_PROFILE and isinstance(parsed, _CompactBoundaryPlan):
        parsed, boundary_audit_calls = _audit_pro_boundaries(
            parsed,
            segments,
            topic,
            settings,
            deadline=deadline,
            cancelled=cancelled,
        )
        calls.extend(boundary_audit_calls)
    require_enrichment = profile in {CORRECTED_PRO_PROFILE, FLASH_SINGLE_PROFILE}
    conversion_settings = dict(settings)
    conversion_settings["_segment_video_grounded"] = video_grounded
    conversion_settings["_segment_model_boundary_authoritative"] = (
        profile == FLASH_SPLIT_PROFILE
    )
    conversion_settings["_segment_trust_gemini_semantics"] = boundary_profile
    conversion_settings["_segment_universal_boundaries"] = (
        profile == PRO_BOUNDARY_PROFILE
    )
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
    if (
        profile in {FLASH_SPLIT_PROFILE, PRO_BOUNDARY_PROFILE}
        and not conversion_settings.get("_segment_trust_gemini_semantics")
    ):
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
        attempt_calls = getattr(exc, "selection_attempt_calls", None)
        if isinstance(attempt_calls, list):
            calls = [dict(item) for item in attempt_calls if isinstance(item, dict)]
            call = calls[-1] if calls else {}
        else:
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
                       cancelled: CancelledCb, *, fallback: bool = False,
                       budget_operation: str | None = None) -> SegmentResult:
    profile = PRO_BOUNDARY_PROFILE if fallback else AUTHORITATIVE_PRO_PROFILE
    operation = budget_operation or (
        "pro_fallback" if fallback else "pro_authoritative"
    )
    if operation not in {"pro_authoritative", "pro_fallback"}:
        raise ValueError(f"Unsupported Pro budget operation: {operation}")
    runtime_settings = dict(settings)
    runtime_settings["_segment_operation"] = operation
    result = run_segment_profile(
        transcript, runtime_settings, profile, topic=topic,
        deadline_monotonic=deadline, cancelled=cancelled,
    )
    for call in result.calls:
        if str(call.get("operation") or "") != "pro_boundary_audit":
            call["operation"] = operation
    return result


def pro_boundary_fallback_detailed(
    transcript: dict,
    settings: dict,
    *,
    topic: str = "",
    video_id: str = "",
    budget_operation: str = "pro_fallback",
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
        budget_operation=budget_operation,
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

    # Cost safety belongs at the public selector boundary, not only in the web
    # orchestration layer. Hosted jobs arrive with a shared context; CLI,
    # evaluation, and future direct callers receive one bounded local ledger.
    # Copy the mapping so installing the private hooks never mutates caller
    # configuration or leaks one transcript's budget into another invocation.
    settings = dict(settings or {})
    reserve_hook = settings.get("_segment_budget_reserve")
    reconcile_hook = settings.get("_segment_budget_reconcile")
    if not (callable(reserve_hook) and callable(reconcile_hook)):
        generation_context = (
            settings.get("generation_context")
            or settings.get("provider_context")
        )
        reserve_hook = getattr(generation_context, "reserve_gemini_call", None)
        reconcile_hook = getattr(generation_context, "reconcile_gemini_call", None)
        if not (callable(reserve_hook) and callable(reconcile_hook)):
            from ..app.clip_engine.provider_runtime import GenerationContext

            generation_mode = (
                "slow"
                if str(settings.get("generation_mode") or "").casefold() == "slow"
                else "fast"
            )
            generation_context = GenerationContext(
                generation_mode,
                generation_id=f"selector:{video_id or 'direct'}",
            )
            settings["_segment_local_budget_context"] = generation_context
            reserve_hook = generation_context.reserve_gemini_call
            reconcile_hook = generation_context.reconcile_gemini_call
        settings["_segment_budget_reserve"] = reserve_hook
        settings["_segment_budget_reconcile"] = reconcile_hook

    # Production defaults to one authoritative normal-Flash boundary selector.
    # Pro, hybrid, and shadow modes remain available only to evaluation callers.
    configured_mode = str(routing_mode or "flash_only").lower()
    flash_only = configured_mode == "flash_only"
    mode = configured_mode
    if mode not in {"pro_only", "shadow", "hybrid", "flash_only"}:
        mode = "flash_only"
        flash_only = True
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
    route = "flash_only" if flash_only else (
        "flash_first" if selected else "pro_authoritative"
    )
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
            result.route = "flash_only" if flash_only else "hybrid_flash"
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
                result.route = (
                    "flash_only_rejected"
                    if flash_only
                    else "hybrid_flash_deferred"
                )
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
        routing_mode=str(
            settings.get("_segment_routing_mode")
            or config.SEGMENT_ROUTING_MODE
        ),
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
