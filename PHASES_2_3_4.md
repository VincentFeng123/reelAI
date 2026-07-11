# Clipper redesign — Phases 2, 3, 4 (detailed implementation guide)

> Phase 1 (transcript-only structure-first) is **done and working**. This document specs
> Phases 2–4 in enough detail to implement without re-deriving the architecture. Read it
> alongside the approved plan `~/.claude/plans/ok-so-read-everything-cozy-nest.md` and the
> memory `clipper-structure-first-redesign.md`.

---

## 0. Phase-1 recap — the exact surfaces you integrate with

You are extending a working pipeline. Do **not** rewrite Phase 1; hook into these.

### Data model — `backend/pipeline/understand/models.py`
```python
SCHEMA_VERSION = 1                      # bump to invalidate structure.json cache when you change shapes

class VisualDependency(BaseModel):      # already exists, empty in Phase 1
    kind: str = "slide"                 # slide|equation|diagram|code|chart|demo
    keyframe_time: float = 0.0
    description: str = ""
    visual_event_id: Optional[str] = None   # link into Structure.visual_events

class VisualEvent(BaseModel):           # already exists, empty list in Phase 1
    event_id: str; start: float; end: float
    kind: str = "other"                 # slide|code|diagram|equation|chart|demo|text_overlay|face|action|other
    text: str = ""                      # on-screen text / OCR / equation, verbatim
    description: str = ""               # semantic caption
    confidence: float = 0.0

class Unit(BaseModel):
    unit_id: str; start: float; end: float; sentence_range: tuple[int,int]
    node_id: str; topic: str; role: str; role_domain: str
    summary: str; transcript: str
    claims: list[str]; concepts_introduced: list[str]; concepts_required: list[str]
    equations: list[str]; references: list[Reference]
    visual_dependencies: list[VisualDependency]     # <- Phase 2 populates
    speaker: Optional[str]                          # <- Phase 3 populates
    source_confidence: float = 1.0                  # scalar 0..1
    warnings: list[str]

class Structure(BaseModel):
    schema_version: int; video_id: str; title: str; duration: float
    detection: DetectionResult
    content_map: ContentMap; units: list[Unit]; dependencies: DependencyGraph
    visual_events: list[VisualEvent] = []           # <- Phase 2 fills
    has_perception: bool = False                    # <- Phase 2 sets True
    degraded: list[str] = []                        # e.g. ["vision","diarization"]
    def units_by_id(self) -> dict[str, Unit]
    def visual_summary(self, start, end) -> str     # already scans visual_events overlapping [start,end]

save_structure(structure); load_structure(video_id) -> Optional[Structure]  # work/<id>/structure.json
```

### Build + orchestration you will modify
```python
# backend/pipeline/understand/build.py
def build_structure(video_id, transcript, sentences, adapter, detection, settings, progress) -> Structure:
    content_map = build_content_map(sentences, settings, sub(0.00, 0.30))
    units       = extract_units(sentences, content_map, adapter, settings, sub(0.30, 0.80))   # <- add perception arg
    deps        = build_dependency_graph(units, settings, sub(0.80, 1.00))
    return Structure(... visual_events=[], has_perception=False ...)                          # <- fill from perception

# backend/orchestrator.py
async def _run_full(job, transcript, video_id, sentences, run, emit):
    structure = load_structure(video_id) if config.STRUCTURE_CACHE else None
    if structure is None or not structure.units:
        adapter, detection = await run(select_adapter, transcript, job.settings)
        structure = await run(build_structure, video_id, transcript, sentences, adapter, detection, job.settings, emit("segmenting",42,68))
        ...
    # Phase 2 changes: download video + perceive BEFORE build_structure; pass perception in.
```

### Reusable infra
- `backend/config.py` — Phase 2/3 keys **already stubbed**: `SCENE_THRESHOLD, KEYFRAME_GRID_S, KEYFRAME_MIN_GAP_S, KEYFRAME_MAX, DHASH_HAMMING_DROP, OCR_ENGINE, VISION_ENGINE, VISION_BATCH, HF_TOKEN, DIARIZATION_MODEL, DIARIZATION_ENABLED, MULTIMODAL, OUTPUT_MODE`. Plus `WORK_DIR, FFMPEG_BIN, GEMINI_MODEL, GEMINI_API_KEY, TARGET_AUDIO_SR(16000), TARGET_AUDIO_CH(1)`.
- `backend/pipeline/download.py` — `download(url, settings, progress) -> {video_id, video_path, audio_path, title, duration}`; caches `work/<id>/video.mp4` + `audio.m4a` (16 kHz mono). `_extract_audio(video_path, audio_path)`.
- `backend/gemini_client.py` — `generate_json(system, user_str, pydantic_schema, temperature)` — **text only today**; you add `generate_json_mm`.
- `backend/llm.py` — `llm_json(system, user, schema, temperature=, est_tokens=)` provider switch (Gemini/Groq). Text passes use this.
- `backend/pipeline/cut.py` — `cut_clips(src, video_id, segments, settings, progress)` for `output_mode="cut"`.

### Verify anytime (no network; uses cached transcripts)
```bash
cd clips
PRECISE_BOUNDARIES=0 .venv/bin/python -m backend.cli "https://youtu.be/1bH_ukYn81c" "the derivative" full
.venv/bin/python -m backend.eval.run_eval 1bH_ukYn81c --topic "the derivative"
```
Providers in this env: `LLM_PROVIDER=gemini` (gemini-2.5-flash), `TRANSCRIBER=supadata`, `GEMINI_API_KEY` + `GROQ_API_KEY` set. Cached transcripts exist under `work/<id>/transcript.json`.

**Golden rule for all phases:** every new capability must `available()`-guard and degrade to Phase-1 behavior on any failure (missing dep/token/key/exception), recording the degradation in `Structure.degraded[]` and lowering `Unit.source_confidence`. Never hard-crash the job.

---

# PHASE 2 — Multimodal perception

**Goal:** give the pipeline eyes. Extract on-screen text / equations / diagrams / code from
the video's frames, attach them as `VisualEvent`s, let unit extraction record
`visual_dependencies`, and let the clip-only judge use on-screen text via
`Structure.visual_summary`. Also enable `output_mode="cut"` (render mp4s) now that the
video is downloaded.

**"Done" looks like:** on a physics lecture, `structure.visual_events` contains the board
equations (`"a = Δv/Δt"`) with timestamps; the acceleration-definition unit has a
`visual_dependency` pointing at that equation event; the judge for a clip that says "using
this equation" now sees `ON-SCREEN TEXT: [equation] a = Δv/Δt` and stops flagging
`visuals_insufficient`.

### 2.1 Dependencies

**No new pip install is required for the recommended path** (ffmpeg + Gemini-vision).
- Scene detection & keyframes: **ffmpeg** (already at `config.FFMPEG_BIN`).
- Vision/OCR of keyframes: **Gemini** via `google-genai` (already installed, 2.10.0).
- Perceptual dedup hash: **PIL + numpy** (already installed).

Optional local OCR (only if you set `OCR_ENGINE=easyocr`):
```bash
.venv/bin/pip install easyocr        # pulls a torch-based model; ~heavy. Default OCR_ENGINE="none".
```
Recommendation: ship with `OCR_ENGINE="none"` — Gemini-vision reads on-screen text
*semantically* (equations, handwriting, code) better than glyph OCR.

### 2.2 `gemini_client.py` — add multimodal (the one enabling edit)

Current `generate_json` builds `contents=user` (a string). Add a sibling that accepts a
list of `types.Part` (interleaved text + images). Verify the exact google-genai API with
Context7/docs if it errors — as of 2.10.0 this is correct:

```python
# append to backend/gemini_client.py
from google.genai import types

def generate_json_mm(system: str, parts: list, schema, temperature: float = 0.2) -> str:
    """Multimodal structured JSON. `parts` = list[types.Part] (text + image/video bytes)."""
    client = get_client()

    def _cfg(thinking: bool):
        kw = dict(system_instruction=system, response_mime_type="application/json",
                  response_schema=schema, temperature=temperature, max_output_tokens=8192)
        if not thinking:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kw)

    def _call():
        try:
            r = client.models.generate_content(model=config.GEMINI_MODEL, contents=parts, config=_cfg(False))
        except Exception:
            r = client.models.generate_content(model=config.GEMINI_MODEL, contents=parts, config=_cfg(True))
        return r.text
    # reuse the same retry loop as generate_json (copy its for-loop over BACKOFF_MAX_RETRIES)
    ...

def image_part(jpeg_bytes: bytes):
    from google.genai import types
    return types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

def text_part(text: str):
    from google.genai import types
    return types.Part.from_text(text=text)
```

Wire `backend/llm.py` optionally to expose an mm helper, or call `gemini_client.generate_json_mm`
directly from `vision.py` (vision is Gemini-only; Groq has no image input here).

### 2.3 `backend/pipeline/understand/scenes.py` (new) — ffmpeg scenes + keyframes

```python
"""Scene-cut detection + keyframe extraction via ffmpeg (zero new deps)."""
import re, subprocess
from pathlib import Path
from ... import config
from .models import Scene   # ADD a Scene model to models.py (below)

def _scene_times(video_path: str) -> list[float]:
    """Parse showinfo pts_time for frames where the scene score exceeds the threshold."""
    cmd = [config.FFMPEG_BIN, "-nostdin", "-hide_banner", "-i", video_path,
           "-filter:v", f"select='gt(scene,{config.SCENE_THRESHOLD})',showinfo", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr or "")]

def _uniform_grid(duration: float, step: float) -> list[float]:
    t, out = 0.0, []
    while t < duration:
        out.append(round(t, 2)); t += step
    return out

def _dhash(path: str) -> int:
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("L").resize((9, 8))
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")

def detect_and_extract(video_path: str, video_id: str, duration: float, progress=None) -> list[Scene]:
    times = sorted(set(_scene_times(video_path)) | set(_uniform_grid(duration, config.KEYFRAME_GRID_S)))
    # enforce min gap, cap count
    kept = []
    for t in times:
        if not kept or (t - kept[-1]) >= config.KEYFRAME_MIN_GAP_S:
            kept.append(t)
    kept = kept[: config.KEYFRAME_MAX]
    kf_dir = config.WORK_DIR / video_id / "keyframes"; kf_dir.mkdir(parents=True, exist_ok=True)
    scenes, last_hash = [], None
    for i, t in enumerate(kept):
        out = kf_dir / f"kf_{i:04d}.jpg"
        subprocess.run([config.FFMPEG_BIN, "-nostdin", "-y", "-ss", f"{t:.3f}", "-i", video_path,
                        "-frames:v", "1", "-q:v", "3", str(out)], capture_output=True)
        if not out.exists():
            continue
        h = _dhash(str(out))
        if last_hash is not None and _hamming(h, last_hash) <= config.DHASH_HAMMING_DROP:
            out.unlink(missing_ok=True); continue          # near-identical to previous → drop
        last_hash = h
        end = kept[i + 1] if i + 1 < len(kept) else duration
        scenes.append(Scene(index=len(scenes), start=t, end=end, keyframe_time=t, keyframe_path=str(out)))
        if progress: progress((i + 1) / len(kept), f"Keyframes {i+1}/{len(kept)}")
    return scenes
```
**Add `Scene` to `models.py`:**
```python
class Scene(BaseModel):
    index: int; start: float; end: float; keyframe_time: float
    keyframe_path: Optional[str] = None
```

**Example output** for a 662 s lecture (~30 kept keyframes after dedup):
```json
[{"index":0,"start":0.0,"end":15.0,"keyframe_time":0.0,"keyframe_path":"work/ID/keyframes/kf_0000.jpg"},
 {"index":1,"start":121.5,"end":148.4,"keyframe_time":121.5,"keyframe_path":".../kf_0007.jpg"}]
```

### 2.4 `backend/pipeline/understand/vision.py` (new) — Gemini keyframe captioning (PRIMARY visual pass)

```python
"""Describe keyframes with Gemini: classify what's on screen + extract on-screen text."""
from pydantic import BaseModel, Field
from ... import config
from .models import Scene, VisualEvent

class VisionItem(BaseModel):
    index: int
    kind: str = "other"          # slide|code|diagram|equation|chart|demo|text_overlay|face|action|other
    text: str = ""               # verbatim on-screen text / equations / code
    description: str = ""        # one-line semantic caption
    confidence: float = 0.5

class VisionBatch(BaseModel):
    items: list[VisionItem] = Field(default_factory=list)

VISION_SYSTEM = (
    "You are a visual analyst for a video. For each numbered keyframe (with its timestamp and the "
    "nearby transcript), classify what is ON SCREEN and extract any on-screen text, equations, or "
    "code VERBATIM. Kinds: slide, code, diagram, equation, chart, demo, text_overlay, face, action, "
    "other. If the frame is just a talking head with nothing on screen, kind='face' and text=''. "
    "Output one item per keyframe index. Structured output only."
)

def available() -> bool:
    return bool(config.GEMINI_API_KEY)

def describe_keyframes(scenes: list[Scene], nearby_text_fn, progress=None) -> list[VisualEvent]:
    if not available() or not scenes:
        return []
    from .. import understand  # noqa
    from ...gemini_client import generate_json_mm, image_part, text_part
    events: list[VisualEvent] = []
    B = config.VISION_BATCH
    batches = [scenes[i:i+B] for i in range(0, len(scenes), B)]
    for bi, batch in enumerate(batches):
        parts = [text_part("Analyze these keyframes:")]
        for s in batch:
            near = nearby_text_fn(s.keyframe_time)[:240]
            parts.append(text_part(f"[keyframe index={s.index} t={s.keyframe_time:.1f}s] transcript: {near}"))
            with open(s.keyframe_path, "rb") as f:
                parts.append(image_part(f.read()))
        try:
            raw = generate_json_mm(VISION_SYSTEM, parts, VisionBatch, temperature=0.1)
            vb = VisionBatch.model_validate_json(raw)
        except Exception:
            vb = VisionBatch()
        by_idx = {it.index: it for it in vb.items}
        for s in batch:
            it = by_idx.get(s.index)
            if not it or (not it.text and it.kind in ("face", "other")):
                continue
            events.append(VisualEvent(event_id=f"ve_{len(events):04d}", start=s.start, end=s.end,
                                      kind=it.kind, text=it.text, description=it.description,
                                      confidence=it.confidence))
        if progress: progress((bi+1)/len(batches), f"Vision {bi+1}/{len(batches)}")
    return events
```
`nearby_text_fn(t)` = a closure over sentences returning the transcript around time `t`
(e.g. the text of sentences within ±8 s). Build it in `perceive.py`.

**Example VisualEvent output** (physics lecture, acceleration section):
```json
[{"event_id":"ve_0007","start":121.5,"end":148.4,"kind":"equation",
  "text":"a = Δv / Δt","description":"Definition of acceleration written on the whiteboard","confidence":0.95},
 {"event_id":"ve_0012","start":355.6,"end":406.2,"kind":"chart",
  "text":"velocity vs time; slope highlighted","description":"v–t graph with the slope region shaded","confidence":0.9}]
```

### 2.5 `backend/pipeline/understand/ocr.py` (optional, default off)

```python
def available(engine: str) -> bool: ...
def ocr_keyframes(scenes, progress=None) -> list["OcrBlock"]: ...   # returns [] if OCR_ENGINE=="none"
```
If enabled (`OCR_ENGINE="easyocr"`), OCR each keyframe and merge text into the nearest
`VisualEvent` (append to `.text` if the vision pass missed it). Keep it fully guarded.

### 2.6 `backend/pipeline/understand/perceive.py` (new) — orchestrates perception + caches

```python
"""Run perception → visual events. Cached to work/<id>/perception.json (keyed by video_id)."""
import json
from pathlib import Path
from ... import config
from .models import Perception   # ADD to models.py
from . import scenes as scenes_mod
from . import vision

def _cache_path(video_id): return config.WORK_DIR / video_id / "perception.json"

def perceive(video_path: str, video_id: str, transcript: dict, sentences, settings, progress=None) -> Perception:
    cache = _cache_path(video_id)
    if config.STRUCTURE_CACHE and cache.exists():
        try: return Perception.model_validate_json(cache.read_text())
        except Exception: pass
    duration = float(transcript.get("duration", 0.0) or (sentences[-1].end if sentences else 0.0))
    degraded = []

    def nearby_text(t):
        return " ".join(s.text for s in sentences if s.start <= t + 8 and s.end >= t - 8)

    try:
        scs = scenes_mod.detect_and_extract(video_path, video_id, duration, lambda f,m="": progress and progress(0.5*f, m))
    except Exception:
        scs, degraded = [], degraded + ["scenes"]
    try:
        ves = vision.describe_keyframes(scs, nearby_text, lambda f,m="": progress and progress(0.5+0.5*f, m))
        if not ves and scs: degraded.append("vision")
    except Exception:
        ves, degraded = [], degraded + ["vision"]

    per = Perception(video_id=video_id, scenes=scs, visual_events=ves, degraded=degraded)
    try: cache.write_text(per.model_dump_json())
    except Exception: pass
    return per
```
**Add `Perception` to `models.py`:**
```python
class Perception(BaseModel):
    video_id: str
    scenes: list[Scene] = []
    visual_events: list[VisualEvent] = []
    diarization: list["SpeakerTurn"] = []     # Phase 3
    degraded: list[str] = []
```

### 2.7 `units.py` — feed visual context in, populate `visual_dependencies`

Change signature to accept perception and inject overlapping visual events per topic:
```python
def extract_units(sentences, content_map, adapter, settings, progress=None, perception=None):
    ...
    for node in topics:
        a, b = node.sentence_range
        vis = ""
        if perception:
            span0, span1 = sentences[a].start, sentences[b].end
            vis_events = [ve for ve in perception.visual_events if ve.end >= span0 and ve.start <= span1]
            vis = "\n".join(f"[{ve.kind} @ {ve.start:.0f}s] {ve.text or ve.description}" for ve in vis_events)[:1500]
        user = (f"TOPIC: {node.title}\n"
                + (f"ON-SCREEN DURING THIS TOPIC:\n{vis}\n\n" if vis else "")
                + f"TRANSCRIPT ...:\n{rendered}\n\nSegment into atomic units.")
        ...
```
Add to the units LLM schema (`UnitLLM`) a `visual_dependencies` field:
```python
class VisDepLLM(BaseModel):
    kind: str = "slide"; on_screen_text: str = ""; description: str = ""
class UnitLLM(BaseModel):
    ...; visual_dependencies: list[VisDepLLM] = Field(default_factory=list)
```
And in the system prompt add: *"If a unit refers to something shown on screen (an equation,
diagram, code, chart), add a visual_dependency naming the on_screen_text or description."*

**Post-process linker** (in `units.py` after building each Unit): match each emitted
`visual_dependency` to the nearest `VisualEvent` by time/text and set `visual_event_id`:
```python
def _link_visual(dep, unit_start, unit_end, perception):
    best, best_score = None, 0.0
    for ve in (perception.visual_events if perception else []):
        if ve.end < unit_start - 5 or ve.start > unit_end + 5: continue
        # crude text overlap; use rapidfuzz.partial_ratio(dep.on_screen_text, ve.text) if present
        score = 1.0 if dep.kind == ve.kind else 0.5
        if score > best_score: best, best_score = ve, score
    return best.event_id if best else None
```
Also **raise `source_confidence`** for units whose visual dependency matched a real event,
and set `source_confidence.visual` accordingly (or keep scalar: `min(1.0, 0.8 + 0.2*matched)`).

### 2.8 `build.py` + `orchestrator.py` wiring

`build_structure` gains a `perception` param and fills the Structure:
```python
def build_structure(video_id, transcript, sentences, adapter, detection, settings, progress=None, perception=None):
    content_map = build_content_map(sentences, settings, sub(0.00, 0.25))
    units       = extract_units(sentences, content_map, adapter, settings, sub(0.25, 0.75), perception)
    deps        = build_dependency_graph(units, settings, sub(0.75, 1.00))
    return Structure(..., visual_events=(perception.visual_events if perception else []),
                     has_perception=bool(perception and perception.visual_events),
                     degraded=(perception.degraded if perception else []))
```

`orchestrator._run_full` — download the video and perceive when multimodal is on:
```python
async def _run_full(job, transcript, video_id, sentences, run, emit):
    structure = load_structure(video_id) if config.STRUCTURE_CACHE else None
    if structure is None or not structure.units:
        want_mm = (job.settings.get("multimodal") if job.settings.get("multimodal") is not None
                   else config.MULTIMODAL) and str(job.settings.get("analysis_profile", config.ANALYSIS_PROFILE)) == "full"
        perception = None
        if want_mm:
            try:
                registry.publish(job, ProgressEvent("perceiving", 30.0, "Downloading video for perception…"))
                dl = await run(download, job.url, job.settings, emit("perceiving", 30, 40))
                registry.set_meta(job, title=dl.get("title",""))
                from .pipeline.understand.perceive import perceive
                perception = await run(perceive, dl["video_path"], video_id, transcript, sentences, job.settings, emit("perceiving", 40, 55))
            except Exception:
                perception = None    # degrade to transcript-only
        adapter, detection = await run(select_adapter, transcript, job.settings)
        structure = await run(build_structure, video_id, transcript, sentences, adapter, detection, job.settings, emit("segmenting",55,72), perception)
        ...
```
**Note:** in `TRANSCRIBER=supadata` mode there's no `video.mp4` yet — the `download()` call
above fetches it (the video, not just audio) once, cached. This is the point where the
"full multimodal path always downloads the video" requirement kicks in.

`output_mode="cut"` — after boundaries, if `job.settings["output_mode"]=="cut"` and you have
`dl["video_path"]`, call `cut_clips(video_path, video_id, clips_spec, settings, emit(...))`
instead of `_build_embed_clips`. Keep `embed` as default.

### 2.9 Progress bands (update)
`ingest 0–30 · perceiving 30–55 · segmenting 55–72 · assembling 72–90 · refining 90–96 · sequencing 96–100`.
Frontends only render `pct`/`message`, so new stage names are safe.

### 2.10 Testing Phase 2
1. Pick a **real** lecture with board/slides (not one of the cached-transcript-only IDs — you need the actual video). Set `TRANSCRIBER=faster_whisper` (word-level) or keep supadata for transcript but the perceive path needs `video.mp4`.
2. `python -m backend.cli "<url>" "<topic>" full` — confirm `structure.visual_events` is non-empty, some units have `visual_dependencies` with `visual_event_id` set, and clip judge messages no longer say `visuals_insufficient` for equation clips.
3. Set `OUTPUT_MODE=cut` and confirm `.mp4`s render under `output/<id>/`.
4. Confirm degrade: rename ffmpeg / unset `GEMINI_API_KEY` → pipeline still returns clips (transcript-only), `structure.degraded` lists `vision`/`scenes`.

### 2.11 Pitfalls
- **ffmpeg `select=scene` is slow on long videos** (decodes everything). Cap with `KEYFRAME_MAX`; for >1 h videos, prefer the uniform grid + fewer scene samples, or add `-skip_frame nokey` to only inspect keyframes.
- **Gemini vision cost**: batch (`VISION_BATCH=8`) and dedup by dhash *before* the vision call. ~30–50 Gemini calls for a 1 h lecture is fine.
- **`generate_json_mm` API drift**: if `Part.from_bytes` errors, check google-genai docs via Context7 — the constructor arg names occasionally change across minor versions.
- **Keyframe seek accuracy**: `-ss` before `-i` is fast but can land on the wrong frame near cuts; acceptable for captioning. Use `-ss` after `-i` only if you need exactness.

---

# PHASE 3 — Speaker diarization

**Goal:** know *who* is speaking. Attach `speaker` to words → sentences → units. This
sharpens `continues` edges (same-speaker runs), enables speaker-change boundary cues, and
unlocks the **interview** adapter family (attribute question vs answer to different people).

**"Done" looks like:** on an interview, units alternate `speaker="SPEAKER_00"` (host) /
`"SPEAKER_01"` (guest); a `direct_answer` unit is a different speaker than its preceding
`question` unit; the interview adapter can anchor "answer with its question" clips.

### 3.1 Dependency + one-time setup (the fragile part — document clearly)
```bash
.venv/bin/pip install "pyannote.audio>=3.1"      # pulls torchaudio; torch already present
```
- Create a **HuggingFace token** (read scope) → set `HF_TOKEN=...` in `clips/.env`.
- **Accept the gated model terms** (once, while logged in) at:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0
- If either step is missing, diarization must degrade silently (single speaker).

### 3.2 `backend/pipeline/understand/diarize.py` (new)

```python
"""Speaker diarization via pyannote (optional; degrades to single speaker)."""
from ... import config
from .models import SpeakerTurn   # ADD to models.py

def available() -> bool:
    if not config.HF_TOKEN:
        return False
    try:
        import pyannote.audio  # noqa
        return True
    except Exception:
        return False

_pipeline = None
def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline
        _pipeline = Pipeline.from_pretrained(config.DIARIZATION_MODEL, use_auth_token=config.HF_TOKEN)
    return _pipeline

def diarize(audio_path: str, video_id: str, progress=None) -> list[SpeakerTurn]:
    if not available():
        return []
    try:
        ann = _get_pipeline()(audio_path)   # audio.m4a (16 kHz mono) from download()
    except Exception:
        return []
    turns = [SpeakerTurn(start=float(t.start), end=float(t.end), speaker=str(spk))
             for t, _, spk in ann.itertracks(yield_label=True)]
    return turns

def assign_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """Majority-overlap speaker for a [start,end] span."""
    best, best_ov = None, 0.0
    mid = (start + end) / 2
    for tr in turns:
        ov = max(0.0, min(end, tr.end) - max(start, tr.start))
        if ov > best_ov or (tr.start <= mid <= tr.end and best is None):
            best, best_ov = tr.speaker, ov
    return best
```
**Add to `models.py`:**
```python
class SpeakerTurn(BaseModel):
    start: float; end: float; speaker: str      # "SPEAKER_00"
```

### 3.3 Integrate into `perceive.py` and `units.py`
- In `perceive.py`, after scenes/vision:
```python
from . import diarize
turns = []
if settings.get("diarization", config.DIARIZATION_ENABLED):
    try: turns = diarize.diarize(audio_path, video_id, ...)
    except Exception: degraded.append("diarization")
per.diarization = turns
```
  (You'll need `audio_path` in `perceive()` — pass it from the orchestrator's `download()` result.)
- In `units.py`, after computing each unit's `[start,end]`, set
  `unit.speaker = diarize.assign_speaker(start, end, perception.diarization)` when turns exist.
- `dependencies.py` **already** uses `unit.speaker` in the `continues` edge rule — it starts
  working automatically once speakers are populated.

### 3.4 Interview adapter (needs Phase 3) — `backend/adapters/interview.py`
```python
from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec
E = ContractElement

class InterviewAdapter(BaseAdapter):
    domain = "interview"
    content_types = ("interview", "podcast", "talk_show")
    def _domain_role_specs(self):
        S = RoleSpec
        return {r.name: r for r in [
            S("question", "An interviewer asks a question.", facet="other", is_anchor=False),
            S("direct_answer", "The interviewee directly answers a question.", facet="other", is_anchor=True, anchor_priority=70),
            S("story_setup", "The speaker sets up a story/anecdote.", facet="other"),
            S("anecdote", "A personal story or example.", facet="other", is_anchor=True, anchor_priority=68),
            S("supporting_detail", "Detail elaborating an answer.", facet="other"),
            S("opinion", "A stated opinion or take.", facet="other", is_anchor=True, anchor_priority=60),
            S("counterargument", "A rebuttal or opposing view.", facet="other", is_anchor=True, anchor_priority=55),
        ]}
    def labeling_hints(self):
        return ("This is an interview/podcast. Attribute questions to the interviewer and answers to the "
                "guest (use the speaker labels). A 'direct_answer' unit should be paired with the 'question' "
                "that prompted it.")
    def _contracts(self):
        return {
          "direct_answer": CompletenessContract("direct_answer", (
              E("question", ("question",), "required", "before"),
              E("answer", ("direct_answer",), "required", "within"),
              E("elaboration", ("supporting_detail","opinion","anecdote"), "recommended", "after"),
          )),
          "anecdote": CompletenessContract("anecdote", (
              E("setup", ("story_setup","question"), "required", "before"),
              E("story", ("anecdote",), "required", "within"),
              E("payoff", ("opinion","supporting_detail","result"), "recommended", "after"),
          )),
        }
```
Register it in `adapters/__init__.py`: `register(InterviewAdapter())`.
`dependencies.py` should add a `answers` edge from a `direct_answer` unit to the nearest
prior `question` unit **by a different speaker** (extend the existing solution→prompt rule
analogously).

### 3.5 Testing Phase 3
- Run on a 2-speaker podcast clip. Confirm `unit.speaker` alternates and `direct_answer`
  units differ in speaker from their `question`.
- Confirm degrade: unset `HF_TOKEN` → all `unit.speaker=None`, `structure.degraded=["diarization"]`, pipeline still works.

### 3.6 Pitfalls
- pyannote is **slow** (~0.2–0.5× realtime) and downloads models on first use. Cache the
  pipeline (module global). Run it in the executor thread (it's blocking).
- The gated-model acceptance is the #1 support issue — surface a clear message in `/health`
  (add a `diarization: available()` field) and in `structure.degraded`.

---

# PHASE 4 — Adapter breadth + evaluation

**Goal:** cover the remaining genres (each is one class on the proven `BaseAdapter`
interface) and build a real golden set so you can tune contracts/anchors against metrics
instead of guessing.

### 4.1 Adapters to add (one file each in `backend/adapters/`, register in `__init__.py`)

Pattern for every adapter: subclass `BaseAdapter`, set `domain` + `content_types`, add
`_domain_role_specs()` (roles with `facet`, `is_anchor`, `anchor_priority`),
`labeling_hints()`, and `_contracts()` (ordered before→within→after elements). The generic
contracts are inherited; override by key.

**`coding.py` (full example):**
```python
class CodingAdapter(BaseAdapter):
    domain = "coding"; content_types = ("coding","programming","software")
    def _domain_role_specs(self):
        S = RoleSpec
        return {r.name: r for r in [
            S("requirement","What we're building / the spec.",facet="overview"),
            S("code_explanation","Explaining a piece of code.",facet="other",is_anchor=True,anchor_priority=58),
            S("implementation","Writing/showing code that implements something.",facet="application",is_anchor=True,anchor_priority=72),
            S("error","An error/bug is shown.",facet="other"),
            S("debugging_step","A step diagnosing/fixing a bug.",facet="application",is_anchor=True,anchor_priority=66),
            S("output_validation","Running it / showing the output is correct.",facet="worked_example"),
        ]}
    def labeling_hints(self):
        return ("This is a coding video. Treat function/class/API names as concepts. An 'implementation' "
                "spans from the requirement through the code to the output check.")
    def _contracts(self):
        return {
          "implementation": CompletenessContract("implementation", (
              E("requirement",("requirement","setup"),"required","before"),
              E("code",("implementation","code_explanation"),"required","within",repeatable=True),
              E("output",("output_validation","result"),"required","after"),
          )),
          "debugging_step": CompletenessContract("debugging_step", (
              E("error",("error",),"required","before"),
              E("fix",("debugging_step","implementation"),"required","within",repeatable=True),
              E("output",("output_validation","result"),"required","after"),
          )),
        }
```
**`recipe.py`, `tutorial.py`, `debate.py`, `review.py`, `story.py`, `sports.py`, `news.py`** —
same shape. Role packs & anchors (from spec §11):
- **recipe**: `ingredients, prep_step, cooking_action, plating, tasting`; anchor `cooking_action`; contract ingredients→action(within,repeatable)→plating.
- **tutorial**: reuse universal `procedure/demonstration`; contract goal→steps→result.
- **debate**: `position, argument, evidence, rebuttal`; anchor `argument`; contract position(before)→argument(within)→evidence(within)→rebuttal(after,rec).
- **review (product_review)**: `product_intro, criterion, test, finding, verdict`; anchor `finding`/`verdict`; contract criterion(before)→test(within)→finding(within)→verdict(after,rec).
- **story**: `setup, rising_action, climax, resolution`; anchor `climax`; contract setup(before)→rising_action(within)→climax(within)→resolution(after,rec).
- **sports**: `situation, play, outcome, analysis`; anchor `analysis`/`play`; contract situation(before)→play(within)→outcome(within)→analysis(after).
- **news**: `lede, detail, context, consequence`; anchor `lede`; contract lede(within)→detail(within,rep)→context(after,rec).

Update `adapters/detect.py::CONTENT_TYPE_TO_DOMAIN` — it already maps most of these to the
right keys; just ensure every new adapter's `domain` has a mapping entry.

### 4.2 Golden set — author real labels

Create `backend/eval/golden/<video_id>.json` for ~5–10 diverse videos. Schema (loader in
`eval/golden.py` already parses this; all fields optional except `video_id`):
```json
{
  "video_id": "1bH_ukYn81c",
  "url": "https://youtu.be/1bH_ukYn81c",
  "domain": "lecture",
  "content_type": "math",
  "topics": ["the derivative"],
  "reference_concepts": ["slope","limit","derivative"],
  "units": [
    {"start": 315.1, "end": 358.1, "role": "example_setup",
     "concepts_introduced": ["derivative of x^2 example"], "concepts_required": ["slope"], "is_anchor": false},
    {"start": 493.7, "end": 543.1, "role": "result",
     "concepts_introduced": [], "concepts_required": ["slope","limit"], "is_anchor": false}
  ],
  "anchors": [
    {"anchor_role": "result", "start": 315.1, "end": 543.1,
     "required_elements_present": ["problem_statement","solution_steps","result"],
     "prerequisites": ["slope"], "must_understand_without_source": true}
  ]
}
```
**Labeling helper (write this):** a script `python -m backend.eval.make_golden <video_id>`
that runs `build_structure` and dumps a golden *skeleton* (predicted units → JSON) for a
human to correct. This makes labeling fast:
```python
# backend/eval/make_golden.py — dump structure.units into the golden schema for hand-editing
```

### 4.3 Metrics already implemented (`eval/metrics.py`) + ones to add
Present: `comprehension` (headline), `ends_on_period_rate`, `unresolved_reference_rate`,
`grounding_ok_rate`, `role_accuracy`, `anchor_recall`.
Add (spec §14): `context_complete_rate` (contract required elements present among gold roles
in-span), `prerequisite_gap_rate`, `worked_example_completeness` (ordered elements present),
`visual_completeness` (Phase 2: referenced equation/diagram on-screen in span),
`sequence_coherence` (no forward-reference across the chronological order without a card).

### 4.4 Tuning loop (the point of Phase 4)
```bash
.venv/bin/python -m backend.eval.run_eval           # all cached videos
```
Read the aggregate table; when `comprehension_rate` is low, inspect which failure_reasons
dominate (add a `--verbose` flag to dump per-clip judge failures) and adjust: anchor
priorities (`RoleSpec.anchor_priority`), contract element `necessity`, closure budgets
(`config.CLOSURE_MAX_*`), or `JUDGE_MIN_SCORE`. **Do not lower the judge bar to inflate
yield** — prefer fixing closure/labeling so real clips pass.

### 4.5 Testing Phase 4
- Add one golden per adapter domain; confirm `detect_content_type` routes correctly and
  `role_accuracy`/`anchor_recall` compute.
- Confirm each new adapter yields sensible clips on a matching video via `cli`.

---

# HANDOFF PROMPT (paste into a fresh Claude Code session, cwd = the practice repo)

> I'm continuing a multi-phase redesign of the video clipper in `clips/`. Phase 1
> (transcript-only structure-first pipeline: `backend/adapters/`, `backend/pipeline/understand/`,
> `backend/pipeline/assemble/`, `backend/llm.py`, `backend/roles.py`, `backend/eval/`, rewritten
> `orchestrator.py`) is **done and working**. Do NOT rebuild Phase 1.
>
> Read these first, in order: `clips/PHASES_2_3_4.md` (the detailed implementation guide),
> the approved plan `~/.claude/plans/ok-so-read-everything-cozy-nest.md`, and the memory note
> `clipper-structure-first-redesign.md`. Then skim the actual Phase-1 code you'll extend:
> `clips/backend/pipeline/understand/build.py`, `.../understand/units.py`,
> `clips/backend/pipeline/understand/models.py`, `clips/backend/orchestrator.py` (`_run_full`),
> and `clips/backend/gemini_client.py`.
>
> **Implement Phase 2 (multimodal perception)** exactly as specified in `PHASES_2_3_4.md §PHASE 2`:
> add `generate_json_mm` to `gemini_client.py`; create `understand/scenes.py`, `understand/vision.py`,
> `understand/perceive.py`, optional `understand/ocr.py`; add `Scene`/`Perception` models;
> thread `perception` through `build_structure` and `units.py` (populate `visual_dependencies` +
> link `visual_event_id`); wire the download+perceive branch into `orchestrator._run_full`; enable
> `output_mode="cut"`. Everything must `available()`-guard and degrade to Phase-1 behavior
> (record in `Structure.degraded`, never hard-crash).
>
> Constraints: `LLM_PROVIDER=gemini`, `TRANSCRIBER=supadata`, Python 3.12 venv at `clips/.venv`.
> The recommended visual path needs NO new pip installs (ffmpeg + Gemini-vision). Verify google-genai
> multimodal API via Context7 if `Part.from_bytes` errors. Test with `PRECISE_BOUNDARIES=0
> .venv/bin/python -m backend.cli "<real lecture url with slides/board>" "<topic>" full` and confirm
> `structure.visual_events` is populated and equation clips stop failing `visuals_insufficient`.
> Keep the clip-dict contract intact (`n,video_id,facet,start,end,cut_end,path,embed_url` + optional
> `role,title,context_card,sequence_index,prerequisite_clips`).
>
> When Phase 2 is verified end-to-end, continue to Phase 3 (diarization) then Phase 4 (adapter
> breadth + golden-set eval) per the same doc. Use a plan + todo list, test each phase before
> moving on, and update the `clipper-structure-first-redesign` memory as phases complete.
> Note: start a fresh session per video transcript so context doesn't bleed between videos.
