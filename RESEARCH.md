# Clip-comprehension research → adoptable changes

Deep-research synthesis (2025-07) on improving the structure-first clipper's self-contained-clip
quality. Targets: comprehension ~44%, unresolved-reference 0.81, prerequisite-gap 0.81, visual-heavy
0-clip videos, noisy run-to-run judge scores.

## How to read this doc (confidence)
The deep-research harness decomposed the question into **5 angles**, fetched 25 sources, extracted
**118 claims**, adversarially verified the **top 25** (3 independent skeptics each; a claim needs
2/3 refutes to die), and synthesized **13 findings**.
- **✓ VERIFIED** = survived 3-vote verification (23 confirmed, 2 killed). High/medium confidence.
- **○ LEAD** = fetched but dropped before verification (top-25 budget / novelty filter). Plausible,
  NOT independently checked — treat as a pointer, verify before adopting.

The literature **validates the architecture** (text-first, understand-then-assemble, OCR as the
high-value visual signal). Concrete techniques exist for every stage.

---

## ⚠️ Headline: the comprehension metric is inflated (self-preference bias) — ✓ VERIFIED (high)
The same Gemini that authors units/context cards also judges the clips, and LLM judges favor their
own generations → leniency → inflated comprehension.
- Sources: G-Eval (EMNLP 2023, arXiv:2303.16634); Panickssery et al. "LLM Evaluators Recognize and
  Favor Their Own Generations" (NeurIPS 2024, arXiv:2404.13076); Self-Preference (arXiv:2410.21819).
- **Fix — SHIPPED & ACTIVE:** judge on a different model (`config.JUDGE_PROVIDER` / `JUDGE_MODEL`).
  Groq key is invalid (401), so default judges on a different **Gemini** model,
  `gemini-2.5-flash-lite` (author is `gemini-2.5-flash`), via the working key. Partial mitigation
  (same family). Stronger: `JUDGE_MODEL=gemini-2.5-pro` (rate-limited) or a valid Groq key
  (`JUDGE_PROVIDER=groq`, different family = strongest). Full fix needs a different family OR humans.

---

## Area 1 — Context closure (unresolved-ref 0.81 + prerequisite-gap 0.81)
- **Coreference — Maverick** — ✓ VERIFIED (high). ACL 2024, aclanthology.org/2024.acl-long.722.
  500M discriminative (DeBERTa-v3) resolver, SOTA OntoNotes, 170× faster than 13B autoregressive.
  Resolve "this/that/the previous equation" → a real antecedent unit; feeds reliable
  `refers_to`/`answers` edges + repair targeting. STATUS: **todo (Tier 2)**.
- **Bridge-concept prerequisite edges** — ✓ VERIFIED (high). EDM 2018, files.eric.ed.gov/fulltext/ED593223
  (bridge measure AUC 0.80/0.81 existence/direction, beats ExtendedOverlap 0.74). A concept introduced
  earlier that reappears later ⇒ a `requires` edge; computable from existing units, no dep.
  STATUS: **SHIPPED** (`dependencies._bridge_edges`, `BRIDGE_PREREQ_EDGES`, cap `BRIDGE_MAX_PER_UNIT=6`).
  Caveat: evidence is one 74-concept course — heuristic, not gospel.
- Heavier alternatives — ✓ VERIFIED: GNN node-attention over a heterogeneous graph (CIKM 2023,
  10.1145/3583780.3614761); unsupervised transcript+visual extraction (AIED 2022, LNCS 13356).

## Area 2 — LLM-as-judge reliability
- **Determinism** — ✓ VERIFIED (high): temp 0 + caching off ⇒ >95% same-verdict; ~5% residual flips
  inherent (Stureborg 2024 arXiv:2405.01724; arXiv:2606.19544). Already temp 0. Average eval over runs.
- **Determinism ≠ validity** (consistency-bias paradox) — ✓ VERIFIED (medium, arXiv:2606.19544):
  Gemini 2.5 Flash test-retest 0.988 but position bias 0.125. Calibrate on humans.
- **Rubric + CoT + form-filling — G-Eval** — ✓ VERIFIED (high, arXiv:2303.16634, Spearman 0.514):
  explicit rubric → auto-CoT eval steps → fill a score form. Add a short reasoning field before the
  verdict. STATUS: **todo (Tier 2)**.
- **Scale** — ✓ VERIFIED (high, arXiv:2405.01724): 1-10 integer beats 1-100 (round-number bias),
  Kendall τ 0.428 vs 0.383. STATUS: **shipped** — judge emits `score_10` (1-10), normalized to the
  legacy 0-1 `score` internally (threshold 0.70 ≡ score_10 ≥ 7).
- **Bias-mitigated config S8** — ✓ VERIFIED (medium, arXiv:2604.23178): CoT + calibrated rubric +
  position-swap → 71% human agreement (Gemini 2.5 Flash). (Preprint; adopt the recipe, not its
  bias-attribution — see Refuted.)
- Caveat: judge numbers derive from summarization NLG, not the exact isolation criterion; newest
  sources (MVVP arXiv:2606.19544, S8 arXiv:2604.23178) are non-peer-reviewed preprints.

## Area 3 — Hierarchical segmentation (upgrade `content_map`)
- **TreeSeg** — ✓ VERIFIED (high, arXiv:2407.12028; github.com/AugmendTech/treeseg): training-free
  hierarchical binary-tree segmentation via embeddings + divisive clustering → native
  chapter→topic→subtopic. Uses installed sentence-transformers. STATUS: **todo (Tier 2)** — would
  replace the current heuristic 3-level map + LLM topic split.
- **Text dominates, multimodal refines** — ✓ VERIFIED (high, Alibaba ACL 2025, arXiv:2408.00365):
  transcript is the most informative modality (+25 BS@30 over visual-only); scene/OCR add ~+5.
- **Don't trust one LLM prompt for boundaries** — ✓ VERIFIED (high, EMNLP 2023 arXiv:2310.11772;
  SegNSP 2026 arXiv:2601.03474): fine-tuned/embedding segmenters beat zero-shot Gemini (Pk 0.14 vs 0.31).
- **VAD-based variable segments + coarse→fine** — ✓ VERIFIED (medium, NCA 2026,
  10.1007/s00521-025-11740-2): Silero VAD variable-length segments; split non-speech/speech, then
  refine by teaching-activity type. Evaluate boundaries with Pk / WindowDiff.

## Area 4 — Visual-dependent content (the 8uNod 0-clip class)
- **OCR is the highest-value visual modality** — ✓ VERIFIED (high, WACV 2023, arXiv:2210.16644):
  OCR-only NMI 78.9 > appearance features; fuse transcript+OCR > transcript-only. `OCR_ENGINE="none"`
  by default today. STATUS: **todo (Tier 3)** — turn on OCR + fuse.
- **Nougat** — ○ LEAD (arXiv:2308.13418): ViT (~350M) OCRs rendered pages → Markdown/**LaTeX**, recovers math.
- **MaViLS** — ○ LEAD (arXiv:2409.16765): video↔slide alignment; OCR F1 0.76 > image 0.64 > audio 0.53.
- **Texo** — ○ LEAD (arXiv:2602.17189): **20M-param** formula-image → LaTeX (printed/handwritten/screen).
  Lightest-weight equation recovery — best first try for on-screen formulas.
- **GLM-OCR** — ○ LEAD (arXiv:2603.10910): 0.9B multimodal — text + formula + code + table from
  slides/screenshots (94.6 OmniDocBench). One model for the whole visual-dependent need.
- **Two-stage slide understanding** — ○ LEAD (arXiv:2201.08574): segment slide into regions
  (title/text/equation/figure/table) → specialized recognizers (Tesseract, DocFigure, equation model,
  TabStruct-Net). Blueprint for structured slide parsing.

## Area 5 — Pipeline engineering (the angle NOT in the original 4; all ○ LEAD)
- **Text-selection, not LLM timestamps** — ○ LEAD (imgly/videoclipper; an n8n Whisper→Gemini flow):
  the LLM selects transcript *text*, then map it back to Whisper word-level timestamps for boundaries;
  claimed more reliable than LLM-emitted timestamps. **This validates what the pipeline already does**
  (`select.py` / `anchor_quote`) — reassurance, not a new fix.
- **PreMind** — ○ LEAD (arXiv:2503.00162): VLM-enhanced PySceneDetect for slide videos (catch slide
  changes inside long shots, merge identical-slide segments), F1 ~98%. Relevant if pushing scene cuts.

---

## Open questions — researched, NO established answer found (design-your-own territory)
1. **Inline-vs-context-card decision + a formal closure stopping rule** — no source. The two-mode
   closure (inline-near / referential-far + span cap) is unmapped territory; tune empirically.
2. **"Unclippable transcript-only" threshold** — no established rule for when a clip *must* include
   the visual (the 8uNod problem).
3. **Grounding a reference to a specific on-screen entity + time window** (math→LaTeX + temporal
   alignment) — only OCR's aggregate value confirmed, not per-reference grounding.
4. **Judge validity on the actual criterion** — all judge findings are from summarization NLG, not
   "can a viewer understand this clip." The only real fix is a small human-labeled calibration set.

## Refuted (killed in verification — do NOT rely on)
- "Prereq extraction efficacy comes from multimodal signals" (1-2) — text alone suffices.
- "Position bias is negligible in modern judges" (1-2) — don't assume it's safe (matters if pairwise).

## Roadmap (impact × fit; status as of this session)
| Tier | Change | Hits | Status |
|---|---|---|---|
| 1 | Cross-model judge (self-preference) | trustworthy metric | **shipped & active** (JUDGE_MODEL=gemini-2.5-flash-lite) |
| 1 | Bridge-concept prerequisite edges | prerequisite-gap | **shipped** |
| 2 | Maverick coref pass | unresolved-ref | todo |
| 2 | TreeSeg content_map | boundary quality | **shipped** (CONTENT_MAP_ENGINE=treeseg default; embeddings segment, LLM only labels; partition byte-deterministic on real data; A/B 3vids×3runs: unresolved-ref/prereq-gap 0.72→0.28, comprehension 0.44→0.22 ± 0.19 = inconclusive at this sample, n_clips std unchanged — downstream unit/judge LLM noise dominates once boundaries are fixed) |
| 2 | Judge rubric+CoT, 1-10 scale | judge validity | **shipped** (reasoning-first CoT via schema order + anchored 1-10 bands + honest error verdict replacing the silent 0.7 outage pass; frozen-specs A/B: determinism preserved ±0.000, stricter at assembly 4→2 surviving clips, mean 0.483→0.400, comprehension flat 0.333, judge_error_rate 0.0) |
| 3 | OCR + Texo/GLM-OCR/Nougat for math | visual-heavy 0-clips | todo |
| 3 | Human calibration set + Pk/WindowDiff/κ | rigor (only real judge-validity fix) | todo |

## Source index (by angle)
- Context/prereq: 2303.16634, 2404.13076, aclanthology 2024.acl-long.722, ED593223 (EDM'18),
  10.1145/3583780.3614761 (CIKM'23), LNCS 13356 (AIED'22).
- Judge: 2405.01724, 2606.19544, 2303.16634, 2604.23178, 2410.21819, survey 2411.15594.
- Segmentation: 2407.12028 + AugmendTech/treeseg, 2408.00365, 2310.11772, 2601.03474,
  10.1007/s00521-025-11740-2.
- Visual: 2210.16644 (WACV'23), 2409.16765 (MaViLS), 2308.13418 (Nougat), 2602.17189 (Texo),
  2603.10910 (GLM-OCR), 2201.08574 (slide understanding).
- Pipeline: imgly/videoclipper, n8n flow, 2510.27106, 2503.00162 (PreMind).
