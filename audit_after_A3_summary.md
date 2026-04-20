# Mass clip audit — suffix=after_A3

- total rows: 166
- clean rows (whisper measured): 166
- error rows: 0

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 72 | 166 | 0.434 |
| ends_mid_sentence | 75 | 166 | 0.452 |
| starts_on_filler | 41 | 166 | 0.247 |
| **usable_clip_rate** | 23 | 166 | **0.139** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms median: 130.0
- end_precision_ms median: 204.0

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| auto_caption_only | 96 | 0.34 | 0.60 | 0.41 | 0.07 |
| lecture | 42 | 0.52 | 0.10 | 0.02 | 0.36 |
| speech | 28 | 0.61 | 0.46 | 0.04 | 0.04 |
