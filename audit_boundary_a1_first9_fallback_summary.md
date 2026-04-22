# Mass clip audit — suffix=boundary_a1_first9_fallback

- total rows: 35
- clean rows (whisper measured): 35
- error rows: 0

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 4 | 35 | 0.114 |
| ends_mid_sentence | 18 | 35 | 0.514 |
| starts_on_filler | 0 | 35 | 0.000 |
| **usable_clip_rate** | 13 | 35 | **0.371** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms median: 30.0
- end_precision_ms median: 60.0

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| lecture | 21 | 0.00 | 0.48 | 0.00 | 0.52 |
| speech | 14 | 0.29 | 0.57 | 0.00 | 0.14 |
