# Mass clip audit — suffix=boundary_a3_first9_startonly

- total rows: 38
- clean rows (whisper measured): 38
- error rows: 0

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 11 | 38 | 0.289 |
| ends_mid_sentence | 16 | 38 | 0.421 |
| starts_on_filler | 0 | 38 | 0.000 |
| **usable_clip_rate** | 5 | 38 | **0.132** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms median: 85.0
- end_precision_ms median: 374.5

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| lecture | 23 | 0.17 | 0.48 | 0.00 | 0.13 |
| speech | 15 | 0.47 | 0.33 | 0.00 | 0.13 |
