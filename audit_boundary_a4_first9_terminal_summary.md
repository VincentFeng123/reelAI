# Mass clip audit — suffix=boundary_a4_first9_terminal

- total rows: 36
- clean rows (whisper measured): 36
- error rows: 0

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 7 | 36 | 0.194 |
| ends_mid_sentence | 2 | 36 | 0.056 |
| starts_on_filler | 0 | 36 | 0.000 |
| **usable_clip_rate** | 25 | 36 | **0.694** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms median: 145.0
- end_precision_ms median: 60.0

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| lecture | 22 | 0.09 | 0.00 | 0.00 | 0.82 |
| speech | 14 | 0.36 | 0.14 | 0.00 | 0.50 |
