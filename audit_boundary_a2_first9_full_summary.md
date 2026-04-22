# Mass clip audit — suffix=boundary_a2_first9_full

- total rows: 38
- clean rows (whisper measured): 38
- error rows: 0

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 10 | 38 | 0.263 |
| ends_mid_sentence | 16 | 38 | 0.421 |
| starts_on_filler | 4 | 38 | 0.105 |
| **usable_clip_rate** | 8 | 38 | **0.211** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms median: 90.0
- end_precision_ms median: 60.0

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| lecture | 23 | 0.13 | 0.48 | 0.17 | 0.26 |
| speech | 15 | 0.47 | 0.33 | 0.00 | 0.13 |
