# Mass clip audit — suffix=boundary_a6_subset_queryopen

- total rows: 14
- clean rows (whisper measured): 14
- error rows: 0

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 10 | 14 | 0.714 |
| ends_mid_sentence | 0 | 14 | 0.000 |
| starts_on_filler | 0 | 14 | 0.000 |
| **usable_clip_rate** | 4 | 14 | **0.286** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms median: 60.0
- end_precision_ms median: 40.0

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| lecture | 4 | 1.00 | 0.00 | 0.00 | 0.00 |
| speech | 10 | 0.60 | 0.00 | 0.00 | 0.40 |
