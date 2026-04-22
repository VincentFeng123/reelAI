# Mass clip audit — suffix=baseline

- total rows: 7
- clean rows (whisper measured): 7
- error rows: 0

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 4 | 7 | 0.571 |
| ends_mid_sentence | 0 | 7 | 0.000 |
| starts_on_filler | 0 | 7 | 0.000 |
| **usable_clip_rate** | 1 | 7 | **0.143** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms median: 200.0
- end_precision_ms median: 360.0

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| lecture | 7 | 0.57 | 0.00 | 0.00 | 0.14 |
