# Mass clip audit — suffix=tier1_smoke

- total rows: 21
- clean rows (whisper measured): 21
- error rows: 0

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 11 | 21 | 0.524 |
| ends_mid_sentence | 3 | 21 | 0.143 |
| starts_on_filler | 0 | 21 | 0.000 |
| **usable_clip_rate** | 5 | 21 | **0.238** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms median: 90.0
- end_precision_ms median: 260.0

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| lecture | 21 | 0.52 | 0.14 | 0.00 | 0.24 |
