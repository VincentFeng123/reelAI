# Mass clip audit — suffix=after_A1

- total rows: 70
- clean rows (whisper measured): 70
- error rows: 0

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 13 | 70 | 0.186 |
| ends_mid_sentence | 18 | 70 | 0.257 |
| starts_on_filler | 5 | 70 | 0.071 |
| **usable_clip_rate** | 37 | 70 | **0.529** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms median: 40.0
- end_precision_ms median: 40.0

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| lecture | 42 | 0.21 | 0.10 | 0.10 | 0.64 |
| speech | 28 | 0.14 | 0.50 | 0.04 | 0.36 |
