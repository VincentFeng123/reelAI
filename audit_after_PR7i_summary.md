# Mass clip audit — suffix=after_PR7i

- total rows: 70
- clean rows (whisper measured): 70
- error rows: 0

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 11 | 70 | 0.157 |
| ends_mid_sentence | 9 | 70 | 0.129 |
| starts_on_filler | 3 | 70 | 0.043 |
| **usable_clip_rate** | 46 | 70 | **0.657** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms median: 20.0
- end_precision_ms median: 30.0

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| lecture | 42 | 0.19 | 0.07 | 0.05 | 0.69 |
| speech | 28 | 0.11 | 0.21 | 0.04 | 0.61 |
