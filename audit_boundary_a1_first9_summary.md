# Mass clip audit — suffix=boundary_a1_first9

- total rows: 35
- clean rows (whisper measured): 0
- error rows: 35

**Statistical-significance note.** Aggregate delta across all 25 videos is the decision signal; per-content-type deltas are directional and require a ≥15 pp swing before reading anything into a single bucket.

**Continuous metrics caveat.** Whisper measures Whisper-refined boundaries — the noise floor is ~±50-100 ms. Do not publish absolute precision numbers; compare only against the prior run's CSV.

## Aggregate binary metrics

| metric | hits | total | rate |
|---|---|---|---|
| starts_mid_word | 0 | 0 | 0.000 |
| ends_mid_sentence | 0 | 0 | 0.000 |
| starts_on_filler | 0 | 0 | 0.000 |
| **usable_clip_rate** | 0 | 0 | **0.000** |

## Aggregate continuous metrics (compare across runs; noisy in absolute)

- start_precision_ms: n/a
- end_precision_ms: n/a

## Per-content-type rollup (directional; ≥15pp swing required)

| type | n | mid_word | mid_sent | filler | usable |
|---|---|---|---|---|---|
| lecture | 0 | - | - | - | - |
| speech | 0 | - | - | - | - |

## Error breakdown

- download_failed: 35
