"""VID4 eval columns for the edge probe (advisory rates). Offline, pure functions."""
from __future__ import annotations

import math

import backend.eval.metrics as metrics


def test_edge_clean_rates_nan_when_no_probe_ran():
    # No spec carries an edge verdict (the default — the probe is OFF and eval never runs it).
    sr, er = metrics.edge_clean_rates([{"start": 0.0, "end": 30.0}, {"start": 40.0, "end": 70.0}])
    assert math.isnan(sr) and math.isnan(er)
    sr0, er0 = metrics.edge_clean_rates([])
    assert math.isnan(sr0) and math.isnan(er0)


def test_edge_clean_rates_compute_over_probed_specs():
    specs = [
        {"starts_clean_audio": True, "ends_clean_audio": True},
        {"starts_clean_audio": False, "ends_clean_audio": True},
        {"starts_clean_audio": True, "ends_clean_audio": False},
        {"start": 0.0, "end": 10.0},                       # unprobed → excluded from both rates
    ]
    sr, er = metrics.edge_clean_rates(specs)
    assert sr == 2 / 3 and er == 2 / 3                      # 3 probed; 2 clean starts, 2 clean ends


def test_edge_clean_rates_partial_verdict_counted_per_field():
    # a spec may carry only one of the two booleans; each rate counts only its own field.
    specs = [{"starts_clean_audio": True}, {"ends_clean_audio": False}]
    sr, er = metrics.edge_clean_rates(specs)
    assert sr == 1.0 and er == 0.0


def test_edge_columns_present_and_null_in_measure_when_no_probe():
    # The run_eval column wiring emits null (None) for the advisory rates on an un-probed run.
    import backend.eval.run_eval as R
    sr, er = metrics.edge_clean_rates([{"start": 0.0, "end": 5.0}])
    assert R._round_nan(sr) is None and R._round_nan(er) is None
