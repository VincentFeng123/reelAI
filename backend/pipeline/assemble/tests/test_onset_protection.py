"""The repair-trim lattice must never trim the clip's leading (onset) unit — trimming it
would advance the start past the setup/problem-read. Offline."""
from __future__ import annotations

from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import _protected_unit_ids
from backend.pipeline.understand.models import Unit
from backend.adapters import get_adapter


def _unit(uid, start, end, role):
    return Unit(unit_id=uid, start=start, end=end, sentence_range=(int(start), int(end)),
                role=role, summary="", transcript="")


def test_leading_onset_unit_is_protected_from_trim():
    """Brief's canonical fixture — kept verbatim.

    NOTE: for this fixture the onset unit (u0, role=example_setup) happens to be
    covered by the 'result' contract's required 'problem_statement' element, so this
    test was already GREEN before the onset-protection implementation.  It serves as a
    permanent regression guard.  See test_onset_protection_without_contract_coverage
    below for the true gap demonstration (RED→GREEN).
    """
    units = [
        _unit("u0", 0, 4, "example_setup"),   # the problem-read (onset)
        _unit("u1", 4, 8, "worked_step"),
        _unit("u2", 8, 12, "result"),         # anchor (payoff)
    ]
    by_id = {u.unit_id: u for u in units}
    # Candidate required fields (types.py): cand_id, anchor_id, role, facet, title, reason,
    # unit_ids, referential, i_start, i_end, start, end. contract_role is optional.
    cand = Candidate(cand_id="c0", anchor_id="u2", role="result", facet="worked_example",
                     title="", reason="", unit_ids=["u0", "u1", "u2"], referential=[],
                     i_start=0, i_end=12, start=0.0, end=12.0, contract_role="result")
    adapter = get_adapter("generic")
    protected = _protected_unit_ids(cand, by_id, adapter)
    assert "u0" in protected      # the leading onset unit must never be trimmed away
    assert "u2" in protected      # the anchor stays protected (existing behavior)


def test_onset_protection_without_contract_coverage():
    """True RED→GREEN gap: the leading unit has role 'setup', which the 'claim' contract
    marks only as RECOMMENDED (not required) in the 'subject' element — so the existing
    contract-driven protection does NOT cover it.  The onset-protection rule must protect
    it regardless.

    'claim' contract required_roles = {'claim', 'result'} — 'setup' is absent.
    This test was FAILING before the onset-protection implementation (RED) and GREEN after.
    """
    units = [
        _unit("u0", 0, 4, "setup"),       # onset — role NOT in claim contract's required elements
        _unit("u1", 4, 8, "evidence"),
        _unit("u2", 8, 12, "claim"),      # anchor
    ]
    by_id = {u.unit_id: u for u in units}
    cand = Candidate(cand_id="c1", anchor_id="u2", role="claim", facet="other",
                     title="", reason="", unit_ids=["u0", "u1", "u2"], referential=[],
                     i_start=0, i_end=12, start=0.0, end=12.0, contract_role="claim")
    adapter = get_adapter("generic")
    protected = _protected_unit_ids(cand, by_id, adapter)
    assert "u0" in protected      # onset must be protected even when contract doesn't require its role
    assert "u2" in protected      # anchor protection unchanged
