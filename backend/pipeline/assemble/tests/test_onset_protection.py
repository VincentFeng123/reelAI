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
    does NOT list in any element — so the existing contract-driven protection does NOT cover
    it.  The onset-protection rule must protect it regardless.

    'claim' contract elements: statement (required: claim/result), support (recommended:
    evidence/explanation/example_setup/demonstration), caveat (optional).  'setup' appears
    in NONE of them.  The onset-protection rule is therefore the sole reason u0 is protected.

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
    # Protection is targeted, not blanket: u1 (evidence) is only in the RECOMMENDED 'support'
    # element, so it must remain trimmable.
    assert "u1" not in protected


def test_onset_protection_explicit_onset_uid():
    """Directly tests the threaded (non-fallback) path: caller provides onset_uid that
    differs from the temporally-earliest in-span unit, simulating a unit grown in before
    the pre-grow onset during assembly.

    Fixture: original onset is u1 (role='setup', not in claim contract required elements).
    u0 (role='setup') was grown in after assembly so it is now temporally earliest but is
    NOT the onset and NOT contract-required.  With onset_uid='u1' explicitly passed,
    _protected_unit_ids must protect u1 (the declared onset) but NOT u0 (grown-in,
    non-onset, non-required).

    This test would FAIL if the onset-protection branch were removed because u1's role
    ('setup') is absent from the claim contract's required elements and the fallback path
    is never reached (onset_uid is not None).
    """
    units = [
        _unit("u0", 0, 4, "setup"),     # grown in after assembly — now earliest but NOT the onset
        _unit("u1", 4, 8, "setup"),     # original pre-grow onset — must be protected
        _unit("u2", 8, 12, "claim"),    # anchor
    ]
    by_id = {u.unit_id: u for u in units}
    cand = Candidate(cand_id="c2", anchor_id="u2", role="claim", facet="other",
                     title="", reason="", unit_ids=["u0", "u1", "u2"], referential=[],
                     i_start=0, i_end=12, start=0.0, end=12.0, contract_role="claim")
    adapter = get_adapter("generic")
    # Pass onset_uid explicitly — exercises the non-fallback branch in _protected_unit_ids.
    protected = _protected_unit_ids(cand, by_id, adapter, onset_uid="u1")
    assert "u1" in protected       # explicit pre-grow onset is protected
    assert "u0" not in protected   # grown-in leading unit: not onset, not required → trimmable
    assert "u2" in protected       # anchor always protected
