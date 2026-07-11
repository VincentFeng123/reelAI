"""Deterministic completeness precheck + contract-by-content binding (spec §7, P1).

A cheap, no-LLM check of whether a candidate's inline units cover the required elements
of its contract. Runs before the judge to cut judge calls and to hint expansion.

The contract itself is chosen by CONTENT (choose_contract), not by the anchor's role:
a "claim"-anchored span that swallowed a worked problem must be judged as a worked
example (problem_statement/reasoning/result gates), not as a bare claim. The binding is
re-run after every span mutation (validate.rebind_contract) so the judge, the
completeness gate, and final scoring always share one contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from ..understand.models import Unit

# Deterministic precedence for exact score+specificity ties: problem-shaped contracts
# dominate whenever setup+steps+answer roles are present (they already win on specificity
# then; precedence settles the remaining exact ties). Contract roles not listed here rank
# after every listed one, ordered alphabetically for determinism.
CONTRACT_PRECEDENCE: tuple[str, ...] = (
    "result", "derivation", "solution", "procedure", "practice_prompt",
    "correction", "definition", "claim",
)


def _precedence_rank(role: str) -> tuple[int, str]:
    try:
        return (CONTRACT_PRECEDENCE.index(role), "")
    except ValueError:
        return (len(CONTRACT_PRECEDENCE), role)


def choose_contract(unit_ids: list[str], units_by_id: dict[str, Unit], adapter) -> Optional[str]:
    """Contract-by-content (P1a): pick the completeness contract that best matches the roles
    ACTUALLY present in the assembled span, independent of the anchor's role.

    Every contract in the adapter's contract set is scored by
    (required elements satisfiable by roles_present) / (required elements); ties break by
    specificity (more required elements wins), then by CONTRACT_PRECEDENCE, then by name.
    Contracts with no satisfiable required element never bind. Returns the winning contract
    role, or None when nothing binds (callers fall back to the anchor role — pre-P1
    behavior for spans that match no contract, and for contract-free test adapters)."""
    contracts_fn = getattr(adapter, "completeness_contracts", None)
    contracts = contracts_fn() if callable(contracts_fn) else {}
    if not contracts:
        return None
    roles_present = {units_by_id[uid].role for uid in unit_ids if uid in units_by_id}
    best_key: Optional[tuple] = None
    best_role: Optional[str] = None
    for role in sorted(contracts):
        contract = contracts[role]
        required = [el for el in contract.elements if el.necessity == "required"]
        if not required:
            continue                       # a contract demanding nothing must never bind
        satisfiable = sum(1 for el in required if any(r in roles_present for r in el.roles))
        if satisfiable == 0:
            continue
        # Fraction: exact score comparison (2/3 == 4/6, no float drift) → determinism
        key = (-Fraction(satisfiable, len(required)), -len(required), _precedence_rank(role))
        if best_key is None or key < best_key:
            best_key, best_role = key, role
    return best_role


def contract_coverage(unit_ids: list[str], contract_role: str, units_by_id: dict[str, Unit],
                      adapter) -> Fraction:
    """P4a tie-break input: the fraction of the bound contract's REQUIRED elements that
    the roles ACTUALLY present in the span satisfy — choose_contract's satisfiable/required
    scoring applied to ONE (already-bound) contract. No adapter / no contract / a contract
    demanding nothing → 0 (the tie-break stays neutral and final_quality decides)."""
    contract = adapter.contract_for(contract_role) if adapter is not None else None
    if not contract:
        return Fraction(0)
    required = [el for el in contract.elements if el.necessity == "required"]
    if not required:
        return Fraction(0)
    roles_present = {units_by_id[uid].role for uid in unit_ids if uid in units_by_id}
    satisfied = sum(1 for el in required if any(r in roles_present for r in el.roles))
    return Fraction(satisfied, len(required))


@dataclass
class ContractReport:
    role: str              # the CONTRACT role the span was checked against (not the anchor's)
    present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)   # required elements not covered
    ok: bool = True


def check_contract(unit_ids: list[str], role: str, units_by_id: dict[str, Unit], adapter) -> ContractReport:
    """Precheck the span against the contract for ``role`` — callers in the assembly/judging
    path pass the content-bound contract_role (P1b), never the raw anchor role."""
    contract = adapter.contract_for(role)
    if not contract:
        return ContractReport(role, [], [], True)
    roles_present = {units_by_id[uid].role for uid in unit_ids if uid in units_by_id}
    present, missing = [], []
    for el in contract.elements:
        if any(r in roles_present for r in el.roles):
            present.append(el.key)
        elif el.necessity == "required":
            missing.append(el.key)
    return ContractReport(role, present, missing, ok=(not missing))
