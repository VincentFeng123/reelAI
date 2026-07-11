"""Adjacency wrapper over the dependency graph (Part B queries)."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from ..understand.models import Edge, Unit


class Graph:
    def __init__(self, edges: list[Edge], units: list[Unit]):
        self.edges = edges
        self.out: dict[str, list[Edge]] = defaultdict(list)
        self.inc: dict[str, list[Edge]] = defaultdict(list)
        for e in edges:
            self.out[e.source].append(e)
            self.inc[e.target].append(e)
        self.order = {u.unit_id: i for i, u in enumerate(units)}
        self._by_id = {u.unit_id: u for u in units}

    def needs(self, uid: str, relations: Optional[Iterable[str]] = None) -> list[Edge]:
        rels = set(relations) if relations else None
        return [e for e in self.out.get(uid, []) if rels is None or e.relation in rels]

    def dependents(self, uid: str, relations: Optional[Iterable[str]] = None) -> list[Edge]:
        rels = set(relations) if relations else None
        return [e for e in self.inc.get(uid, []) if rels is None or e.relation in rels]

    def defines_needed_for(self, provider_ids: set[str], consumer_ids: set[str]) -> bool:
        """True if any consumer unit requires a concept that a provider unit introduces."""
        provided: set[str] = set()
        for pid in provider_ids:
            u = self._by_id.get(pid)
            if u:
                provided.update(u.concepts_introduced)
        if not provided:
            return False
        for cid in consumer_ids:
            u = self._by_id.get(cid)
            if u and (set(u.concepts_required) & provided):
                return True
        return False
