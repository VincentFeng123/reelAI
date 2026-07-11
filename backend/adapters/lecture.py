"""Lecture / physics / math adapter — the fully-worked exemplar (spec §4, §11).

Adds a physics/math role pack on top of the universal roles and defines ordered
completeness contracts for derivations and equation introductions. The generic
contracts (definition, claim, result/worked-example, practice, solution) are inherited
and remain valid for lectures; this adapter tightens the physics-specific ones.
"""
from __future__ import annotations

from .base import BaseAdapter, CompletenessContract, ContractElement, RoleSpec

E = ContractElement


class LectureAdapter(BaseAdapter):
    domain = "lecture"
    content_types = ("lecture", "physics", "math", "science_explainer", "class", "course")

    def _domain_role_specs(self) -> dict[str, RoleSpec]:
        specs = [
            RoleSpec("equation_introduction",
                     "An equation or law is stated / written for the first time.",
                     facet="derivation", is_anchor=True, anchor_priority=64),
            RoleSpec("derivation",
                     "A result is derived step-by-step from prior equations.",
                     facet="derivation", is_anchor=True, anchor_priority=85),
            RoleSpec("variable_definition",
                     "A symbol/variable is bound to a physical quantity and units.",
                     facet="definition"),
            RoleSpec("diagram_interpretation",
                     "A figure / free-body / circuit diagram is read or explained.",
                     facet="application", is_anchor=True, anchor_priority=55),
            RoleSpec("graph_interpretation",
                     "A plot / curve is read (axes, slope, area, trends).",
                     facet="application", is_anchor=True, anchor_priority=55),
            RoleSpec("problem_givens",
                     "The known quantities / constraints of a problem are stated.",
                     facet="worked_example"),
            RoleSpec("calculation",
                     "Numbers are substituted and arithmetic is carried out.",
                     facet="worked_example"),
            RoleSpec("unit_check",
                     "A dimensional / units sanity check on a result.",
                     facet="worked_example"),
            RoleSpec("physical_interpretation",
                     "What the result means physically / a sanity check.",
                     facet="other", is_anchor=True, anchor_priority=46),
        ]
        return {s.name: s for s in specs}

    def labeling_hints(self) -> str:
        return ("This is a lecture. Prefer the physics/math roles when a symbol, equation, "
                "figure, or numeric problem is involved. A worked example spans from the problem "
                "being read (example_setup / problem_givens) through calculation to result. Label "
                "boilerplate ('open your textbook', 'quiz Friday') administrative.")

    def concept_hints(self) -> str:
        return ("Treat named laws, theorems, equations, and defined variables as concepts. A "
                "worked example REQUIRES the concepts of the equations it applies; a derivation "
                "REQUIRES its starting equations.")

    def _contracts(self) -> dict[str, CompletenessContract]:
        return {
            "result": CompletenessContract("result", (
                E("problem_statement", ("example_setup", "problem_givens", "practice_prompt"), "required", "before"),
                E("relevant_principle", ("equation_introduction", "definition", "claim"), "recommended", "before"),
                E("solution_steps", ("worked_step", "calculation", "derivation"), "required", "within", repeatable=True),
                # calculation-as-final (arcs.py / the audited kinematics example): the answer
                # may live in a 'calculation'-role unit — without it here P1a binds such spans
                # to 'procedure' and the problem_statement/reasoning/result gates are bypassed.
                E("result", ("result", "solution", "calculation", "derivation"), "required", "within"),
                E("interpretation", ("physical_interpretation", "unit_check", "explanation"), "recommended", "after"),
            ), max_span_units=16),
            "derivation": CompletenessContract("derivation", (
                E("goal", ("setup", "claim", "equation_introduction"), "recommended", "before"),
                E("starting_point", ("equation_introduction", "variable_definition", "problem_givens"), "required", "before"),
                E("steps", ("worked_step", "calculation", "derivation"), "required", "within", repeatable=True),
                E("result", ("result", "equation_introduction"), "required", "within"),
                E("interpretation", ("physical_interpretation", "explanation"), "recommended", "after"),
            )),
            "equation_introduction": CompletenessContract("equation_introduction", (
                E("statement", ("equation_introduction",), "required", "within"),
                E("variables", ("variable_definition", "definition"), "recommended", "within"),
                E("meaning", ("explanation", "intuition", "physical_interpretation"), "recommended", "after"),
            )),
        }
