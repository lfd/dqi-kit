from __future__ import annotations

import warnings
from collections.abc import Callable

from sage.all import GF
from sage.rings.finite_rings.element_base import FiniteRingElement

type LinSatExpr = int | FiniteRingElement | LinSatVar | LinSatTerms


class LinSatVar:
    def __init__(self, field: GF, id: int, name: str | None = None):
        self.field = field
        self.id = id
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __neg__(self) -> LinSatTerms:
        return -LinSatTerms.from_var(self)

    def __rmul__(self, coef: int | FiniteRingElement) -> LinSatTerms:
        return coef * LinSatTerms.from_var(self)

    def __mul__(self, coef: int | FiniteRingElement) -> LinSatTerms:
        return LinSatTerms.from_var(self) * coef

    def __add__(self, other: LinSatExpr) -> LinSatTerms:
        return LinSatTerms.from_var(self) + other

    def __sub__(self, other: LinSatExpr) -> LinSatTerms:
        return LinSatTerms.from_var(self) - other

    def __radd__(self, other: LinSatExpr) -> LinSatTerms:
        return other + LinSatTerms.from_var(self)

    def __rsub__(self, other: LinSatExpr) -> LinSatTerms:
        return other - LinSatTerms.from_var(self)

    def __eq__(self, other: LinSatExpr) -> LinSatTerms:
        return LinSatTerms.from_var(self) == other

    def __req__(self, other: LinSatExpr) -> LinSatConstraint:
        return other == LinSatTerms.from_var(self)

    def __ne__(self, other: LinSatExpr) -> LinSatConstraint:
        return LinSatTerms.from_var(self) != other

    def __rne__(self, other: LinSatExpr) -> LinSatConstraint:
        return other != LinSatTerms.from_var(self)

    def __lt__(self, other: LinSatExpr) -> LinSatConstraint:
        return LinSatTerms.from_var(self) < other

    def __gt__(self, other: LinSatExpr) -> LinSatConstraint:
        return LinSatTerms.from_var(self) > other

    def __le__(self, other: LinSatExpr) -> LinSatConstraint:
        return LinSatTerms.from_var(self) <= other

    def __ge__(self, other: LinSatExpr) -> LinSatConstraint:
        return LinSatTerms.from_var(self) >= other


class LinSatTerm:
    @staticmethod
    def constant(field: GF, coef: FiniteRingElement) -> LinSatTerm:
        var = LinSatVar(field, None, "1")
        return LinSatTerm(coef, var)

    def __init__(self, coef: FiniteRingElement, var: LinSatVar):
        self.field = var.field
        self.coef = self.field(coef)
        self.var = var

    def __str__(self) -> str:
        if self.var.id is None:
            return str(self.coef)
        return f"{self.coef} {self.var}"


class LinSatTerms:
    @staticmethod
    def from_var(var: LinSatVar) -> LinSatTerms:
        return LinSatTerms.from_term_list(var.field, [LinSatTerm(1, var)])

    @staticmethod
    def from_scalar(field: GF, coef: FiniteRingElement) -> LinSatTerms:
        return LinSatTerms.from_term_list(field, [LinSatTerm.constant(field, coef)])

    @staticmethod
    def from_any(field: GF, other: LinSatExpr) -> LinSatTerms:
        if isinstance(other, LinSatTerms):
            assert other.field == field
            return other
        if isinstance(other, int) or other in field:
            return LinSatTerms.from_scalar(field, other)
        if isinstance(other, LinSatVar):
            assert field == other.field
            return LinSatTerms.from_var(other)
        raise ValueError(f"Cannot convert {other} to Terms")

    @staticmethod
    def from_term_list(field: GF, term_list: list[LinSatTerm]) -> LinSatTerms:
        return LinSatTerms(field, {}).add_term_list(term_list)

    def __init__(self, field: GF, terms: dict[int, LinSatTerm]):
        self.field = field
        self.terms = terms

    def add_term_list(self, term_list: list[LinSatTerm]) -> LinSatTerms:
        terms = self.terms.copy()
        for term in term_list:
            vid = term.var.id
            if vid not in terms:
                terms[vid] = term
            else:
                original = terms[vid]
                terms[vid] = LinSatTerm(original.coef + term.coef, term.var)

        for vid in list(terms.keys()):
            if terms[vid].coef == self.field(0):
                del terms[vid]

        return LinSatTerms(self.field, terms)

    def __mul__(self, coef: int | FiniteRingElement) -> LinSatTerms:
        coef = self.field(coef)

        if coef == self.field(0):
            return LinSatTerms(self.field, {})

        terms = {}
        for key, term in self.terms.items():
            terms[key] = LinSatTerm(term.coef * coef, term.var)

        return LinSatTerms(self.field, terms)

    def __rmul__(self, coef: int | FiniteRingElement) -> LinSatTerms:
        return self * coef

    def __neg__(self) -> LinSatTerms:
        return (-1) * self

    def __add__(self, other: LinSatExpr) -> LinSatTerms:
        other = LinSatTerms.from_any(self.field, other)
        return self.add_term_list(other.terms.values())

    def __radd__(self, other: LinSatExpr) -> LinSatTerms:
        return self + other

    def __sub__(self, other: LinSatExpr) -> LinSatTerms:
        return self + (-other)

    def __rsub__(self, other: LinSatExpr) -> LinSatTerms:
        return -self + other

    def __eq__(self, other: LinSatExpr) -> LinSatConstraint:
        if isinstance(other, set):
            terms = self
            rhs = {self.field(el) for el in other}
        else:
            terms = self - LinSatTerms.from_any(self.field, other)
            rhs = {self.field(0)}

        constant = self.field(0)
        variables = []
        coefs = []

        # remove potential constant term from left-hand side
        for key, term in terms.terms.items():
            if key is None:
                constant = term.coef
            else:
                variables.append(term.var)
                coefs.append(term.coef)

        new_rhs = {el - constant for el in rhs}
        return LinSatConstraint(self.field, variables, coefs, new_rhs)

    def __req__(self, other: LinSatExpr) -> LinSatConstraint:
        return self == other

    def __ne__(self, other: LinSatExpr) -> LinSatConstraint:
        return (self == other).invert()

    def __rne__(self, other: LinSatExpr) -> LinSatConstraint:
        return self != other

    def inequality(
        self, other: LinSatExpr, rhs_filter: Callable[[FiniteRingElement], bool]
    ) -> LinSatConstraint:
        if not (isinstance(other, int) or other in self.field):
            raise ValueError(
                "Only constants are supported as the right-hand side of <, >, >= and <="
            )
        other = self.field(other)
        if other not in self.field.base_ring():
            raise ValueError(
                "For extension fields, <, >, >= and <= support only right-hand side elements from the underlying prime field"
            )

        return self == {el for el in self.field.base_ring() if rhs_filter(int(el))}

    def __lt__(self, other: LinSatExpr) -> LinSatConstraint:
        other = int(other)
        return self.inequality(other, lambda el: el < other)

    def __gt__(self, other: LinSatExpr) -> LinSatConstraint:
        other = int(other)
        return self.inequality(other, lambda el: el > other)

    def __le__(self, other: LinSatExpr) -> LinSatConstraint:
        other = int(other)
        return self.inequality(other, lambda el: el <= other)

    def __ge__(self, other: LinSatExpr) -> LinSatConstraint:
        other = int(other)
        return self.inequality(other, lambda el: el >= other)

    def __str__(self) -> str:
        return " + ".join(str(t) for t in self.terms.values())


class LinSatConstraint:
    def __init__(
        self,
        field: GF,
        vars: list[LinSatVar],
        coefs: list[FiniteRingElement],
        rhs: dict[FiniteRingElement, int],
        n_equations: int = 1,
    ):
        self.field = field

        if isinstance(rhs, set):
            rhs = dict.fromkeys(rhs, 1)

        assert len(vars) == len(coefs)
        assert len(rhs) == 0 or n_equations >= max(rhs.values())
        assert len(rhs) == 0 or min(rhs.values()) >= 0

        self.n_equations = n_equations

        # One entry per variable, no zero coefficients. Scaling and term order
        # are left as given; get_id() and merge() account for them.
        by_id = {}
        for v, c in zip(vars, coefs, strict=False):
            if v.id in by_id:
                by_id[v.id] = (v, by_id[v.id][1] + field(c))
            else:
                by_id[v.id] = (v, field(c))
        nonzero = [(v, c) for v, c in by_id.values() if c != field(0)]
        self.vars = [v for v, _ in nonzero]
        self.coefs = [c for _, c in nonzero]
        self.rhs = {field(v): weight for v, weight in rhs.items() if weight != 0}

        self.is_degenerate = False

        if len(self.rhs) == 0:
            self.is_degenerate = True
        if len(self.rhs) == len(field) and len(set(self.rhs.values())) == 1:
            self.is_degenerate = True
        if len(vars) == 0:
            self.is_degenerate = True

    def show_degeneracy_warnings(self):
        if len(self.rhs) == 0:
            warnings.warn(f"Constraint is always false: {self}", stacklevel=2)
        if len(self.rhs) == len(self.field) and len(set(self.rhs.values())) == 1:
            warnings.warn(f"Constraint is always true: {self}", stacklevel=2)

    def get_first_coef(self) -> FiniteRingElement:
        """Coefficient of the variable with the smallest id (1 if there are no variables)."""
        if not self.vars:
            return self.field(1)
        return min(zip(self.vars, self.coefs, strict=False), key=lambda p: p[0].id)[1]

    def get_id(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
        """Equal for constraints on the same linear form up to scaling and term order."""
        lead = self.get_first_coef()
        pairs = sorted(zip(self.vars, self.coefs, strict=False), key=lambda p: p[0].id)
        return tuple(v.id for v, _ in pairs), tuple(str(c / lead) for _, c in pairs)

    def merge(self, other: LinSatConstraint) -> LinSatConstraint:
        """Merge two constraints with equal ids. The result is sorted by variable
        id and scaled to leading coefficient 1, so it does not depend on the
        order of the operands."""
        assert self.get_id() == other.get_id()
        self_lead = self.get_first_coef()
        other_lead = other.get_first_coef()
        new_rhs = {el / self_lead: weight for el, weight in self.rhs.items()}
        for el, weight in other.rhs.items():
            el = el / other_lead
            if el not in new_rhs and weight > 0:
                new_rhs[el] = 0
            new_rhs[el] += weight
        pairs = sorted(zip(self.vars, self.coefs, strict=False), key=lambda p: p[0].id)
        return LinSatConstraint(
            self.field,
            [v for v, _ in pairs],
            [c / self_lead for _, c in pairs],
            new_rhs,
            self.n_equations + other.n_equations,
        )

    def invert(self) -> LinSatConstraint:
        elements = set(self.field)
        new_rhs = {}
        for el in elements:
            weight = self.rhs.get(el, 0)
            new_weight = self.n_equations - weight
            if new_weight > 0:
                new_rhs[el] = new_weight

        return LinSatConstraint(
            self.field, list(self.vars), list(self.coefs), new_rhs, self.n_equations
        )

    def scale_weight(self, scalar: int) -> LinSatConstraint:
        new_rhs = {el: weight * scalar for el, weight in self.rhs.items()}
        return LinSatConstraint(
            self.field, self.vars, self.coefs, new_rhs, self.n_equations * scalar
        )

    def __str__(self) -> str:
        output = []
        weight_to_rhs = {}
        for el, weight in self.rhs.items():
            if weight not in weight_to_rhs:
                weight_to_rhs[weight] = set()
            weight_to_rhs[weight].add(el)

        for weight, rhs in weight_to_rhs.items():
            line = []
            for c, v in zip(self.coefs, self.vars, strict=False):
                line.append(str(c))
                line.append(str(v))
                line.append("+")

            if len(line) == 0:
                line = ["0"]
            else:
                line.pop()

            if len(rhs) == 1:
                line.append("=")
                line.append(str(next(iter(rhs))))
            else:
                line.append("∈")
                line.append(str(rhs))
            if weight != 1:
                line.append(f"(x{weight})")
            output.append(" ".join(line))

        return "\n".join(output)
