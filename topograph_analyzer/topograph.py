from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from .gauss import is_reduced_positive_definite
from .quadratic_form import Matrix2x2, QuadraticForm, mat_det

VectorTuple = Tuple[int, int]


def _normalise_pair(m: int, n: int) -> VectorTuple:
    if m == 0 and n == 0:
        raise ValueError("zero vector is not a lax vector")
    g = gcd(abs(m), abs(n))
    m //= g
    n //= g
    if m < 0 or (m == 0 and n < 0):
        m, n = -m, -n
    return m, n


@dataclass(frozen=True, order=True)
class LaxVector:
    m: int
    n: int

    def __post_init__(self) -> None:
        m, n = _normalise_pair(self.m, self.n)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "n", n)

    def as_tuple(self) -> VectorTuple:
        return self.m, self.n

    def negative(self) -> VectorTuple:
        return -self.m, -self.n

    def value(self, form: QuadraticForm) -> int:
        return form.evaluate(self.m, self.n)

    def slope_label(self) -> str:
        if self.n == 0:
            return "∞"
        f = Fraction(self.m, self.n)
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


@dataclass(frozen=True)
class TopographEdge:
    """A real topograph edge bordered by two lax-vector regions u and v."""

    u: LaxVector
    v: LaxVector

    def __post_init__(self) -> None:
        if self.u == self.v:
            raise ValueError("edge needs two distinct regions")
        if abs(det(self.u.as_tuple(), self.v.as_tuple())) != 1:
            raise ValueError(f"{self.u} and {self.v} are not a unimodular basis")
        a, b = sorted((self.u, self.v))
        object.__setattr__(self, "u", a)
        object.__setattr__(self, "v", b)

    def endpoints(self) -> Tuple[LaxVector, LaxVector]:
        return self.u, self.v


@dataclass(frozen=True)
class Diamond:
    edge: TopographEdge
    sum_vector: LaxVector
    difference_vector: LaxVector
    value_u: int
    value_v: int
    value_sum: int
    value_difference_direct: int
    value_difference_by_rule: int

    @property
    def valid(self) -> bool:
        return self.value_difference_direct == self.value_difference_by_rule


@dataclass(frozen=True)
class WellResult:
    vector: LaxVector
    value: int
    neighbours: List[Tuple[LaxVector, int]]
    is_local_minimum_in_fragment: bool


@dataclass(frozen=True)
class TopographicReductionCandidate:
    edge: TopographEdge
    basis_matrix: Matrix2x2
    form: QuadraticForm


@dataclass(frozen=True)
class TopographicReductionResult:
    topograph: "Topograph"
    candidates: List[TopographicReductionCandidate]
    selected: Optional[TopographicReductionCandidate]
    well: Optional[WellResult]


@dataclass
class Topograph:
    form: QuadraticForm
    depth: int
    regions: Dict[LaxVector, int] = field(default_factory=dict)
    edges: Set[TopographEdge] = field(default_factory=set)
    diamonds: List[Diamond] = field(default_factory=list)
    adjacency: DefaultDict[LaxVector, Set[LaxVector]] = field(default_factory=lambda: defaultdict(set))

    def add_region(self, vector: LaxVector) -> None:
        self.regions.setdefault(vector, vector.value(self.form))

    def add_edge(self, u: LaxVector, v: LaxVector) -> TopographEdge:
        edge = TopographEdge(u, v)
        self.edges.add(edge)
        self.add_region(edge.u)
        self.add_region(edge.v)
        self.adjacency[edge.u].add(edge.v)
        self.adjacency[edge.v].add(edge.u)
        return edge

    def river_edges(self) -> List[TopographEdge]:
        """Edges separating regions with different signs; meaningful for indefinite forms."""
        result: List[TopographEdge] = []
        for edge in self.edges:
            a = self.regions[edge.u]
            b = self.regions[edge.v]
            if (a == 0 and b < 0) or (b == 0 and a < 0)or a * b < 0:
                result.append(edge)
        return sorted(result, key=lambda e: (e.u.as_tuple(), e.v.as_tuple()))

    def find_well(self) -> Optional[WellResult]:
        if self.form.classify() != "positive definite" or not self.regions:
            return None
        best = min(self.regions, key=lambda v: (self.regions[v], abs(v.m) + abs(v.n), v.as_tuple()))
        neighbours = sorted(
            ((n, self.regions[n]) for n in self.adjacency.get(best, set())),
            key=lambda item: (item[1], item[0].as_tuple()),
        )
        local = all(value >= self.regions[best] for _, value in neighbours)
        return WellResult(best, self.regions[best], neighbours, local)

    def validate(self) -> Dict[str, bool]:
        return {
            "primitive_regions": all(gcd(abs(v.m), abs(v.n)) == 1 for v in self.regions),
            "unimodular_edges": all(abs(det(e.u.as_tuple(), e.v.as_tuple())) == 1 for e in self.edges),
            "diamond_rule": all(d.valid for d in self.diamonds),
        }


def det(u: VectorTuple, v: VectorTuple) -> int:
    return u[0] * v[1] - u[1] * v[0]


def add(u: LaxVector, v: LaxVector) -> LaxVector:
    return LaxVector(u.m + v.m, u.n + v.n)


def sub(u: LaxVector, v: LaxVector) -> LaxVector:
    return LaxVector(u.m - v.m, u.n - v.n)


def generate_topograph(form: QuadraticForm, depth: int = 4) -> Topograph:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    topograph = Topograph(form=form, depth=depth)
    start = (LaxVector(1, 0), LaxVector(0, 1))
    queue = deque([(start[0], start[1], 0)])
    seen: Set[TopographEdge] = set()

    while queue:
        u, v, level = queue.popleft()
        edge = TopographEdge(u, v)
        if edge in seen:
            continue
        seen.add(edge)
        topograph.add_edge(edge.u, edge.v)

        s = add(u, v)
        d = sub(u, v)
        topograph.add_region(s)
        topograph.add_region(d)
        q_u = u.value(form)
        q_v = v.value(form)
        q_s = s.value(form)
        q_d = d.value(form)
        topograph.diamonds.append(Diamond(edge, s, d, q_u, q_v, q_s, q_d, 2 * q_u + 2 * q_v - q_s))

        if level >= depth:
            continue
        for a, b in [(u, s), (v, s), (u, d), (v, d)]:
            if a != b and abs(det(a.as_tuple(), b.as_tuple())) == 1:
                queue.append((a, b, level + 1))

    return topograph


def oriented_basis_matrices(edge: TopographEdge) -> List[Matrix2x2]:
    u0, v0 = edge.endpoints()
    reps_u = [u0.as_tuple(), u0.negative()]
    reps_v = [v0.as_tuple(), v0.negative()]
    matrices: List[Matrix2x2] = []
    seen: Set[Matrix2x2] = set()
    for first_set, second_set in ((reps_u, reps_v), (reps_v, reps_u)):
        for u in first_set:
            for v in second_set:
                matrix: Matrix2x2 = ((u[0], v[0]), (u[1], v[1]))
                if mat_det(matrix) == 1 and matrix not in seen:
                    matrices.append(matrix)
                    seen.add(matrix)
    return matrices


def forms_from_edge(form: QuadraticForm, edge: TopographEdge) -> List[TopographicReductionCandidate]:
    result: List[TopographicReductionCandidate] = []
    seen: Set[Tuple[int, int, int]] = set()
    for matrix in oriented_basis_matrices(edge):
        transformed = form.transform(matrix)
        if transformed.as_tuple() not in seen:
            result.append(TopographicReductionCandidate(edge, matrix, transformed))
            seen.add(transformed.as_tuple())
    return result


def topographic_reduction(form: QuadraticForm, depth: int = 5) -> TopographicReductionResult:
    topograph = generate_topograph(form, depth=depth)
    well = topograph.find_well()
    candidates: List[TopographicReductionCandidate] = []
    if form.classify() == "positive definite":
        for edge in topograph.edges:
            for candidate in forms_from_edge(form, edge):
                if is_reduced_positive_definite(candidate.form):
                    candidates.append(candidate)
    candidates.sort(key=lambda c: (c.form.A, c.form.C, c.form.B, c.basis_matrix))
    return TopographicReductionResult(topograph, candidates, candidates[0] if candidates else None, well)
