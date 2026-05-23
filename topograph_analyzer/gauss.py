from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .quadratic_form import Matrix2x2, QuadraticForm, mat_mul


@dataclass(frozen=True)
class ReductionStep:
    form_before: Tuple[int, int, int]
    operation: str
    matrix: Matrix2x2
    form_after: Tuple[int, int, int]


@dataclass(frozen=True)
class ReductionResult:
    input_form: QuadraticForm
    reduced_form: QuadraticForm
    transformation_matrix: Matrix2x2
    steps: List[ReductionStep]


def is_reduced_positive_definite(form: QuadraticForm) -> bool:
    if form.discriminant() >= 0 or form.A <= 0:
        return False
    a, b, c = form.as_tuple()
    if not (abs(b) <= a <= c):
        return False
    if (a == c or abs(b) == a) and b < 0:
        return False
    return True


def reduce_positive_definite(form: QuadraticForm, max_iterations: int = 10_000) -> ReductionResult:
    if form.discriminant() >= 0 or form.A <= 0:
        raise ValueError("Gauss reduction expects a positive definite form: D<0 and A>0.")

    current = form
    total_matrix: Matrix2x2 = ((1, 0), (0, 1))
    steps: List[ReductionStep] = []

    for _ in range(max_iterations):
        if is_reduced_positive_definite(current):
            return ReductionResult(form, current, total_matrix, steps)

        a, b, c = current.as_tuple()
        if abs(b) > a:
            m = math.floor((a - b) / (2 * a))
            if m == 0 and b <= -a:
                m = 1
            elif m == 0 and b > a:
                m = -1
            local_matrix = ((1, m), (0, 1))
            operation = f"reduce middle coefficient: x=X+({m})Y, y=Y"
        elif a > c:
            local_matrix: Matrix2x2 = ((0, 1), (1, 0))
            operation = "swap variables: x=Y, y=X"
        elif (a == c or abs(b) == a) and b < 0:
            local_matrix = ((1, 0), (0, -1))
            operation = "flip sign: x=X, y=-Y"
        else:
            raise RuntimeError(f"Gauss reduction reached an unexpected state: {current.as_tuple()}.")

        before = current.as_tuple()
        current = current.transform(local_matrix)
        if current.as_tuple() == before:
            raise RuntimeError(f"Gauss reduction made no progress from {before} using operation {operation}.")
        total_matrix = mat_mul(total_matrix, local_matrix)
        steps.append(ReductionStep(before, operation, local_matrix, current.as_tuple()))

    raise RuntimeError("Gauss reduction did not terminate.")
