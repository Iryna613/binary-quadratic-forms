from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterable, List, Tuple

Matrix2x2 = Tuple[Tuple[int, int], Tuple[int, int]]


def mat_mul(left: Matrix2x2, right: Matrix2x2) -> Matrix2x2:
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def mat_det(matrix: Matrix2x2) -> int:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def gcd_many(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result = gcd(result, abs(value))
    return result


@dataclass(frozen=True)
class QuadraticForm:
    """Integral binary quadratic form Q(x,y)=A*x^2+B*x*y+C*y^2."""

    A: int
    B: int
    C: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return self.A, self.B, self.C

    def discriminant(self) -> int:
        return self.B * self.B - 4 * self.A * self.C

    def classify(self) -> str:
        D = self.discriminant()
        if D < 0:
            if self.A > 0:
                return "positive definite"
            if self.A < 0:
                return "negative definite"
            return "definite"
        if D > 0:
            return "indefinite"
        return "degenerate"

    def is_primitive(self) -> bool:
        return gcd_many((self.A, self.B, self.C)) == 1

    def evaluate(self, x: int, y: int) -> int:
        return self.A * x * x + self.B * x * y + self.C * y * y

    def transform(self, matrix: Matrix2x2) -> "QuadraticForm":
        """
        Return coefficients of Q(M*[X,Y]^T). Columns of M are the new basis vectors.
        """
        (alpha, beta), (gamma, delta) = matrix
        a_new = self.evaluate(alpha, gamma)
        c_new = self.evaluate(beta, delta)
        b_new = (
            2 * self.A * alpha * beta
            + self.B * (alpha * delta + beta * gamma)
            + 2 * self.C * gamma * delta
        )
        return QuadraticForm(a_new, b_new, c_new)

    def find_representations(self, number: int, bound: int, primitive_only: bool = False) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        for x in range(-bound, bound + 1):
            for y in range(-bound, bound + 1):
                if primitive_only and gcd(abs(x), abs(y)) != 1:
                    continue
                if self.evaluate(x, y) == number:
                    result.append((x, y))
        return result

    def to_latex(self) -> str:
        parts: List[str] = []
        if self.A:
            parts.append(f"{self.A}x^2")
        if self.B:
            sign = "+" if self.B > 0 and parts else ""
            parts.append(f"{sign}{self.B}xy")
        if self.C:
            sign = "+" if self.C > 0 and parts else ""
            parts.append(f"{sign}{self.C}y^2")
        return "".join(parts) if parts else "0"
