from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

if TYPE_CHECKING:
    from ncpoleon.polynomials import Polynomial

from ncpoleon.polynomials.commutative_polynomials import CommutativePolynomialElement
from ncpoleon.polynomials.noncommutative_polynomials import NonCommutativePolynomialElement

PolynomialElements = TypeVar("PolynomialElements", CommutativePolynomialElement, NonCommutativePolynomialElement)
Scalar = TypeVar("Scalar", float, complex)


@dataclass(eq=False, order=False, kw_only=True)
class PseudomomentMatrixDecomposition(Generic[PolynomialElements, Scalar]):
    """A single (pseudo)moment matrix contribution to a SoS decomposition.

    Docs TODO
    """

    generator: Literal[1] | Polynomial[PolynomialElements, Scalar]
    decomposition: list[Polynomial[PolynomialElements, Scalar]]
    sign: Literal[1] | Literal[-1] = 1


@dataclass(eq=False, order=False, kw_only=True)
class SingleMomentDecomposition(Generic[PolynomialElements, Scalar]):
    """A single scalar moment contribution to a SoS decomposition.

    Docs TODO
    """

    generator: Polynomial[PolynomialElements, Scalar]
    decomposition: Scalar


@dataclass(eq=False, order=False, kw_only=True)
class SoSDecomposition(Generic[PolynomialElements, Scalar]):
    pseudomoment_matrices: list[PseudomomentMatrixDecomposition[PolynomialElements, Scalar]]
    single_moments: list[SingleMomentDecomposition[PolynomialElements, Scalar]]

    def reduce(self) -> Polynomial[PolynomialElements, Scalar]:
        res = 0

        for pseudomoment_matrix_decomposition in self.pseudomoment_matrices:
            for polynomial in pseudomoment_matrix_decomposition.decomposition:
                res += polynomial.adjoint() * pseudomoment_matrix_decomposition.generator * polynomial

        for single_moment_decomposition in self.single_moments:
            res += single_moment_decomposition.generator * single_moment_decomposition.decomposition

        return res
