from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from ncpoleon.polynomials import Polynomial

from ncpoleon.polynomials.commutative_polynomials import CommutativePolynomialElement
from ncpoleon.polynomials.noncommutative_polynomials import NonCommutativePolynomialElement

PolynomialElements = TypeVar("PolynomialElements", CommutativePolynomialElement, NonCommutativePolynomialElement)
Scalar = TypeVar("Scalar", float, complex)


@dataclass(frozen=True, eq=False, order=False, kw_only=True)
class PseudomomentMatrixDecomposition(Generic[PolynomialElements, Scalar]):
    """A single (pseudo)moment matrix contribution to a SoS decomposition.

    Docs TODO
    """

    generator: Polynomial[PolynomialElements, Scalar]
    decomposition: Polynomial[PolynomialElements, Scalar]


@dataclass(frozen=True, eq=False, order=False, kw_only=True)
class SingleMomentDecomposition(Generic[PolynomialElements, Scalar]):
    """A single scalar moment contribution to a SoS decomposition.

    Docs TODO
    """

    generator: Polynomial[PolynomialElements, Scalar]
    decomposition: Scalar


@dataclass(frozen=True, eq=False, order=False, kw_only=True)
class SoSDecomposition(Generic[PolynomialElements, Scalar]):
    pseudomoment_matrices: list[PseudomomentMatrixDecomposition[PolynomialElements, Scalar]]
    single_moments: list[SingleMomentDecomposition[PolynomialElements, Scalar]]
