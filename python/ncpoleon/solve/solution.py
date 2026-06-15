from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from ncpoleon.polynomials.commutative_polynomials import CommutativePolynomialElement
from ncpoleon.polynomials.noncommutative_polynomials import NonCommutativePolynomialElement

from .sos_decomposition import SoSDecomposition

if TYPE_CHECKING:
    from ncpoleon.polynomials import Polynomial
    from ncpoleon.relaxations import BaseSdpRelaxation

PolynomialElements = TypeVar("PolynomialElements", CommutativePolynomialElement, NonCommutativePolynomialElement)
Scalar = TypeVar("Scalar", float, complex)


class BaseSolution(ABC, Generic[PolynomialElements, Scalar]):
    @property
    @abstractmethod
    def value(self) -> np.float64: ...

    @abstractmethod
    def __getitem__(self, monomial: PolynomialElements) -> np.float64 | np.complex128: ...

    @property
    def moment_matrix(self) -> np.ndarray:
        moment_matrices = self.moment_matrix_by_mm_id
        if len(moment_matrices) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `moment_matrix` property will only return the one "
                "associated to index 0. Use `moment_matrix_by_mm_id` to access all of them.",
            )
        return moment_matrices[0]

    @property
    @abstractmethod
    def moment_matrix_by_mm_id(self) -> dict[int, np.ndarray]: ...

    @property
    def moment_matrix_multiplier(self) -> np.ndarray:
        moment_matrix_multipliers = self.moment_matrix_multiplier_by_mm_id
        if len(moment_matrix_multipliers) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `moment_matrix_multiplier` property will only "
                "return the one associated to index 0. Use `moment_matrix_multiplier_by_mm_id` to access all of them.",
            )
        return moment_matrix_multipliers[0]

    @property
    @abstractmethod
    def moment_matrix_multiplier_by_mm_id(self) -> dict[int, np.ndarray]: ...

    @property
    def localizing_matrices_equality(self) -> list[tuple[Polynomial[PolynomialElements, Scalar], np.ndarray]]:
        localizing_matrices = self.localizing_matrices_equality_by_mm_id
        if len(localizing_matrices) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `localizing_matrices_equality` "
                "property will only return the equality localizing moment matrices associated to the moment matrix of "
                "index 0. Use `localizing_matrices_equality_constraints_by_mm_id` to access all of them.",
            )
        return localizing_matrices[0]

    @property
    @abstractmethod
    def localizing_matrices_equality_by_mm_id(
        self,
    ) -> dict[int, list[tuple[Polynomial[PolynomialElements, Scalar], np.ndarray]]]: ...

    @property
    def localizing_matrices_equality_multipliers(
        self,
    ) -> list[tuple[Polynomial[PolynomialElements, Scalar], np.ndarray]]:
        localizing_matrices_multipliers = self.localizing_matrices_equality_multipliers_by_mm_id
        if len(localizing_matrices_multipliers) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `localizing_matrices_equality_multipliers` "
                "property will only return the equality localizing moment matrices multipliers associated to the moment"
                " matrix of index 0. Use `localizing_matrices_equality_multipliers_by_mm_id` to access all of them.",
            )
        return localizing_matrices_multipliers[0]

    @property
    @abstractmethod
    def localizing_matrices_equality_multipliers_by_mm_id(
        self,
    ) -> dict[int, list[tuple[Polynomial[PolynomialElements, Scalar], np.ndarray]]]: ...

    @property
    def localizing_matrices_inequality(self) -> list[tuple[Polynomial[PolynomialElements, Scalar], np.ndarray]]:
        localizing_matrices = self.localizing_matrices_inequality_by_mm_id
        if len(localizing_matrices) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `localizing_matrices_inequality` "
                "property will only return the inequality localizing moment matrices associated to the moment matrix of"
                " index 0. Use `localizing_matrices_inequality_constraints_by_mm_id` to access all of them.",
            )
        return localizing_matrices[0]

    @property
    @abstractmethod
    def localizing_matrices_inequality_by_mm_id(
        self,
    ) -> dict[int, list[tuple[Polynomial[PolynomialElements, Scalar], np.ndarray]]]: ...

    @property
    def localizing_matrices_inequality_multipliers(self) -> list[np.ndarray]:
        localizing_matrices_multipliers = self.localizing_matrices_inequality_multipliers_by_mm_id
        if len(localizing_matrices_multipliers) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `localizing_matrices_inequality_multipliers` "
                "property will only return the inequality localizing moment matrices multipliers associated to the "
                "moment matrix of index 0. Use `localizing_matrices_inequality_multipliers_by_mm_id` to access all of "
                "them.",
            )
        return localizing_matrices_multipliers[0]

    @property
    @abstractmethod
    def localizing_matrices_inequality_multipliers_by_mm_id(self) -> dict[int, list[np.ndarray]]: ...

    @property
    @abstractmethod
    def moment_equalities_multipliers(self) -> list[np.float64 | np.complex128]: ...

    @property
    @abstractmethod
    def moment_inequalities_multipliers(self) -> list[np.float64]: ...

    @property
    @abstractmethod
    def relaxation(self) -> BaseSdpRelaxation[PolynomialElements, Scalar]: ...

    def get_sos_decomposition(self) -> SoSDecomposition[PolynomialElements, Scalar]:
        sos_decompositions = self.get_sos_decomposition_by_mm_id()
        if len(sos_decompositions) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `get_sos_decomposition` "
                "method will only return the SoS decomposition associated to the moment matrix of"
                " index 0. Use `localizing_matrices_inequality_constraints_by_mm_id` to access all of them.",
            )
        return sos_decompositions[0]

    """ def get_sos_decomposition_by_mm_id(self) -> dict[int, SoSDecomposition[PolynomialElements, Scalar]]:
        res = {}

        for mm_index in self.relaxation.moment_matrices:
             """
