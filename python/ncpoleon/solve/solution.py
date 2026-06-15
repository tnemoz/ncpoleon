from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from ncpoleon.polynomials.commutative_polynomials import CommutativePolynomialElement
from ncpoleon.polynomials.noncommutative_polynomials import NonCommutativePolynomialElement
from ncpoleon.solve.utils import sos_vectors_of_hermitian_matrix

from .sos_decomposition import PseudomomentMatrixDecomposition, SingleMomentDecomposition, SoSDecomposition

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
    def moment_matrix(self) -> npt.NDArray[np.float64 | np.complex128]:
        moment_matrices = self.moment_matrix_by_mm_id
        if len(moment_matrices) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `moment_matrix` property will only return the one "
                "associated to index 0. Use `moment_matrix_by_mm_id` to access all of them.",
            )
        return moment_matrices[0]

    @property
    @abstractmethod
    def moment_matrix_by_mm_id(self) -> dict[int, npt.NDArray[np.float64 | np.complex128]]: ...

    @property
    def moment_matrix_multiplier(self) -> npt.NDArray[np.float64 | np.complex128]:
        moment_matrix_multipliers = self.moment_matrix_multiplier_by_mm_id
        if len(moment_matrix_multipliers) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `moment_matrix_multiplier` property will only "
                "return the one associated to index 0. Use `moment_matrix_multiplier_by_mm_id` to access all of them.",
            )
        return moment_matrix_multipliers[0]

    @property
    @abstractmethod
    def moment_matrix_multiplier_by_mm_id(self) -> dict[int, npt.NDArray[np.float64 | np.complex128]]: ...

    @property
    def localizing_matrices_equality(
        self,
    ) -> list[tuple[Polynomial[PolynomialElements, Scalar], npt.NDArray[np.float64 | np.complex128]]]:
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
    ) -> dict[int, list[tuple[Polynomial[PolynomialElements, Scalar], npt.NDArray[np.float64 | np.complex128]]]]: ...

    @property
    def localizing_matrices_equality_multipliers(
        self,
    ) -> list[tuple[Polynomial[PolynomialElements, Scalar], npt.NDArray[np.float64 | np.complex128]]]:
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
    ) -> dict[int, list[tuple[Polynomial[PolynomialElements, Scalar], npt.NDArray[np.float64 | np.complex128]]]]: ...

    @property
    def localizing_matrices_inequality(
        self,
    ) -> list[tuple[Polynomial[PolynomialElements, Scalar], npt.NDArray[np.float64 | np.complex128]]]:
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
    ) -> dict[int, list[tuple[Polynomial[PolynomialElements, Scalar], npt.NDArray[np.float64 | np.complex128]]]]: ...

    @property
    def localizing_matrices_inequality_multipliers(self) -> list[npt.NDArray[np.float64 | np.complex128]]:
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
    def localizing_matrices_inequality_multipliers_by_mm_id(
        self,
    ) -> dict[int, list[npt.NDArray[np.float64 | np.complex128]]]: ...

    @property
    @abstractmethod
    def moment_equalities_multipliers(
        self,
    ) -> list[tuple[Polynomial[PolynomialElements, Scalar], np.float64 | np.complex128]]: ...

    @property
    @abstractmethod
    def moment_inequalities_multipliers(self) -> list[tuple[Polynomial[PolynomialElements, Scalar], np.float64]]: ...

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

    def get_sos_decomposition_by_mm_id(self) -> dict[int, SoSDecomposition[PolynomialElements, Scalar]]:
        res = {}
        moment_matrix_multipliers = self.moment_matrix_multiplier_by_mm_id
        localizing_moment_matrices_multipliers_equality = self.localizing_matrices_equality_by_mm_id
        localizing_moment_matrices_multipliers_inequality = self.localizing_matrices_inequality_by_mm_id
        moment_equality_multipliers = {}

        for polynomial, scalar in self.moment_equalities_multipliers:
            for mm_id, poly_id in polynomial.by_moment_matrix_id().items():
                if mm_id in moment_equality_multipliers:
                    moment_equality_multipliers[mm_id].append((poly_id, scalar))
                else:
                    moment_equality_multipliers[mm_id] = [(poly_id, scalar)]

        moment_inequality_multipliers = {}

        for polynomial, scalar in self.moment_inequalities_multipliers:
            for mm_id, poly_id in polynomial.by_moment_matrix_id().items():
                if mm_id in moment_inequality_multipliers:
                    moment_inequality_multipliers[mm_id].append((poly_id, scalar))
                else:
                    moment_inequality_multipliers[mm_id] = [(poly_id, scalar)]

        for mm_id in self.relaxation.moment_matrices:
            pseudomoment_matrices = []

            sos_vectors = sos_vectors_of_hermitian_matrix(moment_matrix_multipliers[mm_id])[0]
            n_monomials = sos_vectors.shape[1]
            decompositions = (sos_vectors @ self.relaxation.generating_sets[mm_id][:n_monomials]).reshape(-1).tolist()
            pseudomoment_matrices.append(PseudomomentMatrixDecomposition(generator=1, decomposition=decompositions))

            for generator, multiplier in localizing_moment_matrices_multipliers_inequality.get(mm_id, []):
                sos_vectors = sos_vectors_of_hermitian_matrix(multiplier)[0]
                n_monomials = sos_vectors.shape[1]
                decompositions = (
                    (sos_vectors @ self.relaxation.generating_sets[mm_id][:n_monomials]).reshape(-1).tolist()
                )
                pseudomoment_matrices.append(
                    PseudomomentMatrixDecomposition(generator=generator, decomposition=decompositions)
                )

            for generator, multiplier in localizing_moment_matrices_multipliers_equality.get(mm_id, []):
                sos_vectors_pos, sos_vectors_neg = sos_vectors_of_hermitian_matrix(multiplier)
                n_monomials = sos_vectors_pos.shape[1]
                decompositions_pos = (
                    (sos_vectors_pos @ self.relaxation.generating_sets[mm_id][:n_monomials]).reshape(-1).tolist()
                )
                decompositions_neg = (
                    (sos_vectors_neg @ self.relaxation.generating_sets[mm_id][:n_monomials]).reshape(-1).tolist()
                )
                pseudomoment_matrices.append(
                    PseudomomentMatrixDecomposition(generator=generator, decomposition=decompositions_pos)
                )
                pseudomoment_matrices.append(
                    PseudomomentMatrixDecomposition(generator=generator, decomposition=decompositions_neg, sign=-1)
                )

            single_moments = []

            for generator, multiplier in moment_equality_multipliers.get(mm_id, []):
                single_moments.append(SingleMomentDecomposition(generator=generator, decomposition=multiplier))

            for generator, multiplier in moment_inequality_multipliers.get(mm_id, []):
                single_moments.append(SingleMomentDecomposition(generator=generator, decomposition=multiplier))

            res[mm_id] = SoSDecomposition(pseudomoment_matrices=pseudomoment_matrices, single_moments=single_moments)

        return res
