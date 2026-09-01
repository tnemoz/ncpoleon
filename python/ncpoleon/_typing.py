from __future__ import annotations

from typing import TypeAlias, TypeVar

import numpy as np

from ncpoleon.polynomials.commutative_polynomials import CommutativePolynomialElement
from ncpoleon.polynomials.noncommutative_polynomials import NonCommutativePolynomialElement

# Shared TypeVars used across the polynomials, relaxations, solve and export modules.
# Defining them here (rather than re-declaring identically-named TypeVars in each
# module) gives every module the *same* TypeVar object, so type checkers can follow
# generic flow across module boundaries.

PolynomialElements = TypeVar("PolynomialElements", CommutativePolynomialElement, NonCommutativePolynomialElement)
Scalar = TypeVar("Scalar", float, complex)

# `change_variables` substitutes every monomial with the object the caller mapped it to and combines
# the results with that object's own arithmetic, so what comes back is whatever the mapping held: a
# PICOS expression, a MOSEK one, a plain number. The Rust side is typed `PyResult<Bound<PyAny>>` and
# does not constrain it any further
Substituted = TypeVar("Substituted")

# A solved matrix is either real or complex, never a single array holding both dtypes. Spelling it as a
# union *of arrays* (rather than one array over a union of dtypes) is what lets a type checker pick the
# matching overload of `sos_vectors_of_hermitian_matrix`.
RealOrComplexMatrix: TypeAlias = (
    np.ndarray[tuple[int, int], np.dtype[np.float64]] | np.ndarray[tuple[int, int], np.dtype[np.complex128]]
)
