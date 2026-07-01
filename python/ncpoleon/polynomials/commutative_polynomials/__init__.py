from typing import TypeAlias

from ncpoleon._accelerate.polynomials.commutative_polynomials._monomials import CommutativeMonomial
from ncpoleon._accelerate.polynomials.commutative_polynomials._operators import (
    CommutativeOperator,
    generate_commutative_variables,
)

CommutativePolynomialElement: TypeAlias = CommutativeMonomial | CommutativeOperator

__all__ = ["generate_commutative_variables"]
