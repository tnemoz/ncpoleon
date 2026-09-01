from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Literal

from ncpoleon.relaxations import Realness

try:
    # The following import allows to use dunder methods on MOSEK expressions
    import mosek.fusion.pythonic  # noqa: F401
    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense, PSDVariable

    if TYPE_CHECKING:
        from mosek.fusion import Expression, SparseMatrix

    _mosek_available = True
except ImportError:
    _mosek_available = False

    if TYPE_CHECKING:
        from mosek.fusion import Expr, Expression, Matrix, Model, PSDVariable, SparseMatrix

from ncpoleon._typing import PolynomialElements, Scalar

if TYPE_CHECKING:
    from ncpoleon.relaxations import BaseSdpRelaxation, MomentMatrix


logger = logging.getLogger(__name__)


class _ComplexExpr:
    """Pair of MOSEK Expr objects representing a complex-valued Expr"""

    def __init__(self, real: Expression, imag: Expression | None):
        self.real = real
        self.imag = imag

    def __mul__(self, scalar: float | complex) -> _ComplexExpr:
        if isinstance(scalar, complex):
            re, im = scalar.real, scalar.imag
        else:
            re, im = float(scalar), 0.0

        if self.imag is None:
            return _ComplexExpr(
                Expr.mul(re, self.real),
                Expr.mul(im, self.real) if im != 0.0 else None,
            )

        new_real = Expr.sub(Expr.mul(re, self.real), Expr.mul(im, self.imag))
        new_imag = Expr.add(Expr.mul(re, self.imag), Expr.mul(im, self.real))

        return _ComplexExpr(new_real, new_imag)

    def __add__(self, other: _ComplexExpr) -> _ComplexExpr:
        if self.imag is None:
            if other.imag is None:
                return _ComplexExpr(Expr.add(self.real, other.real), None)
            else:
                return _ComplexExpr(Expr.add(self.real, other.real), other.imag)
        if other.imag is None:
            return _ComplexExpr(Expr.add(self.real, other.real), self.imag)
        return _ComplexExpr(
            Expr.add(self.real, other.real),
            Expr.add(self.imag, other.imag),
        )

    def conj(self) -> _ComplexExpr:
        if self.imag is None:
            return self
        return _ComplexExpr(self.real, Expr.mul(-1.0, self.imag))


def convert_row_col_data_to_mosek_symmetric_matrix(
    position_matrix: tuple[list[int], list[int], list[float]], size: int
) -> Matrix:
    rows, cols, data = position_matrix

    return Matrix.sparse(size, size, rows, cols, data)


# TODO: add the docstring
def convert_row_col_data_to_mosek_hermitian_matrix(
    position_matrix: tuple[list[int], list[int], list[complex]], size: int
) -> tuple[Matrix, Matrix, Matrix]:
    rows, cols, data = position_matrix

    return (
        Matrix.sparse(size, size, rows, cols, [x.real for x in data]),
        Matrix.sparse(size, size, rows, cols, [x.imag for x in data]),
        Matrix.sparse(size, size, rows, cols, [-x.imag for x in data]),
    )


# TODO: add the docstring
def convert_row_col_data_to_mosek_hermitianized_matrix(
    position_matrix: tuple[list[int], list[int], list[complex]], size: int, *, real_part: bool
) -> SparseMatrix:
    rows, cols, data = position_matrix
    data_re = [x.real for x in data]
    data_im = [x.imag for x in data]

    if real_part:
        neg_data_im = [-x.imag for x in data]
        real_part = Matrix.sparse(size, size, rows + cols, cols + rows, data_re + data_re)
        imag_part = Matrix.sparse(size, size, rows + cols, cols + rows, data_im + neg_data_im)
        neg_imag_part = Matrix.sparse(size, size, rows + cols, cols + rows, neg_data_im + data_im)
    else:
        neg_data_re = [-x.real for x in data]
        real_part = Matrix.sparse(size, size, rows + cols, cols + rows, data_im + data_im)
        imag_part = Matrix.sparse(size, size, rows + cols, cols + rows, neg_data_re + data_re)
        neg_imag_part = Matrix.sparse(size, size, rows + cols, cols + rows, data_re + neg_data_re)

    return Matrix.sparse([[real_part, neg_imag_part], [imag_part, real_part]])


def rust_moment_matrix_to_mosek(
    moment_matrix: MomentMatrix[PolynomialElements, Scalar],
    mapped_variables: dict[PolynomialElements, Expr | _ComplexExpr],
    is_problem_real_valued: bool,
) -> Expression:
    if is_problem_real_valued:
        mosek_moment_matrix = 0

        for mon, (pos_matrix, realness) in moment_matrix.as_row_col_data_format().items():
            pos_matrix = convert_row_col_data_to_mosek_symmetric_matrix(pos_matrix, moment_matrix.size)
            mosek_moment_matrix = Expr.add(mosek_moment_matrix, Expr.mul(mapped_variables[mon], pos_matrix))

        return mosek_moment_matrix

    mosek_moment_matrix_re = 0
    mosek_moment_matrix_im = 0

    for mon, (pos_matrix, realness) in moment_matrix.as_row_col_data_format().items():
        pos_matrix = convert_row_col_data_to_mosek_hermitian_matrix(pos_matrix, moment_matrix.size)

        if realness == Realness.Real:
            mosek_moment_matrix_re = Expr.add(
                mosek_moment_matrix_re,
                Expr.mul(mapped_variables[mon].real, pos_matrix[0]),
            )
            mosek_moment_matrix_im = Expr.add(
                mosek_moment_matrix_im,
                Expr.mul(mapped_variables[mon].real, pos_matrix[1]),
            )
        else:
            mosek_moment_matrix_re = Expr.add(
                mosek_moment_matrix_re,
                Expr.sub(
                    Expr.add(
                        Expr.mul(mapped_variables[mon].real, pos_matrix[0]),
                        Expr.mul(mapped_variables[mon].real, pos_matrix[0].transpose()),
                    ),
                    Expr.add(
                        Expr.mul(mapped_variables[mon].imag, pos_matrix[1]),
                        Expr.mul(mapped_variables[mon].imag, pos_matrix[1].transpose()),
                    ),
                ),
            )
            mosek_moment_matrix_im = Expr.add(
                mosek_moment_matrix_im,
                Expr.add(
                    Expr.sub(
                        Expr.mul(mapped_variables[mon].real, pos_matrix[1]),
                        Expr.mul(mapped_variables[mon].real, pos_matrix[1].transpose()),
                    ),
                    Expr.sub(
                        Expr.mul(mapped_variables[mon].imag, pos_matrix[0]),
                        Expr.mul(mapped_variables[mon].imag, pos_matrix[0].transpose()),
                    ),
                ),
            )

    # Every entry turned out to be real: the embedding would be block-diagonal with both blocks equal to the real
    # part, which is PSD if and only if that part is, so we can return it directly
    if isinstance(mosek_moment_matrix_im, int):
        return mosek_moment_matrix_re

    return Expr.vstack(
        [
            Expr.hstack([mosek_moment_matrix_re, Expr.mul(-1.0, mosek_moment_matrix_im)]),
            Expr.hstack([mosek_moment_matrix_im, mosek_moment_matrix_re]),
        ]
    )


def get_mosek_psd_variable(model: Model, name: str, size: int, symmetric: bool) -> PSDVariable:
    return model.variable(name, Domain.inPSDCone(size if symmetric else 2 * size))


# FIXME: this can probably be simplified by defining ComplexVariables and HermitianVariables just like PICOS
#  More generally, we can probably provide a blanket implementation for the export, given that the user
#  provides the function with what's a real variable, a complex one, a symmetric one, a hermitian one, and such
#  that the variables can be multiplied together, be taken the trace of, etc.
def to_mosek(
    sdp: BaseSdpRelaxation[PolynomialElements, Scalar],
    objective_direction: str,
    *,
    primal: bool,
    verbosity: Literal[0] | Literal[1] | Literal[2] | Literal[3] = 0,
    **model_kwargs: Any,
) -> Model:
    r"""Export a relaxation to MOSEK.

    :param sdp: The relaxation to be converted to MOSEK, generated with :func:`~ncpoleon.relaxations.get_relaxation`.
    :param objective_direction: Could be either "min" or "max", defines the optimization sense. Note that this is the
        optimization direction of the **primal** problem.
    :param primal: If `True`, then the problem is exported in its primal form. Otherwise, it is exported in its dual
        form.
    :param \**problem_kwargs: Any additional keyword arguments to be passed to :class:`mosek.Model` at instantiation.
    :return: A :class:`mosek.Model` object corresponding to the problem the user has specified.
    """
    if not _mosek_available:
        raise ImportError(
            "mosek is required for to_mosek but is not installed. Install it with: pip install mosek. Note that a MOSEK"
            " license is required to use MOSEK."
        )
    if objective_direction not in ["min", "max"]:
        raise ValueError(
            f"The only supported objective directions are min and max, but {objective_direction} was provided."
        )

    M = Model()

    if verbosity > 0:
        M.setSolverParam("log", verbosity)
        M.setLogHandler(sys.stdout)

    for param, value in model_kwargs.items():
        M.setSolverParam(param, value)

    if primal:
        logger.info("Exporting to a primal MOSEK problem.")

        mapped_variables = {}
        is_problem_real_valued = sdp.is_real

        for moment_matrix_id, moment_matrix in sdp.moment_matrices.items():
            for monomial, (_position_matrix, realness) in moment_matrix.as_row_col_data_format().items():
                if is_problem_real_valued:
                    new_variable = M.variable(str(monomial), Domain.unbounded())
                elif realness == Realness.Real:
                    new_variable = _ComplexExpr(
                        M.variable(str(monomial), Domain.unbounded()),
                        None,
                    )
                else:
                    new_variable = _ComplexExpr(
                        M.variable(f"{str(monomial)}_re", Domain.unbounded()),
                        M.variable(f"{str(monomial)}_im", Domain.unbounded()),
                    )

                mapped_variables[monomial] = new_variable

            mosek_moment_matrix = rust_moment_matrix_to_mosek(moment_matrix, mapped_variables, is_problem_real_valued)
            M.constraint(
                f"MM-{moment_matrix_id}", mosek_moment_matrix, Domain.inPSDCone(mosek_moment_matrix.getShape()[0])
            )
            logger.debug(f"Added moment matrix PSD constraint for moment matrix id {moment_matrix_id}.")

        for moment_matrix_id, equality_moment_matrices in sdp.localising_moment_matrices_equalities.items():
            for index, equality_moment_matrix in enumerate(equality_moment_matrices):
                mosek_new_localising_matrix = rust_moment_matrix_to_mosek(
                    equality_moment_matrix, mapped_variables, is_problem_real_valued
                )
                M.constraint(f"LMME-{moment_matrix_id}-{index}", mosek_new_localising_matrix, Domain.equalsTo(0))
                logger.debug(
                    f"Added constraint {mosek_new_localising_matrix} == 0 for moment matrix id {moment_matrix_id}."
                )

        for moment_matrix_id, inequality_moment_matrices in sdp.localising_moment_matrices_inequalities.items():
            for index, inequality_moment_matrix in enumerate(inequality_moment_matrices):
                mosek_new_localising_matrix = rust_moment_matrix_to_mosek(
                    inequality_moment_matrix, mapped_variables, is_problem_real_valued
                )
                M.constraint(
                    f"LMMI-{moment_matrix_id}-{index}",
                    mosek_new_localising_matrix,
                    Domain.inPSDCone(mosek_new_localising_matrix.getShape()[0]),
                )
                logger.debug(
                    f"Added constraint {mosek_new_localising_matrix} ≽ 0 for moment matrix id {moment_matrix_id}."
                )

        for index, (poly, value) in enumerate(sdp.moment_equalities):
            changed = sdp.change_variables(poly, mapped_variables)

            if not is_problem_real_valued:
                M.constraint(f"ME-{index}_re", changed.real, Domain.equalsTo(value.real))
                logger.debug(f"Added constraint {changed.real} == {value.real}.")

                if changed.imag is not None:
                    M.constraint(f"ME-{index}_im", changed.imag, Domain.equalsTo(value.imag))
                    logger.debug(f"Added constraint {changed.imag} == {value.imag}.")
                elif value.imag != 0.0:
                    raise ValueError(
                        f"Moment equality {index} has an identically real left-hand side but the right-hand"
                        f" side ({value}) isn't real-valued, so it can never be satisfied."
                    )
            else:
                M.constraint(f"ME-{index}", changed, Domain.equalsTo(value))
                logger.debug(f"Added constraint {changed} == {value}.")

        for index, (poly, value) in enumerate(sdp.moment_inequalities):
            changed = sdp.change_variables(poly, mapped_variables)

            if not is_problem_real_valued:
                M.constraint(f"MI-{index}", changed.real, Domain.greaterThan(value))
                logger.debug(f"Added constraint {changed.real} >= {value}.")
            else:
                M.constraint(f"MI-{index}", changed, Domain.greaterThan(value))
                logger.debug(f"Added constraint {changed} >= {value}.")

        objective = sdp.change_variables(sdp.objective, mapped_variables)
        M.objective(
            ObjectiveSense.Minimize if objective_direction == "min" else ObjectiveSense.Maximize,
            objective.real if not is_problem_real_valued else objective,
        )
    else:
        logger.info("Exporting to a dual MOSEK problem.")

        is_problem_real_valued = sdp.is_real
        operator_inequalities = sdp.localising_moment_matrices_inequalities
        operator_equalities = sdp.localising_moment_matrices_equalities
        split_objective_re, split_objective_im = sdp.split_into_real_and_imaginary_parts(sdp.objective)
        assert split_objective_im is None

        split_moment_inequalities = [
            (sdp.split_into_real_and_imaginary_parts(poly), scalar) for (poly, scalar) in sdp.moment_inequalities
        ]
        split_moment_equalities = [
            (sdp.split_into_real_and_imaginary_parts(poly), scalar) for (poly, scalar) in sdp.moment_equalities
        ]

        lambdas = []
        objective = 0.0

        for m, (_, scalar_inequality) in enumerate(split_moment_inequalities):
            new_variable = M.variable(f"lambda_{m}", Domain.greaterThan(0.0))
            lambdas.append(new_variable)
            objective = Expr.add(objective, Expr.mul(new_variable, scalar_inequality))
            logger.debug(f"Added dual variable lambda_{m} >= 0 for moment inequality number {m}.")

        nus = []

        for n, (_, scalar_equality) in enumerate(split_moment_equalities):
            if is_problem_real_valued:
                new_variable = M.variable(f"nu_{n}")
                nus.append(new_variable)
                objective = Expr.add(objective, Expr.mul(new_variable, scalar_equality))
            else:
                new_variable = _ComplexExpr(M.variable(f"nu_{n}^re"), M.variable(f"nu_{n}^im"))
                nus.append(new_variable)
                objective = Expr.add(objective, (new_variable.conj() * scalar_equality).real)
            logger.debug(f"Added dual variable nu_{n} for moment equality number {n}.")

        if objective_direction == "max":
            M.objective(ObjectiveSense.Minimize, -objective)
        else:
            M.objective(ObjectiveSense.Maximize, objective)

        for moment_matrix_index, moment_matrix in sdp.moment_matrices.items():
            Y = get_mosek_psd_variable(M, f"Y_{moment_matrix_index}", moment_matrix.size, is_problem_real_valued)
            logger.debug(f"Added PSD variable Y_{moment_matrix_index} of size {moment_matrix.size}.")

            Ps = [
                get_mosek_psd_variable(
                    M,
                    f"P_{(moment_matrix_index, inequality_index)}",
                    inequality_localizing_matrix.size,
                    is_problem_real_valued,
                )
                for inequality_index, inequality_localizing_matrix in enumerate(
                    operator_inequalities[moment_matrix_index]
                )
            ]
            logger.debug(f"Added {len(Ps)} PSD variable(s) P_* for moment matrix {moment_matrix_index}.")

            Qs = [
                Expr.sub(
                    get_mosek_psd_variable(
                        M,
                        f"Q_{(moment_matrix_index, equality_index)}^0",
                        equality_localizing_matrix.size,
                        is_problem_real_valued,
                    ),
                    get_mosek_psd_variable(
                        M,
                        f"Q_{(moment_matrix_index, equality_index)}^1",
                        equality_localizing_matrix.size,
                        is_problem_real_valued,
                    ),
                )
                for equality_index, equality_localizing_matrix in enumerate(operator_equalities[moment_matrix_index])
            ]
            logger.debug(f"Added {len(Qs)} free Hermitian variable Q_* for moment matrix {moment_matrix_index}.")

            # Precompute localizing matrix row-col formats outside the monomial loop.
            localizing_row_cols = [
                [
                    localizing_matrix.as_row_col_data_format()
                    for localizing_matrix in operator_inequalities[moment_matrix_index]
                ],
                [
                    localizing_matrix.as_row_col_data_format()
                    for localizing_matrix in operator_equalities[moment_matrix_index]
                ],
            ]

            for monomial, (pos_matrix, realness) in moment_matrix.as_row_col_data_format().items():
                if is_problem_real_valued:  #  position matrix is symmetric
                    F = convert_row_col_data_to_mosek_symmetric_matrix(pos_matrix, moment_matrix.size)
                    new_constraint = Expr.dot(Y, F)
                elif realness == Realness.Real:  # position matrix is symmetric but represented as Hermitian
                    F_re, F_im, F_im_neg = convert_row_col_data_to_mosek_hermitian_matrix(
                        pos_matrix, moment_matrix.size
                    )
                    new_constraint = Expr.mul(Expr.dot(Y, Matrix.sparse([[F_re, F_im_neg], [F_im, F_re]])), 1 / 2)
                else:  # position matrix only contains the position of the canonical monomial
                    # We have to multiply by 1/2 to preserve the dot product when considering the representation as
                    # symmetric matrices
                    new_constraint_re = Expr.mul(
                        Expr.dot(
                            Y,
                            convert_row_col_data_to_mosek_hermitianized_matrix(
                                pos_matrix, moment_matrix.size, real_part=True
                            ),
                        ),
                        1 / 2,
                    )
                    new_constraint_im = Expr.mul(
                        Expr.dot(
                            Y,
                            convert_row_col_data_to_mosek_hermitianized_matrix(
                                pos_matrix, moment_matrix.size, real_part=False
                            ),
                        ),
                        1 / 2,
                    )

                for lagrange_mutlipliers, localizing_matrices, precomputed_row_cols in zip(
                    [Ps, Qs],
                    [operator_inequalities[moment_matrix_index], operator_equalities[moment_matrix_index]],
                    localizing_row_cols,
                ):
                    for multiplier, localizing_matrix, localizing_matrix_as_row_col in zip(
                        lagrange_mutlipliers, localizing_matrices, precomputed_row_cols
                    ):
                        pos_matrix_localizing, localizing_realness = localizing_matrix_as_row_col.get(
                            monomial, (None, Realness.Real)
                        )

                        if pos_matrix_localizing is not None:
                            if is_problem_real_valued:
                                assert localizing_realness == Realness.Real
                                G = convert_row_col_data_to_mosek_symmetric_matrix(
                                    pos_matrix_localizing, localizing_matrix.size
                                )
                                new_constraint = Expr.add(new_constraint, Expr.dot(multiplier, G))
                            elif localizing_realness == Realness.Real:
                                G_re, G_im, G_im_neg = convert_row_col_data_to_mosek_hermitian_matrix(
                                    pos_matrix_localizing, localizing_matrix.size
                                )
                                new_constraint = Expr.add(
                                    new_constraint,
                                    Expr.mul(
                                        Expr.dot(multiplier, Matrix.sparse([[G_re, G_im_neg], [G_im, G_re]])), 1 / 2
                                    ),
                                )
                            else:
                                new_constraint_re = Expr.add(
                                    new_constraint_re,
                                    Expr.mul(
                                        Expr.dot(
                                            multiplier,
                                            convert_row_col_data_to_mosek_hermitianized_matrix(
                                                pos_matrix_localizing, localizing_matrix.size, real_part=True
                                            ),
                                        ),
                                        1 / 2,
                                    ),
                                )
                                new_constraint_im = Expr.add(
                                    new_constraint_im,
                                    Expr.mul(
                                        Expr.dot(
                                            multiplier,
                                            convert_row_col_data_to_mosek_hermitianized_matrix(
                                                pos_matrix_localizing, localizing_matrix.size, real_part=False
                                            ),
                                        ),
                                        1 / 2,
                                    ),
                                )

                for lambda_m, ((poly_re, poly_im), _) in zip(lambdas, split_moment_inequalities):
                    assert poly_im is None
                    beta_re, minus_beta_im = poly_re.get(monomial, (None, None))

                    # beta_re can only be None if the monomial isn't present in the moment inequality constraint
                    if beta_re is not None:
                        if is_problem_real_valued or realness == Realness.Real:
                            assert minus_beta_im is None
                            new_constraint = Expr.add(new_constraint, Expr.mul(lambda_m, beta_re))
                        else:
                            assert minus_beta_im is not None
                            new_constraint_re = Expr.add(new_constraint_re, Expr.mul(Expr.mul(lambda_m, beta_re), 2.0))
                            new_constraint_im = Expr.add(
                                new_constraint_im, Expr.mul(Expr.mul(lambda_m, minus_beta_im), 2.0)
                            )

                for nu_n, ((poly_re, poly_im), _) in zip(nus, split_moment_equalities):
                    if is_problem_real_valued or realness == Realness.Real:
                        delta_re, delta_im = poly_re.get(monomial, (None, None))

                        if delta_re is not None:
                            assert delta_im is None
                            new_constraint = Expr.add(
                                new_constraint, Expr.mul(nu_n if is_problem_real_valued else nu_n.real, delta_re)
                            )
                    else:
                        delta_plus_eps_re, minus_delta_minus_eps_im = poly_re.get(monomial, (None, None))

                        if poly_im is not None:
                            delta_plus_eps_im, delta_minus_eps_re = poly_im.get(monomial, (None, None))
                        else:
                            delta_plus_eps_im, delta_minus_eps_re = None, None

                        if delta_plus_eps_re is not None:
                            new_constraint_re = Expr.add(new_constraint_re, Expr.mul(nu_n.real, delta_plus_eps_re))

                        if delta_plus_eps_im is not None:
                            new_constraint_re = Expr.add(new_constraint_re, Expr.mul(nu_n.imag, delta_plus_eps_im))

                        if minus_delta_minus_eps_im is not None:
                            new_constraint_im = Expr.add(
                                new_constraint_im, Expr.mul(nu_n.real, minus_delta_minus_eps_im)
                            )

                        if delta_minus_eps_re is not None:
                            new_constraint_im = Expr.add(new_constraint_im, Expr.mul(nu_n.imag, delta_minus_eps_re))

                alpha_re, alpha_im = split_objective_re.get(monomial, (0.0, None))

                if realness == Realness.Real:
                    if is_problem_real_valued:
                        assert alpha_im is None
                    if objective_direction == "min":
                        M.constraint(f"M-{monomial}", new_constraint, Domain.equalsTo(alpha_re))
                    else:
                        M.constraint(f"M-{monomial}", new_constraint, Domain.equalsTo(-alpha_re))

                    logger.debug(f"Added dual constraint for monomial {monomial}.")
                else:
                    alpha_im = 0.0 if alpha_im is None else alpha_im

                    if objective_direction == "min":
                        M.constraint(f"M-{monomial}-re", new_constraint_re, Domain.equalsTo(2 * alpha_re))
                        M.constraint(f"M-{monomial}-im", new_constraint_im, Domain.equalsTo(2 * alpha_im))
                    else:
                        M.constraint(f"M-{monomial}-re", new_constraint_re, Domain.equalsTo(-2 * alpha_re))
                        M.constraint(f"M-{monomial}-im", new_constraint_im, Domain.equalsTo(-2 * alpha_im))

                    logger.debug(f"Added dual constraints for monomial {monomial}.")

    logger.info("MOSEK problem created.")
    return M
