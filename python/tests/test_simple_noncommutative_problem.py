import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve
from ncpoleon.utils import is_mosek_available

# TODO: Add complex-valued tests, tests for the attributes of the relaxations such that the equality constraints or the
# monomial index


def generate_simple_noncommutative_parameters():
    for solver in ["picos", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 1 / 8)]:
            if solver == "mosek":
                yield pytest.param(
                    solver,
                    level,
                    expected,
                    None,
                    marks=[
                        pytest.mark.skipif(
                            not is_mosek_available(),
                            reason="Mosek is not installed or a Mosek license is not available.",
                        )
                    ],
                )

            elif solver == "picos":
                yield pytest.param(solver, level, expected, "cvxopt")
                yield pytest.param(
                    solver,
                    level,
                    expected,
                    "mosek",
                    marks=[
                        pytest.mark.skipif(
                            not is_mosek_available(),
                            reason="Mosek is not installed or a Mosek license is not available.",
                        )
                    ],
                )


def generate_simple_noncommutative_with_substitution_parameters():
    for solver in ["picos", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 2.15e-05)]:
            if solver == "mosek":
                yield pytest.param(
                    solver,
                    level,
                    expected,
                    None,
                    marks=[
                        pytest.mark.skipif(
                            not is_mosek_available(),
                            reason="Mosek is not installed or a Mosek license is not available.",
                        )
                    ],
                )

            elif solver == "picos":
                yield pytest.param(solver, level, expected, "cvxopt")
                yield pytest.param(
                    solver,
                    level,
                    expected,
                    "mosek",
                    marks=[
                        pytest.mark.skipif(
                            not is_mosek_available(),
                            reason="Mosek is not installed or a Mosek license is not available.",
                        )
                    ],
                )


@pytest.mark.parametrize("solver, level, expected, picos_solver", generate_simple_noncommutative_parameters())
@pytest.mark.parametrize("force_primal", [True, False])
def test_simple_real_noncommutative_problem(
    solver: str, level: int, expected: float, force_primal: bool, picos_solver: str | None
):
    x1, x2 = generate_noncommutative_variables("x", 2, starting_index=1, hermitian=True)
    obj = x2**2 - x1 * x2 / 2 - x2 * x1 / 2 - x2
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0]

    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)

    if solver == "picos":
        solution = solve(sdp, "max", force_primal=force_primal, picos_solver=picos_solver)
    elif solver == "mosek":
        solution = solve(sdp, "max", force_primal=force_primal)

    assert solution.value == pytest.approx(expected)


@pytest.mark.parametrize(
    "solver, level, expected, picos_solver", generate_simple_noncommutative_with_substitution_parameters()
)
@pytest.mark.parametrize("force_primal", [True, False])
def test_simple_real_noncommutative_problem_with_commutative_substitution(
    solver: str, level: int, expected: float, force_primal: bool, picos_solver: str | None
):
    x1, x2 = generate_noncommutative_variables("x", 2, starting_index=1, hermitian=True)
    obj = x2**2 - x1 * x2 / 2 - x2 * x1 / 2 - x2
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0]
    substitutions = {x2 * x1: x1 * x2}

    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints, substitutions=substitutions)

    if solver == "picos":
        solution = solve(sdp, "max", force_primal=force_primal, picos_solver=picos_solver)
    elif solver == "mosek":
        solution = solve(sdp, "max", force_primal=force_primal)

    assert solution.value == pytest.approx(expected, abs=1e-6)
