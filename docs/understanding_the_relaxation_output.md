# Understanding the relaxation output

This page is a **contributor reference**. It explains what a relaxation object
produced by `get_relaxation` actually stores, how moments are deduplicated into a
canonical form, and which SDP — primal *and* dual — that data encodes. It is the
bridge between the maths (the moment/SOS hierarchy) and the Python objects the
rest of the codebase manipulates.

If you only want to *use* the package, read the README instead. This page assumes
you are extending ncpoleon, e.g. adding a solver backend or a new constraint type.

---

## 1. Canonical moments: how monomials are deduplicated

A monomial and its adjoint are not independent variables: they are conjugates of
each other. For a non-commutative problem with Hermitian $X_1$ and $X_2$, the
adjoint of $X_1X_2$ is $X_2X_1$, so in the moment matrix the entries for $X_1X_2$
and $X_2X_1$ are complex conjugates. Storing both would be redundant.

ncpoleon therefore keeps only the **first monomial it encounters** in each
conjugate pair and calls it the *canonical* monomial. Every other monomial is
reduced to its canonical form on lookup.

> **The canonical form is an internal detail — users never see it.** A user can
> query the value of *any* monomial, whether it happens to be the canonical
> representative or its adjoint, and get the right answer with no special
> handling on their side. The lookup reduces the monomial to its canonical form,
> fetches the stored value, and conjugates it automatically when the queried
> monomial is the adjoint (see `BaseSolution.__getitem__`). The rest of this
> section concerns contributors writing such lookups, not callers performing
> them.

Concretely, `MomentMatrix.get_canonical(monomial)` returns

```python
(canonical_monomial, is_adjoint, is_real_valued)
```

- `canonical_monomial` — the representative the moment is actually stored under.
- `is_adjoint` — whether the queried monomial is the *adjoint* of the canonical
  one (so its value is the conjugate of the stored value).
- `is_real_valued` — whether the moment is real. This is `True` when the whole
  problem is real (`relaxation.is_real`) **or** when the monomial is Hermitian
  (it equals its own adjoint), so no imaginary part exists.

### Where a canonical monomial lives in the matrix

For each canonical monomial, `MomentMatrix.data` stores a `PositionMatrixPair`:

```python
data: dict[monomial, PositionMatrixPair]
PositionMatrixPair = tuple[PositionMatrix, PositionMatrix | None]
PositionMatrix     = dict[tuple[int, int], Scalar]   # (row, col) -> coefficient
```

- The **first** `PositionMatrix` lists every `(row, col)` where the canonical
  monomial appears, together with its coefficient.
- The **second** lists every `(row, col)` where its **adjoint** appears.
- The second entry is `None` when the moment is real-valued (real problem or
  Hermitian monomial) — there is no separate conjugate slot to track.

In the notation of [§4](#4-the-sdp-encoded-by-the-relaxation), the first
`PositionMatrix` is the sparse matrix $F_{a,b}$ and the second is $F_{a,b}^\dagger$.
`as_row_col_data_format()` returns the same data as `(rows, cols, values)`
triples, which is what the solver backends consume.

---

## 2. Attributes of the relaxation object

Every relaxation subclasses `BaseSdpRelaxation`. The concrete class encodes two
binary choices — commutative vs. non-commutative variables, and real vs. complex
coefficients:

| Class | Variables | Coefficients |
|---|---|---|
| `RealValuedCommutativeSdpRelaxation` | commutative | real |
| `ComplexValuedCommutativeSdpRelaxation` | commutative | complex |
| `RealValuedNonCommutativeSdpRelaxation` | non-commutative | real |
| `ComplexValuedNonCommutativeSdpRelaxation` | non-commutative | complex |

The real/complex distinction is what drives the two SDP forms in [§4](#4-the-sdp-encoded-by-the-relaxation);
`is_real` tells the backends which one to build.

Main attributes:

| Attribute | Meaning |
|---|---|
| `objective` | The polynomial being optimised; its coefficients are the $\alpha_{a,b}$ (complex moments) and $\mu_{a,b}$ (real moments). |
| `moment_matrices` | `dict[int, MomentMatrix]`, keyed by moment-matrix id $a$. The PSD blocks of the relaxation. |
| `localising_moment_matrices_inequalities` | `dict[int, list[MomentMatrix]]` — the localising matrices from operator **inequalities** ($G$ blocks, PSD). |
| `localising_moment_matrices_equalities` | `dict[int, list[MomentMatrix]]` — localising matrices from operator **equalities** ($H$ blocks, $=0$). |
| `moment_inequalities` | `list[(polynomial, float)]` — scalar moment inequalities $\langle p\rangle \geqslant \gamma_m$. |
| `moment_equalities` | `list[(polynomial, Scalar)]` — scalar moment equalities $\langle p\rangle = \zeta_n$. |
| `is_real` | Whether the relaxation is real-valued (selects the real vs. complex SDP form). |
| `generating_sets` | `dict[int, list[monomial]]` — the monomial basis of each moment matrix. |
| `equalities` / `inequalities` | The operator constraints behind the localising matrices, per moment-matrix id. |

Helper methods worth knowing:

- `reduce_monomial(m)` — rewrite a monomial under the substitution rules, then to
  its canonical form (the first thing every solver does on lookup).
- `split_into_real_and_imaginary_parts(poly)` — split a polynomial's coefficients
  into real/imaginary parts.
- `change_variables(poly, mapping)` — substitute moments elements that can be multiplied by scalars and added together.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $\Gamma_a$ | The moment matrix of index $a$ |
| $z_{a,b}=x_{a,b}+\mathrm{i}y_{a,b}$ | A canonical complex moment in $\Gamma_a$ |
| $r_{a,c}$| A real moment in $\Gamma_a$ |
| $F_{a,b}$  | (Weighted) mask indicating where the complex moment $z_{a, b}$ sits in $\Gamma_a$ |
| $F_{a,b}^\dagger$  | (Weighted) mask indicating where the complex moment $\overline{z_{a, b}}$ sits in $\Gamma_a$ |
| $F^r_{a,c}$  | (Weighted) mask indicating where the real moment $r_{a, c}$ sits in $\Gamma_a$ |
| $p(X)$  | Objective polynomial |
| $\alpha_{a,b}$ | Coefficient of the canonical complex moment $z_{a, b}$ in $p$ |
| $\mu_{a,c}$ | Coefficient of the real moment $r_{a, c}$ in $p$ |
| $g_{a, k}(X)$  | $k$-th operator inequality polynomial associated with $\Gamma_a$ |
| $g_{a, k}$  | The localizing moment matrix associated to $g_{a, k}(X)$ |
| $G_{a,b,k}$ | (Weighted) mask indicating where the complex moment $z_{a, b}$ sits in $g_{a, k}$ |
| $G_{a,b,k}^\dagger$ | (Weighted) mask indicating where the complex moment $\overline{z_{a, b}}$ sits in $g_{a, k}$ |
| $G_{a,c,k}^r$ | (Weighted) mask indicating where the real moment $r_{a, c}$ sits in $g_{a, k}$ |
| $h_{a,\ell}(X)$  | $\ell$-th operator equality polynomial associated with $\Gamma_a$ |
| $h_{a, \ell}$  | The localizing moment matrix associated to $h_{a, \ell}(X)$ |
| $H_{a,b,\ell}$ | (Weighted) mask indicating where the complex moment $z_{a, b}$ sits in $h_{a, \ell}$ |
| $H_{a,b,\ell}^\dagger$ | (Weighted) mask indicating where the complex moment $\overline{z_{a, b}}$ sits in $h_{a, \ell}$ |
| $H_{a,c,\ell}^r$ | (Weighted) mask indicating where the real moment $r_{a, c}$ sits in $h_{a, \ell}$ |
| $q_m$ | $m$-th moment inequality polynomial |
| $\gamma_m$ | $m$-th moment inequality polynomial's lower-bound |
| $\beta_{a, b, m}$ | Coefficient of the complex moment $z_{a, b}$ in $q_m$ |
| $\beta_{a, c, m}^r$ | Coefficient of the real moment $r_{a, c}$ in $q_m$ |
| $r_n$ | $n$-th moment equality polynomial |
| $\zeta_n$ | $n$-th moment inequality polynomial's constraint |
| $\delta_{a, b, m}$ | Coefficient of the complex moment $z_{a, b}$ in $r_n$ |
| $\varepsilon{a, b, m}$ | Coefficient of the complex moment $\overline{z_{a, b}}$ in $r_n$ |
| $\delta_{a, b, m}^r$ | Coefficient of the real moment $r_{a, c}$ in $r_n$ |
| $Y_a$ | Dual variable of $\Gamma_a$ |
| $P_{a,k}$ | Dual variable of $g_{a,k}$ |
| $Q_{a,\ell}$ | Dual variable of $h_{a,\ell}$ |
| $\lambda_m$ | Dual variable of $q_m$ |
| $\nu_n$ | Dual variable of $r_n$ |

Note that technically, we shouldn't consider the same indexing sets for the objective polynomial (or at least the moment matrices) and the localizing matrices, since the latter are generally defined over a smaller set of indexing monomials. This won't matter in the derivation of the SoS decomposition as long as we keep this fact in mind.

All in all, the problem we're considering a relaxation of is

$$
\begin{aligned}
\min&\quad \operatorname{tr}\!\left[\rho p(X)\right]\\
\text{s.t.}&\quad\forall a,k,\ g_{a, k}(X)\succeq0\\
&\quad\forall a,\ell,\ h_{a,\ell}(X)=0\\
&\quad\forall m,\ \operatorname{tr}\!\left[\rho q_m(X)\right]\geqslant\gamma_m\\
&\quad\forall n,\ \operatorname{tr}\!\left[\rho r_n(X)\right]=\zeta_n\\
&\quad\rho\succeq0\ .
\end{aligned}
$$

---

## 4. The SDP encoded by the relaxation

With complex canonical moments $z_{a,b}$ and the real ones $r_{a,c}$, the primal is

$$
\begin{aligned}
\min&\quad \sum_{a, b}\alpha_{a, b}z_{a, b}+\sum_{a, b}\overline{\alpha_{a, b}}\,\overline{z_{a, b}}+\sum_{a, c}\mu_{a, c}r_{a, c}\\
\text{s.t.}&\quad\forall a,\ \sum_bz_{a, b}F_{a, b}+\sum_b\overline{z_{a, b}}F_{a, b}^\dagger+\sum_c r_{a, c}F^r_{a, c}\succeq0
&\text{(Moment matrices)}\\
&\quad\forall a,k,\ \sum_bz_{a, b}G_{a, b, k}+\sum_b\overline{z_{a, b}}G_{a, b, k}^\dagger+\sum_c r_{a, c}G^r_{a, c, k}\succeq0
&\text{(Operator inequalities)}\\
&\quad\forall a,\ell,\ \sum_bz_{a, b}H_{a, b, \ell}+\sum_b\overline{z_{a, b}}H_{a, b, \ell}^\dagger+\sum_c r_{a, c}H^r_{a, c, \ell}=0
&\text{(Operator equalities)}\\
&\quad\forall m,\ \sum_{a, b}z_{a, b}\beta_{a, b, m}+\sum_{a, b}\overline{z_{a, b}}\,\overline{\beta_{a, b, m}}+\sum_{a, c}r_{a, c}\beta^r_{a, c, m}\geqslant\gamma_m
&\text{(Moment inequalities)}\\
&\quad\forall n,\ \sum_{a, b}z_{a, b}\delta_{a, b, n}+\sum_{a, b}\overline{z_{a, b}}\,\varepsilon_{a, b, n}+\sum_{a, c}r_{a, c}\delta^r_{a, c, n}=\zeta_n
&\text{(Moment equalities)}
\end{aligned}
$$

The resulting dual is

$$
\begin{aligned}
\max&\quad\sum_m\lambda_m\gamma_m+\Re\!\left(\sum_n\overline{\nu_n}\,\zeta_n\right)\\
\text{s.t.}&\quad\forall(a,b),\
\alpha_{a,b}=\operatorname{tr}\!\left(Y_aF_{a,b}\right)
+\sum_k\operatorname{tr}\!\left(P_{a,k}G_{a,b,k}\right)
+\sum_\ell\operatorname{tr}\!\left(Q_{a,\ell}H_{a,b,\ell}\right)
+\sum_m\lambda_m\beta_{a,b,m}
+\frac12\sum_n\!\left(\overline{\nu_n}\delta_{a,b,n}+\nu_n\overline{\varepsilon_{a,b,n}}\right)\\
&\quad\forall(a,c),\
\mu_{a,c}=\operatorname{tr}\!\left(Y_aF^r_{a,c}\right)
+\sum_k\operatorname{tr}\!\left(P_{a,k}G^r_{a,c,k}\right)
+\sum_\ell\operatorname{tr}\!\left(Q_{a,\ell}H^r_{a,c,\ell}\right)
+\sum_m\lambda_m\beta^r_{a,c,m}
+\sum_n\Re(\nu_n)\,\delta^r_{a,c,n}\\
&\quad Y_a\succeq0,\ P_{a,k}\succeq0,\ Q_{a,\ell}=Q_{a,\ell}^\dagger,\
\lambda_m\geqslant0,\ \nu_n\in\mathbb{C}
\end{aligned}
$$

The two dual constraints mirror the two kinds of moment: the first (complex) pins
the objective weight $\alpha_{a,b}$ of each complex moment, the second (real) pins
the weight $\mu_{a,b}$ of each real moment.

??? note "Full derivation: real reformulation, Lagrangian, and dual"

    To pose this over real variables, each complex moment is written as

    $$
    z_{a,b}=x_{a,b}+\mathrm{i}\,y_{a,b}
    $$

    Two consequences are worth keeping in mind:

    - **A complex moment becomes two real variables** $x_{a,b}$ and $y_{a,b}$.
    - **A complex equality becomes two real equalities** — its real and imaginary
      parts — whereas a complex inequality stays a single inequality (only its real
      part is constrained).

    This gives the real SDP

    $$
    \begin{aligned}
    \min&\quad 2\sum_{a, b}\left(\Re(\alpha_{a, b})x_{a, b}-\Im(\alpha_{a, b})y_{a, b}\right)+\sum_{a, b}\mu_{a, b}r_{a, b}\\
    \text{s.t.}&\quad\forall a,\sum_bx_{a, b}\left(F_{a, b}+F_{a, b}^\dagger\right)+\sum_by_{a, b}\mathrm{i}\left(F_{a, b}-F_{a, b}^\dagger\right)+\sum_b r_{a, b}F^r_{a, b}\succeq0\\
    &\quad\forall a,k,\sum_bx_{a, b}\left(G_{a, b, k}+G_{a,b,k}^\dagger\right)+\sum_by_{a, b}\mathrm{i}\left(G_{a, b, k}-G_{a, b, k}^\dagger\right)+\sum_b r_{a, b}G^r_{a, b, k}\succeq0\\
    &\quad\forall a,\ell,\sum_bx_{a, b}\left(H_{a, b, \ell}+H_{a, b, \ell}^\dagger\right)+\sum_by_{a, b}\mathrm{i}\left(H_{a, b, \ell}-H_{a, b, \ell}^\dagger\right)+\sum_b r_{a, b}H^r_{a, b, \ell}=0\\
    &\quad\forall m,2\sum_{a, b}\left(\Re(\beta_{a, b, m})x_{a, b}-\Im(\beta_{a, b, m})y_{a, b}\right)+\sum_{a, b}\beta^r_{a, b, m}r_{a, b}\geqslant\gamma_m\\
    &\quad\forall n,\sum_{a, b}\Re(\delta_{a, b, n}+\varepsilon_{a, b, n})x_{a, b}-\Im(\delta_{a, b, n}-\varepsilon_{a, b, n})y_{a, b}+\sum_{a, b}\delta^r_{a, b, n}r_{a, b}=\Re(\zeta_n)\\
    &\quad\forall n,\sum_{a, b}\Im(\delta_{a, b, n}+\varepsilon_{a, b, n})x_{a, b}+\Re(\delta_{a, b, n}-\varepsilon_{a, b, n})y_{a, b}=\Im(\zeta_n)
    \end{aligned}
    $$

    The Lagrangian attaches $Y_a\succeq0$ to each moment matrix, $P_{a,k}\succeq0$
    to each localising inequality, $Q_{a,\ell}$ (Hermitian, free) to each
    localising equality, $\lambda_m\geqslant0$ to each moment inequality, and the
    real pair $\nu_n^{\Re},\nu_n^{\Im}$ to the two halves of each complex moment
    equality:

    $$
    \begin{aligned}
    \mathcal{L}=\,&2\sum_{a, b}\left(\Re(\alpha_{a, b})x_{a, b}-\Im(\alpha_{a, b})y_{a, b}\right)+\sum_{a, b}\mu_{a, b}r_{a, b}\\
    &-\sum_a\operatorname{tr}\!\left(Y_a\left(\sum_bx_{a, b}\left(F_{a, b}+F_{a, b}^\dagger\right)+\sum_by_{a, b}\mathrm{i}\left(F_{a, b}-F_{a, b}^\dagger\right)+\sum_b r_{a, b}F^r_{a, b}\right)\right)\\
    &-\sum_{a,k}\operatorname{tr}\!\left(P_{a,k}\left(\sum_bx_{a, b}\left(G_{a, b, k}+G_{a, b, k}^\dagger\right)+\sum_by_{a, b}\mathrm{i}\left(G_{a, b, k}-G_{a, b, k}^\dagger\right)+\sum_b r_{a, b}G^r_{a, b, k}\right)\right)\\
    &-\sum_{a,\ell}\operatorname{tr}\!\left(Q_{a,\ell}\left(\sum_bx_{a, b}\left(H_{a, b, \ell}+H_{a, b, \ell}^\dagger\right)+\sum_by_{a, b}\mathrm{i}\left(H_{a, b, \ell}-H_{a, b, \ell}^\dagger\right)+\sum_b r_{a, b}H^r_{a, b, \ell}\right)\right)\\
    &-\sum_m\lambda_m\!\left(2\sum_{a, b}\left(\Re(\beta_{a, b, m})x_{a, b}-\Im(\beta_{a, b, m})y_{a, b}\right)+\sum_{a, b}\beta^r_{a, b, m}r_{a, b}-\gamma_m\right)\\
    &-\sum_n\nu^{\Re}_n\!\left(\sum_{a, b}\Re(\delta_{a, b, n}+\varepsilon_{a, b, n})x_{a, b}-\Im(\delta_{a, b, n}-\varepsilon_{a, b, n})y_{a, b}+\sum_{a, b}\delta^r_{a, b, n}r_{a, b}-\Re(\zeta_n)\right)\\
    &-\sum_n\nu^{\Im}_n\!\left(\sum_{a, b}\Im(\delta_{a, b, n}+\varepsilon_{a, b, n})x_{a, b}+\Re(\delta_{a, b, n}-\varepsilon_{a, b, n})y_{a, b}-\Im(\zeta_n)\right)
    \end{aligned}
    $$

    Setting $\partial\mathcal{L}/\partial x_{a,b}=0$, $\partial\mathcal{L}/\partial y_{a,b}=0$
    and $\partial\mathcal{L}/\partial r_{a,b}=0$ gives one stationarity condition
    per real variable.

    $$
    \begin{aligned}
    2\Re(\alpha_{a,b})&=\operatorname{tr}\!\left(Y_a\!\left(F_{a,b}+F_{a,b}^\dagger\right)\right)+\sum_k\operatorname{tr}\!\left(P_{a,k}\!\left(G_{a,b,k}+G_{a,b,k}^\dagger\right)\right)+\sum_\ell\operatorname{tr}\!\left(Q_{a,\ell}\!\left(H_{a,b,\ell}+H_{a,b,\ell}^\dagger\right)\right)\\
    &\quad+2\sum_m\lambda_m\Re(\beta_{a,b,m})+\sum_n\!\left(\nu_n^{\Re}\Re(\delta_{a,b,n}+\varepsilon_{a,b,n})+\nu_n^{\Im}\Im(\delta_{a,b,n}+\varepsilon_{a,b,n})\right)\\
    2\Im(\alpha_{a,b})&=-\operatorname{tr}\!\left(Y_a\,\mathrm{i}\!\left(F_{a,b}-F_{a,b}^\dagger\right)\right)-\sum_k\operatorname{tr}\!\left(P_{a,k}\,\mathrm{i}\!\left(G_{a,b,k}-G_{a,b,k}^\dagger\right)\right)-\sum_\ell\operatorname{tr}\!\left(Q_{a,\ell}\,\mathrm{i}\!\left(H_{a,b,\ell}-H_{a,b,\ell}^\dagger\right)\right)\\
    &\quad+2\sum_m\lambda_m\Im(\beta_{a,b,m})+\sum_n\!\left(\nu_n^{\Re}\Im(\delta_{a,b,n}-\varepsilon_{a,b,n})-\nu_n^{\Im}\Re(\delta_{a,b,n}-\varepsilon_{a,b,n})\right)\\
    \mu_{a,b}&=\operatorname{tr}\!\left(Y_aF^r_{a,b}\right)+\sum_k\operatorname{tr}\!\left(P_{a,k}G^r_{a,b,k}\right)+\sum_\ell\operatorname{tr}\!\left(Q_{a,\ell}H^r_{a,b,\ell}\right)+\sum_m\lambda_m\beta^r_{a,b,m}+\sum_n\nu_n^{\Re}\delta^r_{a,b,n}
    \end{aligned}
    $$

    Defining $\nu_n=\nu_n^{\Re}+\mathrm{i}\nu_n^{\Im}$, the $x$- and $y$-conditions recombine into the single complex constraint

    $$
    \alpha_{a,b}=\operatorname{tr}\!\left(Y_aF_{a,b}\right)+\sum_k\operatorname{tr}\!\left(P_{a,k}G_{a,b,k}\right)+\sum_\ell\operatorname{tr}\!\left(Q_{a,\ell}H_{a,b,\ell}\right)+\sum_m\lambda_m\beta_{a,b,m}+\frac12\sum_n\!\left(\overline{\nu_n}\delta_{a,b,n}+\nu_n\overline{\varepsilon_{a,b,n}}\right)
    $$

    while the $r$-condition becomes (with $\nu_n^{\Re}=\Re(\nu_n)$)

    $$
    \mu_{a,b}=\operatorname{tr}\!\left(Y_aF^r_{a,b}\right)+\sum_k\operatorname{tr}\!\left(P_{a,k}G^r_{a,b,k}\right)+\sum_\ell\operatorname{tr}\!\left(Q_{a,\ell}H^r_{a,b,\ell}\right)+\sum_m\lambda_m\beta^r_{a,b,m}+\sum_n\Re(\nu_n)\delta^r_{a,b,n}
    $$

    The terms of $\mathcal{L}$ independent of the primal variables form the dual
    objective $\sum_m\lambda_m\gamma_m+\Re\!\left(\sum_n\overline{\nu_n}\zeta_n\right)$,
    yielding the complex dual stated above.

## 5. How to extract the SoS decomposition from the dual variables

This section describes how to extract the SoS decomposition of the objective polynomial from the optimal dual variables. Note that the dual formulation allows to express the coefficients of the objective polynomial as a function of the dual variables. That is, we have

$$
\begin{align*}
  &\sum_{a, b}\alpha_{a, b}X_{a, b} + \sum_{a, b}\overline{\alpha_{a, b}}X_{a, b}^\dagger+\sum_{a, c}r_{a, c}X_{a, c}\\
  ={}&\sum_{a, b}\left[\operatorname{tr}\!\left(Y_aF_{a,b}\right)
    +\sum_k\operatorname{tr}\!\left(P_{a,k}G_{a,b,k}\right)
    +\sum_\ell\operatorname{tr}\!\left(Q_{a,\ell}H_{a,b,\ell}\right)
    +\sum_m\lambda_m\beta_{a,b,m}
    +\frac12\sum_n\!\left(\overline{\nu_n}\delta_{a,b,n}+\nu_n\overline{\varepsilon_{a,b,n}}\right)\right]X_{a, b}+{}\\
  &\sum_{a, b}\left[\operatorname{tr}\!\left(Y_aF_{a,b}^\dagger\right)
    +\sum_k\operatorname{tr}\!\left(P_{a,k}G_{a,b,k}^\dagger\right)
    +\sum_\ell\operatorname{tr}\!\left(Q_{a,\ell}H_{a,b,\ell}^\dagger\right)
    +\sum_m\lambda_m\overline{\beta_{a,b,m}}
    +\frac12\sum_n\!\left(\nu_n\overline{\delta_{a,b,n}}+\overline{\nu_n}\varepsilon_{a,b,n}\right)\right]X_{a, b}^\dagger+{}\\
  &\sum_{a, c}\left[\operatorname{tr}\!\left(Y_aF^r_{a,c}\right)
    +\sum_k\operatorname{tr}\!\left(P_{a,k}G^r_{a,c,k}\right)
    +\sum_\ell\operatorname{tr}\!\left(Q_{a,\ell}H^r_{a,c,\ell}\right)
    +\sum_m\lambda_m\beta^r_{a,c,m}
    +\sum_n\Re(\nu_n)\,\delta^r_{a,c,n}\right]X_{a, c}\,.
\end{align*}$$

Note that both sides of this equation are polynomials. By considering the trace as now operating over matrices of polynomials, we can write

$$
\begin{align*}
  &\sum_{a, b}\alpha_{a, b}X_{a, b} + \sum_{a, b}\overline{\alpha_{a, b}}X_{a, b}^\dagger+\sum_{a, b}r_{a, b}X_{a, b}\\
  ={}&\operatorname{tr}\!\left[Y_a\sum_{a, b}\left(X_{a, b}F_{a,b}+X_{a, b}^\dagger F_{a,b}^\dagger+X_{a, b}F^r_{a,b}\right)\right]
    +{}\\
    &\sum_k \operatorname{tr}\!\left[P_{a,k}\sum_{a, b}\left(X_{a, b}G_{a,b,k}+X_{a, b}^\dagger G_{a,b,k}^\dagger+X_{a, b}G^r_{a,b,k}\right)\right]
    +{}\\
    &\sum_\ell \operatorname{tr}\!\left[Q_{a,\ell} \sum_{a, b}\left(X_{a, b}H_{a,b,\ell}+X_{a, b}^\dagger H_{a,b,\ell}^\dagger+X_{a, b}H^r_{a,b,\ell}\right)\right]
    +{}\\
    &\sum_m\lambda_m\left[\sum_{a, b}\left(\beta_{a,b,m}X_{a, b}+\overline{\beta_{a, b, m}}X_{a, b}^\dagger+\sum_{a, c}\beta_{a, b, m}^rX_{a, b}\right)\right]
    +{}\\
    &\sum_{n}\Re\left(\nu_n\right)\left(\sum_{a, b}\left(\frac{\delta_{a, b, n}+\overline{\varepsilon_{a, b, n}}}{2}X_{a, b}+\frac{\overline{\delta_{a, b, n}}+\varepsilon_{a, b, n}}{2}X_{a, b}^\dagger\right)+\sum_{a, c}\delta_{a, c, n}^rX_{a, c}\right)+{}\\
    &\sum_n\mathrm{i}\Im\left(\nu_n\right)\sum_{a, b}\left(\frac{\overline{\varepsilon_{a, b, n}}-\delta_{a, b, n}}{2}X_{a, b}+\frac{\overline{\delta_{a, b, n}}-\varepsilon_{a, b, n}}{2}X_{a, b}^\dagger\right)
\end{align*}
$$

which we can rewrite as, by definition of the relevant quantities

$$
  p(X)=\operatorname{tr}\!\left[Y_a\Gamma_a\right]
    +\sum_k \operatorname{tr}\!\left[P_{a,k}g_{a,k}\right]
    +\sum_\ell \operatorname{tr}\!\left[Q_{a,\ell} h_{a,\ell}\right]
    +\sum_m\lambda_m q_m(X)
    +\sum_{n}\Re\left(\nu_n\right)\left(\frac{r_n(X)+r_n(X)^\dagger}{2}\right)+\mathrm{i}\Im\left(\nu_n\right)\left(\frac{r_n(X)^\dagger-r_n(X)}{2}\right)\ .
$$

We now want to express the first three terms as sums of squares. First of all, let us write the spectral decomposition of $Y_a$ as

$$
Y_a = \sum_j e_{a,j}\,s_{a, j}s_{a, j}^\dagger
$$

with each $s_{a,j}$ being normed. Then we know that $\Gamma_a$ can be written as

$$
\Gamma_a = W_aW_a^\dagger
$$

with $W_a$ being the generating set of monomials for this density matrix. This gives us

$$
\operatorname{tr}\!\left[Y_a\Gamma_a\right] = \operatorname{tr}\!\left[\sum_j e_{a,j}\,s_{a, j}s_{a, j}^\dagger W_aW_a^\dagger\right] = \sum_j e_{a,j}\,W_a^\dagger s_{a,j}s_{a,j}^\dagger W_a = \sum_j\left(\sqrt{e_{a,j}}s_{a,j}^\dagger W_a\right)^\dagger\left(\sqrt{e_{a,j}}s_{a,j}^\dagger W_a\right)
$$

which is a sum of squares. Thus, the elements of the SoS decomposition for this term are the elements of the vector $\sqrt{Y_a}W_a$.

Similarly, we can write the spectral decomposition of $P_{a,k}$ as

$$
P_{a,k} = \sum_j e_{a,k,j}\,s_{a,k, j}s_{a,k, j}^\dagger
$$

and we can write $g_{a,k}$ as $W_{a,k}g_{a,k}(X)W_{a,k}^\dagger$. We then have

$$
\operatorname{tr}\!\left[P_{a,k}g_{a,k}\right] = \operatorname{tr}\!\left[\sum_j e_{a,k,j}\,s_{a,k, j}s_{a,k, j}^\dagger W_{a,k}g_{a,k}(X)W_{a,k}^\dagger\right] = \operatorname{tr}\!\left[\sum_j e_{a,k,j}\,s_{a,k, j}^\dagger W_{a,k}g_{a,k}(X)W_{a,k}^\dagger s_{a,k, j}\right]=\sum_j \left(\sqrt{e_{a,k,j}}\,W_{a, k}^\dagger s_{a,k, j}\right)^\dagger g_{a,k}(X)\left(\sqrt{e_{a,k,j}}\,W_{a,k}^\dagger s_{a,k, j}\right)
$$

which means that the elements we're looking for are the elements of the vector $\sqrt{P_{a,k}}W_{a,k}$.

Finally, let us split $Q_{a,\ell}$ as

$$
Q_{a,\ell} = Q_{a,\ell}^+-Q_{a,\ell}^-
$$

with both components being positive semidefinite. We can then apply the same trick to get a positive term and a negative term in the sum of squares.