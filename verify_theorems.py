"""
Comprehensive numerical and high-precision verification suite for:
"Quantitative Stability, Exact CLT Asymptotics, and l^p-Stam Inequalities in Finite Free Information Theory"

Verifies:
1. Theorem 3.2: Exact eigenvalues {t^(k/2)} and orthonormal Hermite-Lagrange eigenvectors u^(k)
   of E_n(sigma_1, sigma_2) for arbitrary sigma_1, sigma_2 > 0 (Hashemi, 2026, Conj. 4.1; Hashemi-Hyun, 2026, Thm. A.1).
2. Theorem 3.3 & Corollary 3.4: Log-gas Laplacian generator L_n, semigroup identity E_n(sqrt(t), sqrt(1-t)) = exp((ln t)/2 * L_n),
   dual (reversed-weight) truncated Mehler kernel sign law, and strict entrywise positivity.
3. Theorem 4.1, Corollary 4.2, Remark 4.3, Theorem 4.4, Prop. 4.5, Theorem 4.6, Prop. 4.7, Theorem 4.8, Remark 4.9:
   Exact block-diagonal Stam deficit Hessian, sharp 2t(1-t) local stability, all-degree n>=4 local non-attainment at every
   Hermite pair (sigma_1 r, sigma_2 r), n>=4 unnormalized CR obstruction, global n=3 cubic stability, Dyson-flow O(1/n) decay,
   PSD log-gas matrix Stam inequality K(beta) >= 0 and exact c_n^(100) <= 1/(n-1) (plus 50-digit c_4^(100) <= 0.3141332), and
   exact integer quartic bound c_4 <= 876/1855.
4. Theorem 5.1 & Corollary 5.2: Exact finite free CLT convergence rates O(1/m), O(1/m^2), O(1/m^3) with parity-sharpened
   O(m^{-(j-1)}) remainders, universal exact two-term Edgeworth expansion (independent of gamma_5,...,gamma_n), and CLT-tail
   non-attainment cone.
5. Propositions 6.1-6.2, Theorem 6.3, Remark 6.4: Jacobian score identity J_alpha s(alpha) = s(gamma), doubly stochasticity,
   unequal-scale p-Stam phase transition at p* = 2, proof of Conjecture 5.2 for n=3, n=4 uniform rank-1 shift obstruction
   vs non-uniform shift feasibility, n=5 certified linear-extension no-go, and quotient l^q dual checks.

Also generates publication figures:
- fig_spectrum_semigroup.png
- fig_stam_clt.png
"""

import argparse
from fractions import Fraction
from math import factorial
import numpy as np
import mpmath as mp
from mpmath import iv
from numpy.polynomial.hermite_e import hermegauss, hermeval
from scipy.linalg import expm
from scipy.optimize import linprog, minimize, minimize_scalar
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers for finite free additive convolution and score vectors
# ---------------------------------------------------------------------------

def roots_to_elementary(roots: np.ndarray) -> np.ndarray:
    """Return [e_0, e_1, ..., e_n] where p(x) = sum_{k=0}^n (-1)^k e_k x^{n-k}."""
    poly = np.poly(roots)
    n = len(roots)
    return np.array([((-1) ** k) * poly[k] for k in range(n + 1)], dtype=float)


def boxplus_coeffs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Finite free additive convolution in the signed elementary coefficient basis:
    a = [a_0, ..., a_n] with p(x) = sum_{k=0}^n (-1)^k a_k x^{n-k} (a_k = e_k(roots)).
    Note: because (-1)^i (-1)^j = (-1)^{i+j}, the same linear formula also holds on
    plain descending polynomial coefficients [(-1)^k a_k]."""
    n = len(a) - 1
    c = np.zeros(n + 1, dtype=float)
    for k in range(n + 1):
        for i in range(k + 1):
            j = k - i
            c[k] += (
                factorial(n - i) * factorial(n - j) / (factorial(n) * factorial(n - k))
            ) * a[i] * b[j]
    return c


def real_roots_checked(poly: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Compute sorted real roots of a real-rooted polynomial, asserting negligible imaginary parts."""
    raw = np.roots(poly)
    max_im = float(np.max(np.abs(raw.imag)))
    assert max_im < tol, f"Unexpected complex root residual: max |Im| = {max_im:.3e}"
    return np.sort(raw.real)


def boxplus_roots(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute sorted roots and monic polynomial coefficients of p_alpha boxplus_n p_beta."""
    n = len(alpha)
    a = roots_to_elementary(alpha)
    b = roots_to_elementary(beta)
    c = boxplus_coeffs(a, b)
    poly = np.array([((-1) ** k) * c[k] for k in range(n + 1)], dtype=float)
    roots = real_roots_checked(poly)
    return roots, poly


def score_vector(roots: np.ndarray) -> np.ndarray:
    """Compute s_i(alpha) = sum_{j != i} 1 / (alpha_i - alpha_j)."""
    diff = roots[:, None] - roots[None, :]
    np.fill_diagonal(diff, np.inf)
    return np.sum(1.0 / diff, axis=1)


def jacobian_alpha(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the root-map Jacobian d(gamma)/d(alpha) and output roots gamma."""
    n = len(alpha)
    p_poly = np.poly(alpha)
    b = roots_to_elementary(beta)
    gamma, base_poly = boxplus_roots(alpha, beta)
    base_deriv = np.polyder(base_poly)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        q_i, _ = np.polydiv(p_poly, np.array([1.0, -alpha[i]]))
        dp_poly = np.concatenate([[0.0], -q_i])
        da = np.array([((-1) ** k) * dp_poly[k] for k in range(n + 1)], dtype=float)
        dc = boxplus_coeffs(da, b)
        dc_poly = np.array([((-1) ** k) * dc[k] for k in range(n + 1)], dtype=float)
        for m in range(n):
            J[m, i] = -np.polyval(dc_poly, gamma[m]) / np.polyval(base_deriv, gamma[m])
    return J, gamma


def log_gas_laplacian(roots: np.ndarray) -> np.ndarray:
    """Weighted graph Laplacian L_ij = -2/(r_i - r_j)^2 (i != j), L_ii = -sum_{j!=i} L_ij."""
    diff = roots[:, None] - roots[None, :]
    np.fill_diagonal(diff, np.inf)
    L = -2.0 / (diff ** 2)
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(L, -np.sum(L, axis=1))
    return L


def hermite_eigenbasis(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted He_n roots r and orthogonal matrix U with columns u^(0), ..., u^(n-1)."""
    r, _ = hermegauss(n)
    r = np.sort(r)
    U = np.zeros((n, n), dtype=float)
    c_nm1 = np.zeros(n)
    c_nm1[n - 1] = 1.0
    hnp = n * hermeval(r, c_nm1)
    for k in range(n):
        m = n - 1 - k
        c_m = np.zeros(m + 1)
        c_m[m] = 1.0
        U[:, k] = np.sqrt(factorial(n) / factorial(m)) * hermeval(r, c_m) / hnp
    return r, U


# ---------------------------------------------------------------------------
# High-precision mpmath verification
# ---------------------------------------------------------------------------

def mp_he(n: int, x: mp.mpf) -> mp.mpf:
    if n == 0:
        return mp.mpf(1)
    if n == 1:
        return x
    h0, h1 = mp.mpf(1), x
    for k in range(2, n + 1):
        h0, h1 = h1, x * h1 - (k - 1) * h0
    return h1


def mp_real_roots_checked(coeffs: list[mp.mpf], maxsteps: int = 200, tol: str = "1e-30") -> list[mp.mpf]:
    roots = mp.polyroots(coeffs, maxsteps=maxsteps)
    max_im = max(abs(mp.im(z)) for z in roots)
    assert max_im < mp.mpf(tol), f"Unexpected mpmath complex root residual: {max_im}"
    return sorted([mp.re(z) for z in roots])


def mp_he_roots(n: int) -> list[mp.mpf]:
    coeffs = [mp.mpf(0)] * (n + 1)
    for m in range(n // 2 + 1):
        coeffs[2 * m] = (
            ((-1) ** m) * mp.factorial(n)
            / (mp.power(2, m) * mp.factorial(m) * mp.factorial(n - 2 * m))
        )
    return mp_real_roots_checked(coeffs)


def verify_mpmath_high_precision() -> None:
    mp.mp.dps = 50
    print("=== 1. High-Precision (50-digit) Verification of Eigenbasis & L_n ===")
    for n in [2, 3, 5, 10, 15]:
        r = mp_he_roots(n)
        L = mp.matrix(n, n)
        for i in range(n):
            rowsum = mp.mpf(0)
            for j in range(n):
                if i != j:
                    val = -mp.mpf(2) / ((r[i] - r[j]) ** 2)
                    L[i, j] = val
                    rowsum += val
            L[i, i] = -rowsum

        U = mp.matrix(n, n)
        for k in range(n):
            m = n - 1 - k
            scale = mp.sqrt(mp.factorial(n) / mp.factorial(m))
            for i in range(n):
                hnp = n * mp_he(n - 1, r[i])
                U[i, k] = scale * mp_he(m, r[i]) / hnp

        UtU = U.T * U
        err_orth = max(abs(UtU[i, j] - (1 if i == j else 0)) for i in range(n) for j in range(n))
        LU = L * U
        err_eig = max(abs(LU[i, k] - k * U[i, k]) for i in range(n) for k in range(n))
        assert err_orth < mp.mpf("1e-45"), f"Orthogonality failed at n={n}"
        assert err_eig < mp.mpf("1e-45"), f"Eigenvalue equation failed at n={n}"
        print(f"  n={n:2d}: ||U^T U - I||_max = {mp.nstr(err_orth, 3)}, ||L U - U diag(0..n-1)||_max = {mp.nstr(err_eig, 3)}")


def verify_coupling_semigroup_and_mehler() -> None:
    print("\n=== 2. Verification of E_n(sigma_1, sigma_2), Semigroup & Mehler Kernel ===")
    for n in range(2, 9):
        r, U = hermite_eigenbasis(n)
        L = log_gas_laplacian(r)
        for s1, s2 in [(1.0, 1.0), (1.0, 2.0), (2.5, 1.2)]:
            t = (s1 ** 2) / (s1 ** 2 + s2 ** 2)
            J, _ = jacobian_alpha(s1 * r, s2 * r)
            E_spec = U @ np.diag([t ** (k / 2.0) for k in range(n)]) @ U.T
            E_semi = expm(0.5 * np.log(t) * L)

            # Truncated Mehler formula
            c_nm1 = np.zeros(n)
            c_nm1[n - 1] = 1.0
            hnp = n * hermeval(r, c_nm1)
            E_mehler = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for m in range(n):
                        cm = np.zeros(m + 1)
                        cm[m] = 1.0
                        s += (t ** ((n - 1 - m) / 2.0)) * hermeval(r[i], cm) * hermeval(r[j], cm) / factorial(m)
                    E_mehler[i, j] = factorial(n) / (hnp[i] * hnp[j]) * s

            err_spec = np.max(np.abs(J - E_spec))
            err_semi = np.max(np.abs(J - E_semi))
            err_mehler = np.max(np.abs(J - E_mehler))
            row_err = np.max(np.abs(J.sum(axis=1) - 1.0))
            col_err = np.max(np.abs(J.sum(axis=0) - 1.0))
            min_entry = np.min(J)
            assert err_spec < 1e-11 and err_semi < 1e-11 and err_mehler < 1e-11
            assert row_err < 1e-11 and col_err < 1e-11 and min_entry > 0.0
        # Print summary for t = 1/2 (s1=s2=1)
        J_half, _ = jacobian_alpha(r, r)
        sv = np.linalg.svd(J_half, compute_uv=False)
        exp_sv = np.array([2.0 ** (-k / 2.0) for k in range(n)])
        print(f"  n={n}: sv_err(t=1/2) = {np.max(np.abs(sv - exp_sv)):.2e}, min_entry = {np.min(J_half):.6f}")


def verify_stam_hessian() -> None:
    print("\n=== 3. Verification of Exact Stam Deficit Hessian ===")
    eps = 1e-4
    for n in [3, 4, 5, 6]:
        r, U = hermite_eigenbasis(n)
        for s1, s2 in [(1.0, 1.0), (1.0, 2.0)]:
            s2_tot = s1 ** 2 + s2 ** 2
            t = (s1 ** 2) / s2_tot

            def delta_n(alpha: np.ndarray, beta: np.ndarray) -> float:
                gamma, _ = boxplus_roots(alpha, beta)
                return (
                    1.0 / np.sum(score_vector(gamma) ** 2)
                    - 1.0 / np.sum(score_vector(alpha) ** 2)
                    - 1.0 / np.sum(score_vector(beta) ** 2)
                )

            H_num = np.zeros((2 * n, 2 * n))
            for i in range(2 * n):
                vi = U[:, i % n]
                for j in range(i, 2 * n):
                    vj = U[:, j % n]

                    def ev(a_step: float, b_step: float) -> float:
                        a = s1 * r + (a_step * vi if i < n else 0.0) + (b_step * vj if j < n else 0.0)
                        b = s2 * r + (a_step * vi if i >= n else 0.0) + (b_step * vj if j >= n else 0.0)
                        return delta_n(a, b)

                    val = (ev(eps, eps) - ev(eps, -eps) - ev(-eps, eps) + ev(-eps, -eps)) / (4.0 * eps ** 2)
                    H_num[i, j] = H_num[j, i] = val

            H_exact = np.zeros((2 * n, 2 * n))
            # Hessian matrix H = nabla^2 Delta_n carries factor 2 relative to the quadratic
            # Taylor form Delta_n = (1/2) z^T H z displayed in Theorem 4.1 (4/[n(n-1)]^2 -> 8/[n(n-1)]^2).
            pref = 8.0 / ((n * (n - 1)) ** 2)
            for k in range(2, n):
                c_k = pref * ((k + 1) ** 2)
                H_exact[k, k] = c_k * (1.0 - t ** k)
                H_exact[n + k, n + k] = c_k * (1.0 - (1.0 - t) ** k)
                H_exact[k, n + k] = H_exact[n + k, k] = -c_k * ((t * (1.0 - t)) ** (k / 2.0))

            max_err = np.max(np.abs((H_num - H_exact) * ((n * (n - 1)) ** 2)))
            assert max_err < 1e-3, f"Hessian mismatch at n={n}, t={t}: {max_err}"
            print(f"  n={n}, t={t:.2f}: scaled Hessian max error = {max_err:.3e}")

    # Verify Corollary 4.2 sharpness at k=2 along (u_2, v_2) proportional to (t, 1-t)
    r5, U5 = hermite_eigenbasis(5)
    for s1, s2 in [(1.0, 2.0), (1.5, 1.0)]:
        t = (s1 ** 2) / (s1 ** 2 + s2 ** 2)
        u = 1e-4 * t * U5[:, 2]
        v = 1e-4 * (1.0 - t) * U5[:, 2]
        g, _ = boxplus_roots(s1 * r5 + u, s2 * r5 + v)
        delta = (
            1.0 / np.sum(score_vector(g) ** 2)
            - 1.0 / np.sum(score_vector(s1 * r5 + u) ** 2)
            - 1.0 / np.sum(score_vector(s2 * r5 + v) ** 2)
        )
        cr_a = 4.0 * np.sum((s1 * r5 + u) ** 2) / ((5 * 4) ** 2) - 1.0 / np.sum(score_vector(s1 * r5 + u) ** 2)
        cr_b = 4.0 * np.sum((s2 * r5 + v) ** 2) / ((5 * 4) ** 2) - 1.0 / np.sum(score_vector(s2 * r5 + v) ** 2)
        obs_ratio = delta / (cr_a + cr_b)
        pred_ratio = 2.0 * t * (1.0 - t)
        assert abs(obs_ratio - pred_ratio) < 1e-5
        print(f"  Sharpness n=5, t={t:.4f}: ratio={obs_ratio:.6f}, 2t(1-t)={pred_ratio:.6f}")

    # Local R_n < 1/2 non-attainment verification for n >= 4 (Theorem 4.4):
    # (i) Exact Appell-cumulant family D_{p_eps}(z) = exp(-z^2/2 + eps*z^3/6 - (5/32)*eps^2*z^4) mod z^{n+1}
    #     gives R_n(p_eps, p_eps) = 1/2 + ((19 - 5n)/64)*eps^2 + O(eps^4) < 1/2 for n = 4..8.
    mp.mp.dps = 50
    eps_mp = mp.mpf("1e-4")
    for n_loc in [4, 5, 6, 7, 8]:
        f_ser = [mp.mpf(0)] * (n_loc + 1)
        f_ser[2] = -mp.mpf(1) / 2
        f_ser[3] = eps_mp / 6
        f_ser[4] = -mp.mpf(5) * (eps_mp ** 2) / 32
        expf = [mp.mpf(0)] * (n_loc + 1)
        expf[0] = mp.mpf(1)
        cur = [mp.mpf(0)] * (n_loc + 1)
        cur[0] = mp.mpf(1)
        for m_pow in range(1, n_loc + 1):
            nxt = [mp.mpf(0)] * (n_loc + 1)
            for idx_i in range(n_loc + 1):
                if cur[idx_i] == 0:
                    continue
                for idx_j in range(2, min(5, n_loc + 1 - idx_i)):
                    nxt[idx_i + idx_j] += cur[idx_i] * f_ser[idx_j]
            cur = [v / m_pow for v in nxt]
            for idx_i in range(n_loc + 1):
                expf[idx_i] += cur[idx_i]
        fact_n = mp.factorial(n_loc)
        p_c = [expf[k] * fact_n / mp.factorial(n_loc - k) for k in range(n_loc + 1)]
        rts_p = mp_real_roots_checked(p_c, maxsteps=200)
        phi_p = sum(
            (sum(mp.mpf(1) / (rts_p[ii] - rts_p[jj]) for jj in range(n_loc) if jj != ii)) ** 2
            for ii in range(n_loc)
        )
        # Rescaled self-convolution replaces eps by eps/sqrt(2)
        eps_half = eps_mp / mp.sqrt(2)
        f_ser2 = [mp.mpf(0)] * (n_loc + 1)
        f_ser2[2] = -mp.mpf(1) / 2
        f_ser2[3] = eps_half / 6
        f_ser2[4] = -mp.mpf(5) * (eps_half ** 2) / 32
        expf2 = [mp.mpf(0)] * (n_loc + 1)
        expf2[0] = mp.mpf(1)
        cur2 = [mp.mpf(0)] * (n_loc + 1)
        cur2[0] = mp.mpf(1)
        for m_pow in range(1, n_loc + 1):
            nxt = [mp.mpf(0)] * (n_loc + 1)
            for idx_i in range(n_loc + 1):
                if cur2[idx_i] == 0:
                    continue
                for idx_j in range(2, min(5, n_loc + 1 - idx_i)):
                    nxt[idx_i + idx_j] += cur2[idx_i] * f_ser2[idx_j]
            cur2 = [v / m_pow for v in nxt]
            for idx_i in range(n_loc + 1):
                expf2[idx_i] += cur2[idx_i]
        p_c2 = [expf2[k] * fact_n / mp.factorial(n_loc - k) for k in range(n_loc + 1)]
        rts_p2 = mp_real_roots_checked(p_c2, maxsteps=200)
        phi_p2 = sum(
            (sum(mp.mpf(1) / (rts_p2[ii] - rts_p2[jj]) for jj in range(n_loc) if jj != ii)) ** 2
            for ii in range(n_loc)
        )
        phi_he = mp.mpf(n_loc) * (n_loc - 1) / 4
        g1 = 1 - phi_he / phi_p
        g2 = 1 - phi_he / phi_p2
        R_loc = (g1 - g2) / (g1 * (1 - g2))
        dip_obs = (R_loc - mp.mpf("0.5")) / (eps_mp ** 2)
        dip_pred = mp.mpf(19 - 5 * n_loc) / 64
        assert abs(dip_obs - dip_pred) < mp.mpf("1e-5") and R_loc < mp.mpf("0.5")
        print(f"  Local R_{n_loc} < 1/2 (Appell family, eps=1e-4): (R_{n_loc}-1/2)/eps^2 = {mp.nstr(dip_obs, 9)} (theory {mp.nstr(dip_pred, 9)})")

    # (ii) Non-circular exact rational n=4 verification via companion matrix Tr((M_{p''}(2 M_{p'})^{-1})^2) over Q
    def _mat_mul_q(A_q: list[list[Fraction]], B_q: list[list[Fraction]]) -> list[list[Fraction]]:
        m_q = len(A_q)
        return [[sum((A_q[ii][kk] * B_q[kk][jj] for kk in range(m_q)), Fraction(0)) for jj in range(m_q)] for ii in range(m_q)]

    def phi4_companion_exact(q_c: Fraction, r_c: Fraction, s_c: Fraction) -> Fraction:
        z0 = Fraction(0)
        Mx = [
            [z0, z0, z0, -s_c],
            [Fraction(1), z0, z0, -r_c],
            [z0, Fraction(1), z0, -q_c],
            [z0, z0, Fraction(1), z0],
        ]
        Mx2 = _mat_mul_q(Mx, Mx)
        Mx3 = _mat_mul_q(Mx2, Mx)
        I4 = [[Fraction(1) if ii == jj else z0 for jj in range(4)] for ii in range(4)]
        A_m = [[8 * Mx3[ii][jj] + 4 * q_c * Mx[ii][jj] + 2 * r_c * I4[ii][jj] for jj in range(4)] for ii in range(4)]
        B_m = [[12 * Mx2[ii][jj] + 2 * q_c * I4[ii][jj] for jj in range(4)] for ii in range(4)]
        aug = [row[:] + [Fraction(1) if ii == jj else z0 for jj in range(4)] for ii, row in enumerate(A_m)]
        for col in range(4):
            piv_row = next(rw for rw in range(col, 4) if aug[rw][col] != 0)
            aug[col], aug[piv_row] = aug[piv_row], aug[col]
            piv = aug[col][col]
            aug[col] = [val / piv for val in aug[col]]
            for rw in range(4):
                if rw != col:
                    fac = aug[rw][col]
                    aug[rw] = [aug[rw][jj] - fac * aug[col][jj] for jj in range(8)]
        Ainv = [row[4:] for row in aug]
        U_m = _mat_mul_q(B_m, Ainv)
        U2 = _mat_mul_q(U_m, U_m)
        return sum((U2[ii][ii] for ii in range(4)), Fraction(0))

    # Certify the general depressed-quartic Fisher information formula (Remark 4.9) over Q
    for c2_q in [Fraction(-6), Fraction(-4), Fraction(-5, 2)]:
        for c1_q in [Fraction(0), Fraction(1, 2), Fraction(-3, 5)]:
            for c0_q in [Fraction(1), Fraction(3), Fraction(7, 4)]:
                disc_q = (
                    256 * c0_q ** 3
                    - 128 * (c2_q ** 2) * (c0_q ** 2)
                    + 144 * c2_q * (c1_q ** 2) * c0_q
                    - 27 * c1_q ** 4
                    + 16 * (c2_q ** 4) * c0_q
                    - 4 * (c2_q ** 3) * (c1_q ** 2)
                )
                if disc_q == 0:
                    continue
                num_gen_q = (
                    384 * c2_q * (c0_q ** 2)
                    - 64 * (c2_q ** 3) * c0_q
                    - 8 * c2_q ** 5
                    - 432 * (c1_q ** 2) * c0_q
                    - 36 * (c2_q ** 2) * (c1_q ** 2)
                )
                assert phi4_companion_exact(c2_q, c1_q, c0_q) == num_gen_q / disc_q

    # Certify the degree-(2,3) rational identity on a 4x4 rational (eps, tau) grid over Q
    for e_grid in [Fraction(0), Fraction(1, 10), Fraction(1, 4), Fraction(2, 5)]:
        for t_grid in [Fraction(0), Fraction(-1, 5), Fraction(1, 3), Fraction(-3, 7)]:
            phi_comp = phi4_companion_exact(Fraction(-6), 4 * e_grid, 3 + t_grid)
            num_id = Fraction(3) - Fraction(3, 2) * e_grid ** 2 - Fraction(1, 12) * t_grid ** 2 - Fraction(1, 4) * e_grid ** 2 * t_grid
            den_id = Fraction(1) - e_grid ** 2 - Fraction(1, 12) * t_grid ** 2 - Fraction(1, 2) * e_grid ** 2 * t_grid - Fraction(1, 4) * e_grid ** 4 + Fraction(1, 108) * t_grid ** 3
            assert phi_comp == num_id / den_id, f"Quartic rational identity mismatch at ({e_grid}, {t_grid})"

    # Direct exact rational check of p_{1/10} and p_{1/10} boxplus_4 p_{1/10} (plus real-rootedness)
    eps_q = Fraction(1, 10)
    tau_q = Fraction(-15, 4) * (eps_q ** 2)
    rts_q = real_roots_checked(np.array([1.0, 0.0, -6.0, float(4 * eps_q), float(3 + tau_q)]))
    assert np.min(np.diff(rts_q)) > 1.0, f"p_{{1/10}} roots not well-separated: {rts_q}"
    phi_p_q = phi4_companion_exact(Fraction(-6), 4 * eps_q, 3 + tau_q)
    # p boxplus_4 p has coefficients q2 = -12, r2 = 8*eps_q, s2 = 2*(3+tau_q) + (2!2!/4!)*(-6)^2 = 12 + 2*tau_q
    phi_pp_q = phi4_companion_exact(Fraction(-12), 8 * eps_q, 12 + 2 * tau_q)
    inv_p_q, inv_pp_q = Fraction(1) / phi_p_q, Fraction(1) / phi_pp_q
    h1_q = Fraction(1, 3)
    da_q = h1_q - inv_p_q
    Delta_q = inv_pp_q - 2 * inv_p_q
    R4_exact_q = (Delta_q / inv_pp_q) / (da_q / h1_q)
    assert R4_exact_q < Fraction(1, 2)
    print(f"  Exact Q companion-matrix quartic check (4x4 grid + eps=1/10): R_4 = {float(R4_exact_q):.12f} < 0.5")

    # Global obstruction check for unnormalized CR gap at n=4, 5, 6 and colliding R_n check
    for n in [4, 5, 6]:
        ratios = []
        rn_vals = []
        for eps in [1e-1, 1e-2, 1e-3]:
            a = np.array([-1.0 + j * eps for j in range(n - 1)] + [float(n - 1) - 0.5 * (n - 1) * (n - 2) * eps])
            a -= a.mean()
            g, _ = boxplus_roots(a, a)
            pa = 1.0 / np.sum(score_vector(a) ** 2)
            pg = 1.0 / np.sum(score_vector(g) ** 2)
            delta_ab = pg - 2.0 * pa
            h1 = 4.0 / (n * (n - 1))
            ha = (np.sum(a ** 2) / (n * (n - 1))) * h1
            cr_a = ha - pa
            ratios.append(delta_ab / (2.0 * cr_a))
            rn_vals.append((delta_ab / pg) / ((2.0 * cr_a) / (2.0 * ha)))
        assert ratios[-1] < 1e-5, f"Obstruction decay failed at n={n}: {ratios}"
        if n in (4, 5):
            assert min(rn_vals) >= 0.5 - 1e-9, f"Colliding R_n < 1/2 at n={n}: {rn_vals}"
        print(f"  Global CR obstruction n={n}: Delta_n/(2*delta_n) = {[f'{r:.2e}' for r in ratios]}, colliding R_n = {[f'{v:.4f}' for v in rn_vals]}")


def verify_clt_rates() -> None:
    print("\n=== 4. Verification of Exact Finite Free CLT Asymptotics ===")
    rng = np.random.default_rng(1)
    # 4a. Asymmetric polynomial (O(1/m) rate)
    for n in [3, 4, 5]:
        r0, _ = hermegauss(n)
        alpha = rng.normal(size=n)
        alpha -= alpha.mean()
        alpha *= np.sqrt(np.sum(r0 ** 2) / np.sum(alpha ** 2))
        a_base = roots_to_elementary(alpha)
        phi_H = np.sum(score_vector(r0) ** 2)
        pred_c1 = (np.sum(alpha ** 3) ** 2) / ((n ** 2) * ((n - 1) ** 2) * (n - 2))

        m = 512
        cur = np.zeros(n + 1)
        cur[0] = 1.0
        base = a_base.copy()
        mm = m
        while mm > 0:
            if mm & 1:
                cur = boxplus_coeffs(cur, base)
            mm >>= 1
            if mm > 0:
                base = boxplus_coeffs(base, base)
        poly = np.array([((-1) ** k) * cur[k] for k in range(n + 1)])
        roots_m = real_roots_checked(poly) / np.sqrt(m)
        phi_m = np.sum(score_vector(roots_m) ** 2)
        obs_c1 = m * (phi_m / phi_H - 1.0)
        rel_err = abs(obs_c1 - pred_c1) / pred_c1
        assert rel_err < 0.02, f"CLT O(1/m) mismatch at n={n}: obs={obs_c1}, pred={pred_c1}"
        print(f"  Asymmetric n={n} (m={m}): obs={obs_c1:.6f}, pred={pred_c1:.6f} (rel err={rel_err:.2e})")

    # 4b. Symmetric polynomial (O(1/m^2) rate)
    for n in [4, 5]:
        r0, _ = hermegauss(n)
        alpha = np.array([-2.0, -1.0, 1.0, 2.0]) if n == 4 else np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        alpha = alpha * np.sqrt(np.sum(r0 ** 2) / np.sum(alpha ** 2))
        a_base = roots_to_elementary(alpha)
        he_coeffs = roots_to_elementary(r0)
        phi_H = np.sum(score_vector(r0) ** 2)
        delta_a4 = a_base[4] - he_coeffs[4]
        pred_c2 = 16.0 * (delta_a4 ** 2) / (n * (n - 1) * (factorial(n) // factorial(n - 4)))

        m = 256
        cur = np.zeros(n + 1)
        cur[0] = 1.0
        base = a_base.copy()
        mm = m
        while mm > 0:
            if mm & 1:
                cur = boxplus_coeffs(cur, base)
            mm >>= 1
            if mm > 0:
                base = boxplus_coeffs(base, base)
        poly = np.array([((-1) ** k) * cur[k] for k in range(n + 1)])
        roots_m = real_roots_checked(poly) / np.sqrt(m)
        phi_m = np.sum(score_vector(roots_m) ** 2)
        obs_c2 = (m ** 2) * (phi_m / phi_H - 1.0)
        rel_err = abs(obs_c2 - pred_c2) / pred_c2
        assert rel_err < 0.01, f"CLT O(1/m^2) mismatch at n={n}: obs={obs_c2}, pred={pred_c2}"
        print(f"  Symmetric  n={n} (m={m}): obs={obs_c2:.6f}, pred={pred_c2:.6f} (rel err={rel_err:.2e})")

    # 4c. Higher cumulant matching (j=5, O(1/m^3) rate) for n=5
    r5, _ = hermegauss(5)
    he5 = roots_to_elementary(r5)
    a5_pert = he5.copy()
    a5_pert[5] = 0.2
    gamma_5 = -a5_pert[5]
    pred_c3 = factorial(5) / ((factorial(4) ** 2) * 5 * 4) * (gamma_5 ** 2)
    phi_H5 = np.sum(score_vector(r5) ** 2)
    m = 64
    cur = np.zeros(6)
    cur[0] = 1.0
    base = a5_pert.copy()
    mm = m
    while mm > 0:
        if mm & 1:
            cur = boxplus_coeffs(cur, base)
        mm >>= 1
        if mm > 0:
            base = boxplus_coeffs(base, base)
    poly = np.array([((-1) ** k) * cur[k] for k in range(6)])
    roots_m = real_roots_checked(poly) / np.sqrt(m)
    phi_m = np.sum(score_vector(roots_m) ** 2)
    obs_c3 = (m ** 3) * (phi_m / phi_H5 - 1.0)
    rel_err5 = abs(obs_c3 - pred_c3) / pred_c3
    assert rel_err5 < 1e-4, f"CLT O(1/m^3) mismatch at n=5: obs={obs_c3}, pred={pred_c3}"
    print(f"  Cumulant j=5, n=5 (m={m}): obs={obs_c3:.6f}, pred={pred_c3:.6f} (rel err={rel_err5:.2e})")


def verify_score_identity_and_p_stam() -> None:
    print("\n=== 5. Verification of Jacobian Score Identity and Hermite p-Stam Phase Transition ===")
    rng = np.random.default_rng(2026)
    for n in [3, 4, 5, 6]:
        alpha = np.sort(rng.normal(size=n))
        beta = np.sort(rng.normal(size=n))
        Ja, gamma = jacobian_alpha(alpha, beta)
        Jb, _ = jacobian_alpha(beta, alpha)
        sa = score_vector(alpha)
        sb = score_vector(beta)
        sg = score_vector(gamma)
        err_a = np.linalg.norm(Ja @ sa - sg)
        err_b = np.linalg.norm(Jb @ sb - sg)
        assert err_a < 1e-10 and err_b < 1e-10
        P = np.eye(n) - np.ones((n, n)) / n
        max_eig = np.max(np.linalg.eigvalsh(P @ (Ja @ Ja.T + Jb @ Jb.T) @ P))
        assert max_eig <= 1.0 + 1e-11
        print(f"  n={n}: ||Ja s(a) - s(g)||={err_a:.2e}, ||Jb s(b) - s(g)||={err_b:.2e}, max eig(Ja Ja^T + Jb Jb^T on 1^perp)={max_eig:.6f}")

    # Hermite p-Stam phase transition check
    r0, _ = hermegauss(4)
    sr = score_vector(r0)
    gamma, _ = boxplus_roots(r0, r0)
    sg = score_vector(gamma)
    for p in [1.25, 1.5, 2.0, 2.5, 3.0]:
        q = p / (p - 1.0)
        lhs = np.linalg.norm(sg, ord=p) ** (-q)
        rhs = 2.0 * (np.linalg.norm(sr, ord=p) ** (-q))
        pred_ratio = 2.0 ** (q / 2.0 - 1.0)
        obs_ratio = lhs / rhs
        assert abs(obs_ratio - pred_ratio) < 1e-12
        status = "SATISFIED (>0)" if obs_ratio > 1.0 + 1e-9 else ("EQUALITY (=0)" if abs(obs_ratio - 1.0) < 1e-9 else "VIOLATED (<0)")
        print(f"  p={p:.2f} (q={q:.2f}): Hermite ratio = {obs_ratio:.6f} (2^(q/2-1)={pred_ratio:.6f}) -> {status}")

    # Numerical check of l^p(1^perp) contraction and n=3 Theorem + n=4 obstruction
    # 5a. Theorem (n=3 Global p-Stam via l^p(1^perp) Riesz-Thorin)
    def max_alpha_l1(A):
        n_loc = A.shape[0]
        if np.max(np.sum(np.abs(A - 1e-11 / n_loc), axis=0)) > 1.0 + 1e-13:
            return 0.0
        lo, hi = 0.0, 1.0
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            if np.max(np.sum(np.abs(A - mid / n_loc), axis=0)) <= 1.0 + 1e-12:
                lo = mid
            else:
                hi = mid
        return lo

    max_shift_n3 = 0.0
    for _ in range(200):
        a = np.sort(rng.normal(size=3) * np.exp(rng.normal()))
        b = np.sort(rng.normal(size=3) * np.exp(rng.normal()))
        Ja, _ = jacobian_alpha(a, b)
        Jb, _ = jacobian_alpha(b, a)
        amax = min(1.0, 6.0 * (1.0 - np.max(Ja)))
        bmax = min(1.0, 6.0 * (1.0 - np.max(Jb)))
        assert abs(amax - max_alpha_l1(Ja)) < 1e-9
        max_shift_n3 = max(max_shift_n3, (1.0 - amax) ** 2 + (1.0 - bmax) ** 2)
    assert max_shift_n3 <= 1.0 + 1e-12
    print(f"  n=3 Riesz-Thorin shift bound max (1-amax)^2 + (1-bmax)^2 = {max_shift_n3:.8f} <= 1.0")

    for p in [1.05, 1.25, 1.50, 1.75, 1.95]:
        q_exp = p / (p - 1.0)
        min_pstam_n3 = 1e9
        for _ in range(250):
            a = np.sort(rng.normal(size=3) * np.exp(rng.normal()))
            b = np.sort(rng.normal(size=3) * np.exp(rng.normal()))
            g, _ = boxplus_roots(a, b)
            sa_p = np.linalg.norm(score_vector(a), ord=p)
            sb_p = np.linalg.norm(score_vector(b), ord=p)
            sg_p = np.linalg.norm(score_vector(g), ord=p)
            min_pstam_n3 = min(min_pstam_n3, (sg_p ** (-q_exp)) / (sa_p ** (-q_exp) + sb_p ** (-q_exp)))
        assert min_pstam_n3 >= 1.0 - 1e-11
        print(f"  n=3 Conjecture 5.2 (p={p:.2f}): min p-Stam ratio = {min_pstam_n3:.8f} >= 1.0")

    # 5b. Global n=3 Stam stability (Theorem 4.6), Dyson-flow O(1/n) relative decay (Proposition 4.7), and Remark 4.9
    max_disc_err = 0.0
    min_cubic_stab = 1e9
    min_R3 = 1e9
    for _ in range(300):
        a = np.sort(rng.normal(size=3) * np.exp(rng.normal()))
        b = np.sort(rng.normal(size=3) * np.exp(rng.normal()))
        a -= a.mean(); b -= b.mean()
        pa_c = np.poly(a); pb_c = np.poly(b)
        a1, b1 = -pa_c[2], -pa_c[3]
        a2, b2 = -pb_c[2], -pb_c[3]
        phi_a_inv = 1.0 / np.sum(score_vector(a) ** 2)
        phi_b_inv = 1.0 / np.sum(score_vector(b) ** 2)
        max_disc_err = max(max_disc_err, abs(phi_a_inv - ((2.0 / 9.0) * a1 - 1.5 * (b1 ** 2) / (a1 ** 2))))
        g, _ = boxplus_roots(a, b)
        phi_g_inv = 1.0 / np.sum(score_vector(g) ** 2)
        t_c = a1 / (a1 + a2)
        delta_3 = phi_g_inv - (phi_a_inv + phi_b_inv)
        cr_sum = 1.5 * ((b1 / a1) ** 2 + (b2 / a2) ** 2)
        if cr_sum > 1e-13:
            min_cubic_stab = min(min_cubic_stab, delta_3 / (2.0 * t_c * (1.0 - t_c) * cr_sum))
            ha_hb = (2.0 / 9.0) * (a1 + a2)
            R3_val = (delta_3 / phi_g_inv) / (4.0 * t_c * (1.0 - t_c) * (cr_sum / ha_hb))
            min_R3 = min(min_R3, R3_val)
    # Check sharpness of R_3 >= 1/2 near Hermite diagonal
    r3_h, U3_h = hermite_eigenbasis(3)
    u3 = 1e-4 * 0.5 * U3_h[:, 2]
    g3_h, _ = boxplus_roots(r3_h + u3, r3_h + u3)
    pa3 = 1.0 / np.sum(score_vector(r3_h + u3) ** 2)
    pg3 = 1.0 / np.sum(score_vector(g3_h) ** 2)
    ha3 = (np.sum((r3_h + u3) ** 2) / 6.0) * (4.0 / 6.0)
    R3_herm = ((pg3 - 2.0 * pa3) / pg3) / (4.0 * 0.5 * 0.5 * (2.0 * (ha3 - pa3)) / (2.0 * ha3))
    assert max_disc_err < 1e-12 and min_cubic_stab >= 1.0 - 1e-10 and min_R3 >= 0.5 - 1e-10 and abs(R3_herm - 0.5) < 1e-5
    print(f"  Global n=3 Stam theorem: max formula err={max_disc_err:.2e}, min ratio={min_cubic_stab:.10f} >= 1.0, min R_3={min_R3:.6f} >= 0.5 (Hermite limit={R3_herm:.6f})")

    # Check Dyson slope and 1-outlier + He_{n-1} O(1/n) asymptotic decay
    for n_asy in [4, 12, 40]:
        r_sub, _ = hermite_eigenbasis(n_asy - 1)
        R_out = (n_asy - 1) / np.sqrt(2.0)
        w_bulk = np.sqrt(n_asy / (2.0 * (n_asy - 2)))
        q_out = np.concatenate([[-R_out], R_out / (n_asy - 1) + w_bulk * r_sub])
        s_out = score_vector(q_out)
        L_out = log_gas_laplacian(q_out)
        pdot = float((s_out @ L_out @ s_out) / (np.sum(s_out ** 2) ** 2))
        h1_n = 4.0 / (n_asy * (n_asy - 1))
        pq = 1.0 / float(np.sum(s_out ** 2))
        r_lim = (pdot - h1_n) / (4.0 * pq * (1.0 - pq / h1_n))
        # Verify finite-diff Dyson slope match at n=4
        if n_asy == 4:
            r4, _ = hermite_eigenbasis(4)
            r4_unit = r4 / np.sqrt(np.sum(r4 ** 2) / 12.0)
            g_eps, _ = boxplus_roots(q_out, np.sqrt(1e-6) * r4_unit)
            pdot_fd = (1.0 / np.sum(score_vector(g_eps) ** 2) - pq) / 1e-6
            assert abs(pdot_fd - pdot) / pdot < 1e-4
        print(f"  Dyson relative slope n={n_asy:2d}: pq/h1={pq/h1_n:.4f}, R_n(t->0+)={r_lim:.5f}, n*R_n={n_asy*r_lim:.4f}")

    # Remark 4.9: Exact rational integer quartic witness q(x) = x^4 - 6x^2 + 2x + 2 => c_4 <= 876/1855
    # and multi-start BFGS Dyson-limit minimization for n = 4, 5, 6, 8, 12
    c2_w, c1_w, c0_w = Fraction(-6), Fraction(2), Fraction(2)
    q_signs = [x ** 4 - 6 * x ** 2 + 2 * x + 2 for x in (-3, -1, 0, 1, 3)]
    assert q_signs == [23, -5, 2, -1, 35]  # 4 simple real roots by IVT
    # Dual-number heat derivative (c2_dot = -6, c1_dot = 0, c0_dot = -c2 = +6)
    def _q_disc_and_num(c2_f: Fraction, c1_f: Fraction, c0_f: Fraction) -> tuple[Fraction, Fraction]:
        disc_f = (
            256 * c0_f ** 3 - 128 * (c2_f ** 2) * (c0_f ** 2) + 144 * c2_f * (c1_f ** 2) * c0_f
            - 27 * c1_f ** 4 + 16 * (c2_f ** 4) * c0_f - 4 * (c2_f ** 3) * (c1_f ** 2)
        )
        num_f = (
            384 * c2_f * (c0_f ** 2) - 64 * (c2_f ** 3) * c0_f - 8 * c2_f ** 5
            - 432 * (c1_f ** 2) * c0_f - 36 * (c2_f ** 2) * (c1_f ** 2)
        )
        return disc_f, num_f
    disc_w, num_w = _q_disc_and_num(c2_w, c1_w, c0_w)
    ddisc_dc2 = -256 * c2_w * (c0_w ** 2) + 144 * (c1_w ** 2) * c0_w + 64 * (c2_w ** 3) * c0_w - 12 * (c2_w ** 2) * (c1_w ** 2)
    ddisc_dc0 = 768 * (c0_w ** 2) - 256 * (c2_w ** 2) * c0_w + 144 * c2_w * (c1_w ** 2) + 16 * (c2_w ** 4)
    dnum_dc2 = 384 * (c0_w ** 2) - 192 * (c2_w ** 2) * c0_w - 40 * (c2_w ** 4) - 72 * c2_w * (c1_w ** 2)
    dnum_dc0 = 768 * c2_w * c0_w - 64 * (c2_w ** 3) - 432 * (c1_w ** 2)
    disc_dot_w = -6 * ddisc_dc2 + 6 * ddisc_dc0
    num_dot_w = -6 * dnum_dc2 + 6 * dnum_dc0
    inv_phi_w = disc_w / num_w
    pdot_w = (disc_dot_w * num_w - disc_w * num_dot_w) / (num_w ** 2)
    h1_w = Fraction(1, 3)
    R4_int_w = (pdot_w - h1_w) / (4 * inv_phi_w * (1 - inv_phi_w / h1_w))
    assert inv_phi_w == Fraction(53, 180) and pdot_w == Fraction(448, 1125) and R4_int_w == Fraction(876, 1855)
    print(f"  Exact Q integer quartic Dyson witness (Remark 4.9): c_4 <= {R4_int_w} = {float(R4_int_w):.6f} < 0.5")

    rng_dys = np.random.default_rng(123)
    dys_summary = []
    for n_d in [4, 5, 6, 8, 12]:
        def dyson_obj(roots_v: np.ndarray) -> float:
            x = np.sort(roots_v)
            x = x - x.mean()
            n2 = float(np.sum(x ** 2))
            if n2 <= 0.0:
                return 10.0
            x = x * np.sqrt(n_d * (n_d - 1) / n2)
            if np.min(np.diff(x)) < 1e-7:
                return 10.0
            s_v = score_vector(x)
            phi_v = float(np.sum(s_v ** 2))
            h1_d = 4.0 / (n_d * (n_d - 1))
            ip_v = 1.0 / phi_v
            if ip_v >= h1_d * (1.0 - 1e-12):
                return 10.0
            diff_m = x[:, None] - x[None, :]
            sdiff_m = s_v[:, None] - s_v[None, :]
            mask_m = ~np.eye(n_d, dtype=bool)
            pdot_v = float(np.sum((sdiff_m[mask_m] / diff_m[mask_m]) ** 2)) / (phi_v ** 2)
            return (pdot_v - h1_d) / (4.0 * ip_v * (1.0 - ip_v / h1_d))

        r_prev_d, _ = hermegauss(n_d - 1)
        starts_d = [
            np.concatenate([[-(n_d - 1) / np.sqrt(2.0) * sc], 1.0 / np.sqrt(2.0) + np.sqrt(n_d / (2.0 * (n_d - 2))) * r_prev_d])
            for sc in (0.5, 0.8, 1.0, 1.3)
        ] + [np.sort(rng_dys.normal(size=n_d)) for _ in range(6)]
        best_d = 10.0
        for x0_d in starts_d:
            res_d = minimize(dyson_obj, x0_d, method="BFGS", options={"gtol": 1e-8, "maxiter": 300})
            if res_d.fun < best_d:
                best_d = float(res_d.fun)
        dys_summary.append((n_d, best_d, n_d * best_d))
    print("  Dyson-limit BFGS minima (Remark 4.9): " + ", ".join(f"n={nd}: {bd:.4f} (n*R={nrd:.2f})" for nd, bd, nrd in dys_summary))

    # Numerical check of l^p(1^perp) contraction for n=3, 4 and p in (1, 2)
    # Test 20 multi-scale random Gaussian pairs + 10 adversarial clustered/outlier pairs per degree
    def make_test_pairs(n_loc: int, rng_loc: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
        pairs = []
        for _ in range(20):
            s_a = float(np.exp(0.8 * rng_loc.normal()))
            s_b = float(np.exp(0.8 * rng_loc.normal()))
            a_v = np.sort(rng_loc.normal(size=n_loc) * s_a)
            b_v = np.sort(rng_loc.normal(size=n_loc) * s_b)
            pairs.append((a_v, b_v))
        for k_adv in range(10):
            if k_adv == 0 and n_loc == 4:
                pairs.append((
                    np.array([-3.89125, -3.88625, -3.88125, 11.65875]),
                    np.array([-11.635, 3.765, 3.885, 3.985]),
                ))
            else:
                # Clustered (n-1) roots plus 1 outlier, with random scales and orientations
                eps_c = 10.0 ** rng_loc.uniform(-2.5, -0.7)
                a_v = np.concatenate([[-1.0 + j * eps_c for j in range(n_loc - 1)], [float(n_loc - 1)]])
                b_v = np.concatenate([[-1.0 + j * eps_c for j in range(n_loc - 1)], [float(n_loc - 1)]])
                if k_adv % 2 == 1:
                    b_v = -b_v[::-1]
                a_v = np.sort((a_v - a_v.mean()) * float(np.exp(0.5 * rng_loc.normal())))
                b_v = np.sort((b_v - b_v.mean()) * float(np.exp(0.5 * rng_loc.normal())))
                pairs.append((a_v, b_v))
        return pairs

    for n in [3, 4]:
        max_ratio = 0.0
        test_pairs = make_test_pairs(n, rng)
        for alpha, beta in test_pairs:
            Ja, _ = jacobian_alpha(alpha, beta)
            Jb, _ = jacobian_alpha(beta, alpha)
            for p in [1.25, 1.5, 1.75]:
                def neg_ratio(z):
                    x = z[:n] - np.mean(z[:n])
                    y = z[n:] - np.mean(z[n:])
                    denom = (np.sum(np.abs(x) ** p) + np.sum(np.abs(y) ** p)) ** (1.0 / p)
                    if denom < 1e-12:
                        return 0.0
                    num = np.sum(np.abs(Ja @ x + Jb @ y) ** p) ** (1.0 / p)
                    return -num / denom
                for _ in range(6):
                    res = minimize(neg_ratio, rng.normal(size=2 * n), method="Powell", options={"maxiter": 2000})
                    max_ratio = max(max_ratio, float(-res.fun))
        assert max_ratio <= 1.0 + 1e-8
        print(f"  l^p(1^perp) contraction n={n} (30 random+adversarial pairs, Powell), p in {{1.25,1.5,1.75}}: max ratio = {max_ratio:.6f} <= 1")

    # Exact uniform vs non-uniform rank-1 shift check at n=4 and quotient l^q dual bound (Remark 6.4)
    a_obs = np.array([-3.89125, -3.88625, -3.88125, 11.65875])
    b_obs = np.array([-11.635, 3.765, 3.885, 3.985])
    Ja_obs, _ = jacobian_alpha(a_obs, b_obs)
    Jb_obs, _ = jacobian_alpha(b_obs, a_obs)
    MA_obs, MB_obs = float(np.max(Ja_obs)), float(np.max(Jb_obs))
    lb_uni = float(np.sqrt(2.0) * (4.0 * min(MA_obs, MB_obs) - 1.0) / (1.0 + 2.0 * np.sqrt(2.0)))

    def uniform_shift_obj(ab):
        al, be = float(ab[0]), float(ab[1])
        At = Ja_obs - al / 4.0
        Bt = Jb_obs - be / 4.0
        return max(
            float(np.max(np.sum(np.abs(At), axis=0))),
            float(np.max(np.sum(np.abs(Bt), axis=0))),
            float(np.linalg.norm(np.hstack([At, Bt]), ord=2)),
        )

    res_uni = minimize(uniform_shift_obj, [0.28, 0.28], method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-12})
    uni_shift_min = float(res_uni.fun)
    assert lb_uni > 1.008 and uni_shift_min >= lb_uni - 1e-12

    def smooth_shift_obj(uv, beta_sm):
        u, v = uv[:4], uv[4:]
        At = Ja_obs - u[:, None]
        Bt = Jb_obs - v[:, None]
        eps_s = 1e-13
        l1a = np.sum(np.sqrt(At ** 2 + eps_s), axis=0)
        l1b = np.sum(np.sqrt(Bt ** 2 + eps_s), axis=0)
        s2 = np.linalg.eigvalsh(np.hstack([At, Bt]) @ np.hstack([At, Bt]).T)
        vals = np.concatenate([l1a, l1b, [np.sqrt(max(0.0, s2[-1]) + eps_s)]])
        m = np.max(vals)
        return m + np.log(np.sum(np.exp(beta_sm * (vals - m)))) / beta_sm

    uv_opt = np.full(8, 0.25)
    for beta_sm in [30.0, 300.0, 5000.0, 50000.0]:
        uv_opt = minimize(lambda z: smooth_shift_obj(z, beta_sm), uv_opt, method="BFGS", options={"gtol": 1e-11}).x
    At_opt = Ja_obs - uv_opt[:4, None]
    Bt_opt = Jb_obs - uv_opt[4:, None]
    nonuni_shift_min = max(
        float(np.max(np.sum(np.abs(At_opt), axis=0))),
        float(np.max(np.sum(np.abs(Bt_opt), axis=0))),
        float(np.linalg.norm(np.hstack([At_opt, Bt_opt]), ord=2)),
    )
    assert nonuni_shift_min <= 1.0 + 1e-7

    rng_obs = np.random.default_rng(7)
    max_obs_primal = 0.0
    for p in [1.25, 1.5, 1.75]:
        def neg_primal_obs(z):
            x = z[:4] - np.mean(z[:4])
            y = z[4:] - np.mean(z[4:])
            denom = (np.sum(np.abs(x) ** p) + np.sum(np.abs(y) ** p)) ** (1.0 / p)
            if denom < 1e-12:
                return 0.0
            return -float(np.sum(np.abs(Ja_obs @ x + Jb_obs @ y) ** p) ** (1.0 / p)) / denom
        for _ in range(8):
            res_p = minimize(neg_primal_obs, rng_obs.normal(size=8), method="Nelder-Mead", options={"maxiter": 1500})
            max_obs_primal = max(max_obs_primal, float(-res_p.fun))
    assert max_obs_primal <= 1.0 + 1e-8
    print(f"  n=4 rank-1 shift (MA={MA_obs:.4f}, MB={MB_obs:.4f}): uniform min = {uni_shift_min:.6f} >= analytic LB {lb_uni:.6f} > 1; non-uniform min = {nonuni_shift_min:.8f} <= 1, primal = {max_obs_primal:.6f} <= 1")

    # n=5 dual-certified no-go for ALL linear extensions (J_a - u 1^T, J_b - v 1^T) (Remark 6.4)
    a5_obs = np.array([-1.024, -1.004, -0.984, -0.964, 3.976])
    b5_obs = np.array([-3.976, 0.964, 0.984, 1.004, 1.024])
    Ja5, _ = jacobian_alpha(a5_obs, b5_obs)
    Jb5, _ = jacobian_alpha(b5_obs, a5_obs)
    P5 = np.eye(5)[::-1]
    assert np.max(np.abs(Jb5 - P5 @ Ja5 @ P5)) < 1e-12
    P5_even = 0.5 * (np.eye(5) + P5)
    P5_odd = 0.5 * (np.eye(5) - P5)
    s0_5 = np.array([-1.0, -1.0, -1.0, -1.0, 1.0])

    def _M5(u_vec: np.ndarray) -> np.ndarray:
        v_vec = P5 @ u_vec
        At5 = Ja5 - u_vec[:, None]
        Bt5 = Jb5 - v_vec[:, None]
        return At5 @ At5.T + Bt5 @ Bt5.T

    res5_slsqp = minimize(
        lambda z: z[5],
        np.array([0.085, 0.058, 0.047, 0.035, 0.118, 1.003]),
        method="SLSQP",
        constraints=[
            {"type": "ineq", "fun": lambda z: z[5] - float(s0_5 @ (Ja5[:, 4] - z[:5]))},
            {"type": "ineq", "fun": lambda z: z[5] - np.sqrt(max(0.0, float(np.linalg.eigvalsh(P5_even @ _M5(z[:5]) @ P5_even)[-1])))},
            {"type": "ineq", "fun": lambda z: z[5] - np.sqrt(max(0.0, float(np.linalg.eigvalsh(P5_odd @ _M5(z[:5]) @ P5_odd)[-1])))},
        ],
        options={"ftol": 1e-13, "eps": 1e-9, "maxiter": 500},
    )
    u5_opt = res5_slsqp.x[:5]
    v5_opt = P5 @ u5_opt
    At5_opt, Bt5_opt = Ja5 - u5_opt[:, None], Jb5 - v5_opt[:, None]
    primal5_min = max(
        float(np.max(np.sum(np.abs(At5_opt), axis=0))),
        float(np.max(np.sum(np.abs(Bt5_opt), axis=0))),
        float(np.linalg.norm(np.hstack([At5_opt, Bt5_opt]), ord=2)),
    )
    M5_opt = At5_opt @ At5_opt.T + Bt5_opt @ Bt5_opt.T
    _, vec_e5 = np.linalg.eigh(P5_even @ M5_opt @ P5_even)
    _, vec_o5 = np.linalg.eigh(P5_odd @ M5_opt @ P5_odd)
    w_p5 = vec_e5[:, -1] / np.linalg.norm(vec_e5[:, -1])
    w_m5 = vec_o5[:, -1] / np.linalg.norm(vec_o5[:, -1])
    T5_opt = np.hstack([At5_opt, Bt5_opt])
    xy_p5 = T5_opt.T @ w_p5; xy_p5 /= np.linalg.norm(xy_p5)
    xy_m5 = T5_opt.T @ w_m5; xy_m5 /= np.linalg.norm(xy_m5)
    xp5, yp5, xm5, ym5 = xy_p5[:5], xy_p5[5:], xy_m5[:5], xy_m5[5:]
    val_p5 = float(w_p5 @ (Ja5 @ xp5 + Jb5 @ yp5))
    val_m5 = float(w_m5 @ (Ja5 @ xm5 + Jb5 @ ym5))

    c_lp5 = np.zeros(14)
    c_lp5[2], c_lp5[3] = -val_p5, -val_m5
    c_lp5[4:9], c_lp5[9:14] = -Ja5[:, 4], -Jb5[:, 0]
    A_eq5, b_eq5 = np.zeros((11, 14)), np.zeros(11)
    A_eq5[0, :4] = 1.0; b_eq5[0] = 1.0
    for ii in range(5):
        A_eq5[1 + ii, 4 + ii] = 1.0
        A_eq5[6 + ii, 9 + ii] = 1.0
        A_eq5[1 + ii, 2] = np.sum(xp5) * w_p5[ii]
        A_eq5[1 + ii, 3] = np.sum(xm5) * w_m5[ii]
        A_eq5[6 + ii, 2] = np.sum(yp5) * w_p5[ii]
        A_eq5[6 + ii, 3] = np.sum(ym5) * w_m5[ii]
    A_ub5, b_ub5 = np.zeros((20, 14)), np.zeros(20)
    for ii in range(5):
        A_ub5[ii, 4 + ii] = 1.0; A_ub5[ii, 0] = -1.0
        A_ub5[5 + ii, 4 + ii] = -1.0; A_ub5[5 + ii, 0] = -1.0
        A_ub5[10 + ii, 9 + ii] = 1.0; A_ub5[10 + ii, 1] = -1.0
        A_ub5[15 + ii, 9 + ii] = -1.0; A_ub5[15 + ii, 1] = -1.0
    res_lp5 = linprog(c_lp5, A_ub=A_ub5, b_ub=b_ub5, A_eq=A_eq5, b_eq=b_eq5, bounds=[(0, None)] * 4 + [(None, None)] * 10, method="highs")
    sol5 = res_lp5.x
    la5, lb5, lp5, lm5 = sol5[:4]
    sa5_raw, sb5_raw = sol5[4:9] / la5, sol5[9:14] / lb5
    scale5 = max(1.0, float(np.max(np.abs(sa5_raw))), float(np.max(np.abs(sb5_raw))))
    sa5, sb5 = sa5_raw / scale5, sb5_raw / scale5
    u5_res = la5 * sa5 + lp5 * np.sum(xp5) * w_p5 + lm5 * np.sum(xm5) * w_m5
    v5_res = lb5 * sb5 + lp5 * np.sum(yp5) * w_p5 + lm5 * np.sum(ym5) * w_m5
    dual5_raw = float(la5 * (sa5 @ Ja5[:, 4]) + lb5 * (sb5 @ Jb5[:, 0]) + lp5 * val_p5 + lm5 * val_m5)
    # Heuristic float64 primal-dual cross-check (HiGHS LP solutions carry solver tolerance ~1e-8
    # depending on SciPy version; res_penalty5 rigorously accounts for the LP residual on the
    # compact sublevel set f(u,v) <= 2, while verify_interval_certificates_appendix_a() provides
    # the standalone zero-residual 50-digit interval-arithmetic proof of inf f(u,v) >= 1.0021479).
    wsum5 = float(la5 + lb5 + lp5 + lm5)
    res_penalty5 = 3.0 * (float(np.linalg.norm(u5_res)) + float(np.linalg.norm(v5_res)))
    dual5_lb = (dual5_raw - res_penalty5) / wsum5
    assert abs(wsum5 - 1.0) < 1e-10
    assert max(float(np.max(np.abs(sa5))), float(np.max(np.abs(sb5)))) <= 1.0 + 1e-12
    assert float(np.linalg.norm(u5_res)) < 1e-6 and float(np.linalg.norm(v5_res)) < 1e-6
    assert dual5_lb > 1.0025 and primal5_min - dual5_lb < 1e-5

    # Check primal l^p(1^perp) contraction still holds on this n=5 pair
    max_p5_primal = 0.0
    for p in [1.1, 1.25, 1.5, 1.75, 1.9]:
        def neg_p5(z):
            x = z[:5] - np.mean(z[:5])
            y = z[5:] - np.mean(z[5:])
            denom = (np.sum(np.abs(x) ** p) + np.sum(np.abs(y) ** p)) ** (1.0 / p)
            if denom < 1e-12:
                return 0.0
            return -float(np.sum(np.abs(Ja5 @ x + Jb5 @ y) ** p) ** (1.0 / p)) / denom
        for _ in range(10):
            res_p5 = minimize(neg_p5, rng_obs.normal(size=10), method="Powell", options={"maxiter": 2500})
            max_p5_primal = max(max_p5_primal, float(-res_p5.fun))
    assert max_p5_primal <= 1.0 + 1e-8
    print(f"  n=5 linear-extension NO-GO: dual LB = {dual5_lb:.8f} <= primal min = {primal5_min:.8f} (gap={primal5_min-dual5_lb:.2e}), primal l^p ratio = {max_p5_primal:.6f} <= 1")

    # Quotient dual check inf_c1 ||A^T z - c1 1||_q^q + inf_c2 ||B^T z - c2 1||_q^q <= inf_c0 ||z - c0 1||_q^q on n=3,4
    for n in [3, 4]:
        max_quot = 0.0
        quot_pairs = make_test_pairs(n, rng_obs)
        for a_q, b_q in quot_pairs:
            Ja_q, _ = jacobian_alpha(a_q, b_q)
            Jb_q, _ = jacobian_alpha(b_q, a_q)
            for q in [2.5, 4.0]:
                def qnorm(v):
                    vmin, vmax = float(np.min(v)), float(np.max(v))
                    if vmax - vmin < 1e-15:
                        return 0.0
                    return float(
                        minimize_scalar(
                            lambda c: np.sum(np.abs(v - c) ** q),
                            bounds=(vmin, vmax),
                            method="bounded",
                            options={"xatol": 1e-12},
                        ).fun
                    )
                def neg_quot(z):
                    z0 = z - np.mean(z)
                    norm2 = np.linalg.norm(z0)
                    if norm2 < 1e-12:
                        return 0.0
                    z0 /= norm2
                    nz = qnorm(z0)
                    if nz < 1e-14:
                        return 0.0
                    return -(qnorm(Ja_q.T @ z0) + qnorm(Jb_q.T @ z0)) / nz
                for _ in range(6):
                    res_q = minimize(neg_quot, rng_obs.normal(size=n), method="Powell", options={"maxiter": 1500})
                    max_quot = max(max_quot, float(-res_q.fun))
        assert max_quot <= 1.0 + 1e-7
        print(f"  Quotient l^q dual check n={n} (30 random+adversarial pairs, Powell), q in {{2.5,4.0}}: max ratio = {max_quot:.6f} <= 1")


def verify_matrix_stam_edgeworth_and_unequal_scale() -> None:
    print("\n=== 5b. High-Precision Verification of Thm 4.4 (Unequal Scale), Thm 4.8 / Rem 4.9(iv), and Cor 5.2 ===")
    with mp.workdps(50):
        def poly_from_roots_mp(roots):
            p = [mp.mpf(1)]
            for r in roots:
                nxt = [mp.mpf(0)] * (len(p) + 1)
                for i, c in enumerate(p):
                    nxt[i] += c
                    nxt[i + 1] -= c * r
                p = nxt
            return p

        def boxplus_mp(a, b):
            n = len(a) - 1
            c = [mp.mpf(0)] * (n + 1)
            for k in range(n + 1):
                s = mp.mpf(0)
                for i in range(k + 1):
                    j = k - i
                    w = mp.factorial(n - i) * mp.factorial(n - j) / (mp.factorial(n) * mp.factorial(n - k))
                    s += w * a[i] * b[j]
                c[k] = s
            return c

        def poly_from_appell_mp(n, gammas):
            d = [mp.mpf(0)] * (n + 1)
            d[0] = mp.mpf(1)
            for m in range(1, n + 1):
                s = mp.mpf(0)
                for k in range(1, m + 1):
                    gk = gammas.get(k, mp.mpf(0))
                    if gk != 0:
                        s += gk / mp.factorial(k - 1) * d[m - k]
                d[m] = s / m
            return [d[k] * mp.factorial(n) / mp.factorial(n - k) for k in range(n + 1)]

        def roots_mp(coeffs):
            return sorted([mp.re(r) for r in mp.polyroots(coeffs, maxsteps=200, extraprec=40)])

        def phi_star_roots_mp(roots):
            n = len(roots)
            total = mp.mpf(0)
            for i in range(n):
                si = sum(mp.mpf(1) / (roots[i] - roots[j]) for j in range(n) if j != i)
                total += si * si
            return total

        def jac_alpha_mp(alpha, beta):
            n = len(alpha)
            p = poly_from_roots_mp(alpha)
            q = poly_from_roots_mp(beta)
            r_poly = boxplus_mp(p, q)
            gamma = roots_mp(r_poly)
            dr_poly = [r_poly[k] * (n - k) for k in range(n)]

            def eval_poly(poly, x):
                v = mp.mpf(0)
                for c in poly:
                    v = v * x + c
                return v

            J = mp.matrix(n, n)
            for j in range(n):
                sub = [alpha[k] for k in range(n) if k != j]
                dp = [mp.mpf(0)] + [-c for c in poly_from_roots_mp(sub)]
                dr = boxplus_mp(dp, q)
                for i in range(n):
                    J[i, j] = -eval_poly(dr, gamma[i]) / eval_poly(dr_poly, gamma[i])
            return J, gamma

        # 1. Theorem 4.4 unequal-scale local non-attainment at every Hermite pair (sigma_1 r, sigma_2 r)
        eps = mp.mpf("1e-3")
        for n in [4, 5]:
            for t_str in ["0.1", "0.3"]:
                t = mp.mpf(t_str)
                s1, s2 = mp.sqrt(t), mp.sqrt(1 - t)
                pa = poly_from_appell_mp(n, {2: -1, 3: mp.sqrt(t) * eps, 4: -mp.mpf(15) / 4 * t * eps**2})
                pb = poly_from_appell_mp(n, {2: -1, 3: mp.sqrt(1 - t) * eps, 4: -mp.mpf(15) / 4 * (1 - t) * eps**2})
                ra = [s1 * x for x in roots_mp(pa)]
                rb = [s2 * x for x in roots_mp(pb)]
                rg = roots_mp(boxplus_mp(poly_from_roots_mp(ra), poly_from_roots_mp(rb)))
                h1 = mp.mpf(4) / (n * (n - 1))
                inv_a = 1 / phi_star_roots_mp(ra)
                inv_b = 1 / phi_star_roots_mp(rb)
                inv_g = 1 / phi_star_roots_mp(rg)
                da = t * h1 - inv_a
                db = (1 - t) * h1 - inv_b
                Rn = ((inv_g - inv_a - inv_b) / inv_g) / (4 * t * (1 - t) * (da + db) / h1)
                s_val = t * (1 - t)
                u_val = (1 - 2 * t) ** 2
                P4 = -mp.mpf(9) / 32 - mp.mpf(189) / 32 * u_val - mp.mpf(207) / 16 * u_val**2 + mp.mpf(27) / 4 * u_val**3
                Pslope = -mp.mpf(45) / 32 - mp.mpf(297) / 32 * u_val - mp.mpf(261) / 16 * u_val**2 + mp.mpf(45) / 8 * u_val**3
                pred_slope = (P4 + (n - 4) * Pslope) / (72 * (1 - 2 * s_val))
                obs_slope = (Rn - mp.mpf("0.5")) / eps**2
                assert Rn < mp.mpf("0.5")
                assert abs(obs_slope - pred_slope) < mp.mpf("1e-4")
                print(f"  Thm 4.4 unequal-scale (n={n}, t={t_str}): R_n = {mp.nstr(Rn, 10)} < 0.5, slope err = {mp.nstr(abs(obs_slope - pred_slope), 3)}")

        # 2. Theorem 4.8: PSD Matrix Stam K(beta) >= 0, 1+(n-2)+1 limit 1/(n-1), Proof B Jacobian limit
        for n in [4, 5, 6]:
            ep = mp.mpf("1e-5")
            xi = [mp.mpf(j) - mp.mpf(n - 3) / 2 for j in range(n - 2)]
            beta = [mp.mpf(-1)] + [ep * x for x in xi] + [mp.mpf(1)]
            V_b = sum(x * x for x in beta) / (n * (n - 1))
            s_vec = mp.matrix(n, 1)
            L_mat = mp.matrix(n, n)
            for i in range(n):
                si = mp.mpf(0)
                diag = mp.mpf(0)
                for j in range(n):
                    if i != j:
                        si += 1 / (beta[i] - beta[j])
                        w = 2 / (beta[i] - beta[j]) ** 2
                        L_mat[i, j] = -w
                        diag += w
                s_vec[i] = si
                L_mat[i, i] = diag
            K_mat = L_mat - (mp.mpf(4) / (n * (n - 1))) * (s_vec * s_vec.T)
            eigs_K, _ = mp.eigsy(K_mat)
            assert eigs_K[0] > -mp.mpf("1e-35") and abs(eigs_K[1]) < mp.mpf("1e-35")
            dyson_c100 = mp.mpf("0.25") * V_b * eigs_K[2]
            assert abs(dyson_c100 - mp.mpf(1) / (n - 1)) < mp.mpf("1e-7")
            print(f"  Thm 4.8 K(beta) PSD & limit n={n}: min_eig={mp.nstr(eigs_K[0], 3)}, (1/4)V*lam_min_perp={mp.nstr(dyson_c100, 10)} (target 1/{n-1}={mp.nstr(mp.mpf(1)/(n-1), 10)})")

        # Proof B Jacobian limit at n=4, 5
        for n in [4, 5]:
            ep = mp.mpf("1e-5")
            a_raw = [-1 + ep * (j - mp.mpf(n - 2) / 2) for j in range(n - 1)] + [mp.mpf(n - 1)]
            b_raw = [-x for x in reversed(a_raw)]
            Ja_b, _ = jac_alpha_mp(a_raw, b_raw)
            target_MA = mp.mpf("0.5") * (1 + mp.sqrt(1 - mp.mpf(1) / n))
            assert abs(Ja_b[n - 1, n - 1] - target_MA) < mp.mpf("1e-8")

        # 3. Remark 4.9(iv): 50-digit witness c_4^(100) <= 0.3141332030775802 < 1/3
        a0 = [mp.mpf(-1), mp.mpf("-1e-3"), mp.mpf("1e-3"), mp.mpf(1)]
        b0 = [mp.mpf(-3), 1 - mp.mpf("1e-3"), mp.mpf(1), 1 + mp.mpf("1e-3")]
        t_w = mp.mpf("0.725")
        na0 = mp.sqrt(sum(x * x for x in a0))
        nb0 = mp.sqrt(sum(x * x for x in b0))
        a_w = [mp.sqrt(t_w) * x / na0 for x in a0]
        b_w = [mp.sqrt(1 - t_w) * x / nb0 for x in b0]
        Ja_w, _ = jac_alpha_mp(a_w, b_w)
        Jb_w, _ = jac_alpha_mp(b_w, a_w)
        M_w = Ja_w * Ja_w.T + Jb_w * Jb_w.T
        # Project onto 1^perp via orthonormal basis of 1^perp in R^4
        W4 = mp.matrix([
            [1 / mp.sqrt(2), 1 / mp.sqrt(6), 1 / mp.sqrt(12)],
            [-1 / mp.sqrt(2), 1 / mp.sqrt(6), 1 / mp.sqrt(12)],
            [0, -2 / mp.sqrt(6), 1 / mp.sqrt(12)],
            [0, 0, -3 / mp.sqrt(12)],
        ])
        M_perp = W4.T * M_w * W4
        eigs_w, _ = mp.eigsy(M_perp)  # ascending
        lam1_w, lam2_w = eigs_w[2], eigs_w[1]
        c4_100 = (1 - lam2_w) / (4 * t_w * (1 - t_w))
        assert abs(lam1_w - 1) < mp.mpf("1e-40")
        assert c4_100 < mp.mpf("0.3141332030775803") and c4_100 > mp.mpf("0.3141332030775801")
        print(f"  Rem 4.9(iv) 50-digit witness: lam2 = {mp.nstr(lam2_w, 16)}, c_4^(100) <= {mp.nstr(c4_100, 16)} < 1/3")

        # 4. Corollary 5.2: Universal two-term CLT Edgeworth expansion with nonzero gamma_5, gamma_6
        for n in [5, 6]:
            g3 = mp.mpf("0.7")
            g4 = mp.mpf("-1.3")
            g5 = mp.mpf("2.1")
            g6 = mp.mpf("-3.7") if n >= 6 else mp.mpf(0)
            m_clt = mp.mpf("100000")
            h_clt = 1 / mp.sqrt(m_clt)
            coeffs = poly_from_appell_mp(n, {2: -1, 3: g3 * h_clt, 4: g4 * h_clt**2, 5: g5 * h_clt**3, 6: g6 * h_clt**4})
            phi_ratio = phi_star_roots_mp(roots_mp(coeffs)) / (mp.mpf(n * (n - 1)) / 4)
            c1 = mp.mpf(n - 2) / 4 * g3**2
            c2 = (
                mp.mpf((n - 2) * (n - 3)) / 36 * g4**2
                + mp.mpf(5 * (n - 2) * (n - 3)) / 24 * g3**2 * g4
                + mp.mpf((n - 2) * (5 * n - 14)) / 16 * g3**4
            )
            err_m3 = abs(phi_ratio - (1 + c1 / m_clt + c2 / m_clt**2)) * m_clt**3
            assert err_m3 < mp.mpf("1.0")
            print(f"  Cor 5.2 two-term Edgeworth n={n} (g5={float(g5)}, g6={float(g6)}): |err|*m^3 = {mp.nstr(err_m3, 6)} (O(m^-3) confirmed)")


def verify_exact_rational_quartic_certificate() -> None:
    """Exact Q verification of Remark 4.9(i): c_4^(101) <= 876/1855 (and 761590/1625603 at c_0 = 21/10)."""
    print("\n=== 0. Exact Rational Arithmetic Verification (Remark 4.9(i)) ===")

    def _eval_quartic_dyson_q(c2_f: Fraction, c1_f: Fraction, c0_f: Fraction) -> tuple[Fraction, Fraction, Fraction]:
        disc_f = (
            256 * c0_f ** 3 - 128 * (c2_f ** 2) * (c0_f ** 2) + 144 * c2_f * (c1_f ** 2) * c0_f
            - 27 * c1_f ** 4 + 16 * (c2_f ** 4) * c0_f - 4 * (c2_f ** 3) * (c1_f ** 2)
        )
        num_f = (
            384 * c2_f * (c0_f ** 2) - 64 * (c2_f ** 3) * c0_f - 8 * c2_f ** 5
            - 432 * (c1_f ** 2) * c0_f - 36 * (c2_f ** 2) * (c1_f ** 2)
        )
        ddisc_dc2 = -256 * c2_f * (c0_f ** 2) + 144 * (c1_f ** 2) * c0_f + 64 * (c2_f ** 3) * c0_f - 12 * (c2_f ** 2) * (c1_f ** 2)
        ddisc_dc0 = 768 * (c0_f ** 2) - 256 * (c2_f ** 2) * c0_f + 144 * c2_f * (c1_f ** 2) + 16 * (c2_f ** 4)
        dnum_dc2 = 384 * (c0_f ** 2) - 192 * (c2_f ** 2) * c0_f - 40 * (c2_f ** 4) - 72 * c2_f * (c1_f ** 2)
        dnum_dc0 = 768 * c2_f * c0_f - 64 * (c2_f ** 3) - 432 * (c1_f ** 2)
        disc_dot = -6 * ddisc_dc2 - c2_f * ddisc_dc0
        num_dot = -6 * dnum_dc2 - c2_f * dnum_dc0
        inv_phi = disc_f / num_f
        pdot = (disc_dot * num_f - disc_f * num_dot) / (num_f ** 2)
        h1 = Fraction(1, 3)
        r4_lim = (pdot - h1) / (4 * inv_phi * (1 - inv_phi / h1))
        return inv_phi, pdot, r4_lim

    # Integer quartic q(x) = x^4 - 6x^2 + 2x + 2
    q_signs = [x ** 4 - 6 * x ** 2 + 2 * x + 2 for x in (-3, -1, 0, 1, 3)]
    assert q_signs == [23, -5, 2, -1, 35]
    inv_phi_1, pdot_1, r4_1 = _eval_quartic_dyson_q(Fraction(-6), Fraction(2), Fraction(2))
    assert inv_phi_1 == Fraction(53, 180) and pdot_1 == Fraction(448, 1125) and r4_1 == Fraction(876, 1855)
    print(f"  Integer quartic x^4 - 6x^2 + 2x + 2: c_4^(101) <= {r4_1} = {float(r4_1):.6f} < 1/2")

    # Rational quartic with c_0 = 21/10
    q2_signs = [Fraction(x) ** 4 - 6 * Fraction(x) ** 2 + 2 * Fraction(x) + Fraction(21, 10) for x in (-3, -1, 0, 1, 3)]
    assert q2_signs[0] > 0 and q2_signs[1] < 0 and q2_signs[2] > 0 and q2_signs[3] < 0 and q2_signs[4] > 0
    _, _, r4_2 = _eval_quartic_dyson_q(Fraction(-6), Fraction(2), Fraction(21, 10))
    assert r4_2 == Fraction(761590, 1625603)
    print(f"  Rational quartic c_0 = 21/10:        c_4^(101) <= {r4_2} = {float(r4_2):.6f} < 1/2")


def verify_interval_certificates_appendix_a() -> None:
    print("\n=== 0b. Rigorous 50-Digit Interval-Arithmetic Verification of Appendix A Certificates ===")
    prev_iv_dps = iv.dps
    iv.dps = 50
    try:
        with mp.workdps(50):
            def poly_from_roots_iv(roots):
                p = [iv.mpf(1)]
                for r in roots:
                    nxt = [iv.mpf(0)] * (len(p) + 1)
                    for i, c in enumerate(p):
                        nxt[i] += c
                        nxt[i + 1] -= c * r
                    p = nxt
                return p

            def boxplus_iv(a, b):
                n = len(a) - 1
                c = [iv.mpf(0)] * (n + 1)
                for k in range(n + 1):
                    s = iv.mpf(0)
                    for i in range(k + 1):
                        j = k - i
                        w = iv.mpf(mp.factorial(n - i) * mp.factorial(n - j)) / iv.mpf(mp.factorial(n) * mp.factorial(n - k))
                        s += w * a[i] * b[j]
                    c[k] = s
                return c

            def eval_poly_iv(poly, x):
                v = iv.mpf(0)
                for c in poly:
                    v = v * x + c
                return v

            def enclose_roots_iv(poly_iv, approx_roots_mp):
                n = len(poly_iv) - 1
                dpoly_iv = [poly_iv[k] * (n - k) for k in range(n)]
                enclosed = []
                rad = mp.mpf("1e-35")
                for r0 in approx_roots_mp:
                    lo, hi = r0 - rad, r0 + rad
                    assert float((eval_poly_iv(poly_iv, iv.mpf(lo)) * eval_poly_iv(poly_iv, iv.mpf(hi))).b) < 0.0
                    bracket = iv.mpf([lo, hi])
                    dval = eval_poly_iv(dpoly_iv, bracket)
                    assert float(dval.a) > 0.0 or float(dval.b) < 0.0
                    enclosed.append(bracket)
                for i in range(n - 1):
                    assert float(enclosed[i].b) < float(enclosed[i + 1].a)
                return enclosed

            def jacobian_iv(alpha_iv, beta_iv, approx_gamma_mp):
                n = len(alpha_iv)
                p = poly_from_roots_iv(alpha_iv)
                q = poly_from_roots_iv(beta_iv)
                r_poly = boxplus_iv(p, q)
                gamma = enclose_roots_iv(r_poly, approx_gamma_mp)
                dr_poly = [r_poly[k] * (n - k) for k in range(n)]
                J = [[iv.mpf(0)] * n for _ in range(n)]
                for j in range(n):
                    sub = [alpha_iv[k] for k in range(n) if k != j]
                    dp = [iv.mpf(0)] + [-c for c in poly_from_roots_iv(sub)]
                    dr = boxplus_iv(dp, q)
                    for i in range(n):
                        J[i][j] = -eval_poly_iv(dr, gamma[i]) / eval_poly_iv(dr_poly, gamma[i])
                return J, gamma

            # Certificate A.1: Degree-4 eq. (100) bound c_4^(100) <= 0.31413321 < 1/3
            a0_q = [iv.mpf(-1), iv.mpf(-1) / 1000, iv.mpf(1) / 1000, iv.mpf(1)]
            b0_q = [iv.mpf(-3), iv.mpf(999) / 1000, iv.mpf(1), iv.mpf(1001) / 1000]
            t_q = iv.mpf(29) / 40
            na = iv.sqrt(sum(x * x for x in a0_q))
            nb = iv.sqrt(sum(x * x for x in b0_q))
            a_iv = [iv.sqrt(t_q) * x / na for x in a0_q]
            b_iv = [iv.sqrt(1 - t_q) * x / nb for x in b0_q]
            p4_mp = [mp.mpf(c.mid) for c in boxplus_iv(poly_from_roots_iv(a_iv), poly_from_roots_iv(b_iv))]
            g4_approx = sorted([mp.re(r) for r in mp.polyroots(p4_mp, maxsteps=200, extraprec=40)])
            Ja4_iv, g4_iv = jacobian_iv(a_iv, b_iv, g4_approx)
            Jb4_iv, _ = jacobian_iv(b_iv, a_iv, g4_approx)

            z0_iv = [iv.mpf(v) for v in (649, -512, -1000, 863)]
            mean_z = sum(z0_iv) / 4
            z1_iv = [v - mean_z for v in z0_iv]
            g_norm2 = sum(x * x for x in g4_iv)
            dot_zg = sum(z1_iv[i] * g4_iv[i] for i in range(4))
            z_iv = [z1_iv[i] - (dot_zg / g_norm2) * g4_iv[i] for i in range(4)]
            num_iv = iv.mpf(0)
            for j in range(4):
                va = sum(Ja4_iv[i][j] * z_iv[i] for i in range(4))
                vb = sum(Jb4_iv[i][j] * z_iv[i] for i in range(4))
                num_iv += va * va + vb * vb
            rq_iv = num_iv / sum(v * v for v in z_iv)
            c4_ub_iv = (1 - rq_iv) / (4 * t_q * (1 - t_q))
            assert float(rq_iv.a) >= 0.749478769775 and float(c4_ub_iv.b) < 0.31413321
            print(
                f"  Cert A.1 (mpmath.iv): Rayleigh in [{mp.nstr(mp.mpf(rq_iv.a), 14)}, {mp.nstr(mp.mpf(rq_iv.b), 14)}], "
                f"c_4^(100) <= {mp.nstr(mp.mpf(c4_ub_iv.b), 12)} < 0.31413321 < 1/3"
            )

            # Certificate A.2: Zero-residual disjoint-support degree-5 linear-extension obstruction
            a5_iv = [iv.mpf(-128) / 125, iv.mpf(-251) / 250, iv.mpf(-123) / 125, iv.mpf(-241) / 250, iv.mpf(497) / 125]
            b5_iv = [-x for x in reversed(a5_iv)]
            p5_mp = [mp.mpf(c.mid) for c in boxplus_iv(poly_from_roots_iv(a5_iv), poly_from_roots_iv(b5_iv))]
            g5_approx = sorted([mp.re(r) for r in mp.polyroots(p5_mp, maxsteps=200, extraprec=40)])
            Ja5_iv, _ = jacobian_iv(a5_iv, b5_iv, g5_approx)
            Jb5_iv, _ = jacobian_iv(b5_iv, a5_iv, g5_approx)

            l1_iv = iv.mpf(805) / 10000
            l2_iv = iv.mpf(7958) / 10000
            la_iv = iv.mpf(1237) / 20000
            assert 2 * 1237 + 2 * 805 + 2 * 7958 == 20000

            def norm_iv(vec_ints):
                v = [iv.mpf(x) for x in vec_ints]
                nrm = iv.sqrt(sum(x * x for x in v))
                return [x / nrm for x in v]

            w1_iv = norm_iv([0, -1, -1, -1, 0])
            w2_iv = norm_iv([1, 0, 0, 0, -1])
            x1_int = [-100, -100, -100, -100, 23]
            y1_int = list(reversed(x1_int))
            x2_int = [30, 30, 29, 29, -100]
            y2_int = [-v for v in reversed(x2_int)]
            xy1_iv = norm_iv(x1_int + y1_int)
            x1_iv, y1_iv = xy1_iv[:5], xy1_iv[5:]
            xy2_iv = norm_iv(x2_int + y2_int)
            x2_iv, y2_iv = xy2_iv[:5], xy2_iv[5:]

            sum_x1, sum_x2 = sum(x1_iv), sum(x2_iv)
            sum_y1, sum_y2 = sum(y1_iv), sum(y2_iv)
            sa_iv = [-(l1_iv * sum_x1 * w1_iv[i] + l2_iv * sum_x2 * w2_iv[i]) / la_iv for i in range(5)]
            sb_iv = [-(l1_iv * sum_y1 * w1_iv[i] + l2_iv * sum_y2 * w2_iv[i]) / la_iv for i in range(5)]
            max_sa = max(float(abs(x).b) for x in sa_iv)
            max_sb = max(float(abs(x).b) for x in sb_iv)
            assert max_sa < 0.99731 and max_sb < 0.99731

            def dot_iv(a, b):
                return sum(a[i] * b[i] for i in range(len(a)))

            def matvec_iv(M, v):
                return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

            col_a5 = [Ja5_iv[i][4] for i in range(5)]
            col_b1 = [Jb5_iv[i][0] for i in range(5)]
            lb_iv = (
                la_iv * dot_iv(sa_iv, col_a5)
                + la_iv * dot_iv(sb_iv, col_b1)
                + l1_iv * (dot_iv(w1_iv, matvec_iv(Ja5_iv, x1_iv)) + dot_iv(w1_iv, matvec_iv(Jb5_iv, y1_iv)))
                + l2_iv * (dot_iv(w2_iv, matvec_iv(Ja5_iv, x2_iv)) + dot_iv(w2_iv, matvec_iv(Jb5_iv, y2_iv)))
            )
            assert float(lb_iv.a) >= 1.002147929052
            print(f"  Cert A.2 (mpmath.iv, zero residual): max|s_a| <= {max_sa:.6f} < 1, inf f(u,v) >= {mp.nstr(mp.mpf(lb_iv.a), 14)} > 1")
    finally:
        iv.dps = prev_iv_dps


def generate_figures() -> None:
    print("\n=== 6. Generating Publication Figures ===")
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # Figure 1: Spectrum & Entrywise Semigroup Structure of E_n
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # Panel A: Singular values / eigenvalues of E_n(sqrt(t), sqrt(1-t)) vs k
    ax = axes[0]
    t_vals = [0.2, 0.5, 0.8]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    n_plot = 8
    r, U = hermite_eigenbasis(n_plot)
    L = log_gas_laplacian(r)
    ks = np.arange(n_plot)
    for t, col in zip(t_vals, colors):
        s1, s2 = np.sqrt(t), np.sqrt(1.0 - t)
        J, _ = jacobian_alpha(s1 * r, s2 * r)
        evals = np.sort(np.linalg.eigvalsh(J))[::-1]
        ax.semilogy(ks, evals, "o", color=col, markersize=7, label=f"Computed $E_8$ ($t={t}$)")
        ax.semilogy(ks, t ** (ks / 2.0), "--", color=col, alpha=0.75, label=f"Theory $t^{{k/2}}$ ($t={t}$)")
    ax.set_xlabel("Mode index $k$")
    ax.set_ylabel("Eigenvalue $\\lambda_k(E_8)$")
    ax.set_title("(a) Geometric Spectrum of $E_8(\\sqrt{t}, \\sqrt{1-t})$")
    ax.set_xticks(ks)
    ax.set_ylim(1.0e-3, 1.4)
    ax.legend(fontsize=7.5, ncol=3, loc="lower left")

    # Panel B: Heatmap of E_8(1/sqrt(2), 1/sqrt(2)) = exp(-(ln 2)/2 * L_8)
    ax = axes[1]
    E_half = expm(-0.5 * np.log(2.0) * L)
    im = ax.imshow(E_half, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Entry value $(E_8)_{j,i}$")
    ax.set_title("(b) Doubly Stochastic Coupling Matrix $E_8$ ($t=1/2$)")
    ax.set_xlabel("Input root index $i$")
    ax.set_ylabel("Output root index $j$")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks + 1)
    ax.set_yticks(ks)
    ax.set_yticklabels(ks + 1)

    fig.tight_layout()
    fig.savefig("fig_spectrum_semigroup.png", dpi=200)
    plt.close(fig)

    # Figure 2: Finite Free CLT Rates (O(1/m) vs O(1/m^2)) and Stam Hessian Verification
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # Panel A: CLT convergence
    ax = axes[0]
    ms = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
    n = 4
    r0, _ = hermegauss(n)
    phi_H = np.sum(score_vector(r0) ** 2)

    # Asymmetric case
    rng = np.random.default_rng(1)
    a_asym = rng.normal(size=n)
    a_asym -= a_asym.mean()
    a_asym *= np.sqrt(np.sum(r0 ** 2) / np.sum(a_asym ** 2))
    c1 = (np.sum(a_asym ** 3) ** 2) / ((n ** 2) * ((n - 1) ** 2) * (n - 2))

    # Symmetric case
    a_sym = np.array([-2.0, -1.0, 1.0, 2.0])
    a_sym = a_sym * np.sqrt(np.sum(r0 ** 2) / np.sum(a_sym ** 2))
    he_c = roots_to_elementary(r0)
    sym_c = roots_to_elementary(a_sym)
    c2 = 16.0 * ((sym_c[4] - he_c[4]) ** 2) / (n * (n - 1) * factorial(n))

    def clt_excess(alpha: np.ndarray) -> list[float]:
        a_base = roots_to_elementary(alpha)
        vals = []
        for m in ms:
            cur = np.zeros(n + 1)
            cur[0] = 1.0
            base = a_base.copy()
            mm = int(m)
            while mm > 0:
                if mm & 1:
                    cur = boxplus_coeffs(cur, base)
                mm >>= 1
                if mm > 0:
                    base = boxplus_coeffs(base, base)
            poly = np.array([((-1) ** k) * cur[k] for k in range(n + 1)])
            roots_m = real_roots_checked(poly) / np.sqrt(m)
            vals.append(np.sum(score_vector(roots_m) ** 2) / phi_H - 1.0)
        return vals

    ex_asym = clt_excess(a_asym)
    ex_sym = clt_excess(a_sym)
    ax.loglog(ms, ex_asym, "o", color="#1f77b4", label="Asymmetric ($e_3 \\neq 0$)")
    ax.loglog(ms, c1 / ms, "--", color="#1f77b4", label="Theory $C_3 / m$")
    ax.loglog(ms, ex_sym, "s", color="#d62728", label="Symmetric ($e_3 = 0$)")
    ax.loglog(ms, c2 / (ms ** 2), "--", color="#d62728", label="Theory $C_4 / m^2$")
    ax.set_xlabel("Convolution count $m$")
    ax.set_ylabel("Relative Fisher excess $\\Phi_4^*(p_m)/\\Phi_4^*(He_4) - 1$")
    ax.set_title("(a) Finite Free CLT Convergence Rates ($n=4$)")
    ax.legend(fontsize=8.5)

    # Panel B: Stam deficit quadratic scaling along modes k=2,3,4 for n=5
    ax = axes[1]
    n_h = 5
    r5, U5 = hermite_eigenbasis(n_h)
    eps_grid = np.logspace(-4, -1.3, 12)
    for k, col, marker in [(2, "#1f77b4", "o"), (3, "#ff7f0e", "s"), (4, "#2ca02c", "^")]:
        u = U5[:, k]
        obs_vals = []
        for eps in eps_grid:
            g, _ = boxplus_roots(r5 + eps * u, r5 + eps * u)
            d = (
                1.0 / np.sum(score_vector(g) ** 2)
                - 2.0 / np.sum(score_vector(r5 + eps * u) ** 2)
            )
            obs_vals.append(d)
        pred_coeff = 4.0 / ((n_h * (n_h - 1)) ** 2) * ((k + 1) ** 2) * (2.0 - 2.0 ** (2 - k))
        ax.loglog(eps_grid, obs_vals, marker, color=col, label=f"Mode $k={k}$ (num.)")
        ax.loglog(eps_grid, pred_coeff * (eps_grid ** 2), "--", color=col, label=f"Theory $c_{k}\\varepsilon^2$")
    ax.set_xlabel("Perturbation amplitude $\\varepsilon$")
    ax.set_ylabel("Stam deficit $\\Delta_5(r + \\varepsilon u^{(k)}, r + \\varepsilon u^{(k)})$")
    ax.set_title("(b) Exact Modal Stam Deficit Scaling ($n=5$)")
    ax.set_ylim(4e-10, 2.5e-2)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")

    fig.tight_layout()
    fig.savefig("fig_stam_clt.png", dpi=200)
    plt.close(fig)
    print("  Saved fig_spectrum_semigroup.png and fig_stam_clt.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verification suite for Finite Free Stam Stability, CLT Asymptotics, and p-Stam Inequalities."
    )
    parser.add_argument(
        "--certificates-only",
        action="store_true",
        help="Run only the exact rational (Remark 4.9(i)) and 50-digit mpmath.iv interval certificates (Appendix A).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip regenerating publication PNG figures.",
    )
    args = parser.parse_args()

    # Run rigorous exact-Q and 50-digit interval-arithmetic certificates first
    verify_exact_rational_quartic_certificate()
    verify_interval_certificates_appendix_a()

    if not args.certificates_only:
        verify_mpmath_high_precision()
        verify_coupling_semigroup_and_mehler()
        verify_stam_hessian()
        verify_clt_rates()
        verify_score_identity_and_p_stam()
        verify_matrix_stam_edgeworth_and_unequal_scale()
        if not args.no_figures:
            generate_figures()

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY.")
