# Quantitative Stability, Exact CLT Asymptotics, and $\ell^p$-Stam Inequalities in Finite Free Information Theory

**Author / Workflow Director:** Tristan Schubert  
**Status:** Transparent AI-assisted working draft (pending independent verification by a human specialist in finite free probability)

---

## Provenance & Epistemic Status

This repository contains a working draft and self-contained verification suite studying quantitative stability, finite free CLT asymptotics, and $\ell^p$-Stam inequalities around the Hermite extremizer of finite free additive convolution $\boxplus_n$, building on:

- J. Garza-Vargas, N. Srivastava, and Z. Stier (2026), *Finite Free Information Inequalities*, `arXiv:2602.15822`
- B. Hashemi (2026), *Spectral Structure in Finite Free Information Inequalities and $p$-Stam Phase Transitions*, `arXiv:2604.11922`
- B. Hashemi and J. Hyun (2026), *When Finite Free Curves Split*, `arXiv:2609.10367`

This work was produced through a **human-directed, multi-agent AI research workflow** combining proof drafting, exact symbolic/rational checks, 50-digit interval arithmetic (`mpmath.iv`), citation auditing against the primary arXiv LaTeX sources, and multiple rounds of independent adversarial AI review.

- **Analytic proofs (Sections 2–6):** Checked symbolically, numerically, and across multiple AI audit rounds, but **not yet verified line-by-line by a human specialist in finite free probability**.
- **Exact rational proof (Remark 4.9(i)):** Verified in exact $\mathbb{Q}$ arithmetic (`fractions.Fraction`).
- **Rigorous interval-arithmetic certificates (Appendix A, Remarks 4.9(iv) & 6.4):** Verified in 50-digit interval arithmetic (`mpmath.iv`) using explicit short integer/rational data printed in Appendix A (`mp.polyroots` is used only to propose candidate root brackets, which are then rigorously certified via interval sign-change, disjointness, and non-vanishing derivative enclosures, with no floating-point root-finder or eigensolver trusted inside the proof).
- **Exploratory numerics (Table 2):** Clearly separated and labeled as numerical exploration only.

Independent mathematical review, critique, bug reports, or collaboration from specialists in finite free probability are warmly welcomed.

---

## Main Results

1. **Quantitative Finite Free Stam Stability (`arXiv:2609.10367`, Conjecture 8.1):**
   - Exact $2 \times 2$ block-diagonal Hessian of the Stam deficit $\Delta_n$ at every Hermite pair $(\sigma_1 r, \sigma_2 r)$ (Theorem 4.1) and sharp $n$-independent *asymptotic* local constant $c_n^{\mathrm{loc}} = 1/2$ (Corollary 4.2).
   - Algebraic proof via an Appell backward-heat Euler operator that literal $c_n = 1/2$ fails on every open neighborhood of *every* Hermite pair for all $n \ge 4$ (Theorem 4.4).
   - Elementary cubic discriminant proof of global additive stability $\Delta_3 \ge 2t(1-t)(\delta_3(\alpha) + \delta_3(\beta))$ for $n = 3$ (Theorem 4.6), colliding-root obstruction for $n \ge 4$ (Proposition 4.5), Dyson-flow bound $c_n^{(101)} \le 2/n + O(n^{-2})$ (Proposition 4.7; exact rational quartic bound $c_4^{(101)} \le 876/1855$), and exact error-term-free bound $c_n^{(100)} \le \frac{1}{n-1}$ via the divided-difference defect operator $K(\beta) \succeq 0$ (Theorem 4.8; interval-certified $c_4^{(100)} \le 0.31413321 < \frac{1}{3}$ in Appendix A).

2. **Exact Finite Free CLT Fisher-Information Asymptotics (Section 5):**
   - Exact leading Fisher-information asymptotics for $m$-fold finite free convolution for any cumulant-matching order $j$, with parity vanishing of all half-integer powers of $m^{-1}$ (Theorem 5.1).
   - Universal exact two-term Edgeworth expansion up to $O(m^{-3})$ independent of $\gamma_5, \dots, \gamma_n$, and the exact open cone in $(\gamma_3, \gamma_4)$-space where $R_n(p_m, p_m) < 1/2$ along the CLT tail (Corollary 5.2).

3. **$\ell^p$-Stam Contraction (`arXiv:2604.11922`, Conjecture 5.2):**
   - Proof of doubly stochastic $\ell^p(\mathbf{1}^\perp)$ contraction for $n = 3$, $p \in [1, 2]$ via explicit rank-$1$ ambient extensions and complex Riesz–Thorin interpolation, settling Conjecture 5.2 of `arXiv:2604.11922` for all monic real-rooted cubics and $p \in (1, 2]$ (Theorem 6.3).
   - Analytic uniform-shift obstruction at $n = 4$ and an interval-certified, zero-residual dual obstruction at $n = 5$ ($\inf_{u,v \in \mathbb{R}^5} f(u,v) \ge 1.0021479 > 1$, Appendix A) ruling out a universal all-degree ambient linear-extension proof and motivating an intrinsic quotient $\ell^q$ dual formulation (Remark 6.4).

---

## Repository Contents

| File | Description |
| :--- | :--- |
| `paper.pdf` | Compiled 33-page working draft (with Appendix A interval-arithmetic certificates) |
| `paper.tex` | Self-contained LaTeX source (compiles cleanly with `pdflatex`) |
| `paper.md` | Markdown summary of all definitions, theorems, proofs, and certificates |
| `verify_theorems.py` | Self-contained Python verification suite (exact $\mathbb{Q}$, 50-digit `mpmath.iv`, numerical checks, and figure generation) |
| `fig_spectrum_semigroup.png` | Figure 1: Spectrum and coupling matrix of $E_8(\sqrt{t}, \sqrt{1-t})$ |
| `fig_stam_clt.png` | Figure 2: Finite free CLT convergence rates and modal Stam deficit scaling |
| `requirements.txt` | Python dependencies for `verify_theorems.py` |
| `CITATION.cff` | Machine-readable citation metadata |
| `LICENSE` | MIT License (code) and CC BY 4.0 notice (manuscript & figures) |
| `.github/workflows/verify.yml` | GitHub Actions CI workflow running the verification suite |

---

## Reproducing the Verification & Figures

```bash
pip install -r requirements.txt

# Fast check (~2 s): exact-Q quartic bound (Remark 4.9(i)) + 50-digit mpmath.iv Appendix A certificates
python verify_theorems.py --certificates-only

# Full suite (~1–2 min): rigorous certificates, all high-precision checks, exploratory Table 2 runs, and Figures 1–2
python verify_theorems.py
```

The script runs the rigorous exact-$\mathbb{Q}$ and 50-digit `mpmath.iv` interval-arithmetic certificates first, then checks all analytical formulas in Sections 3–6, reproduces the exploratory Table 2 checks, and regenerates Figures 1–2 (pass `--no-figures` to skip PNG generation).

To recompile the PDF from source:

```bash
pdflatex paper.tex
pdflatex paper.tex
```

## License

Code (`verify_theorems.py`) is released under the **MIT License**; the manuscript and figures (`paper.tex`, `paper.pdf`, `paper.md`, `*.png`) are released under **CC BY 4.0** (see [`LICENSE`](LICENSE)).
