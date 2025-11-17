<div>
<script>
window.MathJax = {
  tex: {
    displayMath: [['$$','$$'], ['\\[','\\]']]
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</div>

# Theorem 2 — Polynomial Approximation of the Accuracy Functional  

In this section, we establish that the accuracy functional arising from structured pruning can be uniformly approximated by finite‑order bivariate polynomials.  

---

## Preliminaries

Let  
$$
\mathcal D=\{0,\tfrac{1}{L},\dots,1\}^2 \subset [0,1]^2
$$
denote the discrete set of admissible pruning ratios.  
The accuracy functional evaluated at these ratios is denoted
$$
\mathcal P:\mathcal D\to\mathbb R.
$$

We assume that $\mathcal P$ admits a continuous extension  
$$
\tilde{\mathcal P}:[0,1]^2\to\mathbb R,
\qquad
\tilde{\mathcal P}|_{\mathcal D}=\mathcal P.
$$

### 1.1 Justification of the Continuity Assumption

This assumption is natural and well‑motivated for two reasons.

1. **Intermediate pruning ratios as relaxed constraints.**  Although structured pruning decisions are discrete, any intermediate pruning ratio  
   $$
   (\tilde m_a,\tilde m_g)\in[0,1]^2
   $$
   can be interpreted as a *continuous relaxation* akin to unstructured pruning, where the ratio corresponds to a soft sparsity budget applied to attention and activation parameters.  Continuous sparsity relaxations are standard in pruning theory, and model performance typically varies smoothly with respect to the imposed sparsity level.
   
2. **Empirically observed smoothness in one‑dimensional pruning.**  When pruning is restricted to a single dimension—either  
   $$
   (\tilde m_a,0)
   \quad\text{or}\quad
   (0,\tilde m_g),
   $$
   empirical results show that validation accuracy evolves **smoothly** as a function of the pruning ratio.  Such one‑dimensional smoothness strongly suggests a smooth two‑dimensional accuracy surface with respect to both pruning ratios.

These considerations jointly justify modeling the accuracy surface as a continuous mapping on $[0,1]^2$.

---

# Lemma 1 — Compactness and Regularity of the Domain

**Lemma.**  The domain $[0,1]^2$ is a compact Hausdorff space.

**Proof.**  Each interval $[0,1]$ is compact and Hausdorff.  Products of compact Hausdorff spaces remain compact and Hausdorff.  Thus $[0,1]^2$ satisfies the topological assumptions required for the Stone–Weierstrass theorem.  

---

# Lemma 2 — Polynomial Algebra as a Separating Lattice

**Lemma.**  Let  
$$
A = \mathbb R[x,y]
$$
be the algebra of all finite‑order real polynomials in two variables.  Then $A$ is a separating lattice that contains constants.

**Proof.**  

1. **Point separation.**  For any distinct points $(x_1,y_1)\neq (x_2,y_2)$, the polynomial  
   $$
   p(x,y)=x
   \quad\text{or}\quad
   p(x,y)=y
   $$
   separates the two points.  
   
2. **Contains constants.**  The constant function $1$ belongs to $A$.
   
3. **Lattice property.**  For any $a,b\in A$,
   $$
   \max(a,b)=\frac{a+b}{2}+\frac{|a-b|}{2},\qquad
   \min(a,b)=\frac{a+b}{2}-\frac{|a-b|}{2}.
   $$
   The absolute‑value function $|a-b|$ can be uniformly approximated on $[0,1]^2$ by polynomials (e.g., via even‑power approximants or Bernstein polynomials).  Thus the uniform closure of $A$ is closed under $\max$ and $\min$, and $A$ forms a lattice in the sense of the Stone–Weierstrass theorem.

Therefore, $A$ satisfies all algebraic and lattice requirements.  

---

# Theorem 2 — Uniform Approximation of the Accuracy Functional

**Theorem.**  Let $\tilde{\mathcal P}\in C([0,1]^2)$ be the continuous extension of the accuracy functional.  Then for every $\varepsilon>0$, there exists a finite‑order polynomial  
$$
Q(x,y)\in\mathbb R[x,y]
$$
such that
$$
\sup_{(x,y)\in[0,1]^2}
|\tilde{\mathcal P}(x,y)-Q(x,y)|
<\varepsilon.
$$
In particular,
$$
\max_{(\tilde m_a,\tilde m_g)\in\mathcal D}
|\mathcal P(\tilde m_a,\tilde m_g)-Q(\tilde m_a,\tilde m_g)|
<\varepsilon.
$$

---

# Proof of Theorem 2

**Proof.** By Lemma 1, the domain $[0,1]^2$ is compact and Hausdorff.  By Lemma 2, the polynomial algebra $A=\mathbb R[x,y]$ is a separating lattice that contains constants.

All hypotheses of the **Stone–Weierstrass theorem (lattice version)** are therefore satisfied.  Hence the uniform closure of $A$ equals $C([0,1]^2)$.

Thus, for any $\varepsilon>0$, there exists a polynomial $Q\in A$ such that  
$$
\sup_{(x,y)\in[0,1]^2}
|\tilde{\mathcal P}(x,y)-Q(x,y)|
<\varepsilon.
$$

Since $\mathcal D\subset[0,1]^2$, the same uniform bound applies on $\mathcal D$:
$$
\max_{(\tilde m_a,\tilde m_g)\in\mathcal D}
|\mathcal P(\tilde m_a,\tilde m_g)-Q(\tilde m_a,\tilde m_g)|
<\varepsilon.
$$

Therefore, the accuracy functional can be uniformly approximated over all feasible pruning configurations by a finite‑degree bivariate polynomial.  

---

# Remark

This theorem provides a rigorous mathematical justification for polynomial‑based surrogate modeling (e.g., MAP) in pruning‑ratio optimization.  
In particular, it ensures that the empirical accuracy surface can be captured with arbitrarily small uniform error.