<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
# Theorem 1 — Well‑Posedness of the Pruning‑Ratio Optimization Problem  
This section gives a formal mathematical foundation for the pruning‑ratio optimization problem introduced in *Stage 1: Identification of Redundant Layers*. 
We rigorously characterize the feasible solution space and establish the existence of a globally optimal pruning configuration.

---

## Preliminaries

A Vision Transformer (ViT) contains  
- an attention layer and  
- an activation layer  

at each of the $L$ depth‑ordered positions.

Pruning decisions are encoded by binary vectors
$$
\hat m_a,\,\hat m_g \in \{0,1\}^L,
$$
where entry $1$ indicates that the corresponding layer is removed and entry $0$ indicates it is preserved.

The pruning ratios are defined by
$$
\tilde m_a = \frac{\|\hat m_a\|_0}{L},
\qquad
\tilde m_g = \frac{\|\hat m_g\|_0}{L}.
$$

Since the numerators range from $0$ to $L$,  
$$
\tilde m_a,\tilde m_g \in \Bigl\{0,\tfrac1L,\dots,1\Bigr\}.
$$

Define the discrete pruning‑ratio domain
$$
\mathcal D=\{(i/L,j/L):0\le i,j\le L\},
\qquad |\mathcal D|=(L+1)^2.
$$

---

# Lemma 1 — Finiteness of the Feasible Domain

**Lemma.**  The pruning‑ratio domain $\mathcal D$ is a finite set.

**Proof.**  By construction,
$$
\mathcal D=\{(i/L,j/L):i,j\in\{0,1,\dots,L\}\},
$$
which contains exactly $(L+1)^2$ elements.  Hence $\mathcal D$ is finite.  

---

# Lemma 2 — Well‑Defined Accuracy Functional on the Discrete Domain

**Lemma.**  The validation accuracy produced by the *pruning → finetuning → evaluation* pipeline defines a function
$$
\mathcal P:\mathcal D\to\mathbb R,
$$
which is well‑defined and computable on every point of $\mathcal D$.

**Proof.**  Each $(\tilde m_a,\tilde m_g)\in\mathcal D$ corresponds to a realizable pruning configuration because both ratios derive from binary vectors in $\{0,1\}^L$.  Running the pipeline yields a single deterministic validation accuracy, which is a real number.  Thus $\mathcal P$ is well‑defined on $\mathcal D$ and computable.  

---

# Lemma 3 — Finiteness of the Constrained Feasible Set

A pruning budget $k$ is imposed by the linear constraint
$$
\tilde m_a+\tilde m_g=\frac{k}{L}.
$$

Define the constrained feasible set:
$$
\mathcal D_k
=
\Bigl\{(\tilde m_a,\tilde m_g)\in\mathcal D:
\tilde m_a+\tilde m_g=\tfrac{k}{L}\Bigr\}.
$$

**Lemma.**  $\mathcal D_k$ is a finite set with  
$$
|\mathcal D_k| \le L+1.
$$

**Proof.**  The equality constraint uniquely determines $j = k-i$ for each $i\in\{0,\dots,L\}$, resulting in at most $L+1$ feasible pairs.  Thus $\mathcal D_k$ is finite.  

---

# Theorem 1 — Existence of an Optimal Pruning Configuration

**Theorem.**  For any pruning budget $k$, the optimization problem
$$
(\tilde m_a^\star,\tilde m_g^\star)
=
\underset{(\tilde m_a,\tilde m_g)\in\mathcal D_k}{\arg\max}\,
\mathcal P(\tilde m_a,\tilde m_g)
$$
admits at least one optimal solution.

---

## Proof of Theorem 1

**Proof.**  By Lemma 3, $\mathcal D_k$ is a finite set.  By Lemma 2, $\mathcal P:\mathcal D_k\to\mathbb R$ is a real‑valued function defined on this finite domain.

A real‑valued function on a finite set always attains its maximum; therefore,
$$
\exists\,(\tilde m_a^\star,\tilde m_g^\star)\in\mathcal D_k
\quad\text{s.t.}\quad
\mathcal P(\tilde m_a^\star,\tilde m_g^\star)
=
\max_{(\tilde m_a,\tilde m_g)\in\mathcal D_k}
\mathcal P(\tilde m_a,\tilde m_g).
$$

Thus the pruning‑ratio optimization problem is well‑posed and always has an optimal solution.  

---

# Remark

Since $|\mathcal D_k|\le L+1$, the feasible domain is trivially enumerable.  Thus the maximizer of $\mathcal P$ can be obtained via direct search once the MAP (Model Accuracy Predictor) or empirical evaluation pipeline provides $\mathcal P(\cdot)$ values.

This establishes the mathematical validity of the optimization step used in *Stage 1* of the method.