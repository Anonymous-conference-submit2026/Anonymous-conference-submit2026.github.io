<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
# Theorem 3 — Noise from Subset-based Fast Finetuning Is Averaged Out, Ensuring MAP Provides a Consistent Approximation of the True Functional

This section establishes that although subset-based fast finetuning introduces a constant bias and random noise, such perturbations do not alter the functional structure learned by MAP. In particular, the constant bias can be absorbed into the functional definition, and the random noise vanishes in expectation as the number of sampled pruning ratios increases. Consequently, MAP remains a consistent estimator of the true accuracy functional up to an additive constant, preserving all optimality decisions.

---

# Lemma 1 — Observation Model with Constant Bias and Zero-mean Noise

For any pruning ratio pair $(\tilde m_a,\tilde m_g)$, let  $\mathcal P(\tilde m_a,\tilde m_g)$ denote the validation accuracy obtained under full finetuning.  
Subset-based fast finetuning yields an observed accuracy
$$
\widehat{\mathcal P}(\tilde m_a,\tilde m_g)
=
\mathcal P(\tilde m_a,\tilde m_g) + b + \varepsilon,
$$

where:

- $b$ is a constant bias induced by the small number of finetuning epochs,
- $\varepsilon$ is a zero-mean random noise with finite variance $\sigma^2 < \infty$.

**Lemma.**  The expectation of the observed accuracy satisfies
$$
\mathbb E[\widehat{\mathcal P}(\tilde m_a,\tilde m_g)]
=
\mathcal P(\tilde m_a,\tilde m_g) + b,
$$

i.e., fast finetuning produces a uniformly biased but unbiased-in-shape approximation of the true functional.

**Proof.**  This follows immediately from linearity of expectation and $\mathbb E[\varepsilon] = 0$.  

---

# Lemma 2 — The Bias Term Does Not Affect the Optimal Pruning Solution

Consider the optimization problems

$$
\arg\max_{(\tilde m_a,\tilde m_g)} \mathcal P(\tilde m_a,\tilde m_g)
\quad\text{and}\quad
\arg\max_{(\tilde m_a,\tilde m_g)} (\mathcal P(\tilde m_a,\tilde m_g) + b).
$$

**Lemma.**  These two problems have exactly the same set of maximizers.

**Proof.**  Since $b$ is a constant independent of $(\tilde m_a,\tilde m_g)$,
$$
\mathcal P(m_1) > \mathcal P(m_2)
\iff
\mathcal P(m_1)+b > \mathcal P(m_2)+b,
$$

hence the ordering of all candidate pruning configurations is preserved.  Therefore the maximizers coincide.  

---

# Lemma 3 — Random Noise Vanishes in Expectation as the Number of Samples Grows

Suppose MAP is trained using $N$ sampled pruning ratios, producing observations

$$
y_i
=
\mathcal P(\tilde m_a^{(i)},\tilde m_g^{(i)})
+ b
+ \varepsilon_i.
$$

MAP is optimized by minimizing the empirical squared loss

$$
\frac{1}{N}\sum_{i=1}^{N}
\bigl|y_i - \mathrm{MAP}_\Theta(\tilde m_a^{(i)},\tilde m_g^{(i)})\bigr|^2.
$$

Let $\varepsilon_i$ be i.i.d. with finite variance.

**Lemma.**  As $N\to\infty$,
$$
\frac{1}{N}\sum_{i=1}^N \varepsilon_i
\xrightarrow[]{\ \mathbb P\ }
0.
$$

Thus the contribution of random noise in the training objective vanishes asymptotically.

**Proof.**  This is a direct consequence of the Weak Law of Large Numbers applied to the zero‑mean noise sequence $\{\varepsilon_i\}$.  

---

# Theorem 3 — Consistent Approximation of the Accuracy Functional up to an Additive Constant

**Theorem.**  Under the observation model of Lemma 4, MAP trained with sufficiently many sampled pruning ratios converges (in probability) to the functional
$$
\mathcal P(\tilde m_a,\tilde m_g) + b,
$$

which differs from the true functional by only an additive constant.  Therefore, MAP preserves the identity of the optimal pruning configuration and provides a consistent approximation of the true functional up to functional equivalence.

---

## Proof of Theorem 3

By Lemma 1, each fast-finetuning observation satisfies

$$
\widehat{\mathcal P}
=
\mathcal P + b + \varepsilon.
$$

By Lemma 3, the empirical loss minimized by MAP satisfies

$$
\frac{1}{N} \sum |y_i - \mathrm{MAP}_\Theta|^2
=
\mathbb E\bigl[|\mathcal P + b - \mathrm{MAP}_\Theta|^2\bigr]
+ o_{\mathbb P}(1),
$$

where $o_{\mathbb P}(1)$ denotes terms vanishing in probability.

Hence the optimizer satisfies

$$
\mathrm{MAP}_\Theta^\star
=
\arg\min_\Theta
\ \mathbb E\bigl[
|\mathcal P + b - \mathrm{MAP}_\Theta|^2
\bigr],
$$

so MAP converges to $\mathcal P + b$.

By Lemma 2, adding a constant does not change the pruning configuration that maximizes the functional.  Thus MAP recovers the true optimal pruning solution and the correct functional shape up to a constant vertical translation.

---

# Remark

Although subset-based fast finetuning introduces a nonzero bias, this bias is constant across all pruning configurations and does not change their relative ordering. Combined with the fact that random noise is averaged out with more sampling, MAP remains a theoretically sound and statistically consistent approximation to the true functional $\mathcal P$ for the purpose of pruning‑ratio optimization.