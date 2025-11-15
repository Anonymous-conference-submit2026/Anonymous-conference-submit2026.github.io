# Overall Theoretical Foundation for MAP‑Based Pruning‑Ratio Optimization

The three theorems collectively establish a complete mathematical foundation for our pruning‑ratio optimization framework.  The logic proceeds in three progressive steps, corresponding to Theorem 1–3.

---

## 1. Existence of a Discrete Accuracy Functional and Exhaustive Optimization (Theorem 1)

We first formalize the relationship between pruning ratios and model accuracy as a functional  
$$
\mathcal P:\mathcal D\to\mathbb R,
$$
whose domain  
$$
\mathcal D=\{0,\tfrac1L,\dots,1\}^2
$$
contains a finite number of admissible pruning configurations.  Under a fixed pruning budget, the feasible subset remains finite, and thus $\mathcal P$ must attain its maximum on this domain.  Therefore, the pruning‑ratio optimization problem is **well‑posed**, and the optimal pruning configuration **always exists** and can be found by enumeration. The full proof is provided in [Theorem 1](./prof1.md)

---

## 2. Continuous Relaxation and Polynomial Approximation (Theorem 2)

Although $\mathcal D$ is discrete, empirical and theoretical considerations suggest that accuracy varies smoothly with pruning ratios.  Thus we relax the domain to the continuous square  
$$
(\tilde m_a,\tilde m_g)\in[0,1]^2,
$$
and assume $\mathcal P$ admits a continuous extension.

Applying the Stone–Weierstrass theorem, any continuous functional on $[0,1]^2$ can be **uniformly approximated** by a finite‑degree bivariate polynomial:
$$
Q(x,y)\in \mathbb R[x,y].
$$

This establishes that MAP—implemented as a polynomial‑like parametric model—can in principle approximate the accuracy surface arbitrarily well. The full proof is provided in [Theorem 2](./proof2.md).

---

## 3. Robustness of MAP Under Noisy Subset-based Fast Finetuning (Theorem 3)

The accuracies used to train MAP are obtained via subset-based fast finetuning and satisfy  
$$
\widehat{\mathcal P}=\mathcal P + b + \varepsilon,
$$
where:

- $b$ is a constant bias (few finetuning epochs),
- $\varepsilon$ is zero‑mean noise with finite variance.

We show that:

1. **The constant bias does not affect the optimal pruning decision**, since adding a constant does not change an argmax.  
2. **The random noise vanishes as sample size grows**, by the Weak Law of Large Numbers.

Thus MAP converges (in probability) to  
$$
\mathcal P + b,
$$
which is a harmless vertical shift of the true functional.

Therefore, MAP remains a statistically consistent surrogate for the accuracy functional, even when trained from noisy fast‑finetuning data. The full proof is provided in [Theorem 3](./proof3.md).

---

# Conclusion

Together, **Theorem 1–3** justify our method:

1. The accuracy surface over pruning ratios defines a finite-domain functional with guaranteed optimal solutions.  
2. After a natural continuous relaxation, this functional is uniformly approximable by finite‑degree polynomials.  
3. Subset-based noisy estimates still allow MAP to consistently approximate the true functional and recover optimal pruning choices.

These theoretical results validate MAP as a reliable and efficient surrogate for pruning‑ratio optimization.

<!-- 补充， prof4解释了gradient disparity，prof5解释了recovery  Asymmetry会出现-->
# Additional Notes
Here are two supplementary explanations related to gradient disparity and recovery asymmetry, detailed in [Proof 4](./proof4.md) and [Proof 5](./proof5.md).