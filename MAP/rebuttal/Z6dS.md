
# Reviewer Z6dS

**We sincerely thank the reviewer for the detailed and constructive feedback, as well as for recognizing the strengths of our work—including the comprehensive experiments, the study of activation‑layer redundancy, and the strong empirical performance.**  
Below we reorganize and refine our responses with clearer logic, direct answers, and explicit references to the MAP and theoretical materials you requested.

---

## 1. Clarification of K in Eq. 3

We apologize for the missing definition. $K$ denotes the sparsity constraint specifying the **total number of layers to be retained**, i.e., the sum of attention layers and activation‑function layers preserved after depth pruning.

---

## 2. How Observations Lead to Methodology

We thank the reviewer for pointing out that the causal relationship between our observations and method was previously insufficiently explicit. We have revised Section 3 and Section 4 to clarify that **each observation reveals a form of heterogeneity, each heterogeneity leads to a concrete challenge, and these challenges directly motivate our design principles.**

### Observation 1 → Challenge 1  
- **Observation 1: Gradient disparity.**  
  Attention and activation‑function layers exhibit markedly different gradient scales during backpropagation.

- **Challenge 1: Failure of gradient‑based importance.**  
  This cross‑type gradient heterogeneity systematically biases gradient‑based importance scores, causing attention layers to dominate and leading to suboptimal joint pruning.

### Observation 2 → Challenge 2  
- **Observation 2: Recovery asymmetry.**  
  Pruning activation layers causes large instantaneous accuracy drops but very fast recovery, while pruning attention layers causes mild drops but slow recovery.

- **Challenge 2: Failure of short‑sighted metrics.**  
  Metrics based solely on immediate post‑pruning accuracy fail to represent final recovered accuracy and therefore misestimate true layer importance.

### These two challenges yield our methodological principles  
From these challenges, we derive two central principles:

1. **Avoid cross‑type importance comparison**, preventing gradient‑induced bias.  
2. **Evaluate pruning decisions by recovered accuracy**, accounting for recovery asymmetry.

### These principles naturally lead to our two‑stage algorithm  
BoundaryDPT is designed directly from the principles above:

- **Stage 1, Step 1:** MAP predicts *recovered accuracy* under different pruning ratios, implementing Principle 2 while avoiding cross‑type comparison (Principle 1).  
- **Stage 1, Step 2:** Gradient‑based importance is applied **only within each homogeneous layer type**, ensuring Principle 1.  
- **Stage 2:** Final finetuning and layer merging enable accuracy recovery and inference acceleration.

This revision makes explicit the chain:  
**observation → challenge → methodological motivation → design principles → algorithm**,  
addressing the reviewer’s concern.

---

## 3. Improved Presentation of Part 1 and Motivation for Activation‑Layer Pruning

We appreciate the reviewer’s insightful suggestions. Below we address both the clarity of Part 1 and the motivation for pruning activation‑function layers.

### (1) Improved presentation of Part 1  
We have rewritten Part 1 to present a clearer and more structured problem formulation. The revised section emphasizes three key points:

1. **Depth pruning provides strong speedups but suffers from poor recoverability.**  
   While depth pruning yields higher speedup than width pruning, it causes severe accuracy degradation when many layers are removed.

2. **Accuracy collapse arises from ignoring layer‑type heterogeneity—not from pruning granularity.**  
   Our latency breakdown shows that both attention layers and FFN (including activation) layers contribute significantly. Pruning only one type leads to sharp accuracy drops at high pruning ratios, while *joint* pruning achieves stable accuracy–efficiency tradeoffs.

3. **Joint depth pruning is needed but is hindered by dimension mismatch.**  
   Directly pruning FFN linear layers breaks tensor shapes. We show that removing the activation function between the two linear layers resolves this mismatch, enabling seamless layer merging and feasible joint depth pruning.

### (2) Motivation for pruning activation‑function layers  
With the above context clarified, the motivation for pruning activation‑function layers becomes more explicit:

1. **Activation‑layer removal resolves the dimension‑mismatch barrier.**  
   Removing the activation function allows the two FFN linear layers to merge into one, enabling valid depth pruning without breaking tensor shapes.

2. **Activation layers contribute substantially to inference latency.**  
   FFN branches—including activations—form a major latency bottleneck; ignoring them leaves a large redundancy source unaddressed.

3. **Joint removal of attention and activation layers retains accuracy far better than single‑type pruning.**  
   Our experiments show that pruning either type alone fails catastrophically at high ratios, whereas combined pruning yields smooth, resilient accuracy–speedup curves, outperforming depth‑only and hybrid baselines.

In summary, activation‑layer pruning is a **necessary and principled component** of our method, essential for tensor‑shape compatibility, latency reduction, and stable high‑ratio depth pruning.

---

## 4. Theoretical Foundations and Details of MAP

### Theoretical Foundations of MAP

We appreciate the reviewer’s concern regarding the theoretical grounding of the polynomial MAP formulation.  
We clarify here that **MAP is not heuristic**—its validity follows directly from three formal theorems, fully documented at our theoretical appendix webpage: **[Overall Theoretical Foundation for MAP‑Based Pruning‑Ratio Optimization](https://anonconf2025.github.io/MathProof/overview.html)**  


The three foundational results are:

**Theorem 1 — Existence of an Optimum.**  
The accuracy function is defined over a **finite** domain of pruning ratios; hence a global maximum **always exists** and can be found by enumeration. MAP approximates this well‑posed target, rather than constructing a surrogate objective.

**Theorem 2 — Polynomial Approximability.**  
Accuracy varies smoothly with pruning ratios. By the **Stone–Weierstrass theorem**, any continuous function on \([0,1]^2\) can be uniformly approximated by a finite‑degree polynomial. Thus the polynomial form of MAP is a **principled universal approximator**, not an ad‑hoc fit.

**Theorem 3 — Robustness to Noisy Fast Finetuning.**  
Fast finetuning introduces a constant bias and zero‑mean noise. The MAP estimator converges (in probability) to the true accuracy surface plus the constant shift, and **the argmax remains unchanged**. Thus pruning decisions are stable under noise.

Together, these theorems establish MAP as a **consistent, theoretically rigorous tool** for pruning‑ratio optimization.

---

### More Details of MAP  
A complete description of MAP is available at:  **[MAP: Model Accuracy Predictor — Theory, Implementation, Robustness](https://anonconf2025.github.io/MAP/)** 

The webpage provides:

- full mathematical definition  
- proofs of Theorems 1–3  
- efficient data‑collection procedures  
- a complete numerical example  
- robustness analysis under injected noise  
- runtime and memory overhead  

These materials demonstrate that MAP is stable, reproducible, and easy to implement.

---

### MAP Runtime and Memory Cost  
**Fast data collection.**  
Using a 100k‑image ImageNet subset (≈1/12 of full ImageNet), fast finetuning takes 2 min/epoch on a single 910B2 NPU. MAP requires 368 epochs, totaling **≈12.2 hours**.

**MAP regression and degree selection.**  
Polynomial fitting and L2O‑CV take **<1 minute**, negligible compared to data collection.

**Redundant‑layer identification.**  
After obtaining optimal pruning ratios, the pruned model is trained for 60 epochs, requiring **≈2 hours**.

**Total runtime:** ~14 hours.  
**Peak memory usage:** ~60 GB on a single Ascend 910B2 NPU.

---

## 5. More Experiments Beyond Classification

We agree that activation pruning should be evaluated on tasks beyond classification.  
In the submission, results on:

- **ADE20K semantic segmentation** (Table 4), and  
- **CIFAR transfer learning** (Table 3)

show that BoundaryDPT maintains or improves performance across downstream tasks, indicating generality beyond ImageNet classification.

---

Once again, we sincerely thank the reviewer for the valuable feedback.  
Your comments significantly strengthened the clarity of Section 3, improved the theoretical exposition of MAP, and helped us better articulate the necessity of activation‑layer pruning.

We hope the revisions and explanations above fully address your concerns.