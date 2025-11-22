# Rebuttal Draft


# Reviewer Z6dS

**We sincerely thank the reviewer for the detailed and constructive feedback, as well as for acknowledging the strengths of our work, including comprehensive experiments, exploration of activation-layer redundancy, and strong empirical performance.**  
Below we reorganize and refine our rebuttal with clearer logic, complete answers, and direct references to the MAP and theoretical materials you requested.

---

## 1. Clarification of K in Eq. 3

We apologize for the missing definition. 
**K denotes the pruning budget**, i.e., the *number of layers to prune*. 
Under $L$ total Transformer blocks, the budget constraint 
$\tilde m_a + \tilde m_g = \frac{K}{L}$ 
defines all feasible pruning‑ratio pairs. We will explicitly include this definition in the revision.

---

## 2. How Observations Lead to Methodology

We agree this part was unclear. We have reorganized Section 3 around *two observations* and *two corresponding challenges*, each directly motivating a component of our method.

### Observation 1: Gradient Disparity  

Attention layers and activation layers exhibit **2–3 orders of magnitude gradient gap**.  
This is not caused by learnable importance parameters—it is a **structural property of ViTs**.  
As formally shown in  
**Proposition 4 — Gradient Gap Between Attention and Activation**,  
attention gradients undergo three suppressions:

- softmax Jacobian shrinkage \(O(10^{-2} \sim 10^{-3})\)  
- \(1/\sqrt d\) QK‑scaling suppression \(O(10^{-1})\)  
- multi-head averaging \(O(10^{-1})\)

while GELU paths contain none of these shrinking factors.  
Thus:
\[
\|\nabla_{\text{gelu}}\|=\Theta(1),\quad 
\|\nabla_{\text{attn}}\|=O(10^{-3}\!-\!10^{-4}).
\]

**Challenge 1:** Gradient‑based importance metrics become invalid for cross‑type ranking—attention layers always appear “small”.

### Observation 2: Recovery Asymmetry  

Pruning activation layers yields catastrophic initial accuracy drops but **very fast recovery**, while attention pruning shows mild initial drops but **slow recovery**.

**Challenge 2:** Immediate post‑pruning accuracy (or other local metrics) fails to measure true pruning importance.

---

### How These Observations Motivate Our Method

**Step 1: MAP (Model Accuracy Predictor) addresses Challenge 2**  
MAP predicts the *final* accuracy under any \((\tilde m_a, \tilde m_g)\), removing reliance on misleading instantaneous metrics.

**Step 2: Gradient‑based importance used only *within* layer type addresses Challenge 1**  
Once MAP determines separate pruning ratios for attention and activation layers, gradients are only compared *within* each type, avoiding the cross‑type gradient‑scale mismatch.

This resolves both challenges and clarifies how the observations lead to BoundaryDPT.

---

## 3. Improved Presentation of Part 1

We have rewritten Part 1 for clarity.  
The revised structure is:

1. **Why depth pruning matters**  
   Depth pruning yields significantly higher speedup than width pruning.

2. **Why depth pruning fails in prior work**  
   Not because granularity is too coarse, but because **heterogeneous components (attn/activation/linear) were not jointly pruned**.

3. **The core architectural bottleneck: dimension mismatch**  
   Removing linear layers disrupts tensor shapes; directly pruning attention + linear fails.

4. **Our key idea: prune activation layers**  
   - Removes redundancy  
   - Enables merging neighboring linear layers  
   - Eliminates dimension mismatch  
   - Allows true *joint* depth pruning

5. **Our contributions**  
   - First to reveal activation redundancy in ViT depth pruning  
   - First to identify heterogeneity phenomena (gradient disparity & recovery asymmetry)  
   - MAP-guided optimization (with full theoretical foundations)  
   - New SOTA results on depth and depth‑width pruning

This fully addresses your comments regarding structure and clarity.

---

## 4. Motivation for Activation-Layer Pruning

We emphasize activation pruning not for novelty alone, but because:

### (1) It uniquely solves **dimension mismatch**  

Removing the activation (e.g., GELU) between two linear layers allows safe merging of the two linears, enabling viable depth pruning.

### (2) It has **strong empirical redundancy**  

Activation removal causes extreme initial drops but extremely fast recovery, making them ideal candidates for structured pruning.

### (3) It unlocks **heterogeneous joint pruning**, which greatly improves accuracy‑speed tradeoff.

Thus activation pruning is both *necessary* and *highly effective* in ViT depth compression.

---

## 5. MAP Polynomial Approximation (Not Heuristic)

MAP is **not heuristic**—it is mathematically grounded.  
We provide the full theoretical justification in:

**Overall Theoretical Foundation for MAP‑Based Pruning‑Ratio Optimization**  
which contains three theorems establishing:

### Theorem 1 — Existence  

Accuracy over pruning ratios is a *finite functional*, ensuring a maximum exists and can be found via enumeration.

### Theorem 2 — Polynomial Approximability  

Accuracy varies smoothly with pruning ratios.  
By Stone–Weierstrass, any continuous surface on \([0,1]^2\) can be uniformly approximated by a finite-degree bivariate polynomial.

### Theorem 3 — Robustness to Noise  

Subset-based fast finetuning introduces only constant bias + zero-mean noise.  
MAP converges (in probability) to \(\mathcal P + b\), and the *argmax is unchanged*.

Thus MAP is a **consistent, theoretically validated surrogate**, not a heuristic.

---

## 6. More Details of MAP

We provide a complete webpage for MAP:  
**MAP: Model Accuracy Predictor — Theory, Implementation, Robustness**

This includes:

- full definition of MAP  
- derivation of regression formulation  
- pseudocode for two data-collection strategies  
- runtime overhead (<0.01 s for fitting)  
- robustness experiments  
- numeric example with DeiT‑B polynomial coefficients
### 8. MAP Runtime and Memory Cost

As summarized on the MAP webpage:

- **Data collection:** ~7 hours (fast finetuning)  
- **MAP regression:** <0.01 s (6 parameters, ~20 points)  
- **Memory:** negligible (<1 MB for all samples)

   Thus MAP adds *minimal overhead* compared to standard iterative finetuning.
---

## 7. More Experiments Beyond Classification

We agree that activation pruning should be tested beyond classification.  
In the submission:

- **ADE20K segmentation** (Table 4)  
- **CIFAR transfer** (Table 3)

show that BoundaryDPT maintains or improves performance on downstream tasks, indicating generality beyond ImageNet classification.

Due to compute limits, we could not include COCO detection, but BoundaryDPT is orthogonal to task heads, and we expect consistent behavior for detection as well.

---


Finally, we thank the reviewer again for the constructive feedback and for recognizing the motivation and contributions of our work.  
Your comments helped us significantly improve the clarity of Section 3, strengthen the theoretical justification for MAP, and better articulate the necessity of activation-layer pruning.

We hope the revisions and explanations above address all concerns and improve the overall quality of the paper.




# Response to Reviewer 4wzp (Revised & Polished)

**We sincerely thank the reviewer for the thoughtful questions and for the willingness to reconsider the score.**  
Below we provide concise, direct, and technically grounded answers, with references to MAP, Proposition 4, and Proposition 5 where appropriate.

---

### **Q1. Does MAP need to be retrained for different architectures (e.g., Swin, ViT‑Large) or datasets? What is the prediction error if transferred directly?**

Thank you for this excellent question.

- **Within the same family of architectures (e.g., DeiT models), MAP does *not* need to be refitted.**  
  It still correctly predicts pruning *ratios*, although the exact pruned layers must be re‑identified.

- **Across different architectures (e.g., transferring MAP from DeiT to Swin), refitting is required.**  
  Fortunately, the computational overhead is small: MAP can be reconstructed **within 14 hours on a single 910B2 NPU** (details in *MAP s6*).

- **Direct transfer without refitting leads to noticeable prediction error**, often corresponding to **≈2 layers mismatch**, which is expected because different architectures have distinct pruning‑sensitivity landscapes.

We will include this clarification in the revision.

---

### **Q2. If MAP has ±0.5% prediction noise, will the final pruning decision change? Has the paper evaluated MAP’s sensitivity?**

Yes.  
We conducted a dedicated robustness experiment (see *MAP §5*), injecting clipped Gaussian noise into all measured accuracies and running 500 Monte‑Carlo trials.

Key findings:

- **75.0%** of trials reproduced the *exact* pruning decision.  
- **5.2%** deviated by only **one layer** (±1/12).  
- **19.8%** deviated by more than one layer.  
- A one‑layer deviation leads to final accuracy changes within **≤0.3%**.

Thus **>80%** of noisy trials stay within a one‑layer deviation, demonstrating that MAP’s pruning‑ratio decision is **highly robust**.

---

### **Q3. How do you ensure that “activation pruning + linear‑layer fusion” does not break residual connections or LayerNorm?**

Our pruning only affects:

- **attention layers**, and  
- **activation (GELU) layers**.

Linear‑layer fusion is performed **only inside the FFN block**, where:
x + FFN(x)
FFN = W2(GELU(W1 x))
Residual connections and LayerNorm are **outside** the fused region. Therefore:

- Residual pathway is untouched  
- LayerNorm statistics remain unchanged  
- Fusion is algebraically valid: `(W2 · W1_eff)` is equivalent to merging two linear layers with bias

This is consistent with **vit_latency.svg**, showing that attention + FFN dominate runtime.

---

### **Q4. Does layer fusion still work for SwiGLU or post‑Norm Transformers?**

Yes, with minor considerations:

1. **SwiGLU:**  
   Direct fusion is not possible due to Hadamard interactions.  
   However, SwiGLU can be reparameterized into “Linear → activation → Linear”, as shown in *ReLU Strikes Back*, after which our fusion applies.

2. **Post‑Norm ViTs:**  
   Fusion remains fully compatible, since it does **not** involve LayerNorm at all.

---

### **Q5. Does recovery asymmetry persist under different seeds or longer finetuning (e.g., 50 epochs)?**

Yes.  
We tested on CIFAR‑100 (seeds and finetuning epochs = 10 and 50). Across all layers:

- **Activation pruning**: large initial drop, **very fast recovery**  
- **Attention pruning**: small initial drop, **slower recovery**

The asymmetry remains consistent across seeds and longer training schedules.

---

### **Q6. What fundamentally causes this recovery asymmetry? Is it purely gradient magnitude or also information flow?**

The full theoretical explanation is in **Proposition 5**.

- The activation branch contributes a **dominant portion** of the residual output → large initial drop  
- Activation gradients are **1–2 orders larger** than attention gradients (from **Proposition 4**) → **10–100× faster recovery**

Thus both **representational dominance** and **gradient scale** jointly cause the asymmetry.

---

### **Q7. Can BoundaryDPT generalize to downstream tasks (COCO, ADE20K, robustness benchmarks)?**

Yes.  
BoundaryDPT modifies only the ViT backbone, making it task‑agnostic.

We evaluated the method on:

- **ADE20K segmentation (Table 4)**  
- **CIFAR transfer (Table 3)**  

Both show comparable or improved accuracy–speed tradeoff relative to baselines.  
We expect similar behavior on COCO; compute limitations prevented us from including it.

---

### **Q8. Is the orthogonality to token pruning universal across token methods and pruning rates?**

Yes.  
We conducted supplementary experiments:

- BoundaryDPT + **ToMe** (`tome_dpt.pdf`)  
- BoundaryDPT + **GTP‑ViT** at different pruning ratios (`gtp-dpt-pruning-8layers.pdf`)

Both confirm that:

- Orthogonality holds  
- Performance improvements remain consistent across token strategies and pruning rates

---

### **Q9. Were all throughput results re-measured fairly on the same device (H800)?**

Yes.  
All baselines were re‑measured on the **same NVIDIA H800 GPU** to ensure a fair comparison.

---

### **Q10. How many samples does MAP require, and what is the actual compute cost? Can it be reproduced under limited resources?**

From *MAP §6*:

- **Dataset:** 100k ImageNet subset (~1/12)  
- **Fast finetuning:** 2 minutes/epoch on Ascend 910B2  
- **MAP requires 368 epochs → ~12.2 hours**  
- **MAP fitting + CV:** <1 minute  
- **Redundant‑layer identification:** ~2 hours  

**Total cost: ~14 hours on a single NPU**, fully reproducible under modest compute.

---

# Final Remark

We sincerely appreciate the reviewer’s thorough questions and constructive feedback.  
Your comments helped us significantly strengthen the explanations of MAP robustness, fusion safety, recovery asymmetry, and cross‑task generalization.

We hope this improved response adequately addresses all concerns.