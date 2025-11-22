# Rebuttal Draft


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