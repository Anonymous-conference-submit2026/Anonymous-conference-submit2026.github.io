# Rebuttal Draft


# Reviewer Z6dS

**We sincerely thank the reviewer for the detailed and constructive feedback, as well as for acknowledging the strengths of our work, including comprehensive experiments, exploration of activation-layer redundancy, and strong empirical performance.**  
Below we reorganize and refine our rebuttal with clearer logic, complete answers, and direct references to the MAP and theoretical materials you requested.

---

## 1. Clarification of K in Eq. 3

We apologize for the missing definition. 
$K$ is the sparsity constraint that determines the total number of layers, i.e., the sum of attention layers and activation function layers, to be retained after depth pruning.

---

## 2. How Observations Lead to Methodology

We thank the reviewer for pointing out the unclear connection between our observations and the motivation of our method. We revise Section 3 and Section 4 to make the causal chain explicit: **each observation reveals a type of heterogeneity, each heterogeneity induces a challenge, and these challenges directly motivate our design principles.**

### We have the observation 1 Gradient disparity and it leads to the challenge 1 Failure of gradient-based importance.  
- **Observation 1: Gradient disparity.**  Attention and activation‑function layers exhibit substantially different gradient scales during backpropagation.

- **Challenge 1: Failure of gradient‑based importance.**  This gradient heterogeneity biases cross‑type importance estimation, causing attention layers to dominate and leading to suboptimal joint pruning.

### We have the observation 2 Recovery asymmetry and it leads to the challenge 2 Failure of short-sighted metrics.  
- **Observation 2: Recovery asymmetry.**  Pruning activation layers causes severe immediate accuracy drops but very fast recovery, whereas attention‑layer pruning causes mild drops but slow recovery.

- **Challenge 2: Failure of short‑sighted metrics.**  Metrics based only on immediate post‑pruning accuracy fail to reflect the final recovered accuracy and thus misjudge true layer importance.

### Based on these two challenges, we derive the methodological motivations for our approach.  
These challenges show that heterogeneity exists in both **training dynamics** and **recovery behaviors**, motivating two key principles:

1. **Avoid cross‑type importance comparison** (to eliminate gradient‑induced bias).  
2. **Evaluate pruning decisions using final recovered accuracy** (to account for recovery asymmetry).

### Based on these design principles, we develop our two‑stage algorithm as follows.  
BoundaryDPT is aligned with these principles:

- **Stage 1, Step 1:** A model‑accuracy predictor estimates *recovered accuracy* under different pruning ratios of the two layer types, implementing Principle 2 while avoiding cross‑type comparison (Principle 1).  
- **Stage 1, Step 2:** Gradient‑based importance is applied **within each homogeneous group only**, fully satisfying Principle 1.  
- **Stage 2:** Fine‑tuning and layer merging enable accuracy recovery and inference acceleration.

   
This revision clearly shows how **observation → challenge → methodological motivation → design principle → algorithm** are connected, addressing the reviewer’s concern.

---

## 3. Improved Presentation of Part 1 and Motivation for Activation-Layer Pruning

We have rewritten Part 1 for clarity.  
The revised structure is:

xxx

and based on this, we clarify the motivation for activation-layer pruning as follows.

xxx

reference: \section{Introduction}
Vision Transformers (ViTs) \citep{50650,liu2021Swin,touvron2021going,han2021transformer,chen2021crossvit,li2022efficientformer,wang2022pvt} have demonstrated remarkable performance across various domains.  However, their large parameter counts and high computational costs lead to extended inference latency. Structured pruning \citep{he2023structured} is effective for model compression while maintaining hardware compatibility.


\textcolor{blue}{
\textbf{Depth pruning and its limitation.} As a kind of structured prunign, depth pruning denotes removing entire layers from ViTs. 
Compared to width pruning which only reduces channels or attention heads inside a layer, depth pruning delivers significantly higher speedups under equivalent sparsity budgets, as shown in Figure \ref{fig:combined_dp_latency_analysis} (a). 
However, depth pruning typically incurs substantial accuracy degradation, especially with \textbf{aggressive layer removal}. 
Consequently, comprehensive methods that integrate both width and depth pruning are also limited by the depth pruning challenges, resulting in suboptimal performance. 
}
% As shown in \textbf{Figure xx(a)}, under the same parameter pruning raito, depth pruning incurs significantly higher speedups than width pruning, which reduces channels or attention heads. However, it is generally recognized that depth pruning is notoriously difficult for accuracy recovery \textbf{especially with aggressive layer removal} since its pruning granularity is relatively larger than that of width pruning. Consequently, comprehensive methods that integrate both width and depth pruning are also limited by these depth pruning challenges, resulting in suboptimal performance. 

\textcolor{blue}{
\textbf{Joint depth pruning matters.} While prior research attributes the accuracy collapse in depth pruning to coarse granularity \citep{he2023structured,mao2017exploring}, we challenge this perspective. 
Our analysis reveals that the true bottleneck lies in the neglect of joint pruning of different layers in a ViT by considering cross-layer heterogeneity. 
As shown in Figure \ref{fig:tradeoff}, individually pruning attention layers or activation function layers leads to drastic accuracy drops at high pruning ratios. In contrast, joint pruning of these two types of layers with our method significantly enhances accuracy retention while maintaining efficiency.
}
% as the increase of pruning ratios, the accuracy of DeiT-base drops drastically in case only pruning attention layers or only activation layers. In contrast, joint pruning of this two types of layers with our proposed method significantly enhance the accuracy of ViTs under the same pruning ratio as pruning alone and incurs a more steady accuracy change.    



% \begin{figure*}[h]
% \centering
% \includegraphics[width=0.65\textwidth]{AnonymousSubmission/LaTeX/figure/1_intro/depth_vs_width_pruning.pdf}
% \caption{Speedup Comparsion: Depth Pruning vs Width Pruning}
% \label{fig:depth_vs_width}
% \end{figure*}

\begin{figure*}[h]
\centering
\includegraphics[width=0.95\textwidth]{AnonymousSubmission/LaTeX/figure/1_intro/combined_dp_latency_analysis.pdf}
\caption{ 
\textcolor{blue}{
Practical inference speed analysis of ViTs.  (a) Speedup comparsion: Depth pruning exhibits significantly higher speedup efficiency than width pruning.
(b) Latency breakdown of ViTs.
}
}
\label{fig:combined_dp_latency_analysis}
\end{figure*}

\textcolor{blue}{
\textbf{Dimension mismatch hinders joint depth pruning.} As shown in Figure \ref{fig:combined_dp_latency_analysis} (b), the two most types of time-consuming layers in ViTs are attention layers and linear layers, which together account for over 50\% of total inference time. 
While joint depth pruning of linear layers and attention layers is necessary, direct simultaneous removal may create \textit{dimension mismatch}. 
As illustrated in \textbf{Figure \ref{fig:dim_mismatch}}, if the first linear layer of a feedforward network (FFN) block in ViTs is removed, the output tensor from the previous attention layer cannot be passed through the second linear layer in the FFN. Similarly, pruned second linear layers prevent the output tensor from passing through the subsequent attention layers. In a word, dimension mismatch renders jointly depth-pruned ViTs unworkable..         
}

\textbf{Our contribution.} To address the conundrums of accuracy recovery and dimension mismatch in depth pruning, our contributions are threefold: 



% \begin{figure*}[h]
% \centering
% \includegraphics[width=0.75\textwidth]{AnonymousSubmission/LaTeX/figure/1_intro/vit_latency.pdf}
% \caption{The latency of different componets in ViT}
% \label{fig:latency_of_vit}
% \end{figure*}

\begin{figure*}[h]
\centering
\includegraphics[width=0.75\textwidth]{AnonymousSubmission/LaTeX/figure/1_intro/dim_mismatch.pdf}
\caption{\textcolor{blue}{The visualization of dimensions mismatch.}}
\label{fig:dim_mismatch}
\end{figure*}
\begin{itemize}
\item  \textcolor{blue}{  \textbf{Joint depth pruning of attention and activation function layers is proposed.} }  
In particular, we tackle dimension mismatch by removing activation function layers situated between two linear layers, which allows for the natural merging of those linear layers to reduce model depth while aligning the dimensions of attention layers.
% We empirically prove that joint pruning of attention layers and activation function layer \textbf{constitutes a state-of-the-art strategy to enhance depth pruning performance}, as shown in Figure~\ref{fig:pareto}.
Besides, to the best of our knowledge, we are the \textbf{first} to identify and mitigate the redundancy of the activation function layers during joint pruning in ViTs.  

\item  \textcolor{blue}{ \textbf{ The heterogeneity in joint depth pruning is revealed and addressed. } } We identify two unique phenomena related to the heterogeneity in joint depth pruning: gradient disparity and recovery asymmetry. Such heterogeneity has never been examined in the literature. In light of this, we introduce BoundaryDPT, a two-stage method featuring a model accuracy predictor to manage heterogeneity.

\item \textcolor{blue}{ \textbf{ Two key state-of-the-art records are established. } } With BoundaryDPT, the depth-pruned DeiT-base achieves up to 1.6x speedup while maintaining lossless accuracy, which is the state-of-the-art among depth pruning works. More importantly, building on BoundaryDPT, we further present BoundaryDPT+, a depth-width pruning pipeline that \textbf{establishes a new state-of-the-art benchmark for extreme ViT compression}, as demonstrated in Figure \ref{fig:pareto}. BoundaryDPT+ enhances the ViT inference speedup from 4.60x to 5.44x for the Isomorphic-Pruning-2.6G configuration while achieving near-lossless accuracy.   
\end{itemize}
\begin{figure*}[h]
\centering
\includegraphics[width=1.0\textwidth]{AnonymousSubmission/LaTeX/figure/1_intro/overview.pdf}
\caption{
The complete methodological framework of BoundaryDPT and BoundaryDPT+, involving the research goals, key insights, contributions, and results.
}
% \caption{Overview of BoundaryDPT and BoundaryDPT+, a two-stage methodology for joint pruning, which is to fully unleash the potential of the depth pruning for an enhanced accuracy-speedup Pareto frontier}
\label{fig:overview}
\end{figure*}


---

## 4. 

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