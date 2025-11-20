# Rebuttal Draft

# Reviewer Dp35

**We sincerely thank the reviewer for the positive assessment on soundness, presentation, contribution, and motivation.**  
Your comments helped us clarify the cost comparison and strengthen the explanation of gradient disparity.

---

## 1. Comparison of Training Cost

Thank you for raising the concern regarding pruning cost.

For depth pruning, the main baseline **NOSE** also adopts **iterative pruning with 400‑epoch finetuning**, which is identical to our training schedule.  
Therefore, our method does **not** introduce extra training cost beyond established depth‑pruning pipelines.  

We will additionally include a table comparing total finetuning costs across all baselines for completeness.

---

## 2. Why Gradient Disparity Appears (and why it is structural rather than an artifact)

Thank you for pointing out the need to clarify the origin of this disparity.

### 2.1 The disparity comes from ViT’s architecture, not from the learnable importance parameters

As shown in our formal statement  
**[Proposition 4 — Gradient Gap Between Attention and Activation](https://anonconf2025.github.io/MathProof/prof4.html)**, 
the ViT architecture inherently causes **2–3 orders of magnitude suppression** of attention gradients due to:

- softmax Jacobian shrinking,  
- QK scaling by \($1/\sqrt d$\),  
- multi‑head averaging.

In contrast, GELU paths contain **no such shrinking factors**, resulting in  
$$
\|\nabla_{\text{gelu}}\| = \Theta(1),\quad
\|\nabla_{\text{attn}}\| = O(10^{-3} \text{–} 10^{-4}).
$$


Thus, the gradient disparity is **structural and universal** to standard ViT blocks.

### 2.2 Why it becomes observable when adding importance parameters

In native ViT, gradients are mixed through residual paths, making such disparity hard to measure directly.  
When introducing learnable importance parameters, gradients toward different components become **explicitly comparable**, making the inherent disparity *visible* rather than *created* by the method.

We will make this explanation more explicit in the revised version.

---

## Final Acknowledgement

We sincerely appreciate the reviewer’s positive comments on the writing quality and motivation.  
Your feedback helped us significantly improve the clarity of our explanation regarding training cost and the structural nature of gradient disparity.



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

---

## 7. More Experiments Beyond Classification

We agree that activation pruning should be tested beyond classification.  
In the submission:

- **ADE20K segmentation** (Table 4)  
- **CIFAR transfer** (Table 3)

show that BoundaryDPT maintains or improves performance on downstream tasks, indicating generality beyond ImageNet classification.

Due to compute limits, we could not include COCO detection, but BoundaryDPT is orthogonal to task heads, and we expect consistent behavior for detection as well.

---

## 8. MAP Runtime and Memory Cost

As summarized on the MAP webpage:

- **Data collection:** ~7 hours (fast finetuning)  
- **MAP regression:** <0.01 s (6 parameters, ~20 points)  
- **Memory:** negligible (<1 MB for all samples)

Thus MAP adds *minimal overhead* compared to standard iterative finetuning.

---

# Final Acknowledgement

We thank the reviewer again for the constructive feedback and for recognizing the motivation and contributions of our work.  
Your comments helped us significantly improve the clarity of Section 3, strengthen the theoretical justification for MAP, and better articulate the necessity of activation-layer pruning.

We hope the revisions and explanations above address all concerns and improve the overall quality of the paper.





# Reviewer 4wzp

**We sincerely thank the reviewer for the detailed questions and for the willingness to reconsider the score.**  
Below we address each question concisely and directly, while providing technical grounding drawn from the MAP webpage and the theoretical foundation page.

---

**Q1：** 论文使用一个二次多项式作为模型精度预测器（MAP）。请问该 MAP 是否需要针对不同架构（如 Swin、ViT-Large）或数据集重新拟合？若直接迁移使用，预测误差会有多大？

感谢您的提问，这是一个很好的问题，根据我们的观察，同一种类型的模型(例如Deit系列的模型)，MAP是不需要重新拟合的(但是依然只能确定比例，具体的剪枝位置需要重新确定)，针对不同的架构是需要重新拟合的，但是重新拟合的overhead很小，通常我们只需要一张NPU在14h以内便可以重新拟合了。若直接迁移，预测误差会比较大，例如直接使用Deit拟合得到的MAP去拟合Swin，大概率会有2个层的预测误差。



**Q2：** 如果 MAP 存在偏差（例如 Top-1 误差 ±0.5%），最终的剪枝比例决策是否会被完全改变？论文是否评估过 MAP 对剪枝策略的敏感性？

感谢您的提问，我们补充进行了MAP的鲁棒性实验，具体的实验细节请参考[Robutness Exp of MAP](https://anonconf2025.github.io/MAP/#sec5),通过加噪声并进行蒙特卡洛模拟，我们发现，75%的情况下MAP预测结果不改变，5.2%预测结果偏离一层，剩余的19.8%预测结果偏离1层以上，而结果偏离一层对于最终剪枝模型的精度在0.3%以内，我认为我们的MAP在80%以上的情况不会偏离太多，这证明了我们的MAP结果是比较鲁棒的。


**Q3：** 在执行“剪枝激活层 + 合并线性层”时，如何确保残差连接与 LayerNorm 不会破坏模型的稳定性？是否存在严格的数学条件保证该融合操作无害或近似无害？

感谢您的提问，我们进行剪枝的时候，只针对Attention层和activation层进行操作，而合并线性层的过程只在FFN层内进行操作，而残差连接与 LayerNorm 都在这两个层以外的，因此我们的剪枝并不会影响这两个操作，因此不会破坏模型的稳定性,并且正如[ViT latency](https://anonconf2025.github.io/fig/vit_latency.svg)所展示的，attention和linear占据了80%以上的计算时间，
因此我们剪枝只针对这两种层进行操作是合理的。同时，只针对FFN进行层合并数学上是等价的，即两个矩阵相乘可以等效一个单矩阵相乘(考虑bias同样也成立)


**Q4：** 若应用于带门控机制（如 SwiGLU）或 Post-Norm 的 ViT 变体，是否仍可直接进行层融合？论文是否验证过这种情况下的有效性？

感谢您的提问，这是一个很好的回答
(1) 针对SwiGLU我们确实没有办法进行直接的等价合并(因为有哈达玛积的存在)，但是通常我们可以将SwiGLU转化为Linear+激活+Linear的形式再进行层融合，前者的转换在相关工作(Relu strikes back)中已经进行了探索，因此我们认为我们的工作是仍然有效的
(2) 正如在Q3提到的，我们的层融合是不涉及Layer-Norm的，因此针对Post-Norm的ViT我们仍然是可以进行层融合的，这种情况下仍然是有效的


**Q5：** 论文指出，裁剪激活层会导致初始损失较大但恢复更快，而裁剪注意力层则相反。该现象在不同随机种子或更长训练周期下是否依然存在？若微调周期延长至 50 个 epoch，这种“恢复不对称性”是否仍明显？
在更长训练周期下仍然存在，我们在cifar100上进行了测试，数据如下：
    full_acc	86.80%										
seed=42,epoch=10												
    layer 0	layer 1	layer 2	layer 3	layer 4	layer 5	layer 6	layer 7	layer 8	layer 9	layer 10	layer 11
act	1.00%	1.00%	1.00%	1.00%	1.00%	1.00%	1.31%	1.03%	0.00%	1.00%	1.93%	4.51%
act-ft	83.97%	84.82%	84.87%	83.69%	84.18%	84.99%	85.00%	85.67%	85.11%	86.17%	86.73%	86.85%
attn	85.29%	85.23%	85.71%	85.21%	85.31%	85.64%	85.95%	85.82%	85.68%	85.55%	86.10%	86.33%
attn-ft	86.40%	86.20%	86.10%	86.10%	85.80%	86.10%	86.20%	86.47%	85.90%	86.03%	86.37%	86.35%
                                                
seed=42,epoch=50												
    layer 0	layer 1	layer 2	layer 3	layer 4	layer 5	layer 6	layer 7	layer 8	layer 9	layer 10	layer 11
act	1.00%	1.00%	1.00%	1.00%	1.00%	1.00%	1.31%	1.03%	0.00%	1.00%	1.93%	4.51%
act-ft	85.92%	85.72%	85.70%	86.14%	85.85%	86.16%	86.17%	86.58%	86.92%	86.92%	87.04%	86.83%
attn	85.29%	85.23%	85.71%	85.21%	85.31%	85.64%	85.95%	85.82%	85.68%	85.55%	86.10%	86.33%
attn-ft	86.72%	86.54%	86.46%	86.60%	86.49%	86.39%	86.53%	86.60%	86.64%	86.25%	86.85%	86.89%

xxxxxx



**Q6：** 导致这种恢复差异的根本原因是什么？是仅由梯度幅值差异造成，还是与信息流路径或残差分布相关？论文未提供理论上的解释。

具体的解释可以参考我们的数学证明：https://anonconf2025.github.io/MathProof/prof5.html




**Q7：** BoundaryDPT 仅在 ImageNet 分类上评估。若应用于下游任务（如 COCO 检测、ADE20K 分割或 ImageNet-A 鲁棒性测试），是否仍能保持相似的精度–加速折中性能？

我们将我们的方法已经应用到下游任务中了，Table 4展示了我们在ADE20K上的实验结果，实验证明对比baseline，我们仍然保持相似的精度–加速折中性能



**Q8：** “BoundaryDPT 与 token 剪枝严格正交”的结论仅基于单一 token 方法（GTP-ViT）与单一剪枝率。若更换 token 策略或提高剪枝率，该正交性是否依然成立？

感谢您的提问，我们补充了我们方法与ToMe结合的效果，具体实验结果如tome_dpt.pdf所示，实验证明了此时正交性仍然成立，同时我们也改变了剪枝率进行了测试,具体实验结果如gtp-dpt-pruning-8layers.pdf所示，实验证明更换了剪枝率，正交性仍然成立。 以上的实验结果说明，我们的方法与token剪枝正交



**Q9：** 论文在 Ascend 910B NPU 上训练，却在 NVIDIA H800 GPU 上测量推理吞吐量。所有基线结果是否都重新测量过？若没有，该吞吐量比较是否公平？

感谢您的提问，我们所有的基线都在H800重新测量过了



**Q10：** MAP 的构建需要多轮“剪枝–微调–评估”的采样，但论文未说明具体样本数量或计算开销。在实际环境中训练 MAP 需要多少 GPU 天？是否能在有限算力条件下复现？

https://anonconf2025.github.io/MAP/#sec6 我们补充了样本数量(10w张样本)和runtime overhead的信息，在1张NPU上去构建MAP需要14h，我们认为这是可以在有限算力条件下复现的


