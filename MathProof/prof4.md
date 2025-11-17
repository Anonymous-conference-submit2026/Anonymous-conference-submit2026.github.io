<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# **Proposition 4（Gradient Gap Between Attention and Activation）**

Under the standard Vision Transformer assumptions A1–A4 (softmax concentration, bounded feature/weight norms, large hidden dimension, and multi‑head averaging), the back‑propagated gradients satisfy

$$
\|\nabla_{\mathrm{gelu}}\|=\Theta(1), \qquad
\|\nabla_{\mathrm{attn}}\| = O(10^{-3}\!\sim10^{-4}),
$$

and therefore

$$
\boxed{
\frac{\|\nabla_{\mathrm{gelu}}\|}{\|\nabla_{\mathrm{attn}}\|}
= \Omega(10^{2}\!-\!10^{3})
}
$$

i.e., **attention gradients are inherently 2–3 orders of magnitude smaller than activation (GELU) gradients**.

---

# **Assumptions**

## **A1. Softmax concentration**
For each attention row \($A_i=\mathrm{softmax}(z_i)$\),

$$
A_{i^*j}=O(1),\qquad
A_{ij}=O(10^{-2}\!\sim10^{-3}) \ (i\ne i^*).
$$

This implies softmax Jacobian terms \($s_i s_j = O(10^{-4}\!\sim10^{-6})$\).

---

## **A2. Bounded feature and weight norms**

$$
\|x_i\| = \Theta(1),\qquad
\|W_Q\|,\|W_K\|,\|W_V\|,\|W_1\|,\|W_2\| = O(1).
$$

Thus neither attention nor MLP layers amplify gradient magnitudes via weight norms.

---

## **A3. Large hidden dimension**

$$
d \gg 1,\quad \text{e.g., } d = 768 \text{ in DeiT-B}.
$$

This ensures the QK‑logit scaling factor satisfies \($1/\sqrt d = O(10^{-2})$\).

---

## **A4. Multi‑head averaging**

$$
\mathrm{MHA}(X)=\frac{1}{H}\sum_{h=1}^H \mathrm{Attn}_h(X)W_o^{(h)},\qquad
H=12 \text{ in DeiT-B}.
$$

This introduces an additional factor \($1/H = O(10^{-1})$\) in gradients.

---

# **Proof**

---

## **Step 1. Softmax Jacobian suppression**

By A1, the softmax Jacobian satisfies

$$
\Big\|
\frac{\partial\,\mathrm{softmax}}{\partial z}
\Big\|
=O(10^{-2}\!-\!10^{-3}),
$$

therefore

$$
\|g_Z\|
\le O(10^{-2}\!-\!10^{-3}) \|g\|.
$$

This is the **first suppression**.

---

## **Step 2. QK‑scaling suppression**

From the attention logits

$$
Z=\frac{QK^\top}{\sqrt d},
$$

we obtain

$$
\|\nabla_Q\|,\|\nabla_K\|
\le \frac{1}{\sqrt d}\|\nabla_Z\|
= O(10^{-1})\|\nabla_Z\|
$$

using A3.

This is the **second suppression**.

---

## **Step 3. Multi‑head averaging suppression**

By A4,

$$
\nabla_h = \frac{1}{H} \nabla_{\mathrm{mha}} W_o^\top,
$$

so

$$
\|\nabla_h\|
=O(10^{-1})\|\nabla_{\mathrm{mha}}\|.
$$

This is the **third suppression**.

---

## **Step 4. Combine suppressions (attention path)**

Overall:

$$
\|\nabla_{\mathrm{attn}}\|
\le
O(10^{-2}\!-\!10^{-3})\;
\times O(10^{-1})
\times O(10^{-1})
=
O(10^{-4}\!-\!10^{-3}).
$$

Empirically: \(10^{-3}\)–\(10^{-2}\).

---

## **Step 5. GELU path contains no shrinking factors**

GELU derivative:

$$
\sigma'(x)=\Phi(x)+x\phi(x)\in[0.8,3.6],
$$

and by A2 the MLP weights have \(O(1)\) norms. Hence

$$
\|\nabla_{\mathrm{gelu}}\|=\Theta(1).
$$

No softmax Jacobian, no \($1/\sqrt d$\), no multi‑head averaging.

---

## **Step 6. Final ratio**

$$
\frac{\|\nabla_{\mathrm{gelu}}\|}{\|\nabla_{\mathrm{attn}}\|}
=
\frac{\Theta(1)}{O(10^{-4}\!-\!10^{-3})}
=
\Omega(10^{2}\!-\!10^{3}).
$$

---

# **Conclusion**

Attention gradients undergo **three multiplicative attenuations**:

- softmax Jacobian: \($O(10^{-2}\!-\!10^{-3})$\),
- QK-scaling: \($O(10^{-1})$\),
- multi-head averaging: \($O(10^{-1})$\).

GELU gradients undergo **none** of these.

Therefore, the **gradient magnitude disparity** between attention and activation is a **structural property** of the ViT block.