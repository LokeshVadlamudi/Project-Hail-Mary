<!-- Generated: 2026-04-01 -->
<!-- Phase: Phase 1: Foundations (Weeks 1-3) -->
<!-- Section: 1.1 — Transformer Architecture Internals -->



# Feed‑forward layers, layer norms, residual connections  

## TL;DR  
In a Transformer each token’s representation is first mixed with other tokens (self‑attention) and then passed through a tiny “per‑token” neural network (the feed‑forward block). To keep the signal from vanishing or exploding as it travels through many stacked blocks, we add **skip connections** (residuals) and **layer‑normalisation** that re‑centres and re‑scales each token’s vector. Together these three pieces make deep Transformers stable, fast to train, and easy to run at inference time.

---

## ELI5 — The Simple Version  

### Imagine a relay race with a magical baton  

You have a line of runners (the token positions). Each runner must:

1. **Look around** – see what the other runners are doing (self‑attention).  
2. **Do a quick personal workout** – a few push‑ups that only depend on how tired *they* feel (feed‑forward network).  
3. **Pass the baton** to the next runner **without dropping it** – you add the baton you just received to your own effort (residual connection).  
4. **Make sure the baton isn’t too heavy or too light** – you adjust its weight so every runner feels roughly the same load (layer‑norm).

If you skipped steps 3 or 4, after a few runners the baton would become impossibly heavy (exploding gradients) or feather‑light (vanishing gradients), and the race would fall apart. The residual connection lets each runner keep a copy of the baton they got, while the feed‑forward workout lets them add their own personal improvement. Layer‑norm simply normalises the baton’s weight so the next runner isn’t overwhelmed.

### Why the feed‑forward part?  

After the runners have shared information (attention), each one still needs to **transform** that mixed information in a way that’s specific to *them*. A small two‑layer perceptron (feed‑forward network) does exactly that: it takes the 512‑dimension vector, expands it to a bigger hidden size (usually 4×, i.e. 2048), applies a non‑linearity, then squeezes it back to 512. Because the same network is applied **independently** to every token, the whole step can be run in parallel on a GPU — just like all runners doing their push‑ups at the same time.

---

## Why This Matters for Inference Engineering  

* **Latency budget:** Inference runs the same feed‑forward + norm + residual stack many times (once per transformer layer). Knowing how each piece works lets you spot where you can fuse kernels, quantise, or skip work without breaking the math.  
* **Memory usage:** Layer‑norm introduces two learnable parameters per hidden dimension (γ and β). They are tiny, but they must be read for every token; understanding their cost helps you decide whether to fuse them into the preceding matmul.  
* **Numerical stability:** Residual connections keep the activation magnitude roughly constant across layers, which means you can safely use lower‑precision (FP16/INT8) arithmetic during inference without catastrophic loss of accuracy.  
* **Throughput:** Because the feed‑forward block is token‑wise, it maps perfectly to a **matrix‑multiply** (batched GEMM). Optimising that GEMM is the single biggest win for LLM serving kernels (e.g., FlashAttention‑style optimisations for the FFN).  

---

## How It Actually Works  

We’ll walk through a single transformer **encoder layer** (the decoder is analogous, just with an extra cross‑attention block).  

### 1. Notation  

* `x ∈ ℝ^{T × D}` – input token matrix (`T` = sequence length, `D` = model dimension, usually 512).  
* `W_Q, W_K, W_V ∈ ℝ^{D × D_k}` – projection matrices for queries, keys, values (often `D_k = D / h`, `h` = number of heads).  
* `W_O ∈ ℝ^{h·D_v × D}` – output projection after multi‑head attention.  
* `W₁ ∈ ℝ^{D × D_ff}`, `W₂ ∈ ℝ^{D_ff × D}` – feed‑forward weight matrices (`D_ff = 4·D` in the original paper).  
* `γ, β ∈ ℝ^{D}` – layer‑norm scale and shift.  

### 2. Residual connection (skip‑add)  

```
   x ──► ──► sublayer (attention or FFN) ──► + ──► x_out
   │                                 ▲
   └───────────────(add)─────────────┘
```

Mathematically:

```
x' = x + Sublayer(x)
```

The **identity** path guarantees that, even if the sub‑layer outputs zero, the signal still flows forward. This prevents gradients from shrinking to zero as depth increases (the “vanishing gradient” problem).  

### 3. Layer Normalisation  

Applied **before** the sub‑layer in the “pre‑norm” variant (used in most modern LLMs like GPT‑3, PaLM, LLaMA).  

For each token position `t` we compute:

```
μ_t = mean(x'[t, :])                     # scalar
σ_t = sqrt(var(x'[t, :]) + ε)            # scalar, ε≙1e-5 for numerical stability
y[t, :] = γ * (x'[t, :] - μ_t) / σ_t + β
```

*Interpretation:* we centre each token’s vector to zero mean and scale to unit variance, then stretch/scale by learned `γ` and shift by `β`. This keeps the distribution of activations stable across layers and batches, which is crucial for large‑batch training and for inference with varying sequence lengths.

### 4. Feed‑forward network (FFN)  

Two linear layers with a non‑linearity (usually GELU) in between:

```
FFN(x) = W₂ · GELU( W₁ · x + b₁ ) + b₂
```

* Why the expansion?  
  * The projection to a larger dimension (`D_ff = 4·D`) lets the network model richer, position‑wise transformations.  
  * Because the same weights are used for every token, the operation is a **batched matrix multiply**:  

```
X ∈ ℝ^{T × D}
X₁ = X·W₁ᵀ + b₁          → ℝ^{T × D_ff}
X₂ = GELU(X₁)             → ℝ^{T × D_ff}
X_out = X₂·W₂ᵀ + b₂       → ℝ^{T × D}
```

* Parallelism: each row (token) is independent → perfect for GPU SIMD execution.  

### 5. Putting it together (pre‑norm encoder layer)  

```
# Input: x (T×D)
# 1️⃣ Self‑attention block
z = x + MHSA( LayerNorm(x) )          # MHSA = multi‑head self‑attention
# 2️⃣ Feed‑forward block
out = z + FFN( LayerNorm(z) )
```

If you prefer the **post‑norm** variant (the original “Attention Is All You Need”), the norm comes **after** the residual addition:

```
z = LayerNorm( x + MHSA(x) )
out = LayerNorm( z + FFN(z) )
```

Both work; pre‑norm is empirically easier to train very deep models because the gradient has a smoother path through the norm.

### 6. ASCII diagram of one layer (pre‑norm)

```
   x ────────────────────────────────────────────────────────► out
   │                                                          │
   │   LayerNorm                                              │
   ▼                                                          ▼
  ┌─────┐                                                    ┌─────┐
  │MHSA │                                                    │ FFN │
  └─────┘                                                    └─────┘
   │                                                          │
   │   + (residual)                                           │   + (residual)
   ▼                                                          ▼
  x + MHSA(LayerNorm(x))   ────────►   z + FFN(LayerNorm(z))
```

---

## Paper Breakdown  

### Paper: **“Attention Is All You Need”** (Vaswani et al., 2017)  

| Item | Details |
|------|---------|
| **One‑line summary** | Introduced the Transformer architecture, replacing recurrence with self‑attention and adding feed‑forward, residual, and layer‑norm blocks. |
| **The problem** | Prior state‑of‑the‑art seq2seq models relied on RNNs/LSTMs, which compute tokens sequentially → hard to parallelise, suffer from vanishing gradients, and limited context length. |
| **Key idea** | *Self‑attention* lets each token directly attend to every other token in O(1) depth; a **position‑wise feed‑forward network** processes each token independently; **residual connections** preserve gradient flow; **layer normalisation** stabilises activations across deep stacks. |
| **Impact** | Enabled massive parallel training on TPUs/GPUs, sparked the LLM revolution, and became the default architecture for virtually all modern language models (BERT, GPT, T5, LLaMA, etc.). The paper also showed that stacking 6 identical encoder‑decoder blocks outperformed complex RNN‑based seq2seq models on WMT translation benchmarks while being far faster to train. |

---

## Code You Can Run  

Below is a **minimal, self‑contained PyTorch module** that implements a single pre‑norm Transformer encoder layer (self‑attention + feed‑forward). You can copy‑paste it into a notebook on your DGX Spark and run it with a dummy token sequence.

```python
# ------------------------------------------------------------
# Minimal Transformer Encoder Layer (pre‑norm)
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # projection matrices for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        B, T, D = x.shape

        # 1) Project to Q, K, V
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2) Split into heads -> (B, n_heads, T, d_head)
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 3) Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, n_heads, T, T)
        attn_weights = F.softmax(scores, dim=-1)                            # (B, n_heads, T, T)
        attn_output = torch.matmul(attn_weights, V)                         # (B, n_heads, T, d_head)

        # 4) Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        # 5) Output projection
        out = self.W_o(attn_output)
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model   # default 4× as in the paper
        self.fc1 = nn.Linear(d_model, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerEncoderLayer(nn.Module):
    """
    Pre‑norm encoder layer:
        x' = x + MHSA( LayerNorm(x) )
        out = x' + FFN( LayerNorm(x') )
    """
    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- Self‑attention block ----
        norm_x = self.ln1(x)                 # (B, T, D)
        attn_out = self.attn(norm_x)         # (B, T, D)
        x = x + attn_out                     # residual

        # ---- Feed‑forward block ----
        norm_x = self.ln2(x)                 # (B, T, D)
        ffn_out = self.ffn(norm_x)           # (B, T, D)
        x = x + ffn_out                      # residual
        return x


# ------------------- Demo -------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, d_model = 2, 5, 512   # tiny example
    dummy_input = torch.randn(batch_size, seq_len, d_model)

    layer = TransformerEncoderLayer(d_model=d_model, n_heads=8)
    output = layer(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)
    # Quick sanity check: output should have same shape as input
    assert output.shape == dummy_input.shape
    print("✅ Layer works!")
```

**What the code does**

1. **MultiHeadSelfAttention** builds Q, K, V via linear layers, splits into heads, computes scaled dot‑product attention, and recombines.  
2. **PositionwiseFeedForward** expands to 4× dimensionality, applies GELU, then projects back.  
3. **LayerNorm** (`nn.LayerNorm`) normalises across the feature dimension (`D`).  
4. **Residual connections** are simple tensor additions (`x + ...`).  
5. The demo runs a random tensor through one layer and verifies shape invariance.

Feel free to:

* Change `d_model` to 768 or 1024 to see how memory scales.  
* Replace `F.gelu` with `F.relu` to experiment with activation choices.  
* Stack multiple `TransformerEncoderLayer`s in a `nn.ModuleList` to build a mini‑encoder and compare forward‑pass time vs. a naïve RNN baseline.

---

## Key Takeaways  

- **Residual connections** = identity shortcut that lets gradients flow unchanged through deep stacks.  
- **Layer normalisation** = per‑token mean‑zero, unit‑variance re‑scaling (with learnable gain/bias) that stabilises activations across layers and batches.  
- **Feed‑forward network** = token‑wise two‑layer MLP (usually 4× expansion) that mixes the attended information in a position‑specific way.  
- Together they give **stable, parallelisable, and efficiently trainable** building blocks that form the heart of every modern LLM.  
- For inference engineering, knowing where the **matmul‑heavy** parts (attention QKV projections, FFN expansions) live lets you target kernel fusion, quantisation, and memory‑layout optimisations.  

---

## What's Next  

Next we’ll dive into **multi‑head attention** itself: how the queries, keys, and values are computed, why we split into heads, and the tricks (like FlashAttention) that make the attention step fast on Blackwell GPUs. After that we’ll look at **positional encodings** (sinusoidal vs. learned vs. RoPE) and see how they plug into the attention machinery. Finally, we’ll put all the pieces together and trace a full forward pass from raw text to logits—exactly the kind of end‑to‑end inspection you’ll need when you start optimising LLM serving pipelines.  

Stay tuned, and keep those residual connections (and your curiosity) flowing! 🚀

## Watch These Videos

- **[Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M)** (27:14)
- **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (1:56:20)
- **[Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://www.youtube.com/watch?v=bCz4OMemCcA)** (58:04)
