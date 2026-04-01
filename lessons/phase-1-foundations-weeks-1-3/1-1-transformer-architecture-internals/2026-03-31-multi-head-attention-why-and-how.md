<!-- Generated: 2026-03-31 -->
<!-- Phase: Phase 1: Foundations (Weeks 1-3) -->
<!-- Section: 1.1 — Transformer Architecture Internals -->



# Multi-head attention — why and how  

## TL;DR  
Multi‑head attention lets a Transformer look at a word from many different “perspectives” at once, giving it a richer understanding of context. Each head learns its own set of query/key/value projections, so the model can capture various types of relationships (syntactic, semantic, positional, etc.) in parallel. The outputs of all heads are concatenated and linearly transformed back to the model dimension, preserving the ability to stack many layers efficiently.  

---

## ELI5 — The Simple Version  

### The “detective squad” analogy  
Imagine you’re trying to figure out who stole the cookies from the jar. You could ask just one detective to interview everyone, but that detective might miss clues because they’re only looking for one type of evidence (e.g., footprints). Instead, you assemble a **team of detectives**, each with a different specialty: one looks for fingerprints, another listens for alibis, a third checks security‑camera timestamps, and so on. When they finish, you combine their reports to get a complete picture of what happened.  

In a Transformer, each word is like a detective squad. The **query** is “what am I trying to understand about this word?” The **key** is “what does every other word have to offer?” The **value** is “the actual information I’ll take from a word if it looks relevant.” By having **multiple heads**, we give the model several detective squads that each focus on a different kind of clue (grammar, meaning, distance, etc.).  

### Why multiple heads help  
If we only had one head, the model would be forced to use a single set of similarity scores to decide which words matter. That’s like asking every detective to use the same magnifying glass—some clues would be blurred. With multiple heads, each head learns its own projection matrices (WQ, WK, WV), so they can stretch, rotate, or squash the embedding space in different ways before computing similarity. The result is a set of **parallel, specialized attention maps** that together capture a far richer context than any single map could.  

---

## Why This Matters for Inference Engineering  

* **Parallelism = speed** – Each head can be computed independently on different GPU cores or tensor cores, making the attention layer highly parallelizable.  
* **Memory layout matters** – The way we store queries, keys, and values for many heads (and many tokens) determines whether we hit the GPU’s memory bandwidth wall. Optimizations like FlashAttention, PagedAttention, and multi‑query attention are all about rearranging these tensors to reduce data movement.  
* **Quantization & serving** – When we later compress weights (GPTQ) or cache key/value pairs (PagedAttention), we do it **per head** because each head has its own projection matrices. Understanding the head structure lets you apply these tricks correctly.  
* **Speculative decoding** – Faster token generation often relies on re‑using partial attention results; knowing how heads contribute helps you design approximations that preserve quality.  

In short: if you want to squeeze latency, throughput, or memory out of an LLM serving stack, you must first understand what multi‑head attention actually does under the hood.  

---

## How It Actually Works  

### 1. From embeddings to Q, K, V  

Assume we have an input sequence of length **T** (e.g., 5 tokens). Each token is embedded into a vector of size **d_model** (the model width, e.g., 512).  

```
Embedding matrix X ∈ ℝ^(T × d_model)
```

For each token we compute three projections:

```
Q = X W_Q   (queries)   ∈ ℝ^(T × d_k)
K = X W_K   (keys)      ∈ ℝ^(T × d_k)
V = X W_V   (values)    ∈ ℝ^(T × d_v)
```

`W_Q, W_K, W_V` are learned matrices of shape `d_model × d_k` (or `d_v`). In the original paper they set `d_k = d_v = d_model / h`, where **h** is the number of heads.  

### 2. Splitting into heads  

We reshape the projected matrices so that each head gets its own sub‑space:

```
Q_head[i] = Q[:, i·d_k : (i+1)·d_k]   shape (T, d_k)
K_head[i] = K[:, i·d_k : (i+1)·d_k]
V_head[i] = V[:, i·d_v : (i+1)·d_v]
```

Now we have **h** independent sets of Q, K, V.  

### 3. Scaled dot‑product attention per head  

For a single head we compute:

```
scores = Q_head K_head^T / sqrt(d_k)          (T × T)
attn   = softmax(scores)                      (T × T)
output = attn V_head                          (T × d_v)
```

The `sqrt(d_k)` scaling prevents the dot‑products from growing too large when `d_k` is big, which would push softmax into saturated regions.  

### 4. Concatenating heads & final linear  

We concatenate the `h` outputs along the feature dimension:

```
concat = [output_0 ; output_1 ; … ; output_{h-1}]   shape (T, h·d_v)
```

Because we chose `d_v = d_model / h`, the concatenated size is exactly `d_model`. Finally we apply a output projection matrix `W_O`:

```
MultiHead = concat W_O   ∈ ℝ^(T × d_model)
```

That result is fed into the position‑wise feed‑forward network (FFN) of the encoder/decoder block.  

### ASCII diagram of one Transformer encoder layer  

```
Input X (T × d_model)
   │
   ├─► Linear W_Q  → Q (T × d_model)
   │          │
   │          ├─► split into h heads → Q_i (T × d_k)   (i=0…h-1)
   │          │
   ├─► Linear W_K  → K (T × d_model)   (same split → K_i)
   │          │
   └─► Linear W_V  → V (T × d_model)   (same split → V_i)

For each head i:
   scores_i = Q_i K_i^T / sqrt(d_k)          (T × T)
   attn_i   = softmax(scores_i)              (T × T)
   out_i    = attn_i V_i                     (T × d_v)

Concat out_0…out_{h-1} → (T × h·d_v) = (T × d_model)
   │
   └─► Linear W_O → MultiHead output (T × d_model)
```

### 5. Intuition behind the math  

* **Query** = “What am I looking for?”  
* **Key**   = “What does each token have to offer?”  
* **Value** = “The actual content I’ll copy if the key matches my query.”  

The dot‑product `Q K^T` measures similarity between what we’re looking for and what each token offers. Softmax turns those similarities into a distribution (weights that sum to 1). Multiplying by `V` gives a weighted sum of the values—i.e., a blend of information from the most relevant tokens.  

Having multiple heads means we compute **several** such blends, each with its own learned notion of similarity (different W_Q/W_K). The model can thus attend to syntax in one head, semantics in another, and long‑range dependencies in a third—all simultaneously.  

---

## Code You Can Run  

Below is a minimal, self‑contained PyTorch implementation of multi‑head attention that you can paste into a notebook on your DGX Spark (Blackwell GPU, 128 GB unified memory). It follows the equations above and includes comments linking each step to the paper concepts.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Vanilla multi‑head attention as described in "Attention Is All You Need".
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # dimensionality per head (query/key)
        self.d_v = d_model // num_heads   # dimensionality per head (value)

        # Projection matrices for Q, K, V (shared across heads, then split)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: (batch_size, seq_len, d_model)
        mask: optional (batch_size, 1, 1, seq_len) for padding / causal mask
        Returns: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1) Linear projections -> (batch, seq_len, d_model)
        Q = self.W_q(x)   # queries
        K = self.W_k(x)   # keys
        V = self.W_v(x)   # values

        # 2) Split into heads -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        # 3) Scaled dot‑product attention
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)          # (batch, num_heads, seq_len, seq_len)
        attn = self.dropout(attn)

        # context: (batch, num_heads, seq_len, d_v)
        context = torch.matmul(attn, V)

        # 4) Concatenate heads -> (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 5) Final linear projection
        output = self.W_o(context)
        return output

# ------------------------------
# Quick sanity check
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_model = 2, 4, 8   # tiny model for demo
    mha = MultiHeadAttention(d_model=8, num_heads=2)

    # random input embeddings
    x = torch.randn(batch, seq_len, d_model)

    # optional causal mask (prevent attending to future tokens)
    mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,seq_len)

    out = mha(x, mask=mask)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Sample output:\n", out[0, 0, :4])  # first token of first batch
```

**What to notice while running:**  

* Changing `num_heads` while keeping `d_model` fixed changes the per‑head dimension (`d_k`).  
* The mask implements causal attention (used in decoder blocks) – try removing it to see how each token can attend to future positions.  
* On a Blackwell GPU you’ll see the operation launch many kernels in parallel because the head dimension (`num_heads`) is a natural axis for parallelism.  

---

## Paper Breakdown  

Below we connect the core multi‑head attention mechanism to the six papers you listed. For each we give a one‑line summary, the problem it solved, the key idea (plain language), and its impact.

| Paper | One‑line summary | The problem (what existed & why it sucked) | Key idea (simple) | Impact |
|-------|------------------|--------------------------------------------|-------------------|--------|
| **Attention Is All You Need (2017)** | Introduced the Transformer, showing that stacked self‑attention layers can replace recurrence/convolutions for seq‑2‑seq tasks. | RNNs/CNNs were slow to train because they processed tokens sequentially or with limited receptive fields; attention existed but was added onto RNNs, not the whole model. | **Self‑attention + multi‑head** lets every token directly interact with every other token in a single layer, and multiple heads give diverse interaction patterns. | Sparked the LLM revolution; enabled massive parallel training on GPUs/TPUs. |
| **FlashAttention: Fast and Memory‑Efficient Exact Attention (2022)** | Made the exact quadratic‑time attention algorithm run faster by reducing GPU memory traffic via tiling and awareness of the memory hierarchy. | Standard attention materializes the full `QK^T` matrix (size `T×T`) in HBM, causing many reads/writes and limiting sequence length. | **Tile the computation**: compute attention in chunks that fit into on‑chip SRAM, reuse loaded Q/K/V blocks, and avoid writing the huge intermediate matrix back to HBM. | Gives 2‑3× speedup on typical sequence lengths (512‑2048) without approximation; enables longer contexts and higher throughput. |
| **GPTQ: Accurate Post‑Training Quantization for Generative Pre‑trained Transformers (2023)** | Provides a fast, one‑shot weight quantization method for LLMs that preserves accuracy down to 3‑4 bits per weight. | Quantizing huge LLMs (>100B params) was slow and inaccurate; prior methods needed per‑layer retraining or suffered large drops. | **Approximate second‑order information** (Hessian diagonal) lets us solve optimal quantization per weight block in a single pass, keeping error low. | Enables fitting a 175B‑parameter model on a single GPU for inference, with ~3‑4× speedup vs FP16. |
| **Efficient Memory Management for LLM Serving with PagedAttention (2023) — vLLM** | Treats the KV‑cache like virtual memory pages, eliminating fragmentation and allowing sharing across requests. | Serving many requests leads to a huge, irregularly‑shaped KV cache that wastes memory via internal fragmentation and duplicate storage. | **PagedAttention**: allocate KV cache in fixed‑size pages (like OS pages), map logical token positions to physical pages, and share pages when prefixes match. | Improves serving throughput 2‑4× at same latency; enables longer contexts and larger batch sizes. |
| **Fast Inference from Transformers via Speculative Decoding (2023)** | Uses a small, fast draft model to propose multiple tokens, then validates them in parallel with the big model. | Autoregressive decoding is inherently serial: each token needs a full forward pass, making latency grow linearly with output length. | **Speculative execution**: run the cheap model to guess K tokens, run the big model once on those guesses, accept correct guesses, and repeat. No change to output distribution. | 2‑3× speedup on models like T5‑XXL without retraining; works with any autoregressive LM. |
| **Efficiently Scaling Transformer Inference (2023) — Google** | Combines model parallelism, multi‑query attention, and quantization to achieve low latency and high throughput on massive models (PaLM‑540B). | Scaling inference to hundred‑billion‑parameter models hits memory bandwidth and compute limits; naïve parallelism gives poor utilization. | **Multi‑query attention (MQA)**: share a single set of keys/values across all query heads, drastically reducing KV cache size and memory movement; combine with tensor‑parallel partitioning and int8 quantization. | Achieves 29 ms/token latency on 540B model with 2048‑token context, 76% MFU, and enables 32× longer contexts via MQA. |

### How these papers relate to multi‑head attention  

* **FlashAttention** attacks the *inner loop* of multi‑head attention (the `QK^T` softmax) by making it memory‑efficient.  
* **GPTQ** compresses the projection matrices `W_Q, W_K, W_V, W_O` that produce the per‑head queries/keys/values.  
* **PagedAttention** re‑organizes the *output* of multi‑head attention (the cached keys and values) to serve many requests without waste.  
* **Speculative Decoding** can be viewed as running a *smaller* multi‑head attention draft model in parallel with the big one.  
* **Efficiently Scaling Transformer Inference** shows that *reducing the number of distinct key/value heads* (multi‑query attention) cuts memory bandwidth, which is a direct variant of the standard multi‑head design.  
* The original **Attention Is All You Need** paper is the foundation that defined the multi‑head building block all these later works optimize.

---

## Key Takeaways  

- Multi‑head attention = several independent attention “lenses” that run in parallel and are merged.  
- Each head learns its own query/key/value projections, letting the model capture different types of relationships (syntax, semantics, distance, etc.).  
- The math is a set of scaled dot‑product attentions, followed by concatenation and a final linear projection.  
- Optimizations (FlashAttention, PagedAttention, MQA, quantization) target the **memory movement** and **computation layout** of those projections and the intermediate `QK^T` matrices.  
- Understanding the head‑level structure is essential for applying inference‑engineering tricks like speculative decoding, KV‑caching, and model parallelism.  
- Modern LLMs (e.g., PaLM‑540B) rely on these ideas to achieve low latency, high throughput, and long context lengths.  

---

## What's Next  

Having grasped **what** multi‑head attention computes, the next step is to see **how it fits inside a full Transformer block** (self‑attention → add‑norm → feed‑forward → add‑norm) and how stacking many such layers builds hierarchical representations. After that we’ll dive into **training tricks** (learning‑rate schedules, warm‑up, weight initialization) and then move onto **inference‑specific topics**: KV‑cache management, quantization (GPTQ), speculative decoding, and system‑level optimizations like FlashAttention and PagedAttention.  

Stay tuned—we’ll go from the math inside a single head to serving a 100‑billion‑parameter model on a single GPU! 🚀

## Watch These Videos

- **[Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M)** (27:14)
- **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (1:56:20)
- **[Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://www.youtube.com/watch?v=bCz4OMemCcA)** (58:04)
