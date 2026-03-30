<!-- Generated: 2026-03-30 -->
<!-- Phase: Phase 1: Foundations (Weeks 1-3) -->
<!-- Section: 1.1 — Transformer Architecture Internals -->



# Self-attention mechanism (Q, K, V matrices)

## TL;DR
Self‑attention lets each word look at every other word in a sentence, weigh how useful they are, and mix their information together. It does this by turning each word into three special vectors – **Query** (what I’m looking for), **Key** (what I have to offer), and **Value** (the actual content) – then computing similarity scores (dot‑products) between Queries and Keys, turning those scores into weights, and finally summing the weighted Values. The result is a context‑aware representation for every token that can be computed in parallel, which is why Transformers scale so well for training and inference.

---

## ELI5 — The Simple Version
Imagine you’re at a party and you want to understand the story behind a funny joke someone just told. To get the joke, you don’t just listen to the punch‑line; you glance around, see who’s laughing, recall what they said earlier, and piece together the context.  

In a sentence, each word plays the same role: it needs to “look” at the other words to decide how much each of them should influence its meaning. Self‑attention is the mechanism that lets a word ask, *“Hey, how relevant is every other word to me right now?”* and then blend in the right amount of each word’s meaning.

Think of three helpers for every word:

| Helper | What it does | Real‑world analogue |
|--------|--------------|---------------------|
| **Query** | “What am I trying to find out?” | You, holding a question in your head. |
| **Key**   | “What do I have to offer as a clue?” | Each guest wearing a name tag that hints at their story. |
| **Value** | “Here’s the actual information I can share.” | The guest’s full anecdote you’d hear if you asked them. |

The word computes a similarity score between its Query and every other word’s Key (higher score = more relevant). Those scores are turned into weights that sum to 1, and then the word adds up the Values weighted by those scores. The result is a new representation that already “knows” the context it needed.

---

## Why This Matters for Inference Engineering
When you serve a large language model (LLM) you’re repeatedly running the self‑attention layer for every token generated. Understanding how Q, K, V are built and used lets you:

* **Spot bottlenecks** – the dominant cost is the Q·Kᵀ matrix multiplication (quadratic in sequence length).  
* **Apply optimizations** – FlashAttention, PagedAttention, KV‑cache reuse, quantization, and speculative decoding all target specific parts of this computation.  
* **Design serving systems** – knowing that Keys and Values are cached for later tokens (the KV‑cache) helps you reason about memory layout, batching, and latency trade‑offs.  
* **Debug numerical issues** – softmax scaling, dropout, and head‑splitting are all knobs you can tune when you see NaNs or instability in production.

In short: if you can’t explain self‑attention, you can’t truly optimize LLM inference.

---

## How It Actually Works
We’ll start from the analogy, then introduce the math, and finally show how the recent papers improve each step.

### 1. From words to vectors
Each token → embedding vector **x** ∈ ℝᵈᵐᵒᵈ (e.g., 512).  
We keep a matrix **X** ∈ ℝⁿ×ᵈ (n = sequence length, d = model dim).

```
X = [x₁; x₂; …; xₙ]ᵀ   # each row is a token embedding
```

### 2. Project to Q, K, V
We learn three weight matrices **W_Q**, **W_K**, **W_V** ∈ ℝᵈ×ᵈₖ (often dₖ = d/h where h = number of heads).  
For each token we compute:

```
Q = X W_Q   # shape (n, dₖ)
K = X W_K   # shape (n, dₖ)
V = X W_V   # shape (n, dₖ)
```

Think of these as the three helpers per token.

### 3. Similarity scores (dot‑product)
We ask: how much does token i’s Query match token j’s Key?

```
scores = Q Kᵀ   # shape (n, n)
```

Entry scores[i, j] = q_i · k_j.  
Higher dot‑product → more similarity.

**Why dot‑product?**  
If vectors are normalized, dot‑product = cosine similarity → a natural measure of alignment.

### 4. Scale & softmax
To keep gradients stable we divide by √dₖ (the “scaled dot‑product attention” from the original paper).

```
attn_weights = softmax( scores / √dₖ )   # each row sums to 1
```

Now each row i is a distribution over all tokens telling token i how much to attend to each j.

### 5. Weighted sum of Values
Finally we mix the Values:

```
output = attn_weights V   # shape (n, dₖ)
```

Each token’s new representation is a blend of all Values, weighted by how relevant those tokens were to its Query.

### 6. Multi‑head attention (the “parallel perspectives” trick)
Instead of one big Q/K/V, we split the dimension into **h** heads, each with its own smaller projection. This lets the model attend to different subspaces (e.g., syntax vs. semantics) simultaneously.

```
Q = concat([Q₁, Q₂, …, Q_h])   # each Q_i ∈ ℝⁿ×(dₖ/h)
...
output = concat([head₁, …, head_h]) W_O   # W_O mixes heads back to d
```

### 7. Where the papers fit in

| Paper | What it improves | Plain‑language insight |
|-------|------------------|------------------------|
| **Attention Is All You Need (2017)** | Introduced Q/K/V self‑attention, multi‑head, positional encoding, and the encoder‑decoder stack. | Showed that recurrence isn’t needed; pure attention + feed‑forward can translate languages better and faster. |
| **FlashAttention (2022)** | Makes the Q·Kᵀ → softmax → V multiplication **memory‑efficient** by tiling and recomputing on‑chip. | Instead of storing the huge n×n attention matrix in GPU HBM, we break it into blocks, keep intermediate results in fast SRAM, and reduce HBM traffic → 2‑3× speedup on long sequences. |
| **PagedAttention / vLLM (2023)** | Solves KV‑cache fragmentation by treating cache pages like OS virtual memory pages. | Keys & Values for each request are stored in fixed‑size pages; pages can be reused or swapped, giving near‑zero waste and enabling massive batch sizes. |
| **GPTQ (2023)** | Post‑training quantization of the weight matrices (including W_Q, W_K, W_V, W_O). | Uses second‑order info to round weights to 3‑4 bits with tiny accuracy loss, cutting memory and compute for inference. |
| **Speculative Decoding (2023)** | Generates multiple tokens per forward pass by using a cheap draft model to propose candidates, then verifying them with the target model. | Reduces the serial dependency of autoregressive decoding; the attention computation is still done, but you get 2‑3× more tokens per GPU step. |
| **Efficiently Scaling Transformer Inference (Google, 2023)** | Combines model parallelism, multi‑query attention (MQA), and int8 quantization to hit low latency on huge models. | MQA shares a single Key/Value head across many Query heads, shrinking the KV‑cache and allowing 32× longer contexts without blowing memory. |

These works don’t change the *definition* of self‑attention; they make it **faster**, **less memory‑hungry**, or **more scalable** while preserving the exact same mathematical output (except for quantization, which introduces a controlled approximation).

---

## Code You Can Run
Below is a minimal, self‑contained PyTorch implementation of **single‑head** scaled dot‑product attention. You can run it on a DGX Spark (Blackwell GPU) or CPU; it will automatically use CUDA if available.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Minimal self‑attention (no dropout, no masking).
    Input:  (batch_size, seq_len, embed_dim)
    Output: (batch_size, seq_len, embed_dim)
    """
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim          # dimensionality of Q/K/V per head
        # projection matrices
        self.W_Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, head_dim, bias=False)
        # optional output projection (here we keep dims equal for simplicity)
        self.W_O = nn.Linear(head_dim, embed_dim, bias=False)

    def forward(self, x):
        """
        x: Tensor of shape (B, T, C) where
           B = batch size,
           T = sequence length,
           C = embed_dim
        """
        B, T, C = x.shape
        # 1) Project to Q, K, V
        Q = self.W_Q(x)   # (B, T, head_dim)
        K = self.W_K(x)   # (B, T, head_dim)
        V = self.W_V(x)   # (B, T, head_dim)

        # 2) Scaled dot‑product
        #   Q @ K^T -> (B, T, T)
        scores = torch.bmm(Q, K.transpose(1, 2))          # batch matrix‑mult
        scores = scores / (self.head_dim ** 0.5)         # scale

        # 3) Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)         # (B, T, T)

        # 4) Weighted sum of Values
        context = torch.bmm(attn_weights, V)             # (B, T, head_dim)

        # 5) Optional output projection back to model dim
        out = self.W_O(context)                          # (B, T, C)
        return out, attn_weights

# ------------------- Demo -------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C = 2, 4, 8          # tiny example: 2 sentences, 4 tokens, 8-dim embed
    embed_dim = C
    head_dim = 4               # must divide embed_dim if we used multi‑head
    x = torch.randn(B, T, embed_dim)   # random embeddings

    attn = SelfAttention(embed_dim, head_dim)
    output, weights = attn(x)

    print("Input embeddings:\n", x)
    print("\nAttention weights (batch 0):\n", weights[0].detach().numpy())
    print("\nOutput after attention:\n", output[0].detach().numpy())
```

**What to notice**

* The `scores` matrix is `T×T` – that’s the quadratic term.  
* `attn_weights` rows sum to 1 (softmax).  
* Changing `head_dim` or adding more heads (by splitting `embed_dim`) is how you get multi‑head attention.  
* On a real GPU you’d see the `bmm` calls fused into a highly optimized kernel (FlashAttention does exactly that under the hood).

---

## Paper Breakdown (Relevant to This Topic)

| Paper | One‑line summary | The problem before | Key idea (plain language) | Impact |
|-------|------------------|--------------------|---------------------------|--------|
| **Attention Is All You Need (2017)** | Introduced the Transformer architecture based solely on self‑attention and feed‑forward layers. | Seq2seq models relied on RNNs/LSTMs, which are hard to parallelize and suffer from vanishing gradients on long sequences. | Replace recurrence with a mechanism that lets every token directly attend to every other token via Q/K/V dot‑products, plus positional encodings to retain order. | Enabled massive parallel training, set new BLEU records, and became the foundation for virtually all modern LLMs. |
| **FlashAttention (2022)** | An IO‑aware algorithm that computes exact attention with far fewer GPU memory reads/writes. | Standard attention materializes the full n×n attention matrix in HBM, causing bandwidth bottlenecks on long sequences. | Tile the Q, K, V matrices, compute softmax statistics on‑chip in SRAM, and reuse blocks so that HBM traffic is reduced asymptotically. | Gives 2‑3× speedup on GPT‑2 (seq‑len 1K) and enables longer contexts without changing model quality. |
| **PagedAttention / vLLM (2023)** | Treats the KV‑cache as a set of fixed‑size pages managed like OS virtual memory, eliminating fragmentation. | KV‑cache grows per request; allocating contiguous tensors leads to wasted memory and limits batch size. | Store Keys and Values in page‑sized blocks; allocate pages on demand, reuse them across requests, and swap them like virtual memory pages. | Near‑zero KV‑cache waste, 2‑4× higher throughput for LLM serving, especially with long generations. |
| **GQT (2023) – GPTQ** | One‑shot post‑training quantization of Transformer weights to 3‑4 bits using second‑order information. | Prior quantization methods either needed retraining or caused large accuracy drops at low bitwidth. | Approximate the Hessian of the loss w.r.t. weights to guide optimal rounding, preserving model fidelity while drastically cutting bitwidth. | Enables 175B‑parameter models to run on a single GPU for inference with <1% perplexity loss. |
| **Speculative Decoding (2023)** | Generates multiple tokens per forward pass by using a cheap draft model to propose candidates, then verifying them with the target model. | Autoregressive decoding is strictly serial: each token depends on the previous one, limiting GPU utilization. | Draft model quickly proposes several tokens; the target model runs in parallel to check correctness, accepting or discarding proposals, yielding 2‑3× more tokens per step without altering the distribution. | Cuts latency for tasks like chatbots and code completion while keeping exact model outputs. |
| **Efficiently Scaling Transformer Inference (Google, 2023)** | Combines model parallelism, multi‑query attention, and int8 quantization to hit low latency on huge models. | Scaling inference to >100B parameters hits memory bandwidth and compute limits; naive parallelism hurts utilization. | Multi‑query attention shares a single K/V head across many Q heads, shrinking the KV‑cache; int8 weight quantization reduces memory traffic; careful TPU partitioning balances compute and memory. | Achieves 29 ms/token latency on a 540B‑parameter PaLM model with 2048‑token context, pushing the Pareto frontier of latency vs. MFU. |

---

## Key Takeaways
- Self‑attention = **Query** asks, **Key** answers, **Value** supplies the content; similarity scores decide how much each Value contributes.  
- The core computation is **Q·Kᵀ → softmax → V**, which is **quadratic** in sequence length and thus the main speed/memory bottleneck.  
- Multi‑head attention lets the model capture different types of relationships in parallel.  
- FlashAttention reduces the memory bandwidth cost of the quadratic step by tiling and recomputing on‑chip SRAM.  
- PagedAttention treats the KV‑cache like OS pages, eliminating fragmentation and enabling massive batching.  
- Quantization (GPTQ) and speculative decoding cut compute and latency without changing the underlying attention math.  
- Multi‑query attention (MQA) shares Keys/Values across many Queries, shrinking the KV‑cache and allowing far longer contexts.  
- Understanding these pieces lets you **diagnose**, **optimize**, and **serve** LLMs efficiently on modern hardware like Blackwell GPUs or TPUs.  

---

## What's Next
Now that you know **what** self‑attention does and **how** it’s computed, the next steps are:

1. **Positional encodings** – how the model injects order information since attention itself is permutation‑invariant.  
2. **Layer normalization & residual connections** – why they stabilize deep stacks of attention‑+‑feed‑forward blocks.  
3. **Decoder‑only (causal) masking** – turning self‑attention into the autoregressive “look‑only‑left” mechanism used in GPT‑style models.  
4. **Putting it all together** – walking through a full forward pass from token IDs to logits, and seeing where the KV‑cache lives during generation.  

Master those, and you’ll be ready to tackle the inference‑engineering challenges of KV‑cache management, batching, quantization, and speculative decoding that we just surveyed. Happy coding! 🚀