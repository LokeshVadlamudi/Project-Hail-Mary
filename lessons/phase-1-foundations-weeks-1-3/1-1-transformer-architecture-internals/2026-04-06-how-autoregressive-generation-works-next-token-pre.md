<!-- Generated: 2026-04-06 -->
<!-- Phase: Phase 1: Foundations (Weeks 1-3) -->
<!-- Section: 1.1 — Transformer Architecture Internals -->



# How autoregressive generation works (next-token prediction loop)

## TL;DR
Autoregressive generation is a loop where a Transformer model predicts **one token at a time**, feeds that token back as input, and repeats until a stop condition is met. Each step re‑uses the same weights, but the model’s internal “memory” (the key‑value cache) grows so we don’t recompute everything from scratch. Understanding this loop is the foundation for every inference‑engineering trick—quantization, caching, speculative decoding, etc.—because they all try to make the loop faster or cheaper without changing the output distribution.

---

## ELI5 — The Simple Version  
Imagine you have a magical autocomplete keyboard that, after you type a word, whispers the *most likely* next word in your ear. You listen, type that word, and then ask the keyboard again for the next whisper. You keep doing this until you decide to stop (maybe you hit a period or reach a length limit).  

The keyboard isn’t memorizing whole sentences; it looks at **everything you’ve typed so far**, figures out patterns, and gives a probability distribution over the next word. In a Transformer, that “keyboard” is a stack of self‑attention and feed‑forward layers that turn the current token sequence into a set of logits (scores) for the vocabulary. The loop is:

```
[input tokens] → Transformer → logits → sample/pick next token → append → repeat
```

Because the model is *autoregressive*, each prediction depends only on the tokens that came before it—just like you can’t guess the next word without knowing the sentence so far.

---

## Why This Matters for Inference Engineering  
When you serve a large language model (LLM) to users, the generation loop is the **bottleneck**:

* **Latency** – each new token requires a forward pass through the model.  
* **Memory** – we must keep the intermediate keys and values (the KV‑cache) for every token already generated.  
* **Throughput** – we want to serve many requests at once without blowing up GPU memory.

All the papers we’ll discuss attack one or more of these pain points:

| Paper | What it speeds up / saves |
|-------|---------------------------|
| **FlashAttention** | Makes the *self‑attention* step inside each forward pass faster and memory‑efficient. |
| **GPTQ** | Shrinks the model’s weights (quantization) so each forward pass does less work and fits in less memory. |
| **Speculative Decoding** | Uses a cheap “draft” model to propose several tokens at once, then verifies them with the big model, cutting the number of serial passes. |
| **PagedAttention (vLLM)** | Stores the KV‑cache in OS‑style pages, reducing fragmentation and letting many requests share memory. |
| **Efficiently Scaling Transformer Inference (Google)** | Shows how to partition work across TPUs/GPUs and use multi‑query attention to handle very long contexts with low latency. |
| **Attention Is All You Need** | Gives us the basic transformer block that the loop repeatedly calls. |

Knowing the loop lets you apply these tricks in the right place (e.g., you quantize weights *once*, you cache KV *per request*, you run speculative decoding *inside* the loop).

---

## How It Actually Works  

### 1. The core transformer block (decoder‑only)

For a decoder‑only model like GPT, each layer has:

```
Input (seq_len x d_model) 
   ↓
Masked Multi‑Head Self‑Attention (causal)   ← can only see previous positions
   ↓
Add & Norm
   ↓
Position‑wise Feed‑Forward Network (FFN)
   ↓
Add & Norm
   ↓ Output (same shape)
```

The **mask** (a triangular mask) ensures that when computing attention for position *i*, the model cannot look at positions *j > i* (future tokens). This is what makes the generation *autoregressive*.

### 2. From text to logits  

1. **Tokenize** the prompt → list of token IDs `[t0, t1, …, t_{n-1}]`.  
2. **Embed** each ID → vector of size `d_model` (e.g., 768).  
3. **Add positional encoding** (so the model knows order).  
4. **Pass through N decoder layers** (self‑attention + FFN).  
5. **Final linear layer** (weight matrix `W_vocab` of size `d_model x vocab_size`) → logits for each token in the vocabulary.  
6. **Apply softmax** → probability distribution `P(next token | prompt)`.  
7. **Sample** (greedy, top‑k, nucleus, etc.) → pick token `t_n`.  
8. **Append** `t_n` to the input and repeat.

### 3. The KV‑cache – avoiding recomputation  

Self‑attention needs, for each head, a **Key** and **Value** matrix derived from every token seen so far. If we recomputed them from scratch each step, the cost would be O(seq_len²) per token. Instead we **cache**:

* After processing token `t_i`, we store its `K_i` and `V_i` (per head) in a cache.  
* When generating `t_{i+1}`, we only need to compute the **Query** for the new token and then attend to all cached keys/values.  
* The attention step becomes O(seq_len) per new token (linear in context length) because we just do a dot‑product with the cached keys.

```
Step 0 (prompt):   compute K0..K{n-1}, V0..V{n-1}   → cache
Step 1 (new token): compute Q_n, attend to [K0..K{n-1}] → logits → sample t_n
Step 2:            compute Q_{n+1}, attend to [K0..K{n}] → …
```

The cache grows linearly with the number of generated tokens, which is why long generations consume more memory.

### 4. Where the papers fit in  

| Paper | Where it helps in the loop | Plain‑language insight |
|-------|----------------------------|------------------------|
| **FlashAttention** | Inside each self‑attention step (both prompt processing and per‑token generation). | Instead of loading the huge QKᵀ matrix into slow GPU memory, it tiles the computation so data stays in fast on‑chip SRAM, cutting memory traffic and making attention ~2‑3× faster. |
| **GPTQ** | The weight matrices (`W_Q, W_K, W_V, W_O, W_FFN1, W_FFN2, W_vocab`) used in every layer. | It finds a near‑optimal 3‑ or 4‑bit representation for each weight using second‑order info, so the model is much smaller and each matrix‑vector multiply uses far less energy, with almost no loss in quality. |
| **Speculative Decoding** | The outer generation loop. | A tiny “draft” model proposes *k* tokens in parallel; the big model checks them in one forward pass, accepting the ones that match its distribution. This can give 2‑3× more tokens per second because we do fewer serial passes. |
| **PagedAttention (vLLM)** | Management of the KV‑cache across many concurrent requests. | Instead of allocating a contiguous block per request (which fragments), it splits the cache into fixed‑size pages (like OS virtual memory) and lets pages be shared when requests have common prefixes (e.g., same system prompt). This raises throughput 2‑4×. |
| **Efficiently Scaling Transformer Inference (Google)** | How to lay out the model and cache across TPU/GPU cores, and the use of multi‑query attention (MQA). | MQA lets many query heads share a single set of key/value heads, reducing cache size and enabling far longer contexts (up to 32×) while keeping latency low. Combined with smart partitioning, it hits ~29 ms/token on a 540B model. |
| **Attention Is All You Need** | The fundamental building block that the loop repeatedly calls. | Shows that stacking self‑attention + FFN layers, without recurrence, can learn complex language tasks and is highly parallelizable—making the loop possible at all. |

---

## Code You Can Run  

Below is a **minimal, self‑contained** decoder‑only Transformer in PyTorch (~120 lines) that demonstrates the autoregressive loop, KV‑caching, and sampling.  
You can copy‑paste it into a file (`mini_gpt.py`) and run it on your DGX Spark (it will use the GPU if available).

```python
# mini_gpt.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Helper utilities ----------
def get_pos_emb(seq_len, d_model, device):
    """Sinusoidal positional encoding (same as original paper)."""
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    dim = torch.arange(d_model, dtype=torch.float, device=device).unsqueeze(0)
    angle_rates = 1 / (10000 ** (2 * (dim // 2) / d_model))
    angle_rads = pos * angle_rates
    # apply sin to even indices, cos to odd
    pos_emb = torch.zeros_like(angle_rads)
    pos_emb[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pos_emb[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return pos_emb  # (seq_len, d_model)


# ---------- One Transformer decoder block ----------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # projections for Q, K, V (combined for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cache=None, start_pos=0):
        """
        x: (B, T, d_model)  – input for current step (usually T=1 during generation)
        cache: dict with keys 'k_cache', 'v_cache' each (B, n_heads, S, d_head)
               where S is the total sequence length seen so far (prompt + generated)
        start_pos: index in the cache where the current step's tokens begin
        Returns: y (B, T, d_model), updated cache
        """
        B, T, _ = x.shape
        # ---- Self-attention with causal mask ----
        x_ln = self.ln1(x)
        qkv = self.qkv_proj(x_ln)               # (B, T, 3*d_model)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, T, n_heads, d_head)

        # If we have a cache, append new k/v
        if cache is not None:
            k_cache = cache["k_cache"]
            v_cache = cache["v_cache"]
            k = torch.cat([k_cache, k], dim=2)   # (B, n_heads, S+T, d_head)
            v = torch.cat([v_cache, v], dim=2)
        # Update cache for next step
        new_cache = {"k_cache": k, "v_cache": v}

        # Transpose for batched matmul: (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, n_heads, T, S)
        # Causal mask: prevent looking at future positions
        S = k.size(-2)  # total length seen so far
        mask = torch.tril(torch.ones(T, S, device=x.device, dtype=torch.bool))
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)   # (B, n_heads, T, d_head)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)

        attn_output = self.out_proj(attn_output)
        x = x + self.dropout(attn_output)   # residual

        # ---- Feed‑forward ----
        ff = self.ffn(self.ln2(x))
        x = x + self.dropout(ff)

        return x, new_cache


# ---------- Minimal GPT‑like model ----------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(get_pos_emb(max_seq_len, d_model, torch.device("cpu")), requires_grad=False)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, cache=None):
        """
        idx: (B, T) token IDs
        cache: list of per-layer caches (or None for first pass)
        Returns: logits (B, T, vocab_size), updated cache list
        """
        B, T = idx.shape
        device = idx.device
        # token + positional embeddings
        x = self.token_emb(idx)                     # (B, T, d_model)
        # slice positional encoding to needed length
        pos = self.pos_emb[:T, :].to(device)        # (T, d_model)
        x = x + pos

        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_cache = layer(x, cache=layer_cache, start_pos=0)
            new_cache.append(layer_cache)

        x = self.ln_f(x)
        logits = self.head(x)                       # (B, T, vocab_size)
        return logits, new_cache

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Autoregressive generation loop with KV‑caching.
        prompt: list of token IDs (or tensor)
        Returns: list of generated token IDs (including prompt)
        """
        self.eval()
        device = next(self.parameters()).device
        if isinstance(prompt, list):
            prompt = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
        else:
            prompt = prompt.to(device)

        generated = prompt.clone()
        cache = None  # will hold per‑layer KV caches

        # Process the whole prompt in one go (no caching needed for the prompt itself,
        # but we still build the cache for fast generation afterwards)
        logits, cache = self.forward(generated, cache=None)
        # We only need the last token's logits to sample the next token
        next_token_logits = logits[:, -1, :] / temperature

        for _ in range(max_new_tokens):
            # Optional top‑k filtering
            if top_k is not None:
                top_vals, _ = torch.topk(next_token_logits, top_k)
                min_thresh = top_vals[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_thresh,
                    torch.tensor(-float("inf"), device=device),
                    next_token_logits,
                )
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1,1)

            # Append and run one step through the model (using cache)
            generated = torch.cat([generated, next_token], dim=1)
            logits, cache = self.forward(next_token, cache=cache)
            next_token_logits = logits[:, -1, :] / temperature

        return generated.squeeze(0).tolist()


# ---------- Demo ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 50257   # GPT‑2 tokenizer size (we’ll just use dummy IDs)
    model = MiniGPT(vocab_size, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_seq_len=128)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Simple prompt: token IDs for "Hello world"
    prompt_ids = [15496, 995]  # arbitrary IDs in GPT‑2 vocab
    out_ids = model.generate(prompt_ids, max_new_tokens=20, temperature=0.8, top_k=50)
    print("Generated token IDs:", out_ids)
    # To see actual text you’d need a tokenizer; here we just show IDs.
```

**What the code does**

1. **Builds** a tiny decoder‑only transformer (embedding → N decoder blocks → final linear head).  
2. **Forward pass** returns logits *and* updates a per‑layer KV‑cache (`cache`).  
3. **Generation loop**  
   * Processes the whole prompt once to fill the cache.  
   * Then, for each new token: only the latest token’s query is computed, attention attends to the cached keys/values, we get logits for the next token, sample, append, repeat.  
4. Uses **temperature** and optional **top‑k** sampling to illustrate how you control randomness.  

You can experiment: change `max_new_tokens`, try a larger model (`d_model=256, n_layers=6`), or swap in a real tokenizer from 🤗 HuggingFace to see actual words.

---

## Paper Breakdown (related to the generation loop)

| Paper | One‑line summary | The problem before | Key idea (plain language) | Impact |
|-------|------------------|--------------------|---------------------------|--------|
| **FlashAttention: Fast and Memory‑Efficient Exact Attention** (2022) | Makes the self‑attention operation faster and less memory‑hungry by being IO‑aware. | Standard attention materializes the huge QKᵀ matrix (O(seq_len²)) in GPU memory, causing slowdowns and OOM on long sequences. | Tile the Q, K, V matrices so that dot‑products are computed in chunks that stay in fast on‑chip SRAM, reducing reads/writes from HBM. The math stays exact; only the execution order changes. | Training speedup 15% on BERT‑large, 3× on GPT‑2 (1K seq), enables longer contexts (up to 64K) with better quality. |
| **GPTQ: Accurate Post‑Training Quantization for Generative Pre‑Trained Transformers** (2023) | Compresses LLM weights to 3‑4 bits with negligible loss using second‑order info. | Prior quantization (e.g., naive rounding) caused large accuracy drops, especially for huge models. | For each weight, compute the optimal quantized value considering the Hessian (second‑order) of the loss, then compensate errors in remaining weights (like optimal rounding). | Enables a 175B‑parameter model to run on a single GPU at 3‑4 bits/weight, cutting memory & energy ~4× while keeping perplexity almost unchanged. |
| **Fast Inference from Transformers via Speculative Decoding** (2023) | Uses a cheap draft model to propose multiple tokens, then verifies them in one pass with the target model. | Autoregressive generation is strictly serial: each token needs a full forward pass, limiting throughput. | Draft model (smaller/faster) runs ahead, generating a short sequence of candidate tokens; the big model evaluates the whole sequence in parallel and accepts tokens that match its distribution (using a specific sampling correction). | 2‑3× speedup on T5‑XXL with identical output distribution; no retraining needed. |
| **Attention Is All You Need** (2017) | Introduces the Transformer architecture based solely on self‑attention and feed‑forward layers. | Prior seq2seq models relied on RNNs/CNNs, which hindered parallelization and struggled with long‑range dependencies. | Replace recurrence with stacked self‑attention layers that let every position attend to every other position (via masking for autoregression) and add position‑wise FFNs. | Enabled massive parallel training, better translation quality (BLEU +2‑4), and became the foundation for all modern LLMs. |
| **Efficient Memory Management for LLM Serving with PagedAttention** (2023) – vLLM | Stores the KV‑cache in OS‑style pages, allowing sharing and reducing fragmentation. | Naïve KV‑cache allocation reserves a contiguous tensor per request; memory gets fragmented and wasted, limiting batch size. | Treat KV‑cache like virtual memory: allocate fixed‑size pages, map them to requests, and share pages when prefixes overlap (e.g., same system prompt). | Improves throughput 2‑4× at same latency, especially for long prompts and large models. |
| **Efficiently Scaling Transformer Inference** (2023) – Google | Shows how to partition work across TPUs/GPUs and use multi‑query attention (MQA) for low‑latency, long‑context serving. | Scaling to huge models (hundreds of billions) hits memory bandwidth limits and long context makes KV‑cache huge. | Use MQA (many query heads share a single set of key/value heads) to shrink cache size; combine with model‑parallel and data‑parallel partitioning tuned to TPU v4; int8 quantization further cuts memory. | Achieves 29 ms/token latency on a 540B model with 2048‑token context, 76% MFU at large batch. |

---

## Key Takeaways
- **Autoregressive generation = repeat:** feed current token sequence → transformer → logits → pick next token → append.  
- **KV‑cache is the secret sauce:** stores past keys/values so each step only computes a query and attends to the cache → O(seq_len) per token instead of O(seq_len²).  
- **FlashAttention speeds up the attention kernel** by reducing GPU memory traffic via tiling, keeping the math exact.  
- **GPTQ squeezes the model** into 3‑4 bit weights with almost no quality loss, cutting the compute and memory needed for every forward pass.  
- **Speculative Decoding** lets you generate several tokens at once with a tiny draft model, then verify them in one big‑model pass, breaking the serial bottleneck.  
- **PagedAttention (vLLM)** treats KV‑cache like virtual memory pages, enabling high‑throughput serving with minimal waste.  
- **Multi‑query attention + smart partitioning** (Google) lets you serve enormous models with long contexts at low latency.  
- Understanding the loop lets you apply these optimizations in the right place—weights, cache, or the outer generation loop.  

---

## What's Next
Having grasped how a single token is produced, we’ll look at **batching many generation streams together**, how **beam search** and **top‑p/nucleus sampling** change the loop, and then dive into **system‑level tricks** like continuous batching, paged attention, and speculative decoding in production serving stacks (e.g., vLLM, TensorRT‑LLM). Later we’ll cover **model compression** (quantization, pruning) and **hardware‑aware kernels** (FlashAttention‑2, Tensor cores) to squeeze every drop of performance out of your DGX Spark.  

Stay tuned—next we turn the “next‑token prediction loop” into a high‑throughput serving engine! 🚀

## Watch These Videos

- **[Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs)** (7:58)
- **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (1:56:20)
- **[Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://www.youtube.com/watch?v=bCz4OMemCcA)** (58:04)
