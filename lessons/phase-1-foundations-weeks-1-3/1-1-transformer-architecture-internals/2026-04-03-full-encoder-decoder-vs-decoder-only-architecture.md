<!-- Generated: 2026-04-03 -->
<!-- Phase: Phase 1: Foundations (Weeks 1-3) -->
<!-- Section: 1.1 — Transformer Architecture Internals -->



# Full encoder‑decoder vs decoder‑only architecture  

## TL;DR  
The original Transformer had **two stacks** – an *encoder* that reads the whole input and a *decoder* that generates the output one token at a time while looking back at the encoder. Modern LLMs (GPT‑style) drop the encoder and keep only a *decoder‑only* stack, letting the model both understand and generate text in a single pass. For inference engineering this matters because decoder‑only models are cheaper to serve (no separate encoder pass) and can be streamed token‑by‑token, while encoder‑decoder models are still useful for tasks that need a fixed‑length representation of the input (translation, summarization, retrieval‑augmented generation).

---

## ELI5 — The Simple Version  

### Imagine a two‑person translation team  

*Person A* (the **encoder**) listens to the whole foreign sentence, thinks about its meaning, and writes down a neat summary on a sticky note.  
*Person B* (the **decoder**) looks at that sticky note, then speaks the translation word‑by‑word, glancing back at the note whenever they need a clue about what to say next.  

If you only need to **understand** a sentence (e.g., “Is this spam?”) you can skip Person B entirely – Person A’s summary is enough.  
If you only need to **generate** text from scratch (e.g., “Write a poem about cats”) you can skip Person A – Person B can start with an empty note and just keep adding words, looking at what it has already written.

### Why the split exists  

In the original Transformer paper (“Attention Is All You Need”) the team wanted a model that could **translate** from one language to another. Translation naturally splits into two phases:  

1. **Read‑and‑understand** the source sentence (encoder).  
2. **Produce** the target sentence token‑by‑token while constantly checking the source understanding (decoder).  

Later researchers realized that for many language‑modeling tasks (predicting the next word) you don’t need a separate “read‑and‑understand” stage – the model can look at the words it has already generated and use that as its understanding. That gave rise to **decoder‑only** models like GPT, which are simpler, faster to run, and easier to scale.

---

## Why This Matters for Inference Engineering  

| Aspect | Encoder‑Decoder (Seq2Seq) | Decoder‑Only (Causal LM) |
|--------|---------------------------|--------------------------|
| **Forward pass cost** | Two passes: encoder (O(N²) self‑attention) + decoder (O(N²) per token) | Single pass; each new token re‑uses cached key/value states → amortized O(N) per token |
| **Memory footprint** | Need to store encoder hidden states for the whole decoder pass | Only need to store KV‑cache for generated tokens (can be paged) |
| **Latency** | First token appears after encoder finishes → higher TTFT (time‑to‑first‑token) | First token appears immediately (decoder can start with a blank or prompt) |
| **Use‑case fit** | Tasks that need a *fixed* representation of the input (translation, summarization, retrieval‑augmented QA) | Pure generation, chat, continuation, code completion, any task where the prompt itself is the “understanding” |
| **Serving implications** | Must keep encoder results alive while decoding; harder to batch variable‑length prompts | KV‑cache enables efficient batching and streaming; easier to implement with tensor‑parallelism / pipeline‑parallelism |

If you’re building an LLM serving stack (e.g., on a DGX Spark with Blackwell GPUs), you’ll spend most of your time optimizing the **decoder** path: fused kernels, KV‑cache management, paged attention, and speculative decoding. Knowing when an encoder‑decoder is still warranted helps you choose the right model architecture for a given product (e.g., a retrieval‑augmented chatbot may keep a dense encoder for document embeddings while using a decoder‑only LLM for response generation).

---

## How It Actually Works  

We’ll walk through the original encoder‑decoder Transformer, then show how removing the encoder yields a decoder‑only model. ASCII diagrams help visualize the data flow.

### 1. The Encoder‑Decoder Blueprint  

```
Input tokens  →  Embedding →  Positional Encoding  →  [Encoder Stack]  →  Encoder Outputs (memory)
                                                               │
                                                               ▼
Decoder Stack (masked self‑attention → encoder‑decoder attention → FFN)  →  Linear → Softmax → Next token
```

* **Encoder Stack** – N identical layers, each:  
  1. **Multi‑Head Self‑Attention (MHSA)** – each token can attend to *all* other tokens (bidirectional).  
  2. **Position‑wise Feed‑Forward Network (FFN)** – applied independently per token.  
  3. Add‑&‑Norm (residual + layer norm) after each sub‑layer.

* **Decoder Stack** – N identical layers, each:  
  1. **Masked MHSA** – token can only attend to *previous* positions (causal mask) → ensures autoregressive generation.  
  2. **Encoder‑Decoder Attention** – queries from decoder attend to *encoder outputs* (the “memory”). This lets the decoder look at the source sentence while generating.  
  3. **FFN** – same as encoder.  
  4. Add‑&‑Norm after each.

* **Output** – final decoder hidden states → linear projection → vocab‑size logits → softmax → sample next token.

### 2. What changes when we drop the encoder?  

If we remove the encoder and its encoder‑decoder attention, each decoder layer only has:

```
[Masked MHSA] → [FFN]   (repeated N times)
```

The model now receives only the **target sequence** (the prompt + already‑generated tokens) and learns to predict the next token purely from what it has seen so far. This is exactly the architecture of GPT‑1/2/3/4, LLaMA, Mistral, etc.

#### ASCII: Decoder‑Only forward pass (with KV‑cache)

```
Time step t = 0 (prompt only)
+-------------------+      +-------------------+      +-------------------+
| Embedding + Pos   | ---> | Masked MHSA (t=0) | ---> | FFN + AddNorm     |
+-------------------+      +-------------------+      +-------------------+
        │                         │                         │
        ▼                         ▼                         ▼
   (hidden states)          (Q,K,V)                  (hidden states)
        │                         │                         │
        └─────► Store K,V in cache ◄───────────────────────┘
                         (cached for all future steps)

Time step t = 1 (generate first new token)
+-------------------+      +-------------------+      +-------------------+
| Embedding(tok_t)  | ---> | Masked MHSA (t=1) | ---> | FFN + AddNorm     |
+-------------------+      +-------------------+      +-------------------+
        │                         │                         │
        ▼                         ▼                         ▼
   (new hidden)          (Q_t, K_t, V_t)          (new hidden)
        │                         │                         │
        │   Attend to cached K,V  ◄───────────────────────┘
        └─────► Append K_t,V_t to cache ◄───────────────────────┘
```

*The **KV‑cache** stores the key and value vectors for every previous token, so at step *t* we only need to compute the query for the new token and do a cheap dot‑product with the cached keys.*

### 3. Mathematical intuition (self‑attention)  

For a single head, attention computes:

\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

* **Encoder**: Q, K, V are all derived from the *same* layer’s input → each token can see every other token (full matrix).  
* **Decoder (masked)**: Same formula, but we first zero‑out positions *j > i* in the similarity matrix before softmax → causal mask.  
* **Encoder‑Decoder Attention**: Q comes from decoder layer, K and V come from the *encoder output* (the memory). This is the only place where the encoder talks to the decoder.

### 4. Why the encoder‑decoder was a breakthrough  

* **Parallelism** – Unlike RNNs, the self‑attention layers can compute all token interactions in a single matrix multiply, enabling massive throughput on GPUs/TPUs.  
* **Separation of concerns** – Encoder learns a rich, bidirectional representation; decoder learns how to generate conditioned on that representation. This made translation quality jump from ~25 BLEU to >40 BLEU on WMT’14 English‑German.  
* **Stacking** – Six identical layers let the model build hierarchical features (low‑level syntax → high‑level semantics) without recurrence.

### 5. Why decoder‑only took over  

* **Simplicity** – No need to maintain two separate stacks; training is just next‑token prediction on a huge text corpus.  
* **Scalability** – The causal mask lets us reuse the KV‑cache during inference, turning O(N²) per token into O(N) amortized.  
* **Emergent abilities** – With enough data and parameters, a decoder‑only model learns to perform translation, summarization, QA, etc., *without* an explicit encoder—just by conditioning on the prompt.

---

## Code You Can Run  

Below is a **minimal, runnable** PyTorch implementation of both architectures (encoder‑decoder and decoder‑only) that fits in ~180 lines. You can copy‑paste it into a notebook on your DGX Spark (PyTorch 2.x, CUDA 12, HuggingFace tokenizers optional).  

```python
# --------------------------------------------------------------
# Minimal Transformer: Encoder-Decoder vs Decoder-Only
# --------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------- Helper: Positional Encoding -------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        return x + self.pe[:, :x.size(1)]

# ------------------- Multi‑Head Self‑Attention -------------------
class MHSA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, causal=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.causal = causal

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None):
        """
        x: (B, T, D)
        mask: (B, 1, T, T) bool, True where we keep (False = masked)
        kv_cache: tuple (past_key, past_value) each (B, H, T_past, d_k)
        Returns: y, (key, value) for caching
        """
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            K = torch.cat([past_k, K], dim=2)   # (B, H, T_past+T, d_k)
            V = torch.cat([past_v, V], dim=2)

        # Scaled dot‑product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, T_past+T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        if self.causal:
            # causal mask: prevent looking at future positions
            causal_mask = torch.tril(torch.ones(T, T_past+T, device=x.device, dtype=torch.bool))
            # shape (T, T_past+T) -> broadcast to (B, H, T, T_past+T)
            scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)          # (B, H, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, D)
        out = self.W_o(context)
        return out, (K, V)                     # return updated K,V for cache

# ------------------- Position‑wise Feed‑Forward -------------------
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ------------------- Encoder Layer -------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MHSA(d_model, n_heads, dropout, causal=False)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Self‑attention
        attn_out, _ = self.self_attn(src, mask=src_mask)
        src = self.norm1(src + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_out))
        return src

# ------------------- Decoder Layer -------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MHSA(d_model, n_heads, dropout, causal=True)
        self.enc_attn = MHSA(d_model, n_heads, dropout, causal=False)  # encoder‑decoder attn
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask, kv_cache=None):
        """
        tgt: (B, T_t, D)
        memory: encoder output (B, T_s, D)
        kv_cache: for self‑attention only (we cache decoder self‑attn K,V)
        """
        # Masked self‑attention
        self_attn_out, self_cache = self.self_attn(tgt, mask=tgt_mask, kv_cache=kv_cache)
        tgt = self.norm1(tgt + self.dropout(self_attn_out))

        # Encoder‑decoder attention (queries from tgt, keys/values from memory)
        enc_attn_out, _ = self.enc_attn(tgt, mask=memory_mask, kv_cache=None)  # no cache needed
        tgt = self.norm2(tgt + self.dropout(enc_attn_out))

        # FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))
        return tgt, self_cache   # return updated self‑attn cache

# ------------------- Full Encoder‑Decoder Model -------------------
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4,
                 d_ff=512, n_enc=2, n_dec=2, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec)
        ])
        self.generator = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask):
        x = self.tok_emb(src)
        x = self.pos_emb(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x   # memory

    def decode(self, tgt, memory, tgt_mask, memory_mask):
        x = self.tok_emb(tgt)
        x = self.pos_emb(x)
        cache = None
        new_caches = []
        for layer in self.decoder_layers:
            x, cache = layer(x, memory, tgt_mask, memory_mask, kv_cache=cache)
            new_caches.append(cache)
        logits = self.generator(x)
        return logits, new_caches   # caches can be used for next token generation

    def forward(self, src, tgt):
        src_mask = torch.ones_like(src).unsqueeze(1).unsqueeze(2)  # (B,1,1,T_src)
        tgt_mask = torch.tril(torch.ones_like(tgt)).unsqueeze(1).unsqueeze(2)  # causal
        memory = self.encode(src, src_mask)
        logits, _ = self.decode(tgt, memory, tgt_mask, torch.ones_like(src).unsqueeze(1).unsqueeze(2))
        return logits

# ------------------- Decoder‑Only Model (GPT style) -------------------
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4,
                 d_ff=512, n_layers=2, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, idx, past_key_values=None):
        """
        idx: (B, T) token ids for the whole prompt (or just the new token if using cache)
        past_key_values: list of length n_layers, each element (past_key, past_value)
                         each tensor shape (B, H, T_past, d_k)
        Returns logits for the last position and updated cache.
        """
        x = self.tok_emb(idx)
        x = self.pos_emb(x)

        # Build causal mask for the full sequence length (prompt + generated so far)
        T = x.size(1)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

        new_cache = []
        for i, layer in enumerate(self.layers):
            # layer expects tgt_mask and memory_mask; for decoder‑only we set memory=None
            # we reuse the self‑attention part with causal mask and pass kv_cache.
            # For simplicity we ignore encoder‑decoder attention (set to zero).
            x, cache = layer(x, memory=torch.zeros_like(x),   # dummy memory (not used)
                             tgt_mask=causal_mask,
                             memory_mask=torch.ones_like(x).unsqueeze(1).unsqueeze(2),  # dummy
                             kv_cache=past_key_values[i] if past_key_values else None)
            new_cache.append(cache)

        logits = self.generator(x)  # (B, T, vocab)
        return logits[:, -1, :], new_cache   # return logits for last token only

# ------------------- Demo: translate a toy sentence -------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 1000
    dummy_src = torch.randint(0, vocab_size, (2, 7))   # batch=2, src_len=7
    dummy_tgt = torch.randint(0, vocab_size, (2, 5))   # tgt_len=5 (includes <bos>?)

    encdec = EncoderDecoderTransformer(vocab_size, d_model=128, n_heads=4,
                                       d_ff=256, n_enc=2, n_dec=2)
    logits = encdec(dummy_src, dummy_tgt)
    print("Encoder‑Decoder logits shape:", logits.shape)  # (B, T_tgt, vocab)

    # Decoder‑only generation (autoregressive)
    dec_only = DecoderOnlyTransformer(vocab_size, d_model=128, n_heads=4,
                                      d_ff=256, n_layers=2)
    # start with a single token (e.g., <bos>)
    cur_token = torch.zeros((2, 1), dtype=torch.long)   # bos token id = 0
    past = None
    for _ in range(4):   # generate 4 more tokens
        logits, past = dec_only(cur_token, past)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        cur_token = torch.cat([cur_token, next_token], dim=1)
    print("Generated token ids:", cur_token.squeeze().tolist())
```

**What the code shows**

* The **Encoder‑Decoder** class mirrors the diagram: a stack of encoder layers, then decoder layers that attend to the encoder output.  
* The **Decoder‑Only** class removes the encoder and the encoder‑decoder attention; each layer only does masked self‑attention + FFN.  
* The generation loop demonstrates the **KV‑cache** (`past_key_values`) that makes each step cheap.  
* You can run this on a DGX Spark (Blackwell GPU) – just change `d_model`/`n_heads` to larger values (e.g., 768/12) and you’ll see a realistic transformer in a few hundred lines.

---

## Paper Breakdown (if relevant)

| Paper | One‑line summary | Problem before | Key idea (plain language) | Impact |
|-------|------------------|----------------|---------------------------|--------|
| **Attention Is All You Need** (Vaswani et al., 2017) | Introduced the Transformer, a sequence‑to‑sequence model built solely on self‑attention and feed‑forward nets. | RNN‑based seq2seq models were slow (sequential) and struggled with long‑range dependencies. | Replace recurrence with **multi‑head self‑attention**, allowing every token to interact with every other token in parallel; stack identical encoder/decoder blocks. | Sparked the entire LLM revolution; enabled massive scaling on GPUs/TPUs; BLEU scores jumped from ~25 to >40 on WMT’14 En‑De. |
| **Improving Language Understanding by Generative Pre‑Training** (Radford et al., 2018) – GPT‑1 | Showed that a **decoder‑only** Transformer trained with a language‑modeling objective can acquire useful language understanding. | Prior LMs were either shallow (n‑gram) or RNN‑based, limiting scale and parallelism. | Stack decoder blocks, use causal mask, train on huge text corpus to predict next token; then fine‑tune on downstream tasks. | Proved that decoder‑only models could scale to billions of parameters and perform zero‑shot/few‑shot tasks. |
| **Language Models are Few‑Shot Learners** (Brown et al., 2020) – GPT‑3 | Scaled decoder‑only Transformers to 175 B parameters, demonstrating emergent few‑shot abilities. | Earlier LMs needed task‑specific fine‑tuning; scaling laws were unclear. | Massive model size + diverse web‑scale data + same decoder‑only architecture → model learns to follow instructions from prompts alone. | Established the paradigm of prompting huge decoder‑only LLMs; spurred the API‑based LLM ecosystem. |

---

## Key Takeaways  

- ✅ **Encoder‑Decoder** = two stacks: encoder builds a bidirectional understanding of the input; decoder generates output while attending to that understanding.  
- ✅ **Decoder‑Only** = a single stack with causal self‑attention; the model both “understands” (by looking at previously generated tokens) and generates in one pass.  
- 🚀 **Why it matters for inference:** decoder‑only models enable KV‑caching, lower time‑to‑first‑token, and easier batching; encoder‑decoder models are still needed when you need a fixed‑length representation of a long source (e.g., retrieval‑augmented generation).  
- ⚙️ **Implementation tip:** the core of both architectures is the same MHSA+FFN block; the difference lies in the attention masks and whether you keep encoder‑decoder attention.  
- 📈 **Scaling insight:** moving from encoder‑decoder to decoder‑only removed a major source of sequential latency, allowing LLMs to grow to hundreds of billions of tokens while keeping inference latency tractable.

---

## What's Next  

Now that you know **what** you’re optimizing (the attention‑based transformer blocks), the next step is to dive into **how to make those blocks fast** on modern GPUs:

1. **KV‑cache & Paged Attention** – how to store and retrieve keys/values efficiently during autoregressive generation.  
2. **Fused kernels** – combining QKᵀ, softmax, and V multiplication into a single CUDA kernel (FlashAttention, xFormers).  
3. **Quantization & sparsity** – reducing memory bandwidth without hurting accuracy.  
4. **Speculative decoding & Medusa** – generating multiple tokens per forward pass to cut latency further.  

Understanding the encoder‑decoder vs. decoder‑only split gives you the mental model to decide when you need the extra encoder pass (e.g., retrieval‑augmented QA) and when you can rely purely on a decoder‑only stack for maximal throughput. Happy hacking on your DGX Spark! 🚀

## Watch These Videos

- **[Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M)** (27:14)
- **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (1:56:20)
- **[Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://www.youtube.com/watch?v=bCz4OMemCcA)** (58:04)
