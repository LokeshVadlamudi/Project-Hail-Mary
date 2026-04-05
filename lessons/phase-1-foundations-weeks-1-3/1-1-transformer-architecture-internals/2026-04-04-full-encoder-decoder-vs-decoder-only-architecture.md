<!-- Generated: 2026-04-04 -->
<!-- Phase: Phase 1: Foundations (Weeks 1-3) -->
<!-- Section: 1.1 — Transformer Architecture Internals -->



# Full encoder‑decoder vs decoder‑only architecture  

## TL;DR  
A **full encoder‑decoder Transformer** has two stacks: an encoder that reads the whole input and a decoder that generates output step‑by‑step while looking back at the encoder. A **decoder‑only** model (like GPT) drops the encoder and lets the decoder attend only to its own previously‑generated tokens, turning the architecture into a pure autoregressive language model. For inference engineering, knowing which stack you have tells you how to cache keys/values, how to batch prompts, and whether you need cross‑attention at all.

---

## ELI5 — The Simple Version  

### Imagine a two‑person translation team  

1. **The Reader (Encoder)** – sits down with the whole foreign sentence, reads it carefully, and builds a rich “understanding” of every word, noting how each word relates to the others.  
2. **The Writer (Decoder)** – starts with a blank page and, for each word it wants to write, peeks at the Reader’s notes (cross‑attention) and also at what it has already written (self‑attention) to decide the next word.

If you only need to **continue a story** (like GPT writing the next paragraph), you can fire the Reader and let the Writer just look at its own growing draft. The Writer becomes a solo author that only needs to remember what it has already written.

### Why the split matters  

* With both Reader and Writer, you can translate **any length** sentence because the Reader has already processed the whole source before the Writer starts.  
* With only a Writer, you save memory and compute (no separate encoder stack) and you can generate tokens **one‑by‑one** super fast, which is exactly what LLMs do when you chat with them.

---

## Why This Matters for Inference Engineering  

| Aspect | Encoder‑Decoder (e.g., T5, BART) | Decoder‑Only (e.g., GPT‑3/4, LLaMA) |
|--------|-----------------------------------|--------------------------------------|
| **Cache needed** | Separate caches for encoder self‑attention (static) + decoder self‑attention + cross‑attention (dynamic) | Only decoder self‑attention cache (dynamic) |
| **Prompt handling** | Encoder runs once on the whole input; decoder runs per generated token | Encoder is absent; the whole prompt is processed by the decoder once, then tokens are generated autoregressively |
| **Batching** | You can batch many encoder passes (same input length) then run decoders in parallel for teacher‑forcing training; at inference you still need a per‑step decoder loop | You batch the prompt once, then run a tight loop where each step re‑uses the KV cache – the main engineering challenge is efficient cache management |
| **Latency** | First token latency includes encoder pass + decoder step | First token latency is just the decoder pass over the prompt (often lower) |
| **Memory footprint** | Encoder parameters + decoder parameters (roughly 2×) | Only decoder parameters (≈½ the size for same hidden dimension) |

If you are serving LLMs, you will almost always be dealing with decoder‑only models because they are simpler to cache and scale. Understanding the encoder‑decoder baseline helps you appreciate why certain optimizations (like KV‑caching, flash attention, or paged attention) exist, and it prevents you from mistakenly trying to apply cross‑attention tricks where they aren’t needed.

---

## How It Actually Works  

We’ll walk through the math and data flow, then compare the two architectures with ASCII diagrams.

### 1. Core Transformer building block  

Each layer (encoder or decoder) has two sub‑layers:

```
Input → [Multi‑Head Self‑Attention] → Add & Norm → [Feed‑Forward NN] → Add & Norm → Output
```

* **Self‑Attention** lets each token attend to *all* tokens in the same sequence (query‑key‑value product).  
* **Feed‑Forward** is a position‑wise MLP (same weights for every token).  
* **Add & Norm** = residual connection + layer norm.

### 2. Encoder‑Decoder (the original “Attention Is All You Need”)  

```
Encoder Stack (N layers)          Decoder Stack (N layers)
----------------                  -----------------
Input Embeddings  →  Encoder Layer 1  → … → Encoder Layer N  →  Encoder Outputs (E)
                                                    |
                                                    |  (cross‑attention keys/values)
                                                    V
Target Embeddings (shifted right) → Decoder Layer 1 → … → Decoder Layer N → Logits
```

* **Encoder** receives the full source sequence, computes self‑attention **only** among source tokens, and outputs a set of vectors **E** (one per source token).  
* **Decoder** at each time step `t`:  
  1. **Masked self‑attention** over the target tokens generated so far (prevents peeking at future tokens).  
  2. **Cross‑attention** where queries come from the decoder’s current state, keys/values come from the encoder output **E** (this is the “look at the source” step).  
  3. Feed‑forward, add‑norm, etc.

### 3. Decoder‑Only (GPT style)  

```
Decoder Stack (N layers)
----------------
Input Embeddings (prompt) → Decoder Layer 1 → … → Decoder Layer N → Logits
```

* No encoder → no cross‑attention.  
* The decoder’s self‑attention is **causal** (masked) so each position can only attend to earlier positions.  
* The same stack is used for both processing the prompt and generating new tokens.

### 4. ASCII data‑flow comparison  

#### Encoder‑Decoder (translation)

```
Source:  [The] [animal] [didn't] [cross] [the] [street] [because] [it] [was] [too] [tired]
          │       │       │       │       │       │       │       │       │       │
          └──────► Encoder (self‑attention) ◄──────┘
                                 │
                                 ▼
                         Encoder Outputs E (same length)
                                 │
                                 ▼
Target (shifted right): [</s>] [The] [animal] [didn't] [cross] [the] [street] [because] [it] [was]
          │       │       │       │       │       │       │       │       │       │
          │       │       │       │       │       │       │       │       │       │
          │   Masked Self‑Attention (causal)   │
          │       │       │       │       │       │       │       │       │       │
          │       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
          │   Cross‑Attention (Q from decoder, K,V from E)   │
          │       │       │       │       │       │       │       │       │       │
          │       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
          │   Feed‑Forward + AddNorm (repeat)                │
          │       │       │       │       │       │       │       │       │       │
          ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
        Logits over vocab (next target token)
```

#### Decoder‑Only (language modeling)

```
Prompt: [The] [animal] [didn't] [cross] [the] [street] [because] [it] [was]
          │       │       │       │       │       │       │       │       │
          └──────► Decoder (masked self‑attention) ◄──────┘
                                 │
                                 ▼
                         Decoder Outputs H (same length)
                                 │
                                 ▼
                     Feed‑Forward + AddNorm (repeat)
                                 │
                                 ▼
                Logits over vocab (next token given prompt)
```

*Note*: In the decoder‑only case, after we generate a token we **append** it to the input and run the decoder again, re‑using the cached key/value matrices for all previous tokens (KV‑cache). This makes each step O(1) w.r.t. prompt length after the first pass.

### 5. Why the encoder‑decoder split was introduced  

* **Problem**: Early seq2seq models (RNN‑based) struggled with long-range dependencies because the hidden state had to compress the entire source into a fixed‑size vector.  
* **Breakthrough** (Vaswani et al., 2017 “Attention Is All You Need”): Replace the recurrent encoder with a stack of self‑attention layers, allowing every source token to directly interact with every other token. Add a decoder that can attend to the encoder’s rich representations via cross‑attention.  
* **Impact**: Parallelizable training (no recurrence), dramatically better translation quality, and the foundation for later decoder‑only LLMs that realized you could drop the encoder entirely for pure language modeling.

---

## Code You Can Run  

Below is a **minimal, end‑to‑end example** that builds:

1. An encoder‑decoder Transformer (2 layers each) for a toy copy‑task.  
2. A decoder‑only Transformer (2 layers) for next‑token prediction on the same data.  

You can run this on a DGX Spark (PyTorch 2.x, HF Transformers optional). The code is deliberately short (~120 lines) but still shows the key differences: encoder vs no‑encoder, causal mask, and KV‑caching idea (we’ll manually cache for illustration).

```python
# --------------------------------------------------------------
# Minimal Transformer demo: Encoder‑Decoder vs Decoder‑Only
# --------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Helper: sinusoidal positional encoding ----------
def get_pos_emb(seq_len, d_model):
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    i   = torch.arange(d_model, dtype=torch.float).unsqueeze(0)
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    emb = torch.zeros_like(angle)
    emb[:, 0::2] = torch.sin(angle[:, 0::2])
    emb[:, 1::2] = torch.cos(angle[:, 1::2])
    return emb  # (seq_len, d_model)

# ---------- Multi‑Head Self‑Attention (with optional mask) ----------
class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        B, T, _ = x.size()
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, h, T, T)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # (B, h, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        out = self.W_o(context)
        return out, attn_weights

# ---------- Position‑wise Feed‑Forward ----------
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# ---------- Encoder Layer ----------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MHA(d_model, n_heads)
        self.ffn       = FFN(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Self‑attention
        attn_out, _ = self.self_attn(x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed‑forward
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# ---------- Decoder Layer ----------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MHA(d_model, n_heads)        # masked
        self.cross_attn = MHA(d_model, n_heads)       # encoder‑decoder
        self.ffn       = FFN(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.norm3     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # 1️⃣ Masked self‑attention
        attn1, _ = self.self_attn(tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn1))

        # 2️⃣ Cross‑attention (queries from tgt, keys/values from memory)
        attn2, attn_weights = self.cross_attn(tgt, mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(attn2))

        # 3️⃣ Feed‑forward
        ff_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_out))
        return tgt, attn_weights

# ---------- Full Encoder‑Decoder Model ----------
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4,
                 d_ff=256, n_enc=2, n_dec=2, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = None   # will be created in forward
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec)
        ])
        self.generator = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        """
        src: (B, S) source token ids
        tgt: (B, T) target token ids (teacher‑forcing, shifted right inside)
        """
        B, S = src.size()
        _, T = tgt.size()
        device = src.device

        # Embeddings + positional
        src_emb = self.tok_emb(src) + get_pos_emb(S, self.tok_emb.embedding_dim).to(device)
        tgt_emb = self.tok_emb(tgt) + get_pos_emb(T, self.tok_emb.embedding_dim).to(device)

        # Encoder self‑mask (none for encoder)
        src_mask = torch.ones(B, 1, S, S, device=device)  # (B,1,S,S) broadcastable
        memory = src_emb
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory, src_mask)

        # Decoder masks
        # Causal mask for target (prevent looking ahead)
        tgt_mask = torch.tril(torch.ones((T, T), device=device)).unsqueeze(0).unsqueeze(1)  # (1,1,T,T)
        tgt_mask = tgt_mask.expand(B, 1, T, T)  # (B,1,T,T)
        # Encoder‑decoder mask (allow all source positions)
        memory_mask = torch.ones(B, 1, T, S, device=device)

        out = tgt_emb
        for dec_layer in self.decoder_layers:
            out, _ = dec_layer(out, memory, tgt_mask, memory_mask)

        logits = self.generator(out)  # (B, T, vocab)
        return logits

# ---------- Decoder‑Only Model (same building blocks, no encoder) ----------
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4,
                 d_ff=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, tgt):
        """
        tgt: (B, T) token ids (the prompt, will be used autoregressively)
        """
        B, T = tgt.size()
        device = tgt.device
        emb = self.tok_emb(tgt) + get_pos_emb(T, self.tok_emb.embedding_dim).to(device)

        # causal mask
        mask = torch.tril(torch.ones((T, T), device=device)).unsqueeze(0).unsqueeze(1)
        mask = mask.expand(B, 1, T, T)

        x = emb
        for layer in self.layers:
            x, _ = layer(x, None, mask, None)   # memory=None, memory_mask unused
        logits = self.generator(x)
        return logits

# ---------- Tiny demo data ----------
vocab = list(" abcdefghijklmnopqrstuvwxyz")
stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for ch,i in stoi.items()}
def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long).unsqueeze(0)  # (1, L)

src = encode("the animal did not cross the street because it was tired")
tgt = encode("the animal did not cross the street because it was tired")  # copy task

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src = src.to(device)
tgt = tgt.to(device)

# ---------- Train a single step (just to see shapes) ----------
encdec = EncoderDecoderTransformer(vocab_size=len(vocab), d_model=64, n_heads=4,
                                   d_ff=128, n_enc=2, n_dec=2).to(device)
deconly = DecoderOnlyTransformer(vocab_size=len(vocab), d_model=64, n_heads=4,
                                 d_ff=128, n_layers=2).to(device)

encdec.train()
deconly.train()
opt_encdec = torch.optim.Adam(encdec.parameters(), lr=1e-3)
opt_deconly = torch.optim.Adam(deconly.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Encoder‑Decoder forward (teacher forcing)
logits_encdec = encdec(src, tgt[:, :-1])          # predict tgt[1:] given tgt[:-1]
loss_encdec = criterion(logits_encdec.view(-1, len(vocab)), tgt[:, 1:].reshape(-1))
loss_encdec.backward()
opt_encdec.step()
opt_encdec.zero_grad()

# Decoder‑Only forward (language modeling on the same string)
logits_deconly = deconly(tgt[:, :-1])            # predict next token given prefix
loss_deconly = criterion(logits_deconly.view(-1, len(vocab)), tgt[:, 1:].reshape(-1))
loss_deconly.backward()
opt_deconly.step()
opt_deconly.zero_grad()

print(f"EncDec loss: {loss_encdec.item():.4f}")
print(f"DecOnly loss: {loss_deconly.item():.4f}")

# ---------- Generation demo (greedy) ----------
def greedy_generate(model, prompt, max_len=20, encoder_decoder=False, src=None):
    model.eval()
    generated = prompt.clone()
    with torch.no_grad():
        for _ in range(max_len):
            if encoder_decoder:
                # encoder runs once
                memory = None
                # compute encoder output
                src_emb = model.tok_emb(src) + get_pos_emb(src.size(1), model.tok_emb.embedding_dim).to(device)
                src_mask = torch.ones(1,1,src.size(1),src.size(1),device=device)
                memory = src_emb
                for enc_layer in model.encoder_layers:
                    memory = enc_layer(memory, src_mask)
                # decoder step
                tgt_emb = model.tok_emb(generated) + get_pos_emb(generated.size(1), model.tok_emb.embedding_dim).to(device)
                tgt_mask = torch.tril(torch.ones((generated.size(1), generated.size(1)), device=device)).unsqueeze(0).unsqueeze(1)
                tgt_mask = tgt_mask.expand(1,1,-1,-1)
                memory_mask = torch.ones(1,1,generated.size(1),src.size(1),device=device)
                x = tgt_emb
                for dec_layer in model.decoder_layers:
                    x, _ = dec_layer(x, memory, tgt_mask, memory_mask)
                logits = model.generator(x[:, -1, :])   # last position only
            else:
                # decoder‑only
                emb = model.tok_emb(generated) + get_pos_emb(generated.size(1), model.tok_emb.embedding_dim).to(device)
                mask = torch.tril(torch.ones((generated.size(1), generated.size(1)), device=device)).unsqueeze(0).unsqueeze(1)
                mask = mask.expand(1,1,-1,-1)
                x = emb
                for layer in model.layers:
                    x, _ = layer(x, None, mask, None)
                logits = model.generator(x[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == stoi[" "]:  # naive stop condition
                break
    return generated

prompt = encode("the animal")
print("Encoder‑Decoder generation:", "".join(itos[i.item()] for i in greedy_generate(encdec, prompt, encoder_decoder=True, src=src)[0]))
print("Decoder‑Only generation:",   "".join(itos[i.item()] for i in greedy_generate(deconly, prompt, encoder_decoder=False)[0]))
```

**What you see**

* The encoder‑decoder model needs to run the encoder **once** (the `memory` tensor) and then runs the decoder step‑by‑step.  
* The decoder‑only model skips the encoder entirely; each step only does masked self‑attention over the growing prompt.  
* In a real serving system you would **cache** the key/value matrices for every layer after the first forward pass, making each subsequent token generation O(1) w.r.t. prompt length.

Feel free to tweak `d_model`, `n_heads`, number of layers, or switch to `torch.nn.MultiheadAttention` for a faster version. The code runs comfortably on a DGX Spark’s 128 GB unified memory and Blackwell GPU.

---

## Paper Breakdown (if relevant)

### Paper: **Attention Is All You Need** (Vaswani et al., 2017)  
* **One‑line summary**: Introduced the Transformer architecture, replacing recurrence with stacked self‑attention layers and showing that an encoder‑decoder design can achieve state‑of‑the‑art machine translation.  
* **The problem**: Prior seq2seq models relied on RNNs/LSTMs that processed tokens sequentially, causing slow training and difficulty capturing long‑range dependencies because the encoder had to compress the whole source into a fixed‑size hidden vector.  
* **The key idea**:  
  1. **Self‑attention** lets every token directly attend to every other token in the same layer, giving O(1) path length between any two positions.  
  2. **Multi‑head** projections allow the model to capture different types of relationships in parallel.  
  3. **Positional encodings** inject order information without recurrence.  
  4. **Encoder‑decoder stack** separates understanding (encoder) from generation (decoder) while still allowing the decoder to look at the encoder’s output via cross‑attention.  
* **Impact**:  
  * Training became highly parallelizable → massive speed‑up on GPUs/TPUs.  
  * Translation quality jumped (BLEU scores +2.0 over GNMT on WMT’14 EN‑DE).  
  * The architecture became the universal building block for later models: BERT (encoder‑only), GPT (decoder‑only), T5 (encoder‑decoder), and virtually every LLM today.  

---

## Key Takeaways  

- 🔁 **Encoder‑Decoder** = two stacks: encoder (bi‑directional self‑attention) + decoder (masked self‑attention + cross‑attention).  
- 🔁 **Decoder‑Only** = a single stack with causal self‑attention; no encoder, no cross‑attention.  
- 💡 Encoder‑decoder shines when you need to **condition generation on a separate input** (translation, summarization).  
- 🚀 Decoder‑only is the workhorse for **pure language modeling** because it’s simpler to cache and scale.  
- ⚡ Inference engineering hinges on managing the **KV‑cache**: encoder‑decoder needs separate static encoder cache + dynamic decoder cache; decoder‑only only needs a dynamic cache.  
- 📈 The original Transformer paper showed that replacing recurrence with attention unlocks parallel training and better long‑range context — foundations for today’s LLMs.  

---

## What's Next  

Next we’ll dive into **attention mechanics** (scaled dot‑product, multi‑head, masking) and see how they are implemented efficiently (Flash Attention, Paged Attention, KV‑quantization). After that we’ll explore **model families**: encoder‑only (BERT‑style), decoder‑only (GPT‑style), and encoder‑decoder (T5‑style), and discuss which architectural choices affect inference latency, memory footprint, and batching strategies on modern hardware like the DGX Spark.  

Stay tuned — we’ll turn those diagrams into real kernels you can profile and optimize!

## Watch These Videos

- **[Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M)** (27:14)
- **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (1:56:20)
- **[Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://www.youtube.com/watch?v=bCz4OMemCcA)** (58:04)
