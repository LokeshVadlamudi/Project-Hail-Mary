<!-- Generated: 2026-04-02 -->
<!-- Phase: Phase 1: Foundations (Weeks 1-3) -->
<!-- Section: 1.1 — Transformer Architecture Internals -->



# Full encoder-decoder vs decoder-only architecture  

## TL;DR  
A **full encoder‑decoder** (the original Transformer) has two stacks: an encoder that reads the whole input and a decoder that generates output step‑by‑step while looking back at the encoder. A **decoder‑only** model (like GPT) drops the encoder and lets the decoder attend only to its own previously generated tokens, making it ideal for autoregressive text generation and much simpler to serve at inference time.  

---

## ELI5 — The Simple Version  

Imagine you have a **translation buddy** who works in two stages:

1. **Reader (Encoder)** – sits down with the whole foreign sentence, reads it carefully, and builds a cheat‑sheet that captures the meaning of each word *in context* (e.g., “it” → “the animal”).  
2. **Writer (Decoder)** – looks at that cheat‑sheet, writes the translation word by word, and can peek back at the cheat‑sheet whenever it needs to remember what the original sentence said.

That’s the **encoder‑decoder** setup: a dedicated reader and a dedicated writer, each doing its own job but talking to each other.

Now picture a **story‑telling robot** that only ever writes stories, never reads a separate source. It starts with a prompt (“Once upon a time…”) and, as it adds each new word, it can look back at everything it has already written to decide what comes next. There’s no separate reading phase; the robot’s memory is just the story it’s building so far. That’s a **decoder‑only** model.

The difference matters for inference:  
- Encoder‑decoder needs to run the encoder **once** (to make the cheat‑sheet) and then run the decoder many times (one step per output token).  
- Decoder‑only only ever runs the decoder, but each step must look at all previously generated tokens (self‑attention over the growing sequence).  

---

## Why This Matters for Inference Engineering  

When you serve LLMs you care about **latency**, **memory footprint**, and **throughput**.  

| Aspect | Encoder‑Decoder | Decoder‑Only |
|--------|----------------|--------------|
| **Model size** | Two stacks → roughly 2× the parameters (if same depth) | One stack → smaller for same depth |
| **Encoder cost** | Paid once per input (good if you reuse the same encoder for many decoders, e.g., translation with beam search) | No encoder → no upfront cost |
| **Decoder cost per token** | Attends to encoder outputs (fixed‑size) + self‑attention over generated prefix | Attends only to self‑generated prefix (grows with token count) |
| **KV‑cache friendliness** | Encoder outputs are static → can be cached once; decoder still needs KV cache for self‑attention | Only decoder KV cache needed (simpler) |
| **Typical use‑case** | Seq2seq tasks: translation, summarization, question answering where input and output are both text | Pure generation: story writing, code completion, chat (where the “input” is just the prompt) |

For inference engineering you’ll often pick decoder‑only models because they’re simpler to batch, need only one forward pass per token, and map neatly onto the **KV‑cache** optimization used in serving frameworks (TensorRT‑LLM, vLLM, HuggingFace TGI). Understanding the encoder‑decoder helps you see why some models (e.g., T5, BART) still keep an encoder and when that extra cost is justified.

---

## How It Actually Works  

### 1. The building blocks (shared)

Both architectures use the same **Transformer block**:

```
Input (seq_len x d_model)
   │
   ▼
Multi‑Head Self‑Attention (MHSA)  ←  Q, K, V from same source
   │
   ▼
Add & Norm (residual + layer norm)
   │
   ▼
Position‑wise Feed‑Forward Network (FFN)  (applied independently per position)
   │
   ▼
Add & Norm
   │
   ▼
Output (same shape as input)
```

- **MHSA** lets each token mix information from other tokens via queries (Q), keys (K), values (V).  
- **FFN** is a small MLP (usually 2 linear layers with a GELU) that adds non‑linearity.  
- **Residual + LayerNorm** stabilizes training.

### 2. Encoder‑Decoder flow  

```
Input sentence (source) ──► Embedding ──► [Encoder Stack] ──► Encoder Outputs (memory)
                                                                   │
                                                                   ▼
Target sentence (partial) ──► Embedding + Positional Encoding ──► [Decoder Stack] ──► Logits → next token
```

- **Encoder**: runs **self‑attention only** on the source tokens. No masking; each position can see every other position.  
- **Decoder**: has **two attention sub‑layers**:  
  1. **Masked self‑attention** (can only attend to positions ≤ current target position → prevents peeking at future tokens).  
  2. **Encoder‑decoder attention** (queries from decoder, keys/values from encoder output) → lets the decoder look at the source “cheat‑sheet”.  

```
Encoder stack (N layers):
   X_src → MHSA_src → +&Norm → FFN → +&Norm → … → H_enc

Decoder stack (N layers):
   tgt_emb → MHSA_tgt (masked) → +&Norm → MHSA_encdec (Q from tgt, K/V from H_enc) → +&Norm → FFN → +&Norm → … → logits
```

### 3. Decoder‑Only flow  

```
Prompt (or growing generated text) ──► Embedding + Positional Encoding ──► [Decoder Stack] ──► Logits → next token
```

- Only **masked self‑attention** appears.  
- No separate encoder, no encoder‑decoder attention.  
- The same block repeats N times (e.g., GPT‑2 uses 12 layers, GPT‑3 uses 96).  

```
Decoder‑only stack (N layers):
   X → MHSA_masked → +&Norm → FFN → +&Norm → … → H_final → Linear → vocab logits
```

### 4. Why the encoder‑decoder was a breakthrough  

**Paper**: *Attention Is All You Need* (Vaswani et al., 2017)  

- **Problem**: Prior seq2seq models used RNNs/LSTMs with an attention mechanism that still suffered from sequential computation (hard to parallelize) and a bottleneck fixed‑length context vector.  
- **Key idea**: Replace recurrence entirely with **self‑attention** in both encoder and decoder, allowing **full parallelism** across tokens. Introduce **multi‑head attention**, **position‑wise feed‑forward**, and **layer normalization**.  
- **Impact**: Training speed jumped dramatically (thanks to parallel matrix multiplications) and translation quality surpassed the then‑state‑of‑the‑art Google NMT model. The architecture became the universal building block for virtually all modern LLMs.

### 5. Why decoder‑only became dominant for LLMs  

- **Simplicity**: Only one type of block → easier to scale (just stack more layers).  
- **Autoregressive generation** matches the way humans produce text: token‑by‑token, conditioned on what’s already produced.  
- **Scaling laws** (Kaplan et al., 2020) showed that performance improves predictably with model size, data, and compute when using decoder‑only transformers.  
- **Inference efficiency**: With a KV‑cache, each new token only needs to compute Q for the new position and reuse previously computed K/V, making the cost O(1) per token (aside from the growing cost of the attention matrix, which is mitigated by caching).

---

## Code You Can Run  

Below is a **minimal, self‑contained PyTorch** implementation that shows both architectures side‑by‑side.  
You can run it on a DGX Spark (or any GPU with PyTorch).  
It deliberately uses tiny dimensions so you can see the shapes change; replace them with real sizes (e.g., 768, 12 heads) for experiments.

```python
# --------------------------------------------------------------
# Minimal Encoder‑Decoder vs Decoder‑Only Transformer (PyTorch)
# --------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Helper: Multi‑Head Self‑Attention ----------
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
        """
        x: (B, T, d_model)
        mask: (B, 1, T, T) or None  (True where we want to keep)
        """
        B, T, _ = x.size()
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_h, T, d_k)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot‑product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, n_h, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)          # (B, n_h, T, T)
        context = torch.matmul(attn, V)           # (B, n_h, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        out = self.W_o(context)
        return out, attn

# ---------- Helper: Position‑wise Feed‑Forward ----------
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.lin2(F.gelu(self.lin1(x)))

# ---------- Encoder Layer ----------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MHA(d_model, n_heads)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self‑attention
        attn_out, _ = self.mha(x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# ---------- Decoder Layer (with encoder‑decoder attention) ----------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MHA(d_model, n_heads)   # masked self‑attention
        self.encdec_mha = MHA(d_model, n_heads) # encoder‑decoder attention
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 1️⃣ masked self‑attention
        self_attn, _ = self.self_mha(tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn))

        # 2️⃣ encoder‑decoder attention (query from tgt, key/value from memory)
        encdec_attn, _ = self.encdec_mha(tgt, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(encdec_attn))

        # 3️⃣ FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))
        return tgt

# ---------- Full Encoder‑Decoder Model ----------
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4,
                 n_enc_layers=2, n_dec_layers=2, d_ff=256, max_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_enc_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_dec_layers)
        ])
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (B, S) source token ids
        tgt: (B, T) target token ids (teacher‑forcing during training)
        """
        B, S = src.size()
        _, T = tgt.size()
        # embed + positional
        src_pos = torch.arange(S, device=src.device).unsqueeze(0).expand(B, S)
        tgt_pos = torch.arange(T, device=tgt.device).unsqueeze(0).expand(B, T)
        src_emb = self.tok_emb(src) + self.pos_emb(src_pos)
        tgt_emb = self.tok_emb(tgt) + self.pos_emb(tgt_pos)

        # Encoder stack
        memory = src_emb
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory, src_mask)

        # Decoder stack
        out = tgt_emb
        for dec_layer in self.decoder_layers:
            out = dec_layer(out, memory, tgt_mask, src_mask)

        logits = self.generator(out)   # (B, T, vocab)
        return logits

# ---------- Decoder‑Only Model (GPT‑style) ----------
class GPTStyleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4,
                 n_layers=4, d_ff=256, max_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, idx, attn_mask=None):
        """
        idx: (B, T) token ids (the prompt or already‑generated prefix)
        attn_mask: (B, 1, T, T) mask for causal self‑attention
        """
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x, x, attn_mask, attn_mask)   # memory == x (self‑only)

        logits = self.generator(x)
        return logits

# ---------- Utility: causal mask ----------
def causal_mask(seq_len, device):
    """Returns (1,1,seq_len,seq_len) where True = keep, False = masked."""
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)   # broadcast over batch & heads

# ---------- Demo ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 2000
    batch = 2
    src_len = 8
    tgt_len = 6

    # dummy data
    src = torch.randint(0, vocab_size, (batch, src_len))
    tgt = torch.randint(0, vocab_size, (batch, tgt_len))

    src_mask = None   # encoder sees full source
    tgt_mask = causal_mask(tgt_len, src.device)   # decoder cannot look ahead

    # ---- Encoder‑Decoder ----
    encdec = EncoderDecoderTransformer(vocab_size, d_model=128, n_heads=4,
                                       n_enc_layers=2, n_dec_layers=2,
                                       d_ff=256, max_len=64)
    logits_encdec = encdec(src, tgt, src_mask, tgt_mask)
    print("Encoder‑Decoder logits shape:", logits_encdec.shape)  # (B, T, vocab)

    # ---- Decoder‑Only ----
    gpt = GPTStyleTransformer(vocab_size, d_model=128, n_heads=4,
                              n_layers=4, d_ff=256, max_len=64)
    # pretend we already have a prompt of length 4 and we want to predict token 5
    prompt = torch.randint(0, vocab_size, (batch, 4))
    prompt_mask = causal_mask(4, prompt.device)
    logits_gpt = gpt(prompt, prompt_mask)
    print("Decoder‑Only logits shape:", logits_gpt.shape)  # (B, 4, vocab)

    # ---- Show how KV‑cache would work (conceptual) ----
    # In a real serving loop you would:
    # 1. Run the model on the prompt → get logits for next token + store K,V for each layer.
    # 2. For each new token, feed only the new token’s embedding; the model re‑uses cached K,V.
    # This avoids recomputing attention over the whole prefix each step.
```

**What the code shows**

- Both models share the same building blocks (`MHA`, `FFN`, `LayerNorm`).  
- The encoder‑decoder has **two** stacks; the decoder‑only has just one stack but uses the same `DecoderLayer` (masked self‑attention + optional encoder‑decoder attention, which we set to self‑only by passing `memory == x`).  
- The `causal_mask` implements the “look‑only‑backwards” constraint essential for autoregressive generation.  
- In a real inference server you would keep the **KV‑cache** per layer after the prompt is processed; each step then only computes Q for the new token and does a cheap matrix‑multiply with the cached K/V.

---

## Paper Breakdown (if relevant)

| Paper | One‑line summary | Problem before | Key idea (plain language) | Impact |
|-------|------------------|----------------|---------------------------|--------|
| **Attention Is All You Need** (Vaswani et al., 2017) | Introduced the Transformer, a recurrence‑free seq2seq model built solely on self‑attention and feed‑forward layers. | RNN/LSTM seq2seq models were slow to train (sequential) and struggled with long‑range dependencies due to vanishing gradients and a fixed‑size context vector. | Replace recurrence with **multi‑head self‑attention** that lets every token directly interact with every other token in a single layer; add positional encodings so order isn’t lost; stack identical blocks; use layer norm and residual connections for stability. | Enabled massive parallelism → training speed‑ups of 10‑100×; set new SOTA on translation; became the universal architecture for BERT, GPT, T5, etc. |
| **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019) – GPT‑2 | Showed that scaling a decoder‑only Transformer yields surprisingly strong language understanding. | Prior LMs were either small n‑gram models or task‑specific supervised models; scaling RNN‑LMs was unstable. | Stack many decoder‑only Transformer blocks, train on huge text corpora with a simple next‑token loss; demonstrate emergent abilities (zero‑shot task transfer). | Sparked the LLM scaling era; proved decoder‑only is sufficient for generative tasks. |
| **Training Language Models to Follow Instructions with Human Feedback** (Ouyang et al., 2022) – InstructGPT | Aligned large decoder‑only LMs to human intent via RLHF. | Raw language models produced plausible but often unsafe or off‑topic text. | Collect human preference data, train a reward model, then fine‑tune the LM with Proximal Policy Optimization (PPO) to maximize reward. | Made models usable in real‑world assistants; highlighted that decoder‑only architecture works fine for alignment techniques. |

---

## Key Takeaways  

- 🔁 **Encoder‑Decoder** = separate “reader” (encoder) and “writer” (decoder); encoder builds a context representation that the decoder can attend to at each step.  
- 🤖 **Decoder‑Only** = only a writer; it generates tokens by looking back at what it has already written (masked self‑attention).  
- ⚡️ **Inference impact**: Encoder‑decoder pays a one‑time encoder cost; decoder‑only pays per‑token self‑attention cost but benefits from a simpler KV‑cache and smaller model footprint for the same depth.  
- 📈 **Scaling law**: Decoder‑only models scale predictably with size, data, and compute, making them the default for today’s LLMs (GPT, LLaMA, Mistral, etc.).  
- 🛠️ **Implementation tip**: In serving, cache the key/value tensors of each layer after the prompt; each new token only needs a query projection and a dot‑product with the cache → O(1) per token (ignoring the growing cost of the softmax over the vocabulary).  

---

## What's Next  

Having grasped the **macro‑structure** of encoder‑decoder vs decoder‑only, the next steps are:

1. **Attention mechanics** – dive deeper into multi‑head attention, scaling, and why we split heads.  
2. **Positional encodings** – sinusoidal vs learned vs RoPE, and how they let the model sense order without recurrence.  
3. **KV‑cache & inference optimization** – concrete algorithms for caching keys/values, batching, and paged attention (vLLM, TensorRT‑LLM).  
4. **Model families** – compare encoder‑decoder models (T5, BART, mBART) with decoder‑only families (GPT‑NeoX, LLaMA, Falcon) and see where each shines.  
5. **Hands‑on** – extend the minimal code above to include a real KV‑cache loop, beam search for encoder‑decoder, and top‑k/top‑p sampling for decoder‑only.  

Stay tuned—once you know what you’re optimizing (the attention‑based transformer), you’ll be ready to make those LLMs fly in production! 🚀

## Watch These Videos

- **[Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M)** (27:14)
- **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (1:56:20)
- **[Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://www.youtube.com/watch?v=bCz4OMemCcA)** (58:04)
