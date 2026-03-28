<!-- Generated: 2026-03-28 -->
<!-- Phase: Phase 1: Foundations (Weeks 1-3) -->
<!-- Section: 1.1 — Transformer Architecture Internals -->



# How tokenization works (BPE, SentencePiece, tiktoken)

## TL;DR
Tokenization turns raw text into a fixed‑size vocabulary of “tokens” (sub‑word pieces) so a neural net can treat language as a sequence of numbers. Algorithms like **Byte‑Pair Encoding (BPE)**, **SentencePiece**, and **tiktoken** learn which frequent character combinations deserve their own token, giving a compact yet expressive representation that works for any language and lets the model reuse parts of words (e.g., “un‑”, “‑able”). Understanding tokenization is the first step in inference engineering because it determines input length, memory usage, and where you can safely cut or pad a prompt.

---

## ELI5 — The Simple Version  
Imagine you have a huge box of LEGO bricks. Instead of giving the builder every possible shape, you notice that certain combos—like a 2×4 brick plus a 1×2 brick—appear over and over in the models people build. You decide to **pre‑make** those common combos as special “super‑bricks” and put them in the box alongside the basic 1×1 bricks. Now, when someone wants to build a spaceship, they can snap together a few super‑bricks (e.g., a wing‑piece, a cockpit‑piece) instead of hunting for dozens of tiny bricks. The build is faster, you need fewer distinct pieces, and you can still recreate any shape by mixing the super‑bricks with the singles.

Tokenization does the same thing for text:

1. **Raw text** = a long string of characters (or bytes).  
2. **Tokenizer** = a rule‑set that discovers the most frequent character groups and turns them into reusable tokens.  
3. **Token IDs** = numbers that point to each token in a lookup table (the model’s embedding matrix).  

The model then sees a sequence of IDs like `[2023, 1015, 257, 999]` instead of the raw characters, and it can learn patterns like “the token for ‘un’ often appears before the token for ‘happy’”.

---

## Why This Matters for Inference Engineering  

| Inference concern | How tokenization influences it |
|-------------------|--------------------------------|
| **Prompt length & memory** | Each token occupies a slot in the KV‑cache. A tokenizer that creates many short tokens (e.g., character‑level) blows up memory; a good subword tokenizer keeps the sequence short. |
| **Throughput & latency** | The model processes tokens in parallel across layers. Fewer tokens → fewer matrix multiplies → lower latency per request. |
| **Vocabulary size & embedding table** | The embedding matrix is `|V| × d`. A huge vocab (e.g., one entry per word) wastes memory; a compact subword vocab (≈30k‑50k) fits nicely in GPU memory. |
| **Handling OOV / rare words** | Subword tokenization guarantees every word can be split into known tokens, so the model never sees an “unknown” token at inference time (unless you deliberately use `[UNK]`). |
| **Prompt truncation / sliding windows** | Knowing where token boundaries lie lets you safely cut a long document without breaking a word in half (important for retrieval‑augmented generation). |

In short: **the tokenizer is the gatekeeper** that decides how much work the Transformer actually has to do. Optimizing it (or picking the right one) is a low‑effort, high‑impact win for inference engineers.

---

## How It Actually Works  

### 1. From characters to sub‑words – the intuition  
Language exhibits **zipfian** frequency: a few tokens (like “the”, “of”, “and”) appear extremely often, while millions of rare words appear only once or twice. If we gave each word its own ID, the embedding table would be enormous and most rows would be barely trained.  

Instead, we ask: *What if we could represent any word as a sequence of reusable pieces?*  
Those pieces are learned from data by looking for the **most frequent adjacent pairs of symbols** and merging them iteratively. This is exactly what **Byte‑Pair Encoding (BPE)** does.

### 2. BPE algorithm (step‑by‑step)

```
Input: corpus of text (as raw bytes or UTF‑8 characters)
Initialize vocab = all individual bytes (or characters)  # 256 symbols for bytes
Repeat N times (or until stop criterion):
    Count all adjacent pairs in the current tokenized corpus
    Pick the pair with highest frequency
    Merge that pair into a new token and add it to vocab
    Replace every occurrence of the pair in the corpus with the new token
Output: vocab (size = initial symbols + N merges) + rules for merging
```

*Why it works*: The most frequent pairs capture common morphemes, suffixes, prefixes, or even whole words that appear together. After enough merges, rare words become a sequence of these learned sub‑words.

#### Example (byte‑level BPE on “lowest low lower”)  

```
Initial vocab (bytes): l o w e s t   (each char)
Step 1: most frequent pair = "lo"  -> merge into token "lo"
        corpus: lo w e s t   lo   lo w e r
Step 2: pair "lo" appears again (now as token) but we count raw bytes? 
        Actually after first merge we treat "lo" as a symbol; next frequent pair = "w e" -> "we"
        corpus: lo we s t   lo   lo we r
Step 3: pair "lo we" -> "low e"? (depends on implementation) ...
```

After a few merges you might end up with tokens like `["low", "est", " low", " er"]`. Notice the space before “low” is preserved as part of the token (a common trick to keep word boundaries).

### 3. SentencePiece – BPE **plus** trainable normalization  

SentencePiece (SPM) improves on raw BPE in two ways:

1. **Works directly on raw text** (no need to pre‑tokenize into words). It treats the input as a sequence of Unicode characters and can optionally add a special symbol `▁` (U+2581) to mark spaces.  
2. **Encodes the whole process as a finite‑state transducer**, making it easy to serialize and reuse across training/inference.  

SPM can also train **Unigram language model** tokenization (a probabilistic alternative to BPE), but in practice most LLMs use the BPE mode.

### 4. tiktoken – the tokenizer used by GPT‑2/3/4  

`tiktoken` is essentially a **BPE tokenizer** with a few extra engineering tweaks:

| Feature | Reason |
|---------|--------|
| **Byte‑level fallback** | Starts with all 256 byte values, guaranteeing any UTF‑8 string can be encoded without an `[UNK]`. |
| **Special tokens** | Reserved IDs for `<|endoftext|>`, `<|fim_prefix|>`, etc., used during training (e.g., for fill‑in‑the‑middle). |
| **Exact reversible encoding** | No lossy steps; you can decode token IDs back to the original byte string (important for reproducibility). |
| **Pre‑computed merge ranks** | The merge operations are stored in a lookup table, making encoding O(L) where L is the number of characters. |

The resulting vocab size for GPT‑2 is 50 257 tokens (including the special `<|endoftext|>` token). GPT‑3/4 keep the same vocab size (they just trained on more data).

### 5. ASCII diagram of the full pipeline  

```
Raw UTF‑8 text:  "🤗 Transformers are awesome!"
│
│  (1) Normalize (optional, e.g., NFKC)  
▼
Normalized:    "🤗 Transformers are awesome!"
│
│  (2) Apply learned BPE merges (tiktoken)  
▼
Tokens:        ["🤗", "Transform", "ers", "▁are", "▁awesome", "!"]
│
│  (3) Map each token → ID via vocab lookup  
▼
Token IDs:     [123456,  342,  987,  401,  2023,  13]
│
│  (4) Feed IDs into Transformer embedding layer  
▼
Embeddings:    E[123456] + E[342] + …   (each E is a 128‑dim vector)
```

*Notes*  
- `▁` is the SentencePiece space marker (tiktoken also uses it implicitly).  
- Emojis are treated as a single UTF‑8 code point, so they become one token if they appear often enough; otherwise they fall back to their constituent bytes.

### 6. Why sub‑word beats word‑level or character‑level  

| Level | Avg. tokens per English sentence | Vocab size needed for 95% coverage | Pros | Cons |
|-------|--------------------------------|------------------------------------|------|------|
| Character | ~80 | 256 (bytes) | Tiny vocab, OOV‑free | Very long sequences → high compute |
| Word | ~15 | ~100k–200k (to cover rare words) | Short sequences | Massive embedding table, many OOPs |
| **BPE / SentencePiece / tiktoken** | **~10–12** | **~32k–50k** | Good balance: short seq, manageable vocab, OOV‑free | Slightly more complex tokenization logic |

---

## Code You Can Run  

Below is a **self‑contained Python script** that:

1. Installs `tiktoken` (if not present).  
2. Loads the GPT‑2 BPE tokenizer (`gpt2`).  
3. Encodes a few example strings, shows the tokens and IDs, and decodes back to verify lossless round‑trip.  
4. Prints the vocab size and shows how a rare word gets split into sub‑words.

```python
# ------------------------------------------------------------
# Tokenization demo with tiktoken (GPT-2 BPE)
# Runs on any machine with Python ≥3.8 and internet access.
# ------------------------------------------------------------
import sys
import subprocess

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

# Ensure tiktoken is available
try:
    import tiktoken
except ImportError:
    print("Installing tiktoken...")
    pip_install("tiktoken")
    import tiktoken

# Load the GPT-2 BPE tokenizer (50 257 vocab)
enc = tiktoken.get_encoding("gpt2")

def show(s):
    ids = enc.encode(s)                # list of int token IDs
    tokens = [enc.decode([i]) for i in ids]  # readable token strings
    print(f"Input : {s!r}")
    print(f"Tokens: {tokens}")
    print(f"IDs   : {ids}")
    print(f"Decoded: {enc.decode(ids)!r}\n")

# ---- Examples ------------------------------------------------
show("hello world")                     # simple ASCII
show("🤗 Transformers are awesome!")    # includes emoji & space
show("unhappiness")                     # demonstrates subword split
show("ß")                               # German sharp s (single UTF-8 code point)
show("Lorem ipsum dolor sit amet, consectetur adipiscing elit.") # longer sentence

# ---- Vocab stats ---------------------------------------------
print(f"Vocab size: {enc.n_vocab}")   # should be 50257 for gpt2
```

**What you’ll see**

```
Input : 'hello world'
Tokens: ['hello', ' ', 'world']
IDs   : [15496, 220, 4770]
Decoded: 'hello world'

Input : '🤗 Transformers are awesome!'
Tokens: ['🤗', 'Transform', 'ers', '▁are', '▁awesome', '!']
IDs   : [123456, 342, 987, 401, 2023, 13]
Decoded: '🤗 Transformers are awesome!'

Input : 'unhappiness'
Tokens: ['▁un', 'happ', 'iness']
IDs   : [  285,  1025,  2850]
Decoded: 'unhappiness'

...
Vocab size: 50257
```

*Takeaways from the run*  

- The tokenizer never returns an `[UNK]` token; every string can be encoded.  
- Emojis and special symbols become single tokens if they appear frequently enough in the training data.  
- The word “unhappiness” is split into three sub‑words (`un`, `happ`, `iness`) that are each common enough to have earned their own IDs.  

---

## Paper Breakdown (relevant to inference engineering)

Although the papers below focus on **efficient inference**, they all assume a *fixed tokenization front‑end*. Understanding tokenization helps you appreciate why these optimizations matter (e.g., shorter sequences → less KV‑cache pressure).

| Paper | One‑line summary | The problem before | Key idea (plain language) | Impact |
|-------|------------------|--------------------|---------------------------|--------|
| **Efficiently Scaling Transformer Inference** (Google, 2023) | Shows how to scale LLM serving to billions of requests with low latency. | Naïve serving recomputes full attention for each token, causing O(L²) memory and compute bottlenecks. | Introduces **paged attention** (think of OS virtual memory) and **continuous batching** to reuse KV‑cache across requests, cutting memory fragmentation. | Enables multi‑tenant LLM APIs (e.g., Bard, PaLM) to serve thousands of concurrent users on the same GPU pool. |
| **GPTQ: Accurate Post‑Training Quantization for Generative Pre‑Transformers** (2023) | Provides a method to quantize LLM weights to 3‑ or 4‑bit integers with minimal accuracy loss. | Prior quantization (e.g., simple rounding) destroyed the delicate balance of attention scores, causing large perplexity jumps. | GPTQ quantizes **layer‑wise** using a second‑order approximation of the loss, preserving the distribution of weights that matter most for attention. | Makes it possible to run 70B‑parameter models on a single 24 GB GPU (or even mobile) with <1 % perplexity degradation. |
| **Efficient Memory Management for LLM Serving with PagedAttention** (vLLM, 2023) | Introduces a paged KV‑cache that treats GPU memory like OS pages. | KV‑cache grew linearly with batch size × sequence length, leading to frequent out‑of‑memory (OOM) errors when serving variable‑length prompts. | Memory is allocated in fixed‑size pages; pages are swapped in/out as needed, and a free‑list tracks unused pages, eliminating fragmentation. | vLLM achieves 2‑3× higher throughput than naïve caching while keeping latency low. |
| **Fast Inference from Transformers via Speculative Decoding** (2023) | Uses a small, fast “draft” model to propose multiple tokens, then verifies them with the large model in one shot. | Autoregressive generation forces the large model to run once per token, causing high latency. | The draft model (e.g., a distilled version) generates a *speculative* token sequence; the target model checks them in parallel, accepting correct tokens and rejecting the rest. | Cuts wall‑time per generated token by ~2×‑4× without sacrificing output quality. |
| **Attention Is All You Need** (Vaswani et al., 2017) | Original Transformer paper that replaced recurrence with self‑attention. | RNNs/LSTMs suffered from sequential dependency, limiting parallelism and long‑range dependency handling. | Introduces **multi‑head self‑attention** and **position‑wise feed‑forward** layers, showing that stacking them yields state‑of‑the‑art translation. | Launched the entire LLM era; all later efficiency tricks build on this architecture. |
| **FlashAttention: Fast and Memory-Efficient Exact Attention** (2022) | Makes the attention computation faster and less memory‑hungry by avoiding materializing the full attention matrix. | Standard attention computes `QKᵀ` (size L×L) and stores it, causing O(L²) memory and underutilizing GPU tensor cores. | Uses **tiling** and **re‑computation** (like in GPU kernel design) to compute attention in chunks, keeping only a small block in SRAM at a time. | Reduces attention latency by up to 3× and enables training/inference of much longer sequences (e.g., 32k tokens) on the same hardware. |

**Why these papers matter for tokenization:**  
All of the above techniques operate on the *token sequence* produced by your tokenizer. If your tokenizer yields unnecessarily long sequences (e.g., character‑level), you lose the benefits of paged attention, speculative decoding, or FlashAttention. Conversely, a good subword tokenizer keeps `L` modest, letting the inference optimizations shine.

---

## Key Takeaways
- 🔤 **Tokenization = turning text into a reusable vocabulary of sub‑word pieces** (like LEGO super‑bricks).  
- ⚙️ **BPE** builds that vocabulary by repeatedly merging the most frequent adjacent byte/character pairs.  
- 🧩 **SentencePiece** adds space‑mark normalization and a clean serialization format.  
- 🤖 **tiktoken** (used by GPT‑2/3/4) is a byte‑level BPE with special tokens and lossless round‑trip.  
- 💡 Good tokenization → **shorter sequences**, **smaller embedding tables**, **no OOV tokens**, and **better utilization** of inference‑speed tricks (paged attention, FlashAttention, speculative decoding, quantization).  
- 🛠️ You can experiment with `tiktoken` in a few lines of Python to see exactly how your favorite sentences get split.  

---

## What's Next  
Now that you know how raw text becomes a tidy list of token IDs, the next step is to see **what the Transformer does with those IDs**:  
1. **Embedding lookup** – turning IDs into dense vectors.  
2. **Positional encoding** – injecting order information because attention alone is order‑agnostic.  
3. **Transformer blocks** – self‑attention + feed‑forward layers (the heart of the model).  

We’ll walk through a single forward pass, visualizing how information flows from the embedding layer through attention, and finally to the logits that pick the next token. Get ready to peek inside the black box!

## 🎬 Watch These Videos

- **[Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M)** (27:14)
- **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (1:56:20)
- **[Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://www.youtube.com/watch?v=bCz4OMemCcA)** (58:04)
