<!-- Generated: 2026-03-27 -->
<!-- Phase: Phase 1: Foundations (Weeks 1-3) -->
<!-- Section: 1.1 — Transformer Architecture Internals -->



# How tokenization works (BPE, SentencePiece, tiktoken)

## Overview
Tokenization is the first step that turns raw text into a sequence of numbers a neural network can consume. For a transformer‑based LLM the quality of the tokenizer directly impacts:

* **Vocabulary size** – determines embedding matrix size and memory footprint.  
* **Out‑of‑vocabulary (OOV) rate** – affects how often the model must fall back to special tokens (e.g., `<unk>`).  
* **Inference latency** – longer token sequences mean more transformer layers to run.  
* **Determinism & reproducibility** – essential for serving, caching, and benchmarking.

Understanding how the three most common sub‑word tokenizers (BPE, SentencePiece, tiktoken) are built lets you:

* Choose the right tokenizer for a model you’re deploying.  
* Diagnose token‑related bottlenecks (e.g., excessive splitting of rare words).  
* Extend or fine‑tune a tokenizer without breaking downstream compatibility.

---

## Core Concepts

### 1. Why not character‑ or word‑level tokens?
| Level | Pros | Cons |
|-------|------|------|
| **Character** | No OOV, tiny vocab | Very long sequences → high compute, poor linguistic intuition |
| **Word** | Short sequences, intuitive | Large vocab (hundreds of thousands), many OOV for rare/misspelled words |
| **Sub‑word (BPE, SentencePiece, tiktoken)** | Balances length & vocab size, handles rare words via composition | Slightly more complex algorithm, needs training corpus |

All three algorithms produce a **fixed‑size vocabulary** (e.g., 32 k, 50 k, 100 k tokens) where each token is either a whole word, a sub‑word piece, or a special symbol (`<pad>`, `<eos>`, `<unk>`).

### 2. Byte Pair Encoding (BPE) – the original idea
1. **Start** with a corpus of characters (or UTF‑8 bytes).  
2. **Iteratively** merge the most frequent adjacent pair of symbols into a new symbol.  
3. Stop after *N* merges → vocabulary size = initial symbols + *N* merges.  
4. To tokenize a sentence, greedily apply the learned merges from longest to shortest.

*Result*: Frequent character pairs become single tokens (e.g., `th`, `ing`). Rare words are split into known sub‑words.

### 3. SentencePiece – BPE + flexibility
* Treats the input as a **raw byte stream** (no pre‑tokenization on whitespace/punctuation).  
* Can train **BPE**, **Unigram language model**, or **Word** models.  
* Outputs a **model file** (`.model`) that contains the vocabulary and merge rules; tokenization is a pure lookup → no dependency on external libraries (except the SentencePiece runtime).  
* Handles **language‑agnostic** tokenization (Japanese, Chinese, emojis) because it works on Unicode code points.

### 4. tiktoken – OpenAI’s BPE implementation
* Essentially a **fast, deterministic BPE** built on top of the **gpt‑2** vocab (50 257 tokens).  
* Uses **byte-level fallback**: any unknown UTF‑8 byte sequence is split into its constituent bytes, each of which has a dedicated token (`0`‑`255`).  
* Guarantees **reversibility**: you can always decode a token ID back to the original byte string (no information loss).  
* Implemented in Rust/Cython with a Python wrapper; tokenization is **O(L)** where *L* is the length of the byte string.

---

## How It Works (Technical Deep Dive)

### BPE – Step‑by‑step math (intuitive)

Let the training corpus be a set of words `W = {w₁, w₂, …}` each represented as a sequence of **bytes** (or characters).  
Define the **pair frequency**:

```
freq(pair) = Σ_{w∈W} count_of_pair_in(w)
```

At each iteration:
1. Find `pair* = argmax_{pair} freq(pair)`.  
2. Replace every occurrence of `pair*` in all words with a new symbol `S`.  
3. Add `S` to the vocabulary.

The process stops after `k` merges, giving vocab size `|V| = |Σ| + k` where `|Σ|` is the initial symbol set (usually 256 bytes).

**Complexity** – Naïve recomputation of frequencies is `O(N²)` per iteration; efficient implementations use a **priority queue** (heap) and update only affected pairs after each merge, achieving roughly `O(N log N)` overall.

### SentencePiece – Unigram alternative (brief)

Instead of greedy merges, the Unigram model starts with a large vocab and iteratively **removes** tokens that hurt the likelihood least, using an EM‑style algorithm. The resulting vocab often yields **better perplexity** for the same size because it optimizes a probabilistic objective rather than a purely frequency‑based heuristic.

### tiktoken – Byte‑level fallback

* Vocabulary consists of:
  * 256 **byte tokens** (one per possible UTF‑8 byte).  
  * Merge‑generated tokens (e.g., `“the”`, `“ing”`).  
  * Special tokens (`<|endoftext|>`, `<|fim_prefix|>`, etc.).
* Encoding algorithm:
  1. Convert input string to UTF‑8 bytes.  
  2. While there exists a mergeable pair in the current byte list, replace the **longest** matching merge (preferring earlier merges in the list).  
  3. Emit the token IDs for the final sequence.  
* Decoding is a simple table lookup: `bytes = b''.join(id_to_bytes[token_id] for token_id in ids)`.

Because the byte tokens are always present, **any** Unicode string can be encoded without OOV.

### Trade‑offs Summary

| Property | BPE (raw) | SentencePiece | tiktoken |
|----------|-----------|---------------|----------|
| **Determinism** | Yes (if merge order fixed) | Yes (model file) | Yes (hard‑coded merges) |
| **Speed** | Moderate (depends on implementation) | Fast (pure lookup + DP) | Very fast (Rust/Cython) |
| **OOV handling** | Requires `<unk>` or fallback | Same as BPE unless byte fallback added | Byte fallback → zero OOV |
| **Language neutrality** | Needs pre‑tokenization (whitespace/punct) | Works on raw Unicode | Works on raw UTF‑8 bytes |
| **Model size** | Vocab + merge rules | Vocab + model file (similar) | Fixed 50 257‑token vocab (+ specials) |
| **Customizability** | Easy to re‑train on new corpus | Easy (choose BPE/Unigram/Word) | Fixed; you can only add specials |

---

## Practical Example

Below is a self‑contained Python script that:

1. Trains a tiny BPE tokenizer on a sample corpus (using the `tokenizers` library – HuggingFace’s fast tokenizer backend).  
2. Shows how SentencePiece can be used to train a comparable model.  
3. Demonstrates tiktoken encoding/decoding with the GPT‑2 vocab.  
4. Measures sequence length and vocab size for the same sentence.

> **Tip:** If you have a DGX Spark (128 GB unified memory, Blackwell GPU), you can run the training on the CPU (tokenization is CPU‑bound) and then move the resulting token IDs to the GPU for model inference.

```python
# --------------------------------------------------------------
# tokenization_demo.py
# --------------------------------------------------------------
import time
from pathlib import Path

# 1️⃣  Tiny BPE with HuggingFace tokenizers (pip install tokenizers)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_bpe(corpus_path: Path, vocab_size: int = 2000):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    return tokenizer

# 2️⃣  SentencePiece (pip install sentencepiece)
import sentencepiece as spm

def train_sentencepiece(corpus_path: Path, model_prefix: str, vocab_size: int = 2000):
    spm.SentencePieceTrainer.Train(
        f"--input={corpus_path} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0"
    )
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

# 3️⃣  tiktoken (pip install tiktoken)
import tiktoken

def demo():
    # ---- Sample corpus -------------------------------------------------
    corpus = Path("tiny_corpus.txt")
    corpus.write_text(
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "hello world! 🚀 🌍\n"
        "🚀🌍🚀🌍🚀🌍"
    , encoding="utf-8")

    print("\n=== Training tiny BPE (HF tokenizers) ===")
    start = time.time()
    bpe_tok = train_bpe(corpus, vocab_size=500)
    print(f"Trained in {time.time() - start:.2f}s")
    print(f"Vocab size: {bpe_tok.get_vocab_size()}")

    # Encode a sentence
    sentence = "The quick brown fox jumps over the lazy dog."
    ids = bpe_tok.encode(sentence).ids
    tokens = bpe_tok.encode(sentence).tokens
    print(f"Sentence: {sentence}")
    print(f"Tokens  : {tokens}")
    print(f"IDs     : {ids}")

    print("\n=== Training SentencePiece BPE ===")
    sp = train_sentencepiece(corpus, "sp_model", vocab_size=500)
    print(f"Vocab size: {sp.get_piece_size()}")
    sp_ids = sp.encode(sentence, out_type=int)
    sp_tokens = sp.encode(sentence, out_type=str)
    print(f"Sentence: {sentence}")
    print(f"SP Tokens : {sp_tokens}")
    print(f"SP IDs    : {sp_ids}")

    print("\n=== tiktoken (GPT-2) ===")
    enc = tiktoken.get_encoding("gpt2")
    tik_ids = enc.encode(sentence)
    tik_toks = enc.decode_single_token_bytes(tik_ids)  # not ideal; we'll show decode
    print(f"Sentence: {sentence}")
    print(f"TikTok IDs : {tik_ids}")
    print(f"Decoded back: {enc.decode(tik_ids)}")

    # ---- Length comparison ------------------------------------------------
    print("\n=== Length comparison (number of tokens) ===")
    print(f"HF BPE   : {len(ids)} tokens")
    print(f"SentencePiece: {len(sp_ids)} tokens")
    print(f"tiktoken : {len(tik_ids)} tokens")

if __name__ == "__main__":
    demo()
```

**What to observe**

* The HF BPE and SentencePiece models produce **similar tokenizations** (both split “quick” into `qu`+`ick` etc.) because they learned merges from the same tiny corpus.  
* tiktoken, trained on a massive web corpus, keeps most English words intact (`The`, `quick`, `brown`, …) and only splits punctuation or rare symbols.  
* All three return **integer ID lists** that can be fed straight into a transformer’s embedding layer.  
* Because tiktoken includes a byte fallback, even the emojis 🚀🌍 are encoded as a single token each (they appear in the GPT‑2 vocab), whereas the tiny BPE models may split them into multiple sub‑word tokens or fall back to `[UNK]`.

**Running on DGX Spark**

```bash
# Assuming you have the script above:
python tokenization_demo.py   # runs on CPU; tokenization is lightweight
# To move IDs to GPU for inference (PyTorch example):
import torch
ids = torch.tensor(tik_ids, dtype=torch.long).to("cuda")  # shape: [1, seq_len]
```

You can scale the corpus size up to hundreds of MB; the training will still finish in seconds on a modern CPU because the algorithms are near‑linear.

---

## Key Takeaways
- Tokenization determines **sequence length**, **vocab size**, and **OOV behavior**, all of which directly affect inference latency, memory usage, and model accuracy.  
- **BPE** builds a vocabulary by iteratively merging the most frequent byte/character pairs; it’s simple, deterministic, and works well when you can afford a pre‑tokenization step (whitespace/punct).  
- **SentencePiece** removes the need for language‑specific pre‑tokenization, works directly on raw Unicode, and can train BPE, Unigram, or word models; its model file is self‑contained.  
- **tiktoken** is a highly optimized, byte‑level BPE with a fixed GPT‑2 vocab and guaranteed reversibility—ideal for serving OpenAI‑compatible models.  
- Choose a tokenizer that matches your model’s training data: if you’re re‑using a pretrained LLM, keep its exact tokenizer; if you’re training from scratch, SentencePiece gives the most flexibility with minimal extra code.  
- For inference engineering, profile tokenization latency (usually sub‑millisecond per sentence) and verify that the token IDs you generate match the model’s expected vocabulary (especially special tokens and byte fallback).  

---

## What's Next
With a solid grasp of how text becomes numbers, the next step is to see **how those numbers flow through the transformer**:

* **Positional encodings** – why we add them and how they interact with token embeddings.  
* **Attention mechanics** – scaling, masking, and the cost of longer token sequences.  
* **KV‑cache inference** – how storing key/value tensors per token reduces repeated computation during autoregressive generation.  

Understanding tokenization lets you predict and control the sequence length that drives those downstream costs, making you ready to optimize latency, throughput, and memory in real‑world LLM serving systems. Happy tokenizing!