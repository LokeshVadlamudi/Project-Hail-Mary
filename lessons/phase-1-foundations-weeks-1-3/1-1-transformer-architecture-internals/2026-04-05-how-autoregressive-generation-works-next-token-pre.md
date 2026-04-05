<!-- Generated: 2026-04-05 -->
<!-- Phase: Phase 1: Foundations (Weeks 1-3) -->
<!-- Section: 1.1 — Transformer Architecture Internals -->



# How autoregressive generation works (next-token prediction loop)

## TL;DR
Autoregressive generation is a simple loop: the model looks at everything it has already produced, predicts the probability distribution over the next token, samples (or picks) one token, appends it, and repeats until a stop condition.  Everything that makes this fast or memory‑efficient—speculative decoding, paged KV‑cache, quantization, flash attention—targets one or more steps inside that loop.

## ELI5 — The Simple Version
Imagine you’re telling a story one word at a time, but before you say each word you whisper to yourself: “What would make the most sense given everything I’ve said so far?” You look at the whole story you’ve built, guess the next word, say it out loud, then start the whole process again with the longer story.  

A language model does the same thing, only the “whisper” is a math operation that turns the current sequence of tokens into a set of scores (logits) for every possible next token. The model picks the token with the highest score (greedy) or samples randomly according to those scores (temperature/top‑p), writes it down, and repeats.  

Because the model never looks ahead—it only conditions on what it has already generated—this process is called **autoregressive** (self‑referencing) generation.

## Why This Matters for Inference Engineering
When you serve an LLM you are not just running a single forward pass; you are running the generation loop thousands of times per request. Every micro‑second you shave off the loop (faster attention, less memory movement, better cache reuse) multiplies by the number of generated tokens and directly improves latency, throughput, and cost. Understanding the loop lets you:

* Spot where bottlenecks appear (attention computation, KV‑cache memory, sampling).  
* Apply the right optimization (speculative decoding, paged attention, quantization, flash attention) to the right part of the loop.  
* Debug strange behavior (e.g., repetition, early stop) by tracing what the model sees at each step.

## How It Actually Works
We’ll walk through the loop step‑by‑step, annotating each piece with the relevant papers that make it faster or cheaper.

### 1. Tokenizer → Input IDs
```
text = "The cat sat on the"
ids  = tokenizer.encode(text)   # → [464,  3290,  318,  262,  284,  318]
```
The model only sees integer IDs; the embedding layer turns each ID into a dense vector (size = model hidden size, e.g., 768 for GPT‑2 small).

### 2. Embedding + Positional Encoding
```
E = Embedding(ids)          # shape: (seq_len, hidden)
P = PositionalEncoding(seq_len)   # sinusoidal or learned
X0 = E + P                  # input to first transformer block
```
*Positional encoding* tells the model where each token sits so attention can distinguish “cat sat” from “sat cat”.

### 3. Transformer Block (simplified)
For each layer `l = 0 … L‑1`:
```
# Self‑attention
Q = Xl @ WQ_l   # (seq_len, d_k)
K = Xl @ WK_l
V = Xl @ WV_l
Attn = softmax(QK^T / sqrt(d_k)) @ V   # (seq_len, d_v)

# Add & Norm
Xl' = LayerNorm(Xl + Attn)

# Feed‑forward (FFN)
FFN = GeLU(Xl' @ W1_l + b1) @ W2_l + b2
Xl+1 = LayerNorm(Xl' + FFN)
```
All operations are **matrix multiplications** plus pointwise non‑linearities.  
The **self‑attention** step is where the model looks at every other token to decide how much each should influence the current token’s representation.

#### FlashAttention (2022) – why it matters
*Problem*: Naïve attention computes `QK^T` (size `seq_len²`) and stores the whole matrix in GPU HBM → O(seq_len²) memory and many memory reads/writes.  
*Key idea*: Tile the computation so that blocks of Q, K, V stay in fast on‑chip SRAM, compute the softmax incrementally, and write out only the final output. This reduces HBM traffic dramatically while staying **exact** (no approximation).  
*Impact*: 2‑4× wall‑clock speedup on long sequences, enabling larger context windows without changing model quality.

### 4. LM Head → Logits
After the final transformer block we have the last hidden state `X_L` (shape `(seq_len, hidden)`).  
We only need the representation of the **most recent token** to predict the next one:
```
h_last = X_L[-1, :]          # (hidden,)
logits = h_last @ W_lm^T + b_lm   # (vocab_size,)
```
`W_lm` is the language‑model projection matrix (often tied to the embedding matrix).

### 5. Sampling / Decoding Strategy
Convert logits to probabilities and pick a token:
```
probs = softmax(logits / temperature)   # temperature controls randomness
next_id = multinomial(probs, num_samples=1)   # or argmax for greedy
```
Common strategies:
* **Greedy** – `argmax(logits)`. Fast, deterministic, but can get stuck in loops.
* **Top‑k** – keep only the k highest logits, renormalize.
* **Top‑p (nucleus)** – keep the smallest set whose cumulative probability ≥ p.
* **Temperature** – >1.0 makes distribution flatter, <1.0 sharper.

### 6. Append & Repeat
```
ids = torch.cat([ids, next_id], dim=0)   # grow the sequence
if next_id == eos_token_id or len(ids) > max_len: break
else: go to step 2
```
That’s the **autoregressive loop**. Each iteration re‑runs the transformer over the *entire* history (or a cached slice – see KV‑cache).

### 7. KV‑Cache – avoiding recomputation
If we recomputed attention from scratch each step we’d do O(t²) work at step `t`.  
Instead we store the **key** and **value** matrices for every previous token:
```
cache_k[l] = torch.cat([cache_k[l], K_new], dim=0)   # shape (t, d_k)
cache_v[l] = torch.cat([cache_v[l], V_new], dim=0)
```
At step `t+1` we only need to compute Q for the new token and attention against the cached K,V.  
This reduces per‑step cost from O(t²) to O(t) (still linear in sequence length because we must scan the cache).

#### PagedAttention (vLLM, 2023) – memory‑efficient KV‑cache
*Problem*: Storing a full KV‑cache for each request in a contiguous tensor leads to fragmentation and wasted GPU memory when requests have varying lengths.  
*Key idea*: Allocate KV‑cache in fixed‑size **pages** (e.g., 16‑token blocks) and manage them like an OS virtual memory system. When a request grows, it grabs a new page; when it shrinks, pages are returned to a pool.  
*Impact*: Near‑zero memory waste, enabling >2× higher throughput on the same GPU, especially for bursty traffic.

### 8. Speculative Decoding – generating multiple tokens per forward pass
*Problem*: The autoregressive loop is bandwidth‑bound; each step waits for the previous step’s output before it can start the next matrix multiply.  
*Key idea*: Run a **small, fast draft model** (e.g., a distilled version) to propose several tokens in parallel. Then verify those proposals with the **large target model** in a single forward pass that computes the likelihood of the whole draft sequence. Accepted tokens are kept; rejected ones cause a rollback and the process restarts.  
*Impact*: 2‑4× speedup with negligible quality loss because the draft model is trained to mimic the target’s distribution.

### 9. Quantization (GPTQ, 2023) – making the model smaller & faster
*Problem*: FP16 or BF16 weights still occupy many GBs; loading them dominates latency, and the GPU’s compute units are under‑utilized due to memory bandwidth limits.  
*Key idea*: GPTQ finds, for each weight matrix, the optimal integer quantization (e.g., 3‑ or 4‑bit) by solving a small least‑squares problem that uses **approximate second‑order information** (the Hessian diagonal). It processes layers one‑shot, preserving accuracy while drastically cutting bit‑width.  
*Impact*: A 175B‑parameter model fits in a single GPU’s memory and runs ~3‑4× faster than FP16, enabling low‑cost serving.

### 10. Efficient Scaling – system‑level tricks
*Problem*: Even with a fast model, serving many requests concurrently hits limits in batch size, memory bandwidth, and kernel launch overhead.  
*Key idea*: Papers like **“Efficiently Scaling Transformer Inference”** (Google, 2023) recommend:
* **Continuous batching** – keep the GPU busy by inserting new requests as soon as a slot frees up (instead of waiting for a full batch).  
* **Asynchronous KV‑cache updates** – overlap compute with memory transfers.  
* **Tensor parallelism / pipeline parallelism** – split model layers across GPUs when a single GPU can’t hold the model.  
*Impact*: Near‑linear throughput scaling with number of GPUs while keeping latency low.

---

### Putting it all together – the loop with optimizations
```
while not done:
    # 1. Embedding + pos enc (cached for old tokens)
    # 2. For each layer:
    #       - compute Q for new token only
    #       - attention = softmax(Q @ K_cache^T) @ V_cache   (FlashAttention tiles this)
    #       - add & norm, FFN
    # 3. LM head → logits
    # 4. Sample next token (temperature/top‑p)
    # 5. Append token, update KV‑cache (PagedAttention pages)
    # 6. (Optional) speculative draft: generate N tokens w/ small model,
    #    verify with big model in one shot, accept/reject.
```
Each bullet corresponds to a paper that attacks a specific sub‑step.

---

## Code You Can Run
Below is a **minimal, end‑to‑end example** that shows the generation loop using a pretrained GPT‑2 model from 🤗 Transformers, plus a simple KV‑cache implementation and speculative decoding with a tiny draft model (we’ll use DistilGPT‑2 as the draft).  
You can copy‑paste this into a notebook on your DGX Spark (PyTorch 2.x, CUDA 12, 🤗 Transformers installed).

```python
# --------------------------------------------------------------
# Autoregressive generation loop with KV‑cache & speculative draft
# --------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1️⃣ Load models ------------------------------------------------
target_name = "gpt2"               # 124M param model (fits easily)
draft_name  = "distilgpt2"         # 82M param draft, faster

tokenizer = AutoTokenizer.from_pretrained(target_name)
target = AutoModelForCausalLM.from_pretrained(target_name).to("cuda").eval()
draft  = AutoModelForCausalLM.from_pretrained(draft_name).to("cuda").eval()

# Ensure tokenizer has an EOS token
tokenizer.pad_token = tokenizer.eos_token

def greedy_next(logits):
    """Pick the token with highest logit (greedy)."""
    return torch.argmax(logits, dim=-1)

def sample_top_p(logits, p=0.9, temperature=1.0):
    """Nucleus sampling."""
    logits = logits / temperature
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative prob > p
    mask = cum_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 0
    sorted_logits[mask] = -float('inf')
    probs = torch.softmax(sorted_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return sorted_idx.gather(-1, next_token)

# 2️⃣ Generation settings -----------------------------------------
prompt = "The future of AI is"
max_new_tokens = 30
use_speculative = True   # set False to see vanilla loop
speculative_depth = 4    # how many draft tokens to propose each step

input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
generated = input_ids.clone()

# KV‑cache containers: list per layer, each entry = (key, value) tensors
target_cache = None
draft_cache  = None

# 3️⃣ Autoregressive loop -----------------------------------------
for step in range(max_new_tokens):
    # ------------------- Target model forward (with cache) -------------------
    if step == 0:
        # First pass: no cache, feed whole prompt
        target_out = target(
            input_ids=generated,
            use_cache=True,
            return_dict=True,
        )
    else:
        # Subsequent passes: only feed the newest token
        target_out = target(
            input_ids=generated[:, -1:],
            use_cache=True,
            past_key_values=target_cache,
            return_dict=True,
        )
    target_logits = target_out.logits[:, -1, :]   # (1, vocab)
    target_cache = target_out.past_key_values

    # ------------------- Optional speculative draft -------------------
    if use_speculative and step > 0:
        # Run draft model for `speculative_depth` steps, starting from the
        # *same* cache state as the target (we copy the cache tensors).
        draft_cache = [tuple(t.clone() for t in layer) for layer in target_cache]
        draft_ids = generated.clone()
        accepted = 0
        for _ in range(speculative_depth):
            draft_out = draft(
                input_ids=draft_ids[:, -1:],
                use_cache=True,
                past_key_values=draft_cache,
                return_dict=True,
            )
            draft_logits = draft_out.logits[:, -1, :]
            draft_cache = draft_out.past_key_values
            # Greedy draft token (you could also sample)
            next_draft = torch.argmax(draft_logits, dim=-1)
            draft_ids = torch.cat([draft_ids, next_draft], dim=1)
            accepted += 1
            # Stop early if draft hits EOS
            if next_draft.item() == tokenizer.eos_token_id:
                break

        # Now verify the whole draft sequence with the target model in ONE forward
        # (we already have the target cache up to the original generated length)
        verify_out = target(
            input_ids=draft_ids[:, generated.shape[1]:],   # only the draft tokens
            use_cache=True,
            past_key_values=target_cache,
            return_dict=True,
        )
        verify_logits = verify_out.logits  # shape (1, draft_len, vocab)
        # Compare each position: accept if target's top‑1 matches draft token
        for i in range(verify_logits.shape[1]):
            target_topk = torch.argmax(verify_logits[0, i, :], dim=-1)
            draft_tok   = draft_ids[0, generated.shape[1] + i]
            if target_topk == draft_tok:
                # accept this token
                generated = torch.cat([generated, draft_tok.unsqueeze(0).unsqueeze(0)], dim=1)
                # also advance target cache by one step (already done in verify_out)
                target_cache = verify_out.past_key_values
                if draft_tok.item() == tokenizer.eos_token_id:
                    break
            else:
                # mismatch → reject this token and all following draft tokens
                # Re‑run target model from the last accepted token to get correct next token
                # (we already have the correct logits for the first mismatched position)
                correct_logits = verify_logits[0, i, :]
                correct_token = greedy_next(correct_logits.unsqueeze(0))
                generated = torch.cat([generated, correct_token.unsqueeze(0).unsqueeze(0)], dim=1)
                # update cache to reflect the correct token
                target_cache = verify_out.past_key_values  # cache already includes up to i
                # break out of draft verification; outer loop will continue
                break
        # If we exited the for‑loop via break due to mismatch, continue outer while
        # (the outer loop's `step` will increment, but we have already added tokens)
        # To avoid double counting, we set `step` appropriately:
        # Number of tokens actually added this iteration:
        added = generated.shape[1] - input_ids.shape[1]
        # Adjust the for‑loop counter so we don't overshoot max_new_tokens
        # (simple approach: break when we hit limit)
        if generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
            break
        continue   # go to next iteration of while (for‑loop)

    # ------------------- Vanilla greedy step (no speculation) -------------------
    next_token = greedy_next(target_logits)
    generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    if next_token.item() == tokenizer.eos_token_id:
        break

# 4️⃣ Decode & print -------------------------------------------------
output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("=== Generated ===")
print(output_text)
```

### What the code demonstrates
| Part | What it shows | Linked paper / concept |
|------|----------------|------------------------|
| Loading two models (target + draft) | Sets up speculative decoding | *Fast Inference from Transformers via Speculative Decoding* (2023) |
| `use_cache=True` & `past_key_values` | KV‑cache that avoids recomputing K,V for old tokens | *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM) – our simple tensor cache is the logical precursor |
| `FlashAttention` is **not** explicit here because 🤗 Transformers already uses an optimized kernel (on recent GPUs it will pick FlashAttention when available). | Shows that the attention step can be made memory‑efficient | *FlashAttention* (2022) |
| Quantization isn’t in the snippet, but you could replace `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.int8, load_in_4bit=True)` to try GPTQ‑style 4‑bit inference. | Illustrates weight‑size reduction | *GPTQ* (2023) |
| The loop itself is the autoregressive generation core. | — | *Attention Is All You Need* (the base transformer) |

Feel free to toggle `use_speculative`, change `temperature`, or swap in a quantized model to see how each optimization affects speed and output quality.

---

## Paper Breakdown (if relevant)

| Paper | One‑line summary | The problem before | Key idea (plain language) | Impact |
|-------|------------------|--------------------|---------------------------|--------|
| **Fast Inference from Transformers via Speculative Decoding** (2023) | Uses a small draft model to propose multiple tokens, then verifies them with the big model in one shot. | Autoregressive generation is strictly sequential → GPU sits idle waiting for each step. | Draft model runs cheaply in parallel; the big model checks a whole short sequence at once, accepting correct proposals and fixing mistakes. | 2‑4× speedup with almost no quality loss; enables low‑latency serving. |
| **Efficient Memory Management for LLM Serving with PagedAttention** (vLLM, 2023) | Stores KV‑cache in fixed‑size pages managed like OS virtual memory. | Naïve contiguous KV‑cache wastes memory due to fragmentation and variable request lengths. | Allocate memory in small chunks (pages); allocate/deallocate on demand, sharing free pages across requests. | Near‑zero memory waste → higher throughput on the same GPU, especially for bursty traffic. |
| **Efficiently Scaling Transformer Inference** (Google, 2023) | System‑level tricks: continuous batching, async KV updates, tensor/pipeline parallelism. | Batching strategies either hurt latency (wait for full batch) or underutilize GPU. | Keep GPU fed by inserting new requests as soon as a slot frees; overlap compute with memory moves; split model across GPUs when needed. | Near‑linear scaling of throughput with GPU count while maintaining low latency. |
| **GPTQ: Accurate Post‑Training Quantization for Generative Pre‑Trained Transformers** (2023) | One‑shot weight quantization to 3‑4 bits using approximate second‑order info. | Prior quantization either needed retraining or suffered large accuracy drops. | Solve a small least‑squares problem per weight column using the Hessian diagonal to find the optimal integer rounding; process layers independently. | Enables >175B models to fit on a single GPU and run 3‑4× faster than FP16 with negligible perplexity change. |
| **FlashAttention: Fast and Memory‑Efficient Exact Attention** (2022) | Makes the exact attention algorithm IO‑aware by tiling to reduce HBM traffic. | Standard attention stores the full QKᵀ matrix (O(seq_len²)) causing high memory bandwidth usage and limiting sequence length. | Compute attention in blocks that fit in on‑chip SRAM, accumulating the softmax online; write out only the final output. | 2‑4× wall‑clock speedup on long sequences; enables larger context windows without approximation. |
| **Attention Is All You Need** (2017) | Introduces the Transformer architecture – stacked self‑attention + feed‑forward blocks. | Prior seq2seq models relied on RNNs/CNNs, which hindered parallelization and were slow to train. | Replace recurrence with self‑attention, allowing all positions to be processed in parallel; stack identical blocks for depth. | Foundation for all modern LLMs; enabled scalable pretraining and the inference optimizations above. |

---

## Key Takeaways
- Autoregressive generation is a tight loop: **embed → attend → FFN → LM‑head → sample → append**.  
- Every iteration’s cost is dominated by **attention** (quadratic in sequence length) and **memory moves** for the KV‑cache.  
- **FlashAttention** makes the attention step memory‑efficient and faster without approximation.  
- **PagedAttention** solves the KV‑cache fragmentation problem, letting many requests share GPU memory.  
- **Speculative decoding** lets the draft model do useful work while the big model verifies, breaking the strict sequential dependency.  
- **GPTQ** squeezes the model into few‑bit weights, cutting load time and memory bandwidth pressure.  
- System tricks like **continuous batching** and **tensor/pipeline parallelism** keep the GPU busy at scale.  
- Understanding each sub‑step lets you pick the right optimization for your serving scenario (latency‑critical vs. throughput‑critical).  

---

## What's Next
Having seen how a single token is produced, the next lessons will dive into:

1. **KV‑cache management in depth** – page allocation, eviction policies, and how vLLM’s implementation works.  
2. **Speculative decoding algorithms** – choosing draft model size, acceptance criteria, and handling mis‑speculation.  
3. **Quantization-aware inference** – int8/4bit kernels, dynamic vs. static scaling, and how GPTQ integrates with HuggingFace `bitsandbytes`.  
4. **Batching strategies** – static vs. continuous batching, request scheduling, and latency‑tail optimization.  
5. **Putting it all together** – building a minimal inference engine that combines FlashAttention, paged KV‑cache, speculative decoding, and 4‑bit weights, then benchmarking it on your DGX Spark.

Stay tuned—each piece builds on the loop you just mastered, turning the “next‑token prediction” from a conceptual idea into a high‑throughput, low‑latency serving system. Happy hacking!

## Watch These Videos

- **[Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M)** (27:14)
- **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (1:56:20)
- **[Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://www.youtube.com/watch?v=bCz4OMemCcA)** (58:04)
