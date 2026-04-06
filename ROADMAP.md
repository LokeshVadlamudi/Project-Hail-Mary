# Gen AI Inference Engineering Roadmap

> For a fullstack developer with 5 years of experience.
> Estimated timeline: 4-6 months (2-3 hrs/day alongside work)

---

## Phase 1: Foundations (Weeks 1-3)

### 1.1 — Transformer Architecture Internals

**Goal:** Understand what you're optimizing before you optimize it.

- [x] How tokenization works (BPE, SentencePiece, tiktoken)
- [x] Embedding layers and positional encoding
- [x] Self-attention mechanism (Q, K, V matrices)
- [x] Multi-head attention — why and how
- [x] Feed-forward layers, layer norms, residual connections
- [x] Full encoder-decoder vs decoder-only architecture
- [x] How autoregressive generation works (next-token prediction loop)

**Resources:**
- "Attention Is All You Need" paper (the original transformer paper)
- 3Blue1Brown — "But what is a GPT?" (YouTube series)
- Jay Alammar — "The Illustrated Transformer" (blog post)
- Andrej Karpathy — "Let's build GPT from scratch" (YouTube)
- Umar Jamil — Transformer explainer videos (YouTube)

**Hands-on:**
- Implement a minimal transformer from scratch in PyTorch (~200 lines)
- Trace a single forward pass: input text → tokens → embeddings → attention → output logits → sampled token

---

### 1.2 — LLM Generation Deep Dive

**Goal:** Understand the generation loop that inference engineering optimizes.

- [ ] Prefill phase vs decode phase
- [ ] KV cache — what it stores, why it exists, how it grows
- [ ] Sampling strategies: greedy, top-k, top-p (nucleus), temperature
- [ ] Stop conditions and EOS tokens
- [ ] Context window and how positional encodings limit it
- [ ] RoPE (Rotary Position Embeddings) — used by most modern LLMs

**Resources:**
- HuggingFace `transformers` library source code (generation_utils.py)
- "A Survey on Efficient Inference for LLMs" (2024 survey paper)

**Hands-on:**
- Load a small model (GPT-2 or Llama 3.2 1B) with HuggingFace
- Print the KV cache shape at each generation step
- Measure how generation time scales with sequence length

---

## Phase 2: Model Serving Frameworks (Weeks 4-6)

### 2.1 — vLLM (Primary Focus)

**Goal:** Master the most widely used open-source inference engine.

- [ ] Install and serve a model with vLLM
- [ ] Understand PagedAttention — the core innovation
- [ ] Continuous batching vs static batching
- [ ] vLLM's scheduler and how it manages requests
- [ ] API server (OpenAI-compatible) — how it works under the hood
- [ ] Key parameters: `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`
- [ ] Benchmarking with vLLM's built-in benchmarks

**Resources:**
- vLLM docs (docs.vllm.ai)
- "Efficient Memory Management for LLM Serving with PagedAttention" (paper)
- vLLM GitHub — read the scheduler code

**Hands-on:**
- Serve Llama 3.1 8B with vLLM on your DGX Spark
- Then serve Llama 3.1 70B (INT4 quantized) — your 128GB can handle it
- Benchmark: measure throughput (tokens/sec) and latency (TTFT, TPOT) with varying concurrency
- Compare vLLM vs naive HuggingFace `generate()` — quantify the difference

---

### 2.2 — Other Serving Frameworks (Survey)

**Goal:** Know the landscape so you can pick the right tool.

- [ ] TensorRT-LLM (NVIDIA) — when and why to use it
- [ ] SGLang — RadixAttention, structured generation
- [ ] Text Generation Inference (TGI) by HuggingFace
- [ ] Triton Inference Server — multi-model serving
- [ ] llama.cpp / GGML — CPU and edge inference
- [ ] When to use each (decision matrix)

**Hands-on:**
- Serve the same model on vLLM, TGI, and llama.cpp on your DGX Spark
- Build a comparison table: throughput, latency, memory usage, ease of setup
- Bonus: try TensorRT-LLM — you have native Blackwell hardware, which is its sweet spot

---

## Phase 3: Quantization & Model Optimization (Weeks 7-9)

### 3.1 — Quantization Theory

**Goal:** Understand how to make models smaller and faster without destroying quality.

- [ ] What quantization is (FP32 → FP16 → INT8 → INT4)
- [ ] Data types: FP32, FP16, BF16, FP8, INT8, INT4
- [ ] Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)
- [ ] Weight-only quantization vs weight-activation quantization
- [ ] Calibration datasets and why they matter
- [ ] Perplexity as a quality metric for quantized models

**Resources:**
- "A Survey of Quantization Methods for Efficient Neural Network Inference"
- Tim Dettmers' blog posts on quantization
- HuggingFace quantization docs

---

### 3.2 — Quantization in Practice

**Goal:** Quantize models yourself and measure the trade-offs.

- [ ] GPTQ — how it works, when to use it
- [ ] AWQ (Activation-Aware Weight Quantization)
- [ ] GGUF format and llama.cpp quantization levels (Q4_K_M, Q5_K_S, etc.)
- [ ] FP8 quantization (NVIDIA Hopper/Ada)
- [ ] BitsAndBytes (4-bit, 8-bit via HuggingFace)
- [ ] AutoAWQ and AutoGPTQ tools

**Hands-on:**
- Quantize Llama 3.1 8B to: FP16, INT8, INT4 (GPTQ), INT4 (AWQ)
- For each: measure model size, VRAM usage, throughput, and perplexity
- Build a chart: quality vs speed vs memory for each quantization method

---

### 3.3 — Other Optimization Techniques

- [ ] Speculative decoding — draft model + verification
- [ ] KV cache quantization (FP8 KV cache)
- [ ] Flash Attention and Flash Attention 2 — why they're faster
- [ ] Sliding window attention (Mistral-style)
- [ ] Grouped Query Attention (GQA) vs Multi-Query Attention (MQA)
- [ ] Model pruning (structured vs unstructured)
- [ ] Knowledge distillation (at a high level)

**Hands-on:**
- Enable speculative decoding in vLLM with a draft model — measure speedup
- Toggle Flash Attention on/off and benchmark the difference
- Experiment with KV cache quantization in vLLM

---

## Phase 4: GPU & Systems Knowledge (Weeks 10-12)

### 4.1 — GPU Architecture for Inference Engineers

**Goal:** You don't need to write CUDA kernels, but you need to understand the hardware.

- [ ] GPU vs CPU — why GPUs are better for matrix math
- [ ] CUDA cores, Tensor cores, Streaming Multiprocessors (SMs)
- [ ] GPU memory hierarchy: registers → shared memory → L2 cache → HBM (global memory)
- [ ] Memory bandwidth vs compute — the roofline model
- [ ] Why LLM inference is memory-bandwidth bound (not compute bound)
- [ ] NVIDIA GPU generations: Ampere (A100), Hopper (H100), Blackwell (B200)
- [ ] GPU specs that matter: HBM capacity, HBM bandwidth, Tensor core TFLOPS

**Resources:**
- "Making Deep Learning Go Brrrr From First Principles" (Horace He, blog)
- NVIDIA CUDA programming guide (skim the architecture sections)
- "GPU Performance Background" section of the FlashAttention paper

**Hands-on:**
- Use `nvidia-smi` to monitor GPU utilization during inference
- Use `torch.cuda.memory_summary()` to understand memory allocation
- Calculate: can model X fit on GPU Y? (parameter count × bytes per param + KV cache)

---

### 4.2 — Multi-GPU & Distributed Inference

**Goal:** Serve models that don't fit on a single GPU.

- [ ] Tensor parallelism — splitting layers across GPUs
- [ ] Pipeline parallelism — splitting model stages across GPUs
- [ ] Data parallelism — for throughput scaling
- [ ] NVLink, NVSwitch, InfiniBand — interconnects and why they matter
- [ ] When to use tensor parallel vs pipeline parallel
- [ ] NCCL (NVIDIA Collective Communications Library)

**Hands-on:**
- Serve Llama 3.1 70B on your DGX Spark — experiment with how the unified memory architecture handles large models differently than discrete GPU setups
- For multi-GPU tensor parallelism practice: rent a 2×A100 or 4×A100 node on RunPod for a few hours (~$4-8 total)
- Measure how throughput scales with parallelism strategies

---

## Phase 5: Production Infrastructure (Weeks 13-16)

### 5.1 — Serving Infrastructure

**Goal:** Make inference systems production-ready (this is where your fullstack skills shine).

- [ ] Load balancing across inference replicas
- [ ] Autoscaling based on queue depth / latency / GPU utilization
- [ ] Request queuing and priority scheduling
- [ ] Streaming responses (SSE) and chunked transfer encoding
- [ ] Health checks, graceful shutdown, model loading strategies
- [ ] A/B testing and model routing
- [ ] Rate limiting and quota management

**Hands-on:**
- Build a multi-replica inference service behind a load balancer
- Implement request routing: small requests → small model, complex → large model
- Add autoscaling rules and test with a load generator (locust or k6)

---

### 5.2 — Kubernetes for GPU Workloads

**Goal:** Deploy and manage inference at scale.

- [ ] Kubernetes GPU scheduling (nvidia-device-plugin)
- [ ] Resource requests/limits for GPU, CPU, memory
- [ ] Node pools and GPU node taints/tolerations
- [ ] NVIDIA GPU Operator
- [ ] Model storage: loading from S3/GCS, model caching, PVCs
- [ ] Horizontal Pod Autoscaler with custom metrics

**Hands-on:**
- Deploy vLLM on Kubernetes with GPU support
- Set up autoscaling based on request queue length
- Implement rolling updates with zero-downtime model swaps

---

### 5.3 — Observability & Cost Optimization

**Goal:** Monitor, debug, and reduce cost of inference systems.

- [ ] Key metrics: TTFT, TPOT (time per output token), throughput (tokens/sec), queue depth
- [ ] Prometheus + Grafana dashboards for inference
- [ ] Cost per token / cost per request calculations
- [ ] GPU utilization optimization
- [ ] Batching tuning for throughput vs latency trade-offs
- [ ] Spot/preemptible instances for batch inference
- [ ] Caching: semantic caching, prompt caching, prefix caching

**Hands-on:**
- Set up a Grafana dashboard monitoring vLLM metrics
- Implement a semantic cache (hash similar prompts, return cached responses)
- Calculate cost-per-token for different quantization levels and batch sizes

---

## Phase 6: Advanced Topics (Weeks 17-20)

### 6.1 — Cutting Edge (Pick 2-3)

- [ ] Mixture of Experts (MoE) inference — how routing works, memory implications
- [ ] Prefix caching and RadixAttention (SGLang)
- [ ] Disaggregated inference (separate prefill and decode)
- [ ] LoRA serving — multiple adapters on one base model
- [ ] Structured/constrained generation (JSON mode, grammar-guided decoding)
- [ ] Multimodal inference (vision-language models)
- [ ] Edge inference and on-device deployment
- [ ] Custom CUDA kernels with Triton (the language, not the server)

---

### 6.2 — Stay Current

The field moves fast. Build habits to stay up to date.

- [ ] Follow: vLLM releases, NVIDIA tech blog, r/LocalLLaMA
- [ ] Read new papers monthly (Arxiv Sanity, Papers With Code)
- [ ] Attend/watch: MLSys, NeurIPS systems track, NVIDIA GTC talks
- [ ] Follow key people: Woosuk Kwon (vLLM), Tim Dettmers (quantization), Tri Dao (FlashAttention)
- [ ] Join Discord communities: vLLM, EleutherAI, Nous Research

---

## Portfolio Projects

### Workplace Projects (High Impact — Do These First)

These carry more weight than side projects because they involve real users, real data, and real business outcomes. Pitch them to your manager using the cost reduction or data privacy angle.

#### WP1: Self-Hosted Model Serving (Replace/Reduce OpenAI Spend)

Pitch: *"We're spending $X/month on OpenAI. I'll prototype self-hosting an open model for our simpler use cases and cut costs by Y%."*

- [ ] Audit current LLM API usage — which calls are simple (summarization, classification, extraction) vs complex (reasoning, code gen)
- [ ] Deploy an open model (Llama 3.1 8B or 70B) on internal infrastructure using vLLM
- [ ] Benchmark quality: run the same prompts through OpenAI and the self-hosted model, compare outputs
- [ ] Benchmark cost: calculate cost-per-token for self-hosted vs API
- [ ] Migrate simple use cases to the self-hosted model, keep complex ones on OpenAI
- [ ] Document: before/after cost, latency, quality metrics

**Resume line:** *"Deployed and optimized self-hosted Llama 70B with vLLM, reduced LLM API costs by X% while maintaining quality parity for Y use cases."*
**Skills demonstrated:** Model serving, benchmarking, cost optimization, production deployment.

#### WP2: Internal LLM Gateway with Observability

Pitch: *"We have multiple teams calling LLM APIs with no visibility. I'll build a gateway that gives us cost tracking, latency monitoring, and rate limiting across all LLM usage."*

- [ ] Build an API proxy that sits between your internal services and LLM backends (OpenAI, self-hosted, etc.)
- [ ] Add request logging: tokens used, latency, model, caller, cost per request
- [ ] Build a dashboard: cost by team, latency percentiles (p50/p95/p99), throughput, error rates
- [ ] Add rate limiting and quota management per team
- [ ] Add semantic caching — cache responses for near-duplicate prompts, measure hit rate and savings
- [ ] Bonus: add smart routing — send simple requests to the cheap model, complex to the expensive one

**Resume line:** *"Built internal LLM gateway serving N requests/day across M teams. Added semantic caching (X% hit rate) and cost observability, reducing total LLM spend by Y%."*
**Skills demonstrated:** Infrastructure, observability, caching, cost optimization, fullstack.

#### WP3: RAG Pipeline with Self-Hosted Models

Pitch: *"Legal flagged concerns about sending internal docs to external APIs. I'll build a RAG pipeline that runs entirely on our infrastructure."*

- [ ] Set up a vector store (pgvector, Qdrant, or Weaviate) for internal documents
- [ ] Build the retrieval pipeline: chunking strategy, embedding model, similarity search
- [ ] Serve the generation model locally with vLLM
- [ ] End-to-end pipeline: query → retrieve → generate, all on internal infra
- [ ] Optimize: embedding model quantization, generation model quantization, response latency
- [ ] Add evaluation: retrieval accuracy, generation quality, hallucination rate

**Resume line:** *"Built fully self-hosted RAG pipeline for internal knowledge base — zero external API dependency. Optimized end-to-end latency from X ms to Y ms."*
**Skills demonstrated:** Model serving, optimization, end-to-end systems, data privacy.

#### WP4: Inference Cost & Performance Optimization Sprint

Pitch: *"Our self-hosted models are running but I can make them faster and cheaper with some targeted optimizations."*

(Best if your company already has some self-hosted model infra)

- [ ] Profile current serving setup: GPU utilization, memory usage, batching efficiency
- [ ] Apply quantization (INT8 or INT4) — measure quality impact and speed gain
- [ ] Tune continuous batching parameters — find the throughput/latency sweet spot
- [ ] Implement prompt caching / prefix caching for repeated prompt patterns
- [ ] Add KV cache optimization (FP8 KV cache if hardware supports it)
- [ ] Document everything: before/after metrics for each optimization, decision rationale

**Resume line:** *"Optimized LLM inference pipeline: 3x throughput improvement and 40% memory reduction through quantization, batching tuning, and KV cache optimization."*
**Skills demonstrated:** Quantization, performance profiling, batching, caching — core inference engineering.

---

### Personal Projects (Build on Your DGX Spark)

These fill gaps that workplace projects might not cover and give you public portfolio pieces (GitHub, blog posts).

#### PP1: Inference Benchmark Suite
Build a tool that benchmarks any model across multiple serving frameworks (vLLM, TGI, llama.cpp, TensorRT-LLM), quantization levels, and hardware (DGX Spark vs Mac Mini M4). Output a comparison report with charts.
**Skills demonstrated:** Serving frameworks, quantization, benchmarking, cross-platform analysis.

#### PP2: Optimized 70B Inference Pipeline
Take Llama 3.1 70B on your DGX Spark and squeeze maximum performance out of it. Document every optimization (quantization, batching, caching, KV compression) and its measured impact. Write it up as a blog post.
**Skills demonstrated:** Quantization, batching, caching, KV optimization, GPU proficiency, Blackwell-specific tuning.

#### PP3: Speculative Decoding Benchmark
Implement and benchmark speculative decoding with different draft/target model pairs on your DGX Spark. Measure acceptance rates, speedup, and quality across various tasks.
**Skills demonstrated:** Advanced decoding strategies, benchmarking, model pairing.

#### PP4: Cross-Platform Inference Comparison
Run the same models on DGX Spark (CUDA) vs Mac Mini M4 (Metal/MLX) vs cloud A100. Publish a detailed comparison — this content barely exists online for Blackwell hardware.
**Skills demonstrated:** Cross-platform expertise, benchmarking, technical writing. Bonus: likely to get attention since DGX Spark benchmarks are scarce.

---

## Key Papers (Read in Order)

1. "Attention Is All You Need" (2017) — The transformer
2. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
3. "Efficient Memory Management for LLM Serving with PagedAttention" (2023) — vLLM
4. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2023)
5. "AWQ: Activation-aware Weight Quantization" (2023)
6. "Fast Inference from Transformers via Speculative Decoding" (2023)
7. "Efficiently Scaling Transformer Inference" (2023) — Google
8. "SGLang: Efficient Execution of Structured Language Model Programs" (2024)
9. "A Survey on Efficient Inference for Large Language Models" (2024)

---

## Hardware Setup

### Your Primary Rig: NVIDIA DGX Spark

You have a DGX Spark at home — this is your main lab for the entire roadmap.

| Spec | Detail |
|---|---|
| **GPU** | NVIDIA Blackwell GPU (GB10 Superchip) |
| **Memory** | 128GB unified (CPU+GPU shared) |
| **Max model size** | ~200B parameters quantized, ~65B at FP16 |
| **Connectivity** | ConnectX-7 NIC for multi-node if needed |

**What this means for you:**
- You can run Llama 3.1 70B quantized (INT4) entirely locally — no cloud needed
- You can run Llama 3.1 8B at full FP16 with massive KV cache room
- Quantization experiments are painless — 128GB gives you headroom to compare FP16 vs INT8 vs INT4 side by side
- Multi-model serving is feasible — run a draft + main model for speculative decoding
- All portfolio projects can be built and benchmarked locally
- You can experiment with LoRA multi-adapter serving with plenty of memory

**DGX Spark-specific setup to do early:**
- [ ] Install NVIDIA AI Enterprise stack (comes pre-configured, but verify versions)
- [ ] Set up NVIDIA Container Toolkit for running vLLM/TGI in containers
- [ ] Install vLLM with Blackwell support
- [ ] Verify CUDA version and driver compatibility
- [ ] Run `nvidia-smi` and familiarize yourself with the monitoring output
- [ ] Set up Jupyter Lab for interactive experimentation

### Secondary Rig: Mac Mini M4

| Spec | Detail |
|---|---|
| **Chip** | Apple M4 |
| **Unified Memory** | 16/32GB (shared CPU+GPU+Neural Engine) |
| **Best for** | llama.cpp / MLX inference, edge/on-device experiments |

**How to use it in this roadmap:**
- Run llama.cpp and MLX — Apple Silicon is a first-class target for both
- Great for Phase 2.2: compare the same quantized model on DGX Spark (CUDA) vs Mac Mini (Metal) — real cross-platform benchmarking
- Edge inference experiments (Phase 6.1) — on-device deployment is a growing area
- Use it as a client machine to send requests to your DGX Spark serving models over the network
- Daily driver for reading papers, writing code, light experiments when you don't need the Spark's power

**Mac-specific setup:**
- [ ] Install llama.cpp (Metal-accelerated, native on Apple Silicon)
- [ ] Install MLX and mlx-lm (Apple's ML framework — fast on M-series)
- [ ] Run a 7-8B model quantized (Q4_K_M) via llama.cpp — baseline for edge performance

### Supplementary (Optional, Not Required)

| Option | When to use | Cost |
|---|---|---|
| **Cloud multi-node** (RunPod, Lambda) | Only if you need multi-node distributed inference practice | ~$2-4/hr, use sparingly |

**Recommendation:** Between your DGX Spark and Mac Mini, you cover essentially 100% of this roadmap — CUDA/Blackwell on the Spark, Metal/edge on the Mac. The only reason to touch cloud is multi-node distributed inference (Phase 4.2). Save your money.

---

## Weekly Routine Suggestion

| Day | Focus | Time |
|---|---|---|
| Mon-Fri | Study + hands-on (current phase topic) | 2 hrs |
| Saturday | Project work (portfolio projects) | 3-4 hrs |
| Sunday | Read 1 paper + community catch-up | 1-2 hrs |

---

*Last updated: 2026-03-27*
