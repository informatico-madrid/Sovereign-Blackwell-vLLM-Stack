# Sovereign AI Stack: Enterprise-Grade Inference for Legacy Modernization

A high-performance, on-premise inference stack optimized for **NVIDIA Blackwell (RTX 5090)** architecture and **Linux Kernel 6.14**. Engineered for ultra-low latency agentic workflows (Roo Code / Cline) without data exfiltration.

---

## 📊 Performance Benchmarks (Dual RTX 5090 / sm_120)
* **Model:** Qwen3-Coder-30B-A3B (MoE) - FP8 Precision
* **Ingestion Speed (Prefill):** 5,809 tokens/s 🚀
* **Verified TTFT (250K context):** 43.03s (Time To First Token) 🏆
* **Generation Throughput:** 43.75 tokens/s
* **Context Capacity:** 262,144 tokens (Native Limit Validated)

---

## 🏗️ Project Philosophy
In the era of massive legacy codebases, the "Privacy Gap" prevents enterprises from using cloud-based LLMs for refactoring. This stack provides a **Sovereign Alternative**:
- **Total Data Privacy:** Zero exfiltration. Everything runs on your hardware.
- **Deep Context:** Massive 256K window allows "feeding" entire modules (100+ files) into the LLM at once.
- **Agentic Native:** Profile-driven infrastructure pre-configured for Roo Code/Cline with custom tool-calling parsers.

---

## 🛠️ Prerequisites
- **OS:** Linux (Ubuntu 24.04+ recommended) with **Kernel 6.14+** (for DMABUF support).
- **GPU:** 2x NVIDIA RTX 5090 (SM_120 architecture).
- **Driver:** NVIDIA Driver 590.48+.
- **Software:** Docker Engine + NVIDIA Container Toolkit.

---

## 🚀 Quick Start

1. **Clone and Configure:**
   ```bash
   git clone https://github.com/informatico-madrid/Sovereign-Blackwell-vLLM-Stack
   cd sovereign-ai-stack
   cp .env.example .env  # Configure global hardware/paths
   ```

2. **Launch with a Profile:**
   The stack uses a **Profile-Driven Factory Pattern**. To start the default optimized model:
   ```bash
   ./start.sh start qwen3-30b-fp8
   ```

---

## 🔧 Service Management & Profile Architecture

The infrastructure decouples launch logic from orchestration. Each model is defined by a "Recipe" in `deploy/profiles/*.env`.

### Commands
```bash
./start.sh start <profile_name>   # Start specific model profile
./start.sh stop                 # Stop all services
./start.sh restart <profile>    # Clean VRAM and restart profile
./start.sh logs vllm-engine     # Tail inference logs
```

### Adding New Models (The Factory Pattern)
To add a new model, simply create `deploy/profiles/my-new-model.env`:
```bash
SERVED_MODEL_NAME=my-model
MODEL_PATH=${MODELS_ROOT}/path-to-weights
VLLM_USE_V1=1
VLLM_LAUNCH_COMMAND="--model /model_dir --gpu-memory-utilization 0.86 --max-model-len 262144 ..."
CPU_SET=0-7
```

---

## 🔌 Integration: Connecting Roo Code / Cline

- **API Provider:** OpenAI Compatible
- **Base URL:** `http://localhost:4000/v1` (LiteLLM Proxy)
- **Model ID:** Match the `SERVED_MODEL_NAME` defined in your active profile.

### Native Tool Parsing
Specifically engineered to bridge the "Handshake Gap" between Qwen3's internal XML protocol and standard JSON agents using the optimized `qwen3_coder` parser.

---

## 🔍 Monitoring & Observability
Access **Langfuse** at `http://localhost:3000` to audit agent decisions, monitor token usage, and debug long-context attention degradation in real-time.

---

## ⚙️ Low-Level Optimizations
- **NCCL Tuning:** Forced `NCCL_DMABUF_ENABLE=1` to leverage Blackwell's native memory subsystem.
- **FlashInfer Backend:** Bypassing `flash-attn` symbol conflicts on SM_120 for stable inference.
- **KV Cache Optimization:** Tuned to 0.86 utilization to maximize context without OOM on Blackwell.

---

## ⚖️ Credits & Open Source Compliance
- **Inference Engine:** [vLLM Project](https://github.com/vllm-project/vllm).
- **Tool Parsing:** Native Blackwell-optimized implementations.
- **Model:** [Alibaba Qwen Team](https://github.com/QwenLM/Qwen3).