# Sovereign AI Stack: Enterprise-Grade Inference for Legacy Modernization

A high-performance, on-premise inference stack optimized for **NVIDIA Blackwell (RTX 5090)** architecture and **Linux Kernel 6.14**. Engineered for ultra-low latency agentic workflows (Roo Code / Cline) without data exfiltration.

---

## 📊 Performance Benchmarks (Dual RTX 5090 / sm_120)
* **Model:** Qwen3-Coder-30B-A3B (MoE)
* **Ingestion Speed (Prefill):** 5,809 tokens/s 🚀
* **Verified TTFT (250K context):** 43.03s (Time To First Token) 🏆
* **Generation Throughput:** 43.75 tokens/s
* **Context Capacity:** 262,144 tokens (Hardware Limit Validated)

---

## 🏗️ Project Philosophy
In the era of massive legacy codebases, the "Privacy Gap" prevents enterprises from using cloud-based LLMs for refactoring. This stack provides a **Sovereign Alternative**:
- **Total Data Privacy:** Zero exfiltration. Everything runs on your hardware.
- **Deep Context:** Massive 256K window allows "feeding" entire modules (100+ files) into the LLM at once.
- **Agentic Native:** Pre-configured for Roo Code/Cline with custom tool-calling parsers.

---

## 🛠️ Prerequisites
- **OS:** Linux (Ubuntu 24.04+ recommended) with **Kernel 6.14+** (for DMABUF support).
- **GPU:** 2x NVIDIA RTX 5090 (or SM_120 architecture).
- **Driver:** NVIDIA Driver 590.48+.
- **Software:** Docker Engine + NVIDIA Container Toolkit.

---

## 🚀 Quick Start (One-Click Deploy)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-user/sovereign-ai-stack.git](https://github.com/your-user/sovereign-ai-stack.git)
    cd sovereign-ai-stack
    ```

2.  **Configure Environment:**
    ```bash
    cp .env.example .env
    # Edit .env with your HF_TOKEN and paths
    nano .env 
    ```

3.  **Launch the Stack:**
    ```bash
    docker compose -f deploy/compose/docker-compose.yml up -d
    ```

---

## 🔌 Integration: Connecting Roo Code / Cline

This stack is optimized for **Tool Calling**. Configure your agent as follows:

- **API Provider:** OpenAI Compatible
- **Base URL:** `http://localhost:4000/v1` (LiteLLM Proxy)
- **Model ID:** `bunker-agent`
- **Custom Instructions:** Ensure your agent is aware of the 256K context limit.

### Why the Custom Parser?
Qwen 3 MoE uses a specific XML protocol for tool invocation. This stack includes a **Bespoke Tool Parser** (`core/parsers/qwen3coder_tool_parser.py`) that translates these XML calls into standard JSON for Roo Code, resolving the "Handshake Gap" common in standard vLLM deployments.

---

## 🔍 Monitoring & Observability
Access **Langfuse** at `http://localhost:3000` to:
- Audit agent decisions in real-time.
- Debug long-context attention degradation.
- Monitor token usage and latency per request.

---

## ⚙️ Low-Level Optimizations
- **NCCL Tuning:** Forced `NCCL_DMABUF_ENABLE=1` to leverage Blackwell's native memory subsystem.
- **FlashInfer Backend:** Bypassing `flash-attn` symbol conflicts on SM_120 for stable inference.
- **KV Cache Optimization:** Tuned to 0.90 utilization to maximize context without OOM.

---

## ⚖️ Credits & Open Source Compliance
- **Inference Engine:** [vLLM Project](https://github.com/vllm-project/vllm) (Apache 2.0).
- **Tool Parsing:** Derived from community efforts in the vLLM ecosystem.
- **Model:** [Alibaba Qwen Team](https://github.com/QwenLM/Qwen3).