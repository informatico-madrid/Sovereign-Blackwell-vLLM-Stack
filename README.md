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
- **Agentic Native:** Pre-configured for Roo Code/Cline with custom tool-calling parsers.

---

## 🛠️ Prerequisites
- **OS:** Linux (Ubuntu 24.04+ recommended) with **Kernel 6.14+** (for DMABUF support).
- **GPU:** 2x NVIDIA RTX 5090 (or SM_120 architecture).
- **Driver:** NVIDIA Driver 590.48+.
- **Software:** Docker Engine + NVIDIA Container Toolkit.

---

## 🚀 Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/informatico-madrid/Sovereign-Blackwell-vLLM-Stack
    cd sovereign-ai-stack
    ```

2.  **Configure Environment:**
    ```bash
    cp .env.example .env
    nano .env  # Edit with your setup
    ```
    
    Key variables to configure:
    - `PROJECT_ROOT` - Absolute path to this repository
    - `MODELS_ROOT` - Path to your models directory

3.  **Launch the Stack:**
    ```bash
    ./start.sh start
    ```

---

## 📦 Project Structure

```
sovereign-ai-stack/
├── .env                    # Environment configuration (not versioned)
├── .env.example            # Template for .env
├── start.sh                # Service management script
├── core/
│   ├── parsers/            # Tool call parsers for vLLM
│   │   └── qwen3coder_tool_parser.py
│   └── templates/          # Chat templates
└── deploy/
    └── compose/
        ├── docker-compose.yml
        └── litellm-config.yaml
```

---

## 🔧 Service Management

Use the included `start.sh` script for all operations:

```bash
./start.sh start     # Start all services
./start.sh stop      # Stop all services
./start.sh restart   # Restart all services
./start.sh status    # Show container status
./start.sh logs      # Tail all logs
./start.sh logs vllm-engine  # Tail specific service logs
```

---

## 🔌 Integration: Connecting Roo Code / Cline

This stack is optimized for **Tool Calling**. Configure your agent as follows:

- **API Provider:** OpenAI Compatible
- **Base URL:** `http://localhost:4000/v1` (LiteLLM Proxy)
- **Model ID:** `bunker-agent`
- **Custom Instructions:** Ensure your agent is aware of the 256K context limit.

### Native Tool Parsing
This stack is specifically engineered to bridge the "Handshake Gap" between Qwen3's internal XML protocol and standard JSON agents. 

- **Internal Logic:** The optimized image includes a native `qwen3_coder` parser that handles the translation automatically.
- **Extensibility:** While the stack uses the internal parser by default, developers can inject custom logic by mapping a local parser to the container's internal path (see `Advanced Customization`).
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
- **KV Cache Optimization:** Tuned to 0.84 utilization to maximize context without OOM.

---

## 🛠️ Advanced: Custom Tool Parsers
If you need to modify the tool-calling logic or support a different protocol:
1. Place your parser in `core/parsers/my_custom_parser.py`.
2. Add the volume mount to `deploy/compose/docker-compose.yml`:
   ```yaml
   volumes:
     - ./core/parsers/my_custom_parser.py:/vllm-workspace/vllm/model_executor/layers/fused_moe/custom_parser.py
   ```
3. Update the --tool-call-parser flag in the command section.

---

## ⚖️ Credits & Open Source Compliance
- **Inference Engine:** [vLLM Project](https://github.com/vllm-project/vllm) (Apache 2.0).
- **Tool Parsing:** Derived from community efforts in the vLLM ecosystem.
- **Model:** [Alibaba Qwen Team](https://github.com/QwenLM/Qwen3).