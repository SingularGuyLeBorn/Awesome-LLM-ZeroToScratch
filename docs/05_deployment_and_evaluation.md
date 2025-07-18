# Part 4: 进阶、部署与评估 (Advanced, Deployment & Evaluation)

## 目标 (Goal)

在本教程的最终阶段，我们将把训练好的模型从研究原型转化为可用的服务。这包括实现模型推理、对其性能和能力进行严格评估，并探讨其在实际应用中的部署策略和未来扩展方向。

---

### 1. 长文本能力扩展 (Long-Text Capability Extension)

现代LLM需要处理越来越长的上下文。虽然基础Transformer模型的 `max_position_embeddings` 是固定的，但可以通过技术手段进行扩展。

*   **RoPE Scaling (NTK-aware, Dynamic):**
    *   **原理:** RoPE (Rotary Positional Embeddings) 是一种相对位置编码，通过旋转矩阵将位置信息编码到Attention的Q和K中。通过调整 RoPE 的基（base）或通过“NTK-aware”缩放，可以在不重新训练的情况下，有效地扩展模型的上下文窗口。
    *   **实现:** 这通常在模型加载时，通过修改模型的配置或直接修改 `transformers` 库的 `LlamaRotaryEmbedding` 类来完成。例如，将 `rope_theta` (默认10000) 调整到更大的值，或应用动态缩放因子。
    *   **本教程的实现:** 我们的 `src/models/language_model.py` 中的 `BaseLLM` 已经预留了位置编码的集成点。对于 `transformers` 库加载的模型，这些调整通常在 `AutoModel.from_pretrained` 之后通过修改 `model.config` 或直接在生成时传入 `rope_theta` 参数来完成。

*   **Position Interpolation (PI):**
    *   **原理:** PI 是一种简单而有效的方法，通过线性插值原始位置编码来扩展上下文。它将长序列的位置索引线性地“压缩”到模型原始的最大位置嵌入范围内。
    *   **实现:** PI 通常涉及修改模型的 `forward` 方法中处理位置编码的部分。

*   **训练长度:** 配置文件中的 `max_seq_length` 参数直接决定了模型在训练时能够处理的最大序列长度。如果您要预训练一个真正支持长上下文的模型，必须从一开始就使用长序列数据进行训练。

---

### 2. 模型推理与部署 (Model Inference & Deployment)

训练好的模型最终要投入使用。我们将提供一个简单的推理脚本，并讨论工业级的部署方案。

#### 2.1. 本地推理 CLI

*   **脚本**: `src/inference/inference.py`
*   **功能**: 加载您训练好的SFT或DPO模型（包括LoRA适配器），并提供一个简单的命令行界面，让您可以在终端中与模型进行交互。它会自动处理LoRA权重合并，以便进行高效推理。

*   **运行命令**:
    ```bash
    # 假设你已经完成了SFT训练，并且模型保存在 './checkpoints/sft-tinyllama-guanaco/final_model'
    bash scripts/run_inference.sh ./checkpoints/sft-tinyllama-guanaco/final_model
    ```
    *   你可以选择传递不同的模型路径来测试你的DPO模型或其他训练好的模型。
    *   如果推理过程中遇到显存问题，可以尝试减少 `max_new_tokens` 参数。

#### 2.2. 工业级部署方案

将LLM部署到生产环境需要考虑性能、成本、并发和可维护性。

*   **轻量级部署框架 (推荐):**
    *   **`vLLM`**: 一个高性能、低延迟的LLM推理和服务库。它使用PagedAttention（比FlashAttention更进一步）来高效管理GPU显存，并支持Continuous Batching，显著提高吞吐量。
    *   **`TGI (Text Generation Inference)`**: Hugging Face 开发的LLM推理服务器，由Rust实现，高性能且功能丰富，支持多种优化（如FlashAttention、quantization、stream generation）。
    *   **部署步骤**:
        1.  **打包模型**: 将您的模型（如果使用了LoRA，先执行 `model.merge_and_unload()` 将适配器合并到基础模型中，这在 [`docs/04_track_b_finetuning.md`](./04_track_b_finetuning.md) 中有概念性代码）转换为 `transformers` 格式。
        2.  **构建Docker镜像**: 使用 `vLLM` 或 `TGI` 提供的 Dockerfile 作为基础，将您的模型打包到镜像中。
        3.  **启动服务**: 在GPU实例上运行Docker容器，暴露API接口。
    *   **示例 (概念性 `Dockerfile` 片段):**
        ```dockerfile
        # FROM vllm/vllm-openai:latest
        # COPY your_merged_model_path /model
        # ENV MODEL=/model
        # CMD python -m vllm.entrypoints.api_server --model ${MODEL} --port 8000
        ```

*   **量化 (Quantization):**
    *   **目标**: 显著降低模型的显存占用和计算量，从而降低部署成本。
    *   **技术**:
        *   **`bitsandbytes`**: 运行时量化，支持4-bit (NF4) 和 8-bit 量化，可以在加载模型时直接应用。我们的SFT/DPO训练器中已经使用了`load_in_4bit=True`。
        *   **AWQ/GPTQ**: 离线量化技术，在模型训练后进行，将模型权重压缩到4-bit或更低，对推理性能影响小，但需要专门的量化工具。
    *   **实现**: 对于 `vLLM` 和 `TGI`，它们通常内置了对这些量化模型的支持。您只需在加载时指定量化类型。

---

### 3. 模型评估与对齐 (Model Evaluation & Alignment)

“无懈可击”的系统需要客观的证明。模型评估是科学验证模型能力的关键。

#### 3.1. 标准基准测试 (Standard Benchmarks)

*   **目的**: 衡量模型在特定学术任务上的客观能力，例如常识推理、数学、代码生成等。
*   **工具**: `lm-evaluation-harness` (通常简称为 `lm-eval`) 是业界标准的LLM评估工具。它集成了大量的学术数据集。
    *   **安装**: `pip install lm-eval[hf]`
*   **典型数据集**:
    *   **MMLU (Massive Multitask Language Understanding):** 衡量模型在57个不同学科（科学、人文、社会科学等）上的多任务理解能力。
    *   **GSM8K (Grade School Math 8K):** 衡量小学数学应用题的解决能力。
    *   **HumanEval:** 衡量代码生成能力。
*   **运行评估 (`scripts/run_evaluation.sh` & `src/evaluation/evaluate_llm.py`):**
    我们的 `src/evaluation/evaluate_llm.py` 脚本提供了一个**概念性**的入口。实际运行 `lm-eval` 通常需要直接在命令行调用。

    ```bash
    # 概念性运行脚本 (运行后会提示如何使用lm-eval的实际命令)
    bash scripts/run_evaluation.sh ./checkpoints/sft-tinyllama-guanaco/final_model "mmlu_flan_n_shot,arc_challenge" 20
    ```
    **注意:** 完整的 `lm-evaluation-harness` 运行可能需要数小时到数天，并消耗大量计算资源。

#### 3.2. 对齐评估 (Alignment Evaluation)

*   **目的**: 衡量模型生成内容的“有用性”、“无害性”和“遵循指令”的能力，即模型与人类偏好的对齐程度。
*   **工具与方法**:
    *   **MT-Bench**: 一个多轮对话基准测试，包含各种难度和类型的提示。通常使用一个更强大的“裁判”LLM（如 GPT-4）来评估模型生成的回复质量。
    *   **AlpacaEval**: 另一个自动化评估框架，通过对比模型生成内容与参考答案，并用裁判模型打分。
    *   **人工评估**: 最终极的评估方式，但成本极高。

*   **实现流程**:
    1.  **数据生成**: 使用您的模型对MT-Bench或AlpacaEval的提示集进行推理，生成回复。
    2.  **裁判打分**: 将模型生成的回复、原始提示以及（如果适用）参考回复提交给一个强大的外部LLM（如 GPT-4、Claude Opus）进行打分。
    3.  **结果分析**: 汇总裁判模型的评分，并进行统计分析。
    *   这通常需要编写独立的脚本，与外部API进行交互。

---

### 4. 环境与部署蓝图 (Environment & Deployment Blueprint)

本章节将概述在生产环境中部署和维护LLM所需的关键元素。

#### 4.1. 依赖清单 (Dependency Manifest)

*   在 `requirements.txt` 中已明确指定所有必需的软件依赖及其精确版本。这是实现可复现部署的第一步。
*   **生产环境额外依赖**: 对于生产部署，可能需要 `vLLM`, `text-generation-inference`, `fastapi`, `uvicorn` 等。

#### 4.2. 构建与打包指令 (Build & Packaging Commands)

*   **PyTorch模型**: 通常不需要特殊的“构建”步骤，直接使用Python包管理器安装依赖。
*   **Docker**: 推荐使用Docker将模型、代码和依赖打包成一个可移植的镜像。
    *   **`Dockerfile`**: 定义了构建镜像的步骤。
    *   **`docker build -t your-llm-service .`**: 构建Docker镜像。

#### 4.3. 环境设置指令 (Environment Setup Commands)

*   **生产环境**: 除了Python和PyTorch环境外，还需要：
    *   **CUDA 驱动**: 确保GPU服务器安装了与PyTorch版本兼容的NVIDIA CUDA驱动。
    *   **Docker/Containerd**: 用于运行容器化应用。
    *   **网络配置**: 确保模型服务端口可访问。

#### 4.4. 运行程序指令 (Execution Commands)

*   **本地开发/测试**:
    *   激活Conda/venv环境。
    *   `python src/inference/inference.py <model_path>` (本地CLI)
*   **生产环境 (Docker)**:
    *   `docker run --gpus all -p 8000:8000 your-llm-service:latest`
    *   这会启动一个容器，并将容器的8000端口映射到主机的8000端口，用于接收推理请求。

#### 4.5. 可观测性蓝图 (Observability Blueprint)

在生产环境中，监控和排查问题至关重要。

*   **日志策略 (Logging Strategy):**
    *   **级别**: `INFO` 用于关键业务操作，`WARNING` 用于潜在问题，`ERROR` 用于故障。
    *   **格式**: JSON格式化日志，便于日志收集系统（如ELK Stack, Loki）解析。
    *   **内容**: 记录请求ID、用户ID、输入提示、模型输出、响应时间、错误信息等。
    *   **收集与存储**: 使用 `loguru` 或 `logging` 库，日志写入到标准输出，由容器编排工具（Kubernetes）或日志代理（Promtail, Filebeat）收集并发送到中心化日志系统。
*   **指标与监控 (Metrics & Monitoring):**
    *   **系统指标**: GPU利用率、显存占用、CPU利用率、内存使用。
    *   **业务指标**: 请求QPS（每秒查询数）、平均响应时间、P99延迟、错误率、缓存命中率。
    *   **工具**: Prometheus + Grafana 用于指标收集、存储和可视化。
*   **链路追踪 (Distributed Tracing):**
    *   对于分布式系统，使用OpenTelemetry等工具进行链路追踪，可以追踪一个请求从用户端到后端各个微服务（包括LLM推理服务）的全过程，帮助定位性能瓶颈和错误。

#### 4.6. 弹性与灾备策略 (Resilience & Disaster Recovery Strategy)

设计健壮的系统以应对故障。

*   **错误处理与异常**: 模型推理中的OOM、CUDA错误、无效输入等，都应有优雅的错误捕获和用户友好提示。
*   **熔断 (Circuit Breaker):** 当LLM服务依赖的某个外部服务（如数据库、向量DB）持续故障时，暂时停止向其发送请求，防止级联失败。
*   **限流 (Rate Limiting):** 限制单位时间内对LLM服务的请求数量，防止过载。
*   **重试 (Retry Mechanisms):** 对于瞬时网络错误或轻微的API故障，客户端应具备自动重试机制。
*   **健康检查**: K8s或其他容器编排工具应定期对LLM服务进行健康检查。
*   **数据备份与恢复**: 对于任何涉及数据存储的LLM应用（如微调数据、用户对话历史），必须有定期的备份策略和可验证的恢复流程。模型权重也应存储在持久化存储中。

---