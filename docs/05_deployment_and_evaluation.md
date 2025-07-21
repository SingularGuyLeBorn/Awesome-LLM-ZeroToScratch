# Part 5: 部署、评估与展望 (Deployment, Evaluation & Outlook)

## 目标 (Goal)

在本教程的最终阶段，我们将学习如何与训练好的模型进行交互，评估其性能，并探讨工业级的部署策略和未来方向。

---

### 1. 模型推理：与你的模型对话 (Model Inference)

训练好的模型需要一个入口来展示其能力。我们提供了一个简单的命令行推理脚本。

*   **脚本**: `src/inference/inference.py`
*   **核心功能**:
    *   能自动加载完整的预训练模型，或基础模型+LoRA适配器。
    *   如果加载的是LoRA适配器，它会自动执行`model.merge_and_unload()`，将适配器权重合并到基础模型中，以实现最高效的推理。
    *   提供一个流式（streaming）的命令行界面，让您可以在终端中与模型实时交互。
    *   **CPU/GPU自适应**：脚本能自动检测环境，在CPU或GPU上运行。

#### **运行推理命令**

*   **前提**: 您已经成功运行了SFT或DPO流程，并生成了模型检查点。
*   **命令**:
    ```bash
    # 与SFT（CPU版）训练后的模型对话
    python -m src.inference.inference ./checkpoints/sft-tinyllama-guanaco-cpu/final_model --max_new_tokens 100

    # 或者与DPO（CPU版）训练后的模型对话
    python -m src.inference.inference ./checkpoints/dpo-tinyllama-guanaco-cpu/final_model --max_new_tokens 100
    ```
    *   将路径替换为您实际的模型输出目录。
    *   `--max_new_tokens` 参数控制模型每次生成内容的最大长度。
    *   在CPU上推理速度会较慢，请耐心等待。

---

### 2. 模型评估：我的模型有多好？ (Model Evaluation)

“感觉不错”不是科学的评价。我们需要客观的基准来衡量模型的能力。

#### 2.1. 标准基准测试 (Standard Benchmarks)

*   **目的**: 衡量模型在特定学术任务上的客观能力，例如常识推理、数学、代码生成等。
*   **业界标准工具**: `lm-evaluation-harness`，它集成了大量的学术数据集。
    *   **安装**: `uv pip install lm-eval`
*   **典型数据集**:
    *   **MMLU**: 衡量模型在57个学科上的多任务理解能力。
    *   **GSM8K**: 衡量小学数学应用题的解决能力。
    *   **HumanEval**: 衡量代码生成能力。

*   **运行评估 (概念性)**:
    我们的 `src/evaluation/evaluate_llm.py` 脚本提供了一个**概念性**的入口，它会加载模型并展示如何调用评估工具。
    ```bash
    # 对SFT（CPU版）训练后的模型进行概念性评估
    python -m src.evaluation.evaluate_llm ./checkpoints/sft-tinyllama-guanaco-cpu/final_model --tasks mmlu --num_samples 10
    ```
    **注意**: 完整的 `lm-evaluation-harness` 运行可能需要数小时，并消耗大量计算资源。在CPU上运行完整评估是不现实的，此步骤主要用于验证模型加载和评估流程本身是通的。

#### 2.2. 对齐评估 (Alignment Evaluation)

*   **目的**: 衡量模型生成内容的“有用性”、“无害性”和“遵循指令”的能力，即与人类偏好的对齐程度。
*   **主流方法**:
    *   **MT-Bench**: 一个多轮对话基准测试，通常使用一个更强大的“裁判”LLM（如 GPT-4）来评估模型生成的回复质量。
    *   **AlpacaEval**: 另一个自动化评估框架，通过对比模型生成内容与参考答案，并用裁判模型打分。

---

### 3. 工业级部署与展望 (Industrial Deployment & Outlook)

将LLM部署到生产环境，需要重点考虑性能、成本和稳定性。

*   **推理服务框架 (推荐)**:
    *   **`vLLM`**: 一个为LLM推理设计的、性能极高的服务库。其核心技术`PagedAttention`能极大提升GPU显存利用率和吞吐量。
    *   **`TGI (Text Generation Inference)`**: Hugging Face官方推出的高性能推理服务器。
    *   **部署流程**: 通常是将模型（LoRA合并后）打包到`vLLM`或`TGI`的Docker容器中，通过API对外提供服务。

*   **量化 (Quantization)**:
    *   **目标**: 进一步压缩模型大小，降低显存占用，加速推理。
    *   **技术**:
        *   **`bitsandbytes`**: 在加载时进行量化，方便易用。
        *   **AWQ/GPTQ**: 训练后量化技术，对模型进行离线压缩，通常性能损失更小。
    *   在生产环境中，通常会使用经过AWQ/GPTQ量化并由`vLLM`等框架加载的模型，以达到最佳的成本效益。

*   **长文本能力扩展**:
    *   真实世界的应用往往需要处理长篇文档。
    *   **RoPE Scaling / Position Interpolation (PI)** 等技术可以在不重新训练的情况下，扩展模型的上下文窗口，是长文本应用的关键。

### 最终蓝图：一个完整的LLM应用系统

一个生产级的LLM应用，远不止模型本身。它是一个复杂的系统，通常包括：
1.  **用户前端**: 与用户交互的界面。
2.  **业务后端**: 处理业务逻辑，调用LLM服务。
3.  **LLM推理服务**: 由`vLLM`等框架部署，负责模型计算。
4.  **RAG系统 (可选)**:
    *   **数据源**: 私有文档、数据库等。
    *   **ETL流水线**: 将数据清洗、切块、向量化。
    *   **向量数据库**: (如Chroma, Milvus) 存储向量，用于快速检索。
5.  **可观测性系统**:
    *   **日志 (Logging)**: ELK, Loki。
    *   **指标 (Metrics)**: Prometheus, Grafana。
    *   **追踪 (Tracing)**: Jaeger, OpenTelemetry。
6.  **CI/CD流水线**: 自动化测试、构建、部署。

![LLM System Blueprint](https://miro.medium.com/v2/resize:fit:1400/1*9C2t2W23p32B8k_M4WoL8g.png)
*图片来源: A Blueprint for Building Your Own LLM Apps*

---
**教程至此结束。**

恭喜您！您已经走完了从数据处理、模型预训练、多阶段微调，到最终部署和评估的全过程。更重要的是，您亲历了在真实（且充满挑战）的环境中解决问题的过程。您现在不仅掌握了理论知识，更具备了宝贵的实战经验，这为您未来在大型语言模型领域的深入探索和创新奠定了坚实的基础。

# END OF FILE: docs/05_deployment_and_evaluation.md