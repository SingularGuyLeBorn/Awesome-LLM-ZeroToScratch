# Track A: 从零预训练 LLM / VLM (From-Scratch Pre-training LLM / VLM)

## 目标 (Goal)

这是教程中最具挑战性但也是最能深入理解大模型本质的路线。我们将走过**从零开始预训练一个 Transformer-based 语言模型**的全过程。尽管完整预训练一个SOTA模型成本高昂，本教程旨在让您：

*   理解模型架构设计的核心原则。
*   掌握预训练的完整技术栈和流程。
*   能够成功运行一个小型“玩具”模型，为未来参与大规模训练奠定基础。

---

### 1. 模型架构设计与选择 (Model Architecture Design & Selection)

一个 Transformer 模型的参数量由其核心超参数决定：层数（L）、隐藏维度（d_model）、注意力头数（h）、词汇表大小（V）以及FFN中间层维度。这些参数在**预训练开始前**就必须根据您的**目标参数量**和**计算预算**一次性确定，训练过程中通常**不会调整**。

#### 模块化与热插拔实现 (`src/models/`)

我们的代码库设计高度模块化，允许您通过YAML配置文件“热插拔”不同的组件：

*   **`src/models/language_model.py`**: 这是核心的模型构建入口。它根据配置文件动态地组装模型，例如选择是使用密集型FFN还是MoE层。
*   **`src/models/attention/standard_attention.py`**: 标准的多头自注意力实现。
*   **`src/models/attention/flash_attention.py`**: FlashAttention 的封装。如果您的GPU支持（Ampere架构或更新，如A100, RTX 30/40系列），它能极大加速训练并节省显存。
*   **`src/models/ffn.py`**: 包含了普通FFN和SwiGLU的实现。
*   **`src/models/moe.py`**: Mixture-of-Experts (MoE) 层的实现，这是构建稀疏大模型的关键。

**模型配置示例 (见 `configs/model/`):**

*   **`0.5B_dense.yaml`**: 定义了一个约0.5B参数的密集型LLM架构。
*   **`0.8B_moe.yaml`**: 定义了一个约0.8B参数的MoE型LLM架构。它包含8个专家，每个Token激活其中2个专家。**请注意，MoE模型虽然总参数量大（例如0.8B），但实际激活的参数量通常远小于其总参数量，这使其计算成本更低。**

---

### 2. 预训练配置与执行 (Pre-training Configuration & Execution)

我们将使用 `Hugging Face Accelerate` 来管理分布式训练，它提供了对 `DeepSpeed` 和 `PyTorch FSDP` 的无缝支持。

1.  **检查模型和数据配置**:
    *   打开 `configs/training/pretrain_llm.yaml`。
    *   **`model_config_path`**: 确保它指向您想预训练的模型配置，例如 `configs/model/0.5B_dense.yaml`。
    *   **`dataset_dir`**: 确保它指向您在Part 2中处理好的数据集目录，例如 `./data/processed/wikitext`。

2.  **配置分布式环境 (Accelerate)**:
    `accelerate` 是Hugging Face提供的用于简化分布式训练的工具。第一次使用时，您需要配置它：
    ```bash
    accelerate config
    ```
    按照提示进行配置。对于AutoDL上的单机多卡，通常选择：`multi-GPU`, `yes` (DeepSpeed), `yes` (BF16), 和 `ZeRO-2`。

3.  **启动预训练**:
    在项目根目录下，运行预训练脚本。
    ```bash
    bash scripts/run_pretrain.sh
    ```
    这个脚本会调用 `accelerate launch` 来启动 `src/trainers/pretrain_trainer.py`。训练过程中的损失、学习率等指标会实时上传到 Weights & Biases (W&B)，您可以通过W&B仪表盘进行监控。

#### Bedrock实践：优雅地处理权重共享 (Bedrock Practice: Elegantly Handling Weight Tying)

在许多语言模型中，**词嵌入层 (Token Embeddings)** 和 **输出预测层 (LM Head)** 的权重是共享的，这被称为**权重绑定 (Weight Tying)**。这样做可以显著减少模型参数量并提高性能。

**挑战 (The Challenge):**
在开发自定义模型时，一个非常常见且棘手的 `RuntimeError` 是关于“权重共享与配置不匹配”的。这是因为 `Hugging Face Transformers` 在保存模型时，会严格检查两件事：
1.  模型在内存中的**实际结构**：`lm_head` 的权重是否真的指向 `embed_tokens` 的权重。
2.  模型的**配置文件 (`config.json`)**：`tie_word_embeddings` 字段是否为 `true`。

当使用 `Accelerate` 或 `DeepSpeed` 等分布式框架时，它们会对模型进行“包装”以实现并行化。在这个过程中，模型的原始配置有时会与包装后的模型状态不同步，导致保存时出现“结构”与“配置”不匹配的错误。

**优雅的解决方案 (The Elegant Solution):**
我们最终的解决方案是从根本上解决这个问题，使其符合框架的最佳实践。

1.  **依赖框架内置机制**: 我们修改了 `src/models/language_model.py` 中的 `BaseLLM` 类，使其完全继承自 `transformers.PreTrainedModel`。我们**移除了所有手动的权重绑定逻辑**。
2.  **使用官方API**: `PreTrainedModel` 基类提供了一个名为 `tie_weights()` 的官方方法。我们在 `BaseLLM` 的 `__init__` 方法的末尾调用了 `self.tie_weights()`。

这个方法是专门为处理权重绑定而设计的。它会**自动完成权重共享，并同步更新模型的配置对象**，确保模型的实际结构和它的“身份证”（配置文件）在任何时候都是完全一致的。

通过这个修改，我们的 `BaseLLM` 类变成了一个行为标准、可预测的 `transformers` 模型。它从初始化那一刻起就是自洽的，从而**从根本上消除了**在任何分布式训练场景下可能出现的配置不匹配错误。我们的训练脚本 `pretrain_trainer.py` 也因此变得更加简洁，不再需要任何临时的补丁代码。这是一个典型的从“能用”到“健壮可用”的工程实践案例。

#### 小模型 vs. 大模型：并行策略 (Parallelism Strategies)

当模型或数据过大，无法放入单个GPU时，就需要并行策略。

*   **数据并行 (Data Parallelism - DP / DDP / FSDP / DeepSpeed ZeRO):** 每个GPU拥有模型的（部分或全部）副本，处理不同批次的数据。梯度在各GPU之间同步。DeepSpeed ZeRO技术通过分片（sharding）优化器状态、梯度和模型参数，极大降低了单张GPU的显存需求，是训练大模型的关键。
*   **模型并行 (Model Parallelism - Pipeline / Tensor):** 将模型的不同部分（层或层内操作）切分到不同的GPU上。

本教程的 `Accelerate` 配置可以无缝启用 **DeepSpeed ZeRO**，对于我们演示的小模型，**ZeRO-2** 已经足够。对于70B+的巨型模型，通常需要 **ZeRO-3** 结合张量并行和流水线并行。

---

### 3. 资源估算 (预训练1B模型)

预训练是一个资源密集且耗时的过程。以下估算仅用于让您对成本有一个概念，**实际成本可能更高**。

*   **GPU**: 至少需要 **4 x A100/A800 80G**。
*   **时间成本**: 在5GB文本数据上进行一个**有效的预训练**，预计需要 **7-14天** 的连续训练。
*   **金钱成本**: 以AutoDL A100 80G (约¥10/小时) 计算，10天的成本可能接近 **¥9600**。这仅仅是粗略的下限估算。

**本教程的目标是让您成功跑通流程并获得一个能初步理解语言的“玩具”模型，而非一个工业级可用的SOTA模型。**

---
**您已完成Track A。您现在掌握了从零构建和预训练一个 Transformer 模型的理论和实践。这为您深入理解大模型的底层机制打下了最坚实的基础。**

**接下来，请移步至 [docs/05_deployment_and_evaluation.md](./05_deployment_and_evaluation.md)，学习如何评估您训练好的模型并将其部署为服务。**