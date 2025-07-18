# Track A: 从零预训练 LLM / VLM (From-Scratch Pre-training LLM / VLM)

## 目标 (Goal)

这是教程中最具挑战性但也是最能深入理解大模型本质的路线。我们将走过**从零开始预训练一个 Transformer-based 语言模型**的全过程。尽管完整预训练一个SOTA模型成本高昂，本教程旨在让您：

*   理解模型架构设计的核心原则。
*   掌握预训练的完整技术栈和流程。
*   能够成功运行一个小型“玩具”模型，为未来参与大规模训练奠定基础。

---

### 1. 模型架构设计与选择 (Model Architecture Design & Selection)

一个 Transformer 模型的参数量由其核心超参数决定：层数（L）、隐藏维度（d_model）、注意力头数（h）、词汇表大小（V）以及FFN中间层维度。这些参数在**预训练开始前**就必须根据您的**目标参数量**和**计算预算**一次性确定，训练过程中通常**不会调整**。

#### 参数量计算公式 (Parameter Count Estimation)

一个简化版的 Transformer 模型参数量估算公式（忽略了LayerNorm、Bias等小部分参数）：

$$
\text{Params} \approx V \cdot d_{\text{model}} + L \cdot (2 \cdot d_{\text{model}} \cdot (d_{\text{model}} + \frac{d_{\text{model}}}{h}) + 2 \cdot d_{\text{model}} \cdot d_{\text{ffn}})
$$

其中：
*   $V$: 词汇表大小 (Vocab Size)
*   $d_{\text{model}}$: 隐藏层维度 (Hidden Size)
*   $L$: Transformer 层数 (Number of Hidden Layers)
*   $h$: 注意力头数 (Number of Attention Heads)
*   $d_{\text{ffn}}$: 前馈网络中间层维度 (Intermediate Size)，通常是 $4 \cdot d_{\text{model}}$。

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
    *   **`max_seq_length`**: 根据模型配置的 `max_position_embeddings` 来设置，确保不超过模型的上下文窗口。

2.  **配置分布式环境 (Accelerate)**:
    `accelerate` 是Hugging Face提供的用于简化分布式训练的工具。第一次使用时，您需要配置它：
    ```bash
    accelerate config
    ```
    按照提示进行配置。对于AutoDL上的单机多卡，通常选择：
    *   `all_cuda` 作为处理器。
    *   `multi-GPU` 作为机器类型。
    *   `X` 作为进程数（X是您GPU的数量，例如4）。
    *   **是否使用 DeepSpeed? `yes`。** DeepSpeed是预训练大型模型不可或缺的工具。
    *   **是否使用 BF16 混合精度? `yes`。** 如果您的GPU（如A100, RTX 30/40系列）支持BF16，这将显著节省显存并加速训练。
    *   **DeepSpeed 配置**: 建议选择 `all` (启用ZeRO-3) 或 `ZeRO-2`。您也可以提供自定义的DeepSpeed JSON配置文件。

3.  **启动预训练**:
    在项目根目录下，运行预训练脚本。
    ```bash
    bash scripts/run_pretrain.sh
    ```
    这个脚本会调用 `accelerate launch` 来启动 `src/trainers/pretrain_trainer.py`。训练过程中的损失、学习率等指标会实时上传到 Weights & Biases (W&B)，您可以通过W&B仪表盘进行监控。

#### 关键技术与配置 (`configs/training/pretrain_llm.yaml`)

*   **混合精度训练 (Mixed Precision):**
    *   `bf16: true`: **推荐**。使用 bfloat16 格式进行训练，可以显著降低显存占用并提高计算效率，同时保持与 fp32 相当的精度。需要支持BF16的GPU（NVIDIA Ampere架构及更高版本）。
    *   `fp16: true`: 对于不支持BF16的GPU，可以使用FP16。需要注意梯度缩放以防止下溢。
*   **优化器 (Optimizer):** `AdamW` 是主流选择。配置文件中可以设置学习率、权重衰减等。
*   **学习率调度器 (Scheduler):** `Cosine Annealing` 是最常见的学习率调度策略，它在训练初期逐渐上升（warmup），然后以余弦曲线下降。
*   **Checkpointing:** `save_steps` 参数控制了模型检查点的保存频率。在长时间预训练中，定期保存是防止意外中断导致进度的重要措施。
*   **效率优化:**
    *   **`torch.compile`**: PyTorch 2.0+ 的内置编译器，能自动优化模型计算图，通常能带来显著的训练加速。在`pretrain_trainer.py`中已集成。
    *   **`flash_attention`**: 在模型架构中通过 `attention_type: "flash"` 来启用。

#### 小模型 vs. 大模型：并行策略 (Parallelism Strategies)

当模型或数据过大，无法放入单个GPU时，就需要并行策略。

*   **数据并行 (Data Parallelism, DP / DDP / FSDP):**
    *   每个GPU拥有模型的完整副本。
    *   每个GPU处理不同批次的数据。
    *   梯度在各GPU之间同步（All-reduce）。
    *   **PyTorch DDP** 是基础。
    *   **DeepSpeed ZeRO (Zero Redundancy Optimizer)** 和 **Fully Sharded Data Parallel (FSDP)** 更高效：它们通过分片（sharding）优化器状态、梯度甚至模型参数来降低显存占用。
        *   **ZeRO-1:** 仅分片优化器状态。
        *   **ZeRO-2:** 分片优化器状态和梯度。
        *   **ZeRO-3:** 分片优化器状态、梯度和**模型参数**。这是训练超大模型（千亿参数级别）的关键，因为它能将模型参数的显存占用分摊到所有GPU上。
        *   **FSDP:** PyTorch 原生的模型并行化工具，功能与DeepSpeed ZeRO-2/3相似。

*   **模型并行 (Model Parallelism):**
    *   **流水线并行 (Pipeline Parallelism, PP):** 将模型的层分成若干阶段，每个GPU负责一个阶段。数据在各阶段之间流动，形成一个流水线。
    *   **张量并行 (Tensor Parallelism, TP):** 将单个层（如Attention、FFN）内的矩阵运算拆分到多个GPU上。

**本教程中的实现:**

*   `pretrain_trainer.py` 默认使用 `Accelerate`，它可以无缝对接 `DDP` 和 `DeepSpeed`（如果您在 `accelerate config` 中配置了）。
*   对于教程演示的小模型 (0.5B, 0.8B)，**DeepSpeed ZeRO-2** 或 **FSDP** 通常就足够了。
*   要启用 **DeepSpeed ZeRO-3** 或 **流水线并行**，您需要在 `configs/training/pretrain_llm.yaml` 中指定 `deepspeed_config` 的路径，并创建一个 DeepSpeed JSON 配置文件。我们会在后续章节提供这些配置的示例。

---

### 3. 资源估算 (预训练1B模型)

预训练是一个资源密集且耗时的过程。以下估算仅用于让您对成本有一个概念，**实际成本可能更高**。

*   **GPU**: 至少需要 **4 x A100/A800 80G**。对于1B模型，虽然理论上2张A100 80G可能通过激进的内存优化（如ZeRO-3）勉强跑起来，但4张卡能提供更稳定的训练体验。
*   **时间成本**: 在5GB文本+1M图文数据上进行一个**有效的预训练**（足以让模型学会基本的语言能力，但远未达到SOTA），预计需要 **7-14天** 的连续训练。
*   **金钱成本**: 以AutoDL A100 80G (约¥10/小时) 计算：
    *   4 卡 * ¥10/小时/卡 * 24小时/天 * 10天 = **¥9600**
    *   这仅仅是一个非常粗略的**下限估算**，不包含数据处理、调试、实验迭代的成本。
    *   **本教程的目标是让您成功跑通流程并获得一个能初步理解语言的“玩具”模型，而非一个工业级可用的SOTA模型。** 完整预训练的成本远超普通爱好者的预算。

---
**您已完成Track A。您现在掌握了从零构建和预训练一个 Transformer 模型的理论和实践。这为您深入理解大模型的底层机制打下了最坚实的基础。**

**接下来，请移步至 [docs/05_deployment_and_evaluation.md](./05_deployment_and_evaluation.md)，学习如何评估您训练好的模型并将其部署为服务。**

# END OF FILE: docs/03_track_a_pretraining.md