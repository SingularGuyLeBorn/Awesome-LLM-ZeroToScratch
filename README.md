# Awesome-LLM-ZeroToScratch: 从零到一的LLM大模型训练全流程教程

[![开源协议: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SingularGuyLeBorn/Awesome-LLM-ZeroToScratch)

一份端到端的、代码驱动的大语言模型（LLM）与视觉语言模型（VLM）预训练及微调教程。本项目的设计目标是既能做到**保姆级的平易近人**，又能达到**工业级的稳健可靠**。

本仓库是作者呕心沥血制作，它不仅确保了理论的清晰性、实践的健壮性，更通过我们亲身经历的**史诗级调试之旅**，证明了在各种复杂环境（从**高端云端GPU**到**普通个人电脑的纯CPU环境**）下的**完全可复现性与无缝运行能力**。

## 核心理念：保姆级与工业级的完美融合

本项目基于一种双重目标的哲学精心打造，旨在同时满足不同用户的需求，并解决真实开发中的痛点：

*   **保姆级 (Nanny-Level):**
    *   提供**极其详尽的、手把手的指南**，包含可以直接运行的代码和命令。
    *   针对**AutoDL等云平台**进行了优化，同时确保在**纯CPU环境**下也能成功运行。
    *   我们为你**踩平了数十个真实世界中才会遇到的坑**（从网络下载到内存管理，从库API冲突到数据格式不匹配），并将这些宝贵经验沉淀到代码和[专属的踩坑指南](./docs/06_troubleshooting_guide.md)中，让你无需在环境配置和代码调试的泥潭中挣扎。
*   **工业级 (Industrial-Grade):**
    *   代码库高度模块化，所有实验均由YAML配置文件驱动，为可扩展性而生。
    *   清晰地展示了将模型从0.5B参数扩展到70B+巨型模型所需的关键架构与策略调整（例如，并行策略、内存优化）。
    *   我们的数据加载和环境配置脚本具备**极致的健壮性**（如带**自愈功能的智能数据引擎**），能够从容应对网络波动和缓存损坏等棘手问题，确保数据流水线的稳定高效。

## 项目状态：全系统通行！✅

经过一场漫长而富有成效的调试征途，本项目的所有核心流水线均已在**纯CPU环境**下得到充分验证，并可正常运行，最终输出的日志信息干净、专业、可读。

*   [x] **环境配置**: 针对CPU和GPU环境提供了独立、稳健的配置方案。
*   [x] **数据流水线**: 自动化数据集下载、清洗、处理，并训练分词器。内置**智能数据引擎**，具备预检、串行下载、自动重试和缓存自愈功能。
*   [x] **LLM/VLM 从零预训练**: 核心逻辑已验证。
*   [x] **有监督微调 (SFT)**: 成功实现PEFT (LoRA) 微调，并解决CPU内存挑战。
*   [x] **直接偏好优化 (DPO)**: 成功实现DPO，攻克了复杂的模型加载、数据格式转换及内存分配难题。
*   [x] **近端策略优化 (PPO)**: 成功实现概念性PPO训练，解决了Trl库API调用及数据批处理的最后一道关卡。
*   [x] **模型推理**: 提供交互式命令行界面，支持CPU上的模型加载与对话。
*   [x] **模型评估**: 包含主观评估示例和客观评估（基准测试）的详细指南。

## 项目结构 (最终版)

仓库被组织成一个清晰、模块化的结构，以下是**完整、准确**的文件和目录说明：

```
/Awesome-LLM-ZeroToScratch
|-- .gitignore                  # 定义了Git应忽略的文件和目录
|-- README.md                   # 您正在阅读的这个文件
|-- LICENSE                     # MIT 开源协议
|-- requirements.txt            # GPU环境（如AutoDL）的Python依赖清单
|-- requirements-cpu.txt        # 本地CPU开发环境的Python依赖清单
|-- setup.sh                    # GPU环境的一键安装脚本，专为稳定高效构建而设计
|-- create_project_structure.py # 用于自动生成项目骨架的脚本
|-- daydaydebug.md              # 【你的专属Debug日记】记录了本次史诗级调试的全过程与心路历程
|
|-- configs/                    # 所有YAML配置文件
|   |-- data/
|   |   |-- text_pretrain.yaml  # 文本预训练的数据处理配置
|   |   `-- vlm_pretrain.yaml   # VLM预训练的数据处理配置
|   |-- deepspeed/
|   |   |-- zero_stage2_config.json # DeepSpeed ZeRO Stage 2 配置文件
|   |   `-- zero_stage3_config.json # DeepSpeed ZeRO Stage 3 配置文件
|   |-- model/
|   |   |-- 0.5B_dense.yaml     # 0.5B参数密集型模型架构配置
|   |   |-- 0.8B_moe.yaml       # 0.8B参数专家混合(MoE)模型架构配置
|   |   `-- llama2-7b.yaml      # Llama-2 7B 模型的配置模板
|   `-- training/
|       |-- pretrain_llm.yaml   # 从零预训练的训练流程配置
|       |-- finetune_sft.yaml   # SFT微调的训练流程配置
|       |-- finetune_dpo.yaml   # DPO微调的训练流程配置
|       `-- finetune_ppo.yaml   # PPO微调的训练流程配置
|
|-- data_processing/            # 数据下载、处理、分词器训练脚本
|   |-- __init__.py             # 使该目录成为一个Python包
|   |-- download_and_reproduce.py # 数据处理主入口，自动化下载、清洗和保存
|   |-- process_text.py       # 文本数据清洗逻辑
|   |-- process_vlm.py        # VLM(图文)数据处理逻辑
|   `-- build_tokenizer.py    # SentencePiece分词器训练脚本
|
|-- docs/                       # 详细的Markdown教程文档
|   |-- 01_environment_setup.md # Part 1: 环境搭建指南
|   |-- 02_data_pipeline.md     # Part 2: 数据流水线指南
|   |-- 03_track_a_pretraining.md # Part 3A: 从零预训练路线
|   |-- 04_track_b_finetuning.md  # Part 3B: 微调路线
|   |-- 05_deployment_and_evaluation.md # Part 4: 部署与评估
|   `-- 06_troubleshooting_guide.md   # 【精华】所有问题的解决方案与经验总结
|
|-- scripts/                    # 用于一键启动各项任务的Shell脚本
|   |-- run_pretrain.sh         # 启动预训练
|   |-- run_sft.sh              # 启动SFT微调
|   |-- run_dpo.sh              # 启动DPO微调
|   |-- run_ppo.sh              # 启动PPO微调
|   |-- run_inference.sh        # 启动命令行推理
|   `-- run_evaluation.sh       # 启动模型评估
|
`-- src/                        # 核心模块化源代码
    |-- __init__.py             # 暴露顶层组件
    |-- models/                 # 模型定义
    |   |-- __init__.py         # 暴露模型核心组件
    |   |-- attention/          # 注意力机制实现
    |   |   |-- __init__.py
    |   |   |-- standard_attention.py # 标准多头注意力
    |   |   `-- flash_attention.py    # FlashAttention的封装
    |   |-- ffn.py              # 前馈网络(FFN)实现，支持SwiGLU
    |   |-- moe.py              # 专家混合(MoE)层实现
    |   `-- language_model.py   # 核心模型构建器，组装所有模块
    |-- trainers/               # 训练器
    |   |-- __init__.py         # 暴露训练器函数
    |   |-- pretrain_trainer.py # 从零预训练的训练循环
    |   |-- sft_trainer.py      # SFT微调训练器(使用TRL)
    |   |-- dpo_trainer.py      # DPO微调训练器(使用TRL)
    |   `-- ppo_trainer.py      # PPO微调训练器(使用TRL)
    |-- inference/              # 推理逻辑
    |   |-- __init__.py         # 使该目录成为Python包
    |   `-- inference.py        # 命令行交互式推理脚本
    |-- evaluation/             # 评估逻辑
    |   |-- __init__.py         # 使该目录成为Python包
    |   `-- evaluate_llm.py     # 模型评估脚本
    `-- utils/                  # 工具函数
        `-- __init__.py         # 使该目录成为Python包
```

## 快速开始

### 1\. 环境搭建：你的旅程起点！

我们提供了一键安装脚本，可以自动处理所有Python依赖，包括`PyTorch`, `DeepSpeed` 和 `flash-attn`等。无论你是使用CPU还是GPU，都有专属的最佳实践。

**详细的环境配置指南，请务必首先阅读：**
**➡️ [docs/01_environment_setup.md](./docs/01_environment_setup.md)**

简要步骤如下：

```bash
# 推荐创建一个独立的 Conda 环境
conda create -n awesome-llm-env python=3.10 -y
conda activate awesome-llm-env

# 进入项目根目录
cd /path/to/Awesome-LLM-ZeroToScratch

# 运行一键安装脚本 (根据你的环境，选择 docs/01_environment_setup.md 中对应路径的 `requirements-cpu.txt` 或 `requirements.txt`)
# 例如，对于CPU环境：
uv pip install -r requirements-cpu.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --index-strategy unsafe-best-match

# 或者对于GPU环境，运行 `bash setup.sh`
```

### 2\. 数据准备：模型的养料！

在开始训练之前，你需要准备好数据集和分词器。本项目内置的**智能数据引擎**将为你处理所有复杂的数据下载、校验和管理工作。

详细的数据流水线说明，请参考：
**➡️ [docs/02_data_pipeline.md](./docs/02_data_pipeline.md)**

运行以下命令来自动完成文本数据的处理和分词器训练：

```bash
# 执行文本数据流水线 (下载、清洗、保存处理后数据，具体参数在 configs/data/text_pretrain.yaml 中配置)
python data_processing/download_and_reproduce.py text

# 训练分词器 (参数来自 configs/data/text_pretrain.yaml)
python data_processing/build_tokenizer.py \
    --output_path_prefix ./data/tokenizers/wikitext_spm \
    --corpus_path ./data/processed/wikitext/corpus.txt \
    --vocab_size 8000 \
    --model_type unigram
```

## 使用指南：探索LLM训练的全景图

### Track A: 从零预训练LLM/VLM

此路线将引导你从零开始构建并训练一个全新的语言模型，深入理解其内部机制。

详细的预训练指南，请参考：
**➡️ [docs/03_track_a_pretraining.md](./docs/03_track_a_pretraining.md)**

1.  **配置:**
    *   打开 `configs/training/pretrain_llm.yaml`。
    *   通过修改 `model_config_path` 来选择模型架构，例如 `"configs/model/0.5B_dense.yaml"` 或 `"configs/model/0.8B_moe.yaml"`。
    *   确保 `dataset_dir` 指向你处理好的数据目录。
2.  **启动训练:**
    ```bash
    accelerate launch src/trainers/pretrain_trainer.py configs/training/pretrain_llm.yaml
    ```

### Track B: 微调开源模型

此路线将教你如何高效地微调现有强大的开源模型，使其适应特定任务或风格。

详细的微调指南，请参考：
**➡️ [docs/04_track_b_finetuning.md](./docs/04_track_b_finetuning.md)**

#### 1\. 有监督微调 (SFT)

1.  **配置:**
    *   打开 `configs/training/finetune_sft.yaml`。
    *   确认 `model_name_or_path` 指向你希望微调的基础模型（例如 `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`）。
    *   确认 `dataset_name` 指向SFT数据集。
2.  **启动训练:**
    ```bash
    accelerate launch src/trainers/sft_trainer.py configs/training/finetune_sft.yaml
    ```

#### 2\. 直接偏好优化 (DPO)

DPO在SFT模型的基础上进行，通过人类偏好数据进一步对齐模型行为。

1.  **配置:**
    *   打开 `configs/training/finetune_dpo.yaml`。
    *   **至关重要:** 确保 `model_name_or_path` 指向上一步SFT训练产出的**适配器(adapter)目录**（例如 `"./checkpoints/sft-tinyllama-guanaco-cpu/final_model"`）。
2.  **启动训练:**
    ```bash
    accelerate launch src/trainers/dpo_trainer.py configs/training/finetune_dpo.yaml
    ```

#### 3\. 近端策略优化 (PPO)

PPO作为RLHF家族的另一重要成员，通过强化学习进一步优化模型。

1.  **配置:**
    *   打开 `configs/training/finetune_ppo.yaml`。
    *   确保 `model_name_or_path` 指向上一步SFT训练产出的**适配器(adapter)目录**。
2.  **启动训练:**
    ```bash
    accelerate launch src/trainers/ppo_trainer.py configs/training/finetune_ppo.yaml
    ```

### 模型推理：与你的模型对话！

使用我们提供的命令行界面（CLI）与你训练好的模型进行交互。

详细的推理指南，请参考：
**➡️ [docs/05_deployment_and_evaluation.md](./docs/05_deployment_and_evaluation.md)**

```bash
# 运行推理，将 <path_to_your_model_or_adapter> 替换为你的模型路径
python -m src.inference.inference <path_to_your_model_or_adapter>
```

例如，要与SFT微调后的模型交互：

```bash
python -m src.inference.inference ./checkpoints/sft-tinyllama-guanaco-cpu/final_model
```

### 模型评估：我的模型有多好？

我们提供了一个脚本来演示如何进行主观评估（与模型对话），并指导你如何使用 `lm-evaluation-harness` 等工具进行客观的基准测试。

```bash
# 运行评估脚本
python -m src.evaluation.evaluate_llm <path_to_your_model_or_adapter>
```

## 许可证

本项目采用 MIT 许可证。详情请见 `LICENSE` 文件。

# END OF FILE: README.md