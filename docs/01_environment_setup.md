# Part 1: 基础环境与项目框架搭建 (Environment and Framework Setup)

## 目标 (Goal)

本章节是整个教程的基石。我们将引导您完成从零开始在 [AutoDL](https://www.autodl.com) 平台上配置一个稳定、可复现的深度学习环境，并为后续的模型训练做好准备。

**本教程面向的用户画像:**
*   具备 Python 编程、机器学习和 Transformer 理论基础。
*   缺少大规模模型训练的实践经验。
*   希望获得一个可以立即上手的、代码驱动的实战项目。

---

### 1. AutoDL 平台选择 (Choosing an AutoDL Instance)

AutoDL 是一个性价比很高的GPU租用平台，非常适合个人开发者和研究者。

1.  **注册并登录** AutoDL 平台。
2.  进入“容器实例”页面，点击“租用新实例”。
3.  **选择GPU型号**:
    *   **微调 (Fine-tuning)**: 对于LoRA微调7B级别的模型，推荐选择 **RTX 3090 (24G)** 或 **RTX 4090 (24G)**。性价比高，足以完成大部分微调任务。
    *   **预训练 (Pre-training)**: 对于从零开始预训练1B级别的模型，需要更大的显存和算力。推荐选择 **A100 (80G)** 或 **A800 (80G)**，并建议租用多卡实例（例如4卡）。

### 2. 选择镜像 (Selecting the Docker Image)

镜像是预装了操作系统和基础软件的环境。为保证环境的纯净和可控性，我们选择官方的PyTorch镜像。

*   在镜像选择区域，选择 **`PyTorch` -> `2.3.0` -> `cuda12.1`**。
*   完整的镜像名称类似于 `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`。
*   **为何选择此镜像?** 它是一个纯净的、带有开发工具（`devel`）的官方镜像，版本较新，兼容我们所需的所有库，避免了其他社区镜像可能带来的依赖冲突。
*   选择好后，等待实例创建成功并开机。

### 3. 克隆项目并一键环境配置 (Clone Project & One-Click Setup)

实例开机后，点击“JupyterLab”进入工作区。

1.  **打开终端**: 在JupyterLab启动器中，点击 "Terminal" 打开一个终端窗口。
2.  **创建并激活 Conda 环境 (推荐)**:
    为了保持环境清洁和避免包冲突，强烈建议为本项目创建一个独立的 Conda 环境。
    ```bash
    # 创建名为 'awesome-llm-env' 的 Conda 环境，指定 Python 3.10
    conda create -n awesome-llm-env python=3.10 -y
    # 激活新环境
    conda activate awesome-llm-env
    # 如果遇到 "CondaError: Run 'conda init' before 'conda activate'"，请执行 `conda init`，然后关闭并重新打开终端，或执行 `source ~/.bashrc`
    ```
    *如果您没有安装 Conda，可以使用 `python3 -m venv .venv` 创建一个虚拟环境，然后 `source .venv/bin/activate` 激活它。*
3.  **进入项目目录**:
    请确保您的项目 `Awesome-LLM-ZeroToScratch` 文件夹已位于 `/root/autodl-tmp/` 目录下（如前一步骤所示）。然后进入该目录：
    ```bash
    cd /root/autodl-tmp/Awesome-LLM-ZeroToScratch
    ```
4.  **运行一键安装脚本**:
    确保您的 Conda (或虚拟) 环境已激活，并且您已在项目根目录下，然后运行我们提供的一键安装脚本 `setup.sh`。
    ```bash
    bash setup.sh
    ```
    此脚本将执行以下操作：
    *   配置 `pip` 使用清华大学 PyPI 镜像（用于安装 `uv` 自身）。
    *   **安装 `uv` (一个极速的 Python 包管理器)**。
    *   明确指示 `uv` 将使用清华大学 PyPI 镜像和 PyTorch 官方 CUDA 轮子源 (`https://download.pytorch.org/whl/cu121`)，这将大幅加速下载。
    *   **优先使用 `uv` 安装 `torch` 和 `bitsandbytes`**，以确保这些核心依赖在其他包编译前就位。在此步骤中，您会看到 `uv` 显示下载进度条。
    *   **然后使用 `uv` 安装 `requirements.txt` 中剩余的所有固定版本的依赖库**，同样利用配置的镜像源。
    *   **最后，使用 `uv` 安装 `flash-attn` 并禁用构建隔离**，确保它能正确编译和链接到已安装的 `torch`，同时利用镜像源加速下载。此步骤可能会涉及编译，请耐心等待。

    请耐心等待脚本执行完毕。在 `uv` 下载大文件时，您会看到清晰的进度条。看到 "Environment Setup Complete" 消息即表示成功。

### 4. 数据与代码挂载策略 (Data and Code Mounting Strategy)

在AutoDL上，理解不同磁盘的用途至关重要，这是实现数据持久化和实例无状态切换的关键。

*   **/root/autodl-tmp**: 这是**网盘**，它的数据在实例关机、重启甚至删除后 **依然存在**。它适合存放：
    *   **数据集 (Datasets)**
    *   **预训练权重 (Pre-trained Weights)**
    *   **项目代码 (Your Project Code)**
    *   **训练好的模型 checkpoints**
*   **/root/autodl-fs**: 这是**本地高速盘**，它的数据会随实例的删除而 **永久丢失**。它适合存放：
    *   **Conda/Python 环境**
    *   临时文件和缓存

**推荐工作流:**

1.  将本项目 `Awesome-LLM-ZeroToScratch` 克隆并存放在 `/root/autodl-tmp` 目录下。
2.  所有大型数据集都下载或传输到 `/root/autodl-tmp/data` 目录。
3.  所有训练产出的模型权重都保存到 `/root/autodl-tmp/checkpoints` 目录。
4.  这样，即使您更换了GPU实例，只需在新实例的 `/root/autodl-tmp` 中找到您的代码和数据，重新激活一下 Conda 环境，即可无缝继续工作。

### 5. 验证安装 (Verifying the Installation)

在终端中执行以下Python命令，验证核心库已正确安装：

```bash
python -c "import torch; import transformers; import deepspeed; import flash_attn; print('--- Bedrock Verification Protocol ---'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print('Core libraries imported successfully. System is ready.')"
```

如果您看到打印出的版本号和成功信息，并且`CUDA available`为`True`，那么恭喜您，基础环境已完美搭建。

---
**您已完成第一部分。接下来，请移步至 [docs/02_data_pipeline.md](./docs/02_data_pipeline.md) 开始数据准备工作。**
