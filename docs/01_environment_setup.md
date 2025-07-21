# FILE: docs/01_environment_setup.md

# Part 1: 环境配置：从本地CPU到云端GPU (Environment Setup)

## 目标 (Goal)

一个稳定、可复现的环境是成功的开端。本章节将提供两套并行的、针对不同场景的终极环境配置方案。请根据您的需求选择其一。

*   **方案 A (本地CPU):** 适合初学者、代码调试、或在没有GPU的电脑上快速验证数据处理和代码逻辑。**零成本，几分钟搞定。**
*   **方案 B (云端GPU):** 适合进行真正的模型训练（预训练/微调），充分利用GPU加速。以 [AutoDL](https://www.autodl.com) 平台为例。

---

### 方案A：本地CPU极速验证环境 (推荐新手)

本方案教您如何在自己的电脑 (Windows/macOS/Linux) 上，使用 `uv` 这一现代化的、极速的Python包管理器，搭建一个纯CPU的开发环境。

#### 第一步: 安装 `uv` (若尚未安装)
`uv` 是一个用Rust编写的、比`pip`和`venv`快得多的工具。此操作只需执行一次。

*   **Windows (PowerShell):**
    ```powershell
    irm https://astral.sh/uv/install.ps1 | iex
    ```
*   **macOS/Linux (Terminal):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

#### 第二步: 创建并激活虚拟环境
1.  **进入项目根目录**:
    打开终端，`cd` 进入 `Awesome-LLM-ZeroToScratch` 文件夹。

2.  **用 `uv` 创建虚拟环境**:
    ```bash
    uv venv
    ```
    这会在当前目录下创建一个名为 `.venv` 的文件夹。

3.  **激活环境**:
    *   Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
    *   Windows (CMD): `.\.venv\Scripts\activate.bat`
    *   macOS/Linux: `source .venv/bin/activate`
    激活成功后，命令行前会出现 `(.venv)` 标识。

#### 第三步: 使用 `uv` 一键安装CPU依赖
我们将使用一个专门为CPU环境优化的 `requirements-cpu.txt` 文件。

1.  **创建 `requirements-cpu.txt` 文件**:
    在项目根目录下创建此文件，并复制以下内容。这个文件巧妙地通过 `--extra-index-url` 指定了PyTorch的CPU版本下载地址，确保 `uv` 能找到正确的包。

    ```text
    # FILE: requirements-cpu.txt
    # 终极版：一个自给自足的、用于本地CPU测试的依赖文件。

    # --- 指定额外的包索引地址 ---
    # 告诉uv/pip，要去PyTorch的CPU官方地址寻找包
    --extra-index-url https://download.pytorch.org/whl/cpu

    # --- 核心深度学习框架 (CPU版本) ---
    torch==2.3.0
    torchvision
    torchaudio

    # --- Hugging Face 生态系统 (核心) ---
    transformers==4.41.2
    datasets==2.19.0
    accelerate==0.30.1
    tokenizers==0.19.1
    peft==0.11.1
    trl==0.9.4

    # --- 数据处理与工具库 ---
    sentencepiece==0.2.0
    pyyaml==6.0.1
    Pillow==10.4.0
    tqdm

    # --- 代码质量工具 (可选) ---
    ruff==0.4.4
    ```

2.  **执行安装命令**:
    ```bash
    uv pip install -r requirements-cpu.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --index-strategy unsafe-best-match
    ```
    *   `--index-url`: 使用清华镜像源，国内加速。
    *   `--index-strategy unsafe-best-match`: 允许 `uv` 在所有源（清华源、PyTorch CPU源）中自由寻找最佳匹配，解决依赖冲突。

**至此，您的本地CPU环境已完美就绪！** 您可以运行数据处理、模型推理等不依赖GPU的任务，甚至可以进行非常慢的CPU模型训练来验证代码逻辑。

---

### 方案B：云端GPU生产级训练环境 (AutoDL)

本方案专为 AutoDL 等云平台设计，通过 `setup.sh` 脚本解决网络、依赖和编译等一系列棘手问题。

#### 第一步: 租用并配置AutoDL实例
1.  **租用实例**:
    *   **微调**: 推荐 **RTX 3090/4090 (24G)**。
    *   **预训练**: 推荐 **A100/A800 (80G)**。
2.  **选择镜像**:
    *   选择 **`PyTorch` -> `2.3.0` -> `cuda12.1`**。这是一个纯净的官方镜像，能最大限度避免环境污染。
3.  **数据与代码挂载**:
    *   **代码、数据、模型 checkpoints**: 全部存放在 `/root/autodl-tmp` 目录下，这是一个网盘，数据不会丢失。
    *   **Conda 环境**: 默认安装在本地盘，随实例释放而消失，但可以快速重建。

#### 第二步: 一键部署
实例开机后，进入JupyterLab并打开一个终端。

1.  **创建并激活Conda环境**:
    ```bash
    # 为项目创建一个隔离的、干净的 Conda 环境
    conda create -n awesome-llm-env python=3.10 -y

    # 激活新环境
    conda activate awesome-llm-env
    ```

2.  **运行一键安装脚本**:
    ```bash
    # 进入项目代码所在目录
    cd /root/autodl-tmp/Awesome-LLM-ZeroToScratch

    # 赋予脚本执行权限
    chmod +x setup.sh

    # 执行脚本，开始全自动安装
    ./setup.sh
    ```
    `setup.sh` 脚本经过精心设计，它会：
    *   安装 `uv` 和 `aria2c` 等效率工具。
    *   使用 `uv` 和国内镜像高速安装 `requirements.txt` 中的绝大部分依赖。
    *   **单独、稳健地处理 `flash-attn`**: 先用 `aria2c` 多线程断点续传下载其 `.whl` 文件，再从本地安装。**这是克服国内云服务器网络不稳的“杀手锏”**。

#### 第三步: 验证安装
在激活了 `awesome-llm-env` 环境的终端中，运行：
```bash
python -c "import torch; import transformers; import deepspeed; import flash_attn; print('--- Bedrock Verification Protocol ---'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'bitsandbytes GPU support: {torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8}'); print('Core libraries imported successfully. System is ready.')"