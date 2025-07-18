# Part 2: 数据流水线 (Data Pipeline)

## 目标 (Goal)

模型性能的上限由数据质量决定。本章节将提供一套可复现、高质量的数据处理流程，它是后续所有模型训练工作的基石。我们将处理用于预训练的纯文本和图文对数据，并训练一个自定义分词器。

---

### 1. 数据集准备与复现 (Dataset Preparation & Reproduction)

为了实现“保姆级”的快速验证，我们不直接处理TB级的庞大数据集，而是使用两个小规模、有代表性的标准数据集。我们提供了一个核心脚本来自动化完成整个流程。

*   **纯文本 (LLM):** 我们将使用 `wikitext-2-raw-v1`，这是一个干净、小巧（约10MB）的英文维基百科子集。
*   **图文对 (VLM):** 我们将使用 `COCO` 数据集的一个微小子集（100个样本）作为演示。

**执行命令:**

打开终端，确保您位于 `Awesome-LLM-ZeroToScratch` 项目的根目录。

1.  **执行文本数据流水线:**
    此命令将下载、清洗、处理 `wikitext` 数据，并准备用于训练分词器的语料。
    ```bash
    python data_processing/download_and_reproduce.py text
    ```
    此命令将：
    *   下载 `wikitext-2-raw-v1` 数据集到 `./data/raw/wikitext/`。
    *   对其进行清洗（去除元数据、短行等）。
    *   将处理后的数据保存到 `./data/processed/wikitext/`。
    *   合并所有文本，为训练分词器做准备，语料文件在 `./data/processed/wikitext/corpus.txt`。

2.  **执行VLM数据流水线:**
    此命令将下载并处理 `COCO` 数据集的一个小型子集。
    ```bash
    python data_processing/download_and_reproduce.py vlm
    ```
    此命令将：
    *   下载 `COCO` 数据集的前100个样本到 `./data/raw/coco_demo/`。
    *   对图片和文本进行初步处理（例如图片转换为RGB、文本清洗）。
    *   将处理后的数据保存到 `./data/processed/coco_demo/`。

---

### 2. 数据清洗与增强 (Data Cleaning & Augmentation)

高质量的数据是模型成功的关键。我们的脚本内置了基础但重要的数据清洗步骤。

*   **文本清洗 (`data_processing/process_text.py`):**
    *   **规范化:** 将多余的空格、换行符统一成单个空格。
    *   **规则过滤:** 移除了Wikitext中常见的元数据行（如 `= Headings =`）。
    *   **质量过滤:** 移除了过短（少于10个词）或不含字母的行。

*   **VLM数据处理 (`data_processing/process_vlm.py`):**
    *   确保所有图片都转换为`RGB`格式。
    *   对图片进行了统一的 `Resize((224, 224))` 和 `ToTensor()` 转换。
    *   对文本描述进行了基础的清洗。

*   **数据蒸馏 (Data Distillation) - 概念:**
    对于VLM，高质量的图文对是稀缺的。一种先进的技术是**数据蒸馏**：使用一个强大的VLM（如GPT-4V）为图片生成高质量的描述。我们在 `data_processing/process_vlm.py` 中提供了一个**概念性实现** `conceptual_gpt4v_distillation`。
    *   **它默认不执行**，不会产生任何API费用。
    *   代码结构清晰地展示了如何调用API，您可以填入自己的API密钥来开启此功能。这体现了从“教程”到“工业级”的扩展性。

---

### 3. 分词器训练 (Tokenizer Training)

分词器是连接自然语言和模型输入的桥梁。我们将训练一个针对我们文本数据的 SentencePiece Unigram 分词器，并将其保存为 Hugging Face `transformers` 库兼容的格式。

**执行命令:**

在文本数据流水线成功运行后，语料库已经准备就绪。现在，我们训练分词器。

```bash
python data_processing/build_tokenizer.py \
    --output_path_prefix ./data/tokenizers/wikitext_spm \
    --corpus_path ./data/processed/wikitext/corpus.txt \
    --vocab_size 8000 \
    --model_type unigram \
    --character_coverage 1.0
```

*   **`--output_path_prefix`**: 定义了输出模型和词汇表的前缀。最终的 SentencePiece 模型文件是 `./data/tokenizers/wikitext_spm.model` 和词汇表文件是 `./data/tokenizers/wikitext_spm.vocab`。
*   **Hugging Face 格式**: `build_tokenizer.py` 脚本还会自动将训练好的 tokenizer 转换为 Hugging Face `transformers` 库兼容的格式，并保存到 `./data/tokenizers/wikitext_spm_hf/` 目录中。后续的模型训练和推理将直接从这个 `_hf` 目录加载 tokenizer。

---

### 4. 流式处理 vs. 内存加载 (Streaming vs. In-Memory)

*   **对于本教程的小数据集:** 我们将所有数据加载到内存中，然后进行处理和保存。这简单直接。
*   **对于工业级的大数据集 (TB级):** 绝不可能一次性将数据加载到内存中。此时必须使用 `datasets` 库的 **流式处理（Streaming）** 功能。
    ```python
    # 概念代码
    # ds = load_dataset('some_huge_dataset', split='train', streaming=True)
    # for example in ds:
    #     process(example)
    ```
    流式处理会逐条读取和处理数据，内存占用极低。在后续的训练章节，我们会进一步讨论如何在训练循环中集成流式数据集。

---

### 5. 资源与成本估算 (Resource & Cost Estimation)

*   **GPU**: 本章节所有操作均为CPU密集型，**无需GPU**。
*   **时间成本**: 在AutoDL的CPU实例上，完整运行上述所有流程，预计在 **5-15分钟** 内完成。
*   **金钱成本**:
    *   计算：可忽略不计，约几毛钱的CPU实例租用费。
    *   API调用：数据蒸馏部分默认关闭。如开启，成本取决于您的使用量。

---
**您已完成第二部分。现在，您拥有了干净、处理完毕的数据集和一个定制化的分词器，为模型训练奠定了坚实的基础。接下来，您可以选择进入以下任一赛道：**

*   **赛道A: [docs/03_track_a_pretraining.md](./docs/03_track_a_pretraining.md)** (从零预训练)
*   **赛道B: [docs/04_track_b_finetuning.md](./docs/04_track_b_finetuning.md)** (微调开源模型)

# END OF FILE: docs/02_data_pipeline.md