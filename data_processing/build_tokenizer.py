# FILE: data_processing/build_tokenizer.py
"""
Bedrock Protocol: Module for training a SentencePiece tokenizer. (Upgraded for Robustness)

This script provides a single, focused function to train a tokenizer from a
corpus file. It now includes a dynamic fallback mechanism to automatically
adjust the vocabulary size if the initial request is too large for the corpus,
ensuring the pipeline doesn't crash on smaller datasets.
"""

import sentencepiece as spm
from pathlib import Path
import argparse
import re
import logging  # 导入 logging 模块
import shutil  # 用于文件复制
import json  # 用于生成 JSON 配置
from transformers import PreTrainedTokenizerFast, AutoTokenizer

# 修正: 配置日志, 确保脚本独立运行时也能正常输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # 默认输出到控制台
    ]
)
# 获取日志器
logger = logging.getLogger(__name__)


def train_tokenizer(
        output_path_prefix: str,
        corpus_path: str,
        vocab_size: int,
        model_type: str,
        character_coverage: float,
        add_special_tokens: bool = True
) -> None:
    """
    Trains a SentencePiece tokenizer and saves the model.
    Includes a fallback to automatically reduce vocab_size if it's too high.
    """
    # 将传入的字符串路径转换为 Path 对象，以便使用 .parent 等属性
    output_path_prefix = Path(output_path_prefix)
    corpus_path = Path(corpus_path)

    # Ensure the output directory exists
    output_dir = output_path_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"分词器输出目录已准备: {output_dir}")

    # --- Initial Training Attempt ---
    try:
        logger.info(f"--- [Bedrock] 启动 SentencePiece 训练 (尝试 1) ---")
        logger.info(f"请求词汇量大小: {vocab_size}")
        # SentencePiece Trainer 需要字符串路径
        _run_sentencepiece_training(str(output_path_prefix), str(corpus_path), vocab_size, model_type,
                                    character_coverage, add_special_tokens)

    except RuntimeError as e:
        logger.warning(f"首次训练失败。这在小数据集上很常见。错误详情: {e}")

        # --- Dynamic Vocab Size Adjustment Logic ---
        match = re.search(r'Please set it to a value <= (\d+)', str(e))
        if match:
            suggested_vocab_size = int(match.group(1))
            logger.info(f"--- [Bedrock] 正在尝试使用建议词汇量: {suggested_vocab_size} ---")

            try:
                _run_sentencepiece_training(str(output_path_prefix), str(corpus_path), suggested_vocab_size, model_type,
                                            character_coverage, add_special_tokens)
            except RuntimeError as e2:
                logger.error(f"致命错误: 使用建议词汇量重试训练仍失败。错误详情: {e2}")
                return
        else:
            logger.error("致命错误: 无法从错误信息中解析建议词汇量。已中止。")
            return

    logger.info("\n--- [Bedrock] SentencePiece 训练完成 ---")

    # --- 保存为 Hugging Face Transformers 格式 (最新的鲁棒逻辑) ---
    # 这个逻辑旨在直接生成 HF 格式所需的文件，而不是依赖 PreTrainedTokenizerFast 成功加载原始 .model 文件
    logger.info(f"--- [Bedrock] 正在生成 Hugging Face 分词器文件... ---")
    hf_tokenizer_output_dir = output_dir / f"{output_path_prefix.name}_hf"
    hf_tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Hugging Face 分词器目标目录: {hf_tokenizer_output_dir}")

    # 获取 SentencePiece 生成的 .model 和 .vocab 文件的完整路径
    spm_model_file = output_path_prefix.with_suffix('.model')
    spm_vocab_file = output_path_prefix.with_suffix('.vocab')

    try:
        # 1. 复制 SentencePiece 生成的 .model 和 .vocab 文件到 Hugging Face 输出目录
        shutil.copy(spm_model_file, hf_tokenizer_output_dir / spm_model_file.name)
        shutil.copy(spm_vocab_file, hf_tokenizer_output_dir / spm_vocab_file.name)
        logger.info(f"已复制 SentencePiece 模型文件 ({spm_model_file.name}) 和词汇表文件 ({spm_vocab_file.name})。")

        # 2. 创建 tokenizer_config.json
        # 这是告诉 AutoTokenizer 如何理解这个分词器的元数据文件
        tokenizer_config = {
            "model_max_length": 1024,  # 这是一个合理的默认值，可以根据你的模型配置调整
            "pad_token": "[PAD]",
            "eos_token": "[EOS]",
            "bos_token": "[BOS]",
            "unk_token": "[UNK]",
            "model_type": model_type,  # 使用训练时的模型类型 (e.g., "unigram")
            "vocab_size": vocab_size,  # 使用训练时的词汇量
            "tokenizer_class": "LlamaTokenizerFast" if model_type == "unigram" else f"{model_type.capitalize()}TokenizerFast",
            # 尝试根据类型设置 class
            "name_or_path": str(hf_tokenizer_output_dir),  # 指向自身目录，方便 AutoTokenizer 加载
        }
        with open(hf_tokenizer_output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        logger.info(f"已创建 tokenizer_config.json。")

        # 3. 尝试通过 AutoTokenizer 从复制过来的文件加载并重新保存为标准 HF 格式
        # 这一步旨在生成完整的 tokenizer.json，它内部会解析 .model 文件
        try:
            temp_tokenizer = AutoTokenizer.from_pretrained(
                str(hf_tokenizer_output_dir),
                trust_remote_code=True,
                use_fast=True,
            )
            logger.info("已通过 AutoTokenizer 成功加载临时分词器。")

            # 确保特殊 token 被正确设置
            if add_special_tokens:
                special_tokens_dict = {}
                # 仅当 token 不存在时才添加，避免警告
                if temp_tokenizer.bos_token is None: special_tokens_dict['bos_token'] = '[BOS]'
                if temp_tokenizer.eos_token is None: special_tokens_dict['eos_token'] = '[EOS]'
                if temp_tokenizer.unk_token is None: special_tokens_dict['unk_token'] = '[UNK]'
                if temp_tokenizer.pad_token is None: special_tokens_dict['pad_token'] = '[PAD]'

                if special_tokens_dict:
                    temp_tokenizer.add_special_tokens(special_tokens_dict)
                    logger.info(f"已为临时分词器添加特殊 token: {list(special_tokens_dict.keys())}")

            if temp_tokenizer.pad_token is None and temp_tokenizer.eos_token is not None:
                temp_tokenizer.pad_token = temp_tokenizer.eos_token
                logger.info(f"已设置临时分词器的 pad_token 为 eos_token。")

            temp_tokenizer.save_pretrained(str(hf_tokenizer_output_dir))
            logger.info(f"Hugging Face 格式分词器已成功保存到: {hf_tokenizer_output_dir}")

        except Exception as inner_e:
            logger.warning(
                f"警告：尝试通过 AutoTokenizer 加载并重新保存分词器失败，将尝试手动构建 tokenizer.json。错误详情: {inner_e}")
            # 如果 AutoTokenizer 加载失败，回退到手动构建 tokenizer.json，直接引用 SentencePiece 模型文件
            # 这种方法不涉及在构建时直接解析 .model 文件到 Tokenizer 对象
            tokenizer_json_content = {
                "version": "1.0",
                "normalizer": {
                    "type": "Sequence",
                    "normalizers": [
                        {"type": "Replace", "pattern": " ", "content": " "},
                        {"type": "Strip", "left": False, "right": True}
                    ]
                },
                "pre_tokenizer": {
                    "type": "Sequence",
                    "pre_tokenizers": [
                        {"type": "Split", "pattern": {"String": " ", "SplitMode": "Contiguous"},
                         "add_prefix_space": False},
                        {"type": "Punctuation"}
                    ]
                },
                "model": {
                    "type": "WordPiece" if model_type == "bpe" else "SentencePiece",
                    "files": str(spm_model_file.name),  # 直接引用复制的 .model 文件名
                    "vocab": str(spm_vocab_file.name),  # 直接引用复制的 .vocab 文件名
                },
                "decoder": {
                    "type": "Sequence",
                    "decoders": [
                        {"type": "ByteFallback"},
                        {"type": "Replace", "pattern": " ", "content": " "}
                    ]
                },
                "post_processor": None,
                "token_to_id": {},
                "id_to_token": [],
                "added_tokens": [
                    {"id": 0, "content": "[PAD]", "single_word": False, "lstrip": False, "rstrip": False,
                     "normalized": True},
                    {"id": 1, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False,
                     "normalized": True},
                    {"id": 2, "content": "[BOS]", "single_word": False, "lstrip": False, "rstrip": False,
                     "normalized": True},
                    {"id": 3, "content": "[EOS]", "single_word": False, "lstrip": False, "rstrip": False,
                     "normalized": True},
                    {"id": 4, "content": "<0x0A>", "single_word": False, "lstrip": False, "rstrip": False,
                     "normalized": True}  # 换行符
                ],
                "padding": {
                    "direction": "right",
                    "pad_id": 0,
                    "pad_type_id": 0,
                    "pad_token": "[PAD]",
                    "length": None  # 可以在模型加载时根据 model_max_length 设置
                },
                "truncation": {
                    "max_length": tokenizer_config["model_max_length"],
                    "strategy": "longest_first",
                    "direction": "right"
                }
            }

            # 从 .vocab 文件读取实际的 token_to_id 和 id_to_token
            # 注意：如果 SentencePiece 的 .model 文件本身包含所有信息，这一步可能不是必需的
            # 但为了鲁棒性，我们依然解析 .vocab 文件
            # 确保特殊 token 的 ID 与 added_tokens 保持一致
            id_to_piece = {t["id"]: t["content"] for t in tokenizer_json_content["added_tokens"]}
            piece_to_id = {t["content"]: t["id"] for t in tokenizer_json_content["added_tokens"]}

            with open(spm_vocab_file, "r", encoding="utf-8") as f:
                for line in f:
                    piece, score = line.strip().split('\t')
                    if piece not in piece_to_id:  # 避免覆盖特殊 token
                        id_to_piece[len(id_to_piece)] = piece  # 暂时用连续 ID
                        piece_to_id[piece] = len(piece_to_id)  # 暂时用连续 ID

            # 将 ID 和 Piece 映射回 tokenizer_json_content
            # 先清空，再重新构建
            tokenizer_json_content["token_to_id"] = piece_to_id
            tokenizer_json_content["id_to_token"] = [id_to_piece[i] for i in sorted(id_to_piece.keys())]

            with open(hf_tokenizer_output_dir / "tokenizer.json", "w", encoding="utf-8") as f:
                json.dump(tokenizer_json_content, f, ensure_ascii=False, indent=2)
            logger.info(f"已通过手动构建 tokenizer.json 成功保存分词器到: {hf_tokenizer_output_dir}")

    except Exception as e:
        logger.error(f"致命错误: 尝试生成 Hugging Face 分词器文件失败。错误详情: {e}")
        logger.error(f"请检查文件 {spm_model_file} 和 {spm_vocab_file} 是否有效，并确保目标目录可写。")
        return


def _run_sentencepiece_training(
        output_path_prefix: str,
        corpus_path: str,
        vocab_size: int,
        model_type: str,
        character_coverage: float,
        add_special_tokens: bool
):
    """Helper function to construct and run the SentencePiece training command."""
    cmd_parts = [
        f'--input={corpus_path}',
        f'--model_prefix={output_path_prefix}',
        f'--vocab_size={vocab_size}',
        f'--model_type={model_type}',
        f'--character_coverage={character_coverage}',
    ]

    if add_special_tokens:
        cmd_parts.extend([
            f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3',
            f'--pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS]',
            f'--user_defined_symbols=<0x0A>'
        ])

    command = " ".join(cmd_parts)
    spm.SentencePieceTrainer.Train(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer with dynamic vocab size fallback.")
    parser.add_argument(
        "--output_path_prefix", type=str, required=True,
        help="The base path and prefix for the output files (e.g., './data/tokenizers/my_spm')."
    )
    parser.add_argument(
        "--corpus_path", type=str, required=True,
        help="Path to the text file corpus to train on."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=8000,
        help="The desired total number of tokens in the vocabulary."
    )
    parser.add_argument(
        "--model_type", type=str, default="unigram", choices=["unigram", "bpe", "char", "word"],
        help="The tokenizer model type ('unigram' is recommended)."
    )
    parser.add_argument(
        "--character_coverage", type=float, default=1.0,
        help="The percentage of characters in the corpus to be covered."
    )
    parser.add_argument(
        "--add_special_tokens", type=bool, default=True,
        help="Whether to add default special tokens."
    )
    args = parser.parse_args()

    train_tokenizer(
        output_path_prefix=args.output_path_prefix,
        corpus_path=args.corpus_path,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        add_special_tokens=args.add_special_tokens
    )