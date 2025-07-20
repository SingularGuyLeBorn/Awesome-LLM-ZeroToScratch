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
import sys  # 修正: 导入 sys
from transformers import PreTrainedTokenizerFast, AutoTokenizer, LlamaTokenizerFast

# 修正: 配置日志, 确保脚本独立运行时也能正常输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 默认输出到控制台
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
            logger.info(f"--- [Bedrock] 正在尝试使用建议词汇量: {suggested_vocab_size}")

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
    logger.info(f"--- [Bedrock] 正在生成 Hugging Face 分词器文件... ---")
    hf_tokenizer_output_dir = output_dir / f"{output_path_prefix.name}_hf"
    hf_tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Hugging Face 分词器目标目录: {hf_tokenizer_output_dir}")

    spm_model_file = output_path_prefix.with_suffix('.model')

    try:
        # 修正：直接使用 LlamaTokenizerFast 加载 .model 文件，这是最稳健的方式
        # 它会正确生成所有必要的文件，包括 tokenizer.json
        logger.info(f"尝试使用 LlamaTokenizerFast 直接从 '{spm_model_file}' 加载...")
        tokenizer = LlamaTokenizerFast(vocab_file=str(spm_model_file))

        # 确保特殊 token 被正确设置
        if add_special_tokens:
            special_tokens_dict = {
                'bos_token': '[BOS]',
                'eos_token': '[EOS]',
                'unk_token': '[UNK]',
                'pad_token': '[PAD]'
            }
            tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(f"已为分词器添加特殊 token: {list(special_tokens_dict.keys())}")

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"已设置分词器的 pad_token 为 eos_token。")

        # 保存为 Hugging Face 格式
        tokenizer.save_pretrained(str(hf_tokenizer_output_dir))
        logger.info(f"Hugging Face 格式分词器已成功保存到: {hf_tokenizer_output_dir}")

    except Exception as e:
        logger.error(f"致命错误: 尝试生成 Hugging Face 分词器文件失败。错误详情: {e}")
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

# END OF FILE: data_processing/build_tokenizer.py