# FILE: create_project_structure.py
"""
Bedrock Protocol v: Absolute Provenance Edition
Project: Awesome-LLM-ZeroToScratch
Task: Automated generation of the complete project directory and file skeleton.

This script is a self-contained utility to bootstrap the project structure.
It creates all necessary directories and placeholder files as defined in the
Bedrock Workflow, ensuring a clean, consistent, and reproducible starting point.

This script embodies the Mandate of Intentionality: its sole, deliberate
purpose is to construct the project's foundational scaffold.
"""

import os
import sys
from pathlib import Path
from typing import List

# --- CORE CONFIGURATION ---

# Mandate of Intentionality: The root directory name is explicitly defined.
ROOT_DIR_NAME: str = "Awesome-LLM-ZeroToScratch"

# Mandate of Structural Integrity: The entire project structure is defined
# declaratively. Each entry is a deliberate component of the grand design.
PROJECT_STRUCTURE: List[str] = [
    ".gitignore",
    "README.md",
    "LICENSE",  # Added LICENSE here to ensure it's created.
    "requirements.txt",
    "setup.sh",
    "configs/data/text_pretrain.yaml",
    "configs/data/vlm_pretrain.yaml",
    "configs/model/0.5B_dense.yaml",
    "configs/model/0.8B_moe.yaml",
    "configs/model/llama2-7b.yaml",
    "configs/training/pretrain_llm.yaml",
    "configs/training/finetune_sft.yaml",
    "configs/training/finetune_dpo.yaml",
    "data_processing/download_and_reproduce.py",
    "data_processing/process_text.py",
    "data_processing/process_vlm.py",
    "data_processing/build_tokenizer.py",
    "data_processing/__init__.py",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/attention/__init__.py",
    "src/models/attention/standard_attention.py",
    "src/models/attention/flash_attention.py",
    "src/models/ffn.py",
    "src/models/moe.py",
    "src/models/language_model.py",
    "src/trainers/__init__.py",
    "src/trainers/pretrain_trainer.py",
    "src/trainers/sft_trainer.py",
    "src/trainers/dpo_trainer.py",
    "src/utils/__init__.py",
    "src/inference/__init__.py",  # Added for completeness in structure generation
    "src/inference/inference.py",  # Added for completeness in structure generation
    "src/evaluation/__init__.py",  # Added for completeness in structure generation
    "src/evaluation/evaluate_llm.py",  # Added for completeness in structure generation
    "scripts/run_pretrain.sh",
    "scripts/run_sft.sh",
    "scripts/run_dpo.sh",
    "scripts/run_inference.sh",  # Added for completeness in structure generation
    "scripts/run_evaluation.sh",  # Added for completeness in structure generation
    "docs/01_environment_setup.md",
    "docs/02_data_pipeline.md",
    "docs/03_track_a_pretraining.md",
    "docs/04_track_b_finetuning.md",
    "docs/05_deployment_and_evaluation.md",
]

# Mandate of Zero Ambiguity: Content for critical files is defined explicitly,
# leaving no room for misinterpretation.
GITIGNORE_CONTENT: str = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.idea/
.vscode/
*.swp
*.sublime-workspace
*.sublime-project

# AutoDL / Cloud specific
.autodl/
/autodl-tmp/
/autodl-fs/

# Jupyter Notebook
.ipynb_checkpoints

# Data files
# Raw, processed, or tokenized data should not be in git.
# Use download scripts and mount network storage instead.
*.bin
*.pt
*.arrow
*.parquet
/data/
/datasets/
/wudao_200g/
*.jsonl

# Trained models & logs
# Checkpoints, models, and logs are artifacts, not source code.
/models/
/checkpoints/
/runs/
/logs/

# W&B local artifacts
/wandb/

# Tokenizer files
*.model
*.vocab
"""

README_CONTENT: str = """
# Awesome-LLM-ZeroToScratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, code-driven tutorial for pre-training and fine-tuning
Large Language Models (LLMs) and Vision Language Models (VLMs).

This repository is built following the **Bedrock Protocol v: Absolute Provenance Edition**,
ensuring theoretical and practical robustness, clarity, and reproducibility.

## Core Philosophy

This project is designed with a dual-purpose philosophy:

*   **保姆级 (Nanny-Level):** Providing meticulously detailed, step-by-step guidance with ready-to-run code and commands, specifically tailored for platforms like AutoDL. It's designed for users who have a theoretical understanding but lack hands-on experience with large-scale model training.
*   **工业级 (Industrial-Grade):** The codebase is highly modular, configuration-driven (using YAML), and built for scalability. It clearly demonstrates the key architectural and strategic adjustments needed to scale from a small 0.5B parameter model to a 70B+ giant (e.g., parallelism strategies, memory optimization).

## Project Structure

The repository is organized into a clear, modular structure:
Use code with caution.
Python
/Awesome-LLM-ZeroToScratch
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- setup.sh
|-- configs/ # YAML configs for data, models, and training
|-- data_processing/ # Scripts for data download, processing, tokenization
|-- docs/ # Detailed markdown tutorials
|-- scripts/ # Shell scripts to launch training jobs
`-- src/ # Core, modular source code for models and trainers
Generated code
## Getting Started

To begin your journey, please start with the first document which details the complete environment setup process on the AutoDL platform.

**➡️ Start here: [docs/01_environment_setup.md](./docs/01_environment_setup.md)**

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
"""

LICENSE_CONTENT: str = """
MIT License

Copyright (c) 2025 Bedrock

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def main() -> None:
    """
    Main execution function to generate the project skeleton.
    """
    print("--- Bedrock Protocol: Initiating Project Scaffolding ---")

    # Mandate of Proactive Defense: Check for existing directory to prevent
    # accidental data loss.
    root_path = Path(ROOT_DIR_NAME)
    if root_path.exists():
        print(f"Error: Directory '{ROOT_DIR_NAME}' already exists.")
        print("Scaffolding aborted to prevent overwriting existing work.")
        sys.exit(1)

    print(f"Creating root directory: {root_path}")
    root_path.mkdir()

    # Create all files and directories
    for file_path_str in PROJECT_STRUCTURE:
        # Mandate of Type Purity: Use pathlib for robust, OS-agnostic path handling.
        file_path = root_path / file_path_str

        # Ensure parent directories exist
        print(f"  - Ensuring path: {file_path.parent}")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty file
        print(f"    - Creating file: {file_path}")
        file_path.touch()

    # --- POPULATE SPECIAL FILES ---
    # Mandate of Zero Ambiguity: Write predefined content to essential files.

    print("\nPopulating special files with initial content...")

    # Write .gitignore
    gitignore_path = root_path / ".gitignore"
    gitignore_path.write_text(GITIGNORE_CONTENT.strip())
    print(f"  - Wrote content to {gitignore_path}")

    # Write README.md
    readme_path = root_path / "README.md"
    readme_path.write_text(README_CONTENT.strip())
    print(f"  - Wrote content to {readme_path}")

    # Write LICENSE
    license_path = root_path / "LICENSE"
    license_path.write_text(LICENSE_CONTENT.strip())
    print(f"  - Wrote content to {license_path}")

    # Fill __init__.py files to make packages importable
    for file_path_str in PROJECT_STRUCTURE:
        if file_path_str.endswith("__init__.py"):
            init_path = root_path / file_path_str
            # Mandate of Intentionality: Add a clear comment to __init__.py
            init_path.write_text("# Bedrock: This file makes the directory a Python package.\n")
            print(f"  - Populated package marker in {init_path}")

    print("\n--- Project Scaffolding Complete ---")
    print(f"The project structure for '{ROOT_DIR_NAME}' has been successfully created.")
    print("Each file is a testament to our commitment to absolute provenance.")


if __name__ == "__main__":
    # The script is invoked, beginning the process of creation.
    main()

# END OF FILE: create_project_structure.py