
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
Markdown
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

# END OF FILE: README.md
Use code with caution.
