#!/bin/bash

# ==============================================================================
# Bedrock Protocol: One-Click Environment Setup Script
# ==============================================================================
# This script provides a deterministic, robust, and fast setup for the project
# environment. It embodies the "保姆级" (nanny-level) philosophy by handling
# potential issues like network instability and missing tools.
#
# The script will exit immediately if any command fails.
set -e

echo "--- [Bedrock] Starting Environment Setup ---"

# --- Helper Function to Ensure Command Exists ---
# Before we use any tool, we make sure it's installed.
ensure_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "--- Command '$1' not found. Attempting to install..."
        # 修正：根据不同的命令，安装包名可能不同
        if [ "$1" == "pip" ]; then
            apt-get update -y && apt-get install -y python3-pip
        elif [ "$1" == "uv" ]; then
            # uv 会在后面通过 pip 安装，这里不直接 apt-get
            echo "--- 'uv' will be installed via pip in the next step."
        elif [ "$1" == "aria2c" ]; then
            apt-get update -y && apt-get install -y aria2
        else
            apt-get update -y && apt-get install -y "$2" # 尝试使用提供的第二个参数作为包名
        fi
        echo "--- '$1' installed successfully."
    else
        echo "--- Command '$1' is already installed."
    fi
}


# --- Step 1: Verify and Install Essential Tools ---
echo "[1/5] Verifying essential tools (pip, aria2c)..."
ensure_command pip python3-pip # 确保 pip 已安装
ensure_command aria2c aria2 # 确保 aria2c 已安装

# --- Step 2: Install/Upgrade uv using pip ---
# We use pip for the initial install of uv, our high-speed package manager.
# A mirror is configured first to speed up this step itself.
echo "[2/5] Installing/Upgrading uv..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U uv
echo "--- uv installed/updated successfully."


# --- Step 3: Install All Dependencies EXCEPT Flash Attention ---
# We install everything else first. flash-attn is handled separately
# due to its large size and potential for network-related download failures.
echo "[3/5] Installing all packages from requirements.txt (except flash-attn)..."
# Define mirror arguments for uv. This is crucial for speed.
# NOTE: Ensure 'cu121' matches your target environment's CUDA version.
UV_MIRROR_ARGS="--extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121"

# 我们使用 uv 从 requirements.txt 安装，比 pip 快得多。
# 我们明确排除 flash-attn，在下一步单独处理。
# 修正：增加 --index-strategy unsafe-best-match 参数
uv pip install -r requirements.txt $UV_MIRROR_ARGS --index-strategy unsafe-best-match
echo "--- Core dependencies installed successfully."


# --- Step 4: Robustly Download Flash Attention ---
# This is the most critical step. We download the large pre-compiled wheel
# file using a robust downloader BEFORE trying to install it. This is the
# key to overcoming network timeouts.
echo "[4/5] Robustly downloading Flash Attention wheel..."
# This URL is for torch 2.3.0 and CUDA 12.1/12.2/12.4. CHANGE IF YOUR ENV IS DIFFERENT.
# 修正：确保 URL 匹配 PyTorch 2.3.0 + CUDA 12.1
FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu121torch2.3cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTN_WHEEL_FILE=$(basename "$FLASH_ATTN_WHEEL_URL")

# Use aria2c for fast, multi-connection download.
# The -c flag allows resuming if the download is interrupted.
aria2c -c -x 16 -s 16 -o "$FLASH_ATTN_WHEEL_FILE" "$FLASH_ATTN_WHEEL_URL"
echo "--- Flash Attention wheel downloaded successfully."


# --- Step 5: Install Flash Attention From Local File ---
# Now that the file is local, installation is instant and immune to network issues.
echo "[5/5] Installing Flash Attention from local wheel file..."
uv pip install "$FLASH_ATTN_WHEEL_FILE"
echo "--- Flash Attention installed successfully."


# --- Finalization ---
echo ""
echo "================================================="
echo "✅ [Bedrock] Environment Setup Complete!"
echo "All packages, including the tricky ones, are now installed."
echo "You are ready to proceed with your project."
echo "================================================="

# END OF FILE