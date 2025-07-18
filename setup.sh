# FILE: setup.sh
#!/bin/bash

# Bedrock Protocol: This script provides a deterministic, one-click setup
# for the project environment. It embodies the "保姆级" philosophy.
# It will exit immediately if any command fails.
set -e

echo "--- [Bedrock] Starting Environment Setup ---"

# --- Step 1: Install uv ---
# uv is a fast Python package installer and resolver, which is significantly
# faster than pip for dependency resolution and installation.
echo "[1/4] Installing uv..."
# We use pip to install uv itself into the active conda/venv environment.
# Using --break-system-packages for root user is common in containers, though uv usually handles it well.
pip install uv
echo "uv installed successfully."

# --- Step 2: Install PyTorch and bitsandbytes FIRST ---
# flash-attn (and potentially deepspeed) often require torch during their build process.
# By installing torch and bitsandbytes explicitly here, we ensure they are available
# in the environment before uv attempts to build other compiled packages.
echo "[2/4] Installing PyTorch and bitsandbytes with uv (essential build dependencies)..."
# We explicitly specify versions as in requirements.txt to maintain consistency.
uv pip install torch==2.3.0 bitsandbytes==0.43.1
echo "PyTorch and bitsandbytes installed."

# --- Step 3: Install Remaining Core Dependencies with uv ---
# Install all packages from requirements.txt, excluding torch and bitsandbytes
# which were just installed. This prevents uv from trying to reinstall them
# or encountering conflicts.
echo "[3/4] Installing remaining core Python packages from requirements.txt using uv..."
# Create a temporary filtered requirements file
grep -v -E '^(torch|bitsandbytes|flash-attn)' requirements.txt > /tmp/requirements_filtered.txt
uv pip install -r /tmp/requirements_filtered.txt
rm /tmp/requirements_filtered.txt # Clean up temporary file
echo "Remaining core dependencies installed with uv."

# --- Step 4: Install FlashAttention LAST with --no-build-isolation ---
# This step specifically addresses the ModuleNotFoundError for 'torch' during flash-attn build.
# --no-build-isolation forces uv to use the current environment's packages for building.
echo "[4/4] Installing FlashAttention with uv (disabling build isolation)..."
# We explicitly specify version.
uv pip install flash-attn==2.5.8 --no-build-isolation
echo "FlashAttention installation attempted with uv. Please check for errors above."

echo ""
echo "--- [Bedrock] Environment Setup Complete ---"
echo "The environment has been configured. You are ready to proceed."
echo "Next step: Follow the data preparation guide in docs/02_data_pipeline.md"

# END OF FILE: setup.sh