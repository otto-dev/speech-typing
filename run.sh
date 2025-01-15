#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

# Detect the current Python major.minor version
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

sudo -sE \
  CUDA_VISIBLE_DEVICES=1 \
  DISPLAY=:0 \
  XAUTHORITY=/home/otto/.Xauthority \
  LD_LIBRARY_PATH="$SCRIPT_DIR/venv/lib64/python${PYTHON_VERSION}/site-packages/nvidia/cublas/lib:$SCRIPT_DIR/venv/lib64/python${PYTHON_VERSION}/site-packages/nvidia/cudnn/lib" \
  python "$SCRIPT_DIR/main.py"
