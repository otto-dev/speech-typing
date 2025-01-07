#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
sudo -sE LD_LIBRARY_PATH="$SCRIPT_DIR/venv/lib64/python3.10/site-packages/nvidia/cublas/lib:$SCRIPT_DIR/venv/lib64/python3.10/site-packages/nvidia/cudnn/lib" python "$SCRIPT_DIR/main.py"

