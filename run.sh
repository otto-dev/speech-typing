#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
#  CUDA_VISIBLE_DEVICES=1
sudo -sE DISPLAY=:0 XAUTHORITY=/home/otto/.Xauthority LD_LIBRARY_PATH="$SCRIPT_DIR/venv/lib64/python3.10/site-packages/nvidia/cublas/lib:$SCRIPT_DIR/venv/lib64/python3.10/site-packages/nvidia/cudnn/lib" python "$SCRIPT_DIR/main.py"

