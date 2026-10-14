#!/usr/bin/env bash
set -euo pipefail

# Setup script for Linux/macOS
# Usage:
#   ./setup.sh           # auto (CUDA if available, else CPU)
#   ./setup.sh --cpu     # force CPU PyTorch
#   ./setup.sh --cuda    # force CUDA PyTorch
#   ./setup.sh --dev     # install pytest/dev deps

MODE="auto"
DEV=0

for arg in "$@"; do
  case "$arg" in
    --cpu) MODE="cpu" ;;
    --cuda) MODE="cuda" ;;
    --dev) DEV=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ and retry."
  exit 1
fi

VENV_DIR=".venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

if [[ "$DEV" -eq 1 ]]; then
  pip install -r requirements-dev.txt
else
  pip install -r requirements.txt
fi

if [[ "$MODE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    MODE="cuda"
  else
    MODE="cpu"
  fi
fi

if [[ "$MODE" == "cuda" ]]; then
  echo "Installing PyTorch CUDA (cu121)..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
  echo "Installing PyTorch CPU..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PY

echo
echo "Setup complete."
echo "Activate: source .venv/bin/activate"
echo "Train:    python train.py --model polynomial"
echo "Test:     pytest"
