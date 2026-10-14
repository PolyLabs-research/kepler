#!/usr/bin/env bash
set -euo pipefail

# Create a lean archive for cloud training upload
# Usage: ./scripts/package_for_cloud.sh [output-tar]

OUT=${1:-kepler_cloud_bundle.tar.gz}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

tar \
  --exclude='.venv' \
  --exclude='.venv-linux' \
  --exclude='__pycache__' \
  --exclude='backup_20260129_150049' \
  --exclude='checkpoints' \
  --exclude='data' \
  --exclude='*.pyc' \
  -czf "$OUT" \
  .

echo "Created $OUT"
