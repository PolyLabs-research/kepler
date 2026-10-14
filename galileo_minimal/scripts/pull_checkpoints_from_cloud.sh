#!/usr/bin/env bash
set -euo pipefail

# Pull checkpoints from cloud to local Kepler project
# Usage: ./scripts/pull_checkpoints_from_cloud.sh user@host:/remote/kepler/checkpoints

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 user@host:/remote/kepler/checkpoints"
  exit 1
fi

SRC="$1"
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST="$ROOT_DIR/checkpoints/"
mkdir -p "$DEST"

rsync -avz "$SRC" "$DEST"

echo "Synced checkpoints to $DEST"
