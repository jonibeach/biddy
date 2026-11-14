#!/usr/bin/env bash
# Start Physical AI Studio (backend + prebuilt UI) on http://localhost:7860
set -euo pipefail
STUDIO_DIR="${STUDIO_DIR:-$HOME/physical-ai-studio}"
cd "$STUDIO_DIR/application/backend"
export STATIC_FILES_DIR="$STUDIO_DIR/application/ui/dist/"
export PYTHONUNBUFFERED=1
exec "$HOME/.local/bin/uv" run --no-sync physicalai-studio serve "$@"
