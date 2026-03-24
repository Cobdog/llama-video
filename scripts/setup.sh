#!/usr/bin/env bash
# Full setup: clone llama.cpp, apply patches, build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"

echo "=== llama-video setup ==="

# Clone llama.cpp if not present
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
else
    echo "llama.cpp already cloned at $LLAMA_DIR"
fi

# Apply patches
"$SCRIPT_DIR/apply-patches.sh"

# Build
"$SCRIPT_DIR/build.sh"

# Install Python deps
echo "Installing Python dependencies..."
cd "$PROJECT_DIR"
uv sync --dev

echo ""
echo "=== Setup complete ==="
echo "Start llama-server:  ./scripts/run-server.sh <model.gguf> <mmproj.gguf>"
echo "Start Python service: uv run llama-video-server"
echo "Run tests:           make check"
