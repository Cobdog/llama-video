#!/usr/bin/env bash
# Start patched llama-server with a model and mmproj
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
SERVER_BIN="$LLAMA_DIR/build/bin/llama-server"

if [ ! -f "$SERVER_BIN" ]; then
    echo "Error: llama-server not found at $SERVER_BIN"
    echo "Run ./scripts/build.sh first."
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model.gguf> <mmproj.gguf> [--port PORT] [extra args...]"
    echo ""
    echo "Example:"
    echo "  $0 Qwen3.5-35B-A3B-Q4_K_M.gguf mmproj-F16.gguf"
    echo "  $0 Qwen3.5-35B-A3B-Q4_K_M.gguf mmproj-F16.gguf --port 8080 --ctx-size 8192"
    exit 1
fi

MODEL="$1"
MMPROJ="$2"
shift 2

# Default arguments
PORT="${LLAMA_SERVER_PORT:-8080}"
HOST="${LLAMA_SERVER_HOST:-0.0.0.0}"
CTX_SIZE="${LLAMA_SERVER_CTX_SIZE:-65536}"

# Guard: only one llama-server at a time (each loads 35GB+ into VRAM)
EXISTING_PID=$(pgrep -f "llama-server" 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Error: llama-server is already running (PID: $EXISTING_PID)"
    echo "Stop it first:  kill $EXISTING_PID"
    echo "Or force-kill:  kill -9 $EXISTING_PID"
    exit 1
fi

echo "Starting llama-server..."
echo "  Model:  $MODEL"
echo "  mmproj: $MMPROJ"
echo "  Host:   $HOST:$PORT"
echo "  Ctx:    $CTX_SIZE"

exec "$SERVER_BIN" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --jinja \
    "$@"
