#!/usr/bin/env bash
# Build patched llama.cpp — interactive configuration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="${1:-$PROJECT_DIR/llama.cpp}"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "Error: llama.cpp not found at $LLAMA_DIR"
    echo "Usage: $0 [/path/to/llama.cpp]"
    exit 1
fi

# ── GPU backend ──────────────────────────────────────────────────────

echo ""
echo "=== llama.cpp build configuration ==="
echo ""
echo "Select GPU backend:"
echo "  1) CUDA   (NVIDIA — requires CUDA toolkit)"
echo "  2) HIP    (AMD — requires ROCm)"
echo "  3) Vulkan (cross-platform — requires Vulkan SDK)"
echo "  4) Metal  (macOS — enabled by default on macOS)"
echo "  5) CPU only (no GPU acceleration)"
echo ""
read -rp "Choice [1-5, default=1]: " gpu_choice
gpu_choice="${gpu_choice:-1}"

CMAKE_ARGS=()

case "$gpu_choice" in
    1)
        # Try to find nvcc if not on PATH
        if ! command -v nvcc &>/dev/null; then
            for candidate in /opt/cuda/bin /usr/local/cuda/bin /usr/local/cuda-*/bin; do
                if [ -f "$candidate/nvcc" ]; then
                    echo "Found nvcc at $candidate (not on PATH — adding)"
                    export PATH="$candidate:$PATH"
                    break
                fi
            done
        fi
        if ! command -v nvcc &>/dev/null; then
            echo "WARNING: nvcc not found. CUDA build may fail."
            echo "Install the CUDA toolkit or set PATH to include nvcc."
            read -rp "Continue anyway? [y/N]: " cont
            [[ "$cont" =~ ^[Yy] ]] || exit 1
        else
            echo "Using nvcc: $(command -v nvcc)"
        fi
        CMAKE_ARGS+=("-DGGML_CUDA=ON")
        echo "Backend: CUDA"
        ;;
    2)
        CMAKE_ARGS+=("-DGGML_HIP=ON")
        read -rp "GPU target (e.g. gfx1030, gfx1100, or blank for auto): " hip_target
        if [ -n "$hip_target" ]; then
            CMAKE_ARGS+=("-DGPU_TARGETS=$hip_target")
        fi
        echo "Backend: HIP (ROCm)"
        ;;
    3)
        CMAKE_ARGS+=("-DGGML_VULKAN=ON")
        echo "Backend: Vulkan"
        ;;
    4)
        # Metal is enabled by default on macOS, nothing to add
        echo "Backend: Metal (default on macOS)"
        ;;
    5)
        echo "Backend: CPU only (no GPU acceleration — inference will be slow)"
        ;;
    *)
        echo "Invalid choice: $gpu_choice"
        exit 1
        ;;
esac

# ── Build type ───────────────────────────────────────────────────────

echo ""
read -rp "Build type [Release/Debug, default=Release]: " build_type
build_type="${build_type:-Release}"
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=$build_type")

# ── Parallel jobs ────────────────────────────────────────────────────

# Default to number of cores, capped at 8 to be polite
available_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
default_jobs=$(( available_cores > 8 ? 8 : available_cores ))

echo ""
echo "Detected $available_cores CPU cores."
read -rp "Parallel build jobs [default=$default_jobs]: " jobs
jobs="${jobs:-$default_jobs}"

# ── Confirm and build ────────────────────────────────────────────────

echo ""
echo "─── Build summary ───────────────────────────"
echo "  llama.cpp:  $LLAMA_DIR"
echo "  cmake args: ${CMAKE_ARGS[*]}"
echo "  build type: $build_type"
echo "  jobs:       $jobs"
echo "──────────────────────────────────────────────"
echo ""
read -rp "Proceed? [Y/n]: " confirm
confirm="${confirm:-Y}"
[[ "$confirm" =~ ^[Yy] ]] || { echo "Aborted."; exit 0; }

echo ""
echo "Configuring..."
cd "$LLAMA_DIR"
cmake -B build "${CMAKE_ARGS[@]}"

echo ""
echo "Building with $jobs parallel jobs..."
cmake --build build --config "$build_type" -j "$jobs"

echo ""
echo "=== Build complete ==="
echo "  Server:  $LLAMA_DIR/build/bin/llama-server"
echo "  CLI:     $LLAMA_DIR/build/bin/llama-mtmd-cli"
