#!/usr/bin/env bash
# Apply video patches to llama.cpp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="${1:-$PROJECT_DIR/llama.cpp}"
PATCHES_DIR="$PROJECT_DIR/patches"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "Error: llama.cpp not found at $LLAMA_DIR"
    echo "Usage: $0 [/path/to/llama.cpp]"
    exit 1
fi

# Count available patches
patch_count=$(find "$PATCHES_DIR" -name "*.patch" 2>/dev/null | wc -l)

if [ "$patch_count" -eq 0 ]; then
    echo "No patches found in $PATCHES_DIR — nothing to apply."
    echo "(Patches will be created during Phase 2-3 of implementation.)"
    exit 0
fi

echo "Applying $patch_count patch(es) to llama.cpp..."

cd "$LLAMA_DIR"
for patch in "$PATCHES_DIR"/*.patch; do
    echo "  Applying: $(basename "$patch")"
    git apply "$patch" || {
        echo "  FAILED: $(basename "$patch")"
        echo "  Try: cd llama.cpp && git apply --check ../patches/$(basename "$patch")"
        exit 1
    }
done

echo "All patches applied successfully."
