#!/usr/bin/env bash
# Extract current llama.cpp changes as numbered patch files
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
PATCHES_DIR="$PROJECT_DIR/patches"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "Error: llama.cpp not found at $LLAMA_DIR"
    exit 1
fi

cd "$LLAMA_DIR"

# Check for uncommitted changes
if git diff --quiet && git diff --cached --quiet; then
    echo "No changes in llama.cpp to extract."
    exit 0
fi

echo "Extracting patches from llama.cpp changes..."

# Create a single combined patch (can be split manually later)
timestamp=$(date +%Y%m%d)
patch_file="$PATCHES_DIR/video-support-${timestamp}.patch"

git diff > "$patch_file"

echo "Patch saved to: $patch_file"
echo "Lines: $(wc -l < "$patch_file")"
echo ""
echo "To split into numbered patches, manually separate the diff by file"
echo "and rename to 01-feature.patch, 02-feature.patch, etc."
