"""CLI tool for applying llama-video patches to a llama.cpp checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def find_patches() -> list[Path]:
    """Find patch files shipped with the package."""
    pkg_dir = Path(__file__).parent

    # Pip install: patches bundled inside the package via force-include
    bundled = pkg_dir / "patches"
    if bundled.is_dir():
        patches = sorted(bundled.glob("*.patch"))
        if patches:
            return patches

    # Source checkout: patches/ at the project root
    source = pkg_dir.parent.parent / "patches"
    if source.is_dir():
        patches = sorted(source.glob("*.patch"))
        if patches:
            return patches

    return []


def main() -> None:
    """Apply llama-video patches to a llama.cpp checkout.

    Usage: llama-video-patch /path/to/llama.cpp
    """
    if len(sys.argv) < 2:
        print("Usage: llama-video-patch /path/to/llama.cpp")
        print("")
        print("Applies the llama-video patches to a llama.cpp checkout.")
        print("The patches add video captioning support (super-frames, temporal M-RoPE).")
        sys.exit(1)

    llama_dir = Path(sys.argv[1]).resolve()

    if not llama_dir.is_dir():
        print(f"Error: {llama_dir} is not a directory")
        sys.exit(1)

    if not (llama_dir / ".git").exists():
        print(f"Warning: {llama_dir} doesn't appear to be a git repository")

    patches = find_patches()
    if not patches:
        print("Error: No patch files found. Is the package installed correctly?")
        sys.exit(1)

    print(f"Applying {len(patches)} patch(es) to {llama_dir}...")

    for patch in patches:
        print(f"  Applying: {patch.name}")

        # Check if patch applies cleanly first
        check = subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=str(llama_dir),
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            print(f"  WARNING: Patch may not apply cleanly: {patch.name}")
            print(f"  {check.stderr.strip()}")
            print("  Attempting to apply anyway...")

        result = subprocess.run(
            ["git", "apply", str(patch)],
            cwd=str(llama_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED: {patch.name}")
            print(f"  {result.stderr.strip()}")
            print("")
            print("Patch failed. This may mean:")
            print("  - The llama.cpp version is incompatible")
            print("  - The patch was already applied")
            print("  - There are local modifications in the way")
            sys.exit(1)

    print("")
    print("All patches applied successfully!")
    print("Next: build llama.cpp (cmake -B build && cmake --build build)")
