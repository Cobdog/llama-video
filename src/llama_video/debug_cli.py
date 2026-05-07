"""Debug CLI for llama-video diagnostics.

Usage:
    llama-video-debug extract <video> [--fps N] [--output-dir DIR]
    llama-video-debug preprocess <video> [--fps N] [--model qwen3.5]
    llama-video-debug compare-modes <video> [--server-url URL]
    llama-video-debug trace <video> [--prompt TEXT] [--server-url URL]
    llama-video-debug validate-patch [--server-url URL]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract and inspect frames from a video."""
    from llama_video.config import ExtractorSettings
    from llama_video.extractor import Extractor, ExtractorConfig

    extractor = Extractor(ExtractorSettings())
    config = ExtractorConfig(fps=args.fps, max_frames=args.max_frames)
    frames = extractor.extract_frames(args.video, config)

    print(f"Extracted {len(frames)} frames:")
    for f in frames:
        print(f"  {f}")

    if args.output_dir:
        from PIL import Image

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for f in frames:
            img = Image.fromarray(f.data)
            img.save(out / f"frame_{f.index:04d}.jpg")
        print(f"\nFrames saved to {out}/")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocess video and inspect super-frames."""
    from llama_video.adapters import get_adapter
    from llama_video.config import ExtractorSettings
    from llama_video.extractor import Extractor, ExtractorConfig

    extractor = Extractor(ExtractorSettings())
    config = ExtractorConfig(fps=args.fps, max_frames=args.max_frames)
    frames = extractor.extract_frames(args.video, config)

    adapter = get_adapter(args.model)
    result = adapter.preprocess(frames, fps=args.fps)

    print(f"Source frames: {result.num_source_frames}")
    print(f"Super-frames:  {len(result.super_frames)}")
    print(f"Resolution:    {result.resolution[0]}x{result.resolution[1]}")
    print(f"Grid THW:      T={result.grid_thw[0]}, H={result.grid_thw[1]}, W={result.grid_thw[2]}")
    print(f"Temporal pos:  {result.temporal_positions}")
    total_tokens = result.grid_thw[0] * result.grid_thw[1] * result.grid_thw[2]
    print(f"Vision tokens: {total_tokens}")
    print()
    for sf in result.super_frames:
        print(f"  {sf}")


def cmd_validate_patch(args: argparse.Namespace) -> None:
    """Validate that the llama-server video patch is active."""
    from llama_video.client import LlamaServerClient
    from llama_video.config import ServerConfig

    async def _check() -> None:
        config = ServerConfig(url=args.server_url)
        client = LlamaServerClient(config)
        try:
            healthy = await client.health_check()
            if healthy:
                print(f"llama-server at {args.server_url}: HEALTHY")
                # TODO: Add patch detection once the patch exposes a version endpoint
                print("Patch validation: not yet implemented (requires C patch)")
            else:
                print(f"llama-server at {args.server_url}: UNREACHABLE")
                sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_check())


def main() -> None:
    """Entry point for llama-video-debug CLI."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="llama-video-debug",
        description="Diagnostic tools for llama-video",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract and inspect frames")
    p_extract.add_argument("video", help="Path to video file")
    p_extract.add_argument("--fps", type=float, default=2.0)
    p_extract.add_argument("--max-frames", type=int, default=64)
    p_extract.add_argument("--output-dir", help="Save frames to directory")

    # preprocess
    p_preprocess = subparsers.add_parser("preprocess", help="Preprocess and inspect super-frames")
    p_preprocess.add_argument("video", help="Path to video file")
    p_preprocess.add_argument("--fps", type=float, default=2.0)
    p_preprocess.add_argument("--max-frames", type=int, default=64)
    p_preprocess.add_argument("--model", default="qwen3.5", help="Model profile")

    # validate-patch
    p_validate = subparsers.add_parser("validate-patch", help="Check if video patch is active")
    p_validate.add_argument("--server-url", default="http://localhost:8080")

    args = parser.parse_args()

    commands = {
        "extract": cmd_extract,
        "preprocess": cmd_preprocess,
        "validate-patch": cmd_validate_patch,
    }

    cmd = commands.get(args.command)
    if cmd is None:
        parser.error(f"Unknown command: {args.command}")
    else:
        cmd(args)


if __name__ == "__main__":
    main()
