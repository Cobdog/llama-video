"""Image captioning support.

Handles loading, encoding, and message construction for single images.
No super-framing, no temporal positions — images go directly to llama-server
via the standard image_url content type.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image file and return as RGB numpy array.

    Args:
        path: Path to image file (jpg, png, webp, etc.).

    Returns:
        (H, W, 3) uint8 RGB array.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img: Image.Image = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")

    return np.array(img)


def _encode_image_base64(image_data: np.ndarray, quality: int = 95) -> str:
    """Encode a numpy RGB array as a base64 JPEG data URI."""
    img = Image.fromarray(image_data)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_image_message(path: str, prompt: str) -> dict[str, Any]:
    """Build an OpenAI-compatible message with a single image.

    Args:
        path: Path to image file.
        prompt: Text prompt.

    Returns:
        Message dict ready for /v1/chat/completions.
    """
    image_data = load_image(path)
    image_url = _encode_image_base64(image_data)

    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
        {
            "type": "text",
            "text": prompt,
        },
    ]

    return {
        "role": "user",
        "content": content,
    }
