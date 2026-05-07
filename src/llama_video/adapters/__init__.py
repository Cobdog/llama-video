"""Model adapter package: registry + base class + adapters."""

# Import adapter implementations to trigger registration.
# Each module calls register_adapter() at import time.
from llama_video.adapters import gemma as _gemma  # noqa: F401
from llama_video.adapters import qwen as _qwen  # noqa: F401
from llama_video.adapters.base import AdapterPreset, ModelAdapter
from llama_video.adapters.registry import (
    AdapterNotFoundError,
    default_adapter_name,
    get_adapter,
    list_adapters,
    register_adapter,
)

__all__ = [
    "AdapterNotFoundError",
    "AdapterPreset",
    "ModelAdapter",
    "default_adapter_name",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
