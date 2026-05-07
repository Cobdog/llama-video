"""Adapter registry: name → adapter class mapping + factory."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_video.adapters.base import ModelAdapter

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[ModelAdapter]] = {}

_DEFAULT_ADAPTER = "qwen3.5"


class AdapterNotFoundError(ValueError):
    """Raised when an unknown adapter name is requested."""


def register_adapter(name: str, cls: type[ModelAdapter]) -> None:
    """Register an adapter class under a profile name."""
    if name in _REGISTRY:
        logger.warning("Overwriting adapter '%s': %s → %s", name, _REGISTRY[name], cls)
    _REGISTRY[name] = cls
    logger.debug("Registered adapter '%s': %s", name, cls.__name__)


def get_adapter(name: str | None = None) -> ModelAdapter:
    """Return an adapter instance by profile name.

    Args:
        name: Profile name (e.g. 'qwen3.5', 'gemma4'). None or empty
              returns the default adapter.

    Raises:
        AdapterNotFoundError: If the name is not registered.
    """
    resolved = name.strip().lower() if name else _DEFAULT_ADAPTER
    if resolved not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise AdapterNotFoundError(f"Unknown adapter '{resolved}'. Available: {available}")
    return _REGISTRY[resolved]()


def list_adapters() -> list[str]:
    """Return sorted list of registered adapter profile names."""
    return sorted(_REGISTRY)


def default_adapter_name() -> str:
    """Return the default adapter profile name."""
    return _DEFAULT_ADAPTER
