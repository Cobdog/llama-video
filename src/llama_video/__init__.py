"""llama-video: Temporal video support for llama.cpp.

Frame extraction, multi-model adapters, and captioning service for multimodal GGUF models.
"""

from llama_video.batch import batch_caption, detect_mode, validate_batch_mode
from llama_video.config import (
    PRESETS,
    InferencePreset,
    ModelConfig,
    ServerConfig,
    Settings,
    get_preset,
)
from llama_video.extractor import Extractor, ExtractorConfig
from llama_video.history import CaptionHistory
from llama_video.image import build_image_message, load_image
from llama_video.preprocessor import Preprocessor, VideoInput
from llama_video.templates import (
    BUILT_IN_TEMPLATES,
    PromptTemplate,
    get_template,
    get_templates,
    render_template,
)
from llama_video.tokens import TokenBudget, TokenEstimator
from llama_video.types import CaptionResult, Frame, SuperFrame

__all__ = [
    "BUILT_IN_TEMPLATES",
    "PRESETS",
    "CaptionHistory",
    "CaptionResult",
    "Extractor",
    "ExtractorConfig",
    "Frame",
    "InferencePreset",
    "ModelConfig",
    "Preprocessor",
    "PromptTemplate",
    "ServerConfig",
    "Settings",
    "SuperFrame",
    "TokenBudget",
    "TokenEstimator",
    "VideoInput",
    "batch_caption",
    "build_image_message",
    "detect_mode",
    "get_preset",
    "get_template",
    "get_templates",
    "load_image",
    "render_template",
    "validate_batch_mode",
]

__version__ = "0.1.3"
