"""FastAPI service for video captioning."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from llama_video import __version__
from llama_video.client import LlamaServerClient
from llama_video.config import Settings
from llama_video.errors import ExtractionError, LlamaVideoError, PreprocessingError
from llama_video.extractor import Extractor, ExtractorConfig
from llama_video.preprocessor import Preprocessor
from llama_video.types import (
    CaptionMetadata,
    CaptionRequest,
    CaptionResponse,
    DebugInfo,
    HealthResponse,
)

logger = logging.getLogger(__name__)

# Module-level state (set during lifespan)
_settings: Settings | None = None
_extractor: Extractor | None = None
_preprocessor: Preprocessor | None = None
_client: LlamaServerClient | None = None
_last_debug: DebugInfo = DebugInfo()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and tear down application state."""
    global _settings, _extractor, _preprocessor, _client

    _settings = Settings()
    _extractor = Extractor(_settings.extractor)
    _preprocessor = Preprocessor(_settings.model)
    _client = LlamaServerClient(_settings.server)

    logging.basicConfig(
        level=getattr(logging, _settings.service.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("llama-video service starting (v%s)", __version__)
    logger.info("llama-server URL: %s", _settings.server.url)

    yield

    if _client is not None:
        await _client.close()
    logger.info("llama-video service stopped")


app = FastAPI(
    title="llama-video",
    version=__version__,
    description="Video captioning service for Qwen3.5 GGUF models via patched llama.cpp",
    lifespan=lifespan,
)


@app.post("/v1/caption", response_model=CaptionResponse)
async def caption_video(request: CaptionRequest) -> CaptionResponse:
    """Caption a video file.

    Extracts frames, preprocesses into super-frames, sends to
    patched llama-server, returns caption with metadata.
    """
    global _last_debug
    assert _extractor is not None
    assert _preprocessor is not None
    assert _client is not None

    start_time = time.monotonic()
    debug: dict[str, Any] = {"request": request.model_dump()}

    try:
        # Extract frames
        t0 = time.monotonic()
        extract_config = ExtractorConfig(fps=request.fps, max_frames=request.max_frames)
        frames = await _extractor.extract_frames_async(request.video_path, extract_config)
        debug["frames_extracted"] = len(frames)
        debug["timing_extract_ms"] = (time.monotonic() - t0) * 1000

        # Preprocess
        t0 = time.monotonic()
        video_input = _preprocessor.process(frames, fps=request.fps)
        debug["super_frame_shapes"] = [sf.shape for sf in video_input.super_frames]
        debug["grid_thw"] = video_input.grid_thw
        debug["temporal_positions"] = video_input.temporal_positions
        debug["timing_preprocess_ms"] = (time.monotonic() - t0) * 1000

        # Caption
        t0 = time.monotonic()
        from llama_video.config import get_preset

        preset = get_preset(request.preset)
        caption = await _client.caption_video(
            video_input,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            preset=preset,
        )
        debug["timing_inference_ms"] = (time.monotonic() - t0) * 1000

        total_ms = (time.monotonic() - start_time) * 1000

        metadata = CaptionMetadata(
            frames_extracted=len(frames),
            super_frames=len(video_input.super_frames),
            grid_thw=video_input.grid_thw,
            processing_time_ms=total_ms,
        )

        _last_debug = DebugInfo(
            request=debug.get("request"),
            frames_extracted=debug.get("frames_extracted"),
            super_frame_shapes=debug.get("super_frame_shapes"),
            grid_thw=debug.get("grid_thw"),
            temporal_positions=debug.get("temporal_positions"),
            timing={k: v for k, v in debug.items() if k.startswith("timing_")},
        )

        return CaptionResponse(caption=caption, metadata=metadata)

    except ExtractionError as e:
        logger.error("Extraction failed: %s", e, extra={"context": e.context})
        _last_debug = DebugInfo(error=str(e))
        raise HTTPException(status_code=422, detail=str(e)) from e

    except PreprocessingError as e:
        logger.error("Preprocessing failed: %s", e, extra={"context": e.context})
        _last_debug = DebugInfo(error=str(e))
        raise HTTPException(status_code=422, detail=str(e)) from e

    except LlamaVideoError as e:
        logger.error("Captioning failed: %s", e, extra={"context": e.context})
        _last_debug = DebugInfo(error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service and llama-server health."""
    assert _client is not None
    assert _settings is not None

    llama_ok = await _client.health_check()
    return HealthResponse(
        status="ok" if llama_ok else "degraded",
        llama_server_reachable=llama_ok,
        llama_server_url=_settings.server.url,
        version=__version__,
    )


@app.get("/v1/debug/last-request", response_model=DebugInfo)
async def get_debug_info() -> DebugInfo:
    """Return diagnostic info about the most recent request."""
    return _last_debug


def main() -> None:
    """Entry point for llama-video-server CLI."""
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "llama_video.server:app",
        host=settings.service.host,
        port=settings.service.port,
        workers=settings.service.workers,
        log_level=settings.service.log_level.lower(),
    )
