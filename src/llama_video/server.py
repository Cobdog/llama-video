"""FastAPI service for video captioning."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from llama_video import __version__
from llama_video.adapters import AdapterPreset, get_adapter
from llama_video.client import LlamaServerClient
from llama_video.config import Settings, get_preset
from llama_video.errors import ExtractionError, LlamaVideoError, PreprocessingError
from llama_video.extractor import Extractor, ExtractorConfig
from llama_video.segmenter import segment_video
from llama_video.types import (
    CaptionMetadata,
    CaptionRequest,
    CaptionResponse,
    DebugInfo,
    HealthResponse,
)

logger = logging.getLogger(__name__)

# Module-level state (set during lifespan)
# NOTE: _last_debug is overwritten on every caption request. Under concurrent
# load, /v1/debug/last-request may return a different request's debug info.
_settings: Settings | None = None
_extractor: Extractor | None = None
_client: LlamaServerClient | None = None
_last_debug: DebugInfo = DebugInfo()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and tear down application state."""
    global _settings, _extractor, _client

    _settings = Settings()
    _extractor = Extractor(_settings.extractor)
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
    description="Multi-model video captioning service via patched llama.cpp",
    lifespan=lifespan,
)


@app.post("/v1/caption", response_model=CaptionResponse)
async def caption_video(request: CaptionRequest) -> CaptionResponse:
    """Caption a video file.

    Extracts frames, preprocesses through the model adapter, sends to
    llama-server, returns caption with metadata. Automatically segments
    long videos based on adapter duration limits.
    """
    global _last_debug
    assert _extractor is not None
    assert _client is not None

    start_time = time.monotonic()
    debug: dict[str, Any] = {"request": request.model_dump()}

    try:
        # Resolve adapter — "auto" triggers detection from llama-server
        if request.model_profile == "auto":
            from llama_video.adapters.detect import detect_adapter

            adapter_name = await detect_adapter(_settings.server.url)
            adapter = get_adapter(adapter_name)
            debug["auto_detected"] = adapter_name
        else:
            adapter = get_adapter(request.model_profile)
        debug["adapter"] = adapter.name

        # Get video info for segmentation decision
        from pathlib import Path

        video_path = Path(request.video_path)
        _, _, duration = await _extractor._get_video_info(video_path)

        # Determine chunk size
        chunk_seconds = request.chunk_duration_seconds
        max_duration = adapter.max_duration_seconds

        # Segment if video exceeds adapter limits
        chunks = segment_video(
            total_duration=duration,
            chunk_seconds=chunk_seconds if chunk_seconds is not None else max_duration,
            max_chunk_seconds=max_duration if max_duration != float("inf") else None,
        )
        debug["chunks"] = len(chunks)

        # Build adapter preset from request
        inference_preset = get_preset(request.preset)
        adapter_preset = AdapterPreset(
            temperature=request.temperature
            if request.temperature is not None
            else inference_preset.temperature,
            top_p=inference_preset.top_p,
            top_k=inference_preset.top_k,
            min_p=inference_preset.min_p,
            presence_penalty=inference_preset.presence_penalty,
        )
        model_name = _settings.server.model_name if _settings else ""

        captions: list[str] = []
        total_frames = 0
        total_super_frames = 0
        last_grid_thw: tuple[int, int, int] = (0, 0, 0)

        for chunk in chunks:
            # Extract frames for this chunk
            t0 = time.monotonic()
            extract_config = ExtractorConfig(
                fps=request.fps,
                max_frames=request.max_frames,
                start_time=chunk.start_seconds if len(chunks) > 1 else None,
                duration=chunk.duration if len(chunks) > 1 else None,
            )
            frames = await _extractor.extract_frames_async(request.video_path, extract_config)
            total_frames += len(frames)
            debug.setdefault("timing_extract_ms", 0)
            debug["timing_extract_ms"] += (time.monotonic() - t0) * 1000

            if not frames:
                continue

            # Preprocess through adapter
            t0 = time.monotonic()
            video_input = adapter.preprocess(frames, fps=request.fps)
            total_super_frames += len(video_input.super_frames)
            last_grid_thw = video_input.grid_thw
            debug["grid_thw"] = video_input.grid_thw
            debug["temporal_positions"] = video_input.temporal_positions
            debug.setdefault("timing_preprocess_ms", 0)
            debug["timing_preprocess_ms"] += (time.monotonic() - t0) * 1000

            # Build payload and send
            payload = adapter.build_payload(
                video_input,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                preset=adapter_preset,
                model_name=model_name,
            )

            t0 = time.monotonic()
            result = await _client.send_completion(payload)
            # Use structured reasoning from transport if available, otherwise
            # fall back to in-text tag parsing (for models that embed thinking inline)
            if result.reasoning:
                caption, thinking, truncated = adapter.parse_response(result.content)
                thinking = result.reasoning
            else:
                caption, thinking, truncated = adapter.parse_response(result.content)
            debug.setdefault("timing_inference_ms", 0)
            debug["timing_inference_ms"] += (time.monotonic() - t0) * 1000

            if thinking:
                logger.info(
                    "Chunk %d thinking: %d chars%s",
                    chunk.index,
                    len(thinking),
                    " (truncated)" if truncated else "",
                )

            if caption:
                captions.append(caption)

        # Combine chunk captions
        if len(captions) > 1:
            combined = "\n\n".join(captions)
        elif captions:
            combined = captions[0]
        else:
            combined = ""

        total_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Caption received: %d chars (%.0fms, %d chunks)", len(combined), total_ms, len(chunks)
        )

        metadata = CaptionMetadata(
            frames_extracted=total_frames,
            super_frames=total_super_frames,
            grid_thw=last_grid_thw,
            processing_time_ms=total_ms,
        )

        _last_debug = DebugInfo(
            request=debug.get("request"),
            frames_extracted=total_frames,
            super_frame_shapes=debug.get("super_frame_shapes"),
            grid_thw=debug.get("grid_thw"),
            temporal_positions=debug.get("temporal_positions"),
            timing={k: v for k, v in debug.items() if k.startswith("timing_")},
        )

        return CaptionResponse(caption=combined, metadata=metadata)

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
