"""Gradio WebUI for llama-video experimentation.

Provides an interactive interface for video and image captioning with
full parameter control, frame preview, token budget estimation,
side-by-side comparison, and caption history.

Requires the 'ui' optional dependency group:
    pip install llama-video[ui]
"""

from __future__ import annotations

import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import gradio as gr

from llama_video.adapters import AdapterPreset, get_adapter, list_adapters
from llama_video.adapters.detect import detect_adapter as _detect_adapter
from llama_video.batch import detect_mode
from llama_video.client import LlamaServerClient
from llama_video.config import (
    PRESETS,
    ServerConfig,
    get_preset,
)
from llama_video.errors import LlamaVideoError
from llama_video.extractor import Extractor, ExtractorConfig
from llama_video.history import CaptionHistory
from llama_video.image import build_image_message, load_image
from llama_video.templates import BUILT_IN_TEMPLATES
from llama_video.tokens import TokenEstimator
from llama_video.types import CaptionMetadata, CaptionResult

logger = logging.getLogger(__name__)

_PRESET_NAMES = list(PRESETS.keys())
_TEMPLATE_NAMES = ["custom", *list(BUILT_IN_TEMPLATES.keys())]
_DEFAULT = PRESETS["default"]
_FILE_TYPES = [
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".mkv",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
]


# ── HTML helpers ─────────────────────────────────────────────


def _fmt(n: int) -> str:
    """Format integer with thousands separator."""
    return f"{n:,}"


def build_budget_html(
    vision: int,
    prompt: int,
    generation: int,
    context_limit: int,
) -> str:
    """Render token budget as a stacked HTML bar chart."""
    total = vision + prompt + generation
    headroom = max(0, context_limit - total)
    over = max(0, total - context_limit)
    denom = max(total, context_limit, 1)

    def _seg(val: int, color: str, label: str) -> str:
        if val <= 0:
            return ""
        pct = max(val / denom * 100, 0.4)
        return (
            f'<div style="width:{pct:.2f}%;background:{color};'
            f"display:flex;align-items:center;"
            f"justify-content:center;color:#fff;"
            f"font-size:11px;overflow:hidden;"
            f'white-space:nowrap;padding:0 4px">'
            f"{label}: {_fmt(val)}</div>"
        )

    bar = (
        _seg(vision, "#4a90d9", "Vision")
        + _seg(prompt, "#7b68ee", "Prompt")
        + _seg(generation, "#50c878", "Gen")
        + _seg(headroom, "#2d2d44", "Free")
    )

    pct_used = total / context_limit * 100 if context_limit else 0
    color = "#ff4444" if over else ("#ffaa00" if pct_used > 85 else "#aaa")
    warn = f' — <b style="color:#ff4444">Over by {_fmt(over)}</b>' if over else ""

    return (
        '<div style="font-family:monospace;padding:4px 0">'
        '<div style="display:flex;height:26px;border-radius:4px;'
        f'overflow:hidden;border:1px solid #444">{bar}</div>'
        '<div style="display:flex;justify-content:space-between;'
        f'font-size:11px;margin-top:3px">'
        f'<span style="color:{color}">'
        f"Total: {_fmt(total)} / {_fmt(context_limit)} "
        f"({pct_used:.1f}%){warn}</span></div></div>"
    )


def build_metadata_html(
    mode: str,
    frames: int,
    super_frames: int,
    grid_thw: tuple[int, int, int],
    resolution: tuple[int, int],
    duration_ms: float,
) -> str:
    """Render caption result metadata as an HTML table."""
    t, h, w = grid_thw
    rw, rh = resolution
    rows = [
        ("Mode", mode),
        ("Frames", str(frames)),
        ("Super-frames", str(super_frames)),
        ("Grid T&times;H&times;W", f"{t}&times;{h}&times;{w} = {t * h * w:,} tokens"),
        ("Resolution", f"{rw}&times;{rh}"),
        ("Time", f"{duration_ms / 1000:.1f}s"),
    ]
    trs = "".join(
        f'<tr><td style="color:#888;padding:2px 8px 2px 0">{k}</td><td>{v}</td></tr>'
        for k, v in rows
    )
    return f'<table style="font-family:monospace;font-size:13px">{trs}</table>'


# ── Budget computation (sync — for live updates) ────────────


_RESOLUTION_SCALES: list[tuple[str, float]] = [
    ("Full", 1.0),
    ("3/4", 0.75),
    ("1/2", 0.5),
    ("1/4", 0.25),
    ("1/8", 0.125),
]
_RESOLUTION_NAMES = [name for name, _ in _RESOLUTION_SCALES]
_RESOLUTION_MAP = dict(_RESOLUTION_SCALES)


def _dur_to_frames(caption_dur: float, fps: float) -> int:
    """Convert caption duration + FPS to frame count."""
    return max(2, math.ceil(caption_dur * fps))


def compute_budget(
    info: dict[str, Any],
    fps: float,
    caption_dur: float,
    prompt: str,
    max_tokens: int,
    context_limit: int,
    resolution_scale: float = 1.0,
) -> str:
    """Compute token budget HTML from settings + video info."""
    if not info or not info.get("width"):
        return build_budget_html(0, 0, max_tokens, context_limit)

    mode = info.get("mode", "video")
    w = max(1, int(int(info["width"]) * resolution_scale))
    h = max(1, int(int(info["height"]) * resolution_scale))

    if mode == "video":
        fc = _dur_to_frames(caption_dur, fps)
    else:
        fc = 1
        fps = 1.0

    est = TokenEstimator()
    b = est.estimate_from_settings(
        frame_count=fc,
        resolution=(w, h),
        fps=fps,
        prompt=prompt,
        max_tokens=max_tokens,
        context_limit=context_limit,
    )
    return build_budget_html(
        b.vision_tokens,
        b.prompt_tokens,
        b.generation_budget,
        context_limit,
    )


# ── Async operations ─────────────────────────────────────────


async def _probe_video(path: str) -> dict[str, Any]:
    """Get video dimensions and duration via ffprobe."""
    ext = Extractor()
    w, h, dur = await ext._get_video_info(Path(path))
    return {
        "width": w,
        "height": h,
        "duration": dur,
        "path": path,
        "mode": "video",
    }


async def _extract_preview(
    path: str,
    fps: float,
    max_frames: int,
) -> list[tuple[Any, str]]:
    """Extract frames for gallery preview."""
    mode = detect_mode(path)
    if mode == "video":
        ext = Extractor()
        cfg = ExtractorConfig(fps=fps, max_frames=int(max_frames))
        frames = await ext.extract_frames_async(path, cfg)
        return [(f.data, f"Frame {i} ({f.timestamp:.1f}s)") for i, f in enumerate(frames)]
    img = load_image(path)
    ih, iw = img.shape[:2]
    return [(img, f"Image ({iw}\u00d7{ih})")]


async def _do_caption(
    path: str,
    mode: str,
    prompt: str,
    fps: float,
    max_frames: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    server_url: str,
    timeout: float,
    resolution_scale: float = 1.0,
    cache_prompt: bool = True,
    profile: str = "auto",
) -> tuple[str, str, str, bool, CaptionResult]:
    """Full pipeline: extract, preprocess, infer, return.

    Returns (caption, thinking, metadata_html, truncated, CaptionResult).
    """
    # Resolve adapter profile
    if profile == "auto":
        adapter_name = await _detect_adapter(server_url)
    else:
        adapter_name = profile
    adapter = get_adapter(adapter_name)

    cfg = ServerConfig(url=server_url, timeout=timeout)
    client = LlamaServerClient(cfg)
    adapter_preset = AdapterPreset(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
    )
    start = time.monotonic()
    try:
        if mode == "video":
            return await _caption_video(
                path,
                fps,
                max_frames,
                max_tokens,
                prompt,
                adapter_preset,
                client,
                start,
                resolution_scale,
                cache_prompt,
                adapter,
            )
        return await _caption_image(
            path,
            max_tokens,
            prompt,
            adapter_preset,
            client,
            start,
            cache_prompt,
            adapter,
        )
    finally:
        await client.close()


async def _caption_video(
    path: str,
    fps: float,
    max_frames: int,
    max_tokens: int,
    prompt: str,
    preset: AdapterPreset,
    client: LlamaServerClient,
    start: float,
    resolution_scale: float = 1.0,
    cache_prompt: bool = True,
    adapter: Any = None,
) -> tuple[str, str, str, bool, CaptionResult]:
    """Video captioning sub-pipeline."""
    ext = Extractor()
    frames = await ext.extract_frames_async(
        path,
        ExtractorConfig(fps=fps, max_frames=max_frames),
    )

    vi = adapter.preprocess(frames, fps=fps, resolution_scale=resolution_scale)
    payload = adapter.build_payload(
        vi,
        prompt=prompt,
        max_tokens=max_tokens,
        preset=preset,
        model_name=client._config.model_name if client._config.model_name else "",
        cache_prompt=cache_prompt,
    )
    result = await client.send_completion(payload)
    cap, thinking, truncated = adapter.parse_response(result.content)
    if result.reasoning:
        thinking = result.reasoning
    ms = (time.monotonic() - start) * 1000
    meta = build_metadata_html(
        "video",
        len(frames),
        len(vi.super_frames),
        vi.grid_thw,
        vi.resolution,
        ms,
    )
    res = CaptionResult(
        caption=cap,
        source_path=path,
        mode="video",
        prompt_rendered=prompt,
        template_name=None,
        variables={},
        preset_name="webui",
        settings={
            "fps": fps,
            "max_frames": max_frames,
            "max_tokens": max_tokens,
        },
        metadata=CaptionMetadata(
            frames_extracted=len(frames),
            super_frames=len(vi.super_frames),
            grid_thw=vi.grid_thw,
            processing_time_ms=ms,
        ),
        token_usage=None,
        duration_ms=ms,
    )
    return cap, thinking, meta, truncated, res


async def _caption_image(
    path: str,
    max_tokens: int,
    prompt: str,
    preset: AdapterPreset,
    client: LlamaServerClient,
    start: float,
    cache_prompt: bool = True,
    adapter: Any = None,
) -> tuple[str, str, str, bool, CaptionResult]:
    """Image captioning sub-pipeline using adapter for payload and parsing."""
    # Build image message + system message payload through adapter-compatible path
    image_msg = build_image_message(path, prompt)
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a media captioning assistant. Your reasoning is private and "
                    "will not be shown to the user. Your response must contain the complete, "
                    "detailed caption — do not summarize or abbreviate what you described "
                    "in your reasoning. Write the full description in your response."
                ),
            },
            image_msg,
        ],
        "max_tokens": max_tokens,
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "top_k": preset.top_k,
        "min_p": preset.min_p,
        "presence_penalty": preset.presence_penalty,
        "cache_prompt": cache_prompt,
    }
    model_name = client._config.model_name if client._config.model_name else ""
    if model_name:
        payload["model"] = model_name

    result = await client.send_completion(payload)

    # Parse through adapter if available (handles model-specific thinking tags)
    if adapter is not None:
        cap, thinking, truncated = adapter.parse_response(result.content)
    else:
        from llama_video.client import parse_model_response

        cap, thinking, truncated = parse_model_response(result.content)
    # Use structured reasoning from transport if available
    if result.reasoning:
        thinking = result.reasoning

    ms = (time.monotonic() - start) * 1000
    img = load_image(path)
    ih, iw = img.shape[:2]
    # Use a reasonable grid unit for display metadata
    gu = 28
    tw = max(gu, round(iw / gu) * gu)
    th = max(gu, round(ih / gu) * gu)
    hg, wg = th // gu, tw // gu
    meta = build_metadata_html(
        "image",
        1,
        0,
        (1, hg, wg),
        (tw, th),
        ms,
    )
    res = CaptionResult(
        caption=cap,
        source_path=path,
        mode="image",
        prompt_rendered=prompt,
        template_name=None,
        variables={},
        preset_name="webui",
        settings={"max_tokens": max_tokens},
        metadata=CaptionMetadata(
            frames_extracted=1,
            super_frames=0,
            grid_thw=(1, hg, wg),
            processing_time_ms=ms,
        ),
        token_usage=None,
        duration_ms=ms,
    )
    return cap, thinking, meta, truncated, res


async def _check_health(url: str, timeout: float) -> str:
    """Check llama-server health and loaded model, return status HTML."""
    import httpx as _httpx

    short_timeout = _httpx.Timeout(min(timeout, 10))
    try:
        async with _httpx.AsyncClient(base_url=url, timeout=short_timeout) as c:
            resp = await c.get("/health")
            if resp.status_code != 200:
                return '<span style="color:#ffaa00;font-weight:bold">\u25cf Degraded</span>'

            # Server is up — query loaded model
            model_name = ""
            try:
                mr = await c.get("/v1/models")
                if mr.status_code == 200:
                    data = mr.json().get("data", [])
                    if data:
                        model_name = data[0].get("id", "")
            except Exception:
                pass

            status = '<span style="color:#50c878;font-weight:bold">\u25cf Connected</span>'
            if model_name:
                status += (
                    f'<br><span style="color:#aaa;font-size:12px">Model: <b>{model_name}</b></span>'
                )
            else:
                status += '<br><span style="color:#888;font-size:12px">No model loaded</span>'
            return status
    except (_httpx.ConnectError, _httpx.TimeoutException):
        return '<span style="color:#ff4444;font-weight:bold">\u25cf Unreachable</span>'
    except Exception:
        return '<span style="color:#ff4444;font-weight:bold">\u25cf Unreachable</span>'


def _save_result(result: CaptionResult) -> None:
    """Best-effort save to caption history."""
    try:
        h = CaptionHistory()
        h.save(result)
        h.close()
    except Exception:
        logger.debug("Failed to save to history", exc_info=True)


# ── Shared event handlers ────────────────────────────────────


def _on_template_change(name: str, fp: Any) -> Any:
    """Fill prompt from selected template."""
    if name == "custom":
        return gr.update()
    t = BUILT_IN_TEMPLATES[name]
    try:
        m = detect_mode(fp) if fp else "video"
    except ValueError:
        m = "video"
    return t.template.replace("{media_type}", m)


def _on_preset_change(name: str) -> tuple[Any, ...]:
    """Fill inference sliders from preset."""
    p = get_preset(name)
    return (
        p.temperature,
        p.top_p,
        p.top_k,
        p.min_p,
        p.presence_penalty,
    )


# ── App builder ──────────────────────────────────────────────


def create_app() -> gr.Blocks:
    """Build the complete Gradio application."""
    empty_budget = build_budget_html(0, 0, 2048, 65536)
    checking = '<span style="color:#888">Checking\u2026</span>'
    _default_cfg = ServerConfig()

    with gr.Blocks(
        title="llama-video",
        analytics_enabled=False,
    ) as app:
        gr.Markdown(
            "# llama-video\n"
            "Video & image captioning experimentation "
            "for multimodal models via llama.cpp"
        )

        with gr.Tabs():
            cap_refs = _build_caption_tab(empty_budget, checking, _default_cfg)
            _build_compare_tab(empty_budget, _default_cfg)
            _build_batch_tab(_default_cfg)
            _build_history_tab()

        # Auto-check server on page load
        app.load(
            fn=_check_health,
            inputs=[cap_refs["url"], cap_refs["tout"]],
            outputs=[cap_refs["status"]],
        )

    return app


def _build_caption_tab(
    empty_budget: str,
    not_checked: str,
    default_cfg: ServerConfig,
) -> dict[str, Any]:
    """Build the main Caption tab with all controls.

    Returns dict with 'url', 'tout', and 'status' components
    so the app can wire up auto-check on load.
    """
    with gr.TabItem("Caption"):
        vi_state: gr.State = gr.State(value={})
        # Track visual params from last request to decide cache_prompt.
        # Keys: file, fps, duration, resolution_scale
        last_visual: gr.State = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=3):
                c_file = gr.File(
                    label="Upload Video or Image",
                    type="filepath",
                    file_types=_FILE_TYPES,
                )
                c_mode = gr.Textbox(
                    label="Detected",
                    interactive=False,
                    max_lines=1,
                )
                c_gallery = gr.Gallery(
                    label="Frame Preview",
                    columns=6,
                    rows=2,
                    height="auto",
                    object_fit="contain",
                )
                c_budget = gr.HTML(value=empty_budget)
                c_tmpl = gr.Dropdown(
                    label="Prompt Template",
                    choices=_TEMPLATE_NAMES,
                    value="general",
                )
                c_prompt = gr.Textbox(
                    label="Prompt",
                    value="Describe what happens in this video.",
                    lines=3,
                )
                with gr.Row():
                    c_btn = gr.Button(
                        "Caption",
                        variant="primary",
                        size="lg",
                    )
                    c_cancel = gr.Button(
                        "Cancel",
                        variant="stop",
                        size="lg",
                    )
                c_warn = gr.HTML(visible=False)
                c_out = gr.Textbox(
                    label="Caption Output",
                    lines=10,
                    interactive=False,
                    buttons=["copy"],
                )
                with gr.Accordion("Reasoning", open=False):
                    c_think = gr.Textbox(
                        label="Model Reasoning",
                        lines=8,
                        interactive=False,
                        buttons=["copy"],
                    )
                with gr.Accordion("Metadata", open=False):
                    c_meta = gr.HTML()

            with gr.Column(scale=1, min_width=260):
                _adapter_choices = ["auto", *list_adapters()]
                c_profile = gr.Dropdown(
                    label="Model Profile",
                    choices=_adapter_choices,
                    value="auto",
                )
                gr.Markdown("### Video")
                c_fps = gr.Slider(
                    0.5,
                    10,
                    value=2.0,
                    step=0.5,
                    label="FPS",
                )
                c_dur = gr.Slider(
                    0.5,
                    300,
                    value=30,
                    step=0.5,
                    label="Duration (s)",
                )
                c_fc = gr.Textbox(
                    label="Frames",
                    value="",
                    interactive=False,
                    max_lines=1,
                )
                c_res = gr.Dropdown(
                    label="Resolution",
                    choices=_RESOLUTION_NAMES,
                    value="Full",
                )
                gr.Markdown("### Inference")
                c_preset = gr.Dropdown(
                    label="Preset",
                    choices=_PRESET_NAMES,
                    value="default",
                )
                c_temp = gr.Slider(
                    0,
                    2,
                    value=_DEFAULT.temperature,
                    step=0.05,
                    label="Temperature",
                )
                with gr.Accordion("Advanced", open=False):
                    c_tp = gr.Slider(
                        0,
                        1,
                        value=_DEFAULT.top_p,
                        step=0.01,
                        label="Top P",
                    )
                    c_tk = gr.Slider(
                        1,
                        100,
                        value=_DEFAULT.top_k,
                        step=1,
                        label="Top K",
                    )
                    c_mp = gr.Slider(
                        0,
                        1,
                        value=_DEFAULT.min_p,
                        step=0.01,
                        label="Min P",
                    )
                    c_pp = gr.Slider(
                        0,
                        3,
                        value=_DEFAULT.presence_penalty,
                        step=0.1,
                        label="Presence Penalty",
                    )
                c_stream = gr.Checkbox(
                    label="Stream output",
                    value=True,
                    info="Show tokens as they arrive",
                )
                gr.Markdown("### Generation")
                c_mt = gr.Slider(
                    64,
                    8192,
                    value=2048,
                    step=64,
                    label="Max Tokens",
                )
                c_ctx = gr.Dropdown(
                    label="Context Limit",
                    choices=[
                        "8192",
                        "16384",
                        "32768",
                        "65536",
                        "131072",
                        "262144",
                    ],
                    value="65536",
                    allow_custom_value=True,
                )
                gr.Markdown("### Server")
                c_url = gr.Textbox(
                    label="Server URL",
                    value=default_cfg.url,
                )
                c_tout = gr.Slider(
                    30,
                    1200,
                    value=default_cfg.timeout,
                    step=30,
                    label="Timeout (s)",
                )
                c_status = gr.HTML(value=not_checked)
                c_chk = gr.Button(
                    "Check Server",
                    size="sm",
                )

        # ── Events ──

        async def _on_upload(fp, fps, mt, ctx, prompt, res_name):
            if fp is None:
                return (
                    None,
                    empty_budget,
                    "",
                    {},
                    gr.update(),
                    gr.update(),
                    "",
                )
            try:
                mode = detect_mode(fp)
            except ValueError as e:
                raise gr.Error(str(e)) from e

            ctx_i = int(ctx)
            is_vid = mode == "video"
            rs = _RESOLUTION_MAP.get(res_name, 1.0)

            if is_vid:
                info = await _probe_video(fp)
                raw_dur = float(info.get("duration", 0))
                # Snap up to next 0.5s step so the slider covers the full video
                vid_dur = math.ceil(raw_dur * 2) / 2  # e.g. 5.098 → 5.5
                if vid_dur > 0:
                    fc = _dur_to_frames(vid_dur, fps)
                    dur_update = gr.update(
                        maximum=vid_dur,
                        value=vid_dur,
                        interactive=True,
                    )
                    txt = f"Video: {info['width']}\u00d7{info['height']}, {vid_dur:.1f}s"
                else:
                    # Duration unknown — don't cap the slider
                    vid_dur = 30.0
                    fc = _dur_to_frames(vid_dur, fps)
                    dur_update = gr.update(interactive=True)
                    txt = f"Video: {info['width']}\u00d7{info['height']}"
                gal = await _extract_preview(fp, fps, fc)
                bgt = compute_budget(
                    info,
                    fps,
                    vid_dur,
                    prompt,
                    int(mt),
                    ctx_i,
                    rs,
                )
                fc_txt = f"{fc} frames @ {fps} fps"
            else:
                img = load_image(fp)
                ih, iw = img.shape[:2]
                info = {
                    "width": iw,
                    "height": ih,
                    "duration": 0,
                    "mode": "image",
                }
                gal = [(img, f"Image ({iw}\u00d7{ih})")]
                bgt = compute_budget(
                    info,
                    1,
                    1,
                    prompt,
                    int(mt),
                    ctx_i,
                    rs,
                )
                txt = f"Image: {iw}\u00d7{ih}"
                dur_update = gr.update(interactive=False)
                fc_txt = ""

            return (
                gal,
                bgt,
                txt,
                info,
                gr.update(interactive=is_vid),
                dur_update,
                fc_txt,
            )

        c_file.change(
            fn=_on_upload,
            inputs=[
                c_file,
                c_fps,
                c_mt,
                c_ctx,
                c_prompt,
                c_res,
            ],
            outputs=[
                c_gallery,
                c_budget,
                c_mode,
                vi_state,
                c_fps,
                c_dur,
                c_fc,
            ],
        )

        # Live budget + frame count updates
        def _bgt(vi, fps, dur, prompt, mt, ctx, res_name):
            rs = _RESOLUTION_MAP.get(res_name, 1.0)
            budget = compute_budget(vi, fps, dur, prompt, int(mt), int(ctx), rs)
            fc = _dur_to_frames(dur, fps) if vi.get("mode") == "video" else 0
            fc_txt = f"{fc} frames @ {fps} fps" if fc else ""
            return budget, fc_txt

        bgt_in = [
            vi_state,
            c_fps,
            c_dur,
            c_prompt,
            c_mt,
            c_ctx,
            c_res,
        ]
        for _c in [c_fps, c_dur, c_mt, c_ctx, c_res]:
            _c.change(
                fn=_bgt,
                inputs=bgt_in,
                outputs=[c_budget, c_fc],
            )

        # Re-extract preview on video settings change
        async def _prev(fp, fps, dur):
            if not fp:
                return None
            try:
                fc = _dur_to_frames(dur, fps)
                return await _extract_preview(fp, fps, fc)
            except Exception:
                return None

        for _c in [c_fps, c_dur]:
            _c.change(
                fn=_prev,
                inputs=[c_file, c_fps, c_dur],
                outputs=[c_gallery],
            )

        c_tmpl.change(
            fn=_on_template_change,
            inputs=[c_tmpl, c_file],
            outputs=[c_prompt],
        )
        c_preset.change(
            fn=_on_preset_change,
            inputs=[c_preset],
            outputs=[c_temp, c_tp, c_tk, c_mp, c_pp],
        )

        # Caption button (async generator for streaming support)
        async def _caption(
            fp,
            prompt,
            fps,
            dur,
            mt,
            temp,
            tp,
            tk,
            mp,
            pp,
            url,
            tout,
            res_name,
            do_stream,
            prev_visual,
            profile,
        ):
            if not fp:
                raise gr.Error("Upload a file first")
            mode = detect_mode(fp)
            rs = _RESOLUTION_MAP.get(res_name, 1.0)
            mf = _dur_to_frames(dur, fps)

            # Decide cache_prompt: allow cache only when visual input
            # is identical to last request (i.e. only sampler changes).
            # First run (prev_visual is None) always invalidates.
            cur_visual = (fp, fps, dur, rs)
            use_cache = prev_visual is not None and cur_visual == prev_visual
            logger.info(
                "Caption: fps=%.1f, dur=%.1f, res=%s (scale=%.3f), "
                "frames=%d, max_tokens=%s, stream=%s, cache=%s",
                fps,
                dur,
                res_name,
                rs,
                mf,
                mt,
                do_stream,
                use_cache,
            )

            def _warn_html(truncated: bool) -> dict[str, Any]:
                if truncated:
                    return gr.update(
                        visible=True,
                        value=(
                            '<div style="background:#442200;border:1px solid #664400;'
                            'border-radius:4px;padding:8px;margin:4px 0;color:#ffcc66">'
                            "<b>Truncated:</b> The model used all tokens on reasoning "
                            "and never produced a caption. Increase <b>Max Tokens</b> "
                            "and try again.</div>"
                        ),
                    )
                return gr.update(visible=False)

            if not do_stream:
                try:
                    cap, thinking, meta, truncated, res = await _do_caption(
                        fp,
                        mode,
                        prompt,
                        fps,
                        mf,
                        int(mt),
                        temp,
                        tp,
                        int(tk),
                        mp,
                        pp,
                        url,
                        tout,
                        rs,
                        use_cache,
                        profile,
                    )
                except LlamaVideoError as e:
                    raise gr.Error(str(e)) from e
                except Exception as e:
                    raise gr.Error(f"Error: {e}") from e
                _save_result(res)
                yield cap, thinking, meta, _warn_html(truncated), cur_visual
                return

            # ── Streaming path ──
            cfg = ServerConfig(url=url, timeout=tout)
            client = LlamaServerClient(cfg)
            adapter_preset = AdapterPreset(
                temperature=temp,
                top_p=tp,
                top_k=int(tk),
                min_p=mp,
                presence_penalty=pp,
            )
            # Resolve adapter for streaming too
            if profile == "auto":
                stream_adapter_name = await _detect_adapter(url)
            else:
                stream_adapter_name = profile
            stream_adapter = get_adapter(stream_adapter_name)
            start = time.monotonic()

            try:
                if mode == "video":
                    ext = Extractor()
                    frames = await ext.extract_frames_async(
                        fp,
                        ExtractorConfig(fps=fps, max_frames=mf),
                    )
                    vi = stream_adapter.preprocess(frames, fps=fps, resolution_scale=rs)
                    payload = stream_adapter.build_payload(
                        vi,
                        prompt=prompt,
                        max_tokens=int(mt),
                        preset=adapter_preset,
                        model_name=cfg.model_name,
                        cache_prompt=use_cache,
                    )

                    accumulated_text = ""
                    final_caption = ""
                    final_thinking = ""
                    async for token, is_reasoning in client.stream_completion(payload):
                        accumulated_text += token
                        if is_reasoning:
                            final_thinking += token
                        else:
                            final_caption += token
                        yield final_caption, final_thinking, "", gr.update(visible=False), cur_visual

                    # Streaming already splits reasoning/content via is_reasoning flag.
                    # No need to re-parse through adapter.

                    ms = (time.monotonic() - start) * 1000
                    meta = build_metadata_html(
                        "video",
                        len(frames),
                        len(vi.super_frames),
                        vi.grid_thw,
                        vi.resolution,
                        ms,
                    )
                    res = CaptionResult(
                        caption=final_caption,
                        source_path=fp,
                        mode="video",
                        prompt_rendered=prompt,
                        template_name=None,
                        variables={},
                        preset_name="webui",
                        settings={"fps": fps, "max_frames": mf, "max_tokens": int(mt)},
                        metadata=CaptionMetadata(
                            frames_extracted=len(frames),
                            super_frames=len(vi.super_frames),
                            grid_thw=vi.grid_thw,
                            processing_time_ms=ms,
                        ),
                        token_usage=None,
                        duration_ms=ms,
                    )
                else:
                    image_msg = build_image_message(fp, prompt)
                    payload: dict[str, Any] = {
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a media captioning assistant. Your reasoning is "
                                    "private and will not be shown to the user. Your response "
                                    "must contain the complete, detailed caption."
                                ),
                            },
                            image_msg,
                        ],
                        "max_tokens": int(mt),
                        "temperature": adapter_preset.temperature,
                        "top_p": adapter_preset.top_p,
                        "top_k": adapter_preset.top_k,
                        "min_p": adapter_preset.min_p,
                        "presence_penalty": adapter_preset.presence_penalty,
                        "cache_prompt": use_cache,
                    }
                    if cfg.model_name:
                        payload["model"] = cfg.model_name

                    accumulated_text = ""
                    final_caption = ""
                    final_thinking = ""
                    async for token, is_reasoning in client.stream_completion(payload):
                        accumulated_text += token
                        if is_reasoning:
                            final_thinking += token
                        else:
                            final_caption += token
                        yield final_caption, final_thinking, "", gr.update(visible=False), cur_visual

                    # Streaming already splits reasoning/content via is_reasoning flag.

                    ms = (time.monotonic() - start) * 1000
                    img = load_image(fp)
                    ih, iw = img.shape[:2]
                    gu = 28
                    tw = max(gu, round(iw / gu) * gu)
                    th = max(gu, round(ih / gu) * gu)
                    hg, wg = th // gu, tw // gu
                    meta = build_metadata_html(
                        "image",
                        1,
                        0,
                        (1, hg, wg),
                        (tw, th),
                        ms,
                    )
                    res = CaptionResult(
                        caption=final_caption,
                        source_path=fp,
                        mode="image",
                        prompt_rendered=prompt,
                        template_name=None,
                        variables={},
                        preset_name="webui",
                        settings={"max_tokens": int(mt)},
                        metadata=CaptionMetadata(
                            frames_extracted=1,
                            super_frames=0,
                            grid_thw=(1, hg, wg),
                            processing_time_ms=ms,
                        ),
                        token_usage=None,
                        duration_ms=ms,
                    )

                _save_result(res)
                yield (
                    final_caption,
                    final_thinking,
                    meta,
                    _warn_html(not final_caption and bool(final_thinking)),
                    cur_visual,
                )

            except LlamaVideoError as e:
                raise gr.Error(str(e)) from e
            except Exception as e:
                raise gr.Error(f"Error: {e}") from e
            finally:
                await client.close()

        caption_event = c_btn.click(
            fn=_caption,
            inputs=[
                c_file,
                c_prompt,
                c_fps,
                c_dur,
                c_mt,
                c_temp,
                c_tp,
                c_tk,
                c_mp,
                c_pp,
                c_url,
                c_tout,
                c_res,
                c_stream,
                last_visual,
                c_profile,
            ],
            outputs=[c_out, c_think, c_meta, c_warn, last_visual],
        )
        c_cancel.click(fn=None, cancels=[caption_event])

        c_chk.click(
            fn=_check_health,
            inputs=[c_url, c_tout],
            outputs=[c_status],
        )

    return {"url": c_url, "tout": c_tout, "status": c_status}


def _build_compare_tab(empty_budget: str, default_cfg: ServerConfig) -> None:
    """Build the side-by-side Compare tab."""
    with gr.TabItem("Compare"):
        cmp_info: gr.State = gr.State(value={})

        cmp_file = gr.File(
            label="Upload Video or Image",
            type="filepath",
            file_types=_FILE_TYPES,
        )
        cmp_mode = gr.Textbox(
            label="Detected",
            interactive=False,
            max_lines=1,
        )
        cmp_profile = gr.Dropdown(
            label="Model Profile",
            choices=["auto", *list_adapters()],
            value="auto",
        )

        with gr.Row(equal_height=False):
            a = _compare_column(
                "A",
                "general",
                "Describe what happens in this video.",
                empty_budget,
            )
            b = _compare_column(
                "B",
                "detailed",
                "Describe this video in detail, including "
                "characters, setting, actions, and "
                "atmosphere.",
                empty_budget,
            )

        with gr.Row():
            cmp_url = gr.Textbox(
                label="Server URL",
                value=default_cfg.url,
                scale=3,
            )
            cmp_tout = gr.Slider(
                30,
                1200,
                value=default_cfg.timeout,
                step=30,
                label="Timeout",
                scale=1,
            )
        run_both = gr.Button(
            "Run Both",
            variant="primary",
            size="lg",
        )

        # ── Compare events ──

        async def _cmp_upload(
            fp,
            af,
            adur,
            ap,
            amt,
            bf,
            bdur,
            bp,
            bmt,
        ):
            if fp is None:
                return "", {}, empty_budget, empty_budget, gr.update(), gr.update()
            mode = detect_mode(fp)
            if mode == "video":
                info = await _probe_video(fp)
                vid_dur = float(info.get("duration", 0)) or 30.0
                txt = f"Video: {info['width']}\u00d7{info['height']}, {vid_dur:.1f}s"
                dur_upd = gr.update(maximum=vid_dur, value=vid_dur)
            else:
                img = load_image(fp)
                ih, iw = img.shape[:2]
                info = {
                    "width": iw,
                    "height": ih,
                    "duration": 0,
                    "mode": "image",
                }
                txt = f"Image: {iw}\u00d7{ih}"
                vid_dur = 1.0
                dur_upd = gr.update()

            ba = compute_budget(info, af, adur, ap, int(amt), 65536)
            bb = compute_budget(info, bf, bdur, bp, int(bmt), 65536)
            return txt, info, ba, bb, dur_upd, dur_upd

        cmp_file.change(
            fn=_cmp_upload,
            inputs=[
                cmp_file,
                a["fps"],
                a["dur"],
                a["prompt"],
                a["mt"],
                b["fps"],
                b["dur"],
                b["prompt"],
                b["mt"],
            ],
            outputs=[
                cmp_mode,
                cmp_info,
                a["budget"],
                b["budget"],
                a["dur"],
                b["dur"],
            ],
        )

        # Budget updates on slider changes
        def _cmp_bgt(vi, f, dur, p, mt):
            return compute_budget(vi, f, dur, p, int(mt), 65536)

        for side in [a, b]:
            for _c in [side["fps"], side["dur"], side["mt"]]:
                _c.change(
                    fn=_cmp_bgt,
                    inputs=[
                        cmp_info,
                        side["fps"],
                        side["dur"],
                        side["prompt"],
                        side["mt"],
                    ],
                    outputs=[side["budget"]],
                )

        # Template and preset changes
        for side in [a, b]:
            side["tmpl"].change(
                fn=_on_template_change,
                inputs=[side["tmpl"], cmp_file],
                outputs=[side["prompt"]],
            )
            side["preset"].change(
                fn=_on_preset_change,
                inputs=[side["preset"]],
                outputs=[
                    side["temp"],
                    side["tp"],
                    side["tk"],
                    side["mp"],
                    side["pp"],
                ],
            )

        # Run handler (shared by Run A, Run B, Run Both)
        async def _run(
            fp,
            pr,
            fps,
            dur,
            mt,
            te,
            tp,
            tk,
            mp,
            pp,
            url,
            tout,
            res_name,
            profile,
        ):
            if not fp:
                raise gr.Error("Upload a file first")
            mode = detect_mode(fp)
            rs = _RESOLUTION_MAP.get(res_name, 1.0)
            mf = _dur_to_frames(dur, fps)
            try:
                cap, _thinking, meta, _truncated, res = await _do_caption(
                    fp,
                    mode,
                    pr,
                    fps,
                    mf,
                    int(mt),
                    te,
                    tp,
                    int(tk),
                    mp,
                    pp,
                    url,
                    tout,
                    rs,
                    True,
                    profile,
                )
            except Exception as e:
                raise gr.Error(str(e)) from e
            _save_result(res)
            return cap, meta

        def _side_inputs(
            side: dict[str, Any],
        ) -> list[Any]:
            return [
                cmp_file,
                side["prompt"],
                side["fps"],
                side["dur"],
                side["mt"],
                side["temp"],
                side["tp"],
                side["tk"],
                side["mp"],
                side["pp"],
                cmp_url,
                cmp_tout,
                side["res"],
                cmp_profile,
            ]

        def _side_outputs(
            side: dict[str, Any],
        ) -> list[Any]:
            return [side["out"], side["meta"]]

        a["run"].click(
            fn=_run,
            inputs=_side_inputs(a),
            outputs=_side_outputs(a),
        )
        b["run"].click(
            fn=_run,
            inputs=_side_inputs(b),
            outputs=_side_outputs(b),
        )

        # Run Both = chain A then B
        run_both.click(
            fn=_run,
            inputs=_side_inputs(a),
            outputs=_side_outputs(a),
        ).then(
            fn=_run,
            inputs=_side_inputs(b),
            outputs=_side_outputs(b),
        )


def _compare_column(
    label: str,
    default_tmpl: str,
    default_prompt: str,
    empty_budget: str,
) -> dict[str, Any]:
    """Create one side of the Compare tab. Returns component dict."""
    with gr.Column():
        gr.Markdown(f"### Side {label}")
        tmpl = gr.Dropdown(
            label="Template",
            choices=_TEMPLATE_NAMES,
            value=default_tmpl,
        )
        prompt = gr.Textbox(
            label="Prompt",
            lines=2,
            value=default_prompt,
        )
        with gr.Row():
            fps = gr.Slider(
                0.5,
                10,
                value=2.0,
                step=0.5,
                label="FPS",
            )
            dur = gr.Slider(
                0.5,
                300,
                value=30,
                step=0.5,
                label="Duration (s)",
            )
        res = gr.Dropdown(
            label="Resolution",
            choices=_RESOLUTION_NAMES,
            value="Full",
        )
        preset = gr.Dropdown(
            label="Preset",
            choices=_PRESET_NAMES,
            value="default",
        )
        temp = gr.Slider(
            0,
            2,
            value=1.0,
            step=0.05,
            label="Temperature",
        )
        with gr.Accordion("Advanced", open=False):
            tp = gr.Slider(
                0,
                1,
                value=0.95,
                step=0.01,
                label="Top P",
            )
            tk = gr.Slider(
                1,
                100,
                value=20,
                step=1,
                label="Top K",
            )
            mp = gr.Slider(
                0,
                1,
                value=0.0,
                step=0.01,
                label="Min P",
            )
            pp = gr.Slider(
                0,
                3,
                value=1.5,
                step=0.1,
                label="Presence Pen.",
            )
        mt = gr.Slider(
            64,
            8192,
            value=2048,
            step=64,
            label="Max Tokens",
        )
        budget = gr.HTML(value=empty_budget)
        run = gr.Button(
            f"Run {label}",
            variant="secondary",
        )
        out = gr.Textbox(
            label=f"Caption {label}",
            lines=8,
            interactive=False,
            buttons=["copy"],
        )
        meta = gr.HTML()

    return {
        "tmpl": tmpl,
        "prompt": prompt,
        "fps": fps,
        "dur": dur,
        "res": res,
        "preset": preset,
        "temp": temp,
        "tp": tp,
        "tk": tk,
        "mp": mp,
        "pp": pp,
        "mt": mt,
        "budget": budget,
        "run": run,
        "out": out,
        "meta": meta,
    }


def _build_batch_tab(default_cfg: ServerConfig | None = None) -> None:
    """Build the Batch processing tab."""
    with gr.TabItem("Batch"):
        with gr.Row():
            with gr.Column(scale=2):
                ba_files = gr.File(
                    label="Upload Files",
                    type="filepath",
                    file_count="multiple",
                    file_types=_FILE_TYPES,
                )
                ba_prompt = gr.Textbox(
                    label="Prompt",
                    lines=2,
                    value="Describe what happens in this video.",
                )
                ba_run = gr.Button(
                    "Run Batch",
                    variant="primary",
                    size="lg",
                )
                ba_results = gr.Dataframe(
                    headers=[
                        "File",
                        "Mode",
                        "Caption",
                        "Time",
                    ],
                    datatype=[
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    label="Results",
                )

            with gr.Column(scale=1, min_width=250):
                ba_fps = gr.Slider(
                    0.5,
                    10,
                    value=2.0,
                    step=0.5,
                    label="FPS",
                )
                ba_dur = gr.Slider(
                    0.5,
                    300,
                    value=30,
                    step=0.5,
                    label="Duration (s)",
                )
                ba_res = gr.Dropdown(
                    label="Resolution",
                    choices=_RESOLUTION_NAMES,
                    value="Full",
                )
                ba_preset = gr.Dropdown(
                    label="Preset",
                    choices=_PRESET_NAMES,
                    value="default",
                )
                ba_temp = gr.Slider(
                    0,
                    2,
                    value=1.0,
                    step=0.05,
                    label="Temperature",
                )
                ba_mt = gr.Slider(
                    64,
                    8192,
                    value=2048,
                    step=64,
                    label="Max Tokens",
                )
                _bc = default_cfg or ServerConfig()
                ba_url = gr.Textbox(
                    label="Server URL",
                    value=_bc.url,
                )
                ba_tout = gr.Slider(
                    30,
                    1200,
                    value=_bc.timeout,
                    step=30,
                    label="Timeout",
                )
                ba_profile = gr.Dropdown(
                    label="Model Profile",
                    choices=["auto", *list_adapters()],
                    value="auto",
                )

        async def _batch(
            files,
            prompt,
            fps,
            dur,
            mt,
            temp,
            url,
            tout,
            preset_name,
            res_name,
            profile,
        ):
            if not files:
                raise gr.Error("No files uploaded")
            p = get_preset(preset_name)
            rs = _RESOLUTION_MAP.get(res_name, 1.0)
            mf = _dur_to_frames(dur, fps)
            rows: list[list[str]] = []
            for fp in files:
                name = Path(fp).name
                try:
                    mode = detect_mode(fp)
                    cap, _, _, _, res = await _do_caption(
                        fp,
                        mode,
                        prompt,
                        fps,
                        mf,
                        int(mt),
                        temp,
                        p.top_p,
                        p.top_k,
                        p.min_p,
                        p.presence_penalty,
                        url,
                        tout,
                        rs,
                        True,
                        profile,
                    )
                    _save_result(res)
                    t = f"{res.duration_ms / 1000:.1f}s"
                    rows.append(
                        [
                            name,
                            mode,
                            cap[:200],
                            t,
                        ]
                    )
                except Exception as e:
                    rows.append(
                        [
                            name,
                            "error",
                            str(e)[:200],
                            "",
                        ]
                    )
            return rows

        ba_run.click(
            fn=_batch,
            inputs=[
                ba_files,
                ba_prompt,
                ba_fps,
                ba_dur,
                ba_mt,
                ba_temp,
                ba_url,
                ba_tout,
                ba_preset,
                ba_res,
                ba_profile,
            ],
            outputs=[ba_results],
        )


def _build_history_tab() -> None:
    """Build the caption History tab."""
    with gr.TabItem("History"):
        with gr.Row():
            h_mode = gr.Dropdown(
                label="Mode",
                choices=["all", "video", "image"],
                value="all",
            )
            h_tmpl = gr.Dropdown(
                label="Template",
                choices=[
                    "all",
                    *list(BUILT_IN_TEMPLATES.keys()),
                ],
                value="all",
            )
            h_ref = gr.Button("Refresh", size="sm")

        h_table = gr.Dataframe(
            headers=[
                "ID",
                "Date",
                "Mode",
                "File",
                "Prompt",
                "Caption",
                "Time",
            ],
            datatype=[
                "number",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
            ],
            label="Caption History",
            interactive=False,
        )

        with gr.Row():
            h_json = gr.Button(
                "Export JSON",
                size="sm",
            )
            h_csv = gr.Button(
                "Export CSV",
                size="sm",
            )
            h_scrub = gr.Button(
                "Clear All",
                size="sm",
                variant="stop",
            )

        h_dl = gr.File(
            label="Download Export",
            visible=False,
        )
        h_msg = gr.Textbox(
            label="Status",
            interactive=False,
            visible=False,
        )

        def _load(mode_f, tmpl_f):
            hist = CaptionHistory()
            try:
                rows = hist.list_captions(
                    mode=(mode_f if mode_f != "all" else None),
                    template=(tmpl_f if tmpl_f != "all" else None),
                    limit=100,
                )
                data: list[list[Any]] = []
                for r in rows:
                    cap = r["caption"]
                    prm = r["prompt"]
                    dm = r["duration_ms"]
                    data.append(
                        [
                            r["id"],
                            r["created_at"][:19],
                            r["mode"],
                            Path(r["source_path"]).name,
                            (prm[:60] + "..." if len(prm) > 60 else prm),
                            (cap[:120] + "..." if len(cap) > 120 else cap),
                            (f"{dm / 1000:.1f}s" if dm else ""),
                        ]
                    )
                return data
            finally:
                hist.close()

        def _export(fmt: str) -> Any:
            hist = CaptionHistory()
            try:
                txt = hist.export(format=fmt)
                ext = "json" if fmt == "json" else "csv"
                p = Path(tempfile.gettempdir()) / f"llama-video-export.{ext}"
                p.write_text(txt)
                return gr.update(
                    visible=True,
                    value=str(p),
                )
            finally:
                hist.close()

        def _scrub():
            hist = CaptionHistory()
            try:
                n = hist.scrub()
                return (
                    [],
                    gr.update(
                        visible=True,
                        value=f"Deleted {n} entries",
                    ),
                )
            finally:
                hist.close()

        h_ref.click(
            fn=_load,
            inputs=[h_mode, h_tmpl],
            outputs=[h_table],
        )
        h_mode.change(
            fn=_load,
            inputs=[h_mode, h_tmpl],
            outputs=[h_table],
        )
        h_tmpl.change(
            fn=_load,
            inputs=[h_mode, h_tmpl],
            outputs=[h_table],
        )
        h_json.click(
            fn=lambda: _export("json"),
            outputs=[h_dl],
        )
        h_csv.click(
            fn=lambda: _export("csv"),
            outputs=[h_dl],
        )
        h_scrub.click(
            fn=_scrub,
            outputs=[h_table, h_msg],
        )


def main() -> None:
    """Entry point for llama-video-ui."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
