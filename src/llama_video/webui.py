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
import time
from pathlib import Path
from typing import Any

import gradio as gr

from llama_video.batch import detect_mode
from llama_video.client import LlamaServerClient
from llama_video.config import (
    PRESETS,
    InferencePreset,
    ModelConfig,
    ServerConfig,
    get_preset,
)
from llama_video.errors import LlamaVideoError
from llama_video.extractor import Extractor, ExtractorConfig
from llama_video.history import CaptionHistory
from llama_video.image import load_image
from llama_video.preprocessor import Preprocessor
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


def compute_budget(
    info: dict[str, Any],
    fps: float,
    max_frames: int,
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
        dur = float(info.get("duration", 0))
        fc = min(max_frames, max(1, math.ceil(dur * fps))) if dur > 0 else max_frames
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
) -> tuple[str, str, CaptionResult]:
    """Full pipeline: extract, preprocess, infer, return."""
    cfg = ServerConfig(url=server_url, timeout=timeout)
    client = LlamaServerClient(cfg)
    preset = InferencePreset(
        name="webui",
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
                preset,
                client,
                start,
                resolution_scale,
            )
        return await _caption_image(
            path,
            max_tokens,
            prompt,
            preset,
            client,
            start,
        )
    finally:
        await client.close()


async def _caption_video(
    path: str,
    fps: float,
    max_frames: int,
    max_tokens: int,
    prompt: str,
    preset: InferencePreset,
    client: LlamaServerClient,
    start: float,
    resolution_scale: float = 1.0,
) -> tuple[str, str, CaptionResult]:
    """Video captioning sub-pipeline."""
    ext = Extractor()
    pre = Preprocessor()
    frames = await ext.extract_frames_async(
        path,
        ExtractorConfig(fps=fps, max_frames=max_frames),
    )
    vi = pre.process(frames, fps=fps, resolution_scale=resolution_scale)
    cap = await client.caption_video(
        vi,
        prompt=prompt,
        max_tokens=max_tokens,
        preset=preset,
    )
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
    return cap, meta, res


async def _caption_image(
    path: str,
    max_tokens: int,
    prompt: str,
    preset: InferencePreset,
    client: LlamaServerClient,
    start: float,
) -> tuple[str, str, CaptionResult]:
    """Image captioning sub-pipeline."""
    cap = await client.caption_image(
        path,
        prompt=prompt,
        max_tokens=max_tokens,
        preset=preset,
    )
    ms = (time.monotonic() - start) * 1000
    img = load_image(path)
    ih, iw = img.shape[:2]
    m = ModelConfig.qwen35()
    gu = m.grid_unit
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
    return cap, meta, res


async def _check_health(url: str, timeout: float) -> str:
    """Check llama-server health, return status HTML."""
    cfg = ServerConfig(url=url, timeout=min(timeout, 10))
    client = LlamaServerClient(cfg)
    try:
        ok = await client.health_check()
        if ok:
            return '<span style="color:#50c878;font-weight:bold">\u25cf Connected</span>'
        return '<span style="color:#ffaa00;font-weight:bold">\u25cf Degraded</span>'
    except Exception:
        return '<span style="color:#ff4444;font-weight:bold">\u25cf Unreachable</span>'
    finally:
        await client.close()


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
    not_checked = '<span style="color:#888">Not checked</span>'

    with gr.Blocks(
        title="llama-video",
    ) as app:
        gr.Markdown(
            "# llama-video\n"
            "Video & image captioning experimentation "
            "for Qwen3.5 via patched llama.cpp"
        )

        with gr.Tabs():
            _build_caption_tab(empty_budget, not_checked)
            _build_compare_tab(empty_budget)
            _build_batch_tab()
            _build_history_tab()

    return app


def _build_caption_tab(
    empty_budget: str,
    not_checked: str,
) -> None:
    """Build the main Caption tab with all controls."""
    with gr.TabItem("Caption"):
        vi_state: gr.State = gr.State(value={})

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
                c_btn = gr.Button(
                    "Caption",
                    variant="primary",
                    size="lg",
                )
                c_out = gr.Textbox(
                    label="Caption Output",
                    lines=10,
                    interactive=False,
                    buttons=["copy"],
                )
                with gr.Accordion("Metadata", open=False):
                    c_meta = gr.HTML()

            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Video")
                c_fps = gr.Slider(
                    0.5,
                    10,
                    value=2.0,
                    step=0.5,
                    label="FPS",
                )
                c_mf = gr.Slider(
                    2,
                    256,
                    value=64,
                    step=2,
                    label="Max Frames",
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
                    value="http://localhost:8080",
                )
                c_tout = gr.Slider(
                    30,
                    1200,
                    value=300,
                    step=30,
                    label="Timeout (s)",
                )
                c_status = gr.HTML(value=not_checked)
                c_chk = gr.Button(
                    "Check Server",
                    size="sm",
                )

        # ── Events ──

        async def _on_upload(fp, fps, mf, mt, ctx, prompt, res_name):
            if fp is None:
                return (
                    None,
                    empty_budget,
                    "",
                    {},
                    gr.update(),
                    gr.update(),
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
                gal = await _extract_preview(
                    fp,
                    fps,
                    int(mf),
                )
                bgt = compute_budget(
                    info,
                    fps,
                    int(mf),
                    prompt,
                    int(mt),
                    ctx_i,
                    rs,
                )
                n = len(gal)
                dur = info.get("duration", 0)
                txt = (
                    f"Video: {info['width']}"
                    f"\u00d7{info['height']}, "
                    f"{dur:.1f}s, {n} frames "
                    f"@ {fps} fps"
                )
            else:
                img = load_image(fp)
                ih, iw = img.shape[:2]
                info = {
                    "width": iw,
                    "height": ih,
                    "duration": 0,
                    "mode": "image",
                }
                gal = [
                    (img, f"Image ({iw}\u00d7{ih})"),
                ]
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

            return (
                gal,
                bgt,
                txt,
                info,
                gr.update(interactive=is_vid),
                gr.update(interactive=is_vid),
            )

        c_file.change(
            fn=_on_upload,
            inputs=[
                c_file,
                c_fps,
                c_mf,
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
                c_mf,
            ],
        )

        # Live budget updates
        def _bgt(vi, fps, mf, prompt, mt, ctx, res_name):
            rs = _RESOLUTION_MAP.get(res_name, 1.0)
            return compute_budget(
                vi,
                fps,
                int(mf),
                prompt,
                int(mt),
                int(ctx),
                rs,
            )

        bgt_in = [
            vi_state,
            c_fps,
            c_mf,
            c_prompt,
            c_mt,
            c_ctx,
            c_res,
        ]
        for _c in [c_fps, c_mf, c_mt, c_ctx, c_res]:
            _c.change(
                fn=_bgt,
                inputs=bgt_in,
                outputs=[c_budget],
            )

        # Re-extract preview on video settings change
        async def _prev(fp, fps, mf):
            if not fp:
                return None
            try:
                return await _extract_preview(
                    fp,
                    fps,
                    int(mf),
                )
            except Exception:
                return None

        for _c in [c_fps, c_mf]:
            _c.change(
                fn=_prev,
                inputs=[c_file, c_fps, c_mf],
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

        # Caption button
        async def _caption(
            fp,
            prompt,
            fps,
            mf,
            mt,
            temp,
            tp,
            tk,
            mp,
            pp,
            url,
            tout,
            res_name,
        ):
            if not fp:
                raise gr.Error("Upload a file first")
            mode = detect_mode(fp)
            rs = _RESOLUTION_MAP.get(res_name, 1.0)
            try:
                cap, meta, res = await _do_caption(
                    fp,
                    mode,
                    prompt,
                    fps,
                    int(mf),
                    int(mt),
                    temp,
                    tp,
                    int(tk),
                    mp,
                    pp,
                    url,
                    tout,
                    rs,
                )
            except LlamaVideoError as e:
                raise gr.Error(str(e)) from e
            except Exception as e:
                raise gr.Error(f"Error: {e}") from e
            _save_result(res)
            return cap, meta

        c_btn.click(
            fn=_caption,
            inputs=[
                c_file,
                c_prompt,
                c_fps,
                c_mf,
                c_mt,
                c_temp,
                c_tp,
                c_tk,
                c_mp,
                c_pp,
                c_url,
                c_tout,
                c_res,
            ],
            outputs=[c_out, c_meta],
        )

        c_chk.click(
            fn=_check_health,
            inputs=[c_url, c_tout],
            outputs=[c_status],
        )


def _build_compare_tab(empty_budget: str) -> None:
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
                value="http://localhost:8080",
                scale=3,
            )
            cmp_tout = gr.Slider(
                30,
                1200,
                value=300,
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
            amf,
            ap,
            amt,
            bf,
            bmf,
            bp,
            bmt,
        ):
            if fp is None:
                return "", {}, empty_budget, empty_budget
            mode = detect_mode(fp)
            if mode == "video":
                info = await _probe_video(fp)
                txt = f"Video: {info['width']}\u00d7{info['height']}, {info['duration']:.1f}s"
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

            ba = compute_budget(
                info,
                af,
                int(amf),
                ap,
                int(amt),
                65536,
            )
            bb = compute_budget(
                info,
                bf,
                int(bmf),
                bp,
                int(bmt),
                65536,
            )
            return txt, info, ba, bb

        cmp_file.change(
            fn=_cmp_upload,
            inputs=[
                cmp_file,
                a["fps"],
                a["mf"],
                a["prompt"],
                a["mt"],
                b["fps"],
                b["mf"],
                b["prompt"],
                b["mt"],
            ],
            outputs=[
                cmp_mode,
                cmp_info,
                a["budget"],
                b["budget"],
            ],
        )

        # Budget updates on slider changes
        def _cmp_bgt(vi, f, mf, p, mt):
            return compute_budget(
                vi,
                f,
                int(mf),
                p,
                int(mt),
                65536,
            )

        for side in [a, b]:
            for _c in [side["fps"], side["mf"], side["mt"]]:
                _c.change(
                    fn=_cmp_bgt,
                    inputs=[
                        cmp_info,
                        side["fps"],
                        side["mf"],
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
            mf,
            mt,
            te,
            tp,
            tk,
            mp,
            pp,
            url,
            tout,
            res_name,
        ):
            if not fp:
                raise gr.Error("Upload a file first")
            mode = detect_mode(fp)
            rs = _RESOLUTION_MAP.get(res_name, 1.0)
            try:
                cap, meta, res = await _do_caption(
                    fp,
                    mode,
                    pr,
                    fps,
                    int(mf),
                    int(mt),
                    te,
                    tp,
                    int(tk),
                    mp,
                    pp,
                    url,
                    tout,
                    rs,
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
                side["mf"],
                side["mt"],
                side["temp"],
                side["tp"],
                side["tk"],
                side["mp"],
                side["pp"],
                cmp_url,
                cmp_tout,
                side["res"],
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
            mf = gr.Slider(
                2,
                256,
                value=64,
                step=2,
                label="Max Frames",
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
        "mf": mf,
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


def _build_batch_tab() -> None:
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
                ba_mf = gr.Slider(
                    2,
                    256,
                    value=64,
                    step=2,
                    label="Max Frames",
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
                ba_url = gr.Textbox(
                    label="Server URL",
                    value="http://localhost:8080",
                )
                ba_tout = gr.Slider(
                    30,
                    1200,
                    value=300,
                    step=30,
                    label="Timeout",
                )

        async def _batch(
            files,
            prompt,
            fps,
            mf,
            mt,
            temp,
            url,
            tout,
            preset_name,
            res_name,
        ):
            if not files:
                raise gr.Error("No files uploaded")
            p = get_preset(preset_name)
            rs = _RESOLUTION_MAP.get(res_name, 1.0)
            rows: list[list[str]] = []
            for fp in files:
                name = Path(fp).name
                try:
                    mode = detect_mode(fp)
                    cap, _, res = await _do_caption(
                        fp,
                        mode,
                        prompt,
                        fps,
                        int(mf),
                        int(mt),
                        temp,
                        p.top_p,
                        p.top_k,
                        p.min_p,
                        p.presence_penalty,
                        url,
                        tout,
                        rs,
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
                ba_mf,
                ba_mt,
                ba_temp,
                ba_url,
                ba_tout,
                ba_preset,
                ba_res,
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
                p = Path(f"/tmp/llama-video-export.{ext}")
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
