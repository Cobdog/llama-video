# API Server Subsystem

> **Code:** `src/llama_video/server.py`, `src/llama_video/client.py`
> **Tests:** `tests/unit/test_client.py`, `tests/integration/test_server.py`
> **Last verified:** 2026-03-23

## Purpose

FastAPI service that accepts video files/paths and returns captions. Wraps the full pipeline: extraction → preprocessing → llama-server inference.

## Endpoints

### `POST /v1/caption`
Caption a single video clip.

**Request:**
```json
{
    "video_path": "/path/to/clip.mp4",
    "prompt": "Describe what happens in this video.",
    "fps": 2.0,
    "max_frames": 64,
    "model_config": "qwen3.5"
}
```

**Response:**
```json
{
    "caption": "A person walks across a park and sits on a bench.",
    "metadata": {
        "frames_extracted": 8,
        "super_frames": 4,
        "grid_thw": [4, 16, 16],
        "processing_time_ms": 2340,
        "model_tokens_used": 1024
    }
}
```

### `POST /v1/caption/batch`
Caption multiple clips in sequence.

### `GET /v1/health`
Health check — reports Python service status + llama-server connectivity.

### `GET /v1/debug/last-request`
Returns diagnostic info about the most recent caption request (for debugging).

## Client (llama-server HTTP client)

### `LlamaServerClient`
- `caption_video(video_input: VideoInput, prompt: str) → str`
- `caption_image(image: Image, prompt: str) → str`
- `health_check() → bool`

Communicates with patched llama-server via OpenAI-compatible `/chat/completions`.

### Request Format to llama-server
```json
{
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "video_frames",
                "video_frames": {
                    "frames": ["base64...", "base64...", ...],
                    "fps": 2.0,
                    "is_video": true
                }
            },
            {"type": "text", "text": "Describe this video."}
        ]
    }],
    "max_tokens": 2048,
    "temperature": 0.7
}
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `LLAMA_SERVER_URL` | `http://localhost:8080` | llama-server URL |
| `LLAMA_VIDEO_HOST` | `0.0.0.0` | Python service bind host |
| `LLAMA_VIDEO_PORT` | `9000` | Python service bind port |
| `LLAMA_VIDEO_TIMEOUT` | `120` | Request timeout (seconds) |
| `LLAMA_VIDEO_WORKERS` | `1` | Uvicorn worker count |
