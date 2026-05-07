# Roadmap

Living document. Items are listed roughly in the order they are likely to be tackled, but priorities shift as needs change. Each item is one sentence of intent, not a plan — when an item becomes "next," it gets its own implementation plan.

## In flight

*(nothing yet)*

## Next up

*(nothing yet)*

## Planned (queued)

- **Gemma4 audio support (E2B / E4B).** After the multi-model pipeline is working, extend the Gemma4 adapter to extract audio from video via ffmpeg and send it through the Gemma4 audio encoder. Requires new extraction path, new payload shape for audio, and confirmation that `ggml-org`'s Gemma4 GGUF mmproj includes the audio encoder.

- **Model-family autodetect for router mode.** When a llama-server is running in router mode (multiple models behind `/v1/models`), autodetect the active model family instead of requiring an explicit `model_profile` setting. Probes `/v1/models`, matches a name pattern, selects the adapter. Deferred because the explicit setting is fine for single-model servers — which is the current default and the common case.

## Ideas / unfiled

- Additional model families as they land in llama.cpp (InternVL video, MiniCPM-V, future Gemma / Qwen releases).
- Super-frame strategies beyond 2-frame pairing (e.g., 4-frame temporal patches if a model family adopts them).
- Dataset-scale batch captioning workflows (the `testvid/` Ghibli corpus is already a small benchmark — scale this to larger evaluation runs).

## Recently completed

- **Multi-model architecture (Qwen3.5 + Gemma4).** Extracted a `ModelAdapter` interface, converted Qwen3.5 into the first adapter, added a Gemma4 adapter, and wired model selection through the WebUI, HTTP service, batch API, and debug CLI. Proves the adapter boundary with one patched model and one stock-llama.cpp model.
