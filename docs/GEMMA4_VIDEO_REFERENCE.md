# Gemma 4 Video Processing Reference

Source: <https://ai.google.dev/gemma/docs/capabilities/vision/video>

## Video Input Handling

Gemma 4 processes video as a sequence of **individual image frames at 1 FPS**. The processor handles extraction internally.

- **Input format:** MP4 video (URLs or local file paths accepted by `AutoProcessor`)
- **Frame extraction:** Done automatically by the processor — pass `{"type": "video", "video": "<path_or_url>"}` and it handles the rest
- **FPS:** Fixed at **1 frame per second** (not configurable)
- **Resolution:** Variable. The vision encoder supports arbitrary aspect ratios. Resolution is controlled indirectly via the token budget parameter
- **Max duration:** **60 seconds** = 60 frames maximum at 1 FPS

The 1 FPS constraint is a hard limit in the processor, not a suggestion. If you need finer temporal resolution (e.g., analyzing fast motion), you must pre-extract frames yourself and pass them as individual images rather than using the `video` content type.

## Payload Structure

Internally, the processor constructs a flat text prompt with interleaved special tokens and frame embeddings. The decoded prompt template looks like this:

```
<|turn>user
00:00 <|image> [frame_0_embedding] 00:00 <|image> [frame_0_embedding]
00:01 <|image> [frame_1_embedding] 00:01 <|image> [frame_1_embedding]
...
00:14 <|image> [frame_14_embedding] 00:14 <|image> [frame_14_embedding]
Describe this video.
<|turn>model
```

- Each frame produces **two `<|image>` tokens** preceded by a **timestamp string** (`MM:SS` format)
- The timestamp appears twice per frame, each paired with one `<|image>` token — two patch streams from the vision encoder (e.g., global + local features, or initial + refined patches)
- Frame embeddings are laid out **sequentially in temporal order** within a single user turn
- Text prompts follow after all frame tokens in the same user message
- The `<|turn>` special token marks role boundaries

## Differences from Superframe Approaches

| Aspect | Gemma 4 | Superframe (Qwen) |
|--------|---------|-------------------|
| Frame grouping | **Individual frames**, each independent | Pairs (or groups) of consecutive frames merged into composite tokens |
| Temporal encoding | Explicit **timestamp strings** per frame | Implicit temporal relationships from spatial proximity in the composite |
| Frame processing | Each frame processed **independently** through the vision encoder | Paired frames processed jointly, sharing attention |
| Token structure | `timestamp <|image> timestamp <|image>` per frame | Single composite token per frame pair |
| Temporal resolution | 1 FPS, fixed | Often 2 FPS or variable, depends on pair stride |

Key takeaway for adapter design: you do **not** need to implement any frame pairing, interleaving, or composite token generation. Each frame is a standalone image with a prepended timestamp. Temporal understanding relies on the LLM backbone seeing the ordered sequence, not on any vision-level temporal fusion.

## Model-Specific Parameters

### Token budgets (resolution control)

The primary tuning knob is the **image token budget**, which controls how many tokens each frame's embedding consumes:

| Budget | Tokens per frame | Use case |
|--------|-----------------|----------|
| 70 | 70 | Thumbnails, low-detail |
| 140 | 140 | Moderate detail |
| **280** | **280** | **Default** |
| 560 | 560 | High detail |
| 1120 | 1120 | Maximum detail |

Mechanism: the vision encoder generates **9x the budget** in initial patches, then compresses via **3x3 grid averaging** into the final `budget` tokens. For example, at the default 280 tokens, the encoder produces 2,520 initial patches (280 * 9), which are averaged in 3x3 grids down to 280 final embeddings.

**Processor kwargs:**

- `image_min_tokens` — floor for token budget
- `image_max_tokens` — ceiling for token budget (primary control)

**llama.cpp equivalents:**

- `--image-min-tokens`
- `--image-max-tokens`

### Context window and token accounting

- **Max context window:** 256K tokens
- **Token cost per video (default 280 tokens/frame, 1 FPS):**
  - 60 frames * 280 tokens/frame = **16,800 tokens** for the video alone
  - At budget 1120: 60 * 1120 = **67,200 tokens**
  - Leaves substantial room for text context even at max budget

### Model loading (HuggingFace)

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E2B-it"
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

Note the model class: `AutoModelForMultimodalLM`, **not** `AutoModelForImageTextToText`.

### Available model variants

| Model | Type | Notes |
|-------|------|-------|
| E2B-it | Dense | Smallest, good for testing |
| E4B-it | Dense | Mid-size |
| 31B-it | Dense | Full-size |
| 26B-A4B-it | MoE (4B active) | Sparse, good efficiency |

## API / Prompt Format

The chat message structure uses the standard HuggingFace `apply_chat_template` pattern:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://example.com/video.mp4"},
            {"type": "text", "text": "Describe this video."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs, skip_special_tokens=False)
```

- The `content` field is a **list** of typed content blocks, not a single string
- Video is specified as `{"type": "video", "video": "<source>"}` — the `video` key holds the path/URL
- Multiple content blocks can be mixed (video + text, multiple videos if within token limits)
- The processor handles tokenization, image processing, and template application in one call

### Multi-turn conversation

For follow-up turns, the video frames only appear in the first user message. Subsequent turns are pure text:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "video.mp4"},
            {"type": "text", "text": "Describe this video."}
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "The video shows..."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens at 0:30?"}
        ]
    }
]
```

## Supported Video Lengths and Frame Limits

| Parameter | Value |
|-----------|-------|
| Max video duration | **60 seconds** |
| Frame rate | **1 FPS** (fixed) |
| Max frames | **60** |
| Min video duration | No documented minimum (single frame works) |
| Max token cost (60s at budget 1120) | 67,200 tokens |
| Min token cost (1s at budget 70) | 70 tokens |

### Practical constraints for adapter design

1. **Videos longer than 60 seconds must be segmented.** The adapter needs to split longer videos into <= 60s chunks and either process each independently or summarize across chunks.
2. **The 1 FPS rate means temporal detail is limited.** For a 60-second video, you get 60 frames. Sub-second events will be missed.
3. **Token budget trades off against context.** At default 280 tokens/frame, a 60s video uses ~16.8K tokens out of 256K — plenty of room. At max budget (1120), a 60s video uses ~67.2K tokens, still viable but constrains available text context.
4. **No variable frame rate or keyframe selection.** The processor does uniform sampling at 1 FPS. If you need scene-aware frame selection, you must extract frames yourself and pass them as individual images.

## Sampling Parameters

Standardized sampling configuration across all use cases:

```
temperature=1.0
top_p=0.95
top_k=64
```

These are **very different from Qwen3.5 defaults** (temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5). The adapter must use its own isolated sampler settings.

## Thinking Mode

Gemma 4 uses a different thinking mechanism than Qwen3.5. The model uses standard system/assistant/user roles with control tokens:

- **Enable thinking:** Include `<|think|>` token at the start of the system prompt
- **Thinking output:** `<|channel>thought\n[reasoning]<channel|>`
- **E2B/E4B behavior:** When thinking is disabled, these models still generate empty thought tags: `<|channel>thought\n<channel|>[answer]`
- **31B+ behavior:** When thinking is disabled, output is just the answer without thought tags
- **Multi-turn:** Previous thinking content must NOT be included in history — only the final response

Many libraries (Transformers, llama.cpp) handle the chat template complexities automatically.

## Modality Order

For optimal performance with multimodal inputs, **place image and/or audio content before text** in the prompt. This is the opposite of some other models where text can come first.

## Adapter Design Summary

1. Accept video input (path or URL)
2. Extract frames at 1 FPS (max 60 frames per chunk)
3. Construct chat messages with frames as individual `image_url` entries, timestamp text between frames
4. Use Gemma4-specific sampler settings (temperature=1.0, top_p=0.95, top_k=64)
5. Handle thinking mode parsing with `<|channel>thought` tags
6. For videos >60s, segment into chunks and process independently

Main design decisions:

- **Token budget selection** — configurable, default 280 (lower for video/captioning, higher for OCR/text)
- **Long video segmentation** — automatic chunking at user-specified length (up to 60s), with optional cohesion pass between segments
- **Isolated sampler settings** — completely separate from Qwen3.5 defaults
- **Thinking mode** — parse `<|channel>thought` tags differently from Qwen's `<think/>` tags
- **Prompt ordering** — images before text (model-specific requirement)

## Audio Support (E2B/E4B only)

The E2B and E4B variants support audio input in addition to video. Audio max length is 30 seconds. This is planned for a future phase after the multi-model adapter architecture is established.

### Audio prompt templates

**ASR:** `Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text. Follow these specific instructions for formatting the answer: * Only output the transcription, with no newlines. * When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.`

**AST:** `Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}. When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.`
