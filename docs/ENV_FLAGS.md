# Environment Flags (`ROOP_*`)

All runtime tunables read from environment variables. Every flag is **opt-in or
safe-by-default** — with none set, the app runs its validated default behavior.
Values are read at process start (some at module import), so set them **before**
launching and restart the app to change them.

Defaults below are what the code falls back to when the variable is unset.

> The Pinokio launcher already wires a few of these from the perf settings UI
> (`ROOP_TRT_POOL`, `ROOP_DETMASK_POOL`, `ROOP_ENCODER_PRESET`, `ROOP_PROFILE`,
> `ROOP_BATCH_SWAP`, `ROOP_NVDEC`) — see `app/run.py`.

---

## Throughput / GPU concurrency

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_TRT_POOL` | unset (1) | Pool of N independent TensorRT **swapper** contexts (N≥2) to break single-context serialization. Validated ~+46% video throughput at 2. |
| `ROOP_DETMASK_POOL` | unset (auto) | Pool of N independent detect/mask sessions (FaceAnalysis + mask engines). Set explicitly (e.g. 2–8) to parallelize detection/masking. |
| `ROOP_TRT_WORKSPACE_FRACTION` | ~auto | Fraction of VRAM TensorRT may use as build workspace. Lower if a build OOMs. |
| `ROOP_TRT_PARTITION_ITERATIONS` | 2000 | TensorRT partition search iterations during engine build. |
| `ROOP_BATCH_SWAP` | 0 | Batch multiple face crops through one swap inference call (bit-identical). Phase-1 pixel-boost batching. |
| `ROOP_BATCH_SWAP_XFRAME` | 0 | Cross-frame swap batching collector. Requires `ROOP_BATCH_SWAP=1`. |
| `ROOP_BATCH_SWAP_MAX` | = threads | Max batch size for batched swap. |

## Stabilization (enhancer flicker / kps smoothing)

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_STAB_PARALLEL` | 0 | Run the enhancer flicker stabilizer multi-threaded (windowed contiguous blocks + warm-up). Opt-in / experimental. |
| `ROOP_STAB_2PASS` | 1 (on) | Two-pass kps stabilization (precompute pass 1 + parallel lookup pass 2) when enhancer-stab is off. `0` disables. |
| `ROOP_STAB_WARMUP` | 4 | Warm-up frames per parallel stabilization block. |
| `ROOP_STAB_CHUNK` | auto | Frames per parallel stabilization chunk (auto = `max(threads*24, 192)`). |
| `ROOP_STAB_BLOCKS_PER_THREAD` | 1 | >1 enables work-stealing dispatch to fix idle-thread imbalance in parallel stabilization. |

## Encoding / decoding

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_NVDEC` | auto (on) | Hardware NVDEC video reader. `0` disables; `1` forces (skips the auto probe). |
| `ROOP_NVENC_PRESET` | `p5` | NVENC encoder preset (`p1`–`p7`). |
| `ROOP_ENCODER_PRESET` | model default | CPU x264/x265 preset (e.g. `faster`) — lossless encode speedup at fixed CRF. |
| `ROOP_RESUME` | 1 (on) | Crash-resume: write segments every chunk + manifest so an interrupted run continues. `0` disables. |
| `ROOP_RESUME_CHUNK` | 1000 | Frames per resume segment (min 50). |
| `ROOP_RESUME_KEEP` | 0 (off) | Keep the segment parts + manifest after a deliberate **Stop** so that run can be resumed later. Off = Stop merges the parts into one finished output and deletes them. Crash-resume is unaffected either way. |

## Quality / precision

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_SWAP_FP16` | 0 | `1` lets the swapper run FP16 under TRT (**not recommended** — causes rainbow/smudge from FP16 overflow; default forces FP32). |
| `ROOP_GPEN_FP16` | 0 | `1` lets GPEN run FP16 (not recommended; default FP32 for correct faces). |
| `ROOP_UPSCALE_TRT` | 0 | `1` runs ESRGAN x4 upscalers under TensorRT (**not recommended** — goes all-black under TRT FP16; default forces CUDA/CPU FP32). |
| `ROOP_UPSCALE_TILE` | 256 | Tile size (px) for AI upscalers; lower if VRAM is tight on heavy ×4 models. |
| `ROOP_CAS_STRENGTH` | 0.5 | Contrast-Adaptive Sharpening strength for the `fsr` classical upscaler (0 = plain Lanczos). |

## Masking convention overrides

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_OCCLUDER_RAW` | 0 | `1` skips the Face Occluder mask inversion (flip polarity if the mask is inverted). |
| `ROOP_XSEG3_RAW` | 0 | `1` skips the XSeg3 mask inversion. |

## AI-upscale second pass

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_UPSCALE_NVENC` | 1 (on) | NVENC encode for the post-swap AI upscale pass. `0` opts out. |
| `ROOP_UPSCALE_THREADS` | 4 | Worker threads for the upscale pass (heavy models auto-cap to 1). |
| `ROOP_UPSCALE_MAX_DIM` | 8192 | Max output dimension for GPU-encoded upscale output; raise for very large frames. |

## Detection / tracking

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_TEMPORAL_GAP` | 10 | Max consecutive detection misses gap-filled by the tracked detection pre-pass. |
| `ROOP_TEMPORAL_STEP` | 1 | Scan stride for the "Analyzing faces" pre-pass. The pre-pass is detection-bound, so `2` roughly halves it — but skipped frames are filled by *linear* interpolation, which lags a fast head turn. Capped at `ROOP_TEMPORAL_GAP`. |
| `ROOP_TRACK_READAHEAD` | 1 | Decode frames for the pre-pass on their own thread so decoding overlaps detection (~15% off the pre-pass; bit-identical). `0` decodes inline as before. With it on, the `track_decode` stage times the wait *for* a frame, so it only grows if the decoder is genuinely the slower half. |
| `ROOP_TRACK_ROI_CROP` | 0 | `1` enables ROI-crop pre-pass during identity tracking. |

## Process priority

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_PRIORITY` | `high` | Windows process priority to dodge EcoQoS background throttling (`above_normal` \| `normal` to relax). |

## Ports

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_API_PORT` | 8001 | FastAPI/React backend port. |
| `ROOP_GRADIO_PORT` | CFG `server_port` | Legacy Gradio UI port. |

## Debug / diagnostics

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_PROFILE` | 0 | `1` prints per-stage wall-clock timing (analyze/detect/mask/swap/enhance/…) summed across worker threads. |
| `ROOP_DEBUG_MATCH` | unset | Prints identity-matching diagnostics during the swap pass. |
