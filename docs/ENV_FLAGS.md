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
| `ROOP_DETECTOR_POOL` | = detmask pool | Independent instances of the standalone detector (retinaface/yunet/yoloface), which otherwise serialise behind a module mutex. Defaults to the detmask pool size so a worker never waits. `retinaface_r50.onnx` is ~104 MB per instance — turn this down before the detmask pool when VRAM is tight. |
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
| `ROOP_YAW_ALIGN` | `off` | Seeds the **Profile alignment (90° faces)** selector in the Face Swap tab (the selector overrides it per run; a saved setting wins once you touch it). `off` \| `stabilize` \| `pose` — `1`/`on`/`true` are accepted as legacy aliases for `stabilize`. Affects near-profile faces (~70°+ yaw) only; frontal and mid-angle faces are bit-identical in every mode. Not to be confused with `ROOP_PROFILE` (stage timing). See below. |

### Profile alignment modes

At high yaw the two eyes project to nearly the same point, so fitting the 5
keypoints to a **frontal** template is ill-conditioned. Measured against the
project's own 3-D reference head, on a 512 crop:

| | mean fit error @ 0° | @ 90° | crop rotation swing as the head nods ±25° @ 90° |
|---|---|---|---|
| `off` | 8 px | 60 px | ~30° |
| `stabilize` | 8 px | 60 px | **<0.5°** |
| `pose` | 8 px | **24 px** | **<0.5°** |

- **`stabilize`** keeps the frontal template but takes the rotation from the
  eye→mouth axis, so pitch stops leaking into in-plane roll. This is a
  **temporal** fix — it reduces wobble *between* frames and will look identical
  on a single still, so judge it on video.
- **`pose`** replaces the template with the reference head projected at the
  estimated yaw, making the fit well-posed (error drops ~60%, and to ~0 at zero
  pitch). Face placement barely moves (nose 295.5 → 300.0 px at 90° yaw), so it
  corrects distortion rather than re-framing.

⚠️ Lower fit error is **not** automatically a better swap. The swap models were
trained on crops aligned with the same fixed frontal template, so a
geometrically cleaner profile crop is also slightly off their training
distribution. `pose` is opt-in for that reason — A/B it on real footage before
adopting it.

## Masking convention overrides

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_NONFRONTAL_MASK` | `auto` | Which masking path a dense masker (XSeg/XSeg3/occluder/faceparser/clip2seg) takes. `auto` routes by the non-frontal test; `0` always masks in canonical crop space; `1` always masks on the unwarped bounding-box crop. Use `0` to isolate whether a bad profile frame is caused by the mask *routing* or by the swap itself. |
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
| `ROOP_CAPTURE_PIPE` | auto | How preview/timeline decode video. `auto` uses OpenCV except on HEVC pixel formats where it silently returns the wrong frame (10-bit, 4:2:2, 4:4:4 — measured up to 16 frames off), which go through an ffmpeg pipe instead. `1` forces the pipe for everything (slower seeks, always exact), `0` forces OpenCV (fast, wrong on those formats). |

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
| `ROOP_PROFILE` | 0 | `1` prints per-stage wall-clock timing (analyze/detect/mask/swap/enhance/…) summed across worker threads. Unrelated to `ROOP_YAW_ALIGN`. |
| `ROOP_DEBUG_MATCH` | unset | Prints identity-matching diagnostics during the swap pass. |
| `ROOP_DEBUG_ANGLE` | 0 | `1` prints, per face per masker: the yaw/pitch keypoint proxies, the non-frontal verdict, which masking path ran, and what fraction of the canonical crop the unwarped box covers. Noisy on video — use on a single preview frame. |

## Identity tracking

Cosine distances are scipy convention (0..2). A same-person **profile** frame sits
0.7–1.0 from a frontal one, so these gates are deliberately loose; wrong-person
matches are rejected by *relative* cross-person comparisons at the call sites,
not by tightening these absolute cutoffs.

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_TRACK_VETO` | 0.85 | Distance beyond which a tracked source is refused for a face (guards ID switches, crossings, bystanders). `0` disables the veto — a tracked source then applies wherever spatial association points. |
| `ROOP_TRACK_VETO_MARGIN` | 0.15 | Also veto when a *different* selected person explains the face this much better. |
| `ROOP_TRACK_VETO_SINGLE` | 0 (off) | Absolute veto for the **single selected person** case, which `ROOP_TRACK_VETO` deliberately skips. Catches a tracker identity switch (two people cross; one leaves and another stands where the track was), which otherwise keeps swapping the wrong face for a run of frames with no identity check at all. Set only high enough to catch unambiguous mismatches — different people measured ~0.93–1.07, so `1.0` is a reasonable trial; values near the match threshold make hard poses blink instead. |
| `ROOP_TRACK_OVERLAP_FRAC` | 0.15 | Fraction of a track's frames that must overlap an already-assigned track of the same person before it counts as a genuinely concurrent second body rather than an occlusion handoff. |
| `ROOP_TRACK_TRUEMEAN` | 1 (on) | Identity-lock matches on the true mean embedding. `0` restores the old recency-biased EMA (the "only the first faceset swaps" behaviour). |

## Multi-angle target bank

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_ANGLE_MANUAL_MAX` | 0.90 | Max distance accepted by `/api/target/add_angle` (manual capture). |
| `ROOP_ANGLE_ACCEPT` | 0.60 | Distance under which `/api/target/auto_angles` accepts a harvested angle. |
| `ROOP_ANGLE_SEED_MAX` | 0.85 | Max distance for a seed frame in `/api/target/auto_angles`. |

## Face quality (FIQA)

Composite heuristic score used by the Face Manager's quality gate — a weighted
blend, not a trained network. Weights need not sum to 1; missing components are
renormalised away.

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_FIQA_W_DET` | 0.25 | Weight of the detector confidence term. |
| `ROOP_FIQA_W_SHARP` | 0.30 | Weight of the sharpness term. |
| `ROOP_FIQA_W_RES` | 0.20 | Weight of the resolution term. |
| `ROOP_FIQA_W_POSE` | 0.15 | Weight of the frontality term. |
| `ROOP_FIQA_W_NORM` | 0.10 | Weight of the embedding-norm term. |

## Expression restore (LivePortrait)

The **😀 Expression restore** slider and **Expression region** selector live in
the Face Swap tab. One env flag exists, for the GridSample rewrite below.

| variable | default | meaning |
|---|---|---|
| `ROOP_EXPR_PATCH_GRIDSAMPLE` | `1` | Rewrite the warping module's 5-D `GridSample` nodes so it runs entirely on the GPU. `0` keeps the stock model and its CPU partition. |
| `ROOP_EXPR_POOL` | `0` (off) | Independent restorer instances so the stage runs N-wide instead of serialised. **Costs ~Nx its VRAM** — see below. |

### What the feature costs, and the one stage still running single-file

Measured across three runs of the same pipeline, 192-frame chunks on an RTX 4070:

| run | expression restore | mean chunk | max |
|---|---|---|---|
| off | — | **9.00 s** | 11.34 s |
| on | stock (CPU partition) | 17.49 s | 30.09 s |
| on | rewritten (below) | **13.53 s** | 16.79 s |

So the rewrite cut the feature's cost by ~23%, and what remains is ~4.5 s per
chunk over having it off. Almost all of that adds **in series**: expression
restore is the only GPU stage without a session pool, so while swap, mask and
detect run N-wide behind `_gpu_guard(pooled=True)`, this one holds the global
lock.

`ROOP_EXPR_POOL=N` gives it independent TensorRT contexts and removes that.
Measured with 8 threads sharing one restorer, as ProcessMgr uses it:

| `ROOP_EXPR_POOL` | throughput | per call | VRAM held |
|---|---|---|---|
| 0 *(serialised)* | 28.7 faces/s | 34.8 ms | 1162 MB |
| **2** | **39.2 faces/s** | 25.5 ms | 1822 MB |
| 3 | 38.6 faces/s | 25.9 ms | 2518 MB |
| 4 | 37.4 faces/s | 26.7 ms | 3180 MB |

**2 is the sweet spot: +37% for +660 MB.** Past that the GPU is saturated and
extra contexts cost more than they return. Note that pool 0 with 8 threads lands
at exactly the single-threaded 34.8 ms, which is what "fully serialised" looks
like. Output is **bit-exact** against the unpooled path, and identical across
slots — the pool changes scheduling only.

Still **off by default**, and deliberately not VRAM-auto-tuned like
`ROOP_TRT_POOL`/`ROOP_DETMASK_POOL`, whose defaults were chosen before these
models existed. A 12 GB card running 4 swapper + 4 detmask instances measured
1.0 GB free mid-render, so `=2` fits with roughly 380 MB to spare — enough, but
not comfortable. If it is unstable, drop `ROOP_TRT_POOL` to 3.

The +37% is the isolated gain. In the real pipeline the pooled stage also stops
holding the global lock, so it can overlap with swap/mask/detect as well; that
part is unmeasured and can only be larger, not smaller.

### The constraint

`warping_spade.onnx` warps a 5-D feature volume `(1,32,16,64,64)` with two
`GridSample` nodes, and **nothing in the GPU stack executes them as shipped**:

* TensorRT's `IGridSampleLayer` is documented 4-D only, so its parser rejects
  both nodes (`addGridSample ... nbDims == 4` / `INVALID_NODE` for
  `/dense_motion_network/GridSample` and `/GridSample`).
* onnxruntime's CUDA GridSample kernel is likewise 4-D only and fails at run
  time with *"Only 4-D tensor is supported"*.

onnxruntime's answer is to partition just those two nodes to CPU. Profiled per
call: 134 ms for the `(22,4,16,64,64)` node plus 12 ms for the
`(1,32,16,64,64)` one — 146 ms, against 1.5 ms for the same maths on the GPU.

### The rewrite (default on)

`roop/gridsample5d.py` replaces each 5-D `GridSample` with its definition —
eight corner gathers and a trilinear weighted sum — using only ops TensorRT
builds natively. The result is cached beside the source as
`warping_spade-trt.onnx`, with a `.meta` stamp recording `PATCH_VERSION` and the
source mtime so an edited rewrite cannot serve a stale graph.

Measured on an RTX 4070, warping module only, FP16 TensorRT, error against an
FP32 CPU run of the stock model:

| path | median | max abs err |
|---|---|---|
| stock — TensorRT + 2 nodes on CPU | 165.2 ms | 4.99e-02 |
| **rewritten — full TensorRT** | **26.8 ms** | 4.71e-02 |
| rewritten — CUDA only | 87.8 ms | 4.40e-03 |
| stock — CPU only *(ground truth)* | 2784 ms | — |
| stock — CUDA | ❌ fails outright | — |

**6.16x on TensorRT**, and it gives CUDA-only machines a working GPU path for
the first time. Accuracy is unchanged: the rewritten graph's FP16 error is no
larger than the stock TensorRT path's.

End-to-end, a whole `Expression_LivePortrait.Run()` (appearance + motion x2 +
stitching + warping + the numpy pre/post) goes **237.8 ms -> 34.0 ms per face,
7.0x** — lifting the stage's throughput ceiling from ~4 to ~29 faces/sec.

The gain is larger than the 146 ms of CPU kernel time because the partition cost
more than the kernels. Profiling the provider assignment shows the stock model
was **9 separate TensorRT subgraphs** wrapped around the 2 CPU nodes, with a
device round-trip at every boundary; the rewritten model is **1 engine**.

Two constraints that are not obvious and cost real debugging:

* **The flattened index is computed in INT32, never in float.** TensorRT runs
  this graph with FP16 enabled; FP16's largest value is 65504 with a spacing of
  32 near the top, and the big node addresses 16·64·64 = 65536 voxels. With the
  index in float the module was still 6.7x faster and *agreed to 3e-06 on CPU* —
  but produced 0.71 max abs error on a [0,1] output under TensorRT.
* **Applied only when a GPU provider is present.** On CPU the expansion is ~9%
  *slower* than onnxruntime's own kernel (2.66 s vs 2.44 s): eight gathers beat
  one fused kernel only when there are cores to spread them over.

### Fallback path

With `ROOP_EXPR_PATCH_GRIDSAMPLE=0`, or if the rewrite fails, the processor
picks providers for that one session itself: TensorRT if available, otherwise
CPU, never CUDA. The parser errors are then silenced around session
construction — `SessionOptions.log_severity_level` does **not** work for this,
since the messages come from the TensorRT logger bridged through onnxruntime's
*global* logger, so the global severity is raised for that one session build and
restored afterwards. On the rewritten path they are not silenced, because the
rewrite removes their cause and any remaining error is real.

FasterLivePortrait also publishes `warping_spade-fix.onnx`, which swaps the node
for a custom `GridSample3D` op. That needs a plugin library this project does not
ship — the model fails to load at all — so it is deliberately not used.

Models (~537 MB) download to `app/models/liveportrait/` on first use.
