# GPEN GPU diagnostics and benchmarking

## What was wrong

The shipped GPEN ONNX graphs have a fixed input and output batch of
[1, 3, size, size]. A face cannot be appended to a larger conventional batch
without re-exporting the graph. The correct concurrency mechanism is a bounded
pool of independent TensorRT sessions and execution contexts.

Before the fix, standard GPEN created one session directly from the global
provider list without checking the active providers and without disabling CPU
node fallback. It also ran its full OpenCV preprocessing, I/O binding, inference,
device-to-host copy, and postprocessing under ProcessMgr's global TensorRT lock.
That serialized CPU work between workers and created a new pageable CPU input
binding for every face.

GPEN Ultimate already requested a TensorRT context pool, but each slot still
re-bound pageable CPU input and allocated/copied output for every face. Its
Ultimate CPU finishing pass costs materially more than the GPEN inference
bookkeeping, so sparse batch-1 kernels and CPU finish time can show low sampled
GPU utilization even with most VRAM resident.

The current implementation:

- puts TensorRT/CUDA first and removes explicit CPU from strict GPU sessions;
- sets session.disable_cpu_ep_fallback=1, making unsupported CPU node placement
  fail during session creation;
- runs a real warm-up tensor at initialization, surfacing missing DLLs,
  unsupported kernels, OOM, and NaN/Inf before a render begins;
- preallocates one CUDA input/output OrtValue pair per fixed-shape TensorRT slot
  (or per CUDA worker thread) and binds it once;
- pools standard GPEN 256/512 as well as GPEN Ultimate under TensorRT;
- keeps 1024/2048 single-context and FP32 by default to control VRAM and prevent
  FP16 overflow;
- keeps OpenCV prepare/finish work outside the global TensorRT guard.

The crop still must cross host/device boundaries: alignment and compositing are
OpenCV/NumPy CPU stages and the restored crop is immediately consumed on CPU.
I/O binding removes repeated device allocation and binding overhead; it cannot
eliminate the one H2D input and one D2H output transfer without moving the rest
of the face pipeline to CUDA.

GPEN is ONNX Runtime based, not a PyTorch model. PyTorch's CUDA status is still
checked by the probe because it catches a broken application-wide CUDA install,
but it does not select GPEN's execution provider.

## Source-level findings

The pre-fix locations in the repository revision are:

- app/roop/processors/Enhance_GPEN.py:115-120 constructed the session directly
  from the global list with no active-provider validation or CPU-fallback ban.
- app/roop/processors/Enhance_GPEN.py:188-194 created/rebound pageable host
  input and copied the output back to CPU for each face.
- app/roop/processors/Enhance_GPEN.py:147 restricted context pooling to the
  Ultimate profile, leaving standard GPEN single-context.
- app/roop/ProcessMgr.py:3638-3639 put the entire processor Run method under
  the global TensorRT guard when no pool existed.

The replacements are centered at:

- app/roop/processors/Enhance_GPEN.py:77 for strict session creation;
- app/roop/processors/Enhance_GPEN.py:236 for private reusable CUDA buffers;
- app/roop/processors/Enhance_GPEN.py:459 for bounded TensorRT pooling;
- app/roop/processors/Enhance_GPEN.py:506-579 for split prepare/infer/finish;
- app/roop/ProcessMgr.py:3638-3662 for inference-only guard scope.

## Fail-fast smoke test

Stop an active render first so the probe has enough VRAM. From the project root:

    app\env\Scripts\python.exe app\tools\verify_gpen_gpu.py --provider cuda --size 512 --warmup 5 --repeats 50
    app\env\Scripts\python.exe app\tools\verify_gpen_gpu.py --provider tensorrt --size 512 --trt-fp16 --warmup 5 --repeats 50

Success includes all of these lines:

    active=['CUDAExecutionProvider', ...]
    CPU node fallback=disabled
    Reusable device I/O buffers: True
    PASS: GPEN completed with strict GPU execution.

For TensorRT, the active list begins with TensorrtExecutionProvider and may
include CUDA second. CUDA is a permitted GPU fallback for TensorRT-incompatible
nodes; CPU node placement remains disabled.

## Measure before and after

Use the same input clip, source face, enhancer size, mask settings, thread count,
provider, and warm engine cache for both runs. Ignore the first run after a
TensorRT cache deletion because engine building is not render throughput.

In one PowerShell terminal:

    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm --format=csv -l 1

In a second terminal, follow the launcher log:

    Get-Content logs\api\start_react.js\latest -Wait

At startup, confirm the GPEN-specific line says that TensorRT/CUDA is first and
CPU node fallback is disabled. During the render, record:

- end-to-end output FPS after the first steady-state minute;
- GPEN probe mean/median/p95 latency;
- median GPU utilization and power, not only a single nvidia-smi sample;
- peak memory.used;
- the number of TensorRT pool slots printed by GPEN.

High VRAM with low utilization is not proof of CPU fallback. TensorRT engines
and multiple contexts remain resident in VRAM between short fixed-batch calls.
The fail-fast session option and warm-up are the fallback proof; FPS and latency
are the performance measures.

If VRAM is exhausted, set ROOP_TRT_POOL=2 and restart. If utilization falls
while CPU cores are saturated, the remaining bottleneck is face alignment,
Ultimate finishing, masking, video decode/encode, or frames without a selected
face—not the GPEN execution provider.
