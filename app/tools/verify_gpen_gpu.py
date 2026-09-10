#!/usr/bin/env python
r"""Fail-fast GPEN GPU smoke test and latency benchmark.

Run from the project root, after stopping any active render:

    app\env\Scripts\python.exe app\tools\verify_gpen_gpu.py --provider cuda
    app\env\Scripts\python.exe app\tools\verify_gpen_gpu.py --provider tensorrt --trt-fp16
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import onnxruntime as ort
from roop.processors.Enhance_GPEN import (
    _GPENSlot,
    create_gpen_session,
)

MODEL_NAMES = {
    256: "gpen_bfr_256.onnx",
    512: "GPEN-BFR-512.onnx",
    1024: "gpen_bfr_1024.onnx",
    2048: "gpen_bfr_2048.onnx",
}


def _args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate that GPEN executes with CPU node fallback disabled and "
            "report end-to-end tensor latency (H2D + inference + D2H)."
        )
    )
    parser.add_argument("--model", type=Path, help="Override ONNX model path")
    parser.add_argument("--size", type=int, choices=MODEL_NAMES, default=512)
    parser.add_argument(
        "--provider",
        choices=("cuda", "tensorrt"),
        default="cuda",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--trt-fp16",
        action="store_true",
        help="Build/use a TensorRT FP16 engine (recommended only for 256/512)",
    )
    parser.add_argument(
        "--skip-torch-check",
        action="store_true",
        help="Do not require torch.cuda.is_available(); GPEN itself uses ORT",
    )
    return parser.parse_args()


def _torch_check(skip: bool):
    if skip:
        print("PyTorch CUDA check: skipped")
        return
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch import failed: {exc}") from exc
    available = torch.cuda.is_available()
    print(
        f"PyTorch: {torch.__version__}; CUDA runtime={torch.version.cuda}; "
        f"cuda.is_available()={available}"
    )
    if not available:
        raise RuntimeError(
            "PyTorch cannot access CUDA. Fix the installed Torch/CUDA/driver "
            "stack, or pass --skip-torch-check to test ORT independently."
        )
    print(
        f"CUDA device {torch.cuda.current_device()}: "
        f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
    )


def _providers(args):
    cuda = (
        "CUDAExecutionProvider",
        {"device_id": args.device_id},
    )
    if args.provider == "cuda":
        return [cuda]
    cache = APP_DIR.parent / "cache" / "TRTEngine" / "gpen_probe"
    cache.mkdir(parents=True, exist_ok=True)
    tensorrt = (
        "TensorrtExecutionProvider",
        {
            "device_id": args.device_id,
            "trt_fp16_enable": args.trt_fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache),
        },
    )
    return [tensorrt, cuda]


def _gpu_snapshot():
    command = [
        "nvidia-smi",
        (
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
            "power.draw"
        ),
        "--format=csv,noheader",
    ]
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:  # noqa: BLE001 - CLI error boundary
        return f"nvidia-smi unavailable: {exc}"


def main():
    args = _args()
    if args.warmup < 1 or args.repeats < 1:
        raise SystemExit("--warmup and --repeats must both be >= 1")

    model_path = (args.model or
                  (APP_DIR / "models" / MODEL_NAMES[args.size])).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"GPEN model not found: {model_path}. Start/install the app once "
            "to download it, or pass --model."
        )

    print(f"Python: {sys.executable}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Available ORT providers: {ort.get_available_providers()}")
    _torch_check(args.skip_torch_check)
    print(f"GPU before session: {_gpu_snapshot()}")

    providers = _providers(args)
    session = create_gpen_session(
        str(model_path),
        providers,
        "cuda",
        label=f"GPEN-PROBE-{args.size}",
    )
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(
        f"Input: {input_info.name} {input_info.type} {input_info.shape}; "
        f"output: {output_info.name} {output_info.type} {output_info.shape}"
    )
    if input_info.shape[0] != 1:
        raise RuntimeError(
            f"Expected the shipped fixed batch=1 graph, got {input_info.shape}"
        )

    slot = _GPENSlot(
        session,
        input_info.name,
        output_info.name,
        "cuda",
        providers,
    )
    print(f"Reusable device I/O buffers: {slot.reuses_device_buffers}")
    tensor = np.random.default_rng(7).uniform(
        -1.0,
        1.0,
        size=tuple(int(dim) for dim in input_info.shape),
    ).astype(np.float32)
    tensor = np.ascontiguousarray(tensor)

    for _ in range(args.warmup):
        output = slot.run(tensor)
    if not np.isfinite(output).all():
        raise RuntimeError("GPEN warm-up produced NaN/Inf")

    samples_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        output = slot.run(tensor)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    if not np.isfinite(output).all():
        raise RuntimeError("GPEN timed inference produced NaN/Inf")
    ordered = sorted(samples_ms)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    mean = statistics.fmean(samples_ms)
    print(
        f"Latency including H2D + inference + D2H: "
        f"mean={mean:.2f} ms, median={statistics.median(samples_ms):.2f} ms, "
        f"p95={p95:.2f} ms, tensor throughput={1000.0 / mean:.2f}/s"
    )
    print(
        f"Output: shape={output.shape}, dtype={output.dtype}, "
        f"range=({float(output.min()):.4f}, {float(output.max()):.4f})"
    )
    print(f"GPU after benchmark: {_gpu_snapshot()}")
    print("PASS: GPEN completed with strict GPU execution.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI error boundary
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
