"""UltraMax — CodeFormer (fp16) on a lean host path with colour correction.

WHAT THIS IS. UltraMax runs `codeformer.fp16.onnx` — the same weights as
`Codeformer (fp16)`. What it changes is everything AROUND the network:

1. LEAN HOST PATH: pre/post via 256-entry LUT gather + one saturating C++
   scale instead of five numpy passes. Net: ~27 ms vs CodeFormer's ~36 ms,
   same weights, same fidelity.

2. COLOUR FIX (2026-08-24): CodeFormer's fp16 output is measurably pale
   (chroma drift 2.51, dL +2.22). luma_only_recolour keeps the network's
   luminance and replaces chrominance from the swapper crop. Cost: 0.27 ms.
   ROOP_ULTRAMAX_CHROMA=1 restores CodeFormer's original colour.

3. OPTIONAL DUAL-STREAM ENGINE (ROOP_ULTRAMAX_DUAL=1): runs both CodeFormer
   (structure) and GPEN-512 (detail) on each crop and combines them via an
   edge-aware frequency split. Costs 3x the budget; OFF by default. See the
   docstring in roop-ultimate for full measurement data.

Knobs:
    ROOP_ULTRAMAX_CHROMA    0=colour-fixed (default), 1=CodeFormer's own
    ROOP_ULTRAMAX_DUAL      1 enables the frequency-split dual-stream engine
    ROOP_ULTRAMAX_POOL      override pool size (int)

Ported from roop-ultimate; adapted for roop-unleashed-wip API surface.
"""

from __future__ import annotations

import gc
import os
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.typing import Face, FaceSet, Frame
from roop.processors.enhance_common import (
    looks_collapsed, sized, luma_only_recolour,
)
from roop.utilities import conditional_download, resolve_relative_path
from roop.model_lifecycle import model_lifecycle_manager
from roop.provider_fallback import build_cuda_fallback_providers
from roop import session_pool

try:
    from roop.processors.frequency_split import (
        frequency_split,
        frequency_split_luma,
        reinhard_lab,
    )
    _FREQ_SPLIT_AVAILABLE = True
except ImportError:
    _FREQ_SPLIT_AVAILABLE = False


# Cards below this threshold get one structural context.  A CodeFormer-fp16
# pool of 2 costs ~530 MB extra on top of the other resident nets; on a 12GB
# card a third context pages over PCIe and wedges the render.
_POOL_SINGLE_CONTEXT_BELOW_GB = 15.5

_CODEFORMER_URL = (
    'https://huggingface.co/countfloyd/deepfake/resolve/main/'
    'CodeFormer/codeformer.fp16.onnx'
)
_GPEN512_URL = (
    'https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx'
)


def _select_providers(execution_providers):
    available_fn = getattr(onnxruntime, 'get_available_providers', None)
    available = set(list(available_fn()) if callable(available_fn)
                    else [str(p) if not isinstance(p, (tuple, list)) else str(p[0])
                          for p in (execution_providers or [])])
    providers = []
    for p in (execution_providers or []):
        name = p[0] if isinstance(p, (tuple, list)) else str(p)
        if name in available:
            providers.append(p)
    return providers or ['CPUExecutionProvider']


def _build_session(model_path, providers):
    from roop.face_enhancer import create_enhancer_session
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 2
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess, _ = create_enhancer_session(model_path, requested_providers=providers, session_options=opts, label="UltraMax")
    iob = sess.io_binding()
    iob.bind_output(sess.get_outputs()[0].name, 'cpu')
    return sess, iob


def _pool_size(requested: int) -> int:
    n = max(1, int(requested or 1))
    try:
        gb = session_pool._detect_vram_gb()
    except Exception:
        gb = 0
    if 0 < gb < _POOL_SINGLE_CONTEXT_BELOW_GB:
        n = 1
    try:
        forced = int(os.environ.get('ROOP_ULTRAMAX_POOL', '') or 0)
    except ValueError:
        forced = 0
    return max(1, forced) if forced else n


class Enhance_UltraMax:
    """CodeFormer FP16 on a lean host path with colour fix and optional dual-stream."""

    processorname = 'ultramax'
    # self_excluding = True: ProcessMgr's _gpu_guard skips the global GPU lock
    # for this processor. Each pool slot has its own lock so that io_binding
    # state is never shared between worker threads.
    self_excluding = True
    type = 'enhance'
    model_template = 'ffhq_512'

    # Dual-stream frequency-split engine. OFF by default (see module docstring).
    _DUAL_STREAM = False
    _DUAL_RADIUS = 8
    _DUAL_EPS = 0.04
    _DUAL_GAIN = 1.25
    _DUAL_CLAMP = 24.0
    _DUAL_L_WEIGHT = 0.5
    _DUAL_MODE = 'luma'

    _warned_colour = False
    _warned_dual = False

    def __init__(self):
        self.plugin_options: dict | None = None
        self.devicename: str | None = None
        self._sessions: list[onnxruntime.InferenceSession] = []
        self._bindings: list[onnxruntime.IOBinding] = []
        self._slot_locks: list[threading.Lock] = []
        self._in_name: str | None = None
        self._in_dtype = np.float32
        self._lut: np.ndarray | None = None
        self._pool: session_pool.SessionPool | None = None
        self._faces = 0
        self._global_lock = threading.Lock()
        # Detail stream (GPEN-512), only when dual-stream is active
        self._detail_session: onnxruntime.InferenceSession | None = None
        self._detail_iob: onnxruntime.IOBinding | None = None
        self._detail_in: str | None = None
        self._detail_lut: np.ndarray | None = None
        self._detail_lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────────
    @classmethod
    def _dual_enabled(cls) -> bool:
        v = os.environ.get('ROOP_ULTRAMAX_DUAL')
        if v is None or v == '':
            return bool(cls._DUAL_STREAM)
        return v.strip().lower() in ('1', 'on', 'true', 'yes')

    def Initialize(self, plugin_options: dict) -> None:
        requested_device = plugin_options['devicename'].replace('mps', 'cpu')
        if self.plugin_options is not None and self.devicename != requested_device:
            self.Release()

        self.plugin_options = plugin_options
        self.devicename = requested_device

        if self._sessions:
            return

        from roop.model_registry import ensure_model_downloaded
        model_path = ensure_model_downloaded('codeformer_fp16')

        providers = _select_providers(roop.globals.execution_providers)
        gpu_providers = frozenset({'TensorrtExecutionProvider',
                                   'CUDAExecutionProvider'})
        is_gpu = any(
            (p[0] if isinstance(p, (tuple, list)) else str(p)) in gpu_providers
            for p in providers
        )

        primary_sess, primary_iob = _build_session(model_path, providers)
        self._in_name = primary_sess.get_inputs()[0].name
        in_type = primary_sess.get_inputs()[0].type
        self._in_dtype = np.float16 if 'float16' in str(in_type) else np.float32
        self._sessions.append(primary_sess)
        self._bindings.append(primary_iob)
        self._slot_locks.append(threading.Lock())

        # LUT: uint8 -> model dtype in [-1, 1]
        self._lut = (
            (np.arange(256, dtype=np.float32) / 127.5 - 1.0)
            .astype(self._in_dtype)
        )

        # GPU warm-up
        if is_gpu:
            dummy = np.zeros((1, 3, 512, 512), dtype=self._in_dtype)
            try:
                self._run_slot(0, dummy)
                print('[UltraMax] GPU warm-up passed')
            except Exception as exc:
                print(f'[UltraMax] GPU warm-up warning: {exc}')

        # Multi-context pool
        if session_pool.pooling_enabled() and is_gpu:
            n = _pool_size(session_pool.pool_size())
            for _ in range(n - 1):
                try:
                    s, b = _build_session(model_path, providers)
                    self._sessions.append(s)
                    self._bindings.append(b)
                    self._slot_locks.append(threading.Lock())
                except Exception as e:
                    print(f'[UltraMax] pool slot failed: {e}')
                    break

            if len(self._sessions) > 1:
                pairs = list(zip(self._sessions, self._bindings))
                locks = self._slot_locks

                def _factory(i, _p=pairs, _l=locks):
                    return (_p[i], _l[i])

                self._pool = session_pool.SessionPool(_factory, len(pairs))
                print(f'[UltraMax] pool ready: {len(self._sessions)} contexts')

        # Optional dual-stream detail network (GPEN-512)
        if self._dual_enabled() and _FREQ_SPLIT_AVAILABLE:
            try:
                self._init_detail_stream(providers)
            except Exception as e:
                self._detail_session = None
                print(f'[UltraMax] dual-stream detail network failed ({e}); '
                      f'running single-stream', flush=True)
        elif self._dual_enabled() and not _FREQ_SPLIT_AVAILABLE:
            print('[UltraMax] frequency_split not available; '
                  'dual-stream disabled', flush=True)

        model_lifecycle_manager.register_model(
            name='ultramax',
            unload_cb=self.Release,
            device='cuda' if is_gpu else 'cpu',
        )

    def _init_detail_stream(self, providers):
        """Build the GPEN-512 detail network for the dual-stream engine."""
        from roop.model_registry import ensure_model_downloaded
        model_path = ensure_model_downloaded('gpen_bfr_512')

        self._detail_session, self._detail_iob = _build_session(model_path,
                                                                 providers)
        self._detail_in = self._detail_session.get_inputs()[0].name
        self._detail_lut = (np.arange(256, dtype=np.float32) / 127.5) - 1.0
        print('[UltraMax] dual-stream GPEN-512 detail network ready', flush=True)

    def Release(self) -> None:
        if self._pool is not None:
            self._pool.release()
            self._pool = None
        self._sessions.clear()
        self._bindings.clear()
        self._slot_locks.clear()
        self._lut = None
        self._detail_session = None
        self._detail_iob = None
        self._detail_lut = None
        model_lifecycle_manager.set_unloaded('ultramax')
        gc.collect()

    # ── inference ─────────────────────────────────────────────────────────────
    def _run_slot(self, slot_idx: int, x: np.ndarray) -> np.ndarray:
        sess = self._sessions[slot_idx]
        iob = self._bindings[slot_idx]
        lock = self._slot_locks[slot_idx]
        with lock:
            iob.bind_cpu_input(self._in_name, x)
            sess.run_with_iobinding(iob)
            return iob.copy_outputs_to_cpu()

    def _infer(self, x: np.ndarray) -> np.ndarray:
        if self._pool is not None:
            with self._pool.lease() as slot:
                (sess, iob), slot_lock = slot
                with slot_lock:
                    iob.bind_cpu_input(self._in_name, x)
                    sess.run_with_iobinding(iob)
                    return iob.copy_outputs_to_cpu()
        return self._run_slot(0, x)

    def _detail_stream(self, src512: np.ndarray) -> np.ndarray | None:
        """Run GPEN-512 on src512; return uint8 BGR result or None on failure."""
        if self._detail_session is None:
            return None
        try:
            x = self._detail_lut[src512.transpose(2, 0, 1)[::-1]][None]
            with self._detail_lock:
                self._detail_iob.bind_cpu_input(self._detail_in, x)
                self._detail_session.run_with_iobinding(self._detail_iob)
                outs = self._detail_iob.copy_outputs_to_cpu()
            hwc = np.ascontiguousarray(outs[0][0][::-1].transpose(1, 2, 0),
                                       dtype=np.float32)
            del outs
            if not np.isfinite(hwc.sum()):
                raise ValueError('non-finite detail output')
            np.maximum(hwc, -1.0, out=hwc)
            detail = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)
            if looks_collapsed(detail):
                raise ValueError('detail output collapsed')
            return detail
        except Exception as e:
            if not Enhance_UltraMax._warned_dual:
                Enhance_UltraMax._warned_dual = True
                print(f'[UltraMax] detail stream error ({e}); '
                      f'falling back to single-stream', flush=True)
            return None

    # ── ProcessMgr entry point ────────────────────────────────────────────────
    def Run(
        self,
        source_faceset: FaceSet,
        target_face: Face,
        temp_frame: Frame,
    ) -> tuple[Frame, int]:
        if temp_frame is None:
            return temp_frame, 1

        input_size = temp_frame.shape[1]
        S = 512

        src = (cv2.resize(temp_frame, (S, S), interpolation=cv2.INTER_CUBIC)
               if (temp_frame.shape[0] != S or temp_frame.shape[1] != S)
               else temp_frame)

        # LUT gather: uint8 BGR HWC -> model dtype RGB CHW in [-1, 1]
        x = self._lut[src.transpose(2, 0, 1)[::-1]][None]

        with model_lifecycle_manager.execution_guard('ultramax', required_gb=0.5):
            ort_outs = self._infer(x)

        hwc = np.ascontiguousarray(
            ort_outs[0][0][::-1].transpose(1, 2, 0), dtype=np.float32
        )
        del ort_outs

        if not np.isfinite(hwc.sum()):
            print('[UltraMax] non-finite output — using unenhanced frame')
            return sized(temp_frame, input_size)

        np.maximum(hwc, -1.0, out=hwc)
        restored = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)

        if looks_collapsed(restored):
            print('[UltraMax] output collapsed — using unenhanced frame')
            return sized(temp_frame, input_size)

        # Colour fix: replace CodeFormer's pale chrominance with swapper's
        try:
            chroma = float(os.environ.get('ROOP_ULTRAMAX_CHROMA', '') or 0.0)
        except ValueError:
            chroma = 0.0
        from roop.face_enhancer import apply_ultramax_composite
        blend_ratio = getattr(roop.globals, 'blend_ratio', 1.0)
        if blend_ratio is None:
            blend_ratio = 1.0

        detail = self._detail_stream(src) if (self._dual_enabled() and _FREQ_SPLIT_AVAILABLE) else None

        out = apply_ultramax_composite(
            structure_crop=restored,
            detail_crop=detail,
            reference_crop=src,
            blend_ratio=blend_ratio,
            chroma_weight=chroma,
            dual_stream=(detail is not None),
            gain=self._DUAL_GAIN,
        )

        with self._global_lock:
            self._faces += 1

        return sized(out, input_size)

    def enhance_frame(self, frame: Frame, kps: np.ndarray, blend_ratio: float = 1.0) -> Frame:
        """Standalone 5-point landmark alignment -> inference -> inverse affine warp back."""
        from roop.face_enhancer import align_face_5point, inverse_affine_warp_back
        aligned_crop, M = align_face_5point(frame, kps, crop_size=512)
        enhanced_crop, _ = self.Run(None, None, aligned_crop)
        return inverse_affine_warp_back(
            target_frame=frame,
            enhanced_crop=enhanced_crop,
            M=M,
            original_crop=aligned_crop,
            blend_ratio=blend_ratio,
        )

    # ── compatibility: expose Prepare/Infer/Finish like Enhance_CodeFormer ───
    def Prepare(self, source_faceset, target_face, temp_frame):
        return {'frame': temp_frame, 'target_face': target_face}

    def Finish(self, infer_out, prepared, target_face=None):
        return infer_out

    def cost_summary(self) -> str | None:
        with self._global_lock:
            f = self._faces
        if not f:
            return None
        dual_info = ', dual-stream' if (self._dual_enabled()
                                        and self._detail_session is not None) else ''
        pool_info = f', {len(self._sessions)}-ctx pool' if len(self._sessions) > 1 else ''
        return (f'[UltraMax] {f} faces (CodeFormer fp16 + colour fix'
                f'{pool_info}{dual_info})')
