"""GPEN Realistic — GPEN-512's detail without GPEN's colour.

TWO FINDINGS, and the second corrects the first build of this file.

1. GPEN's problem is COLOUR, not detail. It gets called "plastic" or
   "cartoonish", which sounds like softness; it is not. GPEN pushes the whole
   face pink, paints magenta onto the eyelids and flushes the cheeks. Measured
   against the crop the restorer was handed, chroma drift (LAB a/b, mean abs)
   runs 2.7-3.0 while the input is 0. Keeping GPEN's LUMINANCE and taking
   chrominance from the swapper's own crop removes it — 2.72 -> 0.36 at 512 —
   with detail completely unchanged, for 0.27 ms.

2. But the size that matters is 512, not 256, and the reason is the PASTE.
   `realswap` emits a 256 crop, so a 256 restorer returns 256 and pastes at
   scale 1, while a 512 restorer returns 512 and pastes at scale 2 — twice the
   resolution reaches the frame. Detail carried through to the paste, measured
   on a real crop as high-frequency std at 512:

       swap input 2.67 | GPEN-256 2.82 | CodeFormer-512 4.11 | GPEN-512 5.14

   GPEN-256 sits 1.8x below GPEN-512 and barely above the UNENHANCED input. The
   first version of this processor used 256 plus the colour fix and was reported
   as indistinguishable from plain GPEN-256 — correctly. A post-filter cannot
   recover detail the network never synthesised, and that was the mistake.

So this runs GPEN-512, which measures SHARPER THAN CODEFORMER (5.14 vs 4.11)
and faster than it (30.9 ms vs 37.9 before optimisation). Its bad reputation is
finding 1: the cast, which finding 1 removes.

Knobs, for re-measuring rather than for shipping a different default:
    ROOP_GPENR_CHROMA   0 = the swapper's colour (default), 1 = GPEN's own
    ROOP_GPENR_SIZE     512 (default) or 256 for the fast, much softer tier

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


_SIZES = {
    256: (
        'gpen_bfr_256.onnx',
        'https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_256.onnx',
    ),
    512: (
        'GPEN-BFR-512.onnx',
        'https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx',
    ),
    1024: (
        'gpen_bfr_1024.onnx',
        'https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_1024.onnx',
    ),
}


def _select_providers(execution_providers):
    """Return execution providers filtered to available ones."""
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


class Enhance_GPENRealistic:
    """GPEN-512 with colour cast removed: GPEN's luminance, swapper's chroma."""

    processorname = 'gpen_realistic'
    # Declaring self_excluding = True tells ProcessMgr's _gpu_guard that this
    # processor manages its own concurrency; the global GPU lock is skipped.
    # Each pool slot has its own lock (self._slot_locks) so that two worker
    # threads can never share a mutable io_binding.
    self_excluding = True
    type = 'enhance'
    model_template = 'ffhq_512'

    _SIZE_DEFAULT = 512

    def __init__(self):
        self.plugin_options: dict | None = None
        self.devicename: str | None = None
        self._size: int = self._SIZE_DEFAULT
        self._sessions: list[onnxruntime.InferenceSession] = []
        self._bindings: list[onnxruntime.IOBinding] = []
        self._device_buffers: list[tuple[Any, Any] | None] = []
        self._slot_locks: list[threading.Lock] = []
        self._in_name: str | None = None
        self._out_name: str | None = None
        self._lut: np.ndarray | None = None
        self._faces = 0
        self._pool: session_pool.SessionPool | None = None
        self._global_lock = threading.Lock()

    @property
    def reuses_device_buffers(self) -> bool:
        return any(buf is not None for buf in getattr(self, "_device_buffers", []))

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def Initialize(self, plugin_options: dict) -> None:
        requested_device = plugin_options['devicename'].replace('mps', 'cpu')
        if self.plugin_options is not None and self.devicename != requested_device:
            self.Release()

        self.plugin_options = plugin_options
        self.devicename = requested_device

        if self._sessions:
            return

        try:
            want = int(os.environ.get('ROOP_GPENR_SIZE', '') or self._SIZE_DEFAULT)
        except ValueError:
            want = self._SIZE_DEFAULT
        self._size = want if want in _SIZES else self._SIZE_DEFAULT

        from roop.model_registry import ensure_model_downloaded
        from roop.face_enhancer import create_enhancer_session, align_face_5point, inverse_affine_warp_back
        model_key = 'gpen_bfr_1024' if self._size == 1024 else ('gpen_bfr_512' if self._size == 512 else 'gpen_bfr_256')
        model_path = ensure_model_downloaded(model_key)

        providers = _select_providers(roop.globals.execution_providers)

        def _build_slot():
            opts = onnxruntime.SessionOptions()
            opts.log_severity_level = 2
            sess, active_providers = create_enhancer_session(
                model_path,
                requested_providers=providers,
                session_options=opts,
                label=f"GPENRealistic-{self._size}",
                explicit_shape=(1, 3, self._size, self._size),
            )
            iob = sess.io_binding()
            in_info = sess.get_inputs()[0]
            out_info = sess.get_outputs()[0]
            in_name = in_info.name
            out_name = out_info.name

            dev_buf = None
            ort_value_cls = getattr(onnxruntime, "OrtValue", None)
            device_id = 0
            for p in (active_providers or providers):
                if isinstance(p, (tuple, list)) and len(p) > 1 and isinstance(p[1], dict):
                    if "device_id" in p[1]:
                        device_id = int(p[1]["device_id"])
                        break

            can_reuse = (
                self.devicename == "cuda"
                and ort_value_cls is not None
                and hasattr(ort_value_cls, "ortvalue_from_shape_and_type")
                and hasattr(iob, "bind_ortvalue_input")
                and hasattr(iob, "bind_ortvalue_output")
            )
            if can_reuse:
                try:
                    in_dtype = np.float16 if "float16" in str(in_info.type).lower() else np.float32
                    out_dtype = np.float16 if "float16" in str(out_info.type).lower() else np.float32
                    in_shape = (1, 3, self._size, self._size)
                    out_shape = (1, 3, self._size, self._size)
                    in_val = ort_value_cls.ortvalue_from_shape_and_type(in_shape, in_dtype, "cuda", device_id)
                    out_val = ort_value_cls.ortvalue_from_shape_and_type(out_shape, out_dtype, "cuda", device_id)
                    iob.bind_ortvalue_input(in_name, in_val)
                    iob.bind_ortvalue_output(out_name, out_val)
                    dev_buf = (in_val, out_val)
                except Exception:
                    iob.bind_output(out_name, self.devicename)
                    dev_buf = None
            else:
                iob.bind_output(out_name, self.devicename)
            return sess, iob, dev_buf

        primary_sess, primary_iob, primary_buf = _build_slot()
        self._in_name = primary_sess.get_inputs()[0].name
        self._out_name = primary_sess.get_outputs()[0].name
        self._sessions.append(primary_sess)
        self._bindings.append(primary_iob)
        self._device_buffers.append(primary_buf)
        self._slot_locks.append(threading.Lock())

        # uint8 -> float32 in [-1, 1] pre-normalisation LUT
        self._lut = (np.arange(256, dtype=np.float32) / 127.5) - 1.0

        # GPU warm-up so TRT engine builds now rather than on the first frame
        gpu_providers = frozenset({'TensorrtExecutionProvider', 'CUDAExecutionProvider'})
        is_gpu = any(
            (p[0] if isinstance(p, (tuple, list)) else str(p)) in gpu_providers
            for p in providers
        )
        if is_gpu:
            dummy = np.zeros((1, 3, self._size, self._size), dtype=np.float32)
            try:
                self._run_slot(0, dummy)
                print(f'[GPEN Realistic] GPU warm-up passed (size={self._size})')
            except Exception as exc:
                print(f'[GPEN Realistic] GPU warm-up warning: {exc}')

        # Build extra pool slots when TensorRT pooling is enabled
        if session_pool.pooling_enabled() and is_gpu:
            n = session_pool.pool_size()
            try:
                forced = int(os.environ.get('ROOP_GPENR_POOL', '') or 0)
                if forced:
                    n = max(1, forced)
            except ValueError:
                pass
            # VRAM cap: a GPEN-512 pool of 2 costs ~3 GB extra on top of the
            # other resident networks; cap conservatively on smaller cards.
            try:
                gb = session_pool._detect_vram_gb()
                if 0 < gb < 11.5:
                    n = min(n, 1)
                elif gb < 15.5:
                    n = min(n, 2)
            except Exception:
                pass
            for _ in range(n - 1):
                try:
                    s, b, buf = _build_slot()
                    self._sessions.append(s)
                    self._bindings.append(b)
                    self._device_buffers.append(buf)
                    self._slot_locks.append(threading.Lock())
                except Exception as e:
                    print(f'[GPEN Realistic] pool slot failed: {e}')
                    break

            if len(self._sessions) > 1:
                triplets = list(zip(self._sessions, self._bindings, self._device_buffers))
                locks = self._slot_locks

                def _factory(i, _p=triplets, _l=locks):
                    return (_p[i], _l[i])

                self._pool = session_pool.SessionPool(_factory, len(triplets))
                print(f'[GPEN Realistic] pool ready: {len(triplets)} contexts')

        model_lifecycle_manager.register_model(
            name='gpen_realistic',
            unload_cb=self.Release,
            device='cuda' if is_gpu else 'cpu',
        )

    def Release(self) -> None:
        if self._pool is not None:
            self._pool.release()
            self._pool = None
        self._sessions.clear()
        self._bindings.clear()
        self._device_buffers.clear()
        self._slot_locks.clear()
        self._lut = None
        model_lifecycle_manager.set_unloaded('gpen_realistic')
        gc.collect()

    # ── inference ─────────────────────────────────────────────────────────────
    def _run_slot(self, slot_idx: int, x: np.ndarray) -> np.ndarray:
        sess = self._sessions[slot_idx]
        iob = self._bindings[slot_idx]
        lock = self._slot_locks[slot_idx]
        dev_buf = self._device_buffers[slot_idx] if slot_idx < len(self._device_buffers) else None
        with lock:
            try:
                if dev_buf is not None and hasattr(dev_buf[0], "update_inplace"):
                    in_val, out_val = dev_buf
                    in_val.update_inplace(x)
                    sess.run_with_iobinding(iob)
                    sync = getattr(iob, "synchronize_outputs", None)
                    if callable(sync):
                        sync()
                    return [out_val.numpy()]
                iob.bind_cpu_input(self._in_name, x)
                sess.run_with_iobinding(iob)
                return iob.copy_outputs_to_cpu()
            except Exception as exc:
                from roop.face_enhancer import is_cuda_oom, clear_cuda_cache, CudaOOMError
                if is_cuda_oom(exc):
                    clear_cuda_cache()
                    raise CudaOOMError(f"CUDA Out of Memory in GPENRealistic slot {slot_idx}: {exc}") from exc
                raise

    def _infer(self, x: np.ndarray) -> np.ndarray:
        if self._pool is not None:
            with self._pool.lease() as slot:
                (sess, iob, dev_buf), slot_lock = slot
                with slot_lock:
                    try:
                        if dev_buf is not None and hasattr(dev_buf[0], "update_inplace"):
                            in_val, out_val = dev_buf
                            in_val.update_inplace(x)
                            sess.run_with_iobinding(iob)
                            sync = getattr(iob, "synchronize_outputs", None)
                            if callable(sync):
                                sync()
                            return [out_val.numpy()]
                        iob.bind_cpu_input(self._in_name, x)
                        sess.run_with_iobinding(iob)
                        return iob.copy_outputs_to_cpu()
                    except Exception as exc:
                        from roop.face_enhancer import is_cuda_oom, clear_cuda_cache, CudaOOMError
                        if is_cuda_oom(exc):
                            clear_cuda_cache()
                            raise CudaOOMError(f"CUDA Out of Memory in GPENRealistic pool lease: {exc}") from exc
                        raise
        return self._run_slot(0, x)

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
        S = self._size

        src = (cv2.resize(temp_frame, (S, S), interpolation=cv2.INTER_CUBIC)
               if (temp_frame.shape[0] != S or temp_frame.shape[1] != S)
               else temp_frame)

        # LUT gather: uint8 BGR HWC -> float32 RGB CHW in [-1, 1]
        x = self._lut[src.transpose(2, 0, 1)[::-1]][None]

        try:
            with model_lifecycle_manager.execution_guard('gpen_realistic',
                                                         required_gb=0.5):
                ort_outs = self._infer(x)
        except Exception as exc:
            from roop.face_enhancer import is_cuda_oom, clear_cuda_cache
            if is_cuda_oom(exc):
                print(f'[GPEN Realistic] CUDA OOM encountered ({exc}) — clearing cache and using unenhanced frame')
                clear_cuda_cache()
                return sized(temp_frame, input_size)
            raise

        hwc = np.ascontiguousarray(
            ort_outs[0][0][::-1].transpose(1, 2, 0), dtype=np.float32
        )
        del ort_outs

        # Non-finite check before uint8 cast (np.clip does NOT remove NaN)
        if not np.isfinite(hwc.sum()):
            print('[GPEN Realistic] non-finite output — using unenhanced frame')
            return sized(temp_frame, input_size)

        np.maximum(hwc, -1.0, out=hwc)
        restored = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)

        # Collapse check (FP16 failure mode: finite but flat grey output)
        if looks_collapsed(restored):
            print('[GPEN Realistic] output collapsed — using unenhanced frame')
            return sized(temp_frame, input_size)

        # Colour fix: keep GPEN's luminance, restore swapper's chrominance
        try:
            chroma = float(os.environ.get('ROOP_GPENR_CHROMA', '') or 0.0)
        except ValueError:
            chroma = 0.0

        if chroma < 1.0:
            out = luma_only_recolour(restored, src)
            if chroma > 0.0:
                # Linear blend between fixed (chroma=0) and raw (chroma=1)
                out = cv2.addWeighted(out, 1.0 - chroma, restored, chroma, 0)
        else:
            out = restored

        with self._global_lock:
            self._faces += 1

        return sized(out, input_size)

    def enhance_frame(self, frame: Frame, kps: np.ndarray, blend_ratio: float = 1.0) -> Frame:
        """Standalone 5-point landmark alignment -> inference -> inverse affine warp back."""
        from roop.face_enhancer import (
            align_face_5point,
            inverse_affine_warp_back,
            is_cuda_oom,
            clear_cuda_cache,
            CudaOOMError,
        )
        try:
            aligned_crop, M = align_face_5point(frame, kps, crop_size=self._size)
            enhanced_crop, _ = self.Run(None, None, aligned_crop)
            return inverse_affine_warp_back(
                target_frame=frame,
                enhanced_crop=enhanced_crop,
                M=M,
                original_crop=aligned_crop,
                blend_ratio=blend_ratio,
            )
        except Exception as exc:
            if is_cuda_oom(exc):
                clear_cuda_cache()
                if not isinstance(exc, CudaOOMError):
                    raise CudaOOMError(f"CUDA Out of Memory in GPENRealistic enhance_frame: {exc}") from exc
            raise

    # ── compatibility: expose Prepare/Infer/Finish like Enhance_GPEN ─────────
    def Prepare(self, source_faceset, target_face, temp_frame):
        return {'frame': temp_frame, 'target_face': target_face}

    def Finish(self, infer_out, prepared, target_face=None):
        return infer_out

    def cost_summary(self) -> str | None:
        with self._global_lock:
            f = self._faces
        if not f:
            return None
        pool_info = f', {len(self._sessions)}-ctx pool' if len(self._sessions) > 1 else ''
        return (f'[GPEN Realistic] {f} faces at {self._size}px '
                f'(GPEN luminance, swapper chroma{pool_info})')
