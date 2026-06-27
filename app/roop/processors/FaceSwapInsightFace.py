import os
import roop.globals
import cv2
import numpy as np
import onnx
import onnxruntime

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool


# ── Per-model swap contract ───────────────────────────────────────────────────
# Every swapper here consumes the same ArcFace (buffalo_l / w600k_r50) identity
# embedding, but each differs in resolution, input normalization, identity
# projection (emap) and output range. ProcessMgr reads the published attributes
# (model_output_size / model_mean / model_standard_deviation / model_denormalize)
# to drive align/normalize. Model files download lazily into app/models/ on the
# first run that selects them.
SWAP_MODELS = {
    # inswapper_128 — the original. arcface align, [0,1] input, identity =
    # normed_embedding @ emap, [0,1] output.
    "inswapper": {
        "file": "inswapper_128.onnx",
        "url": "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
        "output_size": 128,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "use_emap": True,
    },
    # ReSwapper 256 — open reproduction of inswapper at 2x resolution. Same
    # identity pipeline (emap), near drop-in.
    "reswapper": {
        "file": "reswapper_256.onnx",
        "url": "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/reswapper_256.onnx",
        "output_size": 256,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "use_emap": True,
    },
    # HyperSwap 256 (FaceFusion) — 256px. [-1,1] input (mean/std 0.5), identity =
    # normed_embedding directly (NO emap), de-normalized output. The model emits
    # (image, mask); we use the image output.
    "hyperswap": {
        "file": "hyperswap_1a_256.onnx",
        "url": "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "use_emap": False,
    },
}


# Opt-in batched swap (ROOP_BATCH_SWAP=1): runs multiple face crops through one
# inference call instead of one-at-a-time, to better saturate the GPU. The stock
# swap ONNX is fixed batch-1; we relax the input/output batch dim to symbolic so
# the session (and TensorRT engine) accept batches. Verified to produce output
# numerically identical to per-crop runs.
_BATCH_SWAP = os.environ.get('ROOP_BATCH_SWAP', '0') == '1'


def _dynamic_batch_model_bytes(model_path):
    """Return the swap ONNX serialized with a symbolic batch dimension."""
    m = onnx.load(model_path)
    for t in list(m.graph.input) + list(m.graph.output):
        dims = t.type.tensor_type.shape.dim
        if len(dims):
            dims[0].dim_param = 'N'
            dims[0].ClearField('dim_value')
    return m.SerializeToString()


def _swap_providers(providers):
    """Return a copy of `providers` with the TensorRT provider forced to FP32.

    inswapper_128 (and the other emap swappers) have layers that overflow in
    FP16, producing rainbow/smudge artifacts when the global precision mode is
    'mixed'/'fp16'. The swapper is tiny (128-256px), so full precision costs
    almost nothing while fixing the corruption; detection and enhancers stay on
    FP16 where they're stable and fast. Opt back into an FP16 swapper with
    ROOP_SWAP_FP16=1 (not recommended)."""
    if os.environ.get('ROOP_SWAP_FP16', '0') == '1':
        return providers
    patched = []
    for p in providers:
        if isinstance(p, (tuple, list)) and len(p) == 2 and 'tensorrt' in str(p[0]).lower():
            name, opts = p[0], dict(p[1])
            opts['trt_fp16_enable'] = False
            # Separate engine cache so the FP32 swap engine never collides with
            # the FP16 engines TensorRT builds for the other models.
            cache = opts.get('trt_engine_cache_path')
            if cache:
                fp32_cache = cache + '_swap_fp32'
                os.makedirs(fp32_cache, exist_ok=True)
                opts['trt_engine_cache_path'] = fp32_cache
            patched.append((name, opts))
        else:
            patched.append(p)
    return patched


class FaceSwapInsightFace():
    processorname = 'faceswap'
    type = 'swap'

    def __init__(self):
        self.plugin_options = None
        self.model_swap_insightface = None
        self.emap = None
        self.image_input_name = "target"
        self.embed_input_name = "source"
        self.loaded_model_key = None
        self.devicename = None
        self.pool = None        # SessionPool of extra sessions (TRT multi-context)
        # Contract consumed by ProcessMgr — defaults match inswapper_128.
        self.model_output_size = 128
        self.model_mean = [0.0, 0.0, 0.0]
        self.model_standard_deviation = [1.0, 1.0, 1.0]
        self.model_denormalize = False

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options

        swap_model = plugin_options.get("swap_model", "inswapper")
        if swap_model not in SWAP_MODELS:
            swap_model = "inswapper"
        spec = SWAP_MODELS[swap_model]

        # Reload when the user switched to a different swap model.
        if self.model_swap_insightface is not None and self.loaded_model_key != swap_model:
            self.Release()

        if self.model_swap_insightface is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [spec["url"]])
            model_path = os.path.join(model_dir, spec["file"])

            graph = onnx.load(model_path).graph
            self.emap = self._find_emap(graph) if spec["use_emap"] else None

            self.devicename = plugin_options["devicename"].replace('mps', 'cpu')

            swap_providers = _swap_providers(roop.globals.execution_providers)
            # When batched swap is enabled, build the session from a batch-dynamic
            # copy of the model so it accepts >1 crop per inference.
            model_arg = _dynamic_batch_model_bytes(model_path) if _BATCH_SWAP else model_path

            def _build(_i=0):
                sess_options = onnxruntime.SessionOptions()
                sess_options.enable_cpu_mem_arena = False
                return onnxruntime.InferenceSession(
                    model_arg, sess_options, providers=swap_providers)

            self.model_swap_insightface = _build()

            # Resolve input tensor names by rank instead of assuming names:
            # rank-4 = the image (NCHW), rank-2 = the identity embedding.
            for inp in self.model_swap_insightface.get_inputs():
                rank = len(inp.shape)
                if rank == 4:
                    self.image_input_name = inp.name
                elif rank == 2:
                    self.embed_input_name = inp.name

            # Optional TensorRT multi-context pool: the primary session plus
            # (N-1) independent extras so up to N worker threads can swap
            # concurrently instead of serialising behind the global GPU lock.
            if session_pool.pooling_enabled():
                n = session_pool.pool_size()
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model_swap_insightface] + extras): _e[i], n)

            # Publish the per-model contract ProcessMgr reads.
            self.model_output_size = spec["output_size"]
            self.model_mean = spec["mean"]
            self.model_standard_deviation = spec["standard_deviation"]
            self.model_denormalize = spec["denormalize"]
            self.loaded_model_key = swap_model

    @staticmethod
    def _find_emap(graph):
        """Locate the 512x512 identity-projection matrix (emap) embedded in the onnx."""
        for init in reversed(graph.initializer):
            arr = onnx.numpy_helper.to_array(init)
            if arr.ndim == 2 and arr.shape == (512, 512):
                return arr
        # Fallback: inswapper_128 stores emap as the last initializer.
        return onnx.numpy_helper.to_array(graph.initializer[-1])

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        latent = source_face.normed_embedding.reshape((1, -1)).astype(np.float32)
        if self.emap is not None:
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)
        # Use the standard run() API rather than io_binding.  io_binding with
        # bind_output() (no device_type) leaves output placement to TensorRT,
        # which registers a device type that copy_outputs_to_cpu() has no
        # transfer path for. run() handles all device transfers internally and
        # works correctly across CPU, CUDA, and TensorRT execution providers.
        feed = {self.image_input_name: temp_frame, self.embed_input_name: latent}
        if self.pool is not None:
            # Lease an independent session so this thread's GPU work runs on its
            # own TensorRT context, concurrently with other workers.
            with self.pool.lease() as sess:
                ort_outs = sess.run(None, feed)
        else:
            ort_outs = self.model_swap_insightface.run(None, feed)
        # Some models (HyperSwap) emit (image, mask); the image is output [0].
        return ort_outs[0][0]

    def RunBatch(self, source_face: Face, target_face: Face, temp_frames: list) -> list:
        """Batched equivalent of Run: temp_frames is a list of [1,3,H,W]
        preprocessed crops sharing the same source identity. Returns a list of
        [3,H,W] outputs, one per crop — numerically identical to calling Run on
        each, but in a single inference (better GPU utilization). Requires the
        session to be batch-dynamic (ROOP_BATCH_SWAP=1)."""
        latent = source_face.normed_embedding.reshape((1, -1)).astype(np.float32)
        if self.emap is not None:
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)
        img_batch = np.concatenate(temp_frames, axis=0).astype(np.float32)   # [B,3,H,W]
        latent_batch = np.repeat(latent, img_batch.shape[0], axis=0)         # [B,512]
        feed = {self.image_input_name: img_batch, self.embed_input_name: latent_batch}
        if self.pool is not None:
            with self.pool.lease() as sess:
                ort_outs = sess.run(None, feed)
        else:
            ort_outs = self.model_swap_insightface.run(None, feed)
        out = ort_outs[0]   # [B,3,H,W]
        return [out[i] for i in range(out.shape[0])]

    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_swap_insightface
        self.model_swap_insightface = None
        self.emap = None
        self.loaded_model_key = None
