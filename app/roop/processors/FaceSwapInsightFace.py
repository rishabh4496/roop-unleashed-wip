import os
import roop.globals
import cv2
import numpy as np
import onnx
import onnxruntime

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path, conditional_download


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
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False
            self.model_swap_insightface = onnxruntime.InferenceSession(
                model_path, sess_options, providers=roop.globals.execution_providers)

            # Resolve input tensor names by rank instead of assuming names:
            # rank-4 = the image (NCHW), rank-2 = the identity embedding.
            for inp in self.model_swap_insightface.get_inputs():
                rank = len(inp.shape)
                if rank == 4:
                    self.image_input_name = inp.name
                elif rank == 2:
                    self.embed_input_name = inp.name

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
        ort_outs = self.model_swap_insightface.run(
            None, {self.image_input_name: temp_frame, self.embed_input_name: latent}
        )
        # Some models (HyperSwap) emit (image, mask); the image is output [0].
        return ort_outs[0][0]

    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None
        self.emap = None
        self.loaded_model_key = None
