import os

import numpy as np
import cv2
import onnxruntime
import threading
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool

THREAD_LOCK_CLIP = threading.Lock()

# DFL's XSeg, as mirrored by the roop-unleashed model host. core.py pre-warms
# this at startup, but only when it finds itself online — so an offline first
# run, or a models folder someone has cleaned out, reaches Initialize with no
# file on disk. Downloading here as well (the same way Mask_Occluder and
# Mask_XSeg3 already do) turns that into a fetch instead of a bare
# onnxruntime load failure raised out of a worker thread, which matters more
# for this engine than for its siblings: DFL XSeg is the DEFAULT mask engine,
# and Mask_RealityUX is built on top of it.
_MODEL_URL = 'https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx'
_MODEL_FILE = 'xseg.onnx'


class Mask_XSeg():
    plugin_options:dict = None

    model_xseg = None

    processorname = 'mask_xseg'
    type = 'mask'


    def __init__(self):
        # Opt-in SessionPool (ROOP_DETMASK_POOL) of independent TensorRT sessions
        # so the mask runs concurrently across worker threads. None → single shared
        # session serialised by the global lock (original safe default).
        self.pool = None


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_xseg is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [_MODEL_URL])
            model_path = os.path.join(model_dir, _MODEL_FILE)
            onnxruntime.set_default_logger_severity(3)

            def _build(_i=0):
                return onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)

            self.model_xseg = _build()
            self.model_inputs = self.model_xseg.get_inputs()
            self.model_outputs = self.model_xseg.get_outputs()

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

            # Optional multi-session pool: primary + (N-1) extras → up to N threads
            # run the mask concurrently, each on its own TensorRT context.
            if session_pool.mask_pooling_enabled():
                n = session_pool.mask_pool_size()
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model_xseg] + extras): _e[i], n)


    def _run_session(self, sess, temp_frame):
        io_binding = sess.io_binding()
        io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        io_binding.bind_output(self.model_outputs[0].name, self.devicename)
        sess.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()


    def Run(self, img1, keywords:str) -> Frame:
        temp_frame = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_CUBIC)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = temp_frame[None, ...]
        if self.pool is not None:
            with self.pool.lease() as sess:
                ort_outs = self._run_session(sess, temp_frame)
        else:
            ort_outs = self._run_session(self.model_xseg, temp_frame)
        # Output: (1, 256, 256, 1) → drop batch + channel dims to a 2D mask.
        # The channel squeeze is not cosmetic. Without it this engine alone hands
        # back a (256, 256, 1) mask where every caller in the project assumes
        # (256, 256), and three of them already carry a workaround written for
        # exactly this: Mask_RealityUX._to_2d (which names it "a measured real
        # bug on XSeg's raw ONNX output"), _recover_undersized_mask's
        # reshape-and-restore dance in procmgr_masking, and _composite_mask
        # leaning on cv2.resize dropping the trailing axis for it. The failure
        # mode when one is missed is silent rather than loud: (h, w, 1) and
        # (h, w) are both individually valid broadcast shapes, so an elementwise
        # combine of the two produces an (h, w, h) array instead of raising.
        # Normalise at the source, the way Mask_Occluder and Mask_XSeg3 do.
        result = ort_outs[0][0]
        if result.ndim == 3:
            result = result[..., 0]
        result = np.clip(result, 0, 1.0)
        result[result < 0.1] = 0
        # invert values to mask areas to keep
        result = 1.0 - result
        return result


    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_xseg
        self.model_xseg = None


