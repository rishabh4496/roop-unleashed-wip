from typing import Any, List, Callable
import cv2
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path, conditional_download


# GPEN blind face restoration at three native resolutions. 512 is the classic
# roop weight; 1024/2048 are the FaceFusion exports for large/close-up faces
# (better detail ceiling, proportionally more VRAM/time).
GPEN_MODELS = {
    512: {
        "file": "GPEN-BFR-512.onnx",
        "url": "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx",
    },
    1024: {
        "file": "gpen_bfr_1024.onnx",
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_1024.onnx",
    },
    2048: {
        "file": "gpen_bfr_2048.onnx",
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_2048.onnx",
    },
}


class Enhance_GPEN():
    plugin_options:dict = None

    model_gpen = None
    name = None
    devicename = None

    processorname = 'gpen'
    type = 'enhance'

    def __init__(self):
        self.model_size = 512
        # One live session per resolution. Sessions are kept resident instead of
        # released on size switch: the enhancer comparison grid renders 512/1024/
        # 2048 back-to-back, and releasing mid-cycle both thrashes TensorRT
        # engine loads and yanks the session out from under a Run() in flight
        # (NoneType io_binding / NaN → black face).
        self.sessions = {}

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options

        size = int(plugin_options.get("size", 512))
        if size not in GPEN_MODELS:
            size = 512

        if size not in self.sessions:
            spec = GPEN_MODELS[size]
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [spec["url"]])
            model_path = f"{model_dir}/{spec['file']}"
            session = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.sessions[size] = session

        # replace Mac mps with cpu for the moment
        self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
        self.model_size = size
        self.model_gpen = self.sessions[size]
        self.name = self.model_gpen.get_inputs()[0].name
        self.output_name = self.model_gpen.get_outputs()[0].name

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]
        sz = self.model_size
        temp_frame = cv2.resize(temp_frame, (sz, sz), interpolation=cv2.INTER_CUBIC)

        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)

        io_binding = self.model_gpen.io_binding()
        io_binding.bind_cpu_input(self.name, temp_frame)
        io_binding.bind_output(self.output_name, self.devicename)
        self.model_gpen.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]

        # post-process
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        scale_factor = int(result.shape[1] / input_size)
        return result.astype(np.uint8), scale_factor


    def Release(self):
        self.sessions.clear()
        self.model_gpen = None
