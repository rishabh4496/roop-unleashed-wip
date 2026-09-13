from typing import Any, List, Callable
import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path, require_local_model
from roop.processors.enhance_common import is_usable, sized
from roop.model_lifecycle import model_lifecycle_manager
from roop.provider_fallback import create_fallback_session


# THREAD_LOCK = threading.Lock()


class Enhance_GFPGAN():
    plugin_options:dict = None

    model_gfpgan = None
    name = None
    devicename = None

    processorname = 'gfpgan'
    type = 'enhance'
    # FFHQ-trained — see Enhance_CodeFormer.model_template for the measured
    # mismatch against each swapper's crop, and ProcessMgr for the re-warp.
    model_template = 'ffhq_512'


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_gfpgan is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.onnx')
            require_local_model(model_path, 'GFPGAN face enhancer', only_when_offline=True)
            self.model_gfpgan = create_fallback_session(
                model_path, None, providers=roop.globals.execution_providers, session_name="GFPGAN")
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

            # Register with ModelLifecycleManager for VRAM guard & offloading
            model_lifecycle_manager.register_model(
                name="gfpgan",
                unload_cb=self.Release,
                device="cuda" if any("CUDA" in str(p) or "Tensorrt" in str(p) for p in roop.globals.execution_providers) else "cpu"
            )

        self.name = self.model_gfpgan.get_inputs()[0].name
        self.output_name = self.model_gfpgan.get_outputs()[0].name

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]
        temp_frame = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        fallback_bgr = temp_frame   # resized input, kept for the non-finite guard

        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)

        if self.model_gfpgan is None:
            self.Initialize(self.plugin_options or {"devicename": "cuda"})

        with model_lifecycle_manager.execution_guard("gfpgan", required_gb=1.5):
            io_binding = self.model_gfpgan.io_binding()
            io_binding.bind_cpu_input(self.name, temp_frame)
            io_binding.bind_output(self.output_name, self.devicename)
            self.model_gfpgan.run_with_iobinding(io_binding)
            ort_outs = io_binding.copy_outputs_to_cpu()
            result = ort_outs[0][0]

        # np.clip does not remove NaN and uint8(NaN) is 0, so a single
        # overflowed value paints black and a saturated graph paints a black
        # FACE — silently. See enhance_common.is_usable.
        if not is_usable(result):
            print("[GFPGAN] non-finite output — using unenhanced frame "
                  "(FP16 overflow? try an fp32 provider)")
            return sized(fallback_bgr.astype(np.uint8), input_size)

        # post-process
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return sized(result.astype(np.uint8), input_size)


    def Release(self):
        if self.model_gfpgan is not None:
            del self.model_gfpgan
            self.model_gfpgan = None
        model_lifecycle_manager.set_unloaded("gfpgan")
