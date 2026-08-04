from typing import Any, List, Callable
import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path
from roop.processors.enhance_common import is_usable, sized


# THREAD_LOCK = threading.Lock()


class Enhance_CodeFormer():
    model_codeformer = None

    plugin_options:dict = None
    fp16 = False
    in_dtype = np.float32

    processorname = 'codeformer'
    type = 'enhance'
    # The 5-point alignment this model was TRAINED on. CodeFormer, like every
    # other restorer here that declares one, learned its prior from FFHQ-
    # aligned 512 faces — but the crop it is handed comes from the SWAPPER's
    # template, because the enhancer reuses the swap crop. Measured against
    # ffhq_512 at 512px, the mismatch is not small:
    #
    #   arcface (inswapper family)  scale 0.856   mean landmark error 22.1 px
    #   arcface_112_v1 (ghost…)     scale 0.828                       16.3 px
    #   arcface_112_v2 (blendswap)  scale 0.744                       30.7 px
    #   mtcnn_512 (hififace)        scale 0.875                       13.2 px
    #   ffhq_512 (uniface…)         scale 1.000                        0.0 px
    #
    # With the default swapper the face arrives ~17% larger than the prior
    # expects and the eyes sit 31 px too high. That matters more here than for
    # a plain CNN restorer: CodeFormer's codebook lookup is a discrete nearest
    # neighbour into a learned dictionary, so an off-distribution input
    # retrieves the wrong entries. ProcessMgr re-warps to this template when
    # `enhancer_align` is on. None = "leave the crop alone".
    model_template = 'ffhq_512'
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        # A precision switch, not a second processor: same graph, same `w`
        # input, same pre/post — only the weights differ. Measured on an
        # RTX 4070 / CUDA at 512: fp32 162.9 ms/call, fp16 102.0 ms/call
        # (1.60x), with a mean output difference of about 3/255.
        want_fp16 = bool(plugin_options.get('fp16', False))
        if self.plugin_options is not None and self.fp16 != want_fp16:
            self.Release()

        self.plugin_options = plugin_options
        self.fp16 = want_fp16
        if self.model_codeformer is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            # conditional_download saves under the URL's basename, so this has
            # to be the name as published, dots and all.
            name = 'codeformer.fp16.onnx' if want_fp16 else 'CodeFormerv0.1.onnx'
            model_path = resolve_relative_path(f'../models/CodeFormer/{name}')
            opts = None
            if want_fp16:
                # This export trips ORT's SimplifiedLayerNormFusion at the
                # default ORT_ENABLE_ALL and the session then fails to build at
                # all — but only on the CPU provider, so it would look like an
                # enhancer that works until the day someone falls back to CPU.
                # EXTENDED builds everywhere and was measured at the speed
                # above; going lower would give the saving back.
                opts = onnxruntime.SessionOptions()
                opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            self.model_codeformer = onnxruntime.InferenceSession(model_path, opts, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_codeformer.get_inputs()
            self.model_outputs = self.model_codeformer.get_outputs()
            self.in_dtype = np.float16 if 'float16' in self.model_inputs[0].type else np.float32


    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        input_size = temp_frame.shape[1]
        # preprocess
        temp_frame = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        fallback_bgr = temp_frame   # resized input, kept for the non-finite guard
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)

        # Fresh io_binding per call: a shared binding is not thread-safe (under
        # the CUDA provider enhancers run concurrently, unlike TensorRT which is
        # serialised by the global GPU lock). Each input is bound exactly once
        # per binding, with a consistent dtype (the fidelity weight w must be
        # float64 for this ONNX export).
        io_binding = self.model_codeformer.io_binding()
        io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame.astype(self.in_dtype))
        cf_fidelity = getattr(roop.globals, 'codeformer_fidelity', 0.5)
        io_binding.bind_cpu_input(self.model_inputs[1].name, np.array([cf_fidelity], dtype=np.float64))
        io_binding.bind_output(self.model_outputs[0].name, self.devicename)
        self.model_codeformer.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        # float32 regardless of the model's precision — every step below
        # (clip, rescale, cvtColor) is written for it, and cv2 rejects float16.
        result = np.asarray(ort_outs[0][0], dtype=np.float32)
        del ort_outs

        # np.clip does not remove NaN and uint8(NaN) is 0, so a single
        # overflowed value paints black and a saturated graph paints a black
        # FACE — silently. See enhance_common.is_usable. Especially worth
        # having on the fp16 tier, which is a half-precision graph.
        if not is_usable(result):
            print("[CodeFormer] non-finite output — using unenhanced frame "
                  "(FP16 overflow? try the fp32 tier)")
            return sized(fallback_bgr.astype(np.uint8), input_size)

        # post-process
        result = result.transpose((1, 2, 0))

        un_min = -1.0
        un_max = 1.0
        result = np.clip(result, un_min, un_max)
        result = (result - un_min) / (un_max - un_min)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = (result * 255.0).round()
        return sized(result.astype(np.uint8), input_size)


    def Release(self):
        del self.model_codeformer
        self.model_codeformer = None

