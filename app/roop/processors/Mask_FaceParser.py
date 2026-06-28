import os
import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download, cuda_only_providers


# BiSeNet (yakhyo/face-parsing, resnet18) trained on CelebAMask-HQ — 19 classes:
#  0 background  1 skin     2 l_brow  3 r_brow  4 l_eye   5 r_eye   6 eye_g(glasses)
#  7 l_ear       8 r_ear    9 ear_r   10 nose   11 mouth  12 u_lip  13 l_lip
#  14 neck       15 neck_l  16 cloth  17 hair   18 hat
#
# The swap region is the inner face: skin + brows + eyes + nose + lips/mouth.
# Everything else — hair, hat, glasses, ears, neck, cloth, background — is kept
# from the original, so part boundaries (hair fringe over the forehead, glasses,
# ears) are excluded precisely. This is sharper than XSeg's coarse whole-face
# blob at those boundaries; XSeg, however, is trained for arbitrary occluders
# (hands) which this model has no class for — hence offered as a separate engine.
_FACE_CLASSES = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13], dtype=np.int64)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MODEL_URL = 'https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx'
_MODEL_FILE = 'resnet18.onnx'


class Mask_FaceParser():
    plugin_options: dict = None

    model = None

    processorname = 'mask_faceparser'
    type = 'mask'
    # Built on thread-safe providers (CUDA/CPU, no TensorRT) so ProcessMgr can run
    # the mask concurrently across worker threads without the global GPU lock.
    threadsafe = True

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [_MODEL_URL])
            model_path = os.path.join(model_dir, _MODEL_FILE)
            onnxruntime.set_default_logger_severity(3)
            self.model = onnxruntime.InferenceSession(
                model_path, None, providers=cuda_only_providers(roop.globals.execution_providers))
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

    def Run(self, img1, keywords: str) -> Frame:
        # img1 is the aligned face crop (BGR uint8). Returned mask matches the
        # other engines' convention: 1.0 = keep ORIGINAL (exclude from swap),
        # 0.0 = use the swapped pixels. process_mask resizes it to the crop.
        resized = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        blob = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

        ort_outs = self.model.run([self.output_name], {self.input_name: blob})
        labels = ort_outs[0][0].argmax(0)                          # (512,512) class ids
        face = np.isin(labels, _FACE_CLASSES).astype(np.float32)   # 1 inside swap region
        # Soften edges so the blend matches the smooth XSeg output.
        face = cv2.GaussianBlur(face, (0, 0), sigmaX=3)
        face = np.clip(face, 0.0, 1.0)
        # invert: keep original everywhere outside the face region
        return (1.0 - face).astype(np.float32)

    def Release(self):
        del self.model
        self.model = None
