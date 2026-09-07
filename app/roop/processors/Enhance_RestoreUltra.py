"""Restore Ultra: ultra-high-definition RestoreFormer++ with forced alignment and anti-halo clarity."""

from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus
from roop.processors.enhance_common import enhance_restore_ultra, inject_reference_detail


class Enhance_RestoreUltra(Enhance_RestoreFormerPPlus):
    """An ultra-high-definition fidelity profile over the RestoreFormer++ weights.

    Features:
    - Forced FFHQ alignment based on original target face keypoints, guaranteeing
      exact anatomical feature registration with original faces.
    - Dedicated eye clarity boost with anti-halo bounding, producing crystal-clear,
      expressive eyes with rich iris definition and natural catchlights (zero halo rings).
    - Edge-preserving fine-line sharpening for eyelashes, eyebrows, and lip borders
      without amplifying noise or creating plastic artifacts.
    - Pooled multi-context TensorRT execution for concurrent worker throughput.
    """

    processorname = 'restore_ultra'
    force_align = True
    model_template = 'ffhq_512'

    def Run(self, source_faceset, target_face, temp_frame):
        reference = temp_frame
        result, scale_factor = super().Run(source_faceset, target_face, temp_frame)
        result = enhance_restore_ultra(result, reference, target_face=target_face)
        return result, scale_factor

