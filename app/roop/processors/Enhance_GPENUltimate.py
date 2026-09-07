"""GPEN Ultimate: razor-sharp face restoration with forced alignment and anti-halo clarity."""

from roop.processors.Enhance_GPEN import Enhance_GPEN
from roop.processors.enhance_common import enhance_gpen_ultimate, inject_reference_detail


class Enhance_GPENUltimate(Enhance_GPEN):
    """An ultimate quality profile built on GPEN-512 with forced FFHQ alignment.

    Features:
    - Forced alignment with original face geometry via target keypoints.
    - Native 512px resolution (matching swap crop, avoiding 256px blur).
    - Dedicated eye clarity boost with anti-halo bounding (zero halos around eyes).
    - Bilateral edge-preserving detail transfer and anti-halo sharpening for crisp skin texture.
    - Pooled multi-context TensorRT inference.
    """

    processorname = 'gpen_ultimate'
    force_align = True
    model_template = 'ffhq_512'

    def Initialize(self, plugin_options: dict):
        options = dict(plugin_options)
        size = int(options.get("size", 512))
        options.update({"size": size, "profile": "ultimate"})
        super().Initialize(options)

    def Run(self, source_faceset, target_face, temp_frame):
        reference = temp_frame
        result, scale_factor = super().Run(source_faceset, target_face, temp_frame)
        result = enhance_gpen_ultimate(result, reference, target_face=target_face)
        return result, scale_factor

