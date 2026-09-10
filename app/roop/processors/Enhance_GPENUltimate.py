"""GPEN Ultimate quality profile.

GPU execution, pooling, and the finishing pass live in the shared GPEN
processor so there is exactly one inference and one Ultimate finish per face.
"""

from roop.processors.Enhance_GPEN import Enhance_GPEN


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
