"""GPEN Ultimate: pooled GPEN-BFR-256 with a fast detail-preserving finish."""

from roop.processors.Enhance_GPEN import Enhance_GPEN


class Enhance_GPENUltimate(Enhance_GPEN):
    """A named throughput/quality profile built on the GPEN 256 checkpoint.

    This is intentionally a profile over the published GPEN weights, not a
    second checkpoint: the gain comes from the 256px network, pooled inference
    contexts, and registered high-frequency detail from the swapped crop.
    """

    processorname = 'gpen_ultimate'

    def Initialize(self, plugin_options: dict):
        options = dict(plugin_options)
        options.update({"size": 256, "profile": "ultimate"})
        super().Initialize(options)
