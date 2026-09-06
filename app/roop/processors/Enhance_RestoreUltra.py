"""Restore Ultra: pooled RestoreFormer++ with a detail-preserving finish."""

from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus
from roop.processors.enhance_common import inject_reference_detail


class Enhance_RestoreUltra(Enhance_RestoreFormerPPlus):
    """A quality-oriented profile over the existing RestoreFormer++ weights.

    RestoreFormer++ already owns the pooled TensorRT path.  This profile keeps
    that path and adds a small high-frequency transfer from the swapped input,
    which restores genuine source detail without reintroducing target colour or
    low-frequency identity information.
    """

    processorname = 'restore_ultra'

    def Run(self, source_faceset, target_face, temp_frame):
        reference = temp_frame
        result, scale_factor = super().Run(source_faceset, target_face, temp_frame)
        result = inject_reference_detail(result, reference,
                                          strength=0.42, crispness=0.28)
        return result, scale_factor
