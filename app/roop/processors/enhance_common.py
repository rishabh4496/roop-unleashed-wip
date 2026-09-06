"""Shared post-processing contract for the face restorers.

Every enhancer here ends the same three lines — clip to [-1, 1], rescale to
[0, 255], cast to uint8 — and hands back `(frame, scale_factor)`. Two things
about that ending are traps, and both were found the expensive way in GPEN
before being written down here.
"""

import cv2
import numpy as np


_CRISP_KERNEL = np.array(
    [[0.0, -1.0, 0.0],
     [-1.0, 5.0, -1.0],
     [0.0, -1.0, 0.0]], dtype=np.float32)


def is_usable(result):
    """False when the model returned anything non-finite.

    `np.clip` does NOT remove NaN — it propagates it — and `uint8(NaN)` is 0.
    So a single overflowed value becomes a black pixel and a saturated graph
    becomes a completely black face, with no exception, no warning, and a
    perfectly normal-looking `(512, 512, 3) uint8` on the way out. Verified:

        np.clip(nan, -1, 1)                  -> nan   (inf clips fine, nan does not)
        np.full(..., nan) -> post -> uint8   -> every value 0

    This is not hypothetical here. GPEN's 1024/2048 weights overflow in FP16
    under TensorRT and painted exactly that, which is why GPEN grew a guard;
    the frame upscaler hit the same thing (ESRGAN x4 goes black under TRT
    FP16). Any enhancer running FP16 on a graph nobody has stress-tested is one
    overflow away from it, and a black face reads as "the app is broken"
    rather than "this model overflowed".
    """
    return bool(np.isfinite(result).all())


def sized(result, input_size):
    """`(frame, scale_factor)` in the form paste_upscale expects.

    scale_factor is `result_width / input_size` as an INTEGER, because
    paste_upscale multiplies the paste matrix by it. That is fine while the
    model output is the same size as the crop or larger (512→1, 1024→2,
    2048→4), but a model SMALLER than the crop gives int(256/512) = 0, which
    collapses the paste matrix to zero and blanks the face.

    So a downscaling model is resized back to the crop size here and reports 1.
    The saving that tier exists for is in the network, not in carrying a
    smaller buffer through the paste — and an INTER_CUBIC upsample of a 256px
    crop costs a fraction of what the 512px net would have.
    """
    if result.shape[1] < input_size:
        result = cv2.resize(result, (input_size, input_size),
                            interpolation=cv2.INTER_CUBIC)
        return result, 1
    return result, max(1, int(result.shape[1] / input_size))


def inject_reference_detail(enhanced, reference, strength=0.0, crispness=0.0):
    """Return *enhanced* with a fast, registered detail-preserving finish.

    Restorers are deliberately conservative about texture: that avoids inventing
    pores, but can also make a swapped face look waxy.  The reference is the
    already-swapped crop, not the untouched target, so this operation preserves
    the new identity while bringing back genuine edges, stubble and fine scene
    texture.  Only a zero-mean high-pass residual is transferred; colour,
    lighting and face geometry remain owned by the model output.

    The operation is intentionally CPU-cheap (one resize when needed and at
    one small Gaussian blur on the existing face crop). It adds no neural
    inference, TensorRT engine, or full-frame pass. Neutral strengths are strict
    no-ops for callers that want the original model output.
    """
    try:
        amount = min(1.0, max(0.0, float(strength)))
    except (TypeError, ValueError):
        amount = 0.0
    try:
        edge_amount = min(1.0, max(0.0, float(crispness)))
    except (TypeError, ValueError):
        edge_amount = 0.0
    if ((amount <= 0.0 and edge_amount <= 0.0)
            or enhanced is None or reference is None):
        return enhanced
    if getattr(enhanced, 'ndim', 0) != 3 or getattr(reference, 'ndim', 0) != 3:
        return enhanced
    if enhanced.shape[2] != 3 or reference.shape[2] != 3:
        return enhanced

    ref = reference
    if ref.shape[:2] != enhanced.shape[:2]:
        ref = cv2.resize(ref, (enhanced.shape[1], enhanced.shape[0]),
                         interpolation=cv2.INTER_CUBIC)

    base = enhanced.astype(np.float32)

    # Scale the radius with the output, keeping the effect consistent for a
    # 256px GPEN output and a 512px RestoreFormer++ output.
    sigma = max(0.8, enhanced.shape[1] / 512.0 * 1.1)
    low = cv2.GaussianBlur(ref, (0, 0), sigmaX=sigma)
    high = ref.astype(np.float32) - low.astype(np.float32)
    # Compression ringing and isolated sensor noise should never become a
    # visible halo. Genuine facial edges remain well inside this clamp.
    high = np.clip(high, -40.0, 40.0)
    out = base + high * amount
    if edge_amount > 0.0:
        # Limit output sharpening to real reference edges. This keeps smooth
        # skin clean while restoring eyes, lips, lashes and hair structure.
        ref_luma_high = (0.114 * high[:, :, 0]
                         + 0.587 * high[:, :, 1]
                         + 0.299 * high[:, :, 2])
        # The swapped reference can be softer than the restored output, so its
        # residual alone is too strict a gate for already-visible eyes/lips.
        # Keep its signal, but allow the output's own local edge response to
        # open the finish without sharpening flat skin or background pixels.
        ref_gate = np.clip((np.abs(ref_luma_high) - 0.35) / 4.0, 0.0, 1.0)
        # Reuse the registered high-pass for the cheap texture lift, and use a
        # single 3x3 output sharpen for true model-edge crispness. There is no
        # second blur and no full-frame operation in this path.
        sharpened = cv2.filter2D(np.clip(out, 0.0, 255.0).astype(np.uint8),
                                 -1, _CRISP_KERNEL)
        output_high = sharpened.astype(np.float32) - out
        output_gate = np.clip((np.max(np.abs(output_high), axis=2) - 2.0)
                              / 10.0, 0.0, 1.0)
        gate = np.maximum(ref_gate, output_gate)
        out += output_high * (edge_amount * gate[:, :, None])
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
