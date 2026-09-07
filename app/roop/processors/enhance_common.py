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

        # Anti-halo bounding: clamp output to local 3x3 min/max envelope
        # so ringing/halos around edges are strictly suppressed.
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        out8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
        local_min = cv2.erode(out8, kernel3).astype(np.float32)
        local_max = cv2.dilate(out8, kernel3).astype(np.float32)

        cand = out + output_high * (edge_amount * gate[:, :, None])
        out = np.clip(cand, local_min - 2.0, local_max + 2.0)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def apply_anti_halo_sharpen(img, amount=0.35, sigma=1.0, limit=2.5):
    """Edge-aware unsharp masking with local min/max bounding to eliminate halos.

    Operates on the Luminance (L) channel in LAB space to avoid color fringing.
    The local min/max envelope ensures that overshoots/halos around step edges
    (pupil vs sclera, face perimeter) cannot form.
    """
    if amount <= 0 or img is None:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    L_min = cv2.erode(L, kernel)
    L_max = cv2.dilate(L, kernel)

    blur = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma)
    high = L - blur

    # Soft coring to ignore sensor noise (|high| < 1.0) and soft saturation on large steps
    high_cored = np.sign(high) * np.maximum(0.0, np.abs(high) - 1.0)
    high_clamped = np.sign(high_cored) * np.minimum(
        np.abs(high_cored), 18.0 + 4.0 * np.tanh((np.abs(high_cored) - 18.0) / 4.0)
    )

    sharpened = L + float(amount) * high_clamped
    # Anti-halo bounding: clamp to local neighborhood range with tiny tolerance
    sharpened = np.clip(sharpened, np.maximum(0.0, L_min - limit), np.minimum(255.0, L_max + limit))

    lab[:, :, 0] = np.clip(sharpened, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_eyes_clarity(img, template='ffhq_512', kps=None, strength=0.55):
    """Specifically enhance eye clarity, iris contrast, and pupil depth with ZERO halo.

    Neural face restorers often leave eyes looking slightly milky or hazy.
    This applies local dynamic contrast (CLAHE on L-channel) in the eye orbits,
    combined with anti-halo bounded sharpening, so pupils are deep, irises are
    richly defined, catchlights are preserved, and NO white or dark halo rings form.
    """
    if strength <= 0 or img is None:
        return img
    h, w = img.shape[:2]
    if template == 'ffhq_512':
        cx1, cy1 = int(0.3769 * w), int(0.4686 * h)
        cx2, cy2 = int(0.6229 * w), int(0.4691 * h)
    else:
        cx1, cy1 = int(0.3419 * w), int(0.4616 * h)
        cx2, cy2 = int(0.6565 * w), int(0.4598 * h)

    rx = int(0.115 * w)
    ry = int(0.075 * h)

    eye_mask = np.zeros((h, w), dtype=np.float32)
    for cx, cy in [(cx1, cy1), (cx2, cy2)]:
        cv2.ellipse(eye_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)

    feather = int(max(rx, ry) * 0.4) | 1
    eye_mask = cv2.GaussianBlur(eye_mask, (feather, feather), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
    L_clahe = clahe.apply(np.clip(L, 0, 255).astype(np.uint8)).astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    L_min = cv2.erode(L_clahe, kernel)
    L_max = cv2.dilate(L_clahe, kernel)

    # Edge-preserving eye sharpening bounded by local min/max
    L_sharp = L_clahe + 0.35 * (L_clahe - cv2.GaussianBlur(L_clahe, (0, 0), sigmaX=1.0))
    L_sharp = np.clip(L_sharp, L_min, L_max)

    L_final = L * (1.0 - eye_mask * float(strength)) + L_sharp * (eye_mask * float(strength))
    lab[:, :, 0] = np.clip(L_final, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_gpen_ultimate(enhanced, reference, target_face=None,
                          strength=0.36, crispness=0.30, eye_boost=0.52):
    """GPEN Ultimate: razor-sharp, photorealistic enhancement.

    Features:
    1. Edge-preserving bilateral texture extraction from reference to inject
       authentic skin pores and micro-details without macro-edge artifacts.
    2. Dedicated eye clarity boost with anti-halo clamping (eyes pop with natural
       depth and catchlights; zero halo rings).
    3. Full-face anti-halo bounded crispness for sharp eyelashes, lips, and contours.
    """
    if enhanced is None:
        return enhanced
    if reference is None:
        reference = enhanced

    ref = reference
    if ref.shape[:2] != enhanced.shape[:2]:
        ref = cv2.resize(ref, (enhanced.shape[1], enhanced.shape[0]),
                         interpolation=cv2.INTER_CUBIC)

    # 1. Edge-preserving texture extraction (bilateral filter preserves boundaries)
    try:
        base_ref = cv2.bilateralFilter(ref, d=5, sigmaColor=22.0, sigmaSpace=4.0)
        detail_ref = ref.astype(np.float32) - base_ref.astype(np.float32)
        # Soft coring & saturation knee
        detail_ref = np.where(
            np.abs(detail_ref) <= 12.0,
            detail_ref,
            np.sign(detail_ref) * (12.0 + 3.0 * np.tanh((np.abs(detail_ref) - 12.0) / 3.0))
        )
        out = enhanced.astype(np.float32) + detail_ref * float(strength)
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    except Exception:
        out = enhanced

    # 2. Dedicated eye clarity enhancement with anti-halo clamping
    out = enhance_eyes_clarity(out, template='ffhq_512', strength=eye_boost)

    # 3. Full-face anti-halo sharpening for razor-sharp micro-textures
    out = apply_anti_halo_sharpen(out, amount=crispness, sigma=1.0, limit=2.5)
    return out


def enhance_restore_ultra(enhanced, reference, target_face=None,
                          strength=0.30, crispness=0.26, eye_clarity=0.48):
    """Restore Ultra: ultra-high-definition fidelity enhancement.

    Features:
    1. High-fidelity edge-preserving texture preservation.
    2. Pristine eye clarity with natural iris luminosity and catchlight
       preservation (strictly zero halo rings).
    3. Subtle anti-halo edge refinement for eyelashes, eyebrows, and lip borders
       without over-sharpening noise or plastic artifacts.
    """
    if enhanced is None:
        return enhanced
    if reference is None:
        reference = enhanced

    ref = reference
    if ref.shape[:2] != enhanced.shape[:2]:
        ref = cv2.resize(ref, (enhanced.shape[1], enhanced.shape[0]),
                         interpolation=cv2.INTER_CUBIC)

    # 1. Subtle bilateral texture injection
    try:
        base_ref = cv2.bilateralFilter(ref, d=5, sigmaColor=18.0, sigmaSpace=4.0)
        detail_ref = ref.astype(np.float32) - base_ref.astype(np.float32)
        detail_ref = np.where(
            np.abs(detail_ref) <= 10.0,
            detail_ref,
            np.sign(detail_ref) * (10.0 + 2.5 * np.tanh((np.abs(detail_ref) - 10.0) / 2.5))
        )
        out = enhanced.astype(np.float32) + detail_ref * float(strength)
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    except Exception:
        out = enhanced

    # 2. Ultra-definition eye clarity with anti-halo bounding
    out = enhance_eyes_clarity(out, template='ffhq_512', strength=eye_clarity)

    # 3. Fine-line edge refinement (eyelashes, eyebrows, lips)
    out = apply_anti_halo_sharpen(out, amount=crispness, sigma=0.8, limit=2.0)
    return out

