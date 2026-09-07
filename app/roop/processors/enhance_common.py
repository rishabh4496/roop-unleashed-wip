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


# ── Precomputed tables for the soft-knee detail curve ─────────────────────
# The knee is evaluated on `ref - bilateral(ref)`, and both operands are uint8
# — so the difference is an INTEGER in [-255, 255] and the curve can only ever
# take 511 distinct values. It was written with np.where, which computes BOTH
# branches over every pixel: a full sign(), abs() and tanh() pass over
# 512x512x3 to decide the ~1% of pixels that are actually over the knee.
# Measured on a 512px crop with cv2 single-threaded (which is how the workers
# run it, see ProcessMgr): 12.5 ms for the knee against 2.2 ms for the
# bilateral filter it post-processes.
#
# The table is built with the identical float32 expression, so `lut[d + 255]`
# is bit-for-bit what np.where returned, and the strength multiply is folded in
# because it is a per-call constant. Keyed by the curve's own parameters so the
# GPEN (12.0 / 3.0) and Restore (10.0 / 2.5) profiles each get their own.
_KNEE_LUTS = {}


def _knee_lut(threshold, softness, strength):
    key = (float(threshold), float(softness), float(strength))
    lut = _KNEE_LUTS.get(key)
    if lut is None:
        d = np.arange(-255, 256, dtype=np.float32)
        lut = np.where(
            np.abs(d) <= threshold,
            d,
            np.sign(d) * (threshold + softness
                          * np.tanh((np.abs(d) - threshold) / softness))
        ) * float(strength)
        lut = np.ascontiguousarray(lut, dtype=np.float32)
        _KNEE_LUTS[key] = lut
    return lut


def _inject_bilateral_detail(enhanced, ref, sigma_color, threshold, softness,
                             strength):
    """`enhanced + knee(ref - bilateral(ref)) * strength`, through that table.

    Bit-identical to the np.where form it replaces (verified elementwise over
    the whole 511-value domain), and now independent of how much of the crop
    sits over the knee.
    """
    base_ref = cv2.bilateralFilter(ref, d=5, sigmaColor=sigma_color,
                                   sigmaSpace=4.0)
    idx = cv2.subtract(ref, base_ref, dtype=cv2.CV_16S)
    idx += 255
    out = enhanced.astype(np.float32)
    out += _knee_lut(threshold, softness, strength).take(idx)
    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.uint8)


# The eye ellipse pair, its feather and the box they live in depend only on the
# crop size and the template, and the strength is a per-profile constant — so
# the whole `eye_mask * strength` term is the same array on every call for a
# given enhancer. Building it per face cost an ellipse rasterisation, a Gaussian
# blur and two more full-size multiplies for a result that never changed.
_EYE_MASK_CACHE = {}


# The two eye keypoints of the warp template the enhancer aligned this crop to
# (the same numbers as face_util.WARP_TEMPLATES), so on that crop they land on
# the eyes by construction — force_align guarantees the crop is in one of these
# two spaces before the finish runs.
_EYE_TEMPLATE_KPS = {
    'ffhq_512':       ((0.37691676, 0.46864664), (0.62285697, 0.46912813)),
    'arcface_112_v2': ((0.34191607, 0.46157411), (0.65653393, 0.45983393)),
}


def _eye_region(h, w, template, strength):
    """`(weight, (x0, y0, x1, y1), sigma)` — the feathered eye mask, the box it
    lives in, and the sharpening radius for this crop size. None when the crop
    is too small to place a pair of eyes on.
    """
    key = (h, w, template, float(strength))
    cached = _EYE_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    centres = [(fx * w, fy * h) for fx, fy
               in _EYE_TEMPLATE_KPS.get(template,
                                        _EYE_TEMPLATE_KPS['arcface_112_v2'])]
    # Radii from the template's OWN interocular distance, at the fractions
    # apply_eyes_area measured an eye at (0.21x IOD across, 0.13x tall). The
    # fixed `0.115 w` x `0.075 h` this replaces was 2.3x an eye wide and 2.4x
    # its height on ffhq_512: the two ellipses met over the nose bridge and
    # reached the temples, so once the feather was added the "eye" region was a
    # 264 x 98 px band straight across the middle of a 512 crop — brows,
    # sockets and upper cheeks included. It also ignored the template it was
    # handed, sizing the much wider arcface pair identically.
    iod = abs(centres[1][0] - centres[0][0])
    rx = int(round(iod * 0.21))
    ry = int(round(iod * 0.13))
    if rx < 2 or ry < 2:
        return None

    feather = int(max(rx, ry) * 0.4) | 1
    pad = feather + 4
    x0 = max(0, int(min(c[0] for c in centres) - rx - pad))
    x1 = min(w, int(max(c[0] for c in centres) + rx + pad) + 1)
    y0 = max(0, int(min(c[1] for c in centres) - ry - pad))
    y1 = min(h, int(max(c[1] for c in centres) + ry + pad) + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for cx, cy in centres:
        cv2.ellipse(mask, (int(round(cx)) - x0, int(round(cy)) - y0),
                    (rx, ry), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)

    # 1.0 px at the 512-crop reference size, scaled with the eye so a 256 and a
    # 1024 crop get the same effect on screen.
    cached = (mask * float(strength), (x0, y0, x1, y1), max(0.8, rx / 26.0))
    _EYE_MASK_CACHE[key] = cached
    return cached


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

    # Soft coring to ignore sensor noise (|high| < 1.0) and soft saturation on
    # large steps.
    #
    # Written as magnitude-then-sign rather than the two sign()/minimum() passes
    # it replaces, because the saturation only ever binds above 18: for
    # |x| <= 18 the bound 18 + 4*tanh((|x|-18)/4) sits ABOVE |x| (their
    # difference is 4*(tanh(u) - u) >= 0 for u <= 0), so np.minimum returned
    # |x| unchanged there — and that is the overwhelming majority of a face
    # crop. Evaluating tanh over the whole 512x512 plane to establish it cost
    # 10.9 ms of a 36 ms post-process. The cut is taken at 17 rather than 18 so
    # the untouched branch is exactly untouched: at |x| = 17 the bound is
    # already 0.02 clear of it, which is many orders of magnitude more than a
    # float32 ulp, so no value that np.minimum would have altered can fall in
    # the cheap branch.
    mag = np.abs(high)
    mag -= 1.0
    np.maximum(mag, 0.0, out=mag)
    over = mag > 17.0
    if over.any():
        m = mag[over]
        mag[over] = np.minimum(m, 18.0 + 4.0 * np.tanh((m - 18.0) / 4.0))
    # copysign, not sign(): where the cored magnitude is 0 the old expression
    # produced sign(0) * 0 = 0 and this produces +-0.0, which adds identically.
    high_clamped = np.copysign(mag, high)

    sharpened = L + float(amount) * high_clamped
    # Anti-halo bounding: clamp to local neighborhood range with tiny tolerance
    sharpened = np.clip(sharpened, np.maximum(0.0, L_min - limit), np.minimum(255.0, L_max + limit))

    lab[:, :, 0] = np.clip(sharpened, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_eyes_clarity(img, template='ffhq_512', kps=None, strength=0.55):
    """Sharpen the eyes — lashes, lid line, iris and pupil edge — and nothing else.

    Neural face restorers leave eyes slightly milky, so lifting their detail
    back is worth doing. What this must NOT do is change the TONE of the region,
    and the first version did exactly that: it ran CLAHE over the whole crop's L
    channel and blended the result into a feathered ellipse pair. CLAHE is a
    local histogram equalisation — it redistributes a region's levels, so its
    output differs from the input by a large LOW-FREQUENCY term, not just by
    detail. Measured on the two frames this was reported from, CLAHE at the
    shipped settings (clipLimit 1.8, 4x4 tiles) moved L over the eye ellipses by
    +28.5 and +23.4 levels on average; at the shipped 0.48 blend that is a
    +12.5 / +10.5 level brightening painted into an oval and feathered at its
    rim. That is the halo the docstring promised could not form: periocular skin
    several levels lighter than the skin around it, with the mask's own blur as
    the only edge between them. The "anti-halo" clamp never saw it, because it
    bounded the sharpen against CLAHE's OWN erode/dilate — which says nothing
    about how far the tone had already moved from the input.

    So the tonal stage is gone. What remains is a high-pass, zero-mean by
    construction, clamped to the ORIGINAL L's 3x3 min/max envelope: a pixel can
    never leave the range its immediate neighbours already span, so there is no
    overshoot at any radius for the feather to spread into a ring. Catchlights
    and pupil depth survive because they ARE local edges; flat periocular skin,
    where the envelope is a couple of levels wide, is left alone.

    `kps` is accepted and unused — the crop is template-aligned, so the eye
    positions are known from the template rather than from the face.
    """
    if strength <= 0 or img is None or getattr(img, 'ndim', 0) != 3:
        return img
    h, w = img.shape[:2]
    region = _eye_region(h, w, template, strength)
    if region is None:
        return img
    weight, (x0, y0, x1, y1), sigma = region

    # Only the eye box is converted, filtered and written back. The old version
    # ran a CLAHE, an erode, a dilate and a blur across the whole 512 plane to
    # use 5% of it, and round-tripped every pixel through BGR->LAB->BGR — which
    # is lossy, so pixels the mask gave zero weight still came back a level or
    # two different from what was handed in.
    lab = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    L_min = cv2.erode(L, kernel)
    L_max = cv2.dilate(L, kernel)

    sharp = L + 0.9 * (L - cv2.GaussianBlur(L, (0, 0), sigmaX=sigma))
    np.clip(sharp, L_min, L_max, out=sharp)

    sharp -= L
    sharp *= weight
    sharp += L
    lab[:, :, 0] = np.clip(sharp, 0, 255).astype(np.uint8)

    out = img.copy()
    out[y0:y1, x0:x1] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


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

    # 1. Edge-preserving texture extraction (bilateral filter preserves
    #    boundaries), with the soft coring / saturation knee taken from the
    #    511-entry table — see _inject_bilateral_detail.
    try:
        out = _inject_bilateral_detail(enhanced, ref, sigma_color=22.0,
                                       threshold=12.0, softness=3.0,
                                       strength=strength)
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

    # 1. Subtle bilateral texture injection, with the soft knee taken from the
    #    511-entry table — see _inject_bilateral_detail.
    try:
        out = _inject_bilateral_detail(enhanced, ref, sigma_color=18.0,
                                       threshold=10.0, softness=2.5,
                                       strength=strength)
    except Exception:
        out = enhanced

    # 2. Ultra-definition eye clarity with anti-halo bounding
    out = enhance_eyes_clarity(out, template='ffhq_512', strength=eye_clarity)

    # 3. Fine-line edge refinement (eyelashes, eyebrows, lips)
    out = apply_anti_halo_sharpen(out, amount=crispness, sigma=0.8, limit=2.0)
    return out

