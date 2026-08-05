"""Lip-sync — regenerate the mouth region to match a driving audio track.

Hand-wired singleton like Expression_LivePortrait, NOT registered in
ProcessMgr.plugins: that registry is a single-select radio for enhancers/mask
engines (Run(source_faceset, target_face, temp_frame) -> (Frame, scale_factor)),
and this needs non-standard inputs (audio features, frame timing) that don't
fit that contract. See ProcessMgr._lipsync_restorer / process_face's
"Lip-sync (post-composite)" block.

Verified against TMElyralab/MuseTalk's own inference code (scripts/
realtime_inference.py) before writing this, rather than guessed:

  * Face crop: bounding-box (get_landmark_and_bbox), NOT a 5-point similarity
    transform — unlike this codebase's estimate_norm/align_crop family. Crop
    is resized to 256x256 with INTER_LANCZOS4.
  * Audio: encoded by a frozen whisper-tiny, chunked per video frame with a
    left/right padding window — not a raw mel-spectrogram (Wav2Lip-style).
  * Generator: the stable-diffusion-v1-4 UNet architecture, cross-attention
    conditioned on the audio embedding, operating in the latent space of
    sd-vae-ft-mse (Stability AI's standard SD VAE — a public checkpoint, not
    MuseTalk-specific).
  * Blend mask: face-parsing model output (get_image_prepare_material), not a
    fixed oval — see the note on Mask_FaceParser below.
  * License: MIT (code); weights usable for any purpose including commercial.

THE OPEN DECISION THIS FILE DOES NOT RESOLVE
----------------------------------------------
Every existing processor in this codebase (GFPGAN, CodeFormer, RestoreFormer++,
LivePortrait, the swappers) is onnxruntime-first: ONNX weights, SessionPool,
the _gpu_guard/TensorRT-serialisation story. MuseTalk has NO official ONNX
export — its weights are PyTorch (musetalk/pytorch_model.bin or unet.pth) plus
a diffusers-format VAE (sd-vae-ft-mse) plus openai-whisper's whisper-tiny.
Third-party ONNX conversions exist but their completeness/fidelity is
unverified from here.

That is a real dependency decision (torch + diffusers + openai-whisper as new
runtime deps, vs. depending on an unverified community ONNX port, vs.
reconsidering the model choice) with real install-size and maintenance
consequences, not something to silently pick. Initialize() raises rather than
guessing — the caller (ProcessMgr.process_face) already wraps this in
try/except and logs via bar_write, so turning the toggle on today safely
no-ops per frame with a clear log line instead of pasting in fabricated model
URLs that would 404.

Everything ELSE in this file (the crop-box math, the class shape, the
prepare/infer/finish split mirroring Expression_LivePortrait) does not depend
on that decision and is real.
"""

import cv2
import numpy as np

from roop.typing import Frame

# musetalk/pytorch_model.bin (or unet.pth for v1.5) + sd-vae-ft-mse +
# whisper-tiny — see the module docstring. No MODELS/_BASE dict yet: unlike
# every onnxruntime processor here, these are not single-file HTTPS downloads
# conditional_download can fetch as-is (the VAE and Whisper checkpoints are
# their own multi-file HF repos), and which files depends on the still-open
# torch-vs-ONNX-port decision above.
INPUT_SIZE = 256  # confirmed: cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)


def face_bbox_crop(frame: Frame, bbox) -> np.ndarray:
    """MuseTalk's own crop: a straight bounding-box crop (no landmark-fitted
    similarity transform), resized to INPUT_SIZE with Lanczos — matches
    get_landmark_and_bbox + the resize call in realtime_inference.py.

    Kept as a standalone function (not a method) so the crop math is testable
    without the model class — mirrors why frame_time/audio_index_for_frame
    live in roop.lipsync_audio rather than on the class that uses them.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)


class Lipsync_MuseTalk:
    processorname = 'lipsync_musetalk'
    type = 'lipsync'
    pool = None  # No pooling for v1 — see the class docstring below.

    def __init__(self):
        self._ready = False

    def Initialize(self, plugin_options: dict):
        """Deliberately not implemented: see the module docstring's "OPEN
        DECISION" section. Raising here (rather than downloading nothing and
        pretending) keeps the failure honest and immediate instead of a
        confusing crash three calls deeper in a fabricated ONNX session.
        """
        raise NotImplementedError(
            "Lip-sync (MuseTalk) is wired into the pipeline but its model "
            "backend is not implemented yet: MuseTalk ships PyTorch+diffusers"
            "+whisper weights with no official ONNX export, unlike every "
            "other model in this app. See Lipsync_MuseTalk.py's module "
            "docstring for what was verified against the upstream repo and "
            "what decision is still open.")

    def build_audio_cache(self, audio_source_path: str):
        """Extract + feature-encode the driving audio once per clip. Depends
        on the same undecided backend as Initialize() (Whisper inference)."""
        raise NotImplementedError("See Initialize().")

    def prepare(self, frame: Frame, target_face, audio_features):
        """CPU-side: crop the face bbox and pair it with its audio chunk.
        The crop math itself (face_bbox_crop) does not depend on the backend
        decision — only what happens to its output in infer() does.
        """
        bbox = getattr(target_face, 'bbox', None)
        if bbox is None:
            return None
        crop = face_bbox_crop(frame, bbox)
        if crop is None:
            return None
        return {"crop": crop, "audio_features": audio_features}

    def infer(self, prepared):
        """GPU-side: VAE encode -> UNet -> VAE decode. Not implemented — see
        Initialize()."""
        raise NotImplementedError("See Initialize().")

    def finish(self, raw, mouth_cutout_hint: np.ndarray) -> Frame:
        """CPU-side: decode the model's output back to a BGR mouth-region
        cutout sized for apply_mouth_area's resize step. Not implemented —
        see Initialize()."""
        raise NotImplementedError("See Initialize().")

    def Run(self, frame: Frame, target_face, audio_features) -> Frame:
        prepared = self.prepare(frame, target_face, audio_features)
        raw = self.infer(prepared)
        return self.finish(raw, None)

    def Release(self):
        self._ready = False
