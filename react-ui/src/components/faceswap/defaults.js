// Baked-in defaults for every setting editable inside the Face Swap tab.
// Snapshot of the user's preferred configuration (taken 2026-07-12 from
// app/config.yaml). The "Reset defaults" button in FaceSwap.jsx merges this
// over the live settings and persists it to the backend CFG.
// Deliberately excludes global/Settings-tab keys (provider, threads, theme,
// output codec/quality, server options, perf knobs) so a reset never touches
// anything outside this tab.
export const FACESWAP_DEFAULTS = {
  // Swap settings
  swap_model: 'hyperswap_1c',
  face_detection_mode: 'Selected face',
  detector_engine: 'retinaface_r50',
  face_detector_size: '640',
  default_det_size: true,
  face_detector_threshold: 0.7,
  face_detector_nms: 0.4,
  refine_landmarks: true,
  rescue_small_faces: true,
  num_swap_steps: 1,
  selected_enhancer: 'Restoreformer++',
  codeformer_fidelity: 0.5,
  max_face_distance: 0.75,
  subsample_upscale: '128px',
  upscale_after_swap: true,
  upscale_model_after: 'esrganx2',
  interp_after_swap: 'off',
  color_transfer_mode: 'lct',
  blend_ratio: 1,

  // Masking parameters
  mask_engine: 'DFL XSeg',
  mask_clip_text: 'cup,hands,hair,banana',
  sam2_model_size: 'tiny',
  show_mask_offsets: false,
  mask_top: 0,
  mask_bottom: 0,
  mask_left: 0,
  mask_right: 0,
  face_mask_blend: 20,

  // Mouth & Angle math
  mouth_top_scale: 1,
  mouth_bottom_scale: 1,
  mouth_left_scale: 1,
  mouth_right_scale: 1,
  mouth_mask_blend: 10,
  use_3d_recon: true,
  use_source_bank: true,
  use_frontalization: false,
  frontalization_threshold: 25,
  jaw_reshape: false,
  jaw_reshape_strength: 0.5,
  detail_transfer_strength: 0,
  // Expression restore is an editable Face Swap control and a heavy GPU stage,
  // but was missing here — so "Reset defaults" left it at whatever it was.
  expression_restore_strength: 0,
  expression_restore_region: 'all',

  // Video parameters
  video_swapping_method: 'In-Memory processing',
  no_face_action: 'Use untouched original frame',
  temporal_detection: true,
  vr_mode: false,
  stabilize_enhancer: true,
  stabilize_enhancer_strength: 0.5,

  // System options
  autorotate_faces: true,
  skip_audio: false,
  keep_frames: false,
  wait_after_extraction: false,

  // Enhancements
  track_identities: true,
  stabilize_face: true,
  stabilize_method: 'one_euro',
  stabilize_min_cutoff: 0.05,
  stabilize_beta: 0.02,
  restore_original_mouth: false,

  // Output
  output_method: 'File',
};
