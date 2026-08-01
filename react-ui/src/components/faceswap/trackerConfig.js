// Single source of truth for the Slider Tracker bar.
//
// These numbers used to live in four places — TRACKER_SLIDERS' defaultVal, the
// 'Default' built-in preset, resetTrackerSliders() and the bypass object inside
// buildPreviewPayload() — and had already drifted apart. Deriving the two value
// maps from the one slider list keeps them honest.
//
// Note that DEFAULT and BYPASS are deliberately NOT the same thing:
//   DEFAULT — the value the slider ships at ("↺ Reset", the "def:" pip).
//   BYPASS  — the value that makes the stage a no-op ("Slider Effect: OFF").
// For jaw reshape and flicker reduction the default is 0.5 but the no-op is 0,
// so collapsing them would silently turn "bypass" into "half strength".

export const TRACKER_SLIDERS = [
  {
    key: 'blend_ratio',
    label: 'Original / Enhanced Blend',
    min: 0,
    max: 1,
    step: 0.01,
    defaultVal: 0.8,
    bypassVal: 0.8,
    format: (v) => Number(v ?? 0.8).toFixed(2),
    info: 'Blends between original target face and enhanced swapped result.',
  },
  {
    key: 'detail_transfer_strength',
    label: 'Skin Detail Transfer',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0,
    bypassVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: "Transfers target's original skin texture (pores, grain, stubble) onto swapped face.",
  },
  {
    key: 'expression_restore_strength',
    label: 'Expression Restore',
    min: 0,
    max: 2,
    step: 0.05,
    defaultVal: 0,
    bypassVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: "Re-applies target's original expression using LivePortrait model.",
  },
  {
    key: 'face_mask_blend',
    label: 'Face Mask Edge Blend',
    min: 0,
    max: 200,
    step: 1,
    defaultVal: 20,
    bypassVal: 20,
    format: (v) => `${Math.round(v ?? 20)}px`,
    info: 'Feathering softness around the border of the face mask.',
  },
  {
    key: 'max_face_distance',
    label: 'Max Face Distance',
    min: 0.01,
    max: 1,
    step: 0.01,
    // Mid-gap. Measured on a hard clip: the same person stayed under ~0.66,
    // different people sat at ~0.93-1.07. 0.75 clears the same-person ceiling
    // with headroom for a worse frame than that clip contained, while staying
    // well under stranger range. See procmgr_runtime._TRACK_VETO_DIST.
    defaultVal: 0.75,
    bypassVal: 0.75,
    format: (v) => Number(v ?? 0.75).toFixed(2),
    info: 'Cosine distance (0-2), NOT a similarity: higher is MORE permissive. '
        + 'Same person measured under ~0.66, different people ~0.93-1.07.',
  },
  {
    key: 'num_swap_steps',
    label: 'Swapping Steps',
    min: 1,
    max: 5,
    step: 1,
    defaultVal: 1,
    bypassVal: 1,
    format: (v) => `${Math.round(v ?? 1)}×`,
    info: 'Swap iteration passes. Higher values increase source likeness.',
  },
  {
    key: 'jaw_reshape_strength',
    label: 'Jaw Reshape Strength',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0.5,
    bypassVal: 0,
    format: (v) => Number(v ?? 0.5).toFixed(2),
    info: 'Blends source jaw & chin contours onto target face.',
  },
  {
    key: 'stabilize_enhancer_strength',
    label: 'Flicker Reduction',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0.5,
    bypassVal: 0,
    format: (v) => Number(v ?? 0.5).toFixed(2),
    info: 'Temporal stabilization strength for video frames.',
  },
];

export const TRACKER_DEFAULT_VALUES = Object.fromEntries(
  TRACKER_SLIDERS.map((s) => [s.key, s.defaultVal])
);

export const TRACKER_BYPASS_VALUES = Object.fromEntries(
  TRACKER_SLIDERS.map((s) => [s.key, s.bypassVal])
);
