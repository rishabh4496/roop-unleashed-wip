/**
 * TypeScript definitions and schemas for the React frontend client.
 */

export type EnhancerType =
  | 'none'
  | 'gpen_realistic'
  | 'ultramax'
  | 'gfpgan'
  | 'codeformer'
  | 'codeformer_fp16'
  | 'dmdnet'
  | 'gpen'
  | 'gpen_256'
  | 'gpen_1024'
  | 'gpen_2048'
  | 'gpen_ultimate'
  | 'restoreformer++'
  | 'restore_ultra'
  | 'keep';

export type EnhancerDisplayName =
  | 'None'
  | 'GPEN Realistic'
  | 'UltraMax'
  | 'Restoreformer++'
  | 'Restore Ultra'
  | 'GPEN Ultimate'
  | 'GPEN'
  | 'GPEN 256'
  | 'GPEN 1024'
  | 'GPEN 2048'
  | 'Codeformer'
  | 'Codeformer (fp16)'
  | 'GFPGAN'
  | 'DMDNet'
  | 'KEEP (sidecar)';

export interface EnhancerSettings {
  enhancer_type: EnhancerType | string;
  enhancer_blend: number; // 0.0 to 1.0, default 0.85
  codeformer_fidelity?: number;
  enhancer_align?: boolean;
  color_match_after_enhance?: boolean;
}

export interface ProcessVideoRequest {
  enhancer_type?: EnhancerType | string;
  enhancer_blend?: number; // default 0.85
  enhancer?: EnhancerDisplayName | string;
  selected_enhancer?: EnhancerDisplayName | string;
  blend_ratio?: number;
  detection?: string;
  output_method?: string;
  video_method?: string;
  upscale?: string;
  mask_engine?: string;
  mask_engine_2?: string;
  clip_text?: string;
  sam2_model_size?: string;
  track_identities?: boolean;
  autorotate?: boolean;
  face_distance?: number;
  num_swap_steps?: number;
  swap_model?: string;
  face_mapping?: any[];
  imagemask?: string | null;
  [key: string]: any;
}

export interface PreviewRequest {
  index?: number;
  frame?: number;
  fake_preview?: boolean;
  enhancer_type?: EnhancerType | string;
  enhancer_blend?: number;
  enhancer?: EnhancerDisplayName | string;
  blend_ratio?: number;
  codeformer_fidelity?: number;
  detection?: string;
  face_distance?: number;
  swap_model?: string;
  mask_engine?: string;
  mask_engine_2?: string;
  clip_text?: string;
  [key: string]: any;
}

export interface FaceSwapState {
  selected_enhancer: EnhancerDisplayName | string;
  enhancer_type?: EnhancerType | string;
  enhancer_blend?: number;
  blend_ratio: number;
  codeformer_fidelity: number;
  max_face_distance: number;
  face_detection_mode: string;
  swap_model: string;
  subsample_upscale: string;
  num_swap_steps: number;
  [key: string]: any;
}

export interface EnhanceRequest {
  enhancer_type?: EnhancerType | string;
  enhancer_blend?: number;
  image?: string;
  frame_data?: string;
  kps?: number[][];
  simulate_oom?: boolean;
  [key: string]: any;
}

export interface EnhanceResponse {
  success: boolean;
  enhanced_frame?: string;
  image?: string;
  enhancer_type?: string;
  enhancer_blend?: number;
  width?: number;
  height?: number;
  error?: string;
  message?: string;
  error_type?: string;
  detail?: string;
}

