class ProcessOptions:

    def __init__(self, processordefines:dict, face_distance,  blend_ratio, swap_mode, selected_index, masking_text, imagemask, num_steps, subsample_size, show_face_area, restore_original_mouth, show_mask=False, use_3d_recon=False,
                 use_source_bank=False, use_frontalization=False, frontalization_threshold=25.0, swap_model='inswapper',
                 stabilize_face=None, stabilize_method=None, stabilize_min_cutoff=None, stabilize_beta=None,
                 stabilize_enhancer=None, stabilize_enhancer_strength=None, temporal_smooth_strength=None):
        self.processors = processordefines
        self.face_distance_threshold = face_distance
        self.blend_ratio = blend_ratio
        self.swap_mode = swap_mode
        self.selected_index = selected_index
        self.masking_text = masking_text
        self.imagemask = imagemask
        self.num_swap_steps = num_steps
        self.show_face_area_overlay = show_face_area
        self.show_face_masking = show_mask
        self.subsample_size = subsample_size
        self.restore_original_mouth = restore_original_mouth
        self.max_num_reuse_frame = 15
        # 3D source pose matching
        self.use_3d_recon = use_3d_recon
        # Multi-angle source bank (Option 1)
        self.use_source_bank = use_source_bank
        # Target frontalization (Option 2)
        self.use_frontalization = use_frontalization
        self.frontalization_threshold = frontalization_threshold
        self.swap_model = swap_model

        import roop.globals
        cfg = getattr(roop.globals, 'CFG', None)

        # One Euro temporal stabilization of face keypoints (video only)
        self.stabilize_face = (getattr(cfg, 'stabilize_face', True) if cfg is not None else True) if stabilize_face is None else stabilize_face
        self.stabilize_method = (getattr(cfg, 'stabilize_method', 'one_euro') if cfg is not None else 'one_euro') if stabilize_method is None else stabilize_method
        self.stabilize_min_cutoff = (getattr(cfg, 'stabilize_min_cutoff', 0.05) if cfg is not None else 0.05) if stabilize_min_cutoff is None else stabilize_min_cutoff
        self.stabilize_beta = (getattr(cfg, 'stabilize_beta', 0.02) if cfg is not None else 0.02) if stabilize_beta is None else stabilize_beta
        # One Euro temporal smoothing of the enhancer output (anti-flicker)
        self.stabilize_enhancer = (getattr(cfg, 'stabilize_enhancer', True) if cfg is not None else True) if stabilize_enhancer is None else stabilize_enhancer
        self.stabilize_enhancer_strength = (getattr(cfg, 'stabilize_enhancer_strength', 0.5) if cfg is not None else 0.5) if stabilize_enhancer_strength is None else stabilize_enhancer_strength
        # Temporal smoothing & anti-flickering strength (0.0 to 1.0, default 0.3)
        self.temporal_smooth_strength = (getattr(cfg, 'temporal_smooth_strength', 0.3) if cfg is not None else 0.3) if temporal_smooth_strength is None else temporal_smooth_strength