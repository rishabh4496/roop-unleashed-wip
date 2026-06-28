import os
import yaml

# --- Make the TensorRT execution provider actually loadable on Windows ---
# onnxruntime advertises 'TensorrtExecutionProvider' as available even when its
# native runtime DLLs cannot be loaded. Loading onnxruntime_providers_tensorrt.dll
# requires BOTH of these to be resolvable at import time:
#   1) nvinfer_*.dll        -> shipped in the `tensorrt_libs` package folder
#   2) the CUDA/cuDNN runtime DLLs -> shipped inside torch's `lib` folder
# If either is missing, ORT fails with "LoadLibrary failed with error 126" and
# silently falls back to the CUDA provider (losing TensorRT acceleration).
#
# Since Python 3.8, Windows ignores PATH for dependent-DLL resolution and only
# searches directories registered via os.add_dll_directory(); the CUDA runtime
# additionally has to be *loaded* into the process, which importing torch does.
# settings is imported very early in every process (including spawned video
# workers), so doing this here fixes the silent CUDA fallback everywhere.
def _enable_tensorrt_runtime():
    dll_dirs = []
    try:
        import tensorrt
        trt_libs = os.path.join(os.path.dirname(os.path.dirname(tensorrt.__file__)), 'tensorrt_libs')
        if os.path.isdir(trt_libs):
            dll_dirs.append(trt_libs)
    except Exception:
        pass
    try:
        # Importing torch loads the CUDA/cuDNN runtime DLLs the TRT EP depends on.
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(torch_lib):
            dll_dirs.append(torch_lib)
    except Exception:
        pass
    for d in dll_dirs:
        try:
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(d)
        except Exception:
            pass
        os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')

_enable_tensorrt_runtime()
class Settings:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load()

    def default_get(_, data, name, default):
        value = default
        try:
            value = data.get(name, default)
        except:
            pass
        return value


    def load(self):
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except:
            data = None

        self.selected_theme = self.default_get(data, 'selected_theme', "Default")
        self.server_name = self.default_get(data, 'server_name', "")
        self.server_port = self.default_get(data, 'server_port', 0)
        self.server_share = self.default_get(data, 'server_share', False)
        self.output_image_format = self.default_get(data, 'output_image_format', 'png')
        self.output_video_format = self.default_get(data, 'output_video_format', 'mp4')
        self.output_video_codec = self.default_get(data, 'output_video_codec', 'libx264')
        self.video_quality = self.default_get(data, 'video_quality', 14)
        self.clear_output = self.default_get(data, 'clear_output', True)
        # Dynamically scale threads to saturate GPU without OOM
        default_threads = 3
        try:
            self.provider = self.default_get(data, 'provider', 'cuda')
            if self.provider in ['cuda', 'tensorrt']:
                import torch
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    import psutil
                    cores = psutil.cpu_count(logical=False) or 4
                    # 1 thread per 1.5GB VRAM is safe for face swapping, bounded by CPU cores
                    default_threads = int(min(max(2, cores - 1), max(2, vram_gb / 1.5)))
        except Exception:
            pass

        saved_threads = self.default_get(data, 'max_threads', -1)
        # Upgrade legacy default '2' to the auto-scaled value to fix CPU bottlenecks for existing users
        if saved_threads == 2 or saved_threads == -1:
            self.max_threads = default_threads
        else:
            self.max_threads = saved_threads

        # Prevent extreme CPU oversubscription by capping max_threads to logical CPU cores
        try:
            import psutil
            logical_cores = psutil.cpu_count(logical=True) or 4
            if self.max_threads > logical_cores:
                self.max_threads = logical_cores
        except Exception:
            pass
        
        self.memory_limit = self.default_get(data, 'memory_limit', 0)
        self.provider = self.default_get(data, 'provider', 'cuda')
        # TensorRT precision mode: 'fp32' | 'fp16' | 'mixed' (only used when provider == 'tensorrt')
        self.trt_precision = self.default_get(data, 'trt_precision', 'mixed')
        self.force_cpu = self.default_get(data, 'force_cpu', False)
        self.output_template = self.default_get(data, 'output_template', '{file}_{time}')
        self.use_os_temp_folder = self.default_get(data, 'use_os_temp_folder', False)
        self.output_show_video = self.default_get(data, 'output_show_video', True)
        self.launch_browser = self.default_get(data, 'launch_browser', False)
        self.max_face_distance = self.default_get(data, 'max_face_distance', 0.85)
        # Faceswap session settings
        self.face_detection_mode = self.default_get(data, 'face_detection_mode', 'All faces')
        # Face-detector input resolution: True = 640x640 (accurate, default),
        # False = 320x320 (~4x faster detection, may miss small/distant faces).
        self.default_det_size = self.default_get(data, 'default_det_size', True)
        self.num_swap_steps = self.default_get(data, 'num_swap_steps', 1)
        self.selected_enhancer = self.default_get(data, 'selected_enhancer', 'GPEN')
        self.subsample_upscale = self.default_get(data, 'subsample_upscale', '256px')
        self.blend_ratio = self.default_get(data, 'blend_ratio', 0.80)
        self.video_swapping_method = self.default_get(data, 'video_swapping_method', 'In-Memory processing')
        self.no_face_action = self.default_get(data, 'no_face_action', 'Retry rotated')
        self.vr_mode = self.default_get(data, 'vr_mode', False)
        self.autorotate_faces = self.default_get(data, 'autorotate_faces', True)
        self.skip_audio = self.default_get(data, 'skip_audio', False)
        self.keep_frames = self.default_get(data, 'keep_frames', False)
        self.wait_after_extraction = self.default_get(data, 'wait_after_extraction', False)
        self.output_method = self.default_get(data, 'output_method', 'File')
        self.mask_engine = self.default_get(data, 'mask_engine', 'DFL XSeg')
        self.mask_clip_text = self.default_get(data, 'mask_clip_text', 'cup,hands,hair,banana')
        self.show_mask_offsets = self.default_get(data, 'show_mask_offsets', False)
        self.restore_original_mouth = self.default_get(data, 'restore_original_mouth', False)
        self.mask_top = self.default_get(data, 'mask_top', 0.0)
        self.mask_bottom = self.default_get(data, 'mask_bottom', 0.0)
        self.mask_left = self.default_get(data, 'mask_left', 0.0)
        self.mask_right = self.default_get(data, 'mask_right', 0.0)
        self.face_mask_blend = self.default_get(data, 'face_mask_blend', 20.0)
        self.mouth_mask_blend = self.default_get(data, 'mouth_mask_blend', 10.0)
        self.mouth_top_scale = self.default_get(data, 'mouth_top_scale', 1.0)
        self.mouth_bottom_scale = self.default_get(data, 'mouth_bottom_scale', 1.0)
        self.mouth_left_scale = self.default_get(data, 'mouth_left_scale', 1.0)
        self.mouth_right_scale = self.default_get(data, 'mouth_right_scale', 1.0)
        # 3D source pose matching
        self.use_3d_recon = self.default_get(data, 'use_3d_recon', False)
        # Multi-angle source bank (Option 1)
        self.use_source_bank = self.default_get(data, 'use_source_bank', False)
        # Target frontalization (Option 2)
        self.use_frontalization = self.default_get(data, 'use_frontalization', False)
        self.frontalization_threshold = self.default_get(data, 'frontalization_threshold', 30.0)
        self.swap_model = self.default_get(data, 'swap_model', 'inswapper')
        # One Euro temporal face stabilization (video)
        self.stabilize_face = self.default_get(data, 'stabilize_face', False)
        self.stabilize_method = self.default_get(data, 'stabilize_method', 'one_euro')
        self.stabilize_min_cutoff = self.default_get(data, 'stabilize_min_cutoff', 0.05)
        self.stabilize_beta = self.default_get(data, 'stabilize_beta', 0.02)
        self.stabilize_enhancer = self.default_get(data, 'stabilize_enhancer', False)
        self.stabilize_enhancer_strength = self.default_get(data, 'stabilize_enhancer_strength', 0.5)





    def save(self):
        data = {
            'selected_theme': self.selected_theme,
            'server_name': self.server_name,
            'server_port': self.server_port,
            'server_share': self.server_share,
            'output_image_format' : self.output_image_format,
            'output_video_format' : self.output_video_format,
            'output_video_codec' : self.output_video_codec,
            'video_quality' : self.video_quality,
            'clear_output' : self.clear_output,
            'max_threads' : self.max_threads,
            'memory_limit' : self.memory_limit,
            'provider' : self.provider,
            'trt_precision' : self.trt_precision,
            'force_cpu' : self.force_cpu,
            'output_template' : self.output_template,
            'use_os_temp_folder' : self.use_os_temp_folder,
            'output_show_video' : self.output_show_video,
            'launch_browser': self.launch_browser,
            'max_face_distance': self.max_face_distance,
            # Faceswap session settings
            'face_detection_mode': self.face_detection_mode,
            'num_swap_steps': self.num_swap_steps,
            'selected_enhancer': self.selected_enhancer,
            'subsample_upscale': self.subsample_upscale,
            'blend_ratio': self.blend_ratio,
            'video_swapping_method': self.video_swapping_method,
            'no_face_action': self.no_face_action,
            'vr_mode': self.vr_mode,
            'autorotate_faces': self.autorotate_faces,
            'skip_audio': self.skip_audio,
            'keep_frames': self.keep_frames,
            'wait_after_extraction': self.wait_after_extraction,
            'output_method': self.output_method,
            'mask_engine': self.mask_engine,
            'mask_clip_text': self.mask_clip_text,
            'show_mask_offsets': self.show_mask_offsets,
            'restore_original_mouth': self.restore_original_mouth,
            'mask_top': self.mask_top,
            'mask_bottom': self.mask_bottom,
            'mask_left': self.mask_left,
            'mask_right': self.mask_right,
            'face_mask_blend': self.face_mask_blend,
            'mouth_mask_blend': self.mouth_mask_blend,
            'mouth_top_scale': self.mouth_top_scale,
            'mouth_bottom_scale': self.mouth_bottom_scale,
            'mouth_left_scale': self.mouth_left_scale,
            'mouth_right_scale': self.mouth_right_scale,
            # 3D source pose matching
            'use_3d_recon': self.use_3d_recon,
            # Multi-angle source bank
            'use_source_bank': self.use_source_bank,
            # Target frontalization
            'use_frontalization': self.use_frontalization,
            'frontalization_threshold': self.frontalization_threshold,
            # Swap model
            'swap_model': self.swap_model,
            # One Euro temporal face stabilization
            'stabilize_face': self.stabilize_face,
            'stabilize_method': self.stabilize_method,
            'stabilize_min_cutoff': self.stabilize_min_cutoff,
            'stabilize_beta': self.stabilize_beta,
            'stabilize_enhancer': self.stabilize_enhancer,
            'stabilize_enhancer_strength': self.stabilize_enhancer_strength,
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(data, f)



