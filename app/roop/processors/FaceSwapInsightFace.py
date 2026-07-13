import os
import threading
import roop.globals
import cv2
import numpy as np
import onnx
import onnxruntime

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool


# ── Per-model swap contract ───────────────────────────────────────────────────
# Every swapper here consumes the buffalo_l / w600k_r50 ArcFace identity
# embedding, but each differs in resolution, alignment template, input
# normalization, identity projection and output range. ProcessMgr reads the
# published attributes (model_output_size / model_mean / model_standard_deviation
# / model_denormalize / model_template) to drive align/normalize. Model files
# download lazily into app/models/ on the first run that selects them.
#
# "embedding" selects how the 512-d identity vector is prepared (specs match
# FaceFusion's prepare_source_embedding):
#   normed_emap    — normed_embedding @ emap, re-normalized (inswapper family)
#   normed         — normed_embedding used directly (hyperswap)
#   converted_raw  — RAW embedding through the crossface converter, no norm (ghost)
#   converted_norm — RAW embedding through the converter, then normalized
#                    (simswap / hififace)
# "template" is the 5-point alignment the model was trained on ('arcface' =
# the existing insightface alignment; others live in face_util.WARP_TEMPLATES).
_FF30 = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
_FF31 = "https://huggingface.co/facefusion/models-3.1.0/resolve/main/"
_FF33 = "https://huggingface.co/facefusion/models-3.3.0/resolve/main/"
_FF34 = "https://huggingface.co/facefusion/models-3.4.0/resolve/main/"

SWAP_MODELS = {
    # inswapper_128 — the original. arcface align, [0,1] input, identity =
    # normed_embedding @ emap, [0,1] output.
    "inswapper": {
        "file": "inswapper_128.onnx",
        "url": "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
        "output_size": 128,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "normed_emap",
        "template": "arcface",
    },
    # ReSwapper 256 — open reproduction of inswapper at 2x resolution. Same
    # identity pipeline (emap), near drop-in.
    "reswapper": {
        "file": "reswapper_256.onnx",
        "url": "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/reswapper_256.onnx",
        "output_size": 256,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "normed_emap",
        "template": "arcface",
    },
    # HyperSwap 1a/1b/1c (FaceFusion) — 256px. [-1,1] input (mean/std 0.5),
    # identity = normed_embedding directly (NO emap), de-normalized output.
    # The model emits (image, mask); we use the image output. The three
    # checkpoints trade identity likeness vs blending differently — offered
    # side by side for A/B.
    "hyperswap": {
        "file": "hyperswap_1a_256.onnx",
        "url": _FF33 + "hyperswap_1a_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "normed",
        "template": "arcface",
    },
    "hyperswap_1b": {
        "file": "hyperswap_1b_256.onnx",
        "url": _FF33 + "hyperswap_1b_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "normed",
        "template": "arcface",
    },
    "hyperswap_1c": {
        "file": "hyperswap_1c_256.onnx",
        "url": _FF33 + "hyperswap_1c_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "normed",
        "template": "arcface",
    },
    # Ghost 1/2/3 (sberbank GHOST, FaceFusion export) — 256px, arcface_112_v1
    # alignment, [-1,1] in/out, identity = RAW embedding through the crossface
    # converter (un-normalized). Different generator depths; 3 = newest.
    "ghost_1": {
        "file": "ghost_1_256.onnx",
        "url": _FF30 + "ghost_1_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "converted_raw",
        "template": "arcface_112_v1",
        "converter_file": "crossface_ghost.onnx",
        "converter_url": _FF34 + "crossface_ghost.onnx",
    },
    "ghost_2": {
        "file": "ghost_2_256.onnx",
        "url": _FF30 + "ghost_2_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "converted_raw",
        "template": "arcface_112_v1",
        "converter_file": "crossface_ghost.onnx",
        "converter_url": _FF34 + "crossface_ghost.onnx",
    },
    "ghost_3": {
        "file": "ghost_3_256.onnx",
        "url": _FF30 + "ghost_3_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "converted_raw",
        "template": "arcface_112_v1",
        "converter_file": "crossface_ghost.onnx",
        "converter_url": _FF34 + "crossface_ghost.onnx",
    },
    # SimSwap 256 — arcface_112_v1 alignment, ImageNet mean/std input, [0,1]
    # output (no denorm), identity = converted + normalized embedding.
    "simswap": {
        "file": "simswap_256.onnx",
        "url": _FF30 + "simswap_256.onnx",
        "output_size": 256,
        "mean": [0.485, 0.456, 0.406],
        "standard_deviation": [0.229, 0.224, 0.225],
        "denormalize": False,
        "embedding": "converted_norm",
        "template": "arcface_112_v1",
        "converter_file": "crossface_simswap.onnx",
        "converter_url": _FF34 + "crossface_simswap.onnx",
    },
    # SimSwap 512 (unofficial) — native 512px decoder, plain [0,1] in/out.
    "simswap_512": {
        "file": "simswap_unofficial_512.onnx",
        "url": _FF30 + "simswap_unofficial_512.onnx",
        "output_size": 512,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "converted_norm",
        "template": "arcface_112_v1",
        "converter_file": "crossface_simswap.onnx",
        "converter_url": _FF34 + "crossface_simswap.onnx",
    },
    # BlendSwap 256 — image-source swapper (NOT embedding-based). The source is
    # an aligned face IMAGE crop (arcface_112_v2 @ 112), the target crop uses
    # the ffhq_512 template. [0,1] in/out. Model inputs named 'source'/'target'.
    "blendswap": {
        "file": "blendswap_256.onnx",
        "url": _FF30 + "blendswap_256.onnx",
        "output_size": 256,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "image",
        "template": "ffhq_512",
        "source_crop_key": "_src_crop_arcface_112_v2",
    },
    # UniFace 256 — image-source swapper. Source = ffhq_512 @ 256 image crop,
    # target = ffhq_512 template, [-1,1] in, de-normalized out.
    "uniface": {
        "file": "uniface_256.onnx",
        "url": _FF30 + "uniface_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "image",
        "template": "ffhq_512",
        "source_crop_key": "_src_crop_ffhq_256",
    },
    # HifiFace (unofficial) — 256px, mtcnn_512 alignment, [-1,1] in/out,
    # identity = converted + normalized embedding.
    "hififace": {
        "file": "hififace_unofficial_256.onnx",
        "url": _FF31 + "hififace_unofficial_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "converted_norm",
        "template": "mtcnn_512",
        "converter_file": "crossface_hififace.onnx",
        "converter_url": _FF34 + "crossface_hififace.onnx",
    },
}


# Opt-in batched swap (ROOP_BATCH_SWAP=1): runs multiple face crops through one
# inference call instead of one-at-a-time, to better saturate the GPU. The stock
# swap ONNX is fixed batch-1; we relax the input/output batch dim to symbolic so
# the session (and TensorRT engine) accept batches. Verified to produce output
# numerically identical to per-crop runs.
_BATCH_SWAP = os.environ.get('ROOP_BATCH_SWAP', '0') == '1'


def _relax_batch_dim(model):
    """Mutate `model` in place, giving every graph input/output a symbolic
    batch dimension so the session accepts batches > 1."""
    for t in list(model.graph.input) + list(model.graph.output):
        dims = t.type.tensor_type.shape.dim
        if len(dims):
            dims[0].dim_param = 'N'
            dims[0].ClearField('dim_value')


def _freeze_convtranspose_reshape(model):
    """Mutate `model` in place so TensorRT can build its ConvTranspose layers.

    GHOST's generator reshapes the identity vector with a [1,-1,1,1] target and
    feeds it straight into a ConvTranspose. onnxruntime resolves the -1 to 512,
    but TensorRT keeps that inferred channel dimension dynamic and refuses to
    build the deconvolution ('IDeconvolutionLayer number of channels in `input`
    tensor must not be dynamic' → INVALID_NODE). Bake the -1 into its
    statically-inferable value for any Reshape feeding a ConvTranspose. This is
    numerically a no-op — the graph inputs are fixed-shape, so every internal
    shape is already static — verified bit-identical on CPU. Returns True if it
    changed anything (inswapper and the other emap swappers match nothing)."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        return False
    static_shapes = {}
    for vi in (list(inferred.graph.value_info) + list(inferred.graph.input)
               + list(inferred.graph.output)):
        dims = vi.type.tensor_type.shape.dim
        static_shapes[vi.name] = [
            (d.dim_value if (d.dim_param == '' and d.dim_value > 0) else None)
            for d in dims]
    convt_inputs = {n.input[0] for n in model.graph.node
                    if n.op_type == 'ConvTranspose' and n.input}
    inits = {i.name: i for i in model.graph.initializer}
    changed = False
    for node in model.graph.node:
        if node.op_type != 'Reshape' or node.output[0] not in convt_inputs:
            continue
        out_shape = static_shapes.get(node.output[0])
        if not out_shape or any(v is None for v in out_shape):
            continue
        shape_init = inits.get(node.input[1])
        if shape_init is None:
            continue
        arr = onnx.numpy_helper.to_array(shape_init)
        if -1 not in arr.tolist() or len(arr) != len(out_shape):
            continue
        shape_init.CopyFrom(onnx.numpy_helper.from_array(
            np.array(out_shape, dtype=arr.dtype), shape_init.name))
        changed = True
    return changed


def _swap_providers(providers):
    """Return a copy of `providers` with the TensorRT provider forced to FP32.

    inswapper_128 (and the other emap swappers) have layers that overflow in
    FP16, producing rainbow/smudge artifacts when the global precision mode is
    'mixed'/'fp16'. The swapper is tiny (128-256px), so full precision costs
    almost nothing while fixing the corruption; detection and enhancers stay on
    FP16 where they're stable and fast. Opt back into an FP16 swapper with
    ROOP_SWAP_FP16=1 (not recommended)."""
    if os.environ.get('ROOP_SWAP_FP16', '0') == '1':
        return providers
    patched = []
    for p in providers:
        if isinstance(p, (tuple, list)) and len(p) == 2 and 'tensorrt' in str(p[0]).lower():
            name, opts = p[0], dict(p[1])
            opts['trt_fp16_enable'] = False
            # Separate engine cache so the FP32 swap engine never collides with
            # the FP16 engines TensorRT builds for the other models.
            cache = opts.get('trt_engine_cache_path')
            if cache:
                fp32_cache = cache + '_swap_fp32'
                os.makedirs(fp32_cache, exist_ok=True)
                opts['trt_engine_cache_path'] = fp32_cache
            patched.append((name, opts))
        else:
            patched.append(p)
    return patched


class FaceSwapInsightFace():
    processorname = 'faceswap'
    type = 'swap'

    def __init__(self):
        self.plugin_options = None
        self.model_swap_insightface = None
        self.emap = None
        self.converter = None            # crossface embedding converter session
        self.converter_input = "input"
        self.embedding_mode = "normed_emap"
        self.source_crop_key = None      # image-source models: which pre-warped crop
        self.image_input_name = "target"
        self.embed_input_name = "source"
        self.loaded_model_key = None
        self.devicename = None
        self.pool = None        # SessionPool of extra sessions (TRT multi-context)
        # Contract consumed by ProcessMgr — defaults match inswapper_128.
        self.model_output_size = 128
        self.model_mean = [0.0, 0.0, 0.0]
        self.model_standard_deviation = [1.0, 1.0, 1.0]
        self.model_denormalize = False
        self.model_template = "arcface"
        # ── DFM (DeepFaceLive) state ──────────────────────────────────────────
        self.is_dfm = False              # True when a .dfm identity model is loaded
        self.model_layout = "nchw"       # 'nhwc' for DFM (in_face is [1,H,W,3])
        self.dfm_out_idx = 0             # which output tensor is the swapped face
        self.dfm_mask_idx = None         # which output tensor is out_celeb_face_mask
        self.model_pixel_boost = True    # DFM crops are native-res; no tiling
        # Per-thread stash for the DFM mask emitted alongside the last swapped
        # face, so process_face can route it into paste-back without changing the
        # generic Run() return signature.
        self._mask_tls = threading.local()

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options

        # ── DFM spike gate ────────────────────────────────────────────────────
        # ROOP_DFM=1 hijacks the swap stage to run a DeepFaceLive .dfm identity
        # model instead of the selected embedding swapper. Everything below is
        # bypassed. Off by default → zero impact on the normal path.
        if os.environ.get("ROOP_DFM", "0") == "1":
            self._initialize_dfm(plugin_options)
            return

        swap_model = plugin_options.get("swap_model", "inswapper")
        if swap_model not in SWAP_MODELS:
            swap_model = "inswapper"
        spec = SWAP_MODELS[swap_model]

        # Reload when the user switched to a different swap model.
        if self.model_swap_insightface is not None and self.loaded_model_key != swap_model:
            self.Release()

        if self.model_swap_insightface is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [spec["url"]])
            model_path = os.path.join(model_dir, spec["file"])

            self.embedding_mode = spec.get("embedding", "normed_emap")
            self.source_crop_key = spec.get("source_crop_key")
            if self.embedding_mode == "normed_emap":
                graph = onnx.load(model_path).graph
                self.emap = self._find_emap(graph)
            else:
                self.emap = None

            # crossface embedding converter (ghost / simswap / hififace): a tiny
            # MLP that maps the buffalo_l ArcFace embedding into the identity
            # space the swap net was trained with. CPU is plenty for it — the
            # result is cached per source face, so it runs once per face per
            # model, not per frame.
            if spec.get("converter_url"):
                conditional_download(model_dir, [spec["converter_url"]])
                self.converter = onnxruntime.InferenceSession(
                    os.path.join(model_dir, spec["converter_file"]),
                    None, providers=["CPUExecutionProvider"])
                self.converter_input = self.converter.get_inputs()[0].name
            else:
                self.converter = None

            self.devicename = plugin_options["devicename"].replace('mps', 'cpu')

            swap_providers = _swap_providers(roop.globals.execution_providers)
            # Load once and apply the transforms this session needs. Freezing the
            # ConvTranspose reshape channel makes TensorRT-incompatible exports
            # (GHOST) buildable; batch relaxation is opt-in. If neither applies we
            # hand onnxruntime the path so it can memory-map the file directly.
            model_arg = model_path
            _model = onnx.load(model_path)
            _changed = _freeze_convtranspose_reshape(_model)
            if _BATCH_SWAP:
                _relax_batch_dim(_model)
                _changed = True
            if _changed:
                model_arg = _model.SerializeToString()

            def _build(_i=0):
                sess_options = onnxruntime.SessionOptions()
                sess_options.enable_cpu_mem_arena = False
                return onnxruntime.InferenceSession(
                    model_arg, sess_options, providers=swap_providers)

            self.model_swap_insightface = _build()

            # Resolve input tensor names by rank instead of assuming names:
            # rank-4 = the image (NCHW), rank-2 = the identity embedding.
            for inp in self.model_swap_insightface.get_inputs():
                rank = len(inp.shape)
                if rank == 4:
                    self.image_input_name = inp.name
                elif rank == 2:
                    self.embed_input_name = inp.name
            # Image-source models (BlendSwap/UniFace) have TWO rank-4 inputs, so
            # rank alone can't tell target from source — resolve by the names the
            # models expose ('target' / 'source').
            if self.embedding_mode == "image":
                names = [inp.name for inp in self.model_swap_insightface.get_inputs()]
                self.image_input_name = "target" if "target" in names else self.image_input_name
                self.embed_input_name = "source" if "source" in names else self.embed_input_name

            # Optional TensorRT multi-context pool: the primary session plus
            # (N-1) independent extras so up to N worker threads can swap
            # concurrently instead of serialising behind the global GPU lock.
            if session_pool.pooling_enabled():
                n = session_pool.pool_size()
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model_swap_insightface] + extras): _e[i], n)

            # Publish the per-model contract ProcessMgr reads.
            self.model_output_size = spec["output_size"]
            self.model_mean = spec["mean"]
            self.model_standard_deviation = spec["standard_deviation"]
            self.model_denormalize = spec["denormalize"]
            self.model_template = spec.get("template", "arcface")
            self.loaded_model_key = swap_model

    @staticmethod
    def _resolve_dfm_path():
        """Locate the .dfm model: ROOP_DFM_MODEL (abs path) wins, else the first
        *.dfm / *.onnx dropped into app/models/dfm/."""
        p = os.environ.get("ROOP_DFM_MODEL")
        if p and os.path.isfile(p):
            return p
        model_dir = resolve_relative_path('../models/dfm')
        if os.path.isdir(model_dir):
            for f in sorted(os.listdir(model_dir)):
                if f.lower().endswith(('.dfm', '.onnx')):
                    return os.path.join(model_dir, f)
        return None

    def _initialize_dfm(self, plugin_options: dict):
        """Load a DeepFaceLive .dfm model as the swapper [Phase-1 SPIKE].

        A .dfm is a plain ONNX export whose identity is baked into the weights —
        there is NO source embedding input. It takes one NHWC face crop
        (in_face, [0,1]) and emits the swapped face (out_celeb_face) plus a
        trained mask (out_celeb_face_mask, unused in the spike). We publish the
        same model_* contract ProcessMgr already reads, plus model_layout/
        model_pixel_boost so preprocess/postprocess can branch."""
        key = "dfm:" + (os.environ.get("ROOP_DFM_MODEL") or "auto")
        if self.model_swap_insightface is not None and self.loaded_model_key != key:
            self.Release()
        if self.model_swap_insightface is not None:
            return   # already loaded this exact model

        path = self._resolve_dfm_path()
        if path is None:
            raise FileNotFoundError(
                "ROOP_DFM=1 but no model found — set ROOP_DFM_MODEL=<abs path to .dfm> "
                "or drop a .dfm file into app/models/dfm/")
        print(f"[DFM] Loading DeepFaceLive model: {path}")

        self.devicename = plugin_options["devicename"].replace('mps', 'cpu')
        swap_providers = _swap_providers(roop.globals.execution_providers)

        # Freeze the decoder's ConvTranspose reshape channel so TensorRT can build
        # the DFM generator (same fix the GHOST export needs). No-op on CPU/CUDA.
        _model = onnx.load(path)
        _freeze_convtranspose_reshape(_model)
        model_arg = _model.SerializeToString()

        def _build(_i=0):
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False
            return onnxruntime.InferenceSession(
                model_arg, sess_options, providers=swap_providers)

        self.model_swap_insightface = _build()

        # Resolve the single image input and its layout/resolution.
        inp = self.model_swap_insightface.get_inputs()[0]
        self.image_input_name = inp.name
        shape = inp.shape
        if len(shape) == 4 and shape[-1] == 3:
            self.model_layout = "nhwc"
            res = int(shape[1])
        elif len(shape) == 4 and shape[1] == 3:
            self.model_layout = "nchw"
            res = int(shape[2])
        else:
            self.model_layout, res = "nhwc", 256   # sane fallback

        # Resolve outputs by name: the swapped face ('celeb' non-mask) and its
        # trained mask ('celeb'+'mask'). Fall back to positional if names differ.
        outs = self.model_swap_insightface.get_outputs()
        self.dfm_out_idx = 0
        self.dfm_mask_idx = None
        for i, o in enumerate(outs):
            n = o.name.lower()
            if "mask" in n:
                self.dfm_mask_idx = i
            elif "celeb" in n:
                self.dfm_out_idx = i
        # If no 'celeb' face was found, take the first non-mask output.
        if not any("celeb" in o.name.lower() and "mask" not in o.name.lower() for o in outs):
            for i, o in enumerate(outs):
                if "mask" not in o.name.lower():
                    self.dfm_out_idx = i
                    break
        # If names gave us no mask but there are >=2 outputs, assume the other one.
        if self.dfm_mask_idx is None and len(outs) >= 2:
            for i in range(len(outs)):
                if i != self.dfm_out_idx:
                    self.dfm_mask_idx = i
                    break

        # Publish the contract.
        self.is_dfm = True
        self.embedding_mode = "dfm"
        self.model_output_size = res
        self.model_mean = [0.0, 0.0, 0.0]
        self.model_standard_deviation = [1.0, 1.0, 1.0]
        self.model_denormalize = False
        self.model_template = "dfl_whole_face"
        self.model_pixel_boost = False

        # Optional TRT multi-context pool, same as the embedding swappers.
        if session_pool.pooling_enabled():
            n = session_pool.pool_size()
            extras = [_build(i) for i in range(n - 1)]
            self.pool = session_pool.SessionPool(
                lambda i, _e=([self.model_swap_insightface] + extras): _e[i], n)

        self.loaded_model_key = key
        _mask_name = outs[self.dfm_mask_idx].name if self.dfm_mask_idx is not None else "none"
        print(f"[DFM] ready — layout={self.model_layout} res={res} "
              f"face='{outs[self.dfm_out_idx].name}' mask='{_mask_name}' "
              f"template=dfl_whole_face(cov={os.environ.get('ROOP_DFM_COVERAGE','1.0')}) "
              f"pool={'on' if self.pool else 'off'}")

    @staticmethod
    def _find_emap(graph):
        """Locate the 512x512 identity-projection matrix (emap) embedded in the onnx."""
        for init in reversed(graph.initializer):
            arr = onnx.numpy_helper.to_array(init)
            if arr.ndim == 2 and arr.shape == (512, 512):
                return arr
        # Fallback: inswapper_128 stores emap as the last initializer.
        return onnx.numpy_helper.to_array(graph.initializer[-1])

    def _compute_latent(self, source_face: Face) -> np.ndarray:
        """Prepare the (1, 512) identity vector per the loaded model's contract.
        Converter results are cached on the Face object (keyed by model) so the
        crossface MLP runs once per source face, not once per frame."""
        mode = self.embedding_mode
        if mode == "normed":
            return source_face.normed_embedding.reshape((1, -1)).astype(np.float32)
        if mode in ("converted_raw", "converted_norm"):
            cache_key = f"_latent_{self.loaded_model_key}"
            cached = source_face.get(cache_key) if hasattr(source_face, 'get') else None
            if cached is not None:
                return cached
            emb = np.asarray(source_face.embedding, dtype=np.float32).reshape(-1, 512)
            converted = self.converter.run(None, {self.converter_input: emb})[0]
            converted = converted.ravel()
            if mode == "converted_norm":
                converted = converted / np.linalg.norm(converted)
            latent = converted.reshape(1, -1).astype(np.float32)
            try:
                source_face[cache_key] = latent
            except Exception:
                pass
            return latent
        # Default: inswapper-family normed_embedding @ emap.
        latent = source_face.normed_embedding.reshape((1, -1)).astype(np.float32)
        if self.emap is not None:
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)
        return latent

    def _prepare_source_crop(self, source_face: Face) -> np.ndarray:
        """Image-source models (BlendSwap/UniFace): build the (1,3,H,W) source
        blob from the pre-warped aligned source crop (BGR→RGB, /255 — no
        mean/std, matching FaceFusion's prepare_source_frame). Cached per source
        face + model. Returns None when the crop is absent (source ingested
        before crops were attached — caller falls back to a no-op swap)."""
        cache_key = f"_srcblob_{self.loaded_model_key}"
        cached = source_face.get(cache_key) if hasattr(source_face, 'get') else None
        if cached is not None:
            return cached
        crop = source_face.get(self.source_crop_key) if hasattr(source_face, 'get') else None
        if crop is None:
            return None
        blob = crop[:, :, ::-1] / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        try:
            source_face[cache_key] = blob
        except Exception:
            pass
        return blob

    def _compute_source_input(self, source_face: Face):
        """Return whatever the loaded model feeds into its source input: an
        image blob for image-source models, else the identity latent vector."""
        if self.embedding_mode == "image":
            return self._prepare_source_crop(source_face)
        return self._compute_latent(source_face)

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        if self.is_dfm:
            # DFM has no source input — identity is in the weights. Feed the crop
            # only, stash the trained mask for paste-back, and return the face.
            feed = {self.image_input_name: temp_frame}
            if self.pool is not None:
                with self.pool.lease() as sess:
                    ort_outs = sess.run(None, feed)
            else:
                ort_outs = self.model_swap_insightface.run(None, feed)
            self._mask_tls.mask = (
                ort_outs[self.dfm_mask_idx][0] if self.dfm_mask_idx is not None else None)
            return ort_outs[self.dfm_out_idx][0]
        latent = self._compute_source_input(source_face)
        if latent is None:
            # Image-source model but no source crop available → return the target
            # crop unchanged (no swap) rather than crashing.
            return temp_frame[0]
        # Use the standard run() API rather than io_binding.  io_binding with
        # bind_output() (no device_type) leaves output placement to TensorRT,
        # which registers a device type that copy_outputs_to_cpu() has no
        # transfer path for. run() handles all device transfers internally and
        # works correctly across CPU, CUDA, and TensorRT execution providers.
        feed = {self.image_input_name: temp_frame, self.embed_input_name: latent}
        if self.pool is not None:
            # Lease an independent session so this thread's GPU work runs on its
            # own TensorRT context, concurrently with other workers.
            with self.pool.lease() as sess:
                ort_outs = sess.run(None, feed)
        else:
            ort_outs = self.model_swap_insightface.run(None, feed)
        # Some models (HyperSwap) emit (image, mask); the image is output [0].
        return ort_outs[0][0]

    def RunBatch(self, source_face: Face, target_face: Face, temp_frames: list) -> list:
        """Batched equivalent of Run: temp_frames is a list of [1,3,H,W]
        preprocessed crops sharing the same source identity. Returns a list of
        [3,H,W] outputs, one per crop — numerically identical to calling Run on
        each, but in a single inference (better GPU utilization). Requires the
        session to be batch-dynamic (ROOP_BATCH_SWAP=1)."""
        if self.is_dfm:
            img_batch = np.concatenate(temp_frames, axis=0).astype(np.float32)
            feed = {self.image_input_name: img_batch}
            if self.pool is not None:
                with self.pool.lease() as sess:
                    ort_outs = sess.run(None, feed)
            else:
                ort_outs = self.model_swap_insightface.run(None, feed)
            out = ort_outs[self.dfm_out_idx]
            return [out[i] for i in range(out.shape[0])]
        latent = self._compute_source_input(source_face)
        if latent is None:
            # Image-source model with no source crop → no-op (return the input
            # target crops unchanged), matching Run's fallback.
            return [t[0] for t in temp_frames]
        img_batch = np.concatenate(temp_frames, axis=0).astype(np.float32)   # [B,3,H,W]
        latent_batch = np.repeat(latent, img_batch.shape[0], axis=0)         # [B,512] or [B,3,Hs,Ws]
        feed = {self.image_input_name: img_batch, self.embed_input_name: latent_batch}
        if self.pool is not None:
            with self.pool.lease() as sess:
                ort_outs = sess.run(None, feed)
        else:
            ort_outs = self.model_swap_insightface.run(None, feed)
        out = ort_outs[0]   # [B,3,H,W]
        return [out[i] for i in range(out.shape[0])]

    def RunBatchMulti(self, requests: list) -> list:
        """Like RunBatch but each crop carries its OWN source identity (for
        cross-frame coalescing where different faces batch together).
        requests = list of (source_face, target_face, blob[1,3,H,W]); the
        target_face is unused by the swap net. Returns a list of [3,H,W]."""
        if self.is_dfm:
            img_batch = np.concatenate([r[2] for r in requests], axis=0).astype(np.float32)
            feed = {self.image_input_name: img_batch}
            if self.pool is not None:
                with self.pool.lease() as sess:
                    ort_outs = sess.run(None, feed)
            else:
                ort_outs = self.model_swap_insightface.run(None, feed)
            out = ort_outs[self.dfm_out_idx]
            return [out[i] for i in range(out.shape[0])]
        latents = [self._compute_source_input(src) for src, _tgt, _blob in requests]
        if any(l is None for l in latents):
            # Image-source model with a crop-less source → no-op passthrough.
            return [r[2][0] for r in requests]
        latent_batch = np.concatenate(latents, axis=0)                       # [B,512]
        img_batch = np.concatenate([r[2] for r in requests], axis=0).astype(np.float32)  # [B,3,H,W]
        feed = {self.image_input_name: img_batch, self.embed_input_name: latent_batch}
        if self.pool is not None:
            with self.pool.lease() as sess:
                ort_outs = sess.run(None, feed)
        else:
            ort_outs = self.model_swap_insightface.run(None, feed)
        out = ort_outs[0]
        return [out[i] for i in range(out.shape[0])]

    def last_dfm_mask(self):
        """The out_celeb_face_mask ([H,W,1], [0,1]) from this thread's most recent
        DFM Run, or None (model has no mask output / not DFM). process_face reads
        it right after the swap to route into paste-back."""
        return getattr(self._mask_tls, 'mask', None)

    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_swap_insightface
        self.model_swap_insightface = None
        self.emap = None
        self.converter = None
        self.loaded_model_key = None
        self.is_dfm = False
        self.model_layout = "nchw"
        self.model_pixel_boost = True
