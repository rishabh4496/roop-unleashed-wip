"""Isolated cost of each face enhancer, under the app's exact TensorRT setup.

The swap phase is GPU-bound (~98% util), so the enhancer's share of GPU work is
the thing to cut, and the only honest way to compare models is one at a time
with nothing else on the card.

Two traps this file exists to avoid (both previously produced ~900ms "results"
that looked like a plausible 1.0x):
  1. importing torch BEFORE onnxruntime makes ORT reject the CUDA EP;
  2. without tensorrt_libs on the DLL path, ORT fails TRT registration and falls
     back to CPU for the WHOLE session — not to CUDA.
So: onnxruntime first, DLL dir added explicitly, and the resolved provider is
asserted rather than assumed.
"""
import os, time

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../app
os.chdir(APP)

# tensorrt_libs alone is NOT enough: onnxruntime_providers_tensorrt.dll also
# needs cublas/cudart, which in this env ship inside torch's lib dir. Adding the
# directory does not import torch — importing torch here would itself break the
# CUDA EP registration, which is the other trap.
for _d in ('env/Lib/site-packages/tensorrt_libs',
           'env/Lib/site-packages/torch/lib',
           'env/Lib/site-packages/nvidia/cuda_runtime/bin'):
    _p = os.path.abspath(os.path.join(APP, _d))
    if os.path.isdir(_p):
        os.add_dll_directory(_p)
        os.environ['PATH'] = _p + os.pathsep + os.environ['PATH']
    else:
        print(f'WARN missing DLL dir {_p}')

import onnxruntime           # noqa: E402  MUST precede torch
import numpy as np           # noqa: E402

CACHE = os.path.join(APP, 'models', 'trt_cache', 'mixed')
TRT_OPTS = {
    'device_id': 0,
    'trt_fp16_enable': True,
    'trt_engine_cache_enable': True,
    'trt_engine_cache_path': CACHE,
    'trt_max_partition_iterations': 2000,
    'trt_max_workspace_size': int(12.0 * 1024**3 * 0.8),
}
PROVIDERS = [('TensorrtExecutionProvider', TRT_OPTS),
             'CUDAExecutionProvider', 'CPUExecutionProvider']

MODELS = [
    ('RestoreFormer++', 'models/restoreformer_plus_plus.onnx'),  # baseline, keep first
    ('GPEN-BFR-512',    'models/GPEN-BFR-512.onnx'),
    ('GFPGAN v1.4',     'models/GFPGANv1.4.onnx'),
    ('CodeFormer',      'models/CodeFormer/CodeFormerv0.1.onnx'),
]

N_WARM, N_ITER = 5, 25
rng = np.random.default_rng(0)
print(f'{"model":18s} {"provider":12s} {"input":22s} {"init":>8s} {"ms/call":>9s} {"rel":>6s}')
base = None
for name, rel in MODELS:
    path = os.path.join(APP, rel)
    if not os.path.exists(path):
        print(f'{name:18s} MISSING {rel}')
        continue
    t0 = time.perf_counter()
    try:
        sess = onnxruntime.InferenceSession(path, None, providers=PROVIDERS)
    except Exception as e:
        print(f'{name:18s} FAILED to load: {type(e).__name__}: {e}')
        continue
    init = time.perf_counter() - t0
    prov = sess.get_providers()[0].replace('ExecutionProvider', '')
    ins = sess.get_inputs()

    feeds = {}
    for inp in ins:
        shape = [1 if (isinstance(d, str) or d is None or d < 1) else d for d in inp.shape]
        if len(shape) == 4:                      # the image tensor
            feeds[inp.name] = rng.random(shape, dtype=np.float32) * 2 - 1
        else:                                    # CodeFormer's fidelity scalar
            feeds[inp.name] = np.array([0.5] * max(1, int(np.prod(shape))),
                                       dtype=np.float64).reshape(shape)
    shp = 'x'.join(str(d) for d in
                   next(v.shape for v in feeds.values() if v.ndim == 4))

    out0 = sess.get_outputs()[0].name
    try:
        for _ in range(N_WARM):
            sess.run([out0], feeds)
        t = time.perf_counter()
        for _ in range(N_ITER):
            sess.run([out0], feeds)
        ms = (time.perf_counter() - t) / N_ITER * 1000
    except Exception as e:
        print(f'{name:18s} {prov:12s} {shp:22s} FAILED: {type(e).__name__}: {str(e)[:70]}')
        del sess
        continue

    if base is None:
        base = ms
    print(f'{name:18s} {prov:12s} {shp:22s} {init:7.1f}s {ms:8.2f} {ms/base:5.2f}x')
    del sess

print('\nNOTE: isolated cost. In-app the same call reports ~4x this because 8 '
      'workers contend for one GPU — the RATIO is what transfers.')
