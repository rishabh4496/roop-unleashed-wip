#!/usr/bin/env python3

import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["AV_LOG_LEVEL"] = "error"

# Windows asyncio fix: Python 3.10 on Windows raises ConnectionResetError
# (WinError 10054) in asyncio ProactorEventLoop when a subprocess pipe closes.
# This is a known CPython bug fixed in 3.11. Patch swallows the spurious error.
import sys as _sys
if _sys.platform == 'win32':
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport as _T
        _orig_ccl = _T._call_connection_lost
        def _patched_ccl(self, exc):
            try:
                _orig_ccl(self, exc)
            except ConnectionResetError:
                pass
        _T._call_connection_lost = _patched_ccl
    except Exception:
        pass

from roop import core
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--execution-provider', default='cuda', help='Execution provider: cpu or cuda')
args = parser.parse_args()
from roop import globals
# Normalize to onnxruntime's exact provider names — naive concatenation makes
# 'cudaExecutionProvider' (wrong case), which get_device() and the GPU guard
# would not recognize during the window before ui.main overwrites this.
_PROVIDER_NAMES = {
    'cpu': 'CPUExecutionProvider',
    'cuda': 'CUDAExecutionProvider',
    'tensorrt': 'TensorrtExecutionProvider',
    'rocm': 'ROCMExecutionProvider',
    'dml': 'DmlExecutionProvider',
}
globals.execution_providers = [_PROVIDER_NAMES.get(
    args.execution_provider.lower(), args.execution_provider + 'ExecutionProvider')]

if __name__ == '__main__':
    import threading
    from api import run_api
    threading.Thread(target=run_api, daemon=True).start()
    core.run()
