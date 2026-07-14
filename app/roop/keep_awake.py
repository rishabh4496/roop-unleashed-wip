"""Keep the machine (and GPU) fully powered during long processing runs.

Symptom this fixes: on Windows, when the display powers off after the idle
timeout — or the user manually turns the monitor off — a long-running batch
freezes and only resumes when the screen comes back on. Two OS behaviours cause
this: (1) with no active display, the NVIDIA GPU is allowed to drop into a deep
low-power state where in-flight CUDA work stalls, and (2) Windows applies power
throttling (EcoQoS) to a process it considers backgrounded. A 10+ hour job can
sit frozen for hours as a result.

The fix is to tell Windows the app is actively working for the whole duration of
a run via SetThreadExecutionState with ES_DISPLAY_REQUIRED, which keeps the
display adapter / GPU powered even while the physical panel is off, plus
ES_SYSTEM_REQUIRED so the machine never sleeps mid-run. This does NOT force the
monitor to stay lit — the user can still turn the screen off; Windows simply
keeps the GPU in a normal power state so processing continues.

Windows-only; a harmless no-op on macOS/Linux. All calls are defensive: a
failure here must never break processing.

The execution state set with ES_CONTINUOUS is associated with the calling
thread and is automatically cleared when that thread exits, so acquire() and
release() must run on the same (processing) thread — which they do, since
batch_process() runs synchronously on one thread from start to finish.
"""
import sys

# SetThreadExecutionState flags (winbase.h)
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002

_active = False


def acquire() -> None:
    """Prevent display-off / sleep power throttling for the current run."""
    global _active
    if sys.platform != 'win32' or _active:
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED)
        _active = True
        print('[KeepAwake] display/sleep power management suppressed for this run '
              '(GPU stays powered while the screen is off)')
    except Exception as e:
        print(f'[KeepAwake] could not suppress power management: {e}')


def release() -> None:
    """Restore normal power management once the run has finished."""
    global _active
    if sys.platform != 'win32' or not _active:
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    except Exception:
        pass
    finally:
        _active = False
