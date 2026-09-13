"""Startup network policy for RoopX.

This module intentionally has only Python standard-library dependencies.  It is
imported by ``run.py`` before torch, transformers, InsightFace, or Gradio are
loaded, so an air-gapped machine is put into offline mode before any third-party
package gets an opportunity to contact the network.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
from typing import Iterable, Optional


_LOGGER = logging.getLogger(__name__)

# These are understood by the libraries used by the optional model loaders.
# Keep them in one place so a new entry point cannot accidentally enable only
# part of the offline policy.
OFFLINE_ENV_VARS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "y"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "n", ""})
_PROBE_HOST = "huggingface.co"
_PROBE_PORT = 443
_PROBE_TIMEOUT = 2.0
_reported_offline = False


def _truthy(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in _FALSE_VALUES:
        return False
    return text in _TRUE_VALUES or bool(text)


def offline_enabled() -> bool:
    """Return whether this process must avoid all remote model access."""

    if _truthy(os.environ.get("ROOP_OFFLINE")):
        return True
    return any(_truthy(os.environ.get(name)) for name in OFFLINE_ENV_VARS)


def internet_available(timeout: float = _PROBE_TIMEOUT) -> bool:
    """Perform a bounded DNS + TCP probe to the primary model host.

    A DNS failure is treated as offline immediately.  A successful DNS lookup
    is not enough: the TCP connect is also checked, which covers airplane mode,
    captive portals, and firewalls that black-hole HTTPS.  This probe is only
    used once during startup; processing never depends on it.
    """

    resolved = []

    def _resolve():
        try:
            resolved.extend(socket.getaddrinfo(
                _PROBE_HOST,
                _PROBE_PORT,
                type=socket.SOCK_STREAM,
            ))
        except (OSError, socket.gaierror):
            return

    # Some Windows DNS providers ignore the caller's socket timeout. Run the
    # resolver in a daemon thread so a disconnected boot cannot wait forever.
    resolver = threading.Thread(target=_resolve, daemon=True)
    resolver.start()
    resolver.join(timeout)
    if resolver.is_alive() or not resolved:
        return False

    deadline = time.monotonic() + timeout
    for family, socktype, proto, _canonname, sockaddr in resolved:
        sock = socket.socket(family, socktype, proto)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            sock.connect(sockaddr)
            return True
        except (OSError, TimeoutError):
            continue
        finally:
            sock.close()
    return False


def _set_offline_environment() -> None:
    os.environ["ROOP_OFFLINE"] = "1"
    for name in OFFLINE_ENV_VARS:
        os.environ[name] = "1"


def mark_offline(reason: Optional[str] = None) -> None:
    """Switch the current process to offline mode after a network failure."""

    global _reported_offline
    _set_offline_environment()
    if reason and not _reported_offline:
        _LOGGER.warning("Network unavailable; continuing in offline mode: %s", reason)
        _reported_offline = True


def configure_startup_environment(argv: Optional[Iterable[str]] = None) -> bool:
    """Configure offline/telemetry flags before importing ML dependencies.

    ``--offline`` and existing offline environment flags always win.  Otherwise
    a failed two-second DNS/TCP probe enables offline mode automatically.  The
    return value is the effective offline state and is useful for diagnostics.
    """

    args = list(sys.argv if argv is None else argv)
    explicit = "--offline" in args or offline_enabled()

    # Gradio and Albumentations should never make analytics/update calls in a
    # local desktop application.  Set, rather than default, these values so a
    # stale shell environment cannot re-enable remote telemetry.
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_TELEMETRY_ENABLED"] = "False"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "3"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3"

    if explicit:
        _set_offline_environment()
        _LOGGER.info("Offline mode enabled by command line or environment")
        return True

    if not internet_available():
        mark_offline("startup DNS/HTTPS probe failed")
        return True

    # Do not force HF offline flags back to zero when a caller intentionally
    # supplied a non-offline environment.  The explicit branch above handles
    # all truthy values; these values only make the normal case unambiguous.
    os.environ["ROOP_OFFLINE"] = "0"
    return False
