"""One-shot installer for the KEEP sidecar.

Creates an ISOLATED virtual environment (so KEEP's basicsr-era dependency pins
can never conflict with the main app env), clones the official KEEP repo,
installs its requirements + a CUDA torch build, and downloads the released
checkpoint. Safe to re-run — every step is idempotent.

Run from the `app` folder with the app's own (trusted) interpreter:
    env\\Scripts\\python.exe sidecar_keep\\setup_sidecar.py
"""

import os
import shutil
import subprocess
import sys
import socket
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
REPO = os.path.join(HERE, "KEEP")
WEIGHTS = os.path.join(HERE, "weights")
REPO_URL = "https://github.com/jnjaby/KEEP.git"
WEIGHT_URLS = [
    # Official v1.0.0 release assets (verified 2026-07-19).
    "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP-b76feb75.pth",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"


def _offline_enabled():
    values = {"1", "true", "yes", "on"}
    return (str(os.environ.get("ROOP_OFFLINE", "")).strip().lower() in values
            or str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in values)


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def venv_python():
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", "python.exe")
    return os.path.join(VENV, "bin", "python")


def main():
    if _offline_enabled():
        missing = []
        if not os.path.isdir(os.path.join(REPO, ".git")):
            missing.append(f"the KEEP source checkout at '{REPO}'")
        if not os.path.exists(venv_python()):
            missing.append(f"the sidecar virtual environment at '{VENV}'")
        for url in WEIGHT_URLS:
            dest = os.path.join(WEIGHTS, os.path.basename(url))
            if not os.path.isfile(dest):
                missing.append(f"the checkpoint '{dest}'")
        if missing:
            raise RuntimeError(
                "KEEP sidecar setup needs online downloads, but offline mode is "
                "enabled. Supply these local assets first: " + "; ".join(missing)
            )
        py = venv_python()
        run([py, "-c", "import torch, basicsr; print('sidecar env OK — torch', torch.__version__, 'cuda', torch.cuda.is_available())"])
        print("\nKEEP sidecar is already installed and usable offline.")
        return

    # 1. Isolated venv (uv is faster and ships with Pinokio; stdlib fallback).
    if not os.path.exists(venv_python()):
        uv = shutil.which("uv")
        if uv:
            run([uv, "venv", VENV, "--python", sys.executable])
        else:
            run([sys.executable, "-m", "venv", VENV])
    py = venv_python()

    def pip(*args):
        uv = shutil.which("uv")
        if uv:
            run([uv, "pip", "install", "--python", py, *args])
        else:
            run([py, "-m", "pip", "install", *args])

    # 2. Clone KEEP (idempotent).
    if not os.path.isdir(os.path.join(REPO, ".git")):
        run(["git", "clone", "--depth", "1", REPO_URL, REPO])

    # 3. Torch (CUDA) first, then the sidecar requirement set, then KEEP's own.
    pip("torch", "torchvision", "--index-url", TORCH_INDEX)
    pip("-r", os.path.join(HERE, "requirements.txt"))
    keep_reqs = os.path.join(REPO, "requirements.txt")
    if os.path.exists(keep_reqs):
        pip("-r", keep_reqs)

    # 4. Checkpoint(s).
    os.makedirs(WEIGHTS, exist_ok=True)
    for url in WEIGHT_URLS:
        dest = os.path.join(WEIGHTS, os.path.basename(url))
        if os.path.isfile(dest):
            print(f"weights: {os.path.basename(dest)} already present")
            continue
        print(f"downloading {url} ...", flush=True)
        part = dest + ".part"
        try:
            with urllib.request.urlopen(url, timeout=3.0) as source, open(part, "wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        except (urllib.error.URLError, OSError, socket.timeout, TimeoutError) as exc:
            if os.path.exists(part):
                os.remove(part)
            raise RuntimeError(
                f"Could not download KEEP checkpoint '{url}'. "
                f"Retry while online or place it at '{dest}'. Error: {exc}"
            ) from exc
        os.replace(part, dest)

    # 5. Smoke check: can the sidecar env import its stack?
    run([py, "-c", "import torch, basicsr; print('sidecar env OK — torch', torch.__version__, 'cuda', torch.cuda.is_available())"])
    print("\nKEEP sidecar installed. Select the 'KEEP (sidecar)' enhancer in the app.")


if __name__ == "__main__":
    main()
