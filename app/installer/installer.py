"""Legacy standalone installer.

The Pinokio launcher uses the root install.js; this module remains for users
who invoke the historical Conda installer directly. Commands are argument
lists so filenames and forwarded CLI arguments never pass through a shell.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence


APP_DIR = Path("roop-unleashed")


def offline_requested(arguments: Sequence[str]) -> bool:
    values = {"1", "true", "yes", "on"}
    return ("--offline" in arguments
            or os.environ.get("ROOP_OFFLINE", "").strip().lower() in values)


def run_cmd(
    cmd: Sequence[str],
    *,
    capture_output: bool = False,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run one command without shell expansion and surface failures."""
    completed = subprocess.run(
        [str(part) for part in cmd],
        shell=False,
        capture_output=capture_output,
        env=dict(env) if env is not None else None,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
    )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def check_env() -> None:
    if run_cmd(["conda", "--version"], capture_output=True,
               check=False).returncode != 0:
        raise RuntimeError("Conda is not installed")
    if os.environ.get("CONDA_DEFAULT_ENV") in (None, "", "base"):
        raise RuntimeError("Activate a non-base Conda environment first")


def install_dependencies() -> None:
    run_cmd(["conda", "install", "-y", "-k", "git"])
    run_cmd(["git", "clone", "https://github.com/C0untFloyd/roop-unleashed.git",
             str(APP_DIR)])
    run_cmd(["git", "checkout", "126fd699c35166772fd60dc6cbe5b0762c9967a1"],
            cwd=APP_DIR)
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=APP_DIR)


def update_dependencies() -> None:
    """Fast-forward only, preserving local application changes."""
    run_cmd(["git", "fetch", "--all"], cwd=APP_DIR)
    run_cmd(["git", "pull", "--ff-only"], cwd=APP_DIR)
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=APP_DIR)


def start_app(arguments: Sequence[str]) -> int:
    print("Launching App")
    return run_cmd([sys.executable, "run.py", *arguments], cwd=APP_DIR,
                   check=False).returncode


def main() -> int:
    arguments = sys.argv[1:]
    offline = offline_requested(arguments)
    check_env()
    if not APP_DIR.exists():
        if offline:
            raise RuntimeError(
                "Offline installer cannot clone the application. Place the "
                f"repository at '{APP_DIR}' first, then rerun with --offline."
            )
        install_dependencies()
    elif not offline and input("Check for Updates? [y/n]").strip().lower() == "y":
        update_dependencies()
    return start_app(arguments)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Installer failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
