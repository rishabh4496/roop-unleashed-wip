module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    // Pin fresh installs to the same channel used by update.js. This avoids
    // installing an older default-branch checkout on a new device.
    {
      method: "shell.run",
      params: {
        message: [
          "git remote set-url origin https://github.com/rishabh4496/roop-unleashed-wip.git",
          "git fetch origin --prune \"+refs/heads/*:refs/remotes/origin/*\"",
          "git show-ref --verify --quiet refs/heads/fix/hyperswap-batch-reshape || git switch --track -c fix/hyperswap-batch-reshape origin/fix/hyperswap-batch-reshape",
          "git switch fix/hyperswap-batch-reshape",
          "git pull --ff-only origin fix/hyperswap-batch-reshape"
        ]
      }
    },
    // Install Python dependencies for the backend (app/ is already in the repo)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Install Node.js dependencies for the React UI
    {
      method: "shell.run",
      params: {
        path: "react-ui",
        message: [
          "npm install"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },
    // Segment Anything 2 (tracked mask engine). Installed AFTER torch.js so it
    // reuses the torch installed there: --no-deps + only its pure-Python deps so
    // the torch/numpy/cv2 already in the env are never touched.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --no-deps sam2 hydra-core omegaconf iopath portalocker antlr4-python3-runtime==4.9.3"
        ]
      }
    }
  ]
}
