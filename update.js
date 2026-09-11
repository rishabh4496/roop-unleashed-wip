module.exports = {
  run: [{
    // Keep every device on the same repository and update channel. A plain
    // `git pull` preserves a device's old branch, so a clone from master would
    // never receive commits made to the current development channel.
    method: "shell.run",
    params: {
      message: [
        "git remote set-url origin https://github.com/rishabh4496/roop-unleashed-wip.git",
        "git fetch origin --prune \"+refs/heads/*:refs/remotes/origin/*\"",
        "git show-ref --verify --quiet refs/heads/fix/hyperswap-batch-reshape || git switch --track -c fix/hyperswap-batch-reshape origin/fix/hyperswap-batch-reshape",
        "git switch fix/hyperswap-batch-reshape",
        "git pull --ff-only origin fix/hyperswap-batch-reshape",
        "git status --short --branch",
        "git log -1 --format=updated-to:%h-%s"
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: "uv pip install -r requirements.txt"
    }
  }, {
    // Re-install Node dependencies in case package.json changed
    method: "shell.run",
    params: {
      path: "react-ui",
      message: "npm install"
    }
  }]
}
