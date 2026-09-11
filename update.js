module.exports = {
  run: [{
    // Keep every device on the canonical main branch. Persist the all-branch
    // fetch refspec because older Pinokio clones may only fetch their original
    // default branch.
    method: "shell.run",
    params: {
      message: [
        "git remote set-url origin https://github.com/rishabh4496/roop-unleashed-wip.git",
        "git config remote.origin.fetch \"+refs/heads/*:refs/remotes/origin/*\"",
        "git fetch origin --prune",
        "git show-ref --verify --quiet refs/heads/main || git switch --track -c main origin/main",
        "git switch main",
        "git branch --set-upstream-to=origin/main main",
        "git pull --ff-only",
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
