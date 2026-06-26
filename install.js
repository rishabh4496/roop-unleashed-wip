module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
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
    }
  ]
}
