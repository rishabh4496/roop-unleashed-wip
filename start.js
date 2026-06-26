module.exports = {
  requires: {
    bundle: "ai",
  },
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: { GRADIO_SERVER_NAME: "127.0.0.1" },
        path: "app",
        message: [
          "python run.py",
        ],
        on: [{
          "event": "/(http:\\/\\/\\S+)/",
          "done": true
        }]
      }
    },
    {
      method: "shell.run",
      params: {
        path: "react-ui",
        message: [
          "npm run dev"
        ],
        on: [{
          "event": "/(http:\\/\\/localhost:[0-9]+)/",
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}