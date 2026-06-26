module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: { },
        path: "app",
        message: [
          "python run.py",
        ],
        on: [{
          "event": "/(http:\\/\\/[0-9.:]+)/", 
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