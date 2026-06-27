module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: { ROOP_TRT_POOL: "4", ROOP_PROFILE: "1", ROOP_BATCH_SWAP_XFRAME: "1", ROOP_BATCH_SWAP: "1", ROOP_STAB_PARALLEL: "1" },
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
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
