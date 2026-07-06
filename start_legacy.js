module.exports = async (kernel) => {
  const GRADIO_PORT = await kernel.port()
  const API_PORT = await kernel.port()

  return {
    daemon: true,
    run: [
      {
        method: "shell.run",
        params: {
          venv: "env",
          env: {
            ROOP_PROFILE: "1",
            ROOP_BATCH_SWAP_XFRAME: "1", 
            ROOP_BATCH_SWAP: "1", 
            ROOP_STAB_PARALLEL: "1",
            OMP_NUM_THREADS: "1",
            OPENBLAS_NUM_THREADS: "1",
            MKL_NUM_THREADS: "1",
            NUMEXPR_NUM_THREADS: "1",
            ROOP_GRADIO_PORT: String(GRADIO_PORT),
            ROOP_API_PORT: String(API_PORT)
          },
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
}
