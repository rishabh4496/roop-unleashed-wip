module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          // GPU pool sizes are auto-tuned by detected VRAM (see
          // app/roop/session_pool.py): small cards (e.g. 6GB) disable pooling,
          // large cards get the validated multi-context settings. To force a
          // value, set ROOP_TRT_POOL / ROOP_DETMASK_POOL here.
          ROOP_PROFILE: "1",
          ROOP_BATCH_SWAP_XFRAME: "1", 
          ROOP_BATCH_SWAP: "1", 
          ROOP_STAB_PARALLEL: "1",
          OMP_NUM_THREADS: "1",
          OPENBLAS_NUM_THREADS: "1",
          MKL_NUM_THREADS: "1",
          NUMEXPR_NUM_THREADS: "1"
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