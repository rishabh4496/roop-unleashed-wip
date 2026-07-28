module.exports = async (kernel) => {
  const API_PORT = await kernel.port()
  const VITE_PORT = API_PORT + 1
  const GRADIO_PORT = API_PORT + 2

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
            // Work-stealing granularity for the parallel stabilizer: 2 blocks per
            // thread recovers most of the idle-thread imbalance (25-59% of chunk
            // time on static splits) for ~+19% warm-up recompute. See ProcessMgr.
            ROOP_STAB_BLOCKS_PER_THREAD: "2",
            // Expression restore was the only GPU stage still running one-wide
            // behind the global lock. Two independent TensorRT contexts measured
            // 28.7 -> 39.2 faces/sec (+37%) for +660MB VRAM; 3 and 4 are slower
            // than 2, so the GPU is saturated there. Output is bit-exact.
            // Costs ~660MB on top of the 4/4 swapper+detmask pools — if a render
            // OOMs or thrashes rebuilding engines, drop this or set
            // ROOP_TRT_POOL to 3. See docs/ENV_FLAGS.md.
            ROOP_EXPR_POOL: "2",
            // Diagnostic for wrong-face swaps: prints [TRACKASSIGN] once per
            // track (which person each track was bound to, and at what cosine
            // distance) and [TRACKMATCH] per face per frame (chosen track,
            // resolved source, and the reason for any veto). Read it to tell a
            // too-permissive distance threshold apart from a tracker identity
            // switch. VERBOSE — [TRACKMATCH] fires for every face of every
            // frame, so set it back to "0" once the question is answered.
            ROOP_DEBUG_MATCH: "1",
            OMP_NUM_THREADS: "1",
            OPENBLAS_NUM_THREADS: "1",
            MKL_NUM_THREADS: "1",
            NUMEXPR_NUM_THREADS: "1",
            ROOP_API_PORT: String(API_PORT),
            ROOP_GRADIO_PORT: String(GRADIO_PORT)
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
          env: {
            ROOP_API_PORT: String(API_PORT),
            PORT: String(VITE_PORT)
          },
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
          url: "{{input.event[1]}}",
          // Direct address of the FastAPI backend (api.py binds 127.0.0.1:ROOP_API_PORT).
          // Surfaced so pinokio.js can offer a graceful "Stop Swap" that POSTs
          // /api/stop — which finalizes the output video (moov atom) instead of
          // the hard process-kill the Terminal square does.
          api_url: `http://127.0.0.1:${API_PORT}`
        }
      }
    ]
  }
}