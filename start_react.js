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
            // ROOP_EXPR_POOL is deliberately NOT set here. It used to be pinned
            // to "2" — measured +28% on the expression stage for +654MB — but
            // this file ships to every install, and forcing two extra restorer
            // contexts on a card too small for even the swapper pool is how you
            // turn a benchmark into an OOM. session_pool._auto_expression_pool()
            // now picks it from the machine's own VRAM (12GB+ still gets 2).
            // Set ROOP_EXPR_POOL here only to override that on THIS machine.
            // Diagnostic for wrong-face swaps: prints [TRACKASSIGN] once per
            // track (which person each track was bound to, and at what cosine
            // distance) and [TRACKMATCH] per face per frame (chosen track,
            // resolved source, and the reason for any veto). Read it to tell a
            // too-permissive distance threshold apart from a tracker identity
            // switch. VERBOSE — [TRACKMATCH] fires for every face of every
            // frame, so set it back to "0" once the question is answered.
            // Off: it was left on after a wrong-face investigation and was
            // producing 461 of every 587 log lines on a real render. Set to "1"
            // when a wrong-face question comes up again. (It no longer breaks
            // the progress bar either way — these go through bar_write now.)
            ROOP_DEBUG_MATCH: "0",
            // Closes the one path max_face_distance cannot reach: with a single
            // selected person the tracked swap applies the source by SPATIAL
            // association with no identity check at all, so a tracker ID switch
            // keeps swapping the wrong face. 1.0 is the conservative setting —
            // it vetoes unambiguous strangers (measured ~0.93-1.07) while
            // leaving even a full profile of the right person alone. Tighten
            // toward 0.9 if strangers still get through AND several angles of
            // the target are captured (the check takes the closest angle, so a
            // multi-angle bank keeps same-person distances low). Too tight and
            // hard frames blink off instead. "0" restores the old behaviour.
            ROOP_TRACK_VETO_SINGLE: "1.0",
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