// ── One piece of GPU work at a time, tab-wide ─────────────────────────────
// Every request this UI makes that runs the detector or the swap pipeline goes
// through here: /api/preview and /api/preview_upscale, but also the endpoints
// that capture faces from a frame (/api/target/use_face, /api/target/add_angle,
// /api/target/auto_angles) and the clip advisor's sampling pass. They all drive
// the same shared FaceAnalysis pool and read the same process-wide roop_globals,
// so they are mutually exclusive in fact whether or not the UI treats them
// that way.
//
// The backend's preview endpoint writes the parameters it was handed into
// PROCESS-WIDE globals (roop_globals.selected_enhancer, .mask_engine,
// .subsample_size …) and only then runs the swap, so two overlapping requests
// interleave: the second one's writes land before the first one's swap reads
// them, and the first comes back rendered with the second's settings.
// live_swap's own lock does not help — it is taken after those writes, and it
// serialises the GPU, not the configuration.
//
// That is what broke the comparison grids. A grid renders its cells one at a
// time, which is careful enough on its own — but the MAIN preview refreshes on
// exactly the same trigger (any settings change), and it did so through a
// completely separate single-flight guard. So the two ran side by side and the
// grid's four cells came back as four renders of the main preview's enhancer:
// a grid that shows four identical pictures, and appears to do nothing at all
// when you change a setting.
//
// One promise chain, tail-appended. Callers run in submission order, strictly
// one at a time, and a rejected call does not break the chain for the next one.
// This is deliberately module scope rather than a hook: it has to cover the
// main preview, all four grids, the single-frame upscale and the face-capture
// actions together, and any of them can be mounted at once.
//
// Long jobs (auto-angles scans a whole video) therefore hold the queue, which is
// the intended behaviour — a preview fired mid-scan would fight it for the GPU.
// Callers that can be slow should show their own busy state; they all do.

let tail = Promise.resolve();

/**
 * Run `fn` once every previously-queued call has settled.
 * @param {() => Promise<any>} fn
 * @returns {Promise<any>} `fn`'s own promise — rejections propagate to the caller.
 */
export function runExclusive(fn) {
  const run = () => fn();
  const result = tail.then(run, run);
  // The chain must never carry a rejection forward, or one failed cell would
  // reject every request queued behind it.
  tail = result.then(() => {}, () => {});
  return result;
}
