import React, { useEffect, useRef, useState } from 'react';
import { motion, spring } from '../../motion';

const fmtBytes = (b) => {
  if (!b) return '0 B';
  if (b >= 1073741824) return `${(b / 1073741824).toFixed(2)} GB`;
  if (b >= 1048576) return `${(b / 1048576).toFixed(1)} MB`;
  if (b >= 1024) return `${Math.round(b / 1024)} KB`;
  return `${b} B`;
};

const fmtEta = (s) => {
  if (!isFinite(s) || s <= 0) return '';
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${String(Math.round(s % 60)).padStart(2, '0')}s left` : `${Math.round(s)}s left`;
};

/**
 * Drop zone / file picker.
 *
 * `progress` is the live upload state from api.js's xhrUpload — see there for
 * why uploads are XHR rather than fetch. Passing it in turns the indeterminate
 * spinner into a real bar; leaving it undefined keeps the old behaviour, which
 * is what the callers that upload small files still do.
 */
export default function FileDrop({ label, accept, multiple, onFiles, busy, hint, progress, onCancel }) {
  const [drag, setDrag] = useState(false);
  const onDrop = (e) => {
    e.preventDefault(); setDrag(false);
    // Mark the native event as consumed so App.jsx's global drop handler
    // doesn't ALSO route these files (which popped the Source/Target dialog
    // on top of a drop that this zone already handled).
    e.nativeEvent.roopConsumed = true;
    if (busy) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length) onFiles(e.dataTransfer.files);
  };

  // Rate and ETA are derived here rather than passed in, so the caller only has
  // to forward what the XHR reports. Sampled against the FIRST observation of
  // this upload, not the previous one: per-event deltas over a fast local
  // socket are tiny and noisy enough that the number flickers unreadably.
  const rateRef = useRef(null);
  const uploading = !!progress && progress.phase === 'upload';
  useEffect(() => {
    if (!busy) rateRef.current = null;
  }, [busy]);
  let rate = 0;
  let eta = 0;
  if (uploading && progress.loaded > 0) {
    if (!rateRef.current) rateRef.current = { t: Date.now(), loaded: progress.loaded };
    const dt = (Date.now() - rateRef.current.t) / 1000;
    const db = progress.loaded - rateRef.current.loaded;
    if (dt > 0.4 && db > 0) {
      rate = db / dt;
      if (progress.total) eta = (progress.total - progress.loaded) / rate;
    }
  }
  const pct = uploading && progress.total
    ? Math.min(100, (progress.loaded / progress.total) * 100)
    : 0;
  const determinate = uploading && progress.total > 0;

  return (
    <label
      onDragOver={(e) => { e.preventDefault(); if (!busy) setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      className={`block ${busy ? 'cursor-wait' : 'cursor-pointer group'}`}
    >
      {/* The picker itself, FIRST and `sr-only peer`, not last and `hidden`.
          `hidden` is display:none, which takes a control out of the tab order
          entirely — and a <label> is not a tab stop, so this zone had nothing
          focusable in it at all. Since FileDrop is BOTH upload zones and source
          faces have no other entry point, adding a face was impossible without
          a mouse. `sr-only` hides it visually but keeps it focusable, and a
          focused file input opens its dialog on Enter/Space; it has to precede
          the visible box for Tailwind's `peer-*` to reach it. Same idiom the
          Toggle in ui.jsx uses. */}
      <input type="file" accept={accept} multiple={multiple}
        onChange={(e) => { if (e.target.files.length) onFiles(e.target.files); e.target.value = ''; }}
        disabled={busy} className="sr-only peer" />
      <motion.div
        animate={{ scale: drag ? 1.035 : 1 }}
        whileHover={busy ? undefined : { scale: 1.012 }}
        transition={spring.snappy}
        className={`relative overflow-hidden px-4 py-3.5 rounded-xl border-2 border-dashed text-center transition-colors duration-200 peer-focus-visible:border-[var(--accent)] peer-focus-visible:ring-2 peer-focus-visible:ring-[var(--accent)]/40 ${busy ? 'border-[var(--accent)]/60 bg-[var(--accent)]/[0.06]' : drag ? 'border-[var(--accent)]/70 bg-[var(--accent)]/[0.10] shadow-[0_0_30px_var(--accent-glow)]' : 'border-white/15 hover:border-[var(--accent)]/50 hover:bg-white/[0.02]'}`}
      >
        {busy ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-center gap-2 text-xs font-semibold text-white/80">
              {!determinate && (
                <span className="h-3.5 w-3.5 rounded-full border-2 border-white/30 border-t-[var(--accent)] animate-spin" />
              )}
              <span>
                {progress?.phase === 'analyse'
                  // The upload finishing is not the job finishing. For a long
                  // video the server-side decode + detection is the longer
                  // half, and a bar parked at 100% reads as a hang.
                  ? 'Analysing — detecting faces…'
                  : determinate ? `Uploading ${pct.toFixed(0)}%` : 'Uploading & analysing…'}
              </span>
              {onCancel && (
                <button
                  type="button"
                  // The label would otherwise open the file dialog behind the
                  // confirmation, and the click would also re-trigger the drop
                  // zone it is sitting inside.
                  onClick={(e) => { e.preventDefault(); e.stopPropagation(); onCancel(); }}
                  className="ml-1 px-2 py-0.5 rounded-md text-micro font-bold text-white/50 hover:text-white bg-white/[0.06] hover:bg-white/15 border border-white/10 transition-colors pointer-events-auto"
                  title="Cancel this upload"
                  aria-label="Cancel this upload"
                >
                  Cancel
                </button>
              )}
            </div>

            {determinate && (
              <>
                <div className="h-1.5 w-full rounded-full bg-white/10 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-150"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <div className="flex items-center justify-center gap-2 text-micro text-white/40 font-mono tabular-nums">
                  <span>{fmtBytes(progress.loaded)} / {fmtBytes(progress.total)}</span>
                  {rate > 0 && <><span>·</span><span>{fmtBytes(rate)}/s</span></>}
                  {eta > 0 && <><span>·</span><span>{fmtEta(eta)}</span></>}
                </div>
              </>
            )}

            {/* Screen readers get the percentage without the layout above. */}
            <span className="sr-only" role="status" aria-live="polite">
              {progress?.phase === 'analyse'
                ? 'Upload complete, analysing'
                : determinate ? `Uploading, ${pct.toFixed(0)} percent` : 'Uploading'}
            </span>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-3 pointer-events-none">
            <motion.div
              animate={drag ? { y: [-1, -5, -1] } : { y: 0 }}
              transition={drag ? { duration: 1.1, repeat: Infinity, ease: 'easeInOut' } : spring.bouncy}
              className={`p-2 rounded-xl bg-black/20 ${drag ? 'text-[var(--accent)] bg-[var(--accent)]/10' : 'text-white/40 group-hover:text-white/70 group-hover:bg-white/5'} transition-colors duration-200`}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </motion.div>
            <div className="text-left">
              <span className={`text-xs font-bold tracking-wide block ${drag ? 'text-[var(--accent)]' : 'text-white/80'}`}>{drag ? 'Drop files now' : label}</span>
              {!drag && hint && <span className="block text-micro text-white/40 mt-0.5">{hint}</span>}
            </div>
          </div>
        )}
      </motion.div>
    </label>
  );
}
