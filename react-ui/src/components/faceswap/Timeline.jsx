import React, { useEffect, useMemo, useRef, useState } from 'react';

/**
 * The clip timeline — an editor-style scrubber for the target video.
 *
 * Replaces the earlier inline block, which read dated for three reasons: the
 * "ruler" was five evenly-spaced timecodes that lined up with nothing on the
 * track, the trim range was stated twice (chips in the header AND a numeric
 * Range box in the toolbar), and the toolbar was four separately-bordered pill
 * groups on one wrapping row, so the eye had no order to follow.
 *
 * What it is now: a measured tick ruler whose marks sit at real time positions,
 * one editable statement of In/Out (the header chips ARE the inputs), a taller
 * filmstrip with pro-editor trim treatment (out-of-range is dimmed AND
 * desaturated), a playhead carrying its own timecode, and a single toolbar
 * surface split into read-out / transport / range zones. New: chapter markers
 * (add at the playhead, click to jump, alt-click to delete) persisted per clip.
 *
 * Scrub/drag geometry still lives in FaceSwap (it owns the pointer capture and
 * the magnetic snap); this component takes the ref and the three handlers and
 * renders on top of them, so the interaction contract is unchanged.
 */

// Label steps a human would choose. The ruler picks the smallest one that keeps
// labels at least ~92px apart at the current track width, so the same clip is
// legible in a narrow rail and a wide window without re-tuning.
const NICE_SECONDS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
const SPEEDS = [0.25, 0.5, 1, 2, 4];
const MINOR_PER_MAJOR = 4;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));

/** m:ss under an hour, h:mm:ss above it. */
export const fmtTC = (f, fps) => {
  const total = Math.max(0, f) / (fps || 25);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = Math.floor(total % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`;
};

/** Timecode with the frame remainder — what the playhead badge shows. */
const fmtTCF = (f, fps) => {
  const r = (fps || 25);
  const idx = Math.max(0, f - 1);
  return `${fmtTC(f, r)}:${String(Math.floor(idx % r)).padStart(2, '0')}`;
};

const markersKey = (k) => `roop_markers_${k || 'default'}`;

/** Chip-shaped inline number field — the header's In/Out are editable in place. */
function RangeField({ label, value, min, max, onCommit, accent = false }) {
  const [draft, setDraft] = useState(null);
  const commit = () => {
    if (draft !== null) {
      const v = clamp(parseInt(draft, 10) || value, min, max);
      if (v !== value) onCommit(v);
    }
    setDraft(null);
  };
  return (
    <label
      className={`inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 transition-colors cursor-text ${
        accent
          ? 'border-[var(--accent)]/30 bg-[var(--accent)]/10'
          : 'border-[var(--border-color)] bg-[var(--surface-2)] hover:border-[var(--border-strong)]'
      }`}
      title={`${label} point — click to type a frame number`}
    >
      <span className="text-[9px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        value={draft ?? value}
        onFocus={(e) => { setDraft(String(value)); e.target.select(); }}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur();
          if (e.key === 'Escape') { setDraft(null); e.currentTarget.blur(); }
        }}
        className="w-[7ch] bg-transparent text-right text-[11px] font-mono font-semibold tabular-nums text-[var(--text-main)] outline-none"
      />
    </label>
  );
}

/** Icon button used across the toolbar clusters. */
const IconBtn = ({ title, onClick, active = false, children, className = '' }) => (
  <button
    type="button"
    onClick={onClick}
    title={title}
    aria-label={title}
    className={`grid place-items-center h-8 w-8 rounded-lg transition-colors ${
      active
        ? 'bg-[var(--accent)]/12 text-[var(--accent)]'
        : 'text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06]'
    } ${className}`}
  >
    {children}
  </button>
);

const Ico = ({ d, fill = false, size = 15 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24"
       fill={fill ? 'currentColor' : 'none'} stroke={fill ? 'none' : 'currentColor'}
       strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    {typeof d === 'string' ? <path d={d} /> : d}
  </svg>
);

export default function Timeline({
  fps = 25,
  maxFrames = 1,
  frame = 1,
  setFrame,
  startFrame = 1,
  endFrame = 1,
  setFrameMarkerVal,
  timelineRef,
  onPointerDown,
  onPointerMove,
  onPointerLeave,
  hoverFrame = null,
  isScrubbing = false,
  storyboardThumbs = [],
  isPlaying = false,
  setIsPlaying,
  isLooping = false,
  setIsLooping,
  playbackRate = 1,
  setPlaybackRate,
  thumbUrl,
  targetKey = '',
}) {
  const span = Math.max(1, maxFrames - 1);
  const pctOf = (f) => ((clamp(f, 1, maxFrames) - 1) / span) * 100;
  const startPct = pctOf(startFrame);
  const endPct = pctOf(endFrame);
  const currentPct = pctOf(frame);
  const rangeLen = Math.max(0, endFrame - startFrame + 1);
  const rangeShare = Math.round((rangeLen / maxFrames) * 100);

  // ── Ruler ticks, measured from the real track width ───────────────────────
  const [trackW, setTrackW] = useState(0);
  useEffect(() => {
    const el = timelineRef?.current;
    if (!el) return;
    setTrackW(el.getBoundingClientRect().width);
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([e]) => setTrackW(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [timelineRef, maxFrames]);

  const ticks = useMemo(() => {
    if (!trackW || maxFrames < 2) return { major: [], minor: [] };
    const want = Math.max(2, Math.floor(trackW / 92));
    const raw = maxFrames / fps / want;
    const sec = NICE_SECONDS.find((s) => s >= raw) ?? NICE_SECONDS[NICE_SECONDS.length - 1];
    const step = Math.max(1, Math.round(sec * fps));
    const major = [];
    const minor = [];
    for (let f = 1; f <= maxFrames; f += step) {
      major.push(f);
      for (let k = 1; k < MINOR_PER_MAJOR; k++) {
        const mf = f + (step * k) / MINOR_PER_MAJOR;
        if (mf < maxFrames) minor.push(Math.round(mf));
      }
    }
    return { major, minor };
  }, [trackW, maxFrames, fps]);

  // ── Chapter markers, per clip ─────────────────────────────────────────────
  const [markers, setMarkers] = useState([]);
  useEffect(() => {
    try {
      const raw = localStorage.getItem(markersKey(targetKey));
      const list = raw ? JSON.parse(raw) : [];
      setMarkers(Array.isArray(list) ? list.filter((n) => Number.isFinite(n)) : []);
    } catch { setMarkers([]); }
  }, [targetKey]);

  const persistMarkers = (list) => {
    const next = [...new Set(list)].filter((f) => f >= 1 && f <= maxFrames).sort((a, b) => a - b);
    setMarkers(next);
    try {
      if (next.length) localStorage.setItem(markersKey(targetKey), JSON.stringify(next));
      else localStorage.removeItem(markersKey(targetKey));
    } catch { /* private mode / quota — markers are a convenience, not state we own */ }
  };

  const markerAtPlayhead = markers.includes(frame);
  const toggleMarker = () =>
    persistMarkers(markerAtPlayhead ? markers.filter((m) => m !== frame) : [...markers, frame]);

  // 'M' toggles a marker at the playhead. Same guards as the app's global
  // shortcut handler so it never fires while typing or behind a modal. The
  // handler is reached through a ref so the listener is bound once instead of
  // re-subscribing on every frame the playback loop paints.
  const toggleRef = useRef(toggleMarker);
  toggleRef.current = toggleMarker;
  useEffect(() => {
    const onKey = (e) => {
      if (e.key.toLowerCase() !== 'm' || e.ctrlKey || e.metaKey || e.altKey) return;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
      if (document.querySelector('[role="dialog"]')) return;
      e.preventDefault();
      toggleRef.current();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const jumpMarker = (dir) => {
    const next = dir > 0 ? markers.find((m) => m > frame) : [...markers].reverse().find((m) => m < frame);
    if (next !== undefined) setFrame(next);
  };

  // ── Frame read-out field ──────────────────────────────────────────────────
  const [frameDraft, setFrameDraft] = useState(null);

  // Labels at the very edges would hang off the track; anchor those to the edge
  // instead of centring them.
  const anchor = (pct) => (pct < 3 ? 'translateX(0)' : pct > 97 ? 'translateX(-100%)' : 'translateX(-50%)');

  const step = (d) => setFrame((f) => clamp(f + d, 1, maxFrames));

  return (
    <div className="space-y-2.5 select-none">
      {/* ── Header: identity + the single, editable statement of the range ── */}
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <div className="flex items-baseline gap-2.5 min-w-0">
          <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">Timeline</span>
          <span className="font-mono text-[10px] tabular-nums text-[var(--text-muted)]/55 truncate">
            {maxFrames.toLocaleString()} frames · {fps} fps · {fmtTC(maxFrames, fps)}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <RangeField label="In" value={startFrame} min={1} max={endFrame}
                      onCommit={(v) => setFrameMarkerVal('start', v)} />
          <RangeField label="Out" value={endFrame} min={startFrame} max={maxFrames}
                      onCommit={(v) => setFrameMarkerVal('end', v)} />
          <span className="inline-flex items-baseline gap-1.5 rounded-lg border border-[var(--accent)]/25 bg-[var(--accent)]/[0.08] px-2 py-1 font-mono text-[11px] tabular-nums text-[var(--accent)]"
                title="Length of the selected range — this is what gets rendered">
            {fmtTC(rangeLen, fps)}
            <span className="text-[9px] opacity-60">{rangeLen.toLocaleString()} f · {rangeShare}%</span>
          </span>
        </div>
      </div>

      {/* ── Ruler: real ticks + marker flags ─────────────────────────────── */}
      <div className="relative h-6">
        <div className="absolute inset-x-0 bottom-0 h-px bg-[var(--border-color)]" />
        {ticks.minor.map((f, i) => (
          <span key={`n${i}`} className="absolute bottom-0 w-px h-1.5 bg-[var(--text-muted)]/20"
                style={{ left: `${pctOf(f)}%` }} />
        ))}
        {ticks.major.map((f) => {
          const pct = pctOf(f);
          return (
            <React.Fragment key={f}>
              <span className="absolute bottom-0 w-px h-2.5 bg-[var(--text-muted)]/40" style={{ left: `${pct}%` }} />
              <span className="absolute bottom-3 font-mono text-[9px] tabular-nums text-[var(--text-muted)]/60 whitespace-nowrap"
                    style={{ left: `${pct}%`, transform: anchor(pct) }}>
                {fmtTC(f, fps)}
              </span>
            </React.Fragment>
          );
        })}

        {/* Chapter markers — click to jump, alt-click to delete. */}
        {markers.map((m) => (
          <button
            key={m}
            type="button"
            onClick={(e) => { if (e.altKey) persistMarkers(markers.filter((x) => x !== m)); else setFrame(m); }}
            title={`Marker at frame ${m} (${fmtTC(m, fps)}) — click to jump, alt-click to remove`}
            className="absolute bottom-0 -translate-x-1/2 h-3.5 w-2.5 z-10 grid place-items-end"
            style={{ left: `${pctOf(m)}%` }}
          >
            <span className={`block h-2.5 w-2 rounded-[2px] transition-transform hover:scale-125 ${
              m === frame ? 'bg-amber-300' : 'bg-amber-400/70'}`} />
          </button>
        ))}
      </div>

      {/* ── Track ────────────────────────────────────────────────────────── */}
      <div className="relative">
        {/* Hover scrub thumbnail */}
        {hoverFrame !== null && thumbUrl && (
          <div
            className="absolute bottom-[104px] z-50 flex flex-col items-center pointer-events-none -translate-x-1/2"
            style={{ left: `${clamp(pctOf(hoverFrame), 9, 91)}%` }}
          >
            <div className="rounded-xl border border-[var(--border-strong)] bg-[var(--card-bg)] p-1.5 backdrop-blur-xl shadow-[0_10px_30px_rgba(0,0,0,0.55)]">
              <img
                src={thumbUrl(hoverFrame)}
                alt=""
                className="w-44 h-[99px] object-cover rounded-lg bg-black/60"
                onError={(e) => { e.currentTarget.style.visibility = 'hidden'; }}
                onLoad={(e) => { e.currentTarget.style.visibility = 'visible'; }}
              />
              <div className="mt-1 flex items-baseline justify-between px-0.5 font-mono text-[10px] tabular-nums">
                <span className="text-[var(--text-main)] font-semibold">{fmtTC(hoverFrame, fps)}</span>
                <span className="text-[var(--text-muted)]/60">f {hoverFrame.toLocaleString()}</span>
              </div>
            </div>
            <span className="w-2.5 h-2.5 rotate-45 -mt-[6px] border-b border-r border-[var(--border-strong)] bg-[var(--card-bg)]" />
          </div>
        )}

        <div
          ref={timelineRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerLeave={onPointerLeave}
          className="relative h-24 w-full rounded-xl bg-[var(--input-bg)] border border-[var(--border-color)] overflow-hidden cursor-ew-resize timeline-track"
        >
          {/* Filmstrip. Shown at full strength — the old 45% wash made every clip
              look like grey mud; legibility comes from the scrim below instead. */}
          {storyboardThumbs.length > 0 && (
            <div className="absolute inset-0 flex pointer-events-none">
              {storyboardThumbs.map((url, i) => (
                <img key={i} src={url} alt="" loading="lazy"
                     className="flex-1 min-w-0 h-full object-cover"
                     onError={(e) => { e.currentTarget.style.opacity = '0'; }} />
              ))}
            </div>
          )}
          <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-black/70 via-black/10 to-black/25" />

          {/* Out-of-range: dimmed AND desaturated, the way an NLE greys the
              material you have trimmed away — reads instantly as "not rendered". */}
          <div className="absolute inset-y-0 left-0 z-10 pointer-events-none bg-black/55 backdrop-saturate-0"
               style={{ width: `${startPct}%` }} />
          <div className="absolute inset-y-0 right-0 z-10 pointer-events-none bg-black/55 backdrop-saturate-0"
               style={{ left: `${endPct}%` }} />

          {/* Active range rails */}
          <div className="absolute z-10 pointer-events-none border-y-2 border-[var(--accent)]/70 inset-y-0"
               style={{ left: `${startPct}%`, width: `${Math.max(0, endPct - startPct)}%` }} />

          {/* In / Out handles */}
          {[['start', startPct], ['end', endPct]].map(([which, pct]) => (
            <div key={which} className="absolute inset-y-0 z-20 w-[3px] -translate-x-1/2 bg-white/90 pointer-events-none"
                 style={{ left: `${pct}%` }}>
              <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-9 w-2.5 rounded-full bg-white shadow-[0_1px_4px_rgba(0,0,0,0.7)] grid place-items-center gap-[2px]">
                <span className="block h-[7px] w-px bg-black/25" />
              </span>
              <span className={`absolute top-1 text-[8px] font-bold uppercase tracking-wider text-white/70 ${
                which === 'start' ? 'left-1.5' : 'right-1.5'}`}>
                {which === 'start' ? 'In' : 'Out'}
              </span>
            </div>
          ))}

          {/* Marker ticks carried down onto the track */}
          {markers.map((m) => (
            <span key={m} className="absolute inset-y-0 w-px bg-amber-300/45 z-10 pointer-events-none"
                  style={{ left: `${pctOf(m)}%` }} />
          ))}

          {/* Hover line */}
          {hoverFrame !== null && (
            <div className="absolute inset-y-0 w-px -translate-x-1/2 bg-white/35 z-20 pointer-events-none"
                 style={{ left: `${pctOf(hoverFrame)}%` }} />
          )}

          {/* Playhead */}
          <div className={`absolute inset-y-0 z-30 pointer-events-none ${isScrubbing ? '' : 'transition-[left] duration-100 ease-out'}`}
               style={{ left: `${currentPct}%` }}>
            <div className="absolute inset-y-0 left-0 -translate-x-1/2 w-[2px] bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.45)]" />
            <div className="absolute -top-px left-0 -translate-x-1/2 h-0 w-0 border-x-[6px] border-x-transparent border-t-[8px] border-t-[var(--accent)]" />
            <div className="absolute bottom-1 left-0 -translate-x-1/2 h-2 w-2 rounded-full bg-white shadow-[0_1px_3px_rgba(0,0,0,0.6)]" />
          </div>
        </div>

        {/* Playhead timecode, riding under the track */}
        <div className={`absolute -bottom-2 z-30 pointer-events-none ${isScrubbing ? '' : 'transition-[left] duration-100 ease-out'}`}
             style={{ left: `${currentPct}%`, transform: anchor(currentPct) }}>
          <span className="rounded-md bg-[var(--accent)] px-1.5 py-0.5 font-mono text-[9px] font-bold tabular-nums text-white shadow-[0_2px_6px_rgba(0,0,0,0.5)]">
            {fmtTCF(frame, fps)}
          </span>
        </div>
      </div>

      {/* ── Toolbar: one surface, three zones ────────────────────────────── */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-2.5 rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] px-3 py-2">
        {/* Read-out */}
        <div className="flex items-baseline gap-2 font-mono">
          <span className="text-[15px] font-semibold tabular-nums text-[var(--text-main)]">{fmtTC(frame, fps)}</span>
          <span className="text-[11px] tabular-nums text-[var(--text-muted)]/45">/ {fmtTC(maxFrames, fps)}</span>
          <span className="mx-1 h-4 w-px bg-[var(--border-color)] self-center" />
          <input
            type="number"
            min={1}
            max={maxFrames}
            value={frameDraft ?? frame}
            onFocus={(e) => { setFrameDraft(String(frame)); e.target.select(); }}
            onChange={(e) => setFrameDraft(e.target.value)}
            onBlur={() => {
              if (frameDraft !== null) setFrame(clamp(parseInt(frameDraft, 10) || frame, 1, maxFrames));
              setFrameDraft(null);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur();
              if (e.key === 'Escape') { setFrameDraft(null); e.currentTarget.blur(); }
            }}
            className="w-[8ch] rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] py-0.5 text-center text-[11px] font-semibold tabular-nums text-[var(--text-main)] outline-none transition-colors focus:border-[var(--accent)]"
            title="Type a frame number and press Enter to jump"
          />
          <span className="text-[10px] tabular-nums text-[var(--text-muted)]/40">/ {maxFrames.toLocaleString()}</span>
        </div>

        {/* Transport */}
        <div className="spring-cluster flex items-center gap-0.5">
          <IconBtn title="Jump to In point" onClick={() => setFrame(startFrame)}>
            <Ico d={<><polygon points="19 20 9 12 19 4" fill="currentColor" stroke="none" /><line x1="5" y1="19" x2="5" y2="5" /></>} />
          </IconBtn>
          <IconBtn title="Previous marker" onClick={() => jumpMarker(-1)} className={markers.length ? '' : 'opacity-30'}>
            <Ico d="M15 6l-6 6 6 6" />
          </IconBtn>
          <IconBtn title="Previous frame (←)" onClick={() => step(-1)}>
            <Ico d={<><rect x="5" y="5" width="2.4" height="14" rx="1" fill="currentColor" stroke="none" /><polygon points="20 5 20 19 10 12" fill="currentColor" stroke="none" /></>} />
          </IconBtn>

          <button
            type="button"
            onClick={() => setIsPlaying(!isPlaying)}
            title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
            className={`mx-1 grid place-items-center h-10 w-10 rounded-full transition-colors ${
              isPlaying
                ? 'bg-[var(--surface-2)] border border-[var(--border-strong)] text-[var(--text-main)] hover:bg-white/[0.08]'
                : 'bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] shadow-[0_2px_10px_var(--accent-glow)]'
            }`}
          >
            {isPlaying
              ? <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16" rx="1.2" /><rect x="14" y="4" width="4" height="16" rx="1.2" /></svg>
              : <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" className="ml-0.5"><polygon points="6 3 21 12 6 21" /></svg>}
          </button>

          <IconBtn title="Next frame (→)" onClick={() => step(1)}>
            <Ico d={<><rect x="16.6" y="5" width="2.4" height="14" rx="1" fill="currentColor" stroke="none" /><polygon points="4 5 4 19 14 12" fill="currentColor" stroke="none" /></>} />
          </IconBtn>
          <IconBtn title="Next marker" onClick={() => jumpMarker(1)} className={markers.length ? '' : 'opacity-30'}>
            <Ico d="M9 6l6 6-6 6" />
          </IconBtn>
          <IconBtn title="Jump to Out point" onClick={() => setFrame(endFrame)}>
            <Ico d={<><polygon points="5 4 15 12 5 20" fill="currentColor" stroke="none" /><line x1="19" y1="5" x2="19" y2="19" /></>} />
          </IconBtn>
        </div>

        {/* Range + playback options */}
        <div className="flex items-center gap-2">
          <div className="spring-cluster flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5">
            <button type="button" onClick={() => setFrameMarkerVal('start', frame)}
                    title="Set In point to the current frame ([)"
                    className="px-2 py-1 rounded-md text-[11px] font-semibold text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              Set in
            </button>
            <button type="button" onClick={() => setFrameMarkerVal('end', frame)}
                    title="Set Out point to the current frame (])"
                    className="px-2 py-1 rounded-md text-[11px] font-semibold text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              Set out
            </button>
            <button type="button"
                    onClick={async () => { await setFrameMarkerVal('start', 1); await setFrameMarkerVal('end', maxFrames); }}
                    title="Reset the range to the whole clip (R)"
                    className="px-2 py-1 rounded-md text-[11px] font-semibold text-[var(--text-muted)]/70 hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              Full
            </button>
          </div>

          <div className="spring-cluster flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5">
            <IconBtn title={markerAtPlayhead ? 'Remove the marker here (M)' : 'Add a marker at the playhead (M)'}
                     onClick={toggleMarker} active={markerAtPlayhead}>
              <Ico size={13} d={<path d="M5 3v18l7-5 7 5V3z" fill={markerAtPlayhead ? 'currentColor' : 'none'} />} />
            </IconBtn>
            {markers.length > 0 && (
              <button type="button" onClick={() => persistMarkers([])}
                      title={`Clear all ${markers.length} markers`}
                      className="px-1.5 py-1 rounded-md font-mono text-[10px] font-bold tabular-nums text-amber-300/80 hover:text-amber-200 hover:bg-white/[0.06] transition-colors">
                {markers.length}✕
              </button>
            )}
          </div>

          <div className="spring-cluster flex items-center rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5" title="Playback speed">
            {SPEEDS.map((r) => (
              <button key={r} type="button" onClick={() => setPlaybackRate(r)}
                      className={`px-1.5 py-1 rounded-md text-[10px] font-bold tabular-nums transition-colors ${
                        playbackRate === r
                          ? 'bg-[var(--accent)] text-white'
                          : 'text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06]'}`}>
                {r}×
              </button>
            ))}
          </div>

          <IconBtn title="Loop playback over the selected range" onClick={() => setIsLooping(!isLooping)} active={isLooping}
                   className={`border ${isLooping ? 'border-[var(--accent)]/30' : 'border-[var(--border-color)]'}`}>
            <Ico d={<><polyline points="17 1 21 5 17 9" /><path d="M3 11V9a4 4 0 0 1 4-4h14" /><polyline points="7 23 3 19 7 15" /><path d="M21 13v2a4 4 0 0 1-4 4H3" /></>} />
          </IconBtn>
        </div>
      </div>
    </div>
  );
}
