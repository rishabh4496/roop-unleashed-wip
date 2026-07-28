import React, { useEffect, useMemo, useState } from 'react';
import { getJSON, postJSON, fileUrl } from '../api';
import { Button, Card } from './ui';
import { confirmDialog } from './confirm';

/**
 * Run History — a browsable record of every completed swap.
 *
 * The live DiagnosticsPanel answers "how is THIS run going" and then vanishes
 * the moment the job ends. This view is the durable other half: each finished
 * run is kept (backend run_history.json) with the exact settings it used, how
 * long it took, and its average throughput — so "was this slower than last
 * time?" and "what did I change between these two renders?" become one glance
 * instead of guesswork. Any two runs can be diffed field-by-field, and a run's
 * settings can be re-loaded straight back into the Face Swap tab.
 */

// Canonical settings keys → short human labels. Anything not listed still shows
// in the diff under its raw key; this map just gives the common ones nice names
// and drives the at-a-glance chip row on each card.
const LABELS = {
  swap_model: 'Swapper',
  selected_enhancer: 'Enhancer',
  face_detection_mode: 'Detection',
  mask_engine: 'Mask',
  max_threads: 'Threads',
  subsample_upscale: 'Pixel boost',
  video_swapping_method: 'Method',
  output_method: 'Output',
  blend_ratio: 'Blend',
  max_face_distance: 'Face distance',
  detector_engine: 'Detector',
  color_transfer_mode: 'Color transfer',
  codeformer_fidelity: 'CodeFormer fidelity',
  face_detector_size: 'Detector size',
  face_detector_threshold: 'Detector threshold',
  num_swap_steps: 'Swap steps',
  track_identities: 'Identity tracking',
  temporal_detection: 'Temporal detection',
  upscale_after_swap: 'AI upscale pass',
  upscale_model_after: 'Upscale model',
  interp_after_swap: 'Interpolation',
  refine_landmarks: 'Refine landmarks',
  yaw_align: 'Profile alignment',
  rescue_small_faces: 'Small-face rescue',
  jaw_reshape: 'Jaw reshape',
  autorotate_faces: 'Autorotate',
  vr_mode: 'VR mode',
};

// The handful of settings that most define what a run looked like, shown as
// chips on the card. Missing/empty ones are skipped so the row stays tight.
const CHIP_KEYS = ['swap_model', 'selected_enhancer', 'face_detection_mode',
                   'mask_engine', 'subsample_upscale', 'max_threads'];

// Keys that are noise in a side-by-side diff (paths, per-run junk, huge blobs).
const DIFF_SKIP = new Set(['selected_theme', 'output_path', 'source_path',
                           'target_path', 'clear_output', 'output_show_video']);

const fmtDur = (s) => {
  if (!s || s <= 0) return null;
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(sec).padStart(2, '0')}s`;
  return `${sec}s`;
};

const fmtRel = (t) => {
  const d = (Date.now() - t * 1000) / 1000;
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  const days = Math.floor(d / 86400);
  if (days < 7) return `${days}d ago`;
  return new Date(t * 1000).toLocaleDateString();
};

const fmtVal = (v) => {
  if (v === true) return 'on';
  if (v === false) return 'off';
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2);
  return String(v);
};

// Primitive-only view of a run's settings, for chips + diffing. Objects/arrays
// (e.g. face_mapping) are not comparable at a glance, so they're dropped.
const primitives = (settings) => {
  const out = {};
  Object.entries(settings || {}).forEach(([k, v]) => {
    if (DIFF_SKIP.has(k)) return;
    if (v === null || ['string', 'number', 'boolean'].includes(typeof v)) out[k] = v;
  });
  return out;
};

export default function RunHistory({ notify, setSettings, setTab }) {
  const [entries, setEntries] = useState([]);
  const [outputPath, setOutputPath] = useState('');
  const [existing, setExisting] = useState({}); // basename -> kind, for still-present outputs
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [compare, setCompare] = useState([]); // up to two run ids

  const load = async () => {
    setLoading(true);
    try {
      const h = await getJSON('/api/history');
      setEntries(h.entries || []);
    } catch (e) {
      notify?.(e.message, 'error');
    } finally {
      setLoading(false);
    }
    // Output listing is best-effort: it lets us show a thumbnail and flag runs
    // whose file has since been deleted. History works fully without it.
    try {
      const o = await getJSON('/api/output');
      setOutputPath(o.output_path || '');
      const map = {};
      (o.files || []).forEach((f) => { map[f.name] = f.kind; });
      setExisting(map);
    } catch { /* no outputs endpoint / folder */ }
  };

  /* eslint-disable react-hooks/exhaustive-deps -- fetch once on mount */
  useEffect(() => { load(); }, []);
  /* eslint-enable react-hooks/exhaustive-deps */

  const toggleCompare = (id) => {
    setCompare((cur) => {
      if (cur.includes(id)) return cur.filter((x) => x !== id);
      if (cur.length >= 2) return [cur[1], id]; // keep the most recent two picks
      return [...cur, id];
    });
  };

  const loadRunSettings = (entry) => {
    if (!setSettings || !entry?.settings) return;
    setSettings((s) => ({ ...(s || {}), ...entry.settings }));
    notify?.(`Loaded the settings from this run (${new Date(entry.time * 1000).toLocaleString()})`);
    setTab?.('faceswap');
  };

  const revealOutput = async (name) => {
    try {
      await postJSON('/api/reveal', { path: `${outputPath}/${name}` });
    } catch (e) {
      notify?.(e.message, 'error');
    }
  };

  const deleteEntry = async (entry) => {
    if (!(await confirmDialog({
      title: 'Remove from history?',
      message: 'This deletes only the history record. The output file itself is kept.',
      confirmLabel: 'Remove', danger: true,
    }))) return;
    try {
      await postJSON('/api/history/delete', { id: entry.id });
      setEntries((prev) => prev.filter((e) => e.id !== entry.id));
      setCompare((cur) => cur.filter((x) => x !== entry.id));
    } catch (e) {
      notify?.(e.message, 'error');
    }
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter((e) => {
      if ((e.outputs || []).some((n) => n.toLowerCase().includes(q))) return true;
      const p = primitives(e.settings);
      return Object.values(p).some((v) => fmtVal(v).toLowerCase().includes(q));
    });
  }, [entries, query]);

  // Roll-up figures for the header. Only runs that recorded a duration count
  // toward render time / throughput, so a fresh install with old entries reads
  // honestly instead of showing zeros.
  const summary = useMemo(() => {
    const timed = entries.filter((e) => e.duration_s > 0);
    const totalTime = timed.reduce((a, e) => a + e.duration_s, 0);
    const totalFrames = entries.reduce((a, e) => a + (e.frames || 0), 0);
    const fpsVals = entries.filter((e) => e.fps > 0).map((e) => e.fps);
    const avgFps = fpsVals.length ? fpsVals.reduce((a, b) => a + b, 0) / fpsVals.length : 0;
    return { count: entries.length, totalTime, totalFrames, avgFps, medianFps: median(fpsVals) };
  }, [entries]);

  const cmpA = entries.find((e) => e.id === compare[0]);
  const cmpB = entries.find((e) => e.id === compare[1]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white/90">🕐 Run History</h2>
          <p className="text-sm text-white/50">Every completed swap, with the settings and speed it ran at. Pick two to compare.</p>
        </div>
        <Button variant="secondary" onClick={load} disabled={loading}>🔄 Refresh</Button>
      </div>

      {/* Summary strip */}
      {entries.length > 0 && (
        <div className="rounded-xl border border-white/[0.07] bg-black/30 flex flex-wrap items-stretch divide-x divide-white/[0.06]">
          <Sum label="Runs" value={summary.count} />
          <Sum label="Total render time" value={fmtDur(summary.totalTime) || '—'} />
          <Sum label="Frames rendered" value={summary.totalFrames ? summary.totalFrames.toLocaleString() : '—'} />
          <Sum label="Median throughput" value={summary.medianFps ? `${summary.medianFps.toFixed(1)} fps` : '—'}
               tone="text-emerald-400" />
        </div>
      )}

      {/* Compare panel */}
      {cmpA && cmpB && <ComparePanel a={cmpA} b={cmpB} onClear={() => setCompare([])} onLoad={loadRunSettings} />}

      {/* Search */}
      {entries.length > 0 && (
        <input
          type="text"
          placeholder="Search by output name or a setting value…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full md:max-w-md px-3 py-2 rounded-xl glass-input text-white text-sm focus:outline-none"
        />
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-20 gap-3">
          <div className="h-8 w-8 rounded-full border-4 border-white/10 border-t-[var(--accent)] animate-spin" />
          <span className="text-white/40 text-sm font-medium">Loading run history…</span>
        </div>
      )}

      {!loading && entries.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 border-2 border-dashed border-white/10 rounded-2xl text-center">
          <span className="text-4xl mb-2">🕐</span>
          <span className="text-white/60 text-sm font-medium">No runs recorded yet</span>
          <span className="text-white/35 text-xs mt-1">Finish a swap and it will appear here with its settings and timing.</span>
        </div>
      )}

      {!loading && entries.length > 0 && filtered.length === 0 && (
        <div className="text-center text-white/40 text-sm py-10">No runs match “{query}”.</div>
      )}

      {/* Run list */}
      {!loading && filtered.length > 0 && (
        <div className="space-y-3">
          {filtered.map((entry) => (
            <RunRow
              key={entry.id}
              entry={entry}
              outputPath={outputPath}
              existing={existing}
              selected={compare.includes(entry.id)}
              compareFull={compare.length >= 2 && !compare.includes(entry.id)}
              onToggleCompare={() => toggleCompare(entry.id)}
              onLoad={() => loadRunSettings(entry)}
              onReveal={revealOutput}
              onDelete={() => deleteEntry(entry)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function Sum({ label, value, tone = 'text-white/85' }) {
  return (
    <div className="px-4 py-2.5 min-w-0">
      <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-white/30">{label}</div>
      <div className={`font-mono text-[15px] font-bold tabular-nums truncate ${tone}`}>{value}</div>
    </div>
  );
}

function Chip({ k, v }) {
  return (
    <span className="inline-flex items-baseline gap-1 px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.07] text-[10px]">
      <span className="text-white/35">{LABELS[k] || k}</span>
      <span className="font-semibold text-white/70">{fmtVal(v)}</span>
    </span>
  );
}

function RunRow({ entry, outputPath, existing, selected, compareFull, onToggleCompare, onLoad, onReveal, onDelete }) {
  const outputs = entry.outputs || [];
  // Pick the first output that still exists on disk for the thumbnail.
  const liveName = outputs.find((n) => existing[n]);
  const kind = liveName ? existing[liveName] : null;
  const allDeleted = outputs.length > 0 && !liveName;

  const p = primitives(entry.settings);
  const chips = CHIP_KEYS.filter((k) => p[k] !== undefined && p[k] !== '' && p[k] !== null);

  return (
    <Card className={`p-3 border transition-colors ${selected ? 'border-[var(--accent)]/50 bg-[var(--accent)]/[0.04]' : 'border-white/[0.06]'}`}>
      <div className="flex gap-3">
        {/* Thumbnail */}
        <div className="relative w-32 h-20 shrink-0 rounded-lg overflow-hidden bg-black/45 border border-white/5 grid place-items-center">
          {liveName ? (
            kind === 'video' ? (
              <video src={fileUrl(`${outputPath}/${liveName}`)} muted playsInline preload="metadata"
                     className="w-full h-full object-contain" />
            ) : (
              <img src={fileUrl(`${outputPath}/${liveName}`)} alt={liveName} loading="lazy"
                   className="w-full h-full object-contain" />
            )
          ) : (
            <span className="text-white/20 text-2xl">{allDeleted ? '🗑️' : '🎬'}</span>
          )}
        </div>

        {/* Body */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-semibold text-white/85 text-sm truncate max-w-[22rem]" title={outputs.join(', ')}>
                  {outputs[0] || 'run'}
                </span>
                {outputs.length > 1 && (
                  <span className="text-[10px] text-white/40">+{outputs.length - 1} more</span>
                )}
                {allDeleted && (
                  <span className="text-[9px] uppercase tracking-wide text-amber-400/70 border border-amber-400/20 rounded px-1 py-0.5">file removed</span>
                )}
              </div>
              <div className="mt-0.5 flex items-center gap-2 text-[11px] text-white/40 font-mono">
                <span title={new Date(entry.time * 1000).toLocaleString()}>{fmtRel(entry.time)}</span>
                {entry.duration_s > 0 && (<><span className="text-white/15">·</span><span>{fmtDur(entry.duration_s)}</span></>)}
                {entry.fps > 0 && (
                  <>
                    <span className="text-white/15">·</span>
                    <span className={entry.fps < 6 ? 'text-amber-400/80' : 'text-emerald-400/80'}>{entry.fps.toFixed(1)} fps</span>
                  </>
                )}
                {entry.frames > 0 && (<><span className="text-white/15">·</span><span>{entry.frames.toLocaleString()} frames</span></>)}
              </div>
            </div>
          </div>

          {/* Setting chips */}
          <div className="mt-2 flex flex-wrap gap-1.5">
            {chips.length ? chips.map((k) => <Chip key={k} k={k} v={p[k]} />)
              : <span className="text-[11px] text-white/25">no settings recorded</span>}
          </div>

          {/* Actions */}
          <div className="mt-2.5 flex items-center gap-1.5 flex-wrap">
            <button
              type="button"
              onClick={onToggleCompare}
              disabled={compareFull}
              title={compareFull ? 'Two runs already selected — clear one first' : 'Select this run to compare'}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold border transition-colors disabled:opacity-30 ${
                selected
                  ? 'bg-[var(--accent)]/20 border-[var(--accent)]/40 text-[var(--accent)]'
                  : 'bg-white/[0.04] border-white/10 text-white/55 hover:text-white hover:border-white/20'
              }`}
            >
              {selected ? '✓ Comparing' : '⇄ Compare'}
            </button>
            <button type="button" onClick={onLoad}
                    className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-white/[0.04] border border-white/10 text-white/55 hover:text-white hover:border-white/20 transition-colors">
              ↻ Load settings
            </button>
            {liveName && (
              <button type="button" onClick={() => onReveal(liveName)}
                      className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-white/[0.04] border border-white/10 text-white/55 hover:text-white hover:border-white/20 transition-colors">
                📁 Reveal
              </button>
            )}
            <button type="button" onClick={onDelete}
                    className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-white/[0.02] border border-white/5 text-white/35 hover:text-red-300 hover:border-red-500/30 transition-colors ml-auto">
              Remove
            </button>
          </div>
        </div>
      </div>
    </Card>
  );
}

function ComparePanel({ a, b, onClear, onLoad }) {
  // Field-by-field diff: the union of primitive keys across both runs, split
  // into those that differ (the interesting part) and those that match.
  const { changed, same } = useMemo(() => {
    const pa = primitives(a.settings), pb = primitives(b.settings);
    const keys = Array.from(new Set([...Object.keys(pa), ...Object.keys(pb)])).sort();
    const changed = [], same = [];
    keys.forEach((k) => {
      const va = pa[k], vb = pb[k];
      (fmtVal(va) === fmtVal(vb) ? same : changed).push({ k, va, vb });
    });
    return { changed, same };
  }, [a, b]);

  const [showSame, setShowSame] = useState(false);

  const Head = ({ e, side }) => (
    <div className="min-w-0">
      <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-white/30">{side}</div>
      <div className="font-semibold text-white/85 text-sm truncate" title={(e.outputs || []).join(', ')}>
        {(e.outputs || [])[0] || 'run'}
      </div>
      <div className="mt-0.5 flex items-center gap-2 text-[10px] font-mono text-white/40">
        <span>{new Date(e.time * 1000).toLocaleString()}</span>
        {e.duration_s > 0 && <span>· {fmtDur(e.duration_s)}</span>}
        {e.fps > 0 && <span className="text-emerald-400/80">· {e.fps.toFixed(1)} fps</span>}
      </div>
      <button type="button" onClick={() => onLoad(e)}
              className="mt-1.5 px-2 py-0.5 rounded-md text-[10px] font-semibold bg-white/[0.05] border border-white/10 text-white/60 hover:text-white transition-colors">
        ↻ Load these settings
      </button>
    </div>
  );

  return (
    <Card className="p-4 border border-[var(--accent)]/25 bg-[var(--accent)]/[0.03]">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="grid grid-cols-2 gap-4 flex-1 min-w-0">
          <Head e={a} side="Run A" />
          <Head e={b} side="Run B" />
        </div>
        <button type="button" onClick={onClear}
                className="shrink-0 text-white/40 hover:text-white text-lg leading-none px-1" title="Close comparison">✕</button>
      </div>

      {/* Speed delta */}
      {a.fps > 0 && b.fps > 0 && (
        <div className="mb-3 text-[11px] font-mono text-white/50">
          {b.fps === a.fps ? 'Same throughput.' : (
            <>Run B is <span className={b.fps > a.fps ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold'}>
              {b.fps > a.fps ? `${((b.fps / a.fps - 1) * 100).toFixed(0)}% faster` : `${((1 - b.fps / a.fps) * 100).toFixed(0)}% slower`}
            </span> than Run A ({a.fps.toFixed(1)} → {b.fps.toFixed(1)} fps).</>
          )}
        </div>
      )}

      {/* Changed settings */}
      <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-white/35 mb-1.5">
        {changed.length} setting{changed.length === 1 ? '' : 's'} differ
      </div>
      {changed.length === 0 ? (
        <div className="text-[12px] text-white/40">These two runs used identical settings.</div>
      ) : (
        <div className="rounded-lg border border-white/[0.07] overflow-hidden divide-y divide-white/[0.05]">
          {changed.map(({ k, va, vb }) => (
            <div key={k} className="grid grid-cols-[1fr_1fr_1fr] gap-2 px-3 py-1.5 text-[11px] font-mono items-baseline">
              <span className="text-white/40 truncate" title={k}>{LABELS[k] || k}</span>
              <span className="text-amber-300/80 truncate text-right">{fmtVal(va)}</span>
              <span className="text-emerald-300/80 truncate text-right">{fmtVal(vb)}</span>
            </div>
          ))}
        </div>
      )}

      {same.length > 0 && (
        <div className="mt-2">
          <button type="button" onClick={() => setShowSame((v) => !v)}
                  className="text-[10px] font-semibold text-white/35 hover:text-white/60 transition-colors">
            {showSame ? '▾ Hide' : '▸ Show'} {same.length} matching setting{same.length === 1 ? '' : 's'}
          </button>
          {showSame && (
            <div className="mt-1.5 rounded-lg border border-white/[0.05] overflow-hidden divide-y divide-white/[0.04]">
              {same.map(({ k, va }) => (
                <div key={k} className="grid grid-cols-[1fr_2fr] gap-2 px-3 py-1 text-[11px] font-mono items-baseline">
                  <span className="text-white/30 truncate" title={k}>{LABELS[k] || k}</span>
                  <span className="text-white/50 truncate text-right">{fmtVal(va)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function median(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
