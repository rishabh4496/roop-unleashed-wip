import React, { useEffect, useMemo, useRef, useState } from 'react';
import { getJSON } from '../../api';

/**
 * Diagnostics shown inside the Preview box while a job runs.
 *
 * The processing panel used to be a status line, a bar and a short console
 * floating in a mostly empty 16:9 box. A long run is exactly when you want to
 * know WHY it is going at the speed it is, so that space now carries the
 * numbers that answer it:
 *
 *   - throughput over time, so a stall or a slow decline (thermal, EcoQoS, a
 *     heavier stretch of footage) is visible as a shape rather than guessed at
 *     from one flickering FPS readout;
 *   - GPU utilisation next to VRAM — a busy GPU means the model is the ceiling,
 *     an idle one under full load means the ceiling is elsewhere (lock, decode,
 *     thread starvation), which are opposite fixes;
 *   - the per-stage cost breakdown, when ROOP_PROFILE=1 is set;
 *   - the settings actually in force, so a screenshot of a run explains itself.
 */

/** Fixed-width sparkline. `data` is plain numbers, oldest first. */
function Spark({ data, max, color = 'var(--accent)', height = 34 }) {
  const w = 100;
  const hi = Math.max(max || 0, ...data, 1e-6);
  const pts = data.map((v, i) => {
    const x = data.length === 1 ? w : (i / (data.length - 1)) * w;
    const y = height - Math.max(0, Math.min(v / hi, 1)) * height;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  return (
    <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none"
         className="w-full block" style={{ height }} aria-hidden="true">
      {data.length > 1 && (
        <>
          <polyline points={`0,${height} ${pts.join(' ')} ${w},${height}`}
                    fill={color} opacity="0.13" stroke="none" />
          <polyline points={pts.join(' ')} fill="none" stroke={color}
                    strokeWidth="1.5" vectorEffect="non-scaling-stroke"
                    strokeLinejoin="round" strokeLinecap="round" />
        </>
      )}
    </svg>
  );
}

/** Labelled value + bar, used for the hardware meters. */
function Meter({ label, value, sub, pct, color, series, max }) {
  return (
    <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] px-2.5 py-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[9px] font-semibold uppercase tracking-[0.14em] text-white/35">{label}</span>
        <span className="text-[11px] font-mono font-bold tabular-nums" style={{ color }}>{value}</span>
      </div>
      {series ? (
        <div className="mt-1 -mx-0.5"><Spark data={series} max={max} color={color} height={26} /></div>
      ) : (
        <div className="mt-1.5 h-1 w-full rounded-full bg-white/[0.06] overflow-hidden">
          <div className="h-full rounded-full transition-[width] duration-500"
               style={{ width: `${Math.max(1, Math.min(pct ?? 0, 100))}%`, background: color }} />
        </div>
      )}
      {sub && <div className="mt-1 text-[9px] font-mono text-white/25 tabular-nums">{sub}</div>}
    </div>
  );
}

const STAGE_COLORS = {
  detect: '#f59e0b', mask: '#a78bfa', swap: '#ef4444', enhance: '#22d3ee',
  decode: '#64748b', frame_total: '#475569', encode: '#38bdf8',
};
const stageColor = (s) => STAGE_COLORS[s] || (
  /detect|track/.test(s) ? STAGE_COLORS.detect
  : /mask/.test(s) ? STAGE_COLORS.mask
  : /swap/.test(s) ? STAGE_COLORS.swap
  : /enhance|upscal/.test(s) ? STAGE_COLORS.enhance
  : /decode|read/.test(s) ? STAGE_COLORS.decode
  : '#94a3b8');

/** One figure in the run-stats strip. */
function Stat({ label, value, tone = 'text-white/80', sub }) {
  return (
    <div className="min-w-0 px-3 py-1.5">
      <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-white/30">{label}</div>
      <div className={`font-mono text-[13px] font-bold tabular-nums truncate ${tone}`}>{value}</div>
      {sub && <div className="font-mono text-[9px] text-white/25 truncate">{sub}</div>}
    </div>
  );
}

const fmtDur = (ms) => {
  if (!isFinite(ms) || ms <= 0) return '—';
  const s = Math.round(ms / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h > 0 ? `${h}h ${String(m).padStart(2, '0')}m` : m > 0 ? `${m}m ${String(sec).padStart(2, '0')}s` : `${sec}s`;
};

export default function DiagnosticsPanel({ desc = '', telemetry, processing, paused, config = [],
                                           elapsedMs = 0, etaMs = 0, prog = 0 }) {
  // ── Throughput history ────────────────────────────────────────────────────
  // The FPS is already in the status line ("Processing frame X / Y (Z FPS)");
  // sampling it here turns a number that jumps every second into a trend.
  const [fpsHist, setFpsHist] = useState([]);
  const [hwHist, setHwHist] = useState({ gpu: [], vram: [], cpu: [] });
  const lastDescRef = useRef('');

  useEffect(() => {
    if (!processing) { setFpsHist([]); setHwHist({ gpu: [], vram: [], cpu: [] }); return; }
  }, [processing]);

  useEffect(() => {
    if (!processing || paused || !desc || desc === lastDescRef.current) return;
    lastDescRef.current = desc;
    const m = desc.match(/\(([\d.]+)\s*FPS\)/i);
    if (!m) return;
    const v = parseFloat(m[1]);
    if (!isFinite(v)) return;
    setFpsHist((h) => [...h, v].slice(-160));
  }, [desc, processing, paused]);

  useEffect(() => {
    if (!processing || !telemetry) return;
    setHwHist((h) => ({
      gpu: [...h.gpu, telemetry.gpu_util ?? 0].slice(-160),
      vram: [...h.vram, telemetry.vram_used ?? 0].slice(-160),
      cpu: [...h.cpu, telemetry.cpu_percent ?? 0].slice(-160),
    }));
  }, [telemetry, processing]);

  const fpsStats = useMemo(() => {
    if (fpsHist.length === 0) return null;
    const recent = fpsHist.slice(-30);
    return {
      cur: fpsHist[fpsHist.length - 1],
      avg: fpsHist.reduce((a, b) => a + b, 0) / fpsHist.length,
      min: Math.min(...fpsHist),
      max: Math.max(...fpsHist),
      // Spread over the recent window: a wide spread means the pipeline is
      // stalling in bursts rather than running at a steady rate.
      spread: Math.max(...recent) - Math.min(...recent),
    };
  }, [fpsHist]);

  // ── Stage breakdown (ROOP_PROFILE=1) ──────────────────────────────────────
  const [profile, setProfile] = useState(null);
  useEffect(() => {
    if (!processing) return;
    let stop = false;
    const poll = async () => {
      try {
        const d = await getJSON('/api/system/profile');
        if (!stop) setProfile(d);
      } catch { /* quiet — diagnostics must never break the run view */ }
    };
    poll();
    const id = setInterval(poll, 4000);
    return () => { stop = true; clearInterval(id); };
  }, [processing]);

  const vramPct = telemetry?.vram_total ? (telemetry.vram_used / telemetry.vram_total) * 100 : 0;
  const hasGpuUtil = telemetry?.gpu_util !== undefined && telemetry?.gpu_util !== null;

  // ── Frame counts, straight out of the status line ─────────────────────────
  const frames = useMemo(() => {
    const m = desc.match(/(\d[\d,]*)\s*\/\s*(\d[\d,]*)/);
    if (!m) return null;
    const done = parseInt(m[1].replace(/,/g, ''), 10);
    const total = parseInt(m[2].replace(/,/g, ''), 10);
    if (!isFinite(done) || !isFinite(total) || total <= 0) return null;
    return { done, total, left: Math.max(0, total - done) };
  }, [desc]);

  // ── Stall watchdog ────────────────────────────────────────────────────────
  // A run that has quietly wedged looks identical to a slow one: the bar sits
  // still and the ETA keeps counting. Timing how long the status line has been
  // unchanged tells them apart — and a long silence during the swap phase is
  // the signature of the encoder-warm-up / display-power stalls this app has
  // hit before, so it is worth surfacing rather than leaving to be noticed.
  const [stallMs, setStallMs] = useState(0);
  const descAtRef = useRef(Date.now());
  useEffect(() => { descAtRef.current = Date.now(); setStallMs(0); }, [desc]);
  useEffect(() => {
    if (!processing || paused) return;
    const id = setInterval(() => setStallMs(Date.now() - descAtRef.current), 1000);
    return () => clearInterval(id);
  }, [processing, paused]);

  // Wall-clock time the run is projected to finish at — far easier to act on
  // than "32m 30s left" when the answer is "come back after lunch".
  const finishAt = etaMs > 0
    ? new Date(Date.now() + etaMs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '—';

  return (
    <div className="w-full grid gap-2.5 lg:grid-cols-3">
      {/* ── Run stats strip ──────────────────────────────────────────────── */}
      <div className="lg:col-span-3 rounded-xl border border-white/[0.07] bg-black/30 flex flex-wrap items-stretch divide-x divide-white/[0.06]">
        <Stat label="Progress" value={`${(prog * 100).toFixed(1)}%`} tone="text-[var(--accent)]" />
        <Stat label="Frames" value={frames ? frames.done.toLocaleString() : '—'}
              sub={frames ? `of ${frames.total.toLocaleString()}` : null} />
        <Stat label="Remaining" value={frames ? frames.left.toLocaleString() : '—'} sub="frames" />
        <Stat label="Elapsed" value={fmtDur(elapsedMs)} />
        <Stat label="Left" value={fmtDur(etaMs)} tone="text-emerald-400" />
        <Stat label="Finishes" value={finishAt} sub="wall clock" />
        <Stat label="Avg" value={fpsStats ? `${fpsStats.avg.toFixed(1)}` : '—'} sub="fps over run" />
        <Stat label="RAM" value={telemetry?.ram_used != null ? `${telemetry.ram_used} GB` : '—'}
              sub={telemetry?.ram_total ? `of ${telemetry.ram_total} GB` : null} />
        <Stat label="Disk free" value={telemetry?.disk_free != null ? `${telemetry.disk_free} GB` : '—'}
              tone={telemetry?.disk_free != null && telemetry.disk_free < 20 ? 'text-red-400' : 'text-white/80'}
              sub="output drive" />
        <Stat label="Status"
              value={paused ? 'Paused' : stallMs > 90000 ? 'Stalled?' : 'Running'}
              tone={paused ? 'text-amber-400' : stallMs > 90000 ? 'text-red-400' : 'text-emerald-400'}
              sub={stallMs > 15000 ? `no output ${fmtDur(stallMs)}` : null} />
      </div>

      {/* ── Throughput ───────────────────────────────────────────────────── */}
      <div className="lg:col-span-2 rounded-xl border border-white/[0.07] bg-black/30 px-3 py-2.5">
        <div className="flex items-baseline justify-between gap-3">
          <span className="text-[9px] font-semibold uppercase tracking-[0.16em] text-white/35">Throughput</span>
          {fpsStats && (
            <span className="flex items-baseline gap-2.5 font-mono text-[10px] text-white/30 tabular-nums">
              <span className="text-[13px] font-bold text-[var(--accent)]">{fpsStats.cur.toFixed(1)}</span>
              <span>fps</span>
              <span className="text-white/15">·</span>
              <span>avg <span className="text-white/60 font-semibold">{fpsStats.avg.toFixed(1)}</span></span>
              <span>min <span className="text-white/60 font-semibold">{fpsStats.min.toFixed(1)}</span></span>
              <span>max <span className="text-white/60 font-semibold">{fpsStats.max.toFixed(1)}</span></span>
            </span>
          )}
        </div>
        {fpsHist.length > 1 ? (
          <>
            <div className="mt-1.5"><Spark data={fpsHist} height={52} /></div>
            <div className="mt-1 flex items-center justify-between font-mono text-[9px] text-white/25">
              <span>{fpsHist.length} samples</span>
              {fpsStats && fpsStats.spread > fpsStats.avg * 0.6 && (
                <span className="text-amber-400/70">
                  unstable — {fpsStats.spread.toFixed(1)} fps swing, pipeline is stalling in bursts
                </span>
              )}
            </div>
          </>
        ) : (
          <div className="mt-2 h-[52px] flex items-center text-[10px] font-mono text-white/25">
            collecting throughput samples…
          </div>
        )}
      </div>

      {/* ── Hardware ─────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-1 lg:gap-1.5">
        {hasGpuUtil ? (
          <Meter label="GPU" value={`${telemetry.gpu_util}%`} color="#4ade80"
                 series={hwHist.gpu.length > 1 ? hwHist.gpu : null} max={100}
                 pct={telemetry.gpu_util}
                 sub={[telemetry.gpu_temp != null && `${telemetry.gpu_temp}°C`,
                       telemetry.gpu_power != null && `${telemetry.gpu_power}W`,
                       telemetry.gpu_clock != null && `${telemetry.gpu_clock}MHz`,
                       // Memory-bandwidth utilisation, not VRAM occupancy: high
                       // here with low core util means the run is memory-bound.
                       telemetry.gpu_mem_util != null && `mem ${telemetry.gpu_mem_util}%`]
                       .filter(Boolean).join('  ·  ')} />
        ) : (
          <Meter label="GPU" value="n/a" color="#64748b" pct={0} sub="nvidia-smi unavailable" />
        )}
        <Meter label="VRAM" value={`${telemetry?.vram_used ?? 0} GB`} color="#60a5fa"
               pct={vramPct} sub={`of ${telemetry?.vram_total ?? 0} GB  ·  ${Math.round(vramPct)}%`} />
        <Meter label="CPU" value={`${telemetry?.cpu_percent ?? 0}%`} color="#fb923c"
               pct={telemetry?.cpu_percent}
               sub={`${telemetry?.threads ?? 0} python threads`} />
      </div>

      {/* ── Stage cost breakdown ─────────────────────────────────────────── */}
      <div className="lg:col-span-2 rounded-xl border border-white/[0.07] bg-black/30 px-3 py-2.5">
        <div className="flex items-baseline justify-between gap-3">
          <span className="text-[9px] font-semibold uppercase tracking-[0.16em] text-white/35">Stage cost</span>
          <span className="font-mono text-[9px] text-white/25">
            {profile?.enabled ? 'wall-clock summed across threads' : 'profiler off'}
          </span>
        </div>
        {profile?.enabled && profile.stages?.length ? (
          <div className="mt-2 space-y-1">
            {profile.stages.filter((s) => s.stage !== 'frame_total').slice(0, 7).map((s) => (
              <div key={s.stage} className="flex items-center gap-2">
                <span className="w-20 shrink-0 truncate font-mono text-[10px] text-white/50">{s.stage}</span>
                <div className="h-2 flex-1 rounded-full bg-white/[0.05] overflow-hidden">
                  <div className="h-full rounded-full transition-[width] duration-700"
                       style={{ width: `${Math.max(1, s.share * 100)}%`, background: stageColor(s.stage) }} />
                </div>
                <span className="w-9 shrink-0 text-right font-mono text-[10px] font-semibold tabular-nums text-white/60">
                  {Math.round(s.share * 100)}%
                </span>
                <span className="w-16 shrink-0 text-right font-mono text-[9px] tabular-nums text-white/25">
                  {s.ms_per_call}ms
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="mt-2 font-mono text-[10px] leading-relaxed text-white/25">
            Set <span className="text-white/45">ROOP_PROFILE=1</span> before starting the app to break the
            run down by stage (decode / detect / mask / swap / enhance) and see which one owns the time.
          </div>
        )}
      </div>

      {/* ── Effective settings ───────────────────────────────────────────── */}
      <div className="rounded-xl border border-white/[0.07] bg-black/30 px-3 py-2.5">
        <span className="text-[9px] font-semibold uppercase tracking-[0.16em] text-white/35">Run config</span>
        <div className="mt-2 space-y-1">
          {config.length === 0 ? (
            <div className="font-mono text-[10px] text-white/25">—</div>
          ) : config.map(([k, v]) => (
            <div key={k} className="flex items-baseline justify-between gap-2 font-mono text-[10px]">
              <span className="text-white/30">{k}</span>
              <span className="truncate text-right font-semibold text-white/60">{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
