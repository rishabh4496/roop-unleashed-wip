import React, { useEffect, useRef } from 'react';

/**
 * Terminal-style live feed shown inside the Preview box while a job runs.
 *
 * Renders the rolling status tail from `/api/progress`'s `log` array (each entry
 * `{ t, msg, seq }`) as a scrolling console, so the preview box mirrors what the
 * real terminal is printing (stage changes, "Processing frame X / Y (Z FPS)",
 * combine/encode, done/errors). Auto-scrolls to the newest line and pins a
 * blinking cursor to the bottom so it reads as a live stream.
 */
export default function ProcessingTerminal({ log = [], paused }) {
  const scrollRef = useRef(null);
  const bottomRef = useRef(null);

  // Keep the newest line in view. Only auto-scroll when the user is already near
  // the bottom, so scrolling up to read history isn't yanked back down.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    if (nearBottom) bottomRef.current?.scrollIntoView({ block: 'end' });
  }, [log.length, log[log.length - 1]?.seq]);

  const lineTone = (msg) => {
    const m = (msg || '').toLowerCase();
    if (m.startsWith('⚠') || /error|fail|abort/.test(m)) return 'text-red-400';
    if (m.startsWith('✓') || /\bdone\b/.test(m)) return 'text-emerald-400';
    if (m.startsWith('▶') || /start/.test(m)) return 'text-[var(--accent)]';
    if (/combin|encod|audio|mux|finaliz/.test(m)) return 'text-sky-300/90';
    if (/upscal|interpolat/.test(m)) return 'text-fuchsia-300/85';
    return 'text-white/70';
  };

  return (
    <div className="w-full rounded-xl border border-white/10 bg-black/60 overflow-hidden font-mono">
      {/* Title bar — traffic lights + live indicator */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-white/10 bg-white/[0.03]">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/70" />
        </span>
        <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-white/40">console</span>
        <span className="ml-auto flex items-center gap-1.5 text-[10px] text-white/40">
          <span className={`h-1.5 w-1.5 rounded-full ${paused ? 'bg-amber-400' : 'bg-emerald-400 animate-pulse'}`} />
          {paused ? 'paused' : 'live'}
        </span>
      </div>

      {/* Scrolling log body */}
      <div ref={scrollRef} className="selectable h-40 overflow-y-auto px-3 py-2 text-[11px] leading-relaxed scroll-smooth">
        {log.length === 0 ? (
          <div className="text-white/30">waiting for output…</div>
        ) : (
          log.map((l) => (
            <div key={l.seq} className="flex gap-2 whitespace-pre-wrap break-words">
              <span className="shrink-0 text-white/25 tabular-nums">{l.t}</span>
              <span className="shrink-0 text-white/25">›</span>
              <span className={lineTone(l.msg)}>{l.msg}</span>
            </div>
          ))
        )}
        {/* Blinking cursor pinned to the tail */}
        <div ref={bottomRef} className="flex gap-2">
          <span className="text-white/25">›</span>
          <span className="inline-block h-3.5 w-2 bg-[var(--accent)]/80 animate-pulse" />
        </div>
      </div>
    </div>
  );
}
