import React, { useEffect, useRef, useState } from 'react';

/**
 * Terminal-style live feed shown inside the Preview box while a job runs.
 * Features real-time log filtering and 1-click copy logs to clipboard.
 */
export default function ProcessingTerminal({ log = [], paused, className = '', bodyClass = 'h-40' }) {
  const scrollRef = useRef(null);
  const bottomRef = useRef(null);
  const [filter, setFilter] = useState('all'); // 'all' | 'errors' | 'stages' | 'frames'
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    if (nearBottom) bottomRef.current?.scrollIntoView({ block: 'end' });
  }, [log.length, log[log.length - 1]?.seq, filter]);

  const lineTone = (msg) => {
    const m = (msg || '').toLowerCase();
    if (m.startsWith('⚠') || /error|fail|abort/.test(m)) return 'text-red-400 font-semibold';
    if (m.startsWith('✓') || /\bdone\b/.test(m)) return 'text-emerald-400 font-semibold';
    if (m.startsWith('▶') || /start/.test(m)) return 'text-[var(--accent)] font-semibold';
    if (/combin|encod|audio|mux|finaliz/.test(m)) return 'text-sky-300/90';
    if (/upscal|interpolat/.test(m)) return 'text-fuchsia-300/85';
    return 'text-white/70';
  };

  const filteredLogs = log.filter((l) => {
    const m = (l.msg || '').toLowerCase();
    if (filter === 'errors') return m.includes('error') || m.includes('fail') || m.includes('⚠');
    if (filter === 'stages') return m.includes('stage') || m.includes('start') || m.includes('done') || m.includes('✓') || m.includes('▶');
    if (filter === 'frames') return m.includes('frame') || m.includes('fps');
    return true;
  });

  const handleCopy = () => {
    const text = log.map((l) => `[${l.t}] ${l.msg}`).join('\n');
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className={`w-full rounded-xl border border-white/10 bg-black/60 overflow-hidden font-mono flex flex-col ${className}`}>
      {/* Title bar — traffic lights + filter chips + copy button */}
      <div className="shrink-0 flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/[0.03]">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/70" />
          </span>
          <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-white/40">Console Logs</span>

          {/* Filter Chips */}
          <div className="hidden sm:flex items-center gap-1 ml-2">
            {[
              { id: 'all', label: 'All' },
              { id: 'errors', label: 'Errors' },
              { id: 'stages', label: 'Stages' },
              { id: 'frames', label: 'Frames' },
            ].map((f) => (
              <button
                key={f.id}
                onClick={() => setFilter(f.id)}
                className={`px-2 py-0.5 rounded text-[9px] font-semibold transition-all ${
                  filter === f.id
                    ? 'bg-indigo-600/40 text-indigo-200 border border-indigo-500/50'
                    : 'text-neutral-400 hover:text-white hover:bg-white/5'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-white/5 border border-white/10 text-neutral-300 hover:bg-white/10 hover:text-white transition-all active:scale-95"
            title="Copy all logs to clipboard"
          >
            <span>{copied ? '✓ Copied!' : '📋 Copy Logs'}</span>
          </button>

          <span className="flex items-center gap-1.5 text-[10px] text-white/40">
            <span className={`h-1.5 w-1.5 rounded-full ${paused ? 'bg-amber-400' : 'bg-emerald-400 animate-pulse'}`} />
            {paused ? 'paused' : 'live'}
          </span>
        </div>
      </div>

      {/* Scrolling log body */}
      <div ref={scrollRef} className={`selectable ${bodyClass} min-h-0 overflow-y-auto px-3 py-2 text-[11px] leading-relaxed scroll-smooth`}>
        {filteredLogs.length === 0 ? (
          <div className="text-white/30 italic">No matching log entries…</div>
        ) : (
          filteredLogs.map((l) => (
            <div key={l.seq} className="flex gap-2 whitespace-pre-wrap break-words">
              <span className="shrink-0 text-white/25 tabular-nums">{l.t}</span>
              <span className="shrink-0 text-white/25">›</span>
              <span className={lineTone(l.msg)}>{l.msg}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} className="flex gap-2">
          <span className="text-white/25">›</span>
          <span className="inline-block h-3.5 w-2 bg-[var(--accent)]/80 animate-pulse" />
        </div>
      </div>
    </div>
  );
}
