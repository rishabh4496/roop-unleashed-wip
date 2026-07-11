import React, { useState } from 'react';

export default function FileDrop({ label, accept, multiple, onFiles, busy, hint }) {
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
  return (
    <label
      onDragOver={(e) => { e.preventDefault(); if (!busy) setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      className={`block ${busy ? 'cursor-wait pointer-events-none' : 'cursor-pointer group'}`}
    >
      <div className={`px-4 py-3.5 rounded-xl border-2 border-dashed text-center transition-all duration-200 ${busy ? 'border-[var(--accent)]/60 bg-[var(--accent)]/[0.06]' : drag ? 'border-[var(--accent)]/60 bg-[var(--accent)]/[0.08]' : 'border-white/15 hover:border-[var(--accent)]/50 hover:bg-white/[0.02]'}`}>
        {busy ? (
          <span className="inline-flex items-center gap-2 text-xs font-semibold text-white/80">
            <span className="h-3.5 w-3.5 rounded-full border-2 border-white/30 border-t-[var(--accent)] animate-spin" />
            Uploading & analysing…
          </span>
        ) : (
          <div className="flex items-center justify-center gap-3 pointer-events-none">
            <div className={`p-2 rounded-xl bg-black/20 ${drag ? 'text-[var(--accent)] bg-[var(--accent)]/10' : 'text-white/40 group-hover:text-white/70 group-hover:bg-white/5'} transition-all duration-200`}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <div className="text-left">
              <span className={`text-xs font-bold tracking-wide block ${drag ? 'text-[var(--accent)]' : 'text-white/80'}`}>{drag ? 'Drop files now' : label}</span>
              {!drag && hint && <span className="block text-[10px] text-white/40 mt-0.5">{hint}</span>}
            </div>
          </div>
        )}
      </div>
      <input type="file" accept={accept} multiple={multiple}
        onChange={(e) => { if (e.target.files.length) onFiles(e.target.files); e.target.value = ''; }}
        disabled={busy} className="hidden" />
    </label>
  );
}
