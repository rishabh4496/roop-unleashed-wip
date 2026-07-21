import React, { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence, spring } from '../motion';

// Keyboard-shortcut cheat-sheet. Opened with `?` (and closable with Esc). The
// shortcuts themselves live in App.jsx / InteractivePreview; this is the single
// discoverable place that lists them.
const GROUPS = [
  {
    title: 'Global',
    items: [
      [['Ctrl', 'K'], 'Command palette / search'],
      [['?'], 'This shortcuts sheet'],
      [['Ctrl', '+'], 'Zoom UI in'],
      [['Ctrl', '−'], 'Zoom UI out'],
      [['Ctrl', '0'], 'Reset UI zoom'],
    ],
  },
  {
    title: 'Preview & timeline',
    items: [
      [['←', '→'], 'Previous / next frame'],
      [['Shift', '←/→'], 'Jump ±10 frames'],
      [['Home', 'End'], 'First / last frame'],
      [['[', ']'], 'Set In / Out point'],
      [['Space'], 'Play / pause'],
      [['C'], 'Toggle before/after compare'],
      [['+', '−'], 'Zoom preview in / out'],
    ],
  },
];

function Keys({ keys }) {
  return (
    <span className="flex items-center gap-1">
      {keys.map((k, i) => (
        <kbd key={i} className="min-w-[22px] text-center px-1.5 py-0.5 rounded-md bg-white/[0.06] border border-white/12 text-[11px] font-mono font-semibold text-white/80">
          {k}
        </kbd>
      ))}
    </span>
  );
}

export default function ShortcutsModal({ open, onClose }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[85] flex items-center justify-center p-4"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
          role="dialog" aria-modal="true" aria-label="Keyboard shortcuts"
        >
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
          <motion.div
            initial={{ scale: 0.94, y: 16, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.96, y: 8, opacity: 0 }}
            transition={spring.snappy}
            className="relative w-full max-w-lg rounded-2xl glass-panel border border-white/10 p-6 shadow-2xl"
          >
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-[15px] font-bold text-white/95 flex items-center gap-2">⌨️ Keyboard shortcuts</h2>
              <button type="button" onClick={onClose} aria-label="Close shortcuts"
                className="h-7 w-7 grid place-items-center rounded-lg text-white/50 hover:text-white hover:bg-white/10 transition-colors">✕</button>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-5">
              {GROUPS.map((g) => (
                <div key={g.title}>
                  <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-white/40 mb-2.5">{g.title}</div>
                  <div className="space-y-2">
                    {g.items.map(([keys, label], i) => (
                      <div key={i} className="flex items-center justify-between gap-3">
                        <span className="text-[12px] text-white/65">{label}</span>
                        <Keys keys={keys} />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
}
