import React from 'react';
import { THEMES } from '../themes';

// Visual theme picker: a grid of live swatch cards instead of a plain dropdown.
export default function ThemeGallery({ value, onChange }) {
  const current = value || 'Default';
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
      {THEMES.map((t) => {
        const active = t.name === current;
        return (
          <button
            key={t.name}
            type="button"
            onClick={() => onChange(t.name)}
            title={t.label}
            className={`tap group relative text-left rounded-xl overflow-hidden border-2 focus:outline-none ${active ? 'shadow-lg' : ''}`}
            style={{ borderColor: active ? t.accent : 'rgba(255,255,255,0.08)', boxShadow: active ? `0 0 0 1px ${t.accent}, 0 8px 24px ${t.accent}33` : undefined }}
          >
            {/* Swatch preview */}
            <div className="h-12 w-full relative" style={{ background: t.bg }}>
              <div className="absolute inset-0 opacity-90" style={{ background: `radial-gradient(circle at 25% 20%, ${t.accent}44, transparent 60%)` }} />
              <div className="absolute bottom-1.5 left-2 flex items-center gap-1">
                <span className="h-4 w-4 rounded-full shadow" style={{ background: t.accent }} />
                <span className="h-2 w-8 rounded-full" style={{ background: `${t.accent}66` }} />
                <span className="h-2 w-4 rounded-full bg-white/20" />
              </div>
              {active && (
                <span className="absolute top-1.5 right-1.5 h-5 w-5 rounded-full flex items-center justify-center text-[10px] font-black text-white shadow" style={{ background: t.accent }}>✓</span>
              )}
            </div>
            {/* Label */}
            <div className="px-2.5 py-2 bg-black/40 backdrop-blur-sm">
              <div className="flex items-center justify-between gap-1">
                <span className="text-[11px] font-bold text-white/90 truncate">{t.name}</span>
                <span className={`text-[8px] font-black uppercase tracking-wider px-1 py-0.5 rounded shrink-0 ${t.mode === 'light' ? 'bg-amber-400/15 text-amber-300' : 'bg-white/10 text-white/40'}`}>{t.mode}</span>
              </div>
              <div className="text-[9px] text-white/35 truncate">{t.label}</div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
