import React from 'react';

export const Card = ({ children, className = '' }) => (
  <div className={`rounded-xl bg-white/[0.04] border border-white/10 backdrop-blur-md ${className}`}>
    {children}
  </div>
);

export const Section = ({ title, children, className = '' }) => (
  <Card className={`p-5 ${className}`}>
    {title && <h3 className="text-sm font-semibold uppercase tracking-wider text-white/50 mb-4">{title}</h3>}
    <div className="space-y-4">{children}</div>
  </Card>
);

export const Field = ({ label, info, children }) => (
  <label className="block">
    <div className="flex items-baseline justify-between mb-1.5">
      <span className="text-sm text-white/80">{label}</span>
      {info && <span className="text-xs text-white/35">{info}</span>}
    </div>
    {children}
  </label>
);

export const Select = ({ label, info, value, onChange, options = [] }) => (
  <Field label={label} info={info}>
    <select
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm focus:outline-none focus:border-[#E94560] transition-colors"
    >
      {options.map((o) => (
        <option key={o} value={o} className="bg-[#16213E]">{o}</option>
      ))}
    </select>
  </Field>
);

export const Slider = ({ label, info, value, onChange, min = 0, max = 1, step = 0.01 }) => (
  <Field label={label} info={info}>
    <div className="flex items-center gap-3">
      <input
        type="range"
        min={min} max={max} step={step}
        value={value ?? 0}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 accent-[#E94560] h-1.5"
      />
      <span className="w-14 text-right text-sm tabular-nums text-white/70">{Number(value ?? 0).toFixed(step < 1 ? 2 : 0)}</span>
    </div>
  </Field>
);

export const Toggle = ({ label, info, checked, onChange }) => (
  <button
    type="button"
    onClick={() => onChange(!checked)}
    className="flex items-center justify-between w-full text-left group"
  >
    <span>
      <span className="text-sm text-white/80 block">{label}</span>
      {info && <span className="text-xs text-white/35">{info}</span>}
    </span>
    <span className={`relative shrink-0 w-10 h-6 rounded-full transition-colors ml-3 ${checked ? 'bg-[#E94560]' : 'bg-white/15'}`}>
      <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${checked ? 'translate-x-4' : ''}`} />
    </span>
  </button>
);

export const TextInput = ({ label, info, value, onChange, placeholder, type = 'text' }) => (
  <Field label={label} info={info}>
    <input
      type={type}
      value={value ?? ''}
      placeholder={placeholder}
      onChange={(e) => onChange(type === 'number' ? Number(e.target.value) : e.target.value)}
      className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm focus:outline-none focus:border-[#E94560] transition-colors"
    />
  </Field>
);

export const Button = ({ children, onClick, variant = 'primary', disabled, className = '', size = 'md' }) => {
  const variants = {
    primary: 'bg-[#E94560] hover:bg-[#d63450] text-white shadow-[0_4px_15px_rgba(233,69,96,0.35)]',
    secondary: 'bg-white/10 hover:bg-white/15 text-white border border-white/10',
    stop: 'bg-transparent hover:bg-red-500/15 text-red-400 border border-red-500/40',
    ghost: 'bg-transparent hover:bg-white/10 text-white/70',
  };
  const sizes = { sm: 'px-3 py-1.5 text-xs', md: 'px-4 py-2.5 text-sm', lg: 'px-6 py-3.5 text-base' };
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed ${variants[variant]} ${sizes[size]} ${className}`}
    >
      {children}
    </button>
  );
};

// Gallery of face thumbnails with selection + move/remove controls.
const PERSON_COLORS = ['#E94560', '#3DA5D9', '#52B788', '#E9C46A', '#9B5DE5', '#F4A261', '#00BBF9', '#F15BB5'];

export const FaceGallery = ({ title, faces, selected, onSelect, onRemove, empty, groups }) => {
  const personCount = groups && groups.length ? new Set(groups).size : 0;
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-white/40">{title}</span>
        {faces.length > 0 && (
          <span className="px-1.5 py-0.5 rounded-full bg-white/10 text-[10px] text-white/60 tabular-nums">
            {faces.length}{personCount > 1 ? ` · ${personCount} people` : ''}
          </span>
        )}
      </div>
      {faces.length === 0 ? (
        <div className="h-24 flex items-center justify-center rounded-lg border border-dashed border-white/10 text-xs text-white/30">
          {empty || 'None yet'}
        </div>
      ) : (
        <div className="grid grid-cols-4 gap-2">
          {faces.map((src, i) => {
            const person = groups && i < groups.length ? groups[i] : null;
            const color = person != null ? PERSON_COLORS[person % PERSON_COLORS.length] : null;
            return (
              <div
                key={i}
                className={`group relative aspect-square rounded-lg overflow-hidden border-2 transition-all cursor-pointer ${selected === i ? 'scale-105' : 'hover:border-white/30'}`}
                style={{ borderColor: selected === i ? (color || '#E94560') : (color ? `${color}66` : 'transparent') }}
                onClick={() => onSelect(i)}
              >
                <img src={src} alt={`face ${i}`} className="w-full h-full object-cover" />
                {person != null && (
                  <span className="absolute bottom-0.5 left-0.5 px-1 rounded text-[9px] font-semibold leading-tight text-white"
                    style={{ backgroundColor: color }}>P{person + 1}</span>
                )}
                {onRemove && (
                  <button
                    type="button"
                    title="Remove this face"
                    onClick={(e) => { e.stopPropagation(); onRemove(i); }}
                    className="absolute top-0.5 right-0.5 h-5 w-5 rounded-full bg-black/70 text-white/80 text-xs leading-none opacity-0 group-hover:opacity-100 hover:bg-[#E94560] transition-opacity flex items-center justify-center"
                  >✕</button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export const Toast = ({ toast }) =>
  !toast ? null : (
    <div className="fixed bottom-6 right-6 z-50 px-4 py-3 rounded-lg shadow-xl text-sm font-medium animate-[fadein_.2s] bg-[#16213E] border border-white/10">
      <span className={toast.type === 'error' ? 'text-red-400' : toast.type === 'info' ? 'text-blue-300' : 'text-green-400'}>
        {toast.message}
      </span>
    </div>
  );
