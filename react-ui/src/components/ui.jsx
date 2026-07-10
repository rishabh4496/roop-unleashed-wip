import React from 'react';

export const Card = ({ children, className = '', ...rest }) => (
  <div className={`rounded-2xl glass-panel apple-transition ${className}`} {...rest}>
    {children}
  </div>
);

export const Section = ({ title, children, className = '' }) => (
  <Card className={`p-6 shadow-xl hover:shadow-2xl border-white/5 border hover:border-white/10 ${className}`}>
    {title && (
      <div className="flex items-center justify-between mb-5 border-b border-white/5 pb-3">
        <h3 className="text-[10px] font-black uppercase tracking-widest text-white/45 flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
          {title}
        </h3>
      </div>
    )}
    <div className="space-y-5">{children}</div>
  </Card>
);

export const Field = ({ label, info, children }) => (
  <label className="block">
    <div className="flex items-baseline justify-between mb-1.5">
      <span className="text-xs font-medium text-white/70">{label}</span>
      {info && (
        <div className="relative group inline-flex items-center">
          <span className="text-[10px] text-white/30 hover:text-white/60 cursor-help bg-white/5 rounded-full w-4.5 h-4.5 flex items-center justify-center font-bold apple-transition">?</span>
          <div className="absolute bottom-full right-0 mb-2 hidden group-hover:block tooltip-content z-50 w-max max-w-xs p-3 rounded-xl bg-black/95 backdrop-blur-lg border border-white/10 shadow-2xl text-xs text-white/70 whitespace-normal leading-relaxed pointer-events-none text-left">
            {info}
          </div>
        </div>
      )}
    </div>
    {children}
  </label>
);

export const Select = ({ label, info, value, onChange, options = [] }) => (
  <Field label={label} info={info}>
    <select
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      className="w-full px-3 py-2 rounded-xl glass-input text-white text-sm focus:outline-none cursor-pointer"
    >
      {options.map((o) => (
        <option key={o} value={o} className="bg-[#121420]">{o}</option>
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
        className="flex-1 accent-[#E94560] h-1.5 apple-transition"
      />
      <span className="w-12 text-right text-xs font-semibold tabular-nums text-white/60">{Number(value ?? 0).toFixed(step < 1 ? 2 : 0)}</span>
    </div>
  </Field>
);

export const Toggle = ({ label, info, checked, onChange }) => (
  <label className="flex items-center justify-between w-full text-left cursor-pointer group/toggle select-none">
    <div className="flex items-center gap-2">
      <span className="text-sm font-semibold tracking-wide text-white/80 group-hover/toggle:text-white transition-colors">{label}</span>
      {info && (
        <div className="relative group inline-flex items-center">
          <span className="text-[10px] text-white/30 hover:text-white/60 cursor-help bg-white/5 rounded-full w-4.5 h-4.5 flex items-center justify-center font-bold apple-transition">?</span>
          <div className="absolute bottom-full right-0 mb-2 hidden group-hover:block tooltip-content z-50 w-max max-w-xs p-3 rounded-xl bg-black/95 backdrop-blur-lg border border-white/10 shadow-2xl text-xs text-white/70 whitespace-normal leading-relaxed pointer-events-none text-left">
            {info}
          </div>
        </div>
      )}
    </div>
    <div className={`relative shrink-0 w-10.5 h-6 rounded-full transition-all duration-300 ml-3 border ${checked ? 'bg-[var(--accent)] border-[var(--accent)] shadow-[0_0_12px_var(--accent-glow)]' : 'bg-black/30 border-white/5'}`}>
      <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-[0_2px_4px_rgba(0,0,0,0.4)] transition-all duration-300 cubic-bezier(0.16, 1, 0.3, 1) ${checked ? 'translate-x-4.5' : ''}`} />
    </div>
    <input type="checkbox" className="hidden" checked={checked} onChange={(e) => onChange(e.target.checked)} />
  </label>
);

export const TextInput = ({ label, info, value, onChange, placeholder, type = 'text' }) => (
  <Field label={label} info={info}>
    <input
      type={type}
      value={value ?? ''}
      placeholder={placeholder}
      onChange={(e) => onChange(type === 'number' ? Number(e.target.value) : e.target.value)}
      className="w-full px-3 py-2 rounded-xl glass-input text-white text-sm focus:outline-none"
    />
  </Field>
);

export const Button = ({ children, onClick, variant = 'primary', disabled, className = '', size = 'md' }) => {
  const variants = {
    primary: 'bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] text-white shadow-[0_4px_20px_rgba(233,69,96,0.35)] hover:shadow-[0_8px_30px_rgba(233,69,96,0.5)] border border-white/10 shimmer-sweep',
    secondary: 'bg-white/5 hover:bg-white/10 text-white/90 border border-white/5 backdrop-blur-md hover:border-white/10 shimmer-sweep',
    stop: 'bg-red-500/5 hover:bg-red-500/15 text-red-400 border border-red-500/20 hover:border-red-500/40 shimmer-sweep',
    ghost: 'bg-transparent hover:bg-white/5 text-white/60 hover:text-white',
  };
  const sizes = { 
    sm: 'px-3 py-1.5 text-[10px] tracking-wider uppercase', 
    md: 'px-5 py-3 text-xs tracking-wider uppercase', 
    lg: 'px-7 py-4 text-sm tracking-widest uppercase' 
  };
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-xl font-extrabold text-center apple-transition apple-spring-active disabled:opacity-30 disabled:active:scale-100 disabled:cursor-not-allowed ${variants[variant]} ${sizes[size]} ${className}`}
    >
      {children}
    </button>
  );
};

// Gallery of face thumbnails with selection + move/remove controls.
export const PERSON_COLORS = ['#E94560', '#3DA5D9', '#52B788', '#E9C46A', '#9B5DE5', '#F4A261', '#00BBF9', '#F15BB5'];

export const FaceGallery = ({ title, faces, selected, onSelect, onRemove, empty, groups, vertical = false, info = [], draggable = false }) => {
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
        <div className={vertical ? "flex flex-col gap-2" : "grid grid-cols-4 3xl:grid-cols-5 4xl:grid-cols-6 gap-2"}>
          {faces.map((src, i) => {
            const person = groups && i < groups.length ? groups[i] : null;
            const color = person != null ? PERSON_COLORS[person % PERSON_COLORS.length] : null;
            const itemInfo = info && info[i];
            const hasMultiFaces = itemInfo && itemInfo.count > 1;
            return (
              <div
                key={i}
                draggable={draggable}
                onDragStart={draggable ? (e) => {
                  e.dataTransfer.setData('text/roop-source', String(i));
                  e.dataTransfer.effectAllowed = 'link';
                } : undefined}
                title={draggable ? `Face ${i + 1} — drag onto a person to assign` : undefined}
                className={`group relative ${vertical ? 'flex items-center gap-3 p-2' : 'aspect-square'} rounded-xl overflow-hidden border-2 apple-transition apple-spring-active cursor-pointer ${draggable ? 'active:cursor-grabbing' : ''} ${selected === i ? (vertical ? 'bg-white/5' : 'scale-105') : 'hover:border-white/30'}`}
                style={{ borderColor: selected === i ? (color || 'var(--accent)') : (color ? `${color}66` : 'transparent') }}
                onClick={() => onSelect(i)}
              >
                {vertical ? (
                  <>
                    <img src={src} alt={`face ${i}`} className="w-10 h-10 rounded-lg object-cover shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-white/90">
                        {person != null ? `Person ${person + 1}` : `Face ${i + 1}`}
                      </div>
                    </div>
                    {onRemove && (
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); onRemove(i); }}
                        className="h-7 w-7 shrink-0 rounded-full bg-black/40 text-white/60 hover:bg-[var(--accent-hover)] hover:text-white transition-colors flex items-center justify-center"
                      >✕</button>
                    )}
                  </>
                ) : (
                  <>
                    <img src={src} alt={`face ${i}`} className="w-full h-full object-cover" />
                    {hasMultiFaces && (
                      <span className="absolute top-1 left-1 px-1 py-0.5 rounded bg-black/75 backdrop-blur text-[8px] font-black text-[var(--accent)] border border-[var(--accent)]/30 shadow-md pointer-events-none select-none">
                        {itemInfo.count}F
                      </span>
                    )}
                    {person != null && (
                      <span className="absolute bottom-1 left-1 px-1.5 rounded-md text-[9px] font-bold leading-tight text-white shadow-sm"
                        style={{ backgroundColor: color }}>P{person + 1}</span>
                    )}
                    {onRemove && (
                      <button
                        type="button"
                        title="Remove this face"
                        onClick={(e) => { e.stopPropagation(); onRemove(i); }}
                        className="absolute top-1 right-1 h-6 w-6 rounded-full bg-black/70 text-white/80 text-xs leading-none opacity-0 group-hover:opacity-100 hover:bg-[var(--accent-hover)] transition-opacity flex items-center justify-center"
                      >✕</button>
                    )}
                  </>
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
    <div className="fixed bottom-6 right-6 z-50 px-4 py-3 rounded-xl shadow-2xl animate-slide-up bg-[#16213E]/95 backdrop-blur-md border border-white/10 flex items-center gap-3 min-w-[250px]">
      {toast.type === 'error' && <span className="text-lg">❌</span>}
      {toast.type === 'info' && <span className="text-lg">ℹ️</span>}
      {(!toast.type || toast.type === 'success') && <span className="text-lg">✅</span>}
      <span className={`text-sm font-medium ${toast.type === 'error' ? 'text-red-400' : toast.type === 'info' ? 'text-blue-300' : 'text-green-400'}`}>
        {toast.message}
      </span>
    </div>
  );
