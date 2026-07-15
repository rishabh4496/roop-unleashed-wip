import React, { useState, useEffect, useRef } from 'react';
import { PERSON_COLORS } from './constants';
import { motion, AnimatePresence, fadeUp, spring } from '../motion';

// Panels rise + de-blur in on mount (cinematic reveal). Hover depth is done in
// CSS (.card-lift) — a shadow/border swell, deliberately no positional "bob",
// so large form panels don't jitter as the cursor crosses them. Pass
// animate={false} to opt a card out of the entrance (e.g. rows inside a list
// that already animates itself).
export const Card = ({ children, className = '', animate = true, hover = true, ...rest }) => (
  <motion.div
    className={`rounded-2xl glass-panel ${hover ? 'card-lift' : ''} ${className}`}
    variants={fadeUp}
    initial={animate ? 'hidden' : false}
    animate={animate ? 'show' : false}
    {...rest}
  >
    {children}
  </motion.div>
);

export const Section = ({ title, action, children, className = '', collapsible = false, defaultOpen = true }) => {
  const [open, setOpen] = useState(defaultOpen);
  const showBody = !collapsible || open;
  const marker = <span className="h-3.5 w-[3px] rounded-full bg-[var(--accent)]/80 shrink-0" />;
  const label = 'text-[11px] font-semibold uppercase tracking-[0.14em] text-white/40';
  return (
    <Card className={`p-5 ${className}`}>
      {title && (
        <div className={`flex items-center justify-between gap-3 ${showBody ? 'mb-4' : 'mb-0'}`}>
          {collapsible ? (
            <button type="button" onClick={() => setOpen((o) => !o)} className="flex items-center gap-2 min-w-0 group/sec">
              {marker}
              <span className={`${label} group-hover/sec:text-white/60 transition-colors`}>{title}</span>
              <svg className={`w-3 h-3 text-white/30 transition-transform duration-200 ${open ? 'rotate-90' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M9 6l6 6-6 6" /></svg>
            </button>
          ) : (
            <h3 className={`${label} flex items-center gap-2`}>{marker}{title}</h3>
          )}
          {action}
        </div>
      )}
      {collapsible ? (
        <AnimatePresence initial={false}>
          {showBody && (
            <motion.div
              key="section-body"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ ...spring.smooth, opacity: { duration: 0.2 } }}
              style={{ overflow: 'hidden' }}
            >
              <div className="space-y-4">{children}</div>
            </motion.div>
          )}
        </AnimatePresence>
      ) : (
        showBody && <div className="space-y-4">{children}</div>
      )}
    </Card>
  );
};

// Shared "?" hover-tooltip badge. shrink-0 so it never wraps to its own line,
// and stays pinned to the top of a multi-line label.
export const InfoBadge = ({ info }) => (
  <div className="relative group inline-flex items-center shrink-0 mt-0.5">
    <span className="text-[10px] text-white/30 hover:text-white/60 cursor-help bg-white/5 rounded-full w-4.5 h-4.5 flex items-center justify-center font-bold apple-transition">?</span>
    <div className="absolute bottom-full right-0 mb-2 hidden group-hover:block tooltip-content z-50 w-max max-w-xs p-3 rounded-xl bg-black/95 backdrop-blur-lg border border-white/10 shadow-2xl text-xs text-white/70 whitespace-normal leading-relaxed pointer-events-none text-left">
      {info}
    </div>
  </div>
);

export const Field = ({ label, info, children }) => (
  <label className="block">
    <div className="flex items-start justify-between gap-2 mb-1.5">
      <span className="text-xs font-medium text-white/70 leading-snug min-w-0">{label}</span>
      {info && <InfoBadge info={info} />}
    </div>
    {children}
  </label>
);

export const Select = ({ label, info, value, onChange, options = [] }) => (
  <Field label={label} info={info}>
    <select
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      className="w-full px-3 py-2.5 rounded-xl glass-input text-white text-[13px] focus:outline-none cursor-pointer"
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
  <label className="flex items-start justify-between gap-3 w-full text-left cursor-pointer group/toggle select-none">
    <span className="flex items-start gap-1.5 min-w-0 flex-1">
      <span className="text-[13px] font-semibold tracking-wide leading-snug text-white/80 group-hover/toggle:text-white transition-colors">{label}</span>
      {info && <InfoBadge info={info} />}
    </span>
    <motion.div
      whileTap={{ scale: 0.9 }}
      transition={spring.snappy}
      className={`relative shrink-0 mt-0.5 w-10 h-[22px] rounded-full transition-colors duration-200 border ${checked ? 'bg-[var(--accent)] border-[var(--accent)]' : 'bg-white/[0.06] border-white/10 group-hover/toggle:border-white/20'}`}
    >
      <motion.span
        className="absolute top-[2px] left-[2px] w-[16px] h-[16px] rounded-full bg-white shadow-[0_1px_3px_rgba(0,0,0,0.5)]"
        animate={{ x: checked ? 18 : 0 }}
        transition={spring.bouncy}
      />
    </motion.div>
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
      className="w-full px-3 py-2.5 rounded-xl glass-input text-white text-[13px] focus:outline-none placeholder:text-white/25"
    />
  </Field>
);

export const Button = ({ children, onClick, variant = 'primary', disabled, className = '', size = 'md' }) => {
  const variants = {
    primary: 'bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white shadow-[0_2px_10px_var(--accent-glow)] hover:shadow-[0_4px_18px_var(--accent-glow)] border border-white/10',
    secondary: 'bg-white/[0.05] hover:bg-white/[0.09] text-white/85 hover:text-white border border-white/10 hover:border-white/20',
    stop: 'bg-red-500/10 hover:bg-red-500/18 text-red-300 border border-red-500/25 hover:border-red-500/40',
    ghost: 'bg-transparent hover:bg-white/[0.06] text-white/55 hover:text-white',
  };
  const sizes = {
    sm: 'px-3 py-1.5 text-[11px] tracking-wide rounded-lg',
    md: 'px-4 py-2.5 text-[13px] tracking-wide rounded-xl',
    lg: 'px-6 py-3.5 text-sm tracking-wide rounded-xl',
  };
  return (
    <motion.button
      type="button"
      onClick={onClick}
      disabled={disabled}
      whileHover={disabled ? undefined : { y: -2, scale: 1.03 }}
      whileTap={disabled ? undefined : { scale: 0.95, y: 0 }}
      transition={spring.snappy}
      className={`font-semibold text-center transition-colors duration-200 disabled:opacity-35 disabled:cursor-not-allowed ${variants[variant]} ${sizes[size]} ${className}`}
    >
      {children}
    </motion.button>
  );
};

// Gallery of face thumbnails with selection + move/remove controls.
export const FaceGallery = ({ title, faces, selected, onSelect, onRemove, empty, groups, vertical = false, info = [], draggable = false, large = false }) => {
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
        <div className={vertical ? "flex flex-col gap-2" : (large ? "grid grid-cols-2 3xl:grid-cols-3 gap-2.5" : "grid grid-cols-4 3xl:grid-cols-5 4xl:grid-cols-6 gap-2")}>
          {faces.map((src, i) => {
            const person = groups && i < groups.length ? groups[i] : null;
            const color = person != null ? PERSON_COLORS[person % PERSON_COLORS.length] : null;
            const itemInfo = info && info[i];
            const hasMultiFaces = itemInfo && itemInfo.count > 1;
            return (
              <motion.div
                key={i}
                layout
                initial={{ opacity: 0, scale: 0.7 }}
                animate={{ opacity: 1, scale: 1, transition: { ...spring.snappy, delay: Math.min(i * 0.025, 0.35) } }}
                whileHover={vertical ? { x: 3 } : { y: -4, scale: 1.04, zIndex: 5 }}
                whileTap={{ scale: 0.95 }}
                transition={spring.snappy}
                draggable={draggable}
                onDragStart={draggable ? (e) => {
                  e.dataTransfer.setData('text/roop-source', String(i));
                  e.dataTransfer.effectAllowed = 'link';
                } : undefined}
                title={draggable ? `Face ${i + 1} — drag onto a person to assign` : undefined}
                className={`group relative ${vertical ? 'flex items-center gap-3 p-2' : 'aspect-square'} rounded-xl overflow-hidden border-2 transition-colors duration-200 cursor-pointer ${draggable ? 'active:cursor-grabbing' : ''} ${selected === i ? (vertical ? 'bg-white/5' : 'ring-2 ring-[var(--accent)]/30') : 'hover:border-white/30'}`}
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
                        className="h-7 w-7 shrink-0 rounded-full bg-black/40 text-white/60 hover:bg-[var(--accent-hover)] hover:text-white hover:scale-110 active:scale-90 flex items-center justify-center"
                      >✕</button>
                    )}
                  </>
                ) : (
                  <>
                    <img src={src} alt={`face ${i}`} className="w-full h-full object-cover" />
                    {hasMultiFaces && (
                      <span className="absolute top-1 left-1 px-1 py-0.5 rounded bg-black/75 backdrop-blur text-[8px] font-bold text-[var(--accent)] border border-[var(--accent)]/30 shadow-md pointer-events-none select-none">
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
                        className="absolute top-1 right-1 h-6 w-6 rounded-full bg-black/70 text-white/80 text-xs leading-none opacity-0 group-hover:opacity-100 hover:bg-[var(--accent-hover)] hover:scale-110 active:scale-90 transition-all duration-300 flex items-center justify-center"
                      >✕</button>
                    )}
                  </>
                )}
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// Smoothly tweens a displayed number toward `value` (eased). Great for %/fps/ETA.
export const AnimatedNumber = ({ value, duration = 500, decimals = 0, suffix = '', className = '' }) => {
  const [disp, setDisp] = useState(value || 0);
  const fromRef = useRef(value || 0);
  const rafRef = useRef();
  useEffect(() => {
    const from = fromRef.current;
    const to = value || 0;
    if (from === to) return;
    const start = performance.now();
    const tick = (t) => {
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisp(from + (to - from) * eased);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
      else fromRef.current = to;
    };
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value, duration]);
  return <span className={className}>{Number(disp).toFixed(decimals)}{suffix}</span>;
};

// Shimmering placeholder block for loading states.
export const Skeleton = ({ className = '' }) => (
  <div className={`rounded-lg bg-white/5 relative overflow-hidden ${className}`}>
    <div className="absolute inset-0 -translate-x-full animate-[skeleton_1.4s_infinite] bg-gradient-to-r from-transparent via-white/10 to-transparent" />
  </div>
);

// One-shot celebratory confetti burst. Renders nothing when inactive.
export const Confetti = ({ active }) => {
  if (!active) return null;
  const colors = ['#E94560', '#3DA5D9', '#52B788', '#E9C46A', '#9B5DE5', '#F4A261', '#00BBF9'];
  const pieces = Array.from({ length: 90 });
  return (
    <div className="fixed inset-0 z-[70] pointer-events-none overflow-hidden">
      {pieces.map((_, i) => {
        const left = Math.random() * 100;
        const delay = Math.random() * 0.5;
        const dur = 1.8 + Math.random() * 1.4;
        const size = 6 + Math.random() * 6;
        const color = colors[i % colors.length];
        const rot = Math.random() * 360;
        return (
          <span
            key={i}
            style={{
              position: 'absolute', top: '-5%', left: `${left}%`,
              width: size, height: size * 0.5, background: color,
              transform: `rotate(${rot}deg)`,
              animation: `confetti-fall ${dur}s cubic-bezier(0.4,0,0.6,1) ${delay}s forwards`,
              borderRadius: 1,
            }}
          />
        );
      })}
    </div>
  );
};

export const Toast = ({ toast }) => (
  <AnimatePresence>
    {toast && (
      <motion.div
        initial={{ opacity: 0, y: 24, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 12, scale: 0.95 }}
        transition={spring.bouncy}
        className="fixed bottom-6 right-6 z-50 px-4 py-3 rounded-xl shadow-2xl bg-[#0E0F15]/95 backdrop-blur-xl border border-white/10 flex items-center gap-3 min-w-[250px]"
      >
        {toast.type === 'error' && <span className="text-lg">❌</span>}
        {toast.type === 'info' && <span className="text-lg">ℹ️</span>}
        {(!toast.type || toast.type === 'success') && <span className="text-lg">✅</span>}
        <span className={`text-sm font-medium ${toast.type === 'error' ? 'text-red-400' : toast.type === 'info' ? 'text-blue-300' : 'text-green-400'}`}>
          {toast.message}
        </span>
      </motion.div>
    )}
  </AnimatePresence>
);
