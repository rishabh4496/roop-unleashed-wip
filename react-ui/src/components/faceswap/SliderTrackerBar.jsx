import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TRACKER_SLIDERS, TRACKER_DEFAULT_VALUES } from './trackerConfig';

const STORAGE_KEY = 'roop_user_slider_presets';

const num = (v, fallback) => (typeof v === 'number' && !isNaN(v) ? v : fallback);

// Slider steps are 0.01/0.05, so values that came back through parseFloat or
// JSON can be a float ULP off what the preset table holds. Compare with a
// tolerance rather than ===, or a preset would never register as "active".
const valuesMatch = (params, values) =>
  TRACKER_SLIDERS.every((s) =>
    Math.abs(num(params[s.key], s.defaultVal) - num(values[s.key], s.defaultVal)) < 1e-6);

// Custom presets come from localStorage, which is user-writable and survives
// across versions — anything malformed in there would otherwise crash the whole
// Face Swap tab on the spread below.
const sanitizeCustomPresets = (raw) => {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((p) => p && typeof p.name === 'string' && p.values && typeof p.values === 'object')
    .map((p, i) => ({
      name: p.name,
      values: Object.fromEntries(
        TRACKER_SLIDERS.map((s) => [s.key, num(p.values[s.key], s.defaultVal)])),
      isCustom: true,
      id: String(p.id ?? `legacy-${i}`),
    }));
};

const BUILTIN_PRESETS = [
  { name: 'Default', values: TRACKER_DEFAULT_VALUES },
  {
    name: '✨ Ultra Realism',
    values: {
      blend_ratio: 0.85,
      detail_transfer_strength: 0.4,
      expression_restore_strength: 0.8,
      face_mask_blend: 25,
      max_face_distance: 0.8,
      num_swap_steps: 2,
      jaw_reshape_strength: 0.4,
      stabilize_enhancer_strength: 0.6,
    },
  },
  {
    name: '🎬 Cinematic',
    values: {
      blend_ratio: 0.9,
      detail_transfer_strength: 0.25,
      expression_restore_strength: 1.2,
      face_mask_blend: 30,
      max_face_distance: 0.85,
      num_swap_steps: 2,
      jaw_reshape_strength: 0.5,
      stabilize_enhancer_strength: 0.5,
    },
  },
  {
    name: '🕊️ Subtle Touchup',
    values: {
      blend_ratio: 0.6,
      detail_transfer_strength: 0.15,
      expression_restore_strength: 0.4,
      face_mask_blend: 15,
      max_face_distance: 0.9,
      num_swap_steps: 1,
      jaw_reshape_strength: 0.2,
      stabilize_enhancer_strength: 0.3,
    },
  },
  {
    name: '⚡ Strong Likeness',
    values: {
      blend_ratio: 1.0,
      detail_transfer_strength: 0.5,
      expression_restore_strength: 0.9,
      face_mask_blend: 20,
      max_face_distance: 0.7,
      num_swap_steps: 3,
      jaw_reshape_strength: 0.75,
      stabilize_enhancer_strength: 0.7,
    },
  },
  {
    name: '🌿 Natural Soft',
    values: {
      blend_ratio: 0.75,
      detail_transfer_strength: 0.2,
      expression_restore_strength: 0.6,
      face_mask_blend: 40,
      max_face_distance: 0.85,
      num_swap_steps: 1,
      jaw_reshape_strength: 0.3,
      stabilize_enhancer_strength: 0.5,
    },
  },
];

export default function SliderTrackerBar({
  params = {},
  onSetParam,
  sliderEffectEnabled = true,
  onToggleSliderEffect,
  onResetSliders,
  onRefreshPreview,
}) {
  const [expanded, setExpanded] = useState(true);
  const [customPresets, setCustomPresets] = useState([]);
  const [isSaving, setIsSaving] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');

  // Load custom presets from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) setCustomPresets(sanitizeCustomPresets(JSON.parse(saved)));
    } catch {
      /* ignore */
    }
  }, []);

  // Save custom presets to localStorage
  const saveCustomPresetsToStorage = (updated) => {
    setCustomPresets(updated);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch {
      /* ignore */
    }
  };

  const allPresets = useMemo(() => [...BUILTIN_PRESETS, ...customPresets], [customPresets]);

  // DERIVED, never stored. As state it went stale in four ways: it started at
  // 'Default' regardless of the loaded config, it did not follow edits made in
  // the main settings panel, clicking every "def:" pip left it reading 'Custom',
  // and "↺ Reset" and preset-delete had to remember to correct it by hand.
  // Reading it off `params` means the highlight is always the truth.
  const activePreset = useMemo(() => {
    const hit = allPresets.find((pr) => valuesMatch(params, pr.values));
    return hit ? hit.name : 'Custom';
  }, [params, allPresets]);

  const applyPreset = (pObj) => {
    Object.entries(pObj.values).forEach(([k, v]) => {
      if (onSetParam) onSetParam(k, v);
    });
    if (onRefreshPreview) onRefreshPreview();
  };

  const handleSaveCurrentPreset = () => {
    // Strip any ★ the user typed (or pasted from an existing pill's label) so
    // re-saving "★ mine" cannot produce "★ ★ mine".
    const name = newPresetName.replace(/^[★\s]+/, '').trim();
    if (!name) return;

    const values = Object.fromEntries(
      TRACKER_SLIDERS.map((s) => [s.key, num(params[s.key], s.defaultVal)]));

    const newPreset = {
      name: `★ ${name}`,
      values,
      isCustom: true,
      // Date.now() alone collides when two presets are saved inside the same
      // millisecond, which would make one delete remove both.
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    };

    saveCustomPresetsToStorage(
      [...customPresets.filter((p) => p.name !== newPreset.name), newPreset]);
    setNewPresetName('');
    setIsSaving(false);
  };

  const handleDeleteCustomPreset = (presetId, e) => {
    e.stopPropagation();
    saveCustomPresetsToStorage(customPresets.filter((p) => p.id !== presetId));
  };

  return (
    <div className="w-full rounded-2xl glass-panel p-4 mb-4 border border-white/10 shadow-2xl transition-all duration-300 relative overflow-hidden group">
      {/* Subtle top glow line */}
      <div
        className={`absolute top-0 inset-x-0 h-0.5 transition-colors duration-500 ${
          sliderEffectEnabled
            ? 'bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent opacity-80'
            : 'bg-white/10 opacity-30'
        }`}
      />

      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 select-none">
        {/* Left title & active status badge */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-base">🎚️</span>
            <span className="text-xs font-extrabold uppercase tracking-[0.16em] text-white/90">
              Slider Tracker
            </span>
          </div>

          {/* Toggle status indicator badge */}
          <button
            type="button"
            onClick={onToggleSliderEffect}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold tracking-wider uppercase border transition-all duration-200 cursor-pointer ${
              sliderEffectEnabled
                ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.25)]'
                : 'bg-amber-500/15 border-amber-500/40 text-amber-300'
            }`}
            title="Click to toggle slider effects ON or OFF"
          >
            <span
              className={`h-2 w-2 rounded-full ${
                sliderEffectEnabled
                  ? 'bg-emerald-400 animate-pulse shadow-[0_0_6px_#10b981]'
                  : 'bg-amber-400'
              }`}
            />
            {sliderEffectEnabled ? 'Slider Effect: ON' : 'Slider Effect: BYPASSED'}
          </button>
        </div>

        {/* Right action controls */}
        <div className="flex items-center gap-2">
          {/* Master Slider Effect Switch */}
          <button
            type="button"
            onClick={onToggleSliderEffect}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all duration-200 cursor-pointer ${
              sliderEffectEnabled
                ? 'bg-[var(--accent)]/20 border-[var(--accent)]/50 text-white shadow-[0_0_14px_var(--accent-glow)]'
                : 'bg-white/[0.04] border-white/10 text-white/50 hover:border-white/20 hover:text-white/80'
            }`}
          >
            <div
              className={`w-7 h-4 rounded-full p-0.5 transition-colors duration-200 ${
                sliderEffectEnabled ? 'bg-[var(--accent)]' : 'bg-white/20'
              }`}
            >
              <div
                className={`w-3 h-3 rounded-full bg-white transition-transform duration-200 ${
                  sliderEffectEnabled ? 'translate-x-3' : 'translate-x-0'
                }`}
              />
            </div>
            <span>Effect {sliderEffectEnabled ? 'ON' : 'OFF'}</span>
          </button>

          {/* Reset sliders button */}
          {onResetSliders && (
            <button
              type="button"
              onClick={onResetSliders}
              className="px-2.5 py-1.5 rounded-xl text-[11px] font-semibold bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-white/60 hover:text-white transition-colors"
              title="Reset all trackers to default values"
            >
              ↺ Reset
            </button>
          )}

          {/* Refresh preview button */}
          {onRefreshPreview && (
            <button
              type="button"
              onClick={onRefreshPreview}
              className="px-2.5 py-1.5 rounded-xl text-[11px] font-semibold bg-[var(--accent)]/10 hover:bg-[var(--accent)]/20 border border-[var(--accent)]/30 text-[var(--accent)] transition-colors"
              title="Force re-render preview with current slider tracker settings"
            >
              🔄 Refresh Preview
            </button>
          )}

          {/* Collapse/Expand button */}
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="px-2.5 py-1.5 rounded-xl text-[11px] font-semibold bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-white/70 transition-colors flex items-center gap-1"
          >
            <span>{expanded ? 'Collapse' : 'Expand'}</span>
            <svg
              className={`w-3 h-3 transition-transform duration-200 ${
                expanded ? 'rotate-180' : 'rotate-0'
              }`}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
        </div>
      </div>

      {/* Preset Profiles Selector & Save Strip */}
      {expanded && (
        <div className="flex flex-wrap items-center justify-between gap-2 mt-3 pt-3 border-t border-white/5 select-none">
          {/* Left preset pills list */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px] font-bold uppercase tracking-wider text-white/40 mr-1">
              Presets:
            </span>
            {activePreset === 'Custom' && (
              <span
                className="px-2 py-1 rounded-lg text-[10px] font-bold border border-dashed border-white/25 text-white/60 bg-white/[0.03]"
                title="Current slider values do not match any saved preset"
              >
                Custom
              </span>
            )}
            {allPresets.map((p) => {
              const isSel = activePreset === p.name;
              return (
                // The delete control is a SIBLING of the preset button, not a
                // child of it: a disabled <button> swallows clicks on its
                // descendants, so nesting it made "×" dead whenever the effect
                // was bypassed — and a button inside a button is invalid markup.
                <div
                  key={p.id || p.name}
                  className={`relative inline-flex items-center rounded-lg border transition-all duration-200 ${
                    isSel && sliderEffectEnabled
                      ? 'bg-[var(--accent)] text-white border-white/20 shadow-[0_0_10px_var(--accent-glow)]'
                      : 'bg-white/[0.03] border-white/10 text-white/70 hover:border-white/20 hover:text-white'
                  } ${sliderEffectEnabled ? '' : 'opacity-35'}`}
                >
                  <button
                    type="button"
                    onClick={() => applyPreset(p)}
                    disabled={!sliderEffectEnabled}
                    className="px-2.5 py-1 text-[10px] font-bold disabled:cursor-not-allowed"
                  >
                    {p.name}
                  </button>
                  {p.isCustom && (
                    <button
                      type="button"
                      onClick={(e) => handleDeleteCustomPreset(p.id, e)}
                      className="mr-1.5 px-1 rounded bg-black/40 text-[10px] font-extrabold hover:text-red-300 transition-colors"
                      title="Delete custom preset"
                    >
                      ×
                    </button>
                  )}
                </div>
              );
            })}
          </div>

          {/* Right Save Preset Button / Inline Input */}
          <div className="flex items-center gap-1.5">
            {isSaving ? (
              <div className="flex items-center gap-1.5 bg-black/60 p-1 rounded-xl border border-[var(--accent)]/40 shadow-lg">
                <input
                  type="text"
                  placeholder="Preset Name..."
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSaveCurrentPreset()}
                  className="px-2 py-1 rounded-lg bg-black/80 text-white text-[11px] border border-white/10 focus:outline-none focus:border-[var(--accent)]"
                  autoFocus
                />
                <button
                  type="button"
                  onClick={handleSaveCurrentPreset}
                  disabled={!newPresetName.trim()}
                  className="px-2.5 py-1 rounded-lg text-[10px] font-bold bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors disabled:opacity-40"
                >
                  Save
                </button>
                <button
                  type="button"
                  onClick={() => setIsSaving(false)}
                  className="px-2 py-1 rounded-lg text-[10px] font-bold bg-white/10 text-white/60 hover:text-white transition-colors"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setIsSaving(true)}
                disabled={!sliderEffectEnabled}
                className="px-2.5 py-1 rounded-lg text-[10px] font-bold bg-[var(--accent)]/15 hover:bg-[var(--accent)]/25 border border-[var(--accent)]/40 text-white transition-all shadow-[0_0_10px_var(--accent-glow)] flex items-center gap-1 disabled:opacity-40 cursor-pointer"
                title="Save current slider values as a custom preset"
              >
                <span>💾 Save Preset</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Grid of slider controls */}
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div
              className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mt-3 pt-3 border-t border-white/5 transition-opacity duration-300 ${
                sliderEffectEnabled ? 'opacity-100' : 'opacity-55 grayscale-[25%]'
              }`}
            >
              {TRACKER_SLIDERS.map((s) => {
                const rawVal = params[s.key];
                const val = num(rawVal, s.defaultVal);
                const percent = Math.max(
                  0,
                  Math.min(100, ((val - s.min) / (s.max - s.min)) * 100)
                );
                const isModified = val !== s.defaultVal;

                return (
                  <div
                    key={s.key}
                    className={`group/card relative p-3 rounded-xl bg-black/40 border transition-all duration-200 ${
                      isModified && sliderEffectEnabled
                        ? 'border-[var(--accent)]/40 bg-black/60 shadow-[0_0_12px_rgba(233,69,96,0.08)]'
                        : 'border-white/5 hover:border-white/15'
                    }`}
                  >
                    {/* Card Header: Icon, Label, Value */}
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span className="text-xs shrink-0">{s.icon}</span>
                        <span
                          className="text-[11px] font-semibold text-white/80 truncate group-hover/card:text-white transition-colors"
                          title={s.info}
                        >
                          {s.label}
                        </span>
                      </div>
                      <span className="text-xs font-mono font-bold tabular-nums text-[var(--accent)] shrink-0 bg-[var(--accent)]/10 px-1.5 py-0.5 rounded border border-[var(--accent)]/20">
                        {s.format(val)}
                      </span>
                    </div>

                    {/* Range Input Slider with gradient background fill */}
                    <div className="relative flex items-center">
                      <input
                        type="range"
                        min={s.min}
                        max={s.max}
                        step={s.step}
                        value={val}
                        disabled={!sliderEffectEnabled}
                        onChange={(e) => onSetParam && onSetParam(s.key, parseFloat(e.target.value))}
                        className="w-full h-1.5 rounded-lg appearance-none bg-white/10 cursor-pointer accent-[var(--accent)] focus:outline-none disabled:cursor-not-allowed"
                        style={{
                          background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${percent}%, rgba(255,255,255,0.1) ${percent}%, rgba(255,255,255,0.1) 100%)`,
                        }}
                      />
                    </div>

                    {/* Footer: Min, Default indicator, Max */}
                    <div className="flex items-center justify-between text-[9px] font-mono text-white/35 mt-1.5">
                      <span>{s.min}</span>
                      <button
                        type="button"
                        onClick={() => sliderEffectEnabled && onSetParam && onSetParam(s.key, s.defaultVal)}
                        disabled={!sliderEffectEnabled}
                        className={`hover:text-white transition-colors cursor-pointer ${
                          isModified ? 'text-[var(--accent)] font-bold' : ''
                        }`}
                        title="Click to reset to default"
                      >
                        def: {s.defaultVal}
                      </button>
                      <span>{s.max}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
