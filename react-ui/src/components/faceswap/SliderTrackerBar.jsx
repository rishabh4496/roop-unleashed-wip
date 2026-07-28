import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const TRACKER_SLIDERS = [
  {
    key: 'blend_ratio',
    label: 'Original / Enhanced Blend',
    icon: '🧪',
    min: 0,
    max: 1,
    step: 0.01,
    defaultVal: 0.8,
    format: (v) => Number(v ?? 0.8).toFixed(2),
    info: 'Blends between original target face and enhanced swapped result.',
  },
  {
    key: 'detail_transfer_strength',
    label: 'Skin Detail Transfer',
    icon: '✨',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: "Transfers target's original skin texture (pores, grain, stubble) onto swapped face.",
  },
  {
    key: 'expression_restore_strength',
    label: 'Expression Restore',
    icon: '😀',
    min: 0,
    max: 2,
    step: 0.05,
    defaultVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: "Re-applies target's original expression using LivePortrait model.",
  },
  {
    key: 'face_mask_blend',
    label: 'Face Mask Edge Blend',
    icon: '🎭',
    min: 0,
    max: 200,
    step: 1,
    defaultVal: 20,
    format: (v) => `${Math.round(v ?? 20)}px`,
    info: 'Feathering softness around the border of the face mask.',
  },
  {
    key: 'max_face_distance',
    label: 'Max Face Similarity',
    icon: '🎯',
    min: 0.01,
    max: 1,
    step: 0.01,
    defaultVal: 0.85,
    format: (v) => Number(v ?? 0.85).toFixed(2),
    info: 'Face matching distance threshold. Lower is stricter.',
  },
  {
    key: 'num_swap_steps',
    label: 'Swapping Steps',
    icon: '⚡',
    min: 1,
    max: 5,
    step: 1,
    defaultVal: 1,
    format: (v) => `${Math.round(v ?? 1)}×`,
    info: 'Swap iteration passes. Higher values increase source likeness.',
  },
  {
    key: 'jaw_reshape_strength',
    label: 'Jaw Reshape Strength',
    icon: '🗿',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0.5,
    format: (v) => Number(v ?? 0.5).toFixed(2),
    info: 'Blends source jaw & chin contours onto target face.',
  },
  {
    key: 'stabilize_enhancer_strength',
    label: 'Flicker Reduction',
    icon: '🎬',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0.5,
    format: (v) => Number(v ?? 0.5).toFixed(2),
    info: 'Temporal stabilization strength for video frames.',
  },
];

const BUILTIN_PRESETS = [
  {
    name: 'Default',
    values: {
      blend_ratio: 0.8,
      detail_transfer_strength: 0,
      expression_restore_strength: 0,
      face_mask_blend: 20,
      max_face_distance: 0.85,
      num_swap_steps: 1,
      jaw_reshape_strength: 0.5,
      stabilize_enhancer_strength: 0.5,
    },
  },
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
  const [activePreset, setActivePreset] = useState('Default');
  const [customPresets, setCustomPresets] = useState([]);
  const [isSaving, setIsSaving] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');

  // Load custom presets from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('roop_user_slider_presets');
      if (saved) {
        setCustomPresets(JSON.parse(saved));
      }
    } catch {
      /* ignore */
    }
  }, []);

  // Save custom presets to localStorage
  const saveCustomPresetsToStorage = (updated) => {
    setCustomPresets(updated);
    try {
      localStorage.setItem('roop_user_slider_presets', JSON.stringify(updated));
    } catch {
      /* ignore */
    }
  };

  const applyPreset = (pObj) => {
    setActivePreset(pObj.name);
    Object.entries(pObj.values).forEach(([k, v]) => {
      if (onSetParam) onSetParam(k, v);
    });
    if (onRefreshPreview) onRefreshPreview();
  };

  const handleSaveCurrentPreset = () => {
    const name = newPresetName.trim();
    if (!name) return;

    // Collect current values of the 8 sliders from params
    const values = {};
    TRACKER_SLIDERS.forEach((s) => {
      values[s.key] = num(params[s.key], s.defaultVal);
    });

    const newPreset = {
      name: `★ ${name}`,
      values,
      isCustom: true,
      id: Date.now().toString(),
    };

    const updated = [...customPresets.filter((p) => p.name !== newPreset.name), newPreset];
    saveCustomPresetsToStorage(updated);
    setActivePreset(newPreset.name);
    setNewPresetName('');
    setIsSaving(false);
  };

  const handleDeleteCustomPreset = (presetId, e) => {
    e.stopPropagation();
    const updated = customPresets.filter((p) => p.id !== presetId);
    saveCustomPresetsToStorage(updated);
    if (activePreset.includes(presetId)) {
      setActivePreset('Default');
    }
  };

  const num = (v, fallback) => (typeof v === 'number' && !isNaN(v) ? v : fallback);

  const allPresets = [...BUILTIN_PRESETS, ...customPresets];

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
              onClick={() => {
                setActivePreset('Default');
                onResetSliders();
              }}
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
            {allPresets.map((p) => {
              const isSel = activePreset === p.name;
              return (
                <div key={p.name || p.id} className="relative inline-flex items-center">
                  <button
                    type="button"
                    onClick={() => applyPreset(p)}
                    disabled={!sliderEffectEnabled}
                    className={`px-2.5 py-1 rounded-lg text-[10px] font-bold border transition-all duration-200 flex items-center gap-1.5 ${
                      isSel && sliderEffectEnabled
                        ? 'bg-[var(--accent)] text-white border-white/20 shadow-[0_0_10px_var(--accent-glow)]'
                        : 'bg-white/[0.03] border-white/10 text-white/70 hover:border-white/20 hover:text-white'
                    } disabled:opacity-35 disabled:cursor-not-allowed`}
                  >
                    <span>{p.name}</span>
                    {p.isCustom && (
                      <span
                        onClick={(e) => handleDeleteCustomPreset(p.id, e)}
                        className="hover:text-red-300 font-extrabold ml-1 px-1 rounded bg-black/40"
                        title="Delete custom preset"
                      >
                        ×
                      </span>
                    )}
                  </button>
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
                        onChange={(e) => {
                          setActivePreset('Custom');
                          if (onSetParam) onSetParam(s.key, parseFloat(e.target.value));
                        }}
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
                        onClick={() => {
                          if (sliderEffectEnabled && onSetParam) {
                            setActivePreset('Custom');
                            onSetParam(s.key, s.defaultVal);
                          }
                        }}
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
