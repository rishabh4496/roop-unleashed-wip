import React, { useState } from 'react';
import { motion, AnimatePresence } from '../../motion';

/**
 * FloatingActionDock
 * Ultra-premium floating glassmorphic HUD dock anchored at bottom center.
 * Gives instant access to layout workspace switching, render action, preview toggles,
 * ambilight background glow, panel drawer controls, and pop-out window.
 */
export default function FloatingActionDock({
  workspaceMode,
  setWorkspaceMode,
  isRendering,
  onStartSwap,
  onCancelSwap,
  progress,
  onPreview,
  previewing,
  ambilightEnabled,
  setAmbilightEnabled,
  onOpenPopout,
  onOpenPresetStudio,
  drawers,
  setDrawers,
}) {
  const [showModeMenu, setShowModeMenu] = useState(false);

  // Descriptions state what each mode ACTUALLY does — the table these map onto
  // lives in FaceSwap.jsx (WORKSPACE_LAYOUT). Keep the two in step.
  const MODES = [
    { id: 'default', label: '🎛️ Standard Studio', desc: 'Faces, settings and timeline' },
    { id: 'cinema', label: '🎬 Cinema Canvas', desc: 'Canvas only — panels and timeline hidden' },
    { id: 'dual', label: '👥 Dual Inspector', desc: 'Faces + parameters, no timeline' },
    { id: 'timeline', label: '🎞️ Timeline Deck', desc: 'Canvas and timeline, panels hidden' },
  ];

  const currentMode = MODES.find((m) => m.id === workspaceMode) || MODES[0];

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 pointer-events-auto transition-all duration-300">
      <div className="relative flex items-center gap-2.5 rounded-full border border-white/15 bg-neutral-950/85 px-4 py-2.5 shadow-2xl backdrop-blur-xl ring-1 ring-white/10">
        
        {/* Render Button with Live Status Ring */}
        <div className="relative flex items-center">
          {isRendering ? (
            <button
              onClick={onCancelSwap}
              className="relative flex items-center gap-2 rounded-full bg-rose-600/90 px-4 py-2 text-xs font-semibold text-white shadow-lg transition-all hover:bg-rose-500 active:scale-95"
              title="Cancel current swap job"
            >
              <span className="h-2 w-2 rounded-full bg-white animate-ping" />
              <span>Cancel ({Math.round(progress)}%)</span>
            </button>
          ) : (
            <button
              onClick={onStartSwap}
              className="group relative flex items-center gap-2 overflow-hidden rounded-full bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 px-5 py-2 text-xs font-bold text-white shadow-lg shadow-indigo-500/25 transition-all hover:shadow-indigo-500/40 hover:scale-105 active:scale-95"
              title="Start Face Swap processing"
            >
              <span className="text-base group-hover:scale-110 transition-transform">▶</span>
              <span>Start Swap</span>
            </button>
          )}
        </div>

        <div className="h-4 w-px bg-white/15" />

        {/* Instant Preview Button */}
        <button
          onClick={onPreview}
          disabled={previewing}
          className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-all ${
            previewing
              ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
              : 'text-neutral-300 hover:bg-white/10 hover:text-white'
          }`}
          title="Generate instant preview frame"
          aria-label="Generate instant preview frame"
        >
          <span>👁️</span>
          <span>{previewing ? 'Rendering...' : 'Preview'}</span>
        </button>

        <div className="h-4 w-px bg-white/15" />

        {/* Workspace Layout Switcher Menu */}
        <div className="relative">
          <button
            onClick={() => setShowModeMenu(!showModeMenu)}
            className="flex items-center gap-1.5 rounded-full bg-white/5 border border-white/10 px-3 py-1.5 text-xs font-medium text-neutral-200 transition-all hover:bg-white/10 hover:text-white"
            title="Switch Workspace Layout Mode"
          >
            <span>{currentMode.label.split(' ')[0]}</span>
            <span className="hidden sm:inline">{currentMode.label.split(' ').slice(1).join(' ')}</span>
            <span className="text-[10px] opacity-60">▼</span>
          </button>

          <AnimatePresence>
            {showModeMenu && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                className="absolute bottom-full left-0 mb-3 w-56 rounded-2xl border border-white/15 bg-neutral-900/95 p-2 shadow-2xl backdrop-blur-2xl"
              >
                <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-neutral-400">
                  Workspace Layout Mode
                </div>
                <div className="space-y-1">
                  {MODES.map((mode) => (
                    <button
                      key={mode.id}
                      onClick={() => {
                        setWorkspaceMode(mode.id);
                        setShowModeMenu(false);
                      }}
                      className={`w-full rounded-xl px-3 py-2 text-left transition-all ${
                        workspaceMode === mode.id
                          ? 'bg-indigo-600/30 text-indigo-200 border border-indigo-500/40 font-semibold'
                          : 'text-neutral-300 hover:bg-white/10 hover:text-white'
                      }`}
                    >
                      <div className="text-xs">{mode.label}</div>
                      <div className="text-[10px] text-neutral-400">{mode.desc}</div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="h-4 w-px bg-white/15" />

        {/* Ambilight Glow Toggle */}
        <button
          onClick={() => setAmbilightEnabled(!ambilightEnabled)}
          className={`flex items-center gap-1 rounded-full px-2.5 py-1.5 text-xs transition-all ${
            ambilightEnabled
              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
              : 'text-neutral-400 hover:bg-white/10 hover:text-neutral-200'
          }`}
          title="Toggle Canvas Ambilight Glow"
        >
          <span>✨</span>
          <span className="hidden md:inline">Glow</span>
        </button>

        {/* Preset Studio Trigger */}
        {onOpenPresetStudio && (
          <button
            onClick={onOpenPresetStudio}
            className="flex items-center gap-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 px-2.5 py-1.5 text-xs text-indigo-300 transition-all hover:bg-indigo-500/20 hover:text-indigo-200"
            title="Open Preset Studio & Quality Recipes"
          >
            <span>🎨</span>
            <span className="hidden md:inline">Recipes</span>
          </button>
        )}

        {/* External Pop-out Window */}
        <button
          onClick={onOpenPopout}
          className="flex items-center gap-1 rounded-full px-2.5 py-1.5 text-xs text-neutral-400 transition-all hover:bg-white/10 hover:text-neutral-200"
          title="Open Detached Pop-out Preview Monitor"
        >
          <span>↗️</span>
          <span className="hidden md:inline">Pop-out</span>
        </button>

        <div className="h-4 w-px bg-white/15" />

        {/* Panel Drawer Collapse Toggles */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setDrawers((d) => ({ ...d, left: !d.left }))}
            className={`rounded-lg p-1.5 text-xs transition-all ${
              drawers.left ? 'text-indigo-400 bg-white/10' : 'text-neutral-500 hover:text-neutral-300'
            }`}
            title="Toggle Left Faces Sidebar"
          >
            📂
          </button>

          <button
            onClick={() => setDrawers((d) => ({ ...d, right: !d.right }))}
            className={`rounded-lg p-1.5 text-xs transition-all ${
              drawers.right ? 'text-indigo-400 bg-white/10' : 'text-neutral-500 hover:text-neutral-300'
            }`}
            title="Toggle Right Settings Inspector"
          >
            ⚙️
          </button>

          <button
            onClick={() => setDrawers((d) => ({ ...d, bottom: !d.bottom }))}
            className={`rounded-lg p-1.5 text-xs transition-all ${
              drawers.bottom ? 'text-indigo-400 bg-white/10' : 'text-neutral-500 hover:text-neutral-300'
            }`}
            title="Toggle Bottom Timeline & Logs Deck"
          >
            🎞️
          </button>
        </div>

      </div>
    </div>
  );
}
