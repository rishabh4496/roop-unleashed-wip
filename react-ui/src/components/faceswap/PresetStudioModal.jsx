import React, { useState } from 'react';
import { motion, AnimatePresence } from '../../motion';

/**
 * PresetStudioModal
 * Preset Studio & Recipe Manager modal for Roop Unleashed.
 * Includes curated 1-click quality recipes, visual parameter diffs, export/import, and recipe sharing.
 */
export default function PresetStudioModal({
  isOpen,
  onClose,
  activeParams = {},
  onApplyRecipe,
  onExportRecipe,
  onImportRecipe,
  notify,
}) {
  const [selectedRecipe, setSelectedRecipe] = useState(null);

  const CURATED_RECIPES = [
    {
      id: 'cinematic_4k',
      title: '🎬 Cinematic 4K Ultra',
      desc: 'Maximum fidelity & sharpness. Uses GPEN face enhancer + Real-ESRGAN x4 upscale + temporal anti-flicker.',
      badge: 'PRO QUALITY',
      color: 'from-amber-500/20 to-orange-500/20 border-amber-500/40 text-amber-300',
      params: {
        selected_enhancer: 'GPEN-BFR-512',
        enhancer_blend: 0.85,
        face_upscaler: 'esrganx4',
        temporal_detection: true,
        stabilize_enhancer: true,
        jaw_reshape: true,
        jaw_reshape_strength: 0.5,
        mask_blur: 0.3,
      },
    },
    {
      id: 'fast_draft',
      title: '⚡ Fast Live Draft',
      desc: 'Optimized for instant preview speed. Disables heavy enhancers and uses fast Lanczos scaling.',
      badge: 'HIGH SPEED',
      color: 'from-blue-500/20 to-cyan-500/20 border-blue-500/40 text-blue-300',
      params: {
        selected_enhancer: 'None',
        enhancer_blend: 0.5,
        face_upscaler: 'lanczos_x2',
        temporal_detection: false,
        stabilize_enhancer: false,
        jaw_reshape: false,
      },
    },
    {
      id: 'anime_art',
      title: '🎨 Anime & Stylized Art',
      desc: 'Tailored for illustration and anime characters using Real-ESRGAN Anime x4 and DFL XSeg mask.',
      badge: 'ART & ANIME',
      color: 'from-pink-500/20 to-purple-500/20 border-pink-500/40 text-pink-300',
      params: {
        selected_enhancer: 'Restoreformer++',
        enhancer_blend: 0.75,
        face_upscaler: 'esrgan_anime_x4',
        mask_engine: 'DFL XSeg',
        temporal_detection: true,
      },
    },
    {
      id: 'group_scene',
      title: '👥 Multi-Person Crowd',
      desc: 'Best for scenes with multiple people. Identity locking enabled with temporal gap filling.',
      badge: 'MULTI-FACE',
      color: 'from-emerald-500/20 to-teal-500/20 border-emerald-500/40 text-emerald-300',
      params: {
        selected_enhancer: 'GFPGAN',
        enhancer_blend: 0.8,
        temporal_detection: true,
        stabilize_enhancer: true,
        no_face_action: 'skip',
      },
    },
  ];

  if (!isOpen) return null;

  const currentRecipe = selectedRecipe || CURATED_RECIPES[0];

  // Calculate parameter differences between active params and selected recipe
  const diffs = Object.entries(currentRecipe.params).map(([key, value]) => {
    const activeVal = activeParams[key];
    const isDifferent = activeVal !== value;
    return { key, value, activeVal, isDifferent };
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 15 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 15 }}
        className="relative w-full max-w-4xl max-h-[85vh] flex flex-col overflow-hidden rounded-3xl border border-white/15 bg-neutral-900 shadow-2xl"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-neutral-950/60">
          <div className="flex items-center gap-3">
            <span className="text-2xl">✨</span>
            <div>
              <h2 className="text-lg font-bold text-white">Preset Studio & Recipe Manager</h2>
              <p className="text-xs text-neutral-400">Curated 1-click quality recipes & configuration diff inspector</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="flex items-center justify-center h-8 w-8 rounded-full bg-white/5 border border-white/10 text-neutral-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6 overflow-y-auto flex-1">
          {/* Recipe List */}
          <div className="space-y-3">
            <div className="text-xs font-bold uppercase tracking-wider text-neutral-400">
              Curated Studio Recipes
            </div>
            <div className="space-y-2.5">
              {CURATED_RECIPES.map((recipe) => {
                const isSelected = currentRecipe.id === recipe.id;
                return (
                  <div
                    key={recipe.id}
                    onClick={() => setSelectedRecipe(recipe)}
                    className={`cursor-pointer rounded-2xl p-4 border transition-all duration-200 ${
                      isSelected
                        ? `bg-gradient-to-r ${recipe.color} shadow-lg ring-1 ring-white/20`
                        : 'bg-white/[0.03] border-white/10 hover:border-white/20 hover:bg-white/[0.06]'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-bold text-white">{recipe.title}</span>
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wider uppercase bg-white/10 border border-white/15 text-white/90">
                        {recipe.badge}
                      </span>
                    </div>
                    <p className="text-xs text-neutral-300 mt-1.5 leading-relaxed">{recipe.desc}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Parameter Diff Inspector */}
          <div className="flex flex-col space-y-4 rounded-2xl border border-white/10 bg-black/40 p-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-neutral-400">
                Recipe Parameter Diff
              </span>
              <span className="text-xs font-semibold text-indigo-400">{currentRecipe.title}</span>
            </div>

            <div className="space-y-2 flex-1 overflow-y-auto pr-1">
              {diffs.map(({ key, value, activeVal, isDifferent }) => (
                <div
                  key={key}
                  className={`flex items-center justify-between p-2.5 rounded-xl border text-xs font-mono transition-all ${
                    isDifferent
                      ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-200'
                      : 'bg-white/[0.02] border-white/5 text-neutral-400'
                  }`}
                >
                  <span className="font-semibold text-white/80">{key}</span>
                  <div className="flex items-center gap-2 text-right">
                    {isDifferent && (
                      <span className="line-through text-neutral-500 text-[10px]">
                        {String(activeVal ?? 'off')}
                      </span>
                    )}
                    <span className={`font-bold ${isDifferent ? 'text-indigo-400' : 'text-neutral-300'}`}>
                      {String(value)}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Action Buttons */}
            <div className="pt-3 border-t border-white/10 flex items-center justify-between gap-3">
              <div className="flex gap-2">
                <button
                  onClick={onExportRecipe}
                  className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-white/5 border border-white/10 text-neutral-300 hover:bg-white/10 hover:text-white transition-all"
                  title="Export active settings as JSON recipe"
                >
                  📤 Export
                </button>
                <label className="cursor-pointer px-3 py-1.5 rounded-xl text-xs font-semibold bg-white/5 border border-white/10 text-neutral-300 hover:bg-white/10 hover:text-white transition-all">
                  📥 Import
                  <input type="file" accept=".json" onChange={onImportRecipe} className="hidden" />
                </label>
              </div>

              <button
                onClick={() => {
                  onApplyRecipe(currentRecipe.params);
                  notify(`Applied recipe: ${currentRecipe.title}`, 'success');
                  onClose();
                }}
                className="px-5 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-indigo-500/20 hover:scale-105 active:scale-95 transition-all"
              >
                Apply Recipe ✨
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
