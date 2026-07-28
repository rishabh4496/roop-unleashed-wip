import React from 'react';

/**
 * LiveProcessingPeek
 * Renders a visual frame thumbnail snapshot inside the processing box while a job runs.
 * Shows active frame position, resolution, and real-time processing status badge.
 */
export default function LiveProcessingPeek({
  previewSrc,
  rawUrl,
  frame = 1,
  maxFrames = 1,
  progressDesc = '',
  paused = false,
}) {
  const activeImage = previewSrc || rawUrl;

  return (
    <div className="relative group overflow-hidden rounded-2xl border border-white/15 bg-black/60 shadow-xl backdrop-blur-md transition-all duration-300">
      {/* Media Frame Container */}
      <div className="relative aspect-video w-full flex items-center justify-center overflow-hidden bg-neutral-950">
        {activeImage ? (
          <img
            src={activeImage}
            alt="Live Processing Frame Peek"
            className="h-full w-full object-contain transition-all duration-300 transform-gpu"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-neutral-500">
            <div className="h-8 w-8 rounded-full border-2 border-white/10 border-t-indigo-500 animate-spin" />
            <span className="text-xs">Buffering live frame...</span>
          </div>
        )}

        {/* Live Status Overlay Badges */}
        <div className="absolute top-3 left-3 flex items-center gap-2 z-10">
          <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider backdrop-blur-md border ${
            paused
              ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
              : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
          }`}>
            {paused ? '⏸ PAUSED' : '🔴 LIVE FRAME PEEK'}
          </span>

          {maxFrames > 1 && (
            <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-black/70 backdrop-blur-md border border-white/15 text-white">
              Frame {frame} / {maxFrames}
            </span>
          )}
        </div>

        {/* Bottom Description Pill */}
        {progressDesc && (
          <div className="absolute bottom-3 left-3 right-3 z-10">
            <div className="px-3 py-1.5 rounded-xl bg-black/80 backdrop-blur-md border border-white/10 text-xs font-mono text-neutral-200 truncate">
              <span className="text-indigo-400 font-bold mr-2">STATE:</span>
              <span>{progressDesc}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
