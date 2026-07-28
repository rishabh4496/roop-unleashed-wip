import React from 'react';

/**
 * MediaTabSessionBar
 * Allows working with multiple media targets in a tabbed session workspace.
 */
export default function MediaTabSessionBar({
  targets = [],
  selTarget = 0,
  onSelectTarget,
  onRemoveTarget,
  onAddTarget,
}) {
  if (!targets || targets.length <= 1) return null;

  return (
    <div className="flex items-center gap-1.5 overflow-x-auto rounded-xl border border-white/10 bg-neutral-900/60 p-1.5 backdrop-blur-md no-scrollbar">
      <div className="flex items-center gap-1 px-2 text-[11px] font-bold uppercase tracking-wider text-neutral-400">
        <span>🎬</span>
        <span>Sessions</span>
      </div>

      {targets.map((target, idx) => {
        const isSelected = selTarget === idx;
        const fileName = typeof target === 'string' ? target.split(/[\\/]/).pop() : `Target ${idx + 1}`;

        return (
          <div
            key={idx}
            onClick={() => onSelectTarget(idx)}
            className={`group relative flex cursor-pointer items-center gap-2 rounded-lg px-3 py-1.5 text-xs transition-all ${
              isSelected
                ? 'bg-indigo-600/30 text-indigo-200 border border-indigo-500/40 font-semibold shadow-sm'
                : 'text-neutral-300 hover:bg-white/10 hover:text-white border border-transparent'
            }`}
          >
            <span className="truncate max-w-[140px]">{fileName}</span>
            {targets.length > 1 && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRemoveTarget(idx);
                }}
                className="opacity-0 group-hover:opacity-15 hover:!opacity-100 hover:text-rose-400 transition-opacity p-0.5 rounded"
                title="Close target session"
              >
                ✕
              </button>
            )}
          </div>
        );
      })}

      {onAddTarget && (
        <button
          onClick={onAddTarget}
          className="flex items-center gap-1 rounded-lg border border-dashed border-white/20 px-2.5 py-1.5 text-xs text-neutral-400 transition-all hover:border-indigo-400 hover:text-indigo-300"
          title="Add another target file session"
        >
          <span>+ Add File</span>
        </button>
      )}
    </div>
  );
}
