import React, { useEffect, useRef, useState } from 'react';

/**
 * Generic side-by-side comparison grid with a single shared zoom/pan, used by
 * both the Enhancer grid and the Mask-engine grid. Each cell shows one variant's
 * rendered preview (or a live "Rendering…" spinner while it's in flight).
 *
 * Props:
 *   items          string[]  — variant labels to show, one cell each
 *   previews       {label: dataURL}
 *   times          {label: "1.23s" | "Cached"}
 *   timers         {label: "0.4s"}  — live elapsed while rendering
 *   gridColsClass  tailwind grid-cols class
 *   emptyHint      optional string shown center when there are no items
 */
export default function CompareGrid({ items, previews, times, timers, gridColsClass, emptyHint }) {
  // Zoom/pan is shared by every cell, so the same region is magnified in all
  // variants at once — that's what makes fine differences readable.
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  useEffect(() => {
    const handleUp = () => setIsPanning(false);
    window.addEventListener('mouseup', handleUp);
    window.addEventListener('touchend', handleUp);
    return () => {
      window.removeEventListener('mouseup', handleUp);
      window.removeEventListener('touchend', handleUp);
    };
  }, []);

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomSpeed = 0.15;
    const newZoom = Math.min(Math.max(1, zoom + (e.deltaY < 0 ? zoomSpeed : -zoomSpeed)), 8);
    if (newZoom === 1) setPan({ x: 0, y: 0 });
    setZoom(newZoom);
  };

  const handlePointerDown = (e) => {
    if (zoom > 1) {
      setIsPanning(true);
      setStartPan({ x: (e.clientX ?? e.touches?.[0]?.clientX) - pan.x, y: (e.clientY ?? e.touches?.[0]?.clientY) - pan.y });
    }
  };

  const handlePointerMove = (e) => {
    if (!isPanning || zoom <= 1) return;
    const cx = e.clientX ?? e.touches?.[0]?.clientX;
    const cy = e.clientY ?? e.touches?.[0]?.clientY;
    setPan({ x: cx - startPan.x, y: cy - startPan.y });
  };

  // Double click toggles fit ↔ 2.5x centered near the clicked spot of that cell
  const handleDoubleClick = (e) => {
    if (zoom > 1) {
      setZoom(1);
      setPan({ x: 0, y: 0 });
    } else {
      setZoom(2.5);
      const cell = e.currentTarget;
      const rect = cell.getBoundingClientRect();
      const clickX = (e.clientX ?? e.touches?.[0]?.clientX) - rect.left;
      const clickY = (e.clientY ?? e.touches?.[0]?.clientY) - rect.top;
      setPan({
        x: (rect.width / 2 - clickX) * 1.5,
        y: (rect.height / 2 - clickY) * 1.5
      });
    }
  };

  const resetZoom = () => { setZoom(1); setPan({ x: 0, y: 0 }); };
  const transformStyle = { transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`, transformOrigin: 'center' };

  return (
    <div
      ref={containerRef}
      className={`relative grid ${gridColsClass} gap-3 aspect-video max-h-[45vh] rounded-2xl overflow-hidden bg-black/45 border border-white/5 p-2`}
      onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove}
      style={{ touchAction: 'none', cursor: zoom > 1 ? (isPanning ? 'grabbing' : 'grab') : 'zoom-in' }}
    >
      {items.length === 0 && emptyHint && (
        <div className="col-span-full flex items-center justify-center text-xs text-white/40 font-semibold p-6 text-center">
          {emptyHint}
        </div>
      )}
      {items.map((label) => (
        <div key={label} className="relative rounded-xl overflow-hidden bg-black/50 border border-white/5 flex items-center justify-center"
             onDoubleClick={handleDoubleClick}>
          {previews[label] ? (
            <div className="w-full h-full flex items-center justify-center transition-transform duration-75 select-none" style={transformStyle}>
              <img src={previews[label]} alt={label} className="max-w-full max-h-full object-contain pointer-events-none" draggable={false} />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center gap-2 text-white/40 text-xs px-3 text-center">
              <span className="h-4 w-4 rounded-full border-2 border-white/20 border-t-[var(--accent)] animate-spin" />
              <span className="font-semibold">Rendering {label}…</span>
              {timers[label] && (
                <span className="text-micro text-white/30 font-mono">
                  Elapsed: {timers[label]}
                </span>
              )}
            </div>
          )}
          <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur border border-white/10 text-micro font-bold text-white uppercase pointer-events-none max-w-[calc(100%-1rem)] truncate">{label}</span>
          {times[label] && (
            <span className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur border border-white/10 text-micro font-bold text-white/60 font-mono pointer-events-none">
              ⏱️ {times[label]}
            </span>
          )}
        </div>
      ))}
      {zoom > 1 && (
        <button
          type="button"
          onClick={resetZoom}
          className="absolute top-3 right-3 z-30 px-2.5 py-1 rounded-full bg-black/70 backdrop-blur border border-white/10 text-mini font-bold text-white/80 hover:text-white hover:border-white/30 transition-all"
          title="Reset zoom (or double-click)"
        >
          🔍 {zoom.toFixed(1)}× — Reset
        </button>
      )}
      {zoom === 1 && items.length > 0 && (
        <span className="absolute top-3 right-3 z-30 px-2.5 py-1 rounded-full bg-black/50 backdrop-blur text-micro font-semibold text-white/40 pointer-events-none select-none">
          Scroll or double-click to zoom all
        </span>
      )}
    </div>
  );
}
