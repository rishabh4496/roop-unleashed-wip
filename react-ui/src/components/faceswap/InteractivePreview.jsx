import React, { useEffect, useRef, useState, useMemo } from 'react';
import AIScannerOverlay from './AIScannerOverlay';

export default function InteractivePreview({
  beforeSrc,
  afterSrc,
  faces = [],
  personIds = [],
  onSelectPerson,
  splitView = false,
  compare = false,
  onToggleCompare,
  frame = 1,
  setFrame,
  maxFrames = 1,
  isPlaying = false,
  setIsPlaying,
  previewing = false,
  previewSecs = 0,
  processing = false,
  liveFrame = null,
  liveSeq = 0
}) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [showBoxes, setShowBoxes] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [imgDim, setImgDim] = useState(null);

  const handleSliderMove = (clientX) => {
    const target = imageRef.current || containerRef.current;
    if (!target) return;
    const rect = target.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = Math.max(0, Math.min((x / rect.width) * 100, 100));
    setSliderPosition(percent);
  };

  useEffect(() => {
    const handleMouseUp = () => { setIsDraggingSlider(false); setIsPanning(false); };
    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('touchend', handleMouseUp);
    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, []);

  // Keep isFullscreen in sync when the user exits fullscreen via Esc (the
  // browser fires no click on our toggle in that case).
  useEffect(() => {
    const onFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  // Keyboard Navigation: Zoom controls
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignore if user is typing in input fields
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;

      if (e.key === '=' || e.key === '+') {
        e.preventDefault();
        setZoom((z) => Math.min(z + 0.5, 5));
      } else if (e.key === '-') {
        e.preventDefault();
        setZoom((z) => {
          const nz = Math.max(1, z - 0.5);
          if (nz === 1) setPan({ x: 0, y: 0 });
          return nz;
        });
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomSpeed = 0.15;
    const newZoom = Math.min(Math.max(1, zoom + (e.deltaY < 0 ? zoomSpeed : -zoomSpeed)), 5);
    if (newZoom === 1) setPan({ x: 0, y: 0 });
    setZoom(newZoom);
  };

  const handlePointerDown = (e) => {
    if (zoom > 1 && !isDraggingSlider) {
      setIsPanning(true);
      setStartPan({ x: (e.clientX || e.touches?.[0]?.clientX) - pan.x, y: (e.clientY || e.touches?.[0]?.clientY) - pan.y });
    }
  };

  // Pan/slider updates are coalesced to one per animation frame. Pointer-move
  // fires faster than React can re-render this whole subtree, so without this
  // the events queue up and the image visibly lags behind the cursor. rAF
  // batching keeps panning locked to the display refresh instead.
  const rafRef = useRef(null);
  const pendingRef = useRef(null);
  useEffect(() => () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); }, []);

  const handlePointerMove = (e) => {
    const cx = e.clientX ?? e.touches?.[0]?.clientX;
    const cy = e.clientY ?? e.touches?.[0]?.clientY;
    if (cx === undefined) return;
    if (!isDraggingSlider && !(isPanning && zoom > 1)) return;
    pendingRef.current = { cx, cy };
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const p = pendingRef.current;
      if (!p) return;
      if (isDraggingSlider) {
        handleSliderMove(p.cx);
      } else if (isPanning && zoom > 1) {
        setPan({ x: p.cx - startPan.x, y: p.cy - startPan.y });
      }
    });
  };

  // Double click toggles between fit-to-screen and 2.5x zoom
  const handleDoubleClick = (e) => {
    if (zoom > 1) {
      setZoom(1);
      setPan({ x: 0, y: 0 });
    } else {
      setZoom(2.5);
      // Center pan coordinates roughly where clicked
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const clickX = (e.clientX ?? e.touches?.[0]?.clientX) - rect.left;
        const clickY = (e.clientY ?? e.touches?.[0]?.clientY) - rect.top;
        setPan({
          x: (rect.width / 2 - clickX) * 1.5,
          y: (rect.height / 2 - clickY) * 1.5
        });
      }
    }
  };

  // Face boxes are pure geometry derived from the detections + image size, so
  // memoise them. Otherwise every pan/zoom re-render (many per second) would
  // rebuild the entire box list, which is wasted work while the user drags.
  const faceBoxes = useMemo(() => {
    if (!faces.length || !imgDim || !showBoxes) return null;
    const clickable = typeof onSelectPerson === 'function';
    return faces.map((bbox, i) => {
      const [sx, sy, ex, ey] = bbox;
      const left = (sx / imgDim.w) * 100;
      const top = (sy / imgDim.h) * 100;
      const width = ((ex - sx) / imgDim.w) * 100;
      const height = ((ey - sy) / imgDim.h) * 100;
      const label = (personIds[i] ?? i) + 1;
      return (
        <div
          key={i}
          className={`absolute border-2 border-[var(--accent)] shadow-[0_0_10px_var(--accent-glow)] z-20 ${clickable ? 'pointer-events-auto cursor-pointer group/face hover:bg-[var(--accent)]/10 transition-colors' : 'pointer-events-none'}`}
          style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
          title={clickable ? `Click to add Person ${label} to target faces` : undefined}
          onPointerDown={clickable ? (e) => e.stopPropagation() : undefined}
          onClick={clickable ? (e) => { e.stopPropagation(); onSelectPerson(i); } : undefined}
        >
          <span className="absolute -top-6 left-0 bg-[var(--accent)] text-white text-[10px] font-bold px-1.5 py-0.5 rounded whitespace-nowrap">
            Person {label}
          </span>
          {clickable && (
            <span className="absolute left-1/2 -translate-x-1/2 -bottom-6 opacity-0 group-hover/face:opacity-100 transition-opacity bg-black/80 backdrop-blur text-[var(--accent)] text-[9px] font-bold px-1.5 py-0.5 rounded whitespace-nowrap pointer-events-none">
              ＋ Add to targets
            </span>
          )}
        </div>
      );
    });
  }, [faces, imgDim, showBoxes, personIds, onSelectPerson]);

  // Skip the smoothing transition while the user is actively panning or the
  // slider is being dragged — otherwise every frame chases a 75ms ease and the
  // image trails the cursor. Discrete zoom (buttons/wheel) still eases nicely.
  const interacting = isPanning || isDraggingSlider;
  const transformStyle = {
    transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
    transformOrigin: 'center',
    willChange: zoom > 1 || interacting ? 'transform' : 'auto',
  };
  const aspectStyle = { aspectRatio: imgDim ? `${imgDim.w}/${imgDim.h}` : '1', maxHeight: '100%', maxWidth: '100%', display: 'flex' };

  const triggerFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  const stepFrame = (delta) => setFrame && setFrame((f) => Math.max(1, Math.min(maxFrames, f + delta)));
  const isVideo = maxFrames > 1;

  // Shared floating HUD control bar (zoom / boxes / compare / transport / fullscreen).
  const hudBar = () => (
    <div className="spring-cluster absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 p-1.5 rounded-xl hud-glass opacity-60 group-hover:opacity-100 hover:opacity-100 transition-all duration-300 z-50">
      <button onClick={() => setZoom(z => Math.min(z + 0.5, 5))} className="p-2 text-xs font-bold font-mono rounded-lg hud-glass-button" title="Zoom In">+</button>
      <button onClick={() => setZoom(z => { const nz = Math.max(1, z - 0.5); if (nz === 1) setPan({ x: 0, y: 0 }); return nz; })} className="p-2 text-xs font-bold font-mono rounded-lg hud-glass-button" title="Zoom Out">-</button>
      <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} className="px-2 py-1 text-[10px] font-bold rounded-lg hud-glass-button" title="Reset view">FIT</button>
      <div className="w-px h-4 bg-white/10 mx-1" />
      <button onClick={() => setShowBoxes(b => !b)} className={`px-2 py-1 text-[10px] font-bold rounded-lg apple-transition ${showBoxes ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'}`}>BOXES</button>
      {onToggleCompare && (
        <button onClick={() => onToggleCompare()} title="Toggle before/after compare (C)"
                className={`px-2 py-1 text-[10px] font-bold rounded-lg apple-transition ${compare ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'}`}>COMPARE</button>
      )}
      {isVideo && (
        <>
          <div className="w-px h-4 bg-white/10 mx-1" />
          <button onClick={() => stepFrame(-1)} className="p-2 rounded-lg hud-glass-button" title="Previous frame (←)">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zM20 6v12l-9-6z"/></svg>
          </button>
          <button onClick={() => setIsPlaying && setIsPlaying(p => !p)} className="p-2 rounded-lg hud-glass-button" title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}>
            {isPlaying
              ? <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>
              : <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>}
          </button>
          <button onClick={() => stepFrame(1)} className="p-2 rounded-lg hud-glass-button" title="Next frame (→)">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M16 6h2v12h-2zM4 6l9 6-9 6z"/></svg>
          </button>
          <span className="px-1 text-[10px] font-bold font-mono text-white/60 tabular-nums whitespace-nowrap">{frame}/{maxFrames}</span>
        </>
      )}
      <div className="w-px h-4 bg-white/10 mx-1" />
      <button onClick={triggerFullscreen} className="p-2 rounded-lg hud-glass-button" title="Toggle Fullscreen">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
      </button>
    </div>
  );

  // Split View comparisons
  if (compare && splitView) {
    return (
      <div
        ref={containerRef}
        className={`relative w-full aspect-video max-h-[45vh] rounded-2xl overflow-hidden bg-black/40 border border-white/5 group shadow-xl ${isFullscreen ? 'h-screen w-screen' : ''}`}
        onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onDoubleClick={handleDoubleClick} style={{ touchAction: 'none' }}
      >
        <div className={`flex w-full h-full ${interacting ? '' : 'transition-transform duration-75'}`} style={transformStyle}>
          <div className="flex-1 relative border-r border-white/10 flex items-center justify-center overflow-hidden bg-black/50">
            <div className="relative" style={aspectStyle}>
              <img src={beforeSrc} alt="Before" className="w-full h-full object-contain pointer-events-none"
                   onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })} />
              <div className="absolute inset-0 pointer-events-none">{faceBoxes}</div>
            </div>
            <span className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur text-[11px] font-bold text-white/80 uppercase">Before</span>
          </div>
          <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-black/50">
            <div className="relative" style={aspectStyle}>
              <img src={afterSrc} alt="After" className="w-full h-full object-contain pointer-events-none" />
            </div>
            <span className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-[var(--accent)]/80 backdrop-blur text-[11px] font-bold text-white uppercase">After</span>
          </div>
        </div>

        {/* HUD control bar overlays */}
        {hudBar()}
      </div>
    );
  }

  // Standard or Slide-Comparison View
  return (
    <div
      className={`relative w-full aspect-video max-h-[45vh] rounded-2xl overflow-hidden bg-black/40 border border-white/5 select-none group shadow-xl ${isFullscreen ? 'h-screen w-screen' : ''} ${previewing ? 'preview-glow' : ''}`}
      ref={containerRef} onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onDoubleClick={handleDoubleClick} style={{ touchAction: 'none' }}
    >
      {previewing && <AIScannerOverlay />}
      {previewing && (
        <div className="absolute top-3 right-3 flex flex-col items-end gap-1.5 z-50">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/70 backdrop-blur-md text-xs font-bold text-white/95 tabular-nums border border-white/10 shadow-2xl">
            <span className="h-3 w-3 rounded-full border-2 border-white/30 border-t-[var(--accent)] animate-spin" />
            Rendering… {previewSecs}s
          </div>
        </div>
      )}
      {processing && liveFrame && (
        <div className="absolute top-3 right-3 flex flex-col items-end gap-1.5 z-50">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--accent)] backdrop-blur-md text-[11px] font-semibold tracking-wide text-white border border-white/15 shadow-lg">
            <span className="h-1.5 w-1.5 rounded-full bg-white animate-ping" />
            Live swapping{liveSeq > 0 ? ` · ${liveSeq} frames` : ''}
          </div>
        </div>
      )}

      {/* Frame navigation shortcuts popup guide (visible when video) */}
      {maxFrames > 1 && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 px-2.5 py-1 rounded-lg bg-black/50 backdrop-blur text-[10px] font-bold text-white/50 border border-white/5 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-30 flex items-center gap-2">
          <span>Shortcuts:</span>
          <span className="bg-white/10 px-1 py-0.5 rounded text-white font-mono">←</span>
          <span className="bg-white/10 px-1 py-0.5 rounded text-white font-mono">→</span>
          <span>Frame</span>
          <span className="bg-white/10 px-2 py-0.5 rounded text-white font-mono">Space</span>
          <span>Compare</span>
        </div>
      )}

      <div className={`absolute inset-0 flex items-center justify-center ${interacting ? '' : 'transition-transform duration-75'}`} style={transformStyle}>

        {/* Before Image & Bounding Boxes Wrapper */}
        <div className="relative z-10" style={aspectStyle} ref={imageRef}>
          <img src={beforeSrc} alt="Before" className="w-full h-full object-contain pointer-events-none"
               onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })} draggable={false} />
          <div className="absolute inset-0 pointer-events-none z-30">{faceBoxes}</div>

          {/* After Image Overlay with Clip-path */}
          <div className="absolute inset-0 pointer-events-none z-20"
               style={{ clipPath: compare ? `polygon(${sliderPosition}% 0, 100% 0, 100% 100%, ${sliderPosition}% 100%)` : 'none' }}>
            <img src={afterSrc} alt="After" className="w-full h-full object-contain" draggable={false} />
          </div>

          {/* Slider Line & Handle (only when compare) */}
          {compare && (
            <div className="absolute top-0 bottom-0 w-0.5 bg-gradient-to-b from-[var(--accent)] via-white to-[var(--accent)] shadow-[0_0_8px_rgba(233,69,96,0.6)] z-40 pointer-events-none" style={{ left: `${sliderPosition}%` }}>
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-black/60 backdrop-blur border border-white/20 rounded-full flex items-center justify-center shadow-[0_4px_15px_rgba(0,0,0,0.5)] transition-transform duration-200 group-hover:scale-110 cursor-ew-resize pointer-events-auto"
                   onPointerDown={(e) => { e.stopPropagation(); setIsDraggingSlider(true); handleSliderMove(e.clientX ?? e.touches?.[0]?.clientX); }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="mr-0.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="rotate-180 ml-0.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
              </div>
            </div>
          )}
        </div>

      </div>

      {compare && <span className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur text-[11px] font-bold tracking-wider text-white/80 uppercase shadow z-30 pointer-events-none">Before</span>}
      <span className="absolute top-3 right-3 px-2.5 py-1 rounded-full bg-[var(--accent)]/80 backdrop-blur text-[11px] font-bold tracking-wider text-white uppercase shadow z-30 transition-opacity duration-300 pointer-events-none"
            style={{ opacity: compare && sliderPosition > 85 ? 0 : 1 }}>{compare ? 'After' : 'Swapped'}</span>

      {/* HUD control bar overlays */}
      {hudBar()}
    </div>
  );
}
