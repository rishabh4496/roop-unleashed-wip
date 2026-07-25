import React, { useEffect, useRef, useState, useMemo } from 'react';
import AIScannerOverlay from './AIScannerOverlay';

// Cross-fades between src changes using TWO persistent <img> layers that are
// never remounted. The incoming frame is loaded into the hidden (back) layer
// while the current frame stays painted on the front layer; only once the back
// layer has fully decoded do we flip which layer is on top, so the fade always
// runs from one complete frame to the next and never blanks to black in between.
// (Remounting a single <img> via a changing `key` is what caused the old black
// flicker — a fresh <img> paints nothing until it decodes.)
function CrossfadeImage({ src, className, style, fadeMs = 200, onLoad }) {
  const [layers, setLayers] = useState({ a: src, b: src, front: 'a' });

  useEffect(() => {
    setLayers((s) => {
      if (src === s[s.front]) return s;              // already the visible frame
      const back = s.front === 'a' ? 'b' : 'a';
      if (src === s[back]) return s;                 // already loading it
      return { ...s, [back]: src };                  // start decoding into back
    });
  }, [src]);

  const promote = (which, e) => {
    if (onLoad) onLoad(e);
    setLayers((s) => {
      if (s.front === which) return s;               // already front
      if (s[which] !== src) return s;                // stale load — newer src pending
      return { ...s, front: which };
    });
  };

  const renderLayer = (which) => (
    <img
      key={which}
      src={layers[which]}
      alt=""
      aria-hidden
      draggable={false}
      onLoad={(e) => promote(which, e)}
      className={className}
      style={{ ...style, opacity: layers.front === which ? 1 : 0, transition: `opacity ${fadeMs}ms ease-out` }}
    />
  );

  return <>{renderLayer('a')}{renderLayer('b')}</>;
}

// Pixel-peeping a 4K frame in a ~1000px stage needs ~4×; 5 was not quite enough
// to reach 1:1 on large sources, so the ceiling is 8.
const ZOOM_MAX = 8;

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
  scrubbing = false,
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
        setZoom((z) => Math.min(z + 0.5, ZOOM_MAX));
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

  // Wheel-to-zoom must be a NATIVE, non-passive listener. React registers its
  // onWheel handlers as passive, so calling preventDefault() there is silently
  // ignored — the page scrolls behind the zoom and the console warns. Binding
  // the listener ourselves with { passive: false } lets us actually stop the
  // page from scrolling while zooming the preview. Re-bind when the compare /
  // split branch swaps the container node. Functional setState avoids stale
  // zoom/pan captured in the effect closure.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      const dir = e.deltaY < 0 ? 0.15 : -0.15;
      setZoom((z) => {
        const nz = Math.min(Math.max(1, z + dir), ZOOM_MAX);
        if (nz === 1) setPan({ x: 0, y: 0 });
        return nz;
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [compare, splitView]);

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

  // Zoom to actual pixels: the frame is letterboxed to fit, so 1:1 is however
  // much magnification puts one source pixel on one screen pixel. Face-swap
  // artefacts (seams, mask edges, enhancer texture) live at that scale, and
  // guessing at it with +/- steps is exactly what made close inspection tedious.
  const zoomToActual = () => {
    // imageRef is only attached in the single-stage layout; fall back to the
    // container so the button still does something in split view.
    const box = (imageRef.current || containerRef.current)?.getBoundingClientRect();
    if (!box || !imgDim || !box.width) return;
    setZoom(Math.min(Math.max(imgDim.w / box.width, 1), ZOOM_MAX));
    setPan({ x: 0, y: 0 });
  };

  const zoomBy = (d) => setZoom((z) => {
    const nz = Math.min(Math.max(1, z + d), ZOOM_MAX);
    if (nz === 1) setPan({ x: 0, y: 0 });
    return nz;
  });

  // Shared floating HUD. One glass surface split into labelled zones (view /
  // overlays / transport / window) instead of a single undifferentiated row of
  // look-alike buttons, and it now reports the state it controls — the zoom
  // factor and the frame — rather than only offering the controls.
  const hudBar = () => (
    <div className="absolute inset-x-0 bottom-0 z-50 flex justify-center p-3 pointer-events-none">
      {/* Rests at reduced opacity rather than fully hidden: the bar is not just
          controls, it REPORTS state (zoom factor, face count, whether Compare
          is on), and state you have to hover to discover is state you will
          miss. Comes to full strength on hover or keyboard focus. */}
      <div className="spring-cluster pointer-events-auto flex items-center gap-0.5 rounded-xl hud-glass p-1 opacity-45 translate-y-0.5 group-hover:opacity-100 group-hover:translate-y-0 focus-within:opacity-100 focus-within:translate-y-0 transition-all duration-300">
        {/* View */}
        <button onClick={() => zoomBy(-0.5)} disabled={zoom <= 1}
                className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button text-sm font-bold disabled:opacity-30" title="Zoom out (−)">−</button>
        <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
                className={`px-2 h-7 rounded-lg text-[10px] font-bold font-mono tabular-nums ${zoom > 1 ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'}`}
                title="Reset to fit">
          {zoom > 1 ? `${zoom.toFixed(1)}×` : 'FIT'}
        </button>
        <button onClick={() => zoomBy(0.5)} disabled={zoom >= ZOOM_MAX}
                className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button text-sm font-bold disabled:opacity-30" title="Zoom in (+)">+</button>
        <button onClick={zoomToActual} className="px-2 h-7 rounded-lg text-[10px] font-bold hud-glass-button" title="Zoom to actual pixels (1 source pixel = 1 screen pixel)">1:1</button>

        <span className="w-px h-5 bg-white/10 mx-1.5" />

        {/* Overlays */}
        <button onClick={() => setShowBoxes(b => !b)} title="Show detected face boxes"
                className={`px-2 h-7 rounded-lg text-[10px] font-bold ${showBoxes ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'}`}>
          Faces{faces.length > 0 && <span className="ml-1 opacity-60 tabular-nums">{faces.length}</span>}
        </button>
        {onToggleCompare && (
          <button onClick={() => onToggleCompare()} title="Toggle before/after compare (C)"
                  className={`px-2 h-7 rounded-lg text-[10px] font-bold ${compare ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'}`}>Compare</button>
        )}

        {isVideo && (
          <>
            <span className="w-px h-5 bg-white/10 mx-1.5" />
            <button onClick={() => stepFrame(-1)} className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button" title="Previous frame (←)">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zM20 6v12l-9-6z"/></svg>
            </button>
            <button onClick={() => setIsPlaying && setIsPlaying(p => !p)}
                    className={`grid place-items-center h-7 w-7 rounded-lg ${isPlaying ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'}`}
                    title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}>
              {isPlaying
                ? <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>
                : <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>}
            </button>
            <button onClick={() => stepFrame(1)} className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button" title="Next frame (→)">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M16 6h2v12h-2zM4 6l9 6-9 6z"/></svg>
            </button>
            <span className="px-1.5 text-[10px] font-bold font-mono text-white/55 tabular-nums whitespace-nowrap">
              {frame.toLocaleString()}<span className="opacity-40">/{maxFrames.toLocaleString()}</span>
            </span>
          </>
        )}

        <span className="w-px h-5 bg-white/10 mx-1.5" />
        <button onClick={triggerFullscreen} className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button" title="Toggle fullscreen">
          {isFullscreen
            ? <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/></svg>
            : <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>}
        </button>
      </div>
    </div>
  );

  // Small always-on status cluster, top-left: what you are looking at and at
  // what magnification. Pure read-out, so it stays out of the way at low
  // opacity until the pointer enters the stage.
  const stageInfo = () => (
    <div className="absolute top-3 left-3 z-40 flex items-center gap-1.5 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300">
      {imgDim && (
        <span className="rounded-md bg-black/65 backdrop-blur px-2 py-1 font-mono text-[10px] tabular-nums text-white/65 border border-white/10">
          {imgDim.w}×{imgDim.h}
        </span>
      )}
      {zoom > 1 && (
        <span className="rounded-md bg-[var(--accent)]/80 backdrop-blur px-2 py-1 font-mono text-[10px] font-bold tabular-nums text-white">
          {zoom.toFixed(1)}×
        </span>
      )}
    </div>
  );

  // Split View comparisons
  if (compare && splitView) {
    return (
      <div
        ref={containerRef}
        className={`relative w-full aspect-video max-h-[54vh] min-h-[260px] rounded-2xl overflow-hidden preview-stage group ${isFullscreen ? 'h-screen w-screen max-h-none' : ''}`}
        onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onDoubleClick={handleDoubleClick} style={{ touchAction: 'none' }}
      >
        {stageInfo()}
        <div className={`flex w-full h-full ${interacting ? '' : 'transition-transform duration-75'}`} style={transformStyle}>
          <div className="flex-1 relative border-r border-white/10 flex items-center justify-center overflow-hidden bg-black/50">
            <div className="relative" style={aspectStyle}>
              <img src={beforeSrc} alt="Before" className="w-full h-full object-contain pointer-events-none"
                   onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })} />
              <div className="absolute inset-0 pointer-events-none">{faceBoxes}</div>
            </div>
            <span className="absolute bottom-3 left-3 px-2 py-1 rounded-md bg-black/65 backdrop-blur text-[10px] font-semibold uppercase tracking-[0.14em] text-white/70">Before</span>
          </div>
          <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-black/50">
            <div className="relative" style={aspectStyle}>
              <img src={afterSrc} alt="After" className="w-full h-full object-contain pointer-events-none" />
            </div>
            <span className="absolute bottom-3 left-3 px-2 py-1 rounded-md bg-[var(--accent)]/85 backdrop-blur text-[10px] font-semibold uppercase tracking-[0.14em] text-white">After</span>
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
      className={`relative w-full aspect-video max-h-[54vh] min-h-[260px] rounded-2xl overflow-hidden preview-stage select-none group ${isFullscreen ? 'h-screen w-screen max-h-none' : ''}`}
      ref={containerRef} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onDoubleClick={handleDoubleClick} style={{ touchAction: 'none' }}
    >
      {/* Render indicator: an indeterminate bar along the top edge plus a single
          scanner sweep. Replaces the pulsing glow that used to breathe around
          the whole panel — the border is chrome, not a status light. */}
      {previewing && (
        <div className="absolute inset-x-0 top-0 h-[3px] z-50 overflow-hidden bg-white/[0.06]">
          <div className="h-full w-1/3 rounded-full bg-[var(--accent)] preview-indeterminate" />
        </div>
      )}
      {previewing && <AIScannerOverlay />}
      {previewing && (
        <div className="absolute top-4 right-3 z-50">
          <div className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-black/70 backdrop-blur-md text-[11px] font-semibold text-white/90 tabular-nums border border-white/10">
            <span className="h-3 w-3 rounded-full border-2 border-white/25 border-t-[var(--accent)] animate-spin" />
            Rendering {previewSecs}s
          </div>
        </div>
      )}

      {stageInfo()}

      {/* Frame navigation shortcuts guide (video only) */}
      {maxFrames > 1 && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 px-2.5 py-1 rounded-lg bg-black/55 backdrop-blur text-[10px] font-semibold text-white/45 border border-white/5 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-30 flex items-center gap-1.5 whitespace-nowrap">
          <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">←/→</span>
          <span>frame</span>
          <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">⇧</span>
          <span>×10</span>
          <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">[ ]</span>
          <span>in/out</span>
          <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">M</span>
          <span>marker</span>
          <span className="bg-white/10 px-1.5 py-0.5 rounded text-white/80 font-mono">Space</span>
          <span>play</span>
          <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">2×click</span>
          <span>zoom</span>
        </div>
      )}

      <div className={`absolute inset-0 flex items-center justify-center ${interacting ? '' : 'transition-transform duration-75'}`} style={transformStyle}>

        {/* Before Image & Bounding Boxes Wrapper.

            Both frames use a single PERSISTENT <img> each (no React `key`, so the
            element is never remounted on a frame change). Changing an <img>'s src
            makes the browser keep the previously decoded frame painted until the
            new one is ready, then swap atomically — so clicking to a new frame
            never flashes the black container through while the next frame decodes.
            (Remounting via `key` mounts a fresh, empty <img> that paints blank
            until load — that blank was the black flicker.) */}
        <div className="relative z-10" style={aspectStyle} ref={imageRef}>
          <img
            src={beforeSrc}
            alt="Before"
            className="relative z-[1] w-full h-full object-contain pointer-events-none"
            onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
            draggable={false}
          />
          <div className="absolute inset-0 pointer-events-none z-30">{faceBoxes}</div>

          {/* Swapped "after" overlay (clip-path only in compare mode). Cross-faded
              between frames via two persistent layers (see CrossfadeImage) — smooth
              transition on frame/preview changes with no black flash. Falls back to
              the before frame when no swap is available so it never blanks. Snap the
              fade short while actively scrubbing so the image doesn't trail the
              playhead; ease gently on discrete jumps / preview refreshes. */}
          <div className="absolute inset-0 pointer-events-none z-20"
               style={{ clipPath: compare ? `polygon(${sliderPosition}% 0, 100% 0, 100% 100%, ${sliderPosition}% 100%)` : 'none' }}>
            <CrossfadeImage
              src={afterSrc || beforeSrc}
              fadeMs={isPlaying ? 0 : (scrubbing ? 60 : 200)}
              className="absolute inset-0 w-full h-full object-contain"
            />
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

      {/* Provenance chips sit along the BOTTOM edge: the top corners now belong
          to the render indicator and the stage read-out, and a label that
          overlaps the status you are reading is worse than no label. */}
      {compare && (
        <span className="absolute bottom-3 left-3 px-2 py-1 rounded-md bg-black/65 backdrop-blur text-[10px] font-semibold uppercase tracking-[0.14em] text-white/70 z-30 pointer-events-none">
          Before
        </span>
      )}
      <span className="absolute bottom-3 right-3 px-2 py-1 rounded-md bg-[var(--accent)]/85 backdrop-blur text-[10px] font-semibold uppercase tracking-[0.14em] text-white z-30 transition-opacity duration-300 pointer-events-none"
            style={{ opacity: compare && sliderPosition > 85 ? 0 : 1 }}>{compare ? 'After' : 'Swapped'}</span>

      {/* HUD control bar overlays */}
      {hudBar()}
    </div>
  );
}
