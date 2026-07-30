import React, { useEffect, useRef, useState, useMemo } from 'react';
import AIScannerOverlay from './AIScannerOverlay';

// Cross-fades between src changes using TWO persistent <img> layers that are
// never remounted.
function CrossfadeImage({ src, className, style, fadeMs = 200, onLoad }) {
  const [layers, setLayers] = useState({ a: src, b: src, front: 'a' });

  useEffect(() => {
    setLayers((s) => {
      if (src === s[s.front]) return s;
      const back = s.front === 'a' ? 'b' : 'a';
      // The back layer is ALREADY showing this exact image — stepping back onto
      // a frame whose render is still cached puts the previous src back. An
      // <img> whose src is re-assigned to what it already holds fires no load
      // event, so the promote below would never run: the stage stayed on the
      // other frame while the playhead said otherwise, and stepping between two
      // frames looked like the picture was flicking back and forth at random.
      // It is loaded, so promote it here instead of waiting for a load.
      if (src === s[back]) return { ...s, front: back };
      return { ...s, [back]: src };
    });
  }, [src]);

  const promote = (which, e) => {
    if (onLoad) onLoad(e);
    setLayers((s) => {
      if (s.front === which) return s;
      if (s[which] !== src) return s;
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
      style={{
        ...style,
        opacity: layers.front === which ? 1 : 0,
        transition: `opacity ${fadeMs}ms ease-out`,
      }}
    />
  );

  return (
    <>
      {renderLayer('a')}
      {renderLayer('b')}
    </>
  );
}

const ZOOM_MAX = 8;
const LENS_R = 88;        // lens radius in px (the glass is 2R across)
const LENS_ZOOM = 3.5;    // magnification ON TOP of the stage's own zoom

export default function InteractivePreview({
  beforeSrc,
  afterSrc,
  faces = [],
  personIds = [],
  onSelectPerson,
  splitView = false,
  compare = false,
  onToggleCompare,
  sliderEffectEnabled = true,
  onToggleSliderEffect,
  frame = 1,
  setFrame,
  maxFrames = 1,
  isPlaying = false,
  setIsPlaying,
  previewing = false,
  previewSecs = 0,
  scrubbing = false,
  onMaskChange,
  maskApplied = false,
}) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [compareMode, setCompareMode] = useState('slider'); // 'slider' | 'blend' | 'diff'
  const [compareDir, setCompareDir] = useState('vertical'); // 'vertical' | 'horizontal'
  const [autoSwipe, setAutoSwipe] = useState(false);
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const maskExportRef = useRef(null);
  const lastPtRef = useRef(null);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [showBoxes, setShowBoxes] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isPeekingOriginal, setIsPeekingOriginal] = useState(false);

  // Feature 1: Interactive Magnifier Glass Lens State
  const [magnifierActive, setMagnifierActive] = useState(false);
  const [lensPos, setLensPos] = useState({ x: 0, y: 0, relX: 0, relY: 0, visible: false });

  // Feature 2: Interactive Face Mask Brush Tool State
  const [maskBrushActive, setMaskBrushActive] = useState(false);
  const [brushMode, setBrushMode] = useState('paint'); // 'paint' | 'erase'
  const [brushSize, setBrushSize] = useState(30);
  const [isDrawingMask, setIsDrawingMask] = useState(false);

  const [imgDim, setImgDim] = useState(null);

  const handleSliderMove = (clientX, clientY) => {
    const target = imageRef.current || containerRef.current;
    if (!target) return;
    const rect = target.getBoundingClientRect();
    if (compareDir === 'horizontal' && clientY !== undefined) {
      const y = Math.max(0, Math.min(clientY - rect.top, rect.height));
      const percent = Math.max(0, Math.min((y / rect.height) * 100, 100));
      setSliderPosition(percent);
    } else if (clientX !== undefined) {
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      const percent = Math.max(0, Math.min((x / rect.width) * 100, 100));
      setSliderPosition(percent);
    }
  };

  // Auto-swipe animation.
  //
  // Runs in both slider and blend mode — sliderPosition drives the wipe in one
  // and the crossfade in the other, and the Auto button is offered in both, so
  // gating the loop to 'slider' alone left the button lit and doing nothing.
  // Only 'diff' has nothing for it to move, and there the button is hidden.
  //
  // Suspended whenever the page is hidden or a render is in flight: this ticks
  // React state 60x a second, and the swap already wants every frame of GPU it
  // can get (see the render-lite mode in FaceSwap/index.css).
  const autoSwipeRuns = autoSwipe && compare && compareMode !== 'diff';
  useEffect(() => {
    if (!autoSwipeRuns) return;
    let animId;
    let startTime;
    const animate = (time) => {
      if (document.hidden
          || document.documentElement.hasAttribute('data-render-lite')) {
        startTime = undefined;          // resume from the current phase, no jump
        animId = requestAnimationFrame(animate);
        return;
      }
      if (!startTime) startTime = time;
      const elapsed = (time - startTime) / 1000;
      const pos = 50 + 40 * Math.sin(elapsed * (Math.PI * 2 / 2.5));
      setSliderPosition(pos);
      animId = requestAnimationFrame(animate);
    };
    animId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animId);
  }, [autoSwipeRuns]);

  useEffect(() => {
    const handleMouseUp = () => {
      setIsDraggingSlider(false);
      setIsPanning(false);
      setIsPeekingOriginal(false);
      setIsDrawingMask(false);
      lastPtRef.current = null;   // next press starts a new stroke
    };
    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('touchend', handleMouseUp);
    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, []);

  useEffect(() => {
    const onFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
      // Bare keys only. Without this, Ctrl/Cmd+C hit the 'c' branch below and
      // its preventDefault() swallowed the copy — the same for Ctrl+M/Ctrl+B.
      if (e.ctrlKey || e.metaKey || e.altKey) return;

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
      } else if (e.key.toLowerCase() === 'c' && onToggleCompare) {
        e.preventDefault();
        onToggleCompare();
      } else if (e.key.toLowerCase() === 'm') {
        setMagnifierActive((m) => !m);
      } else if (e.key.toLowerCase() === 'b') {
        setMaskBrushActive((b) => !b);
      } else if (e.key.toLowerCase() === 'h' && compare) {
        setCompareDir((d) => (d === 'vertical' ? 'horizontal' : 'vertical'));
      } else if (e.key.toLowerCase() === 'o' && compare) {
        setCompareMode((m) => (m === 'blend' ? 'slider' : 'blend'));
      } else if (e.key.toLowerCase() === 'a' && compare && compareMode !== 'diff') {
        // Same condition the Auto button renders under — a shortcut that
        // silently flips hidden state is worse than one that does nothing.
        setAutoSwipe((a) => !a);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onToggleCompare, compare, compareMode]);

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
    if (maskBrushActive) {
      setIsDrawingMask(true);
      drawMaskStroke(e, true);   // start a fresh stroke, no join to the last one
      return;
    }
    if (zoom > 1 && !isDraggingSlider) {
      setIsPanning(true);
      setStartPan({
        x: (e.clientX || e.touches?.[0]?.clientX) - pan.x,
        y: (e.clientY || e.touches?.[0]?.clientY) - pan.y,
      });
    }
  };

  const rafRef = useRef(null);
  const pendingRef = useRef(null);
  useEffect(() => () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); }, []);

  const handlePointerMove = (e) => {
    const cx = e.clientX ?? e.touches?.[0]?.clientX;
    const cy = e.clientY ?? e.touches?.[0]?.clientY;
    if (cx === undefined) return;

    // Update Magnifier glass position.
    // We record the DISPLAYED image box (imageRef, so the rect already includes
    // the zoom/pan transform) in container-local coordinates. The lens then
    // reproduces exactly that layout scaled by LENS_ZOOM and offsets it so the
    // pixel under the cursor lands dead centre. The previous version scaled a
    // copy of the image squeezed into the 176px SQUARE lens and used a
    // transform-origin taken from the 16:9 container, so the glass showed a
    // letterboxed, differently-proportioned region that did not correspond to
    // what was under the pointer.
    if (magnifierActive && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const ir = imageRef.current?.getBoundingClientRect();
      setLensPos({
        x: cx - rect.left,
        y: cy - rect.top,
        imgW: ir?.width || rect.width,
        imgH: ir?.height || rect.height,
        imgX: (ir?.left ?? rect.left) - rect.left,
        imgY: (ir?.top ?? rect.top) - rect.top,
        visible: cx >= rect.left && cx <= rect.right && cy >= rect.top && cy <= rect.bottom,
      });
    }

    // Mask Brush drawing
    if (maskBrushActive && isDrawingMask) {
      drawMaskStroke(e);
      return;
    }

    if (!isDraggingSlider && !(isPanning && zoom > 1)) return;
    pendingRef.current = { cx, cy };
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const p = pendingRef.current;
      if (!p) return;
      if (isDraggingSlider) {
        handleSliderMove(p.cx, p.cy);
      } else if (isPanning && zoom > 1) {
        setPan({ x: p.cx - startPan.x, y: p.cy - startPan.y });
      }
    });
  };

  // Every stroke is painted onto TWO canvases with identical geometry:
  //   maskCanvasRef  — visible, translucent accent, purely a viewing aid
  //   maskExportRef  — offscreen, solid white on transparent, what gets sent
  // The backend decodes the mask with cv2.IMREAD_GRAYSCALE, so exporting the
  // translucent pink layer would land at grey ~121 and be read as a HALF-
  // strength mask everywhere the user painted. White gives a clean 0/255.
  const drawMaskStroke = (e, isStart = false) => {
    const canvas = maskCanvasRef.current;
    const exportCanvas = maskExportRef.current;
    if (!canvas || !exportCanvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX ?? e.touches?.[0]?.clientX;
    const cy = e.clientY ?? e.touches?.[0]?.clientY;
    if (cx === undefined || !rect.width) return;

    // brushSize is a SCREEN measurement; the canvas is at the media's native
    // resolution, so it has to be converted or the stroke is the wrong size by
    // exactly the display scale factor (e.g. 2.4x off for 1920px media in an
    // 800px box). The rect already includes the stage zoom/pan transform.
    const scale = canvas.width / rect.width;
    const x = (cx - rect.left) * scale;
    const y = (cy - rect.top) * scale;
    const r = Math.max(0.5, (brushSize / 2) * scale);

    // Join to the previous sample, otherwise a quick drag leaves a dotted trail
    // of disconnected circles instead of a stroke.
    const prev = isStart ? null : lastPtRef.current;
    const erase = brushMode === 'erase';

    for (const [c, color] of [[canvas, 'rgba(233, 69, 96, 0.45)'], [exportCanvas, '#ffffff']]) {
      const ctx = c.getContext('2d');
      if (!ctx) continue;
      ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over';
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = r * 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      if (prev) {
        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    lastPtRef.current = { x, y };
  };

  // Hand the painted region to the parent as a PNG data URL (null when nothing
  // is painted). Explicit rather than per-stroke: each commit re-renders the
  // preview through the backend.
  const commitMask = () => {
    if (!onMaskChange) return;
    const c = maskExportRef.current;
    if (!c) return;
    let painted = false;
    try {
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      // Step 4 pixels; the smallest brush covers far more than that.
      for (let i = 3; i < d.length; i += 16) {
        if (d[i] > 8) { painted = true; break; }
      }
    } catch {
      painted = true;   // tainted canvas shouldn't silently drop the mask
    }
    onMaskChange(painted ? c.toDataURL('image/png') : null);
  };

  const clearMaskCanvas = () => {
    for (const c of [maskCanvasRef.current, maskExportRef.current]) {
      const ctx = c?.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, c.width, c.height);
    }
    lastPtRef.current = null;
    if (onMaskChange) onMaskChange(null);
  };

  const handleDoubleClick = (e) => {
    if (zoom > 1) {
      setZoom(1);
      setPan({ x: 0, y: 0 });
    } else {
      setZoom(2.5);
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const clickX = (e.clientX ?? e.touches?.[0]?.clientX) - rect.left;
        const clickY = (e.clientY ?? e.touches?.[0]?.clientY) - rect.top;
        setPan({
          x: (rect.width / 2 - clickX) * 1.5,
          y: (rect.height / 2 - clickY) * 1.5,
        });
      }
    }
  };

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
          className={`absolute border-2 border-[var(--accent)] shadow-[0_0_10px_var(--accent-glow)] z-20 ${
            clickable
              ? 'pointer-events-auto cursor-pointer group/face hover:bg-[var(--accent)]/10 transition-colors'
              : 'pointer-events-none'
          }`}
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

  const zoomToActual = () => {
    const box = (imageRef.current || containerRef.current)?.getBoundingClientRect();
    if (!box || !imgDim || !box.width) return;
    setZoom(Math.min(Math.max(imgDim.w / box.width, 1), ZOOM_MAX));
    setPan({ x: 0, y: 0 });
  };

  const zoomBy = (d) =>
    setZoom((z) => {
      const nz = Math.min(Math.max(1, z + d), ZOOM_MAX);
      if (nz === 1) setPan({ x: 0, y: 0 });
      return nz;
    });

  // The 50/50 split branch returns early and renders neither the lens overlay
  // nor the mask canvas, so offering their buttons there just lit up an "active"
  // badge for a tool that could not do anything.
  const splitMode = compare && splitView;

  // Main HUD Control Bar inside Preview Box
  const hudBar = () => (
    <div className="absolute inset-x-0 bottom-0 z-50 flex justify-center p-3 pointer-events-none">
      <div className="spring-cluster pointer-events-auto flex items-center gap-0.5 rounded-xl hud-glass p-1 opacity-70 translate-y-0.5 group-hover:opacity-100 group-hover:translate-y-0 focus-within:opacity-100 focus-within:translate-y-0 transition-all duration-300 shadow-2xl border border-white/10">
        {/* View Zoom controls */}
        <button
          onClick={() => zoomBy(-0.5)}
          disabled={zoom <= 1}
          className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button text-sm font-bold disabled:opacity-30"
          title="Zoom out (−)"
        >
          −
        </button>
        <button
          onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
          className={`px-2 h-7 rounded-lg text-[10px] font-bold font-mono tabular-nums ${
            zoom > 1
              ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20'
              : 'hud-glass-button'
          }`}
          title="Reset to fit"
        >
          {zoom > 1 ? `${zoom.toFixed(1)}×` : 'FIT'}
        </button>
        <button
          onClick={() => zoomBy(0.5)}
          disabled={zoom >= ZOOM_MAX}
          className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button text-sm font-bold disabled:opacity-30"
          title="Zoom in (+)"
        >
          +
        </button>
        <button
          onClick={zoomToActual}
          className="px-2 h-7 rounded-lg text-[10px] font-bold hud-glass-button"
          title="Zoom 1:1"
        >
          1:1
        </button>

        <span className="w-px h-5 bg-white/10 mx-1" />

        {/* Faces Overlays Toggle */}
        <button
          onClick={() => setShowBoxes((b) => !b)}
          title="Show detected face boxes"
          className={`px-2 h-7 rounded-lg text-[10px] font-bold ${
            showBoxes
              ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20'
              : 'hud-glass-button'
          }`}
        >
          Faces
          {faces.length > 0 && <span className="ml-1 opacity-60 tabular-nums">{faces.length}</span>}
        </button>

        {/* COMPARE SWAP FACES Button inside Preview Box */}
        {onToggleCompare && (
          <button
            onClick={() => onToggleCompare()}
            title="Toggle Compare Swap Faces (C)"
            className={`px-2.5 h-7 rounded-lg text-[10px] font-bold flex items-center gap-1.5 transition-all duration-200 ${
              compare
                ? 'text-white bg-[var(--accent)] shadow-[0_0_12px_var(--accent-glow)] border border-white/20'
                : 'hud-glass-button text-white/80 hover:text-white'
            }`}
          >
            <span>🎭 Compare Faces</span>
            {compare && <span className="text-[9px] bg-black/40 px-1 rounded font-mono">{Math.round(sliderPosition)}%</span>}
          </button>
        )}

        {/* Peek Original Button */}
        <button
          onMouseDown={() => setIsPeekingOriginal(true)}
          onMouseUp={() => setIsPeekingOriginal(false)}
          onMouseLeave={() => setIsPeekingOriginal(false)}
          onTouchStart={() => setIsPeekingOriginal(true)}
          onTouchEnd={() => setIsPeekingOriginal(false)}
          title="Press and hold to view original target image"
          className={`px-2 h-7 rounded-lg text-[10px] font-bold transition-all duration-200 select-none ${
            isPeekingOriginal
              ? 'bg-amber-500 text-black border border-amber-300 font-extrabold shadow-[0_0_10px_rgba(245,158,11,0.5)]'
              : 'hud-glass-button text-white/70'
          }`}
        >
          👁️ Hold Peek
        </button>

        <span className="w-px h-5 bg-white/10 mx-1" />

        {/* Feature 1: Magnifier Lens Button */}
        {!splitMode && (
        <button
          onClick={() => setMagnifierActive((m) => !m)}
          title="Toggle interactive 3.5× Magnifier Glass Lens (M)"
          className={`px-2 h-7 rounded-lg text-[10px] font-bold flex items-center gap-1 transition-colors ${
            magnifierActive
              ? 'text-cyan-300 bg-cyan-500/20 border border-cyan-400/40 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
              : 'hud-glass-button text-white/70'
          }`}
        >
          🔍 Lens
        </button>
        )}

        {/* Feature 2: Face Mask Brush Tool Button */}
        {!splitMode && (
        <button
          onClick={() => setMaskBrushActive((b) => !b)}
          title="Toggle interactive Face Mask Brush tool (B)"
          className={`px-2 h-7 rounded-lg text-[10px] font-bold flex items-center gap-1 transition-colors ${
            maskBrushActive
              ? 'text-pink-300 bg-pink-500/20 border border-pink-400/40 shadow-[0_0_10px_rgba(236,72,153,0.3)]'
              : 'hud-glass-button text-white/70'
          }`}
        >
          🖌️ Mask Brush
        </button>
        )}

        {/* Slider Effect Toggle Button inside Preview Box HUD */}
        {onToggleSliderEffect && (
          <button
            onClick={() => onToggleSliderEffect()}
            title="Toggle Slider Effects ON/OFF"
            className={`px-2 h-7 rounded-lg text-[10px] font-bold flex items-center gap-1 transition-colors ${
              sliderEffectEnabled
                ? 'text-emerald-400 bg-emerald-500/15 border border-emerald-500/30'
                : 'text-amber-300 bg-amber-500/15 border border-amber-500/30'
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${sliderEffectEnabled ? 'bg-emerald-400' : 'bg-amber-400'}`} />
            <span>Effect {sliderEffectEnabled ? 'ON' : 'OFF'}</span>
          </button>
        )}

        {/* Video Frame controls */}
        {isVideo && (
          <>
            <span className="w-px h-5 bg-white/10 mx-1" />
            <button onClick={() => stepFrame(-1)} className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button" title="Previous frame (←)">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zM20 6v12l-9-6z"/></svg>
            </button>
            <button
              onClick={() => setIsPlaying && setIsPlaying((p) => !p)}
              className={`grid place-items-center h-7 w-7 rounded-lg ${
                isPlaying ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'
              }`}
              title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
            >
              {isPlaying ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
              )}
            </button>
            <button onClick={() => stepFrame(1)} className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button" title="Next frame (→)">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M16 6h2v12h-2zM4 6l9 6-9 6z"/></svg>
            </button>
            <span className="px-1 text-[10px] font-bold font-mono text-white/55 tabular-nums whitespace-nowrap">
              {frame.toLocaleString()}<span className="opacity-40">/{maxFrames.toLocaleString()}</span>
            </span>
          </>
        )}

        <span className="w-px h-5 bg-white/10 mx-1" />

        {/* Fullscreen Button */}
        <button onClick={triggerFullscreen} className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button" title="Toggle fullscreen">
          {isFullscreen ? (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/></svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
          )}
        </button>
      </div>
    </div>
  );

  // Top info overlay
  const stageInfo = () => (
    <div className="absolute top-3 left-3 z-40 flex items-center gap-2 pointer-events-none">
      {imgDim && (
        <span className="rounded-lg bg-black/75 backdrop-blur-md px-2.5 py-1 font-mono text-[10px] tabular-nums text-white/80 border border-white/10 shadow-lg">
          {imgDim.w}×{imgDim.h}
        </span>
      )}
      {zoom > 1 && (
        <span className="rounded-lg bg-[var(--accent)]/90 backdrop-blur-md px-2.5 py-1 font-mono text-[10px] font-bold tabular-nums text-white shadow-lg">
          {zoom.toFixed(1)}×
        </span>
      )}
      {compare && (
        <span className="rounded-lg bg-[var(--accent)] text-white backdrop-blur-md px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider shadow-lg border border-white/20 animate-pulse">
          Compare Active
        </span>
      )}
      {magnifierActive && !splitMode && (
        <span className="rounded-lg bg-cyan-500/20 text-cyan-300 backdrop-blur-md px-2 py-1 text-[9px] font-bold uppercase tracking-wider border border-cyan-400/30">
          🔍 Magnifier Lens (3.5×)
        </span>
      )}
      {maskBrushActive && !splitMode && (
        <span className="rounded-lg bg-pink-500/20 text-pink-300 backdrop-blur-md px-2 py-1 text-[9px] font-bold uppercase tracking-wider border border-pink-400/30">
          🖌️ Mask Brush Active ({brushMode})
        </span>
      )}
    </div>
  );

  // Split View comparison (50/50 dual pane)
  if (compare && splitView) {
    return (
      <div
        ref={containerRef}
        className={`relative w-full aspect-video max-h-[54vh] min-h-[260px] rounded-2xl overflow-hidden preview-stage group border border-white/10 shadow-2xl ${
          isFullscreen ? 'h-screen w-screen max-h-none' : ''
        }`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onDoubleClick={handleDoubleClick}
        style={{ touchAction: 'none' }}
      >
        {stageInfo()}
        <div className={`flex w-full h-full ${interacting ? '' : 'transition-transform duration-75'}`} style={transformStyle}>
          <div className="flex-1 relative border-r border-white/10 flex items-center justify-center overflow-hidden bg-black/60">
            <div className="relative" style={aspectStyle}>
              <img
                src={beforeSrc}
                alt="Before"
                className="w-full h-full object-contain pointer-events-none"
                onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
              />
              <div className="absolute inset-0 pointer-events-none">{faceBoxes}</div>
            </div>
            <span className="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg bg-black/75 backdrop-blur-md text-[10px] font-bold uppercase tracking-[0.14em] text-white/80 border border-white/10">
              Before (Target Original)
            </span>
          </div>
          <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-black/60">
            <div className="relative" style={aspectStyle}>
              <img
                src={isPeekingOriginal ? beforeSrc : afterSrc}
                alt="After"
                className="w-full h-full object-contain pointer-events-none"
              />
            </div>
            <span className="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg bg-[var(--accent)]/90 backdrop-blur-md text-[10px] font-bold uppercase tracking-[0.14em] text-white border border-white/20 shadow-lg">
              {isPeekingOriginal ? 'Original (Peek)' : 'After (Swapped)'}
            </span>
          </div>
        </div>

        {hudBar()}
      </div>
    );
  }

  // Standard or Slide-Comparison View (Vertical slider handle)
  const currentClipPosition = isPeekingOriginal ? 0 : compare ? sliderPosition : 100;

  return (
    <div
      className={`relative w-full aspect-video max-h-[54vh] min-h-[260px] rounded-2xl overflow-hidden preview-stage select-none group border border-white/10 shadow-2xl ${
        isFullscreen ? 'h-screen w-screen max-h-none' : ''
      }`}
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onDoubleClick={handleDoubleClick}
      style={{ touchAction: 'none' }}
    >
      {/* Render indicator */}
      {previewing && (
        <div className="absolute inset-x-0 top-0 h-[3px] z-50 overflow-hidden bg-white/[0.06]">
          <div className="h-full w-1/3 rounded-full bg-[var(--accent)] preview-indeterminate" />
        </div>
      )}
      {previewing && <AIScannerOverlay />}
      {previewing && (
        <div className="absolute top-4 right-3 z-50">
          <div className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-black/80 backdrop-blur-md text-[11px] font-semibold text-white/90 tabular-nums border border-white/10 shadow-2xl">
            <span className="h-3 w-3 rounded-full border-2 border-white/25 border-t-[var(--accent)] animate-spin" />
            Rendering {previewSecs}s
          </div>
        </div>
      )}

      {stageInfo()}

      {/* Compare Mode Toolbar Controls Overlay */}
      {compare && !splitMode && !maskBrushActive && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-black/85 backdrop-blur-md border border-white/15 shadow-2xl animate-fade-in">
          <button
            type="button"
            onClick={() => setCompareMode('slider')}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all ${
              compareMode === 'slider' ? 'bg-[var(--accent)] text-white shadow-md' : 'bg-white/10 text-white/70 hover:text-white'
            }`}
            title="Split Slider mode"
          >
            ✂️ Split
          </button>
          <button
            type="button"
            onClick={() => setCompareMode('blend')}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all ${
              compareMode === 'blend' ? 'bg-[var(--accent)] text-white shadow-md' : 'bg-white/10 text-white/70 hover:text-white'
            }`}
            title="Opacity Blend mode (O)"
          >
            🌓 Blend
          </button>
          <button
            type="button"
            onClick={() => setCompareMode('diff')}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all ${
              compareMode === 'diff' ? 'bg-[var(--accent)] text-white shadow-md' : 'bg-white/10 text-white/70 hover:text-white'
            }`}
            title="Difference Map mode"
          >
            ⚡ Diff
          </button>

          {compareMode === 'slider' && (
            <>
              <span className="w-px h-4 bg-white/15 mx-0.5" />
              <button
                type="button"
                onClick={() => setCompareDir((d) => (d === 'vertical' ? 'horizontal' : 'vertical'))}
                className="px-2 py-1 rounded-lg text-[10px] font-bold bg-white/10 hover:bg-white/20 text-white/80 transition-all flex items-center gap-1"
                title="Toggle Split Direction: Vertical vs Horizontal (H)"
              >
                {compareDir === 'vertical' ? '↔ Vert' : '↕ Horiz'}
              </button>
            </>
          )}

          {compareMode !== 'diff' && (
            <>
              <span className="w-px h-4 bg-white/15 mx-0.5" />
              <button
                type="button"
                onClick={() => setSliderPosition(25)}
                className={`px-1.5 py-1 rounded text-[9px] font-mono font-bold ${
                  sliderPosition === 25 ? 'bg-white/20 text-white' : 'text-white/50 hover:text-white'
                }`}
              >
                25%
              </button>
              <button
                type="button"
                onClick={() => setSliderPosition(50)}
                className={`px-1.5 py-1 rounded text-[9px] font-mono font-bold ${
                  sliderPosition === 50 ? 'bg-white/20 text-white' : 'text-white/50 hover:text-white'
                }`}
              >
                50%
              </button>
              <button
                type="button"
                onClick={() => setSliderPosition(75)}
                className={`px-1.5 py-1 rounded text-[9px] font-mono font-bold ${
                  sliderPosition === 75 ? 'bg-white/20 text-white' : 'text-white/50 hover:text-white'
                }`}
              >
                75%
              </button>
              <span className="w-px h-4 bg-white/15 mx-0.5" />
              <button
                type="button"
                onClick={() => setAutoSwipe((a) => !a)}
                className={`px-2 py-1 rounded-lg text-[10px] font-bold transition-all flex items-center gap-1 ${
                  autoSwipe ? 'bg-emerald-500 text-black shadow-md font-extrabold animate-pulse' : 'bg-white/10 text-white/70 hover:text-white'
                }`}
                title="Toggle Auto-Swipe oscillation loop (A)"
              >
                {autoSwipe ? '⏸️ Auto' : '▶️ Auto'}
              </button>
            </>
          )}
        </div>
      )}

      {/* Mask Brush Toolbar Controls Overlay */}
      {maskBrushActive && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-black/85 backdrop-blur-md border border-white/15 shadow-2xl">
          <button
            type="button"
            onClick={() => setBrushMode('paint')}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-colors ${
              brushMode === 'paint' ? 'bg-pink-500 text-white' : 'bg-white/10 text-white/70'
            }`}
          >
            🖌️ Paint
          </button>
          <button
            type="button"
            onClick={() => setBrushMode('erase')}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-colors ${
              brushMode === 'erase' ? 'bg-amber-500 text-black' : 'bg-white/10 text-white/70'
            }`}
          >
            🧹 Erase
          </button>
          <div className="flex items-center gap-1.5 px-2 border-l border-white/10">
            <span className="text-[9px] font-mono text-white/50 uppercase font-bold">Size:</span>
            <input
              type="range"
              min={10}
              max={80}
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value))}
              className="w-16 h-1 accent-pink-500 cursor-pointer"
            />
            <span className="text-[10px] font-mono font-bold text-white/80 tabular-nums">{brushSize}px</span>
          </div>
          <button
            type="button"
            onClick={clearMaskCanvas}
            className="px-2 py-1 rounded-lg text-[10px] font-bold bg-white/10 hover:bg-white/20 text-white/70 hover:text-white transition-colors"
          >
            ↺ Clear
          </button>
          {onMaskChange && (
            <button
              type="button"
              onClick={commitMask}
              className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-colors ${
                maskApplied
                  ? 'bg-emerald-500 text-black'
                  : 'bg-[var(--accent)] text-white hover:brightness-110'
              }`}
              title="Send the painted region to the swap — painted areas keep the ORIGINAL face"
            >
              {maskApplied ? '✓ Mask Applied' : '✅ Apply Mask'}
            </button>
          )}
        </div>
      )}

      {/* Frame shortcuts guide */}
      {maxFrames > 1 && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 px-2.5 py-1 rounded-lg bg-black/65 backdrop-blur-md text-[10px] font-semibold text-white/50 border border-white/10 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-30 flex items-center gap-1.5 whitespace-nowrap">
          <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">←/→</span>
          <span>frame</span>
          <span className="bg-white/10 px-1.5 py-0.5 rounded text-white/80 font-mono">Space</span>
          <span>play</span>
          <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">M</span>
          <span>lens</span>
          <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">B</span>
          <span>brush</span>
          {compare && (
            <>
              <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">O</span>
              <span>blend</span>
              <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">H</span>
              <span>axis</span>
              <span className="bg-white/10 px-1 py-0.5 rounded text-white/80 font-mono">A</span>
              <span>auto</span>
            </>
          )}
        </div>
      )}

      <div
        className={`absolute inset-0 flex items-center justify-center ${
          interacting ? '' : 'transition-transform duration-75'
        }`}
        style={transformStyle}
      >
        <div className="relative z-10" style={aspectStyle} ref={imageRef}>
          {/* Base Target Image */}
          <img
            src={beforeSrc}
            alt="Before"
            className="relative z-[1] w-full h-full object-contain pointer-events-none"
            onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
            draggable={false}
          />
          <div className="absolute inset-0 pointer-events-none z-30">{faceBoxes}</div>

          {/* Swapped Image Overlay with Multi-Mode Comparison Styling */}
          <div
            className="absolute inset-0 pointer-events-none z-20"
            style={
              isPeekingOriginal
                ? { opacity: 0 }
                : !compare
                ? { opacity: 1 }
                : compareMode === 'blend'
                ? { opacity: sliderPosition / 100 }
                : compareMode === 'diff'
                ? { mixBlendMode: 'difference', opacity: 1 }
                : compareDir === 'horizontal'
                ? { clipPath: `polygon(0 ${currentClipPosition}%, 100% ${currentClipPosition}%, 100% 100%, 0 100%)` }
                : { clipPath: `polygon(${currentClipPosition}% 0, 100% 0, 100% 100%, ${currentClipPosition}% 100%)` }
            }
          >
            <CrossfadeImage
              src={afterSrc || beforeSrc}
              fadeMs={isPlaying ? 0 : scrubbing ? 60 : 200}
              className="absolute inset-0 w-full h-full object-contain"
            />
          </div>

          {/* Interactive Mask Paint Canvas Overlay */}
          {imgDim && (
            <>
              <canvas
                ref={maskCanvasRef}
                width={imgDim.w}
                height={imgDim.h}
                className={`absolute inset-0 w-full h-full object-contain z-[35] ${
                  maskBrushActive
                    ? 'cursor-crosshair pointer-events-auto'
                    : 'pointer-events-none opacity-60'
                }`}
              />
              {/* Offscreen solid-white twin — this is what gets exported. */}
              <canvas
                ref={maskExportRef}
                width={imgDim.w}
                height={imgDim.h}
                style={{ display: 'none' }}
                aria-hidden
              />
            </>
          )}

          {/* Draggable Vertical or Horizontal Compare Slider Handle */}
          {compare && !isPeekingOriginal && compareMode === 'slider' && (
            compareDir === 'vertical' ? (
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-gradient-to-b from-[var(--accent)] via-white to-[var(--accent)] shadow-[0_0_12px_rgba(233,69,96,0.8)] z-40 pointer-events-none"
                style={{ left: `${sliderPosition}%` }}
              >
                <div
                  className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 bg-black/80 backdrop-blur-md border border-white/30 rounded-full flex items-center justify-center shadow-[0_4px_20px_rgba(0,0,0,0.7)] transition-transform duration-200 group-hover:scale-110 cursor-ew-resize pointer-events-auto"
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    setIsDraggingSlider(true);
                    handleSliderMove(e.clientX ?? e.touches?.[0]?.clientX, e.clientY ?? e.touches?.[0]?.clientY);
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    setSliderPosition(50);
                  }}
                  title="Drag left/right to compare faces (Double-click to reset to 50%)"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="mr-0.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="rotate-180 ml-0.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
                </div>
              </div>
            ) : (
              <div
                className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-[var(--accent)] via-white to-[var(--accent)] shadow-[0_0_12px_rgba(233,69,96,0.8)] z-40 pointer-events-none"
                style={{ top: `${sliderPosition}%` }}
              >
                <div
                  className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 bg-black/80 backdrop-blur-md border border-white/30 rounded-full flex items-center justify-center shadow-[0_4px_20px_rgba(0,0,0,0.7)] transition-transform duration-200 group-hover:scale-110 cursor-ns-resize pointer-events-auto"
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    setIsDraggingSlider(true);
                    handleSliderMove(e.clientX ?? e.touches?.[0]?.clientX, e.clientY ?? e.touches?.[0]?.clientY);
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    setSliderPosition(50);
                  }}
                  title="Drag up/down to compare faces (Double-click to reset to 50%)"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="-mt-0.5 rotate-90"><polyline points="15 18 9 12 15 6"></polyline></svg>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="-mb-0.5 -rotate-90"><polyline points="15 18 9 12 15 6"></polyline></svg>
                </div>
              </div>
            )
          )}
        </div>
      </div>

      {/* Feature 1: Interactive Magnifier Glass Lens Overlay */}
      {magnifierActive && lensPos.visible && (
        <div
          className="absolute rounded-full border-2 border-cyan-400 shadow-[0_0_30px_rgba(6,182,212,0.6)] overflow-hidden pointer-events-none z-50 bg-black"
          style={{
            width: LENS_R * 2,
            height: LENS_R * 2,
            left: `${lensPos.x - LENS_R}px`,
            top: `${lensPos.y - LENS_R}px`,
          }}
        >
          <img
            src={isPeekingOriginal ? beforeSrc : (afterSrc || beforeSrc)}
            alt="Magnifier lens view"
            draggable={false}
            style={{
              position: 'absolute',
              width: lensPos.imgW * LENS_ZOOM,
              height: lensPos.imgH * LENS_ZOOM,
              maxWidth: 'none',
              objectFit: 'contain',
              // Put the pixel under the cursor at the centre of the glass.
              left: -((lensPos.x - lensPos.imgX) * LENS_ZOOM - LENS_R),
              top: -((lensPos.y - lensPos.imgY) * LENS_ZOOM - LENS_R),
            }}
          />
          {/* Crosshair so the exact sampled pixel is unambiguous */}
          <span className="absolute left-1/2 top-1/2 w-4 h-px -translate-x-1/2 -translate-y-1/2 bg-cyan-300/70" />
          <span className="absolute left-1/2 top-1/2 h-4 w-px -translate-x-1/2 -translate-y-1/2 bg-cyan-300/70" />
          <span className="absolute bottom-1.5 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded bg-black/80 backdrop-blur border border-cyan-400/40 text-[9px] font-mono font-bold text-cyan-300 tracking-wider">
            {(LENS_ZOOM * zoom).toFixed(1)}× LENS
          </span>
        </div>
      )}

      {/* Bottom labels */}
      {compare && (
        <span className="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg bg-black/75 backdrop-blur-md text-[10px] font-bold uppercase tracking-[0.14em] text-white/80 z-30 pointer-events-none border border-white/10 shadow-lg">
          Before (Target Original)
        </span>
      )}
      <span
        className="absolute bottom-3 right-3 px-2.5 py-1 rounded-lg bg-[var(--accent)]/90 backdrop-blur-md text-[10px] font-bold uppercase tracking-[0.14em] text-white z-30 transition-opacity duration-300 pointer-events-none border border-white/20 shadow-lg"
        style={{ opacity: compare && sliderPosition > 85 ? 0 : 1 }}
      >
        {isPeekingOriginal ? 'Original (Peek)' : compare ? 'After (Swapped)' : 'Swapped Result'}
      </span>

      {hudBar()}
    </div>
  );
}
