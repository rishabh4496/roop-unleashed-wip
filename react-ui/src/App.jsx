import React, { useEffect, useState, useCallback, useRef, useMemo, Suspense, lazy } from 'react';
import { getJSON, postJSON } from './api';
import { Toast, Confetti } from './components/ui';
import CommandPalette from './components/CommandPalette';
import { playChime, notifyDesktop, fmtTime } from './components/faceswap/utils';
// Tab panels are code-split so the initial bundle only ships the shell + the
// first tab's dependencies. Each is fetched on first visit (Vite emits one
// chunk per lazy import), trimming a ~530 KB single bundle into per-tab pieces.
const FaceSwap = lazy(() => import('./components/FaceSwap'));
const Settings = lazy(() => import('./components/Settings'));
const FaceManager = lazy(() => import('./components/FaceManager'));
const Extras = lazy(() => import('./components/Extras'));
const Gallery = lazy(() => import('./components/Gallery'));
import { THEME_CLASSES, THEMES, themeByName } from './themes';
import { motion, AnimatePresence, MotionConfig, spring, viewTransition } from './motion';

// Lightweight fallback while a tab chunk loads — mirrors the app's connecting
// spinner so the swap reads as intentional, not a flash of empty space.
function TabFallback() {
  return (
    <div className="flex flex-col items-center justify-center h-[40vh] gap-3">
      <div className="h-7 w-7 rounded-full border-4 border-white/10 border-t-[var(--accent)] animate-spin" />
      <div className="text-white/35 text-xs font-medium">Loading…</div>
    </div>
  );
}

const TABS = [
  { id: 'faceswap', label: '🎭 Face Swap' },
  { id: 'facemgr', label: '👥 Face Manager' },
  { id: 'extras', label: '✏️ Editor' },
  { id: 'gallery', label: '📂 Outputs' },
  { id: 'settings', label: '⚙️ Settings' },
];

export default function App() {
  const [tab, setTab] = useState('faceswap');
  const [meta, setMeta] = useState(null);
  const [settings, setSettings] = useState(null);
  const [toast, setToast] = useState(null);
  const [error, setError] = useState('');

  const [progress, setProgress] = useState({ processing: false, progress: 0, desc: '', output: null });
  const [startTime, setStartTime] = useState(null);
  const [confetti, setConfetti] = useState(false);
  const pollRef = useRef(null);

  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const pr = await getJSON('/api/progress');
        setProgress(pr);
        if (!pr.processing) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch { /* ignore */ }
    }, 1000);
  }, []);

  useEffect(() => {
    if (progress.processing && !pollRef.current) {
      startPolling();
    }
    return () => {
      if (!progress.processing && pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [progress.processing, startPolling]);

  const prevProcessingRef = useRef(false);
  useEffect(() => {
    const was = prevProcessingRef.current;
    prevProcessingRef.current = progress.processing;
    if (progress.processing && !was) {
      if (!startTime) setStartTime(Date.now());
    } else if (!progress.processing && was) {
      setStartTime(null);
    }
  }, [progress.processing, startTime]);

  const prevProcessingCelebrationRef = useRef(false);
  useEffect(() => {
    const was = prevProcessingCelebrationRef.current;
    prevProcessingCelebrationRef.current = progress.processing;
    if (was && !progress.processing && !progress.error && (progress.progress || 0) >= 0.99) {
      playChime();
      notifyDesktop('✨ Swap complete', progress.output?.name ? `${progress.output.name} is ready` : 'Your render is ready');
      setConfetti(false);
      requestAnimationFrame(() => setConfetti(true));
      setTimeout(() => setConfetti(false), 2600);
    }
  }, [progress.processing, progress.error, progress.progress, progress.output]);

  const [isDraggingOver, setIsDraggingOver] = useState(false);
  useEffect(() => { isDraggingOverRef.current = isDraggingOver; }, [isDraggingOver]);
  const [showPalette, setShowPalette] = useState(false);
  const fileListenersRef = useRef([]);
  const dragHideTimerRef = useRef(null);
  const isDraggingOverRef = useRef(false);

  // App-level UI zoom (Chrome-style). Uses the CSS `zoom` property, which
  // reflows layout instead of just visually scaling, so the whole UI grows /
  // shrinks cleanly. Persisted so it survives reloads.
  const [zoom, setZoom] = useState(() => {
    const v = parseFloat(localStorage.getItem('roop_zoom'));
    return v && v >= 0.5 && v <= 1.6 ? v : 1;
  });
  const bumpZoom = useCallback((d) => setZoom((z) => Math.min(1.6, Math.max(0.5, Math.round((z + d) * 20) / 20))), []);
  useEffect(() => {
    document.documentElement.style.zoom = String(zoom);
    localStorage.setItem('roop_zoom', String(zoom));
  }, [zoom]);

  // Global keyboard shortcuts: ⌘/Ctrl-K palette, and ⌘/Ctrl +/-/0 zoom.
  useEffect(() => {
    const onKey = (e) => {
      if (!(e.metaKey || e.ctrlKey)) return;
      const k = e.key.toLowerCase();
      if (k === 'k') { e.preventDefault(); setShowPalette((v) => !v); }
      else if (e.key === '=' || e.key === '+') { e.preventDefault(); bumpZoom(0.05); }
      else if (e.key === '-' || e.key === '_') { e.preventDefault(); bumpZoom(-0.05); }
      else if (e.key === '0') { e.preventDefault(); setZoom(1); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [bumpZoom]);

  const applyTheme = useCallback((name) => {
    setSettings((s) => ({ ...(s || {}), selected_theme: name }));
    postJSON('/api/settings', { selected_theme: name }).catch(() => {});
  }, []);

  // Faceswap-tab actions are decoupled via a window event bus so the palette
  // stays independent of the FaceSwap component's internal handlers.
  const runFaceswap = useCallback((id) => {
    setTab('faceswap');
    setTimeout(() => window.dispatchEvent(new CustomEvent('roop:command', { detail: { id } })), 60);
  }, []);

  const commands = useMemo(() => {
    const cmds = [];
    TABS.forEach((t) => cmds.push({ id: `nav-${t.id}`, section: 'Navigate', icon: t.label.split(' ')[0], title: `Go to ${t.label.replace(/^\S+\s/, '')}`, run: () => setTab(t.id) }));
    cmds.push({ id: 'act-start', section: 'Actions', icon: '▶', title: 'Start swapping', subtitle: 'Run the current job', run: () => runFaceswap('start') });
    cmds.push({ id: 'act-stop', section: 'Actions', icon: '⏹', title: 'Stop processing', run: () => runFaceswap('stop') });
    cmds.push({ id: 'act-queue', section: 'Actions', icon: '➕', title: 'Add current to batch queue', run: () => runFaceswap('queue') });
    cmds.push({ id: 'act-compare', section: 'Actions', icon: '🔍', title: 'Toggle before/after compare', run: () => runFaceswap('compare') });
    cmds.push({ id: 'act-split', section: 'Actions', icon: '⬍', title: 'Toggle split view', run: () => runFaceswap('split') });
    cmds.push({ id: 'act-preview', section: 'Actions', icon: '🔄', title: 'Refresh preview', run: () => runFaceswap('preview') });
    cmds.push({ id: 'act-shortcuts', section: 'Actions', icon: '⌨️', title: 'Show keyboard shortcuts', run: () => runFaceswap('shortcuts') });
    THEMES.forEach((t) => cmds.push({ id: `theme-${t.name}`, section: 'Theme', icon: '🎨', title: t.name, subtitle: t.label, run: () => applyTheme(t.name) }));
    return cmds;
  }, [applyTheme, runFaceswap]);

  const registerFileListener = useCallback((cb) => {
    fileListenersRef.current.push(cb);
    return () => {
      fileListenersRef.current = fileListenersRef.current.filter((l) => l !== cb);
    };
  }, []);

  useEffect(() => {
    // The overlay is kept alive by a self-resetting timer: dragover fires
    // continuously while a file is over the window, so we re-arm a short hide
    // timer on every event. When the drag leaves the window, is dropped
    // elsewhere, or is CANCELLED (Escape) — all of which stop the dragover
    // stream — the timer fires and clears the overlay. This is far more robust
    // than the old clientX/Y===0 corner check, which never fired on a cancelled
    // drag and left the overlay stuck.
    const armHideTimer = () => {
      // 450ms comfortably exceeds Chromium's ~350ms stationary-dragover interval,
      // so holding a file still over the window won't flicker the overlay off;
      // once the drag actually leaves/cancels (no more dragover) it clears fast.
      if (dragHideTimerRef.current) clearTimeout(dragHideTimerRef.current);
      dragHideTimerRef.current = setTimeout(() => setIsDraggingOver(false), 450);
    };
    const clearOverlay = () => {
      if (dragHideTimerRef.current) { clearTimeout(dragHideTimerRef.current); dragHideTimerRef.current = null; }
      setIsDraggingOver(false);
    };
    const handleDragOver = (e) => {
      // Only react to OS file drags — internal drags (e.g. dragging a source
      // face onto a person) carry custom types, not 'Files', and must not
      // trigger the full-screen "drop media" overlay.
      if (!Array.from(e.dataTransfer?.types || []).includes('Files')) return;
      e.preventDefault();
      setIsDraggingOver(true);
      armHideTimer();
    };
    const handleDragLeave = (e) => {
      e.preventDefault();
    };
    const handleDrop = (e) => {
      e.preventDefault();
      clearOverlay();
      // A dedicated dropzone (FileDrop) already handled this drop.
      if (e.roopConsumed) return;
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        fileListenersRef.current.forEach((listener) => listener(files));
      }
    };
    const handlePaste = (e) => {
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
      const items = e.clipboardData?.items;
      if (!items) return;
      const files = [];
      for (let i = 0; i < items.length; i++) {
        if (items[i].kind === 'file') {
          files.push(items[i].getAsFile());
        }
      }
      if (files.length > 0) {
        fileListenersRef.current.forEach((listener) => listener(files));
      }
    };
    // Extra safety nets so the overlay can never get stuck: a cancelled drag
    // fires dragend, and any click/keydown dismisses a lingering overlay.
    const handleDragEnd = () => clearOverlay();
    const handlePointerDown = () => { if (dragHideTimerRef.current || isDraggingOverRef.current) clearOverlay(); };
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('drop', handleDrop);
    window.addEventListener('dragend', handleDragEnd);
    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('paste', handlePaste);
    return () => {
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('drop', handleDrop);
      window.removeEventListener('dragend', handleDragEnd);
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('paste', handlePaste);
      if (dragHideTimerRef.current) clearTimeout(dragHideTimerRef.current);
    };
  }, []);

  const toastTimerRef = useRef(null);
  const notify = useCallback((message, type = 'success') => {
    setToast({ message, type });
    // Reset the dismiss timer, otherwise an earlier toast's timeout hides a
    // newer toast prematurely.
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToast(null), 3000);
  }, []);

  useEffect(() => {
    Promise.all([getJSON('/api/meta'), getJSON('/api/settings')])
      .then(([m, s]) => { setMeta(m); setSettings(s); })
      .catch(() => setError('Cannot reach backend on 127.0.0.1:8001. Make sure the server (run.py) is running.'));

    getJSON('/api/progress').then((pr) => {
      setProgress(pr);
      if (pr.processing) startPolling();
    }).catch(() => {});

    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [startPolling]);

  // Autosave settings to the backend CFG (debounced) on every in-session edit.
  // Previously settings were only persisted at swap-start / explicit Save, so
  // switching Pinokio's Run<->Dev view — which reloads this webview and remounts
  // React — reverted any unsaved edits to the backend's last-saved values (the
  // load effect above re-fetches /api/settings on every mount). Persisting each
  // change makes the backend the durable source of truth across reloads.
  const settingsLoadedRef = useRef(false);
  const settingsSaveRef = useRef(null);
  const settingsDirtyRef = useRef(null); // latest not-yet-persisted settings, or null when clean
  // Persist the pending edit now. keepalive lets it survive the webview teardown
  // when this fires from pagehide/visibilitychange during a Run<->Dev reload.
  const flushSettings = useCallback((keepalive = false) => {
    const body = settingsDirtyRef.current;
    if (!body) return;
    settingsDirtyRef.current = null;
    if (settingsSaveRef.current) { clearTimeout(settingsSaveRef.current); settingsSaveRef.current = null; }
    postJSON('/api/settings', body, { keepalive }).catch(() => { /* offline — persists on next edit/run */ });
  }, []);
  useEffect(() => {
    if (!settings) return;
    // Skip the first value (just fetched from the backend) so we don't re-POST
    // exactly what we loaded on mount.
    if (!settingsLoadedRef.current) { settingsLoadedRef.current = true; return; }
    settingsDirtyRef.current = settings;
    if (settingsSaveRef.current) clearTimeout(settingsSaveRef.current);
    settingsSaveRef.current = setTimeout(() => flushSettings(false), 500);
    return () => { if (settingsSaveRef.current) clearTimeout(settingsSaveRef.current); };
  }, [settings, flushSettings]);
  // Flush a pending edit immediately when the webview is hidden/torn down, so a
  // change made <500ms before a Run<->Dev toggle isn't dropped by the debounce.
  useEffect(() => {
    const onPageHide = () => flushSettings(true);
    const onVisibility = () => { if (document.visibilityState === 'hidden') flushSettings(true); };
    window.addEventListener('pagehide', onPageHide);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.removeEventListener('pagehide', onPageHide);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [flushSettings]);

  // Apply selected theme class to body/html
  useEffect(() => {
    if (!settings || !settings.selected_theme) return;
    document.documentElement.classList.remove(...THEME_CLASSES);
    document.body.classList.remove(...THEME_CLASSES);
    const cls = themeByName(settings.selected_theme).className;
    if (cls) {
      document.documentElement.classList.add(cls);
      document.body.classList.add(cls);
    }
  }, [settings]);

  return (
    <MotionConfig reducedMotion="user">
    <div className="min-h-screen flex flex-col relative overflow-hidden select-none">
      {/* Floating Ambient Background Glows — static.
          These are two very large filter-blur discs. When they animated
          (transform translate/scale on an infinite loop) the compositor had to
          re-rasterize an enormous blurred layer every single frame, which
          taxed the GPU continuously and made scrolling, zooming and the live
          preview feel laggy across the whole app. Static blurred layers are
          rasterized once and cached for free, so we keep the exact same look
          without the per-frame cost. */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <div className="absolute top-[-28%] left-[-18%] w-[55%] h-[55%] rounded-full bg-[var(--accent)]/[0.04] blur-[150px]" />
        <div className="absolute bottom-[-24%] right-[-18%] w-[60%] h-[60%] rounded-full bg-[#8B5CF6]/[0.03] blur-[160px]" />
      </div>

      {/* Floating Header Capsule */}
      <header className="sticky top-4 z-40 mx-auto max-w-none w-[98%] rounded-2xl glass-panel px-5 py-3 flex flex-col md:flex-row items-center justify-between gap-4 border-white/10">
        <div className="flex items-center gap-3">
          <span className="grid place-items-center h-9 w-9 rounded-xl bg-[var(--accent)]/12 border border-[var(--accent)]/25 text-lg">⚡</span>
          <div>
            <h1 className="text-[15px] font-bold tracking-tight text-white/95 flex items-center gap-1.5">
              Roop Unleashed <span className="text-white/35 font-medium">Studio</span>
            </h1>
            {meta?.git_version && (
              <span className="text-[9px] font-mono text-white/35 tracking-wider block mt-0.5">
                Engine {meta.git_version}
              </span>
            )}
          </div>
          {progress.processing && (
            <div className="ml-2 flex items-center gap-2 px-2.5 py-1 rounded-lg bg-[var(--accent)]/10 border border-[var(--accent)]/20 text-[10px] font-bold tracking-wide uppercase">
              <span className={`h-1.5 w-1.5 rounded-full ${progress.paused ? 'bg-amber-400' : 'bg-[var(--accent)] animate-ping'}`} />
              <span className={progress.paused ? 'text-amber-400/90' : 'text-[var(--accent)]'}>
                {progress.paused ? 'Paused' : `Processing ${Math.round((progress.progress || 0) * 100)}%`}
              </span>
              {startTime && (progress.progress || 0) > 0.01 && (
                <span className="text-white/40 normal-case font-mono font-medium ml-1">
                  ETA: {fmtTime(((Date.now() - startTime) * (1 - progress.progress)) / progress.progress)}
                </span>
              )}
              <div className="flex items-center gap-1.5 border-l border-white/10 pl-2 ml-1">
                {progress.paused ? (
                  <button
                    type="button"
                    onClick={async (e) => {
                      e.stopPropagation();
                      try {
                        await postJSON('/api/resume', {});
                        setProgress((pr) => ({ ...pr, paused: false, desc: 'Resuming…' }));
                      } catch {}
                    }}
                    className="hover:text-white text-white/60 transition-colors cursor-pointer"
                    title="Resume Job"
                  >
                    ▶
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={async (e) => {
                      e.stopPropagation();
                      try {
                        await postJSON('/api/pause', {});
                        setProgress((pr) => ({ ...pr, paused: true, desc: 'Paused' }));
                      } catch {}
                    }}
                    className="hover:text-white text-white/60 transition-colors cursor-pointer"
                    title="Pause Job"
                  >
                    ⏸
                  </button>
                )}
                <button
                  type="button"
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (window.confirm('Stop the active job?')) {
                      try {
                        await postJSON('/api/stop', {});
                      } catch {}
                    }
                  }}
                  className="hover:text-red-400 text-white/60 transition-colors cursor-pointer"
                  title="Stop Job"
                >
                  ⏹
                </button>
              </div>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2 w-full md:w-auto">
        <button
          type="button"
          onClick={() => setShowPalette(true)}
          title="Command palette (Ctrl/⌘ + K)"
          className="hidden md:flex items-center gap-2 px-3 py-2 rounded-xl bg-white/[0.03] border border-white/10 text-white/45 hover:text-white hover:border-white/20 hover:bg-white/[0.06] transition-colors text-xs font-medium"
        >
          <span className="text-sm">⌘</span> Search
          <kbd className="text-[9px] font-mono bg-white/5 px-1.5 py-0.5 rounded border border-white/10">Ctrl K</kbd>
        </button>
        <div className="hidden md:flex items-center gap-0.5 px-1 py-1 rounded-xl bg-white/[0.03] border border-white/10" title="UI zoom (Ctrl + / − / 0)">
          <button type="button" onClick={() => bumpZoom(-0.05)} title="Zoom out (Ctrl −)" className="h-6 w-6 grid place-items-center rounded-lg text-white/50 hover:text-white hover:bg-white/10 text-base leading-none transition-colors">−</button>
          <button type="button" onClick={() => setZoom(1)} title="Reset zoom (Ctrl 0)" className="min-w-[44px] text-[11px] font-semibold text-white/60 hover:text-white tabular-nums transition-colors">{Math.round(zoom * 100)}%</button>
          <button type="button" onClick={() => bumpZoom(0.05)} title="Zoom in (Ctrl +)" className="h-6 w-6 grid place-items-center rounded-lg text-white/50 hover:text-white hover:bg-white/10 text-base leading-none transition-colors">+</button>
        </div>
        <nav className="flex gap-0.5 bg-black/25 p-1 rounded-xl border border-white/[0.06] w-full md:w-auto overflow-x-auto">
          {TABS.map((t) => {
            const active = tab === t.id;
            return (
              <motion.button
                key={t.id}
                onClick={() => setTab(t.id)}
                whileTap={{ scale: 0.94 }}
                transition={spring.snappy}
                className={`relative px-3.5 py-2 rounded-lg text-[12px] font-semibold tracking-wide whitespace-nowrap flex items-center gap-1.5 transition-colors duration-200 ${
                  active ? 'text-white' : 'text-white/45 hover:text-white/90'
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="tab-pill"
                    className="absolute inset-0 rounded-lg bg-white/[0.08] border border-white/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]"
                    transition={spring.snappy}
                  />
                )}
                <span className="relative z-10 flex items-center gap-1.5">{t.label}</span>
              </motion.button>
            );
          })}
        </nav>
        </div>
      </header>

      {/* Main Container Layout */}
      <main className="flex-1 w-[98%] max-w-none mx-auto px-6 py-8 mt-4 z-10 relative">
        {error && (
          <div className="rounded-2xl bg-red-500/10 border border-red-500/20 p-5 text-sm text-red-300 animate-slide-up">
            ⚠️ {error}
          </div>
        )}
        {!error && !meta && (
          <div className="flex flex-col items-center justify-center h-[50vh] gap-4">
            <div className="h-8 w-8 rounded-full border-4 border-white/10 border-t-[var(--accent)] animate-spin" />
            <div className="text-white/40 text-sm font-medium">Establishing secure gateway connection…</div>
          </div>
        )}
        {!error && meta && settings && (
          <AnimatePresence mode="wait">
            <motion.div
              key={tab}
              variants={viewTransition}
              initial="initial"
              animate="animate"
              exit="exit"
            >
              <Suspense fallback={<TabFallback />}>
                {tab === 'faceswap' && (
                  <FaceSwap
                    meta={meta}
                    settings={settings}
                    setSettings={setSettings}
                    notify={notify}
                    registerFileListener={registerFileListener}
                    progress={progress}
                    setProgress={setProgress}
                    startTime={startTime}
                    setStartTime={setStartTime}
                  />
                )}
                {tab === 'facemgr' && <FaceManager notify={notify} registerFileListener={registerFileListener} />}
                {tab === 'extras' && <Extras notify={notify} registerFileListener={registerFileListener} />}
                {tab === 'gallery' && <Gallery notify={notify} setSettings={setSettings} setTab={setTab} />}
                {tab === 'settings' && <Settings meta={meta} settings={settings} setSettings={setSettings} notify={notify} />}
              </Suspense>
            </motion.div>
          </AnimatePresence>
        )}
      </main>

      <Toast toast={toast} />

      <Confetti active={confetti} />

      {progress.processing && (
        <div
          className="fixed top-0 left-0 h-[3px] bg-[var(--accent)] z-[60] transition-all duration-300 shadow-[0_0_8px_var(--accent-glow)]"
          style={{ width: `${Math.round((progress.progress || 0) * 100)}%` }}
        />
      )}

      <CommandPalette open={showPalette} onClose={() => setShowPalette(false)} commands={commands} />

      {isDraggingOver && (
        <div className="fixed inset-0 bg-[var(--accent)]/10 backdrop-blur-[2px] border-4 border-dashed border-[var(--accent)] z-50 flex items-center justify-center pointer-events-none">
          <div className="bg-black/80 px-6 py-4 rounded-2xl border border-white/10 shadow-2xl flex flex-col items-center gap-2">
            <span className="text-4xl animate-bounce">📥</span>
            <span className="text-lg font-bold text-white uppercase tracking-wider">Drop media here</span>
          </div>
        </div>
      )}
    </div>
    </MotionConfig>
  );
}
