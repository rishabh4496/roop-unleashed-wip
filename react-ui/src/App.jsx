import React, { useEffect, useState, useCallback, useRef, useMemo } from 'react';
import { getJSON, postJSON } from './api';
import { Toast } from './components/ui';
import CommandPalette from './components/CommandPalette';
import FaceSwap from './components/FaceSwap';
import Settings from './components/Settings';
import FaceManager from './components/FaceManager';
import Extras from './components/Extras';
import Gallery from './components/Gallery';
import { THEME_CLASSES, THEMES, themeByName } from './themes';
import { motion, AnimatePresence, MotionConfig, spring, viewTransition } from './motion';

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

  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [showPalette, setShowPalette] = useState(false);
  const fileListenersRef = useRef([]);

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
    const handleDragOver = (e) => {
      // Only react to OS file drags — internal drags (e.g. dragging a source
      // face onto a person) carry custom types, not 'Files', and must not
      // trigger the full-screen "drop media" overlay.
      if (!Array.from(e.dataTransfer?.types || []).includes('Files')) return;
      e.preventDefault();
      setIsDraggingOver(true);
    };
    const handleDragLeave = (e) => {
      e.preventDefault();
      if (e.clientX === 0 && e.clientY === 0) {
        setIsDraggingOver(false);
      }
    };
    const handleDrop = (e) => {
      e.preventDefault();
      setIsDraggingOver(false);
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
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('drop', handleDrop);
    window.addEventListener('paste', handlePaste);
    return () => {
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('drop', handleDrop);
      window.removeEventListener('paste', handlePaste);
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
  }, []);

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
      {/* Floating Ambient Background Glows — subtle, slow */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <div className="absolute top-[-28%] left-[-18%] w-[55%] h-[55%] rounded-full bg-[var(--accent)]/[0.04] blur-[170px] animate-float-1" />
        <div className="absolute bottom-[-24%] right-[-18%] w-[60%] h-[60%] rounded-full bg-[#8B5CF6]/[0.03] blur-[190px] animate-float-2" />
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
              {tab === 'faceswap' && <FaceSwap meta={meta} settings={settings} setSettings={setSettings} notify={notify} registerFileListener={registerFileListener} />}
              {tab === 'facemgr' && <FaceManager notify={notify} registerFileListener={registerFileListener} />}
              {tab === 'extras' && <Extras notify={notify} registerFileListener={registerFileListener} />}
              {tab === 'gallery' && <Gallery notify={notify} />}
              {tab === 'settings' && <Settings meta={meta} settings={settings} setSettings={setSettings} notify={notify} />}
            </motion.div>
          </AnimatePresence>
        )}
      </main>

      <Toast toast={toast} />

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
