import React, { useEffect, useState, useCallback, useRef } from 'react';
import { getJSON } from './api';
import { Toast } from './components/ui';
import FaceSwap from './components/FaceSwap';
import Settings from './components/Settings';
import FaceManager from './components/FaceManager';
import Extras from './components/Extras';
import Gallery from './components/Gallery';

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
  const fileListenersRef = useRef([]);

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
    const theme = settings.selected_theme;
    
    const themeClasses = [
      'theme-glass-light',
      'theme-cyberpunk-dark',
      'theme-cyberpunk-light',
      'theme-emerald-dark',
      'theme-emerald-light',
      'theme-nordic-dark',
      'theme-nordic-light'
    ];
    document.documentElement.classList.remove(...themeClasses);
    document.body.classList.remove(...themeClasses);
    
    const themeClassMap = {
      'Glass Light': 'theme-glass-light',
      'Cyberpunk Dark': 'theme-cyberpunk-dark',
      'Cyberpunk Light': 'theme-cyberpunk-light',
      'Emerald Dark': 'theme-emerald-dark',
      'Emerald Light': 'theme-emerald-light',
      'Nordic Dark': 'theme-nordic-dark',
      'Nordic Light': 'theme-nordic-light'
    };
    
    const cls = themeClassMap[theme];
    if (cls) {
      document.documentElement.classList.add(cls);
      document.body.classList.add(cls);
    }
  }, [settings]);

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden select-none">
      {/* Floating Ambient Background Glows */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <div className="absolute top-[-25%] left-[-15%] w-[60%] h-[60%] rounded-full bg-[var(--accent)]/[0.06] blur-[150px] animate-float-1" />
        <div className="absolute bottom-[-20%] right-[-15%] w-[65%] h-[65%] rounded-full bg-[#C5A880]/[0.04] blur-[170px] animate-float-2" />
      </div>
 
      {/* Floating Header Capsule */}
      <header className="sticky top-4 z-40 mx-auto max-w-none w-[98%] rounded-2xl glass-panel px-6 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4 shadow-2xl border-white/10">
        <div className="flex items-center gap-3">
          <span className="text-2xl animate-pulse">⚡</span>
          <div>
            <h1 className="text-lg font-black tracking-widest uppercase text-white/95 flex items-center gap-2">
              Roop Unleashed <span className="bg-gradient-to-r from-[#C5A880] to-[var(--accent)] bg-clip-text text-transparent font-black">Studio Pro</span>
            </h1>
            {meta?.git_version && (
              <span className="text-[9px] font-mono text-white/40 tracking-widest block uppercase mt-0.5">
                Engine: {meta.git_version}
              </span>
            )}
          </div>
        </div>
        <nav className="flex gap-1 bg-black/40 p-1 rounded-2xl border border-white/5 w-full md:w-auto overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2.5 rounded-xl text-xs font-bold uppercase tracking-wider whitespace-nowrap transition-all duration-300 apple-transition apple-spring-active flex items-center gap-1.5 ${
                tab === t.id
                  ? 'bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] text-white shadow-[0_4px_15px_var(--accent-glow)] scale-[1.03]'
                  : 'text-white/50 hover:text-white hover:bg-white/5 hover:scale-[1.02]'
              }`}
            >
              {tab === t.id && <span className="h-1.5 w-1.5 rounded-full bg-white animate-ping" />}
              {t.label}
            </button>
          ))}
        </nav>
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
          <div key={tab} className="animate-slide-up">
            {tab === 'faceswap' && <FaceSwap meta={meta} settings={settings} setSettings={setSettings} notify={notify} registerFileListener={registerFileListener} />}
            {tab === 'facemgr' && <FaceManager notify={notify} registerFileListener={registerFileListener} />}
            {tab === 'extras' && <Extras notify={notify} registerFileListener={registerFileListener} />}
            {tab === 'gallery' && <Gallery notify={notify} />}
            {tab === 'settings' && <Settings meta={meta} settings={settings} setSettings={setSettings} notify={notify} />}
          </div>
        )}
      </main>

      <Toast toast={toast} />

      {isDraggingOver && (
        <div className="fixed inset-0 bg-[var(--accent)]/10 backdrop-blur-[2px] border-4 border-dashed border-[var(--accent)] z-50 flex items-center justify-center pointer-events-none">
          <div className="bg-black/80 px-6 py-4 rounded-2xl border border-white/10 shadow-2xl flex flex-col items-center gap-2">
            <span className="text-4xl animate-bounce">📥</span>
            <span className="text-lg font-bold text-white uppercase tracking-wider">Drop media here</span>
          </div>
        </div>
      )}
    </div>
  );
}
