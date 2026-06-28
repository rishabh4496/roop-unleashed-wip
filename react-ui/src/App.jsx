import React, { useEffect, useState, useCallback } from 'react';
import { getJSON } from './api';
import { Toast } from './components/ui';
import FaceSwap from './components/FaceSwap';
import Settings from './components/Settings';
import FaceManager from './components/FaceManager';
import Extras from './components/Extras';

const TABS = [
  { id: 'faceswap', label: '🎭 Face Swap' },
  { id: 'facemgr', label: '👥 Face Manager' },
  { id: 'extras', label: '✏️ Editor' },
  { id: 'settings', label: '⚙️ Settings' },
];

export default function App() {
  const [tab, setTab] = useState('faceswap');
  const [meta, setMeta] = useState(null);
  const [settings, setSettings] = useState(null);
  const [toast, setToast] = useState(null);
  const [error, setError] = useState('');

  const notify = useCallback((message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
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
    <div className="min-h-screen flex flex-col">
      {/* Floating Header Capsule */}
      <header className="sticky top-4 z-40 mx-auto max-w-[1450px] 3xl:max-w-[1800px] 4xl:max-w-[2300px] 5xl:max-w-[2800px] w-[92%] sm:w-[96%] rounded-2xl glass-panel px-6 py-3 flex flex-col md:flex-row items-center justify-between gap-4 shadow-xl">
        <div className="flex items-center gap-3">
          <span className="text-2xl">⚡</span>
          <h1 className="text-lg font-black tracking-wider uppercase text-white/90">
            Roop Unleashed <span className="bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] bg-clip-text text-transparent">Pro</span>
          </h1>
        </div>
        <nav className="flex gap-1 bg-black/20 p-1.5 rounded-xl border border-white/5 w-full md:w-auto overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 rounded-lg text-xs sm:text-sm font-bold tracking-wide whitespace-nowrap transition-all duration-300 apple-transition apple-spring-active ${
                tab === t.id
                  ? 'bg-[var(--accent)] text-white shadow-[0_4px_12px_var(--accent-glow)]'
                  : 'text-white/60 hover:text-white hover:bg-white/5'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {/* Main Container Layout */}
      <main className="flex-1 max-w-[1500px] 3xl:max-w-[1850px] 4xl:max-w-[2350px] 5xl:max-w-[2850px] w-full mx-auto px-6 py-8 mt-4">
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
          <div className="animate-slide-up">
            {tab === 'faceswap' && <FaceSwap meta={meta} settings={settings} setSettings={setSettings} notify={notify} />}
            {tab === 'facemgr' && <FaceManager notify={notify} />}
            {tab === 'extras' && <Extras notify={notify} />}
            {tab === 'settings' && <Settings meta={meta} settings={settings} setSettings={setSettings} notify={notify} />}
          </div>
        )}
      </main>

      <Toast toast={toast} />
    </div>
  );
}
