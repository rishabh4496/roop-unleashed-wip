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

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-40 px-6 py-3 border-b border-white/10 bg-black/30 backdrop-blur-xl flex items-center gap-6">
        <h1 className="text-xl font-bold tracking-wide whitespace-nowrap">Roop Unleashed <span className="text-[#E94560]">Pro</span></h1>
        <nav className="flex gap-1 overflow-x-auto">
          {TABS.map((t) => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${tab === t.id ? 'bg-[#E94560] text-white' : 'text-white/60 hover:bg-white/10'}`}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1 max-w-[1500px] w-full mx-auto px-6 py-6">
        {error && (
          <div className="rounded-xl bg-red-500/10 border border-red-500/30 p-4 text-sm text-red-300">{error}</div>
        )}
        {!error && !meta && <div className="text-white/40 text-sm">Connecting to backend…</div>}
        {!error && meta && settings && (
          <>
            {tab === 'faceswap' && <FaceSwap meta={meta} settings={settings} setSettings={setSettings} notify={notify} />}
            {tab === 'facemgr' && <FaceManager notify={notify} />}
            {tab === 'extras' && <Extras notify={notify} />}
            {tab === 'settings' && <Settings meta={meta} settings={settings} setSettings={setSettings} notify={notify} />}
          </>
        )}
      </main>

      <Toast toast={toast} />
    </div>
  );
}
