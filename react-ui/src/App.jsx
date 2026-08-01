import React, { useEffect, useState, useCallback, useRef, useMemo, Suspense, lazy } from 'react';
import { getJSON, postJSON } from './api';
import { Toasts, Confetti } from './components/ui';
import CommandPalette from './components/CommandPalette';
import ErrorBoundary from './components/ErrorBoundary';
import { ConfirmHost, confirmDialog } from './components/confirm';
import { playChime, notifyDesktop, fmtTime } from './components/faceswap/utils';
import { themeByName, allThemes, applyThemeToDom } from './themes';
import { SETTINGS_CATALOG, focusSetting } from './components/settingsCatalog';
import { motion, AnimatePresence, MotionConfig, spring, viewTransition } from './motion';
import { Icon } from './icons';

// Tab panels are code-split so the initial bundle only ships the shell + the
// first tab's dependencies. Each is fetched on first visit (Vite emits one
// chunk per lazy import), trimming a ~530 KB single bundle into per-tab pieces.
//
// The raw importer is kept alongside the lazy component so we can WARM a chunk
// before it is needed (pointer/focus on its tab button, and an idle sweep after
// first paint). Code-splitting bought a smaller first load but paid for it with
// a spinner on every first tab visit; prefetching keeps the win and removes the
// spinner, so a tab change is a pure animation instead of a network round trip.
// A rejected prefetch is ignored — React.lazy retries on real render, and the
// ErrorBoundary below owns the failure case.
const loadFaceSwap = () => import('./components/FaceSwap');
const loadSettings = () => import('./components/Settings');
const loadFaceManager = () => import('./components/FaceManager');
const loadExtras = () => import('./components/Extras');
const loadGallery = () => import('./components/Gallery');
const loadRunHistory = () => import('./components/RunHistory');

const FaceSwap = lazy(loadFaceSwap);
const Settings = lazy(loadSettings);
const FaceManager = lazy(loadFaceManager);
const Extras = lazy(loadExtras);
const Gallery = lazy(loadGallery);
const RunHistory = lazy(loadRunHistory);

// Lightweight fallback while a tab chunk loads — mirrors the app's connecting
// spinner so the swap reads as intentional, not a flash of empty space. It
// fades in on a delay (see `.deferred-fallback`) so a chunk that resolves in a
// few frames shows nothing at all rather than a jarring spinner blink.
function TabFallback() {
  return (
    <div className="deferred-fallback flex flex-col items-center justify-center h-[40vh] gap-3">
      <div className="h-7 w-7 rounded-full border-4 border-white/10 border-t-[var(--accent)] animate-spin" />
      <div className="text-white/35 text-xs font-medium">Loading…</div>
    </div>
  );
}

const TABS = [
  { id: 'faceswap', label: 'Face Swap', icon: Icon.faceswap, preload: loadFaceSwap },
  { id: 'facemgr', label: 'Face Manager', icon: Icon.faces, preload: loadFaceManager },
  { id: 'extras', label: 'Editor', icon: Icon.editor, preload: loadExtras },
  { id: 'gallery', label: 'Outputs', icon: Icon.outputs, preload: loadGallery },
  { id: 'history', label: 'History', icon: Icon.history, preload: loadRunHistory },
  { id: 'settings', label: 'Settings', icon: Icon.settings, preload: loadSettings },
];

// Fire each importer at most once; repeated hovers must not re-request.
const warmed = new Set();
const warmTab = (id) => {
  if (warmed.has(id)) return;
  warmed.add(id);
  const t = TABS.find((x) => x.id === id);
  t?.preload?.().catch(() => warmed.delete(id));
};

export default function App() {
  const [tab, setTab] = useState('faceswap');
  const [meta, setMeta] = useState(null);
  const [settings, setSettings] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [error, setError] = useState('');

  const [progress, setProgress] = useState({ processing: false, progress: 0, desc: '', output: null });
  const [startTime, setStartTime] = useState(null);
  const [confetti, setConfetti] = useState(false);
  const pollRef = useRef(null);

  // ── Connection health ────────────────────────────────────────────────────
  // Every backend call in this shell reports its outcome here. A single blip is
  // ignored (the server briefly stalls during heavy GPU work); three in a row
  // flips the UI to "reconnecting", which starts a cheap heartbeat that clears
  // itself the moment the backend answers again. Costs nothing while healthy —
  // the heartbeat only exists while we believe we are offline.
  const [offline, setOffline] = useState(false);
  const failsRef = useRef(0);
  const beatRef = useRef(null);
  const reportNet = useCallback((ok) => {
    if (ok) {
      failsRef.current = 0;
      setOffline(false);
    } else if (++failsRef.current >= 3) {
      setOffline(true);
    }
  }, []);

  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const pr = await getJSON('/api/progress', { timeout: 8000 });
        reportNet(true);
        setProgress(pr);
        if (!pr.processing) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch {
        // Keep polling: a job can outlive a transient backend stall, and the
        // health banner tells the user what is happening meanwhile.
        reportNet(false);
      }
    }, 1000);
  }, [reportNet]);

  // ── Catch up the moment this view is looked at again ─────────────────────
  // Switching to the Terminal (or another Pinokio tab) either reloads this
  // webview or leaves it hidden. If it survives, the browser throttles its
  // timers hard — a 1 s poll can drop to one a minute in a background tab, and
  // to nothing at all while occluded. Coming back then showed a progress bar
  // and a live frame stuck wherever they were when you left, until the next
  // throttled tick happened to fire.
  //
  // So: on every return to visibility, fetch once immediately and make sure the
  // poll is running. Cheap (one request per switch) and it makes the state on
  // screen current before the eye lands on it. `pageshow` covers the
  // back/forward cache, where no visibilitychange fires at all.
  useEffect(() => {
    const catchUp = async () => {
      if (document.visibilityState === 'hidden') return;
      try {
        const pr = await getJSON('/api/progress', { timeout: 8000 });
        reportNet(true);
        setProgress(pr);
        if (pr.processing && !pollRef.current) startPolling();
      } catch {
        reportNet(false);
      }
    };
    document.addEventListener('visibilitychange', catchUp);
    window.addEventListener('focus', catchUp);
    window.addEventListener('pageshow', catchUp);
    return () => {
      document.removeEventListener('visibilitychange', catchUp);
      window.removeEventListener('focus', catchUp);
      window.removeEventListener('pageshow', catchUp);
    };
  }, [startPolling, reportNet]);

  // Heartbeat while offline — reconnects and refreshes core state on recovery.
  useEffect(() => {
    if (!offline) {
      if (beatRef.current) { clearInterval(beatRef.current); beatRef.current = null; }
      return undefined;
    }
    beatRef.current = setInterval(async () => {
      try {
        const pr = await getJSON('/api/progress', { timeout: 5000 });
        setProgress(pr);
        reportNet(true);
      } catch { /* still down */ }
    }, 3000);
    return () => { if (beatRef.current) { clearInterval(beatRef.current); beatRef.current = null; } };
  }, [offline, reportNet]);

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
      // started_at is the backend's clock, so a run already in flight when this
      // view (re)loaded keeps its real elapsed time instead of restarting at 0.
      if (!startTime) setStartTime(progress.started_at ? progress.started_at * 1000 : Date.now());
    } else if (!progress.processing && was) {
      setStartTime(null);
    }
  }, [progress.processing, progress.started_at, startTime]);

  const prevProcessingCelebrationRef = useRef(false);
  useEffect(() => {
    const was = prevProcessingCelebrationRef.current;
    prevProcessingCelebrationRef.current = progress.processing;
    if (was && !progress.processing && !progress.error && (progress.progress || 0) >= 0.99) {
      playChime();
      // Backend /api/progress returns output as { path, kind } (no `name`), so
      // derive the filename from the path for the notification body.
      const outName = (progress.output?.path || '').split(/[\\/]/).pop();
      notifyDesktop('Swap complete', outName ? `${outName} is ready` : 'Your render is ready');
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
  // (The `?` shortcuts HUD is owned by the FaceSwap tab, which has the full
  // playback/timeline/queue shortcut list — no duplicate here.)
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

  // ── Theme resolution ─────────────────────────────────────────────────────
  // Declared above the command palette because that memo lists themes too.
  //
  // Track the OS light/dark preference. Nothing in the app used to read this,
  // so a machine that switches to light mode at sunset left the UI dark. It is
  // only ACTED on when the user opts into the pairing below, but it is always
  // tracked so flipping that switch takes effect immediately.
  const [systemDark, setSystemDark] = useState(
    () => !window.matchMedia?.('(prefers-color-scheme: light)')?.matches,
  );
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)');
    if (!mq) return undefined;
    const onChange = () => setSystemDark(mq.matches);
    onChange();
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  // Which theme is actually live: either the explicit pick, or — when the user
  // has paired a light and a dark theme — whichever half the OS is asking for.
  const themeName = settings?.theme_follow_system
    ? (systemDark ? (settings?.theme_dark || 'Default') : (settings?.theme_light || 'Glass Light'))
    : settings?.selected_theme;
  const customThemes = settings?.custom_themes;

  useEffect(() => {
    if (!themeName) return;
    // applyThemeToDom owns both mechanisms (a preset's class, a custom theme's
    // inline variables) and sets data-theme-mode for either — see themes.js.
    applyThemeToDom(themeByName(themeName, customThemes));
  }, [themeName, customThemes]);

  const applyTheme = useCallback((name) => {
    // Picking a theme by hand also leaves the system pairing: otherwise the
    // choice would be overridden on the next render and the click would look
    // like it did nothing.
    const patch = { selected_theme: name, theme_follow_system: false };
    setSettings((s) => ({ ...(s || {}), ...patch }));
    postJSON('/api/settings', patch).catch(() => {});
  }, []);

  // Faceswap-tab actions are decoupled via a window event bus so the palette
  // stays independent of the FaceSwap component's internal handlers. `extra`
  // carries arguments for the commands that need one (applying a named preset).
  const runFaceswap = useCallback((id, extra) => {
    setTab('faceswap');
    setTimeout(() => window.dispatchEvent(new CustomEvent('roop:command', { detail: { id, ...extra } })), 60);
  }, []);

  // Presets are mirrored into localStorage by useProfiles, so the palette can
  // list them without a fetch and without waiting on the Face Swap chunk. Read
  // once on mount; the `roop:presets-changed` ping keeps it current when that
  // tab saves or deletes one.
  const [presets, setPresets] = useState([]);
  useEffect(() => {
    const read = () => {
      try {
        const v = JSON.parse(localStorage.getItem('roop_profiles') || '[]');
        setPresets(Array.isArray(v) ? v.filter((pr) => pr && pr.name) : []);
      } catch { setPresets([]); }
    };
    read();
    window.addEventListener('roop:presets-changed', read);
    // `storage` only fires for OTHER documents, which covers the pop-out
    // preview window writing presets while this one is open.
    window.addEventListener('storage', read);
    return () => {
      window.removeEventListener('roop:presets-changed', read);
      window.removeEventListener('storage', read);
    };
  }, []);

  const commands = useMemo(() => {
    const cmds = [];
    // The tab's own icon and label carry straight into its palette row, so the
    // two surfaces can never drift apart (this used to slice the emoji off the
    // front of the label string and re-derive the title with a regex).
    TABS.forEach((t) => cmds.push({ id: `nav-${t.id}`, section: 'Navigate', icon: t.icon, title: `Go to ${t.label}`, run: () => { warmTab(t.id); setTab(t.id); } }));
    cmds.push({ id: 'act-start', section: 'Actions', icon: Icon.play, title: 'Start swapping', subtitle: 'Run the current job', run: () => runFaceswap('start') });
    cmds.push({ id: 'act-stop', section: 'Actions', icon: Icon.stop, title: 'Stop processing', run: () => runFaceswap('stop') });
    cmds.push({ id: 'act-queue', section: 'Actions', icon: Icon.queue, title: 'Add current to batch queue', run: () => runFaceswap('queue') });
    cmds.push({ id: 'act-compare', section: 'Actions', icon: Icon.compare, title: 'Toggle before/after compare', run: () => runFaceswap('compare') });
    cmds.push({ id: 'act-split', section: 'Actions', icon: Icon.split, title: 'Toggle split view', run: () => runFaceswap('split') });
    cmds.push({ id: 'act-preview', section: 'Actions', icon: Icon.refresh, title: 'Refresh preview', run: () => runFaceswap('preview') });
    cmds.push({ id: 'act-shortcuts', section: 'Actions', icon: Icon.shortcuts, title: 'Show keyboard shortcuts', run: () => runFaceswap('shortcuts') });
    // Custom themes are selectable from the palette too — they are themes, and
    // leaving them out would make the studio's output feel second-class.
    allThemes(customThemes).forEach((t) => cmds.push({
      id: `theme-${t.name}`, section: 'Theme', icon: Icon.theme,
      title: t.name, subtitle: t.custom ? 'Your theme' : t.label,
      run: () => applyTheme(t.name),
    }));

    // Every setting by name. The panel has its own search box, but you have to
    // be standing in the panel to use it — and half the reason to look a
    // setting up is that you are somewhere else and cannot remember which of
    // four columns it lives in. Selecting one opens Settings and flashes the
    // control itself.
    SETTINGS_CATALOG.forEach((s) => cmds.push({
      id: `setting-${s.key}`, section: 'Settings', icon: Icon.settings,
      title: s.label, subtitle: s.section,
      run: () => {
        warmTab('settings');
        setTab('settings');
        // The panel is a lazy chunk and may be mounting for the first time, so
        // give it a beat to subscribe before firing at it. Its own handler
        // waits two frames again for the filters to settle.
        setTimeout(() => focusSetting(s.key), 80);
      },
    }));

    // Saved presets, straight from the same localStorage the Face Swap tab
    // reads. No fetch: this memo runs on every settings change, and the list is
    // already mirrored locally precisely so it works offline.
    presets.forEach((pr) => cmds.push({
      id: `preset-${pr.name}`, section: 'Presets', icon: Icon.layout,
      title: pr.name, subtitle: 'Apply preset',
      run: () => runFaceswap('preset', { name: pr.name }),
    }));

    return cmds;
  }, [applyTheme, runFaceswap, customThemes, presets]);

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

  // Latest message mirrored into an aria-live region for screen readers.
  const [liveMsg, setLiveMsg] = useState('');
  const dismissToast = useCallback((id) => setToasts((ts) => ts.filter((t) => t.id !== id)), []);
  const notify = useCallback((message, type = 'success') => {
    const id = Date.now() + Math.random();
    // Stack toasts (newest wins visually) instead of clobbering — each dismisses
    // on its own timer, and the whole set is capped so a burst can't pile up.
    setToasts((ts) => [...ts, { id, message, type }].slice(-4));
    setLiveMsg(`${type}: ${message}`);
    setTimeout(() => setToasts((ts) => ts.filter((t) => t.id !== id)), 4000);
  }, []);

  // Core bootstrap (meta + settings + in-flight job), retryable. The backend is
  // often still binding its port when this webview first paints — especially on
  // a cold Pinokio start — so a first failure schedules its own backoff retry
  // instead of parking the user on a dead-end error screen.
  const [retrying, setRetrying] = useState(false);
  const bootAttemptRef = useRef(0);
  const bootTimerRef = useRef(null);
  const loadCore = useCallback(async () => {
    setRetrying(true);
    try {
      const [m, s] = await Promise.all([
        getJSON('/api/meta', { timeout: 10000 }),
        getJSON('/api/settings', { timeout: 10000 }),
      ]);
      setMeta(m);
      // Mark the incoming value as "just fetched" so the autosave effect below
      // skips it — otherwise a retry would immediately POST back the settings
      // we only just read.
      settingsLoadedRef.current = false;
      setSettings(s);
      setError('');
      reportNet(true);
      bootAttemptRef.current = 0;
      try {
        const pr = await getJSON('/api/progress', { timeout: 8000 });
        setProgress(pr);
        if (pr.processing) startPolling();
      } catch { /* progress is non-critical for boot */ }
    } catch {
      setError('Cannot reach backend on 127.0.0.1:8001. Make sure the server (run.py) is running.');
      // 1s, 2s, 4s … capped at 8s, forever — a launcher window left open should
      // heal itself the moment the server comes up.
      const wait = Math.min(8000, 1000 * 2 ** bootAttemptRef.current++);
      if (bootTimerRef.current) clearTimeout(bootTimerRef.current);
      bootTimerRef.current = setTimeout(() => loadCore(), wait);
    } finally {
      setRetrying(false);
    }
  }, [reportNet, startPolling]);

  useEffect(() => {
    loadCore();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (bootTimerRef.current) clearTimeout(bootTimerRef.current);
    };
  }, [loadCore]);

  // Warm the remaining tab chunks once the app is idle, so the very first visit
  // to any tab is instant even without a hover. requestIdleCallback keeps this
  // off the critical path; Safari/older webviews fall back to a timeout.
  useEffect(() => {
    if (!meta) return undefined;
    const run = () => TABS.forEach((t) => warmTab(t.id));
    const ric = window.requestIdleCallback;
    if (ric) {
      const h = ric(run, { timeout: 3000 });
      return () => window.cancelIdleCallback?.(h);
    }
    const h = setTimeout(run, 1500);
    return () => clearTimeout(h);
  }, [meta]);

  // A tab change swaps the entire view; keeping the old scroll offset drops the
  // user into the middle of a panel they have never seen. Snap to the top as
  // the outgoing view fades so the incoming one always starts at its header.
  const firstTabRenderRef = useRef(true);
  useEffect(() => {
    if (firstTabRenderRef.current) { firstTabRenderRef.current = false; return; }
    if (window.scrollY > 4) window.scrollTo({ top: 0, behavior: 'auto' });
  }, [tab]);

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
          <span className="grid place-items-center h-9 w-9 rounded-xl bg-[var(--accent)]/12 border border-[var(--accent)]/25 text-[var(--accent)]">
            <Icon.brand size={18} />
          </span>
          <div>
            <h1 className="text-lead font-bold tracking-tight text-white/95 flex items-center gap-1.5">
              Roop Unleashed <span className="text-white/35 font-medium">Studio</span>
            </h1>
            {meta?.git_version && (
              <span className="text-nano font-mono text-white/35 tracking-wider block mt-0.5">
                Engine {meta.git_version}
              </span>
            )}
          </div>
          {progress.processing && (
            <div className="ml-2 flex items-center gap-2 px-2.5 py-1 rounded-lg bg-[var(--accent)]/10 border border-[var(--accent)]/20 text-micro font-bold tracking-wide uppercase">
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
                    className="grid place-items-center hover:text-white text-white/60 transition-colors cursor-pointer"
                    title="Resume Job" aria-label="Resume job"
                  >
                    <Icon.play size={13} />
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
                    className="grid place-items-center hover:text-white text-white/60 transition-colors cursor-pointer"
                    title="Pause Job" aria-label="Pause job"
                  >
                    <Icon.pause size={13} />
                  </button>
                )}
                <button
                  type="button"
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (await confirmDialog({ title: 'Stop job?', message: 'Stop the active job? The partial output so far is finalized and kept.', confirmLabel: 'Stop', danger: true })) {
                      try {
                        await postJSON('/api/stop', {});
                      } catch {}
                    }
                  }}
                  className="grid place-items-center hover:text-red-400 text-white/60 transition-colors cursor-pointer"
                  title="Stop Job" aria-label="Stop job"
                >
                  <Icon.stop size={13} />
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
          <Icon.search size={14} /> Search
          <kbd className="text-nano font-mono bg-white/5 px-1.5 py-0.5 rounded border border-white/10">Ctrl K</kbd>
        </button>
        <div className="hidden md:flex items-center gap-0.5 px-1 py-1 rounded-xl bg-white/[0.03] border border-white/10" title="UI zoom (Ctrl + / − / 0)">
          <button type="button" onClick={() => bumpZoom(-0.05)} title="Zoom out (Ctrl −)" aria-label="Zoom out" className="h-6 w-6 grid place-items-center rounded-lg text-white/50 hover:text-white hover:bg-white/10 text-base leading-none transition-colors">−</button>
          <button type="button" onClick={() => setZoom(1)} title="Reset zoom (Ctrl 0)" aria-label="Reset zoom" className="min-w-[44px] text-mini font-semibold text-white/60 hover:text-white tabular-nums transition-colors">{Math.round(zoom * 100)}%</button>
          <button type="button" onClick={() => bumpZoom(0.05)} title="Zoom in (Ctrl +)" aria-label="Zoom in" className="h-6 w-6 grid place-items-center rounded-lg text-white/50 hover:text-white hover:bg-white/10 text-base leading-none transition-colors">+</button>
        </div>
        <nav className="flex gap-0.5 bg-black/25 p-1 rounded-xl border border-white/[0.06] w-full md:w-auto overflow-x-auto">
          {TABS.map((t) => {
            const active = tab === t.id;
            return (
              <motion.button
                key={t.id}
                onClick={() => setTab(t.id)}
                // Start fetching the panel's chunk the instant the pointer or
                // keyboard focus lands on the tab — by the time the click
                // registers the module is usually already parsed, so the view
                // swap is a pure animation with no loading state in between.
                onPointerEnter={() => warmTab(t.id)}
                onFocus={() => warmTab(t.id)}
                aria-current={active ? 'page' : undefined}
                whileTap={{ scale: 0.94 }}
                transition={spring.snappy}
                className={`relative px-3.5 py-2 rounded-lg text-note font-semibold tracking-wide whitespace-nowrap flex items-center gap-1.5 transition-colors duration-200 ${
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
                {/* The icon takes the accent only while the tab is active, so
                    the selected tab is legible from colour and from the pill
                    behind it, not from colour alone. */}
                <span className="relative z-10 flex items-center gap-1.5">
                  <t.icon size={14} className={active ? 'text-[var(--accent)]' : undefined} />
                  {t.label}
                </span>
              </motion.button>
            );
          })}
        </nav>
        </div>
      </header>

      {/* Main Container Layout */}
      <main className="flex-1 w-[98%] max-w-none mx-auto px-6 py-8 mt-4 z-10 relative">
        {error && (
          <div role="alert" className="rounded-2xl bg-red-500/10 border border-red-500/20 p-5 text-sm text-red-300 animate-slide-up flex flex-wrap items-center justify-between gap-3">
            {/* items-start: on a wrapped multi-line error the icon belongs on
                the first line, not floating in the vertical middle. */}
            <span className="flex items-start gap-2 min-w-0">
              <Icon.warning size={16} className="mt-px text-red-400" />
              <span className="selectable">{error}</span>
            </span>
            <div className="flex items-center gap-2 shrink-0">
              <span className="text-mini text-red-300/50">
                {retrying ? 'Retrying…' : 'Retrying automatically…'}
              </span>
              <button
                type="button"
                onClick={() => { bootAttemptRef.current = 0; loadCore(); }}
                disabled={retrying}
                className="px-3 py-1.5 rounded-lg bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 text-mini font-semibold text-red-200 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Retry now
              </button>
            </div>
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
              <ErrorBoundary resetKey={tab}>
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
                {tab === 'history' && <RunHistory notify={notify} setSettings={setSettings} setTab={setTab} />}
                {tab === 'settings' && <Settings meta={meta} settings={settings} setSettings={setSettings} notify={notify} />}
              </Suspense>
              </ErrorBoundary>
            </motion.div>
          </AnimatePresence>
        )}
      </main>

      <Toasts toasts={toasts} onDismiss={dismissToast} />

      {/* Backend went quiet mid-session (server restart, GPU stall, sleep).
          Non-blocking: the UI stays usable and this clears itself the moment a
          heartbeat lands, so a brief hiccup never forces a manual reload. */}
      <AnimatePresence>
        {offline && !error && (
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.96 }}
            transition={spring.snappy}
            role="status"
            className="fixed bottom-6 left-6 z-50 px-4 py-3 rounded-xl bg-[#0E0F15]/95 backdrop-blur-xl border border-amber-400/25 shadow-2xl flex items-center gap-3"
          >
            <span className="h-2 w-2 rounded-full bg-amber-400 animate-ping" />
            <span className="text-xs font-semibold text-amber-200/90">Reconnecting to the engine…</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Screen-reader status channel: toast messages + live processing state,
          announced politely without stealing focus. */}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">{liveMsg}</div>
      <div className="sr-only" role="status" aria-live="polite">
        {progress.processing ? `${progress.paused ? 'Paused' : 'Processing'} ${Math.round((progress.progress || 0) * 100)} percent` : ''}
      </div>

      <Confetti active={confetti} />

      {progress.processing && (
        <div
          className="fixed top-0 left-0 h-[3px] bg-[var(--accent)] z-[60] transition-all duration-300 shadow-[0_0_8px_var(--accent-glow)]"
          style={{ width: `${Math.round((progress.progress || 0) * 100)}%` }}
        />
      )}

      <CommandPalette open={showPalette} onClose={() => setShowPalette(false)} commands={commands} />

      <ConfirmHost />

      {isDraggingOver && (
        <div className="fixed inset-0 bg-[var(--accent)]/10 backdrop-blur-[2px] border-4 border-dashed border-[var(--accent)] z-50 flex items-center justify-center pointer-events-none">
          <div className="bg-black/80 px-6 py-4 rounded-2xl border border-white/10 shadow-2xl flex flex-col items-center gap-2">
            {/* animate-bounce is a transform, so it stays on the compositor —
                this overlay is on screen during a drag and must not repaint. */}
            <Icon.drop size={34} className="animate-bounce text-[var(--accent)]" />
            <span className="text-lg font-bold text-white uppercase tracking-wider">Drop media here</span>
          </div>
        </div>
      )}
    </div>
    </MotionConfig>
  );
}
