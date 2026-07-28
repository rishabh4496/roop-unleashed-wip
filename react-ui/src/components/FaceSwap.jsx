import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { getJSON, postJSON, postFiles, API } from '../api';
import { Section, Select, Slider, Toggle, TextInput, Button, FaceGallery, Card, AnimatedNumber, Skeleton } from './ui';
import PersonGroups from './PersonGroups';
import QualityReport from './QualityReport';
import FileDrop from './faceswap/FileDrop';
import CompareGrid from './faceswap/CompareGrid';
import InteractivePreview from './faceswap/InteractivePreview';
import Timeline from './faceswap/Timeline';
import FacesetLibrary from './faceswap/FacesetLibrary';
import ProcessingTerminal from './faceswap/ProcessingTerminal';
import DiagnosticsPanel from './faceswap/DiagnosticsPanel';
import { num, fmtTime } from './faceswap/utils';
import useProfiles from './faceswap/useProfiles';
import useTelemetry from './faceswap/useTelemetry';
import { FACESWAP_DEFAULTS } from './faceswap/defaults';
import { motion, spring, TiltCard } from '../motion';

// AI upscale models folded into the swap pass (mirrors the Extras post-processor
// list). value = backend subtype, label = friendly name.
const AI_UPSCALE_MODELS = [
  { value: 'esrganx2', label: 'Real-ESRGAN ×2' },
  { value: 'esrganx4', label: 'Real-ESRGAN ×4' },
  { value: 'esrgan_anime_x4', label: 'Real-ESRGAN Anime ×4' },
  { value: 'ultrasharp_x4', label: 'Ultra-Sharp ×4' },
  { value: 'lsdirx4', label: 'LSDIR ×4' },
  { value: 'clear_reality_x4', label: 'Clear Reality ×4' },
  { value: 'span_x4', label: 'SPAN ×4' },
  { value: 'compact_x4', label: 'Compact ×4 (fast AI)' },
  { value: 'nomos8k_x4', label: 'Nomos8k ×4' },
  { value: 'lanczos_x2', label: '⚡ Fast Lanczos ×2 (no AI)' },
  { value: 'lanczos_x4', label: '⚡ Fast Lanczos ×4 (no AI)' },
  { value: 'fsr_x2', label: '⚡ FSR-lite ×2 · Lanczos+CAS (no AI)' },
  { value: 'fsr_x4', label: '⚡ FSR-lite ×4 · Lanczos+CAS (no AI)' },
  { value: 'spline_x2', label: '⚡ Spline36 ×2 (no AI)' },
  { value: 'spline_x4', label: '⚡ Spline36 ×4 (no AI)' },
  { value: 'sinc_x2', label: '⚡ Sinc ×2 · sharpest (no AI)' },
  { value: 'sinc_x4', label: '⚡ Sinc ×4 · sharpest (no AI)' },
];

export default function FaceSwap({
  meta, settings, setSettings, notify, registerFileListener,
  progress, setProgress, startTime, setStartTime
}) {
  const [sourceFaces, setSourceFaces] = useState([]);
  const [sourceFacesInfo, setSourceFacesInfo] = useState([]);
  const [targetFaces, setTargetFaces] = useState([]);
  const [targetGroups, setTargetGroups] = useState([]);
  const [targetNames, setTargetNames] = useState([]);
  const [targetFacesInfo, setTargetFacesInfo] = useState([]);
  const [targets, setTargets] = useState([]);
  const [selSource, setSelSource] = useState(0);
  const [selTarget, setSelTarget] = useState(0);
  const [selTargetFace, setSelTargetFace] = useState(0);
  const [frame, setFrame] = useState(1);
  const [maxFrames, setMaxFrames] = useState(1);
  const [previewSrc, setPreviewSrc] = useState('');
  const [previewFaces, setPreviewFaces] = useState([]);
  const [previewPersonIds, setPreviewPersonIds] = useState([]);
  // Single-frame AI-upscale spot-check (result shown in a modal).
  const [upscaledSrc, setUpscaledSrc] = useState('');
  const [upscaledDims, setUpscaledDims] = useState(null);
  const [upscaling, setUpscaling] = useState(false);
  const [fakePreview, setFakePreview] = useState(true);
  const [uploadingSrc, setUploadingSrc] = useState(false);
  const [uploadingTgt, setUploadingTgt] = useState(false);

  const [previewing, setPreviewing] = useState(false);
  const [previewSecs, setPreviewSecs] = useState(0);
  const [compare, setCompare] = useState(false);
  const [splitView, setSplitView] = useState(false);
  const [comparingEnhancers, setComparingEnhancers] = useState(false);
  const [selectedGridEnhancers, setSelectedGridEnhancers] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('roop_grid_enhancers') || 'null');
      if (Array.isArray(saved) && saved.length >= 1 && saved.length <= 4 && saved.every(x => typeof x === 'string')) {
        return saved;
      }
    } catch { /* fall through to default */ }
    return ['None', 'GPEN', 'Restoreformer++', 'GFPGAN'];
  });
  useEffect(() => {
    localStorage.setItem('roop_grid_enhancers', JSON.stringify(selectedGridEnhancers));
  }, [selectedGridEnhancers]);
  const [enhancerPreviews, setEnhancerPreviews] = useState({});
  const [enhancerTimes, setEnhancerTimes] = useState({});
  const [liveRenderingTimers, setLiveRenderingTimers] = useState({});
  const activeIntervalsRef = useRef({});

  // ── Mask-engine comparison grid (mirrors the enhancer grid) ──
  const [comparingMasks, setComparingMasks] = useState(false);
  const [selectedGridMasks, setSelectedGridMasks] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('roop_grid_masks') || 'null');
      if (Array.isArray(saved) && saved.length >= 1 && saved.length <= 4 && saved.every(x => typeof x === 'string')) {
        return saved;
      }
    } catch { /* fall through to default */ }
    return ['None', 'DFL XSeg', 'Face Occluder', 'Face Parser (BiSeNet)'];
  });
  useEffect(() => {
    localStorage.setItem('roop_grid_masks', JSON.stringify(selectedGridMasks));
  }, [selectedGridMasks]);
  const [maskPreviews, setMaskPreviews] = useState({});
  const [maskTimes, setMaskTimes] = useState({});
  const [maskRenderTimers, setMaskRenderTimers] = useState({});
  const maskIntervalsRef = useRef({});

  // ── Swapper-model comparison grid (mirrors the enhancer/mask grid) ──
  const [comparingSwappers, setComparingSwappers] = useState(false);
  const [selectedGridSwappers, setSelectedGridSwappers] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('roop_grid_swappers') || 'null');
      if (Array.isArray(saved) && saved.length >= 1 && saved.length <= 4 && saved.every(x => typeof x === 'string')) {
        return saved;
      }
    } catch { /* fall through to default */ }
    return ['inswapper', 'reswapper', 'hyperswap', 'simswap'];
  });
  useEffect(() => {
    localStorage.setItem('roop_grid_swappers', JSON.stringify(selectedGridSwappers));
  }, [selectedGridSwappers]);
  const [swapperPreviews, setSwapperPreviews] = useState({});
  const [swapperTimes, setSwapperTimes] = useState({});
  const [swapperRenderTimers, setSwapperRenderTimers] = useState({});
  const swapperIntervalsRef = useRef({});

  // ── AI-upscale comparison grid (mirrors the enhancer/mask/swapper grid) ──
  // Keyed by the friendly MODEL LABEL (e.g. "Real-ESRGAN ×2"), which is also
  // the caption CompareGrid shows; label→subtype is resolved when calling the
  // backend. Each cell swaps the frame ONCE then upscales it with one model.
  const [comparingUpscalers, setComparingUpscalers] = useState(false);
  const [selectedGridUpscalers, setSelectedGridUpscalers] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('roop_grid_upscalers') || 'null');
      const valid = AI_UPSCALE_MODELS.map(m => m.label);
      if (Array.isArray(saved) && saved.length >= 1 && saved.length <= 4 && saved.every(x => valid.includes(x))) {
        return saved;
      }
    } catch { /* fall through to default */ }
    return AI_UPSCALE_MODELS.slice(0, 2).map(m => m.label);
  });
  useEffect(() => {
    localStorage.setItem('roop_grid_upscalers', JSON.stringify(selectedGridUpscalers));
  }, [selectedGridUpscalers]);
  const [upscalePreviews, setUpscalePreviews] = useState({});
  const [upscaleTimes, setUpscaleTimes] = useState({});
  const [upscaleRenderTimers, setUpscaleRenderTimers] = useState({});
  const upscaleIntervalsRef = useRef({});

  // Telemetry HUD — GPU/VRAM/CPU/RAM/threads poller (see faceswap/useTelemetry).
  const telemetry = useTelemetry();

  // Target-to-Source visual mapping state
  const [faceMapping, setFaceMapping] = useState({});

  const getFaceMappingArray = () => {
    const uniqPersons = Array.from(new Set(targetGroups))
      .filter(x => typeof x === 'number')
      .sort((a, b) => a - b);
    return uniqPersons.map(pId => {
      const mappedSrc = faceMapping[pId];
      return mappedSrc !== undefined ? mappedSrc : pId;
    });
  };

  // Profile Management — named setting presets (see faceswap/useProfiles).
  const {
    profiles, newProfileName, setNewProfileName,
    saveProfile, loadProfile, deleteProfile, exportProfiles, importProfiles,
  } = useProfiles({ settings, setSettings, notify });

  // Keyboard Shortcuts Modal visibility
  const [showShortcutHUD, setShowShortcutHUD] = useState(false);

  // Drag and drop overlay state removed (managed globally by App.jsx)

  // Batch Queue Manager
  const [queue, setQueue] = useState([]);
  const [currentQueueIndex, setCurrentQueueIndex] = useState(null);
  const [isQueueRunning, setIsQueueRunning] = useState(false);
  const [queuePaused, setQueuePaused] = useState(false);

  // Pasted Files Dialog State
  const [pastedFiles, setPastedFiles] = useState(null);

  // Preview Cache Ref
  const previewCacheRef = useRef({});
  // Set when a preview refresh is skipped because a swap run owns the GPU.
  const previewDeferredRef = useRef(false);

  const clearPreviewCache = () => {
    previewCacheRef.current = {};
  };

  // Content signature of a face gallery. The count alone is NOT enough to
  // identify which faces are loaded: uploading a new faceset and then deleting
  // the old one leaves the count unchanged, so a length-keyed preview cache kept
  // serving — and a length-keyed refresh effect kept skipping — the *previous*
  // face. Hashing the thumbnail data URLs (size + tail, cheap) also catches
  // reorders and in-place replacements.
  const gallerySig = (faces) =>
    `${faces.length}:` + faces.map((f) => (f ? `${f.length}.${f.slice(-24)}` : '-')).join('|');
  const sourceSig = useMemo(() => gallerySig(sourceFaces), [sourceFaces]);
  const targetSig = useMemo(() => gallerySig(targetFaces), [targetFaces]);

  const getCacheKey = (idx = selTarget, fr = frame) => {
    return `${idx}_${fr}_${previewKey}_${sourceSig}_${targetSig}_${selSource}_${selTargetFace}`;
  };

  const getCachedPreview = (idx = selTarget, fr = frame) => {
    const key = getCacheKey(idx, fr);
    return previewCacheRef.current[key];
  };

  const setCachedPreview = (idx, fr, data) => {
    const key = getCacheKey(idx, fr);
    const cache = previewCacheRef.current;
    const keys = Object.keys(cache);
    if (keys.length > 200) {
      delete cache[keys[0]]; // cap at 200 items to prevent memory bloat
    }
    cache[key] = data;
  };

  // Invalidate cache when source/target faces or selections change
  useEffect(() => {
    clearPreviewCache();
  }, [sourceSig, targetSig, selSource, selTargetFace]);

  // ── Shareable "recipe": full current settings + person→source mapping ──
  const exportRecipe = () => {
    try {
      const recipe = {
        type: 'roop-recipe',
        version: 1,
        exported_at: new Date().toISOString(),
        settings: { ...p },
        face_mapping: getFaceMappingArray(),
      };
      const blob = new Blob([JSON.stringify(recipe, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `roop_recipe_${Date.now()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      notify('Recipe exported — share the .json to reproduce this exact setup');
    } catch (e) { notify('Failed to export recipe: ' + e.message, 'error'); }
  };

  const importRecipe = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const r = JSON.parse(event.target.result);
        if (r.type !== 'roop-recipe' || !r.settings) throw new Error('Not a valid roop recipe file.');
        setSettings((s) => ({ ...s, ...r.settings }));
        if (Array.isArray(r.face_mapping)) {
          const fm = {};
          r.face_mapping.forEach((v, i) => { fm[i] = v; });
          setFaceMapping(fm);
        }
        clearPreviewCache();
        notify('Recipe applied — settings and mapping restored');
      } catch (err) {
        notify('Failed to import recipe: ' + err.message, 'error');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  // Register clipboard & drop listener with global App registry
  useEffect(() => {
    if (!registerFileListener) return;
    return registerFileListener((files) => {
      setPastedFiles(files);
      return true; // consumed
    });
  }, [registerFileListener]);

  // Add current target, source, and settings to the queue
  const addToQueue = () => {
    if (targets.length === 0) {
      notify('Load target media first', 'error');
      return;
    }
    const newJob = {
      id: Date.now() + Math.random().toString(36).substr(2, 9),
      targetIndex: selTarget,
      targetName: targets[selTarget]?.name || 'Unknown',
      sourceIndex: selSource,
      sourceName: sourceFaces[selSource] ? `Face ${selSource + 1}` : 'Selected Face',
      params: { ...p },
      faceMapping: getFaceMappingArray(),
      status: 'Pending'
    };
    setQueue((prev) => [...prev, newJob]);
    notify(`Added "${newJob.targetName}" to Batch Queue`);
  };

  const removeFromQueue = (id) => {
    setQueue((prev) => prev.filter(job => job.id !== id));
  };

  const clearQueue = () => {
    setQueue([]);
    setIsQueueRunning(false);
    setQueuePaused(false);
    setCurrentQueueIndex(null);
  };

  const startQueue = async () => {
    if (queue.length === 0) {
      notify('Queue is empty', 'error');
      return;
    }
    setQueuePaused(false);
    setIsQueueRunning(true);
    setCurrentQueueIndex(0);
  };

  // Pause the whole queue: pause the job that's currently swapping (backend
  // honours roop_globals.pause in ProcessMgr) and hold off dispatching the next
  // job until resumed. The runner effect below bails out while queuePaused.
  const pauseQueue = async () => {
    setQueuePaused(true);
    try {
      if (progress.processing && !progress.paused) {
        await postJSON('/api/pause', {});
        setProgress((pr) => ({ ...pr, paused: true, desc: 'Paused' }));
      }
      notify('Queue paused', 'info');
    } catch (e) { notify(e.message, 'error'); }
  };

  const resumeQueue = async () => {
    setQueuePaused(false);
    try {
      if (progress.processing && progress.paused) {
        await postJSON('/api/resume', {});
        setProgress((pr) => ({ ...pr, paused: false, desc: 'Resuming…' }));
      }
      notify('Queue resumed');
    } catch (e) { notify(e.message, 'error'); }
  };

  // Stop the queue: halt further dispatch and never leave the current job frozen
  // in a paused state (resume the backend so it can finish under run-bar control).
  const stopQueue = async () => {
    setIsQueueRunning(false);
    setQueuePaused(false);
    if (progress.processing && progress.paused) {
      try { await postJSON('/api/resume', {}); } catch { /* ignore */ }
      setProgress((pr) => ({ ...pr, paused: false }));
    }
  };

  // Queue Runner State Machine
  /* eslint-disable react-hooks/exhaustive-deps -- intentional: queue state machine reads latest state each tick without re-subscribing on every dep change */
  useEffect(() => {
    if (!isQueueRunning || currentQueueIndex === null) return;
    // Hold dispatch while the queue is paused — don't start the next job.
    if (queuePaused) return;

    if (currentQueueIndex >= queue.length) {
      setIsQueueRunning(false);
      setCurrentQueueIndex(null);
      notify('All batch queue jobs finished!', 'success');
      return;
    }

    const job = queue[currentQueueIndex];
    // Skip jobs that already reached a terminal state (e.g. re-running a queue
    // with some Finished/Failed items). A 'Running' job — which is what we see
    // when this effect re-fires on resume — must NOT be re-dispatched or skipped;
    // the progress-monitor effect below advances it once it truly finishes.
    if (job.status === 'Finished' || job.status === 'Failed') {
      setCurrentQueueIndex((idx) => idx + 1);
      return;
    }
    if (job.status === 'Running') return;

    const executeJob = async () => {
      setQueue((prev) => prev.map(j => j.id === job.id ? { ...j, status: 'Running' } : j));

      // Resolve the target by NAME at run time — stored indices go stale when
      // targets are removed after the job was queued, which made a queued job
      // silently swap the wrong file.
      const resolvedIndex = targets.findIndex(t => t.name === job.targetName);
      if (resolvedIndex === -1) {
        notify(`Job "${job.targetName}" skipped: target no longer loaded`, 'error');
        setQueue((prev) => prev.map(j => j.id === job.id ? { ...j, status: 'Failed' } : j));
        setCurrentQueueIndex((idx) => idx + 1);
        return;
      }

      try {
        await postJSON('/api/target/select', { index: resolvedIndex });
        setSelTarget(resolvedIndex);
        
        await postJSON('/api/source/select', { index: job.sourceIndex });
        setSelSource(job.sourceIndex);

        await postJSON('/api/settings', job.params);
        setSettings(job.params);
        
        await postJSON('/api/swap', {
          ...job.params,
          enhancer: job.params.selected_enhancer,
          detection: job.params.face_detection_mode,
          output_method: job.params.output_method,
          video_method: job.params.video_swapping_method,
          upscale: job.params.subsample_upscale,
          mask_engine: job.params.mask_engine,
          clip_text: job.params.mask_clip_text,
          sam2_model_size: job.params.sam2_model_size,
          track_identities: job.params.track_identities,
          autorotate: job.params.autorotate_faces,
          face_distance: num(job.params.max_face_distance, 0.85),
          blend_ratio: num(job.params.blend_ratio, 0.8),
          num_swap_steps: num(job.params.num_swap_steps, 1),
          face_mapping: job.faceMapping || [],
          target_index: resolvedIndex,
        });

        setStartTime(Date.now());
        setProgress({ processing: true, paused: false, progress: 0, desc: 'Starting queue job…', output: null });
      } catch (e) {
        notify(`Job "${job.targetName}" failed to start: ${e.message}`, 'error');
        setQueue((prev) => prev.map(j => j.id === job.id ? { ...j, status: 'Failed' } : j));
        setCurrentQueueIndex((idx) => idx + 1);
      }
    };
    executeJob();
  }, [isQueueRunning, currentQueueIndex, queuePaused]);

  // Monitor job progress to move to next queue item
  useEffect(() => {
    if (!isQueueRunning || currentQueueIndex === null) return;
    const job = queue[currentQueueIndex];
    if (!job || job.status !== 'Running') return;

    if (!progress.processing) {
      const isFailed = progress.error || (progress.desc && progress.desc.toLowerCase().includes('fail'));
      setQueue((prev) => prev.map(j => j.id === job.id ? { ...j, status: isFailed ? 'Failed' : 'Finished' } : j));
      setCurrentQueueIndex((idx) => idx + 1);
    }
  }, [progress.processing]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // Custom Timeline and Playback States
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(1); // 0.25 | 0.5 | 1 | 2 | 4
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [dragType, setDragType] = useState('playhead'); // 'playhead', 'start', 'end'
  const [storyboardThumbs, setStoryboardThumbs] = useState([]);
  const [hoverFrame, setHoverFrame] = useState(null);
  // Timeline view window (inclusive frame range the track currently shows).
  // On a 95k-frame clip the full-width track puts ~100 frames in every pixel,
  // so it can only ever be a rough locator; zooming makes it a working surface.
  // Every frame<->pixel conversion below goes through this, and it is read via
  // a ref so a zoom during a drag can't tear down the scrub listeners.
  const [view, setView] = useState({ start: 1, end: 1 });
  const viewRef = useRef(view);
  viewRef.current = view;
  const timelineRef = useRef(null);
  // Coalesces timeline hover/scrub pointer-move work to one update per frame.
  // Pointer-move fires faster than this large component can re-render, so
  // without batching the events pile up and scrubbing/hovering feels sticky.
  const timelineRafRef = useRef(null);
  const timelinePendingRef = useRef(null);
  // Buffered, video-style playback (see the playback effect below): a rolling
  // window of upcoming frames decoded into blob URLs so the playhead can run at
  // real time instead of refetching each frame from the server per tick.
  const playBufRef = useRef(new Map());     // frame -> object URL
  const playFetchRef = useRef(new Set());   // frames currently in flight
  const playBufIdxRef = useRef('');         // signature of the video the buffer was built for
  // Latest `targets`, readable from inside the playback rAF loop, which must not
  // re-bind on it: every In/Out drag and unrelated target-state update would
  // otherwise tear playback down and restart it.
  const targetsRef = useRef([]);
  // Seek plumbing. The playback loop owns the playhead and writes `frame` itself
  // every time it paints, so it cannot simply watch `frame` for seeks — it would
  // see its own writes. `playWroteRef` is the last value the LOOP wrote (set
  // synchronously at write time, so it is already current when the resulting
  // commit runs its effects); anything else arriving in `frame` therefore came
  // from outside — a timeline drag, the frame box, an arrow key, a step/jump
  // button — and is handed to the loop through `playSeekRef`.
  const playWroteRef = useRef(null);
  const playSeekRef = useRef(null);
  const [bufferedSrc, setBufferedSrc] = useState(null);
  // True while the playhead is held on a frame that hasn't arrived yet.
  const [playStalled, setPlayStalled] = useState(false);
  const stalledRef = useRef(false);
  const [isGeneratingPreviewClip, setIsGeneratingPreviewClip] = useState(false);
  const [origStartEnd, setOrigStartEnd] = useState(null);


  const previewBusyRef = useRef(false);   // a /api/preview call is in flight
  const previewPendingRef = useRef(null); // latest queued request while busy (coalesced)

  // p = the swap parameters, seeded from CFG (settings) and patched locally.
  const p = settings || {};
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));

  // The knobs that actually decide a run's speed and look, shown alongside the
  // live diagnostics so a screenshot of a slow or wrong-looking run says what
  // produced it — otherwise the numbers have to be paired with the settings tab
  // from memory. Only the ones with real cost/quality weight; off is shown as
  // "off" rather than hidden, because an unexpectedly disabled stage is itself
  // the answer often enough.
  const runConfigSummary = useMemo(() => ([
    ['swapper', p.swap_model || '—'],
    ['detector', p.detector_engine || '—'],
    ['enhancer', p.selected_enhancer && p.selected_enhancer !== 'None' ? p.selected_enhancer : 'off'],
    ['mask', p.mask_engine || '—'],
    ['pixel boost', p.subsample_upscale || '—'],
    ['upscale', p.upscale_after_swap ? (p.upscale_model_after || 'on') : 'off'],
    ['tracking', p.track_identities ? 'on' : 'off'],
    ['temporal', p.temporal_detection ? 'on' : 'off'],
    ['stabilize', p.stabilize_face || p.stabilize_enhancer ? 'on' : 'off'],
  ]), [p.swap_model, p.detector_engine, p.selected_enhancer, p.mask_engine,
       p.subsample_upscale, p.upscale_after_swap, p.upscale_model_after,
       p.track_identities, p.temporal_detection, p.stabilize_face, p.stabilize_enhancer]);

  // While face-swap preview is on, auto-refresh when the swapped result would
  // change: new source faces, target faces, or any swap/mask parameter.
  // ── The one place UI params become an /api/preview request ──────────────
  // Both the request body and the cache/refresh signature are built from this
  // single function, and the signature is literally the request body with the
  // frame coordinates zeroed. That makes the failure this replaces structurally
  // impossible: a setting cannot be sent to the backend without also
  // invalidating the cache (a control that silently does nothing), nor keyed
  // without being sent. There used to be six hand-maintained copies of this
  // list -- and they had already drifted: the three comparison-grid keys were
  // missing sam2_model_size, face_detector_nms and face_mapping, so those grids
  // served stale cells when any of the three changed.
  //
  // `params` is the settings object to read from: `p` normally, or a grid's
  // `localParams` (p with that grid's one override applied), which reproduces
  // the override in the request for free.
  const buildPreviewPayload = (params, { index, frame: fr, fake, ...overrides } = {}) => ({
    index, frame: fr, fake_preview: fake,
    enhancer: params.selected_enhancer, codeformer_fidelity: num(params.codeformer_fidelity, 0.5),
    detection: params.face_detection_mode,
    face_distance: num(params.max_face_distance, 0.85), blend_ratio: num(params.blend_ratio, 0.8),
    mask_engine: params.mask_engine, clip_text: params.mask_clip_text,
    no_face_action: params.no_face_action, vr_mode: params.vr_mode, autorotate: params.autorotate_faces,
    show_mask_offsets: params.show_mask_offsets, restore_original_mouth: params.restore_original_mouth,
    num_swap_steps: num(params.num_swap_steps, 1), upscale: params.subsample_upscale,
    use_3d_recon: params.use_3d_recon, use_source_bank: params.use_source_bank,
    use_frontalization: params.use_frontalization, frontalization_threshold: num(params.frontalization_threshold, 30),
    jaw_reshape: params.jaw_reshape, jaw_reshape_strength: num(params.jaw_reshape_strength, 0.5),
    detail_transfer_strength: num(params.detail_transfer_strength, 0),
    swap_model: params.swap_model, default_det_size: params.default_det_size,
    face_detector_size: params.face_detector_size, face_detector_threshold: params.face_detector_threshold,
    face_detector_nms: params.face_detector_nms,
    color_transfer_mode: params.color_transfer_mode, sam2_model_size: params.sam2_model_size,
    refine_landmarks: params.refine_landmarks, yaw_align: params.yaw_align,
    rescue_small_faces: params.rescue_small_faces,
    detector_engine: params.detector_engine,
    face_mapping: getFaceMappingArray(),
    mask_top: params.mask_top,
    mask_bottom: params.mask_bottom,
    mask_left: params.mask_left,
    mask_right: params.mask_right,
    face_mask_blend: params.face_mask_blend,
    mouth_mask_blend: params.mouth_mask_blend,
    mouth_top_scale: params.mouth_top_scale,
    mouth_bottom_scale: params.mouth_bottom_scale,
    mouth_left_scale: params.mouth_left_scale,
    mouth_right_scale: params.mouth_right_scale,
    ...overrides,
  });

  // index/frame are separate cache dimensions, so they are zeroed here rather
  // than being part of the settings signature.
  const previewSignature = (params, fake) =>
    JSON.stringify(buildPreviewPayload(params, { index: 0, frame: 0, fake }));

  const previewKey = previewSignature(p, fakePreview);

  // One-click speed/quality profiles. Each bundles the core levers (detection
  // resolution, pixel-boost upscale, enhancer, swap steps); other settings (mask
  // engine, target selection, etc.) are left as-is. Pure UI — no runtime cost.
  const PRESETS = {
    Fast:     { default_det_size: false, face_detector_size: '320', face_detector_threshold: 0.50, subsample_upscale: '128px', selected_enhancer: 'None',            num_swap_steps: 1 },
    Balanced: { default_det_size: true,  face_detector_size: '640', face_detector_threshold: 0.50, subsample_upscale: '256px', selected_enhancer: 'GPEN',            num_swap_steps: 1 },
    Quality:  { default_det_size: true,  face_detector_size: '640', face_detector_threshold: 0.50, subsample_upscale: '512px', selected_enhancer: 'Restoreformer++', num_swap_steps: 2 },
  };
  const activePreset = Object.keys(PRESETS).find((name) =>
    Object.entries(PRESETS[name]).every(([k, v]) =>
      k === 'default_det_size' ? (p[k] !== false) === (v !== false) : p[k] === v));
  const applyPreset = (name) => {
    setSettings((s) => ({ ...s, ...PRESETS[name] }));
    notify(`Applied ${name} preset`, 'info');
  };

  // User-defined default: a snapshot of the Face Swap tab settings the user
  // clicked "Save as default" on, kept in localStorage so it survives reloads.
  // Only ever holds FACESWAP_DEFAULTS keys, so it can never capture or restore
  // global/Settings-tab settings. null until the user saves one.
  const USER_DEFAULTS_KEY = 'roop_faceswap_user_defaults';
  const [userDefaults, setUserDefaults] = useState(() => {
    try {
      const raw = JSON.parse(localStorage.getItem(USER_DEFAULTS_KEY) || 'null');
      return raw && typeof raw === 'object' ? raw : null;
    } catch { return null; }
  });

  // Snapshot the current Face Swap tab settings as the user's new default.
  // Restricted to the FACESWAP_DEFAULTS key set so we never bake a global
  // setting into the tab default.
  const saveAsDefault = () => {
    const snapshot = {};
    for (const k of Object.keys(FACESWAP_DEFAULTS)) {
      snapshot[k] = p[k] !== undefined ? p[k] : FACESWAP_DEFAULTS[k];
    }
    try { localStorage.setItem(USER_DEFAULTS_KEY, JSON.stringify(snapshot)); } catch { /* storage blocked — non-fatal */ }
    setUserDefaults(snapshot);
    notify('Saved current settings as your default', 'info');
  };

  // Restore every Face Swap tab setting to the user's saved default if they
  // have one, otherwise the baked-in factory defaults (faceswap/defaults.js).
  // Persist immediately so the backend CFG matches even if the user never runs
  // a preview/swap afterwards.
  const resetToDefaults = () => {
    const target = userDefaults || FACESWAP_DEFAULTS;
    setSettings((s) => ({ ...s, ...target }));
    postJSON('/api/settings', target).catch(() => { /* backend offline — will persist on next run */ });
    notify(userDefaults ? 'Restored your saved default' : 'Face Swap settings reset to factory defaults', 'info');
  };

  // Forget the user's saved default so "Reset" falls back to factory defaults.
  const clearUserDefault = () => {
    try { localStorage.removeItem(USER_DEFAULTS_KEY); } catch { /* non-fatal */ }
    setUserDefaults(null);
    notify('Cleared your saved default — Reset now uses factory defaults', 'info');
  };

  // ── initial rehydrate ──
  // Pinokio reloads the webview whenever you switch the RUN/DEV/FILES tabs,
  // which remounts this component and wipes its React state. The backend keeps
  // running, so we restore both the faces/targets AND the live job state.
  useEffect(() => {
    getJSON('/api/state').then((st) => {
      checkDesync(st);
      setSourceFaces(st.source_faces || []);
      if (st.source_faces_info) setSourceFacesInfo(st.source_faces_info);
      setTargetFaces(st.target_faces || []);
      setTargetGroups(st.target_groups || []);
      setTargetNames(st.target_names || []);
      setTargetFacesInfo(st.target_faces_info || []);
      const tg = st.targets || [];
      setTargets(tg);
      if (tg.length > 0) {
        const sel = st.selected_target_index || 0;
        setSelTarget(sel);
        setMaxFrames(tg[sel]?.frames || 1);
        setFrame(1);
      }
    }).catch(() => {});

    // Restore an in-flight swap so the run bar shows Pause/Resume/Stop and the
    // progress %/desc again instead of falling back to "Start Swapping".
    getJSON('/api/progress').then((pr) => {
      setProgress(pr);
      if (pr.processing) {
        if (!startTime) setStartTime(Date.now());
      }
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshPreview = async (opts = {}) => {
    if (targets.length === 0) { setPreviewSrc(''); return; }
    
    const idx = opts.index ?? selTarget;
    const fr = opts.frame ?? frame;
    const fake = opts.fake ?? fakePreview;

    // Check client-side preview cache first.
    //
    // `force` skips it AND evicts the entry. The cache key covers every setting
    // that reaches the backend, so an automatic refresh after a settings change
    // is always a miss and re-renders correctly. But the manual 🔄 Refresh button
    // used to go through this same lookup, which made it a no-op whenever the
    // cache was warm: nothing the user could do would re-run the swap for the
    // current frame. Anything the key cannot see — a source file edited on disk,
    // a model swapped underneath us, a backend restart — left a stale image with
    // no way to clear it.
    if (!opts.force) {
      const cached = getCachedPreview(idx, fr);
      if (cached) {
        setPreviewFaces(cached.faces);
        setPreviewPersonIds(cached.personIds || []);
        setPreviewSrc(cached.image);
        return;
      }
    } else {
      delete previewCacheRef.current[getCacheKey(idx, fr)];
    }

    // Single-flight: the backend's live_swap shares one (non-thread-safe)
    // ProcessMgr on the GPU. Two overlapping /api/preview calls corrupt/hang
    // TensorRT/CUDA. So never run two at once — queue the latest request and
    // run it once the current one finishes.
    if (previewBusyRef.current) { 
      previewPendingRef.current = { ...opts, index: idx, frame: fr, fake: fake }; 
      return; 
    }
    
    previewBusyRef.current = true;
    setPreviewing(true);
    // Safety net: the first run of a new model downloads it and builds a
    // TensorRT/CUDA engine (minutes). Abort after 15 min so a genuine hang can
    // never wedge the single-flight guard permanently.
    const ctrl = new AbortController();
    const killer = setTimeout(() => ctrl.abort(), 15 * 60 * 1000);
    try {
      const res = await postJSON('/api/preview', buildPreviewPayload(p, { index: idx, frame: fr, fake }), { signal: ctrl.signal });
      if (res.faces) setPreviewFaces(res.faces);
      setPreviewPersonIds(res.person_ids || []);
      setPreviewSrc(res.image || '');
      if (res.image) {
        setCachedPreview(idx, fr, { faces: res.faces || [], personIds: res.person_ids || [], image: res.image });
      }
    } catch (e) {
      notify(e.name === 'AbortError' ? 'Preview timed out (model build took too long)' : e.message, 'error');
    }
    finally {
      clearTimeout(killer);
      previewBusyRef.current = false;
      setPreviewing(false);
      if (previewPendingRef.current) {
        const next = previewPendingRef.current;
        previewPendingRef.current = null;
        refreshPreview(next);
      }
    }
  };

  const loadEnhancerPreviews = async (activeCheck) => {
    if (targets.length === 0) return;
    const available = selectedGridEnhancers.filter(e => meta.enhancers?.includes(e));

    setEnhancerPreviews((prev) => {
      const reset = {};
      for (const enh of available) {
        if (prev[enh]) reset[enh] = prev[enh];
      }
      return reset;
    });
    setEnhancerTimes((prev) => {
      const reset = {};
      for (const enh of available) {
        if (prev[enh]) reset[enh] = prev[enh];
      }
      return reset;
    });
    setLiveRenderingTimers((prev) => {
      const reset = {};
      for (const enh of available) {
        if (prev[enh]) reset[enh] = prev[enh];
      }
      return reset;
    });

    for (const enh of available) {
      if (!activeCheck()) return;
      const localParams = { ...p, selected_enhancer: enh };
      const cacheKey = `${selTarget}_${frame}_${previewSignature(localParams, fakePreview)}_${sourceSig}_${targetSig}_${selSource}_${selTargetFace}`;
      
      if (previewCacheRef.current[cacheKey]) {
        if (!activeCheck()) return;
        setEnhancerPreviews((prev) => ({ ...prev, [enh]: previewCacheRef.current[cacheKey].image }));
        setEnhancerTimes((prev) => ({ ...prev, [enh]: 'Cached' }));
        continue;
      }

      try {
        const start = Date.now();
        setLiveRenderingTimers(prev => ({ ...prev, [enh]: '0.0s' }));
        activeIntervalsRef.current[enh] = setInterval(() => {
          setLiveRenderingTimers(prev => ({
            ...prev,
            [enh]: ((Date.now() - start) / 1000).toFixed(1) + 's'
          }));
        }, 100);

        const res = await postJSON('/api/preview', buildPreviewPayload(localParams, { index: selTarget, frame, fake: fakePreview }));
        const duration = ((Date.now() - start) / 1000).toFixed(2);
        if (activeIntervalsRef.current[enh]) {
          clearInterval(activeIntervalsRef.current[enh]);
          delete activeIntervalsRef.current[enh];
        }
        if (!activeCheck()) return;
        if (res.image) {
          setEnhancerPreviews((prev) => ({ ...prev, [enh]: res.image }));
          setEnhancerTimes((prev) => ({ ...prev, [enh]: `${duration}s` }));
          setLiveRenderingTimers((prev) => ({ ...prev, [enh]: null }));
          previewCacheRef.current[cacheKey] = { faces: res.faces || [], image: res.image };
        }
      } catch {
        if (activeIntervalsRef.current[enh]) {
          clearInterval(activeIntervalsRef.current[enh]);
          delete activeIntervalsRef.current[enh];
        }
        setLiveRenderingTimers((prev) => ({ ...prev, [enh]: null }));
        // Fail silently
      }
    }
  };

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: loadEnhancerPreviews is a stable closure invoked on mount/trigger */
  useEffect(() => {
    if (!comparingEnhancers || targets.length === 0) return;
    let active = true;
    loadEnhancerPreviews(() => active);
    return () => {
      active = false;
      if (activeIntervalsRef.current) {
        Object.values(activeIntervalsRef.current).forEach(clearInterval);
        activeIntervalsRef.current = {};
      }
    };
  }, [comparingEnhancers, selectedGridEnhancers, frame, selTarget, targets.length, sourceSig, targetSig, selSource, selTargetFace, previewKey]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // Render one preview per selected mask engine, holding every other setting
  // (including the currently-selected enhancer) fixed — the mirror image of
  // loadEnhancerPreviews, but varying `mask_engine` instead of `enhancer`.
  const loadMaskPreviews = async (activeCheck) => {
    if (targets.length === 0) return;
    const available = selectedGridMasks.filter(m => meta.mask_engines?.includes(m));

    const keepOnly = (prev) => {
      const reset = {};
      for (const m of available) if (prev[m]) reset[m] = prev[m];
      return reset;
    };
    setMaskPreviews(keepOnly);
    setMaskTimes(keepOnly);
    setMaskRenderTimers(keepOnly);

    for (const me of available) {
      if (!activeCheck()) return;
      const localParams = { ...p, mask_engine: me };
      const cacheKey = `${selTarget}_${frame}_${previewSignature(localParams, fakePreview)}_${sourceSig}_${targetSig}_${selSource}_${selTargetFace}`;

      if (previewCacheRef.current[cacheKey]) {
        if (!activeCheck()) return;
        setMaskPreviews((prev) => ({ ...prev, [me]: previewCacheRef.current[cacheKey].image }));
        setMaskTimes((prev) => ({ ...prev, [me]: 'Cached' }));
        continue;
      }

      try {
        const start = Date.now();
        setMaskRenderTimers(prev => ({ ...prev, [me]: '0.0s' }));
        maskIntervalsRef.current[me] = setInterval(() => {
          setMaskRenderTimers(prev => ({ ...prev, [me]: ((Date.now() - start) / 1000).toFixed(1) + 's' }));
        }, 100);

        const res = await postJSON('/api/preview', buildPreviewPayload(localParams, { index: selTarget, frame, fake: fakePreview }));
        const duration = ((Date.now() - start) / 1000).toFixed(2);
        if (maskIntervalsRef.current[me]) {
          clearInterval(maskIntervalsRef.current[me]);
          delete maskIntervalsRef.current[me];
        }
        if (!activeCheck()) return;
        if (res.image) {
          setMaskPreviews((prev) => ({ ...prev, [me]: res.image }));
          setMaskTimes((prev) => ({ ...prev, [me]: `${duration}s` }));
          setMaskRenderTimers((prev) => ({ ...prev, [me]: null }));
          previewCacheRef.current[cacheKey] = { faces: res.faces || [], image: res.image };
        }
      } catch {
        if (maskIntervalsRef.current[me]) {
          clearInterval(maskIntervalsRef.current[me]);
          delete maskIntervalsRef.current[me];
        }
        setMaskRenderTimers((prev) => ({ ...prev, [me]: null }));
        // Fail silently (e.g. SAM2-tracked needs a video pre-pass and may skip a single frame)
      }
    }
  };

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: loadMaskPreviews is a stable closure invoked on trigger */
  useEffect(() => {
    if (!comparingMasks || targets.length === 0) return;
    let active = true;
    loadMaskPreviews(() => active);
    return () => {
      active = false;
      if (maskIntervalsRef.current) {
        Object.values(maskIntervalsRef.current).forEach(clearInterval);
        maskIntervalsRef.current = {};
      }
    };
  }, [comparingMasks, selectedGridMasks, frame, selTarget, targets.length, sourceSig, targetSig, selSource, selTargetFace, previewKey]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // ── Swapper-model grid preview loader ──────────────────────────────────
  // Identical shape to loadMaskPreviews/loadEnhancerPreviews, but varies
  // `swap_model` instead of mask_engine/enhancer. Each selected swapper model
  // downloads on first use, so the live per-cell timer matters here.
  const loadSwapperPreviews = async (activeCheck) => {
    if (targets.length === 0) return;
    const available = selectedGridSwappers.filter(m => meta.swap_models?.includes(m));

    const keepOnly = (prev) => {
      const reset = {};
      for (const m of available) if (prev[m]) reset[m] = prev[m];
      return reset;
    };
    setSwapperPreviews(keepOnly);
    setSwapperTimes(keepOnly);
    setSwapperRenderTimers(keepOnly);

    for (const sm of available) {
      if (!activeCheck()) return;
      const localParams = { ...p, swap_model: sm };
      const cacheKey = `${selTarget}_${frame}_${previewSignature(localParams, fakePreview)}_${sourceSig}_${targetSig}_${selSource}_${selTargetFace}`;

      if (previewCacheRef.current[cacheKey]) {
        if (!activeCheck()) return;
        setSwapperPreviews((prev) => ({ ...prev, [sm]: previewCacheRef.current[cacheKey].image }));
        setSwapperTimes((prev) => ({ ...prev, [sm]: 'Cached' }));
        continue;
      }

      try {
        const start = Date.now();
        setSwapperRenderTimers(prev => ({ ...prev, [sm]: '0.0s' }));
        swapperIntervalsRef.current[sm] = setInterval(() => {
          setSwapperRenderTimers(prev => ({ ...prev, [sm]: ((Date.now() - start) / 1000).toFixed(1) + 's' }));
        }, 100);

        const res = await postJSON('/api/preview', buildPreviewPayload(localParams, { index: selTarget, frame, fake: fakePreview }));
        const duration = ((Date.now() - start) / 1000).toFixed(2);
        if (swapperIntervalsRef.current[sm]) {
          clearInterval(swapperIntervalsRef.current[sm]);
          delete swapperIntervalsRef.current[sm];
        }
        if (!activeCheck()) return;
        if (res.image) {
          setSwapperPreviews((prev) => ({ ...prev, [sm]: res.image }));
          setSwapperTimes((prev) => ({ ...prev, [sm]: `${duration}s` }));
          setSwapperRenderTimers((prev) => ({ ...prev, [sm]: null }));
          previewCacheRef.current[cacheKey] = { faces: res.faces || [], image: res.image };
        }
      } catch {
        if (swapperIntervalsRef.current[sm]) {
          clearInterval(swapperIntervalsRef.current[sm]);
          delete swapperIntervalsRef.current[sm];
        }
        setSwapperRenderTimers((prev) => ({ ...prev, [sm]: null }));
        // Fail silently (a model may fail to download or init on a single frame)
      }
    }
  };

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: loadSwapperPreviews is a stable closure invoked on trigger */
  useEffect(() => {
    if (!comparingSwappers || targets.length === 0) return;
    let active = true;
    loadSwapperPreviews(() => active);
    return () => {
      active = false;
      if (swapperIntervalsRef.current) {
        Object.values(swapperIntervalsRef.current).forEach(clearInterval);
        swapperIntervalsRef.current = {};
      }
    };
  }, [comparingSwappers, selectedGridSwappers, frame, selTarget, targets.length, sourceSig, targetSig, selSource, selTargetFace, previewKey]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // ── AI-upscale grid preview loader ─────────────────────────────────────
  // Unlike the enhancer/swapper grids (which re-run the full swap per cell),
  // this swaps the frame ONCE then upscales that single result with each
  // selected model — so the grid isolates the upscaler's effect (and is much
  // cheaper: one swap + N upscales instead of N full swaps).
  const loadUpscalePreviews = async (activeCheck) => {
    if (targets.length === 0) return;
    const labels = AI_UPSCALE_MODELS.map(m => m.label);
    const available = selectedGridUpscalers.filter(l => labels.includes(l));

    const keepOnly = (prev) => {
      const reset = {};
      for (const l of available) if (prev[l]) reset[l] = prev[l];
      return reset;
    };
    setUpscalePreviews(keepOnly);
    setUpscaleTimes(keepOnly);
    setUpscaleRenderTimers(keepOnly);

    // Swap the frame ONCE to get the base image every cell upscales. fake_preview
    // is forced on so the grid always compares upscalers on the swapped result
    // (falls back to the raw frame server-side when there are no source faces).
    let baseImage = '';
    try {
      const baseRes = await postJSON('/api/preview', buildPreviewPayload(p, { index: selTarget, frame, fake: true }));
      baseImage = baseRes.image || '';
    } catch {
      // handled below (no base → nothing to upscale)
    }
    if (!activeCheck()) return;
    if (!baseImage) return;

    for (const label of available) {
      if (!activeCheck()) return;
      const subtype = AI_UPSCALE_MODELS.find(m => m.label === label)?.value || 'esrganx2';
      try {
        const start = Date.now();
        setUpscaleRenderTimers(prev => ({ ...prev, [label]: '0.0s' }));
        upscaleIntervalsRef.current[label] = setInterval(() => {
          setUpscaleRenderTimers(prev => ({ ...prev, [label]: ((Date.now() - start) / 1000).toFixed(1) + 's' }));
        }, 100);

        const res = await postJSON('/api/preview_upscale', { image: baseImage, subtype });

        const duration = ((Date.now() - start) / 1000).toFixed(2);
        if (upscaleIntervalsRef.current[label]) {
          clearInterval(upscaleIntervalsRef.current[label]);
          delete upscaleIntervalsRef.current[label];
        }
        if (!activeCheck()) return;
        if (res.image) {
          setUpscalePreviews((prev) => ({ ...prev, [label]: res.image }));
          setUpscaleTimes((prev) => ({ ...prev, [label]: `${duration}s` }));
          setUpscaleRenderTimers((prev) => ({ ...prev, [label]: null }));
        }
      } catch {
        if (upscaleIntervalsRef.current[label]) {
          clearInterval(upscaleIntervalsRef.current[label]);
          delete upscaleIntervalsRef.current[label];
        }
        setUpscaleRenderTimers((prev) => ({ ...prev, [label]: null }));
        // Fail silently (a model may fail to download or init on a single frame)
      }
    }
  };

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: loadUpscalePreviews is a stable closure invoked on trigger */
  useEffect(() => {
    if (!comparingUpscalers || targets.length === 0) return;
    let active = true;
    loadUpscalePreviews(() => active);
    return () => {
      active = false;
      if (upscaleIntervalsRef.current) {
        Object.values(upscaleIntervalsRef.current).forEach(clearInterval);
        upscaleIntervalsRef.current = {};
      }
    };
  }, [comparingUpscalers, selectedGridUpscalers, frame, selTarget, targets.length, sourceSig, targetSig, selSource, selTargetFace, previewKey]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // Live elapsed timer for the "Rendering…" badge so a slow first run reads as
  // working, not hung.
  useEffect(() => {
    if (!previewing) { setPreviewSecs(0); return; }
    const started = Date.now();
    setPreviewSecs(0);
    const id = setInterval(() => setPreviewSecs(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(id);
  }, [previewing]);

  // Auto-refresh preview when the target list, selected target, or frame
  // changes (targets.length covers initial rehydrate after a page refresh).
  useEffect(() => {
    if (targets.length === 0 || progress.processing || isScrubbing || isPlaying) return;
    const t = setTimeout(() => refreshPreview(), 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selTarget, frame, targets.length, isScrubbing, isPlaying]);

  useEffect(() => {
    if (targets.length === 0 || isScrubbing || isPlaying) return;
    // A run is in flight — the GPU is busy, so remember that the preview is now
    // out of date and re-render it once the run finishes. Without this, faces
    // changed during a run leave the previous result frozen on screen.
    if (progress.processing) { previewDeferredRef.current = true; return; }
    const t = setTimeout(() => refreshPreview(), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewKey, sourceSig, targetSig, isScrubbing, isPlaying]);

  useEffect(() => {
    if (progress.processing || !previewDeferredRef.current) return;
    previewDeferredRef.current = false;
    if (targets.length === 0 || isScrubbing || isPlaying) return;
    const t = setTimeout(() => refreshPreview(), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress.processing]);

  // ── source / target file handling ──
  const onAddSource = async (files) => {
    if (!files || !files.length) return;
    const before = sourceFaces.length;
    setUploadingSrc(true);
    try {
      const res = checkDesync(await postFiles('/api/source/add', files));
      setSourceFaces(res.source_faces);
      if (res.source_faces_info) setSourceFacesInfo(res.source_faces_info);
      const added = res.source_faces.length - before;
      if (added > 0) notify(`Loaded ${added} face(s) — ${res.faceset_count} faceset(s) total`);
      else notify('No face detected in the uploaded file(s)', 'error');
    } catch (err) { notify(err.message, 'error'); }
    finally { setUploadingSrc(false); }
  };

  const onAddTarget = async (files) => {
    if (!files || !files.length) return;
    setUploadingTgt(true);
    try {
      const beforeCount = targets.length;
      const res = await postFiles('/api/target/add', files);
      const newTargetsList = res.targets || [];
      setTargets(newTargetsList);
      setSelTarget(res.selected_target_index || 0);
      const mf = newTargetsList[res.selected_target_index || 0]?.frames || 1;
      setMaxFrames(mf); setFrame(1);
      refreshPreview({ index: res.selected_target_index || 0, frame: 1 });
      notify(`Added ${newTargetsList.length - beforeCount} target(s)`);

      // Automatically add videos to batch queue if more than 1 video is uploaded
      const newTargets = newTargetsList.slice(beforeCount);
      const newVideos = newTargets.filter(t => t.frames > 1);
      if (newVideos.length > 1) {
        const jobs = newVideos.map((t, idx) => {
          const absoluteIndex = beforeCount + newTargets.indexOf(t);
          return {
            id: Date.now() + Math.random().toString(36).substr(2, 9) + '_' + idx,
            targetIndex: absoluteIndex,
            targetName: t.name || 'Unknown',
            sourceIndex: selSource,
            sourceName: sourceFaces[selSource] ? `Face ${selSource + 1}` : 'Selected Face',
            params: { ...p },
            faceMapping: getFaceMappingArray(),
            status: 'Pending'
          };
        });
        setQueue((prev) => [...prev, ...jobs]);
        notify(`Automatically queued ${jobs.length} uploaded videos`, 'success');
      }
    } catch (err) { notify(err.message, 'error'); }
    finally { setUploadingTgt(false); }
  };

  const removeTarget = async (i) => {
    const res = await postJSON('/api/target/remove', { index: i });
    setTargets(res.targets);
    const newSel = res.selected_target_index || 0;
    setSelTarget(newSel);
    if (res.targets.length === 0) { setPreviewSrc(''); setMaxFrames(1); }
    else { setMaxFrames(res.targets[newSel]?.frames || 1); setFrame(1); }
  };

  const selectTarget = async (i) => {
    setSelTarget(i);
    const res = await postJSON('/api/target/select', { index: i });
    setTargets(res.targets);
    const mf = res.targets[i]?.frames || 1;
    setMaxFrames(mf); setFrame(1);
    refreshPreview({ index: i, frame: 1 });
  };

  // The backend reports it if the faceset list and the gallery thumbnails have
  // fallen out of step. That state used to be invisible — the gallery showed a
  // face the swap didn't have, and the only symptom was that person silently
  // not being swapped — so say so loudly wherever a gallery payload arrives.
  const checkDesync = (res) => {
    if (res?.desync) notify(`Source faces: ${res.desync}`, 'error');
    return res;
  };

  const sourceAction = async (path, body) => {
    try {
      const res = checkDesync(await postJSON(path, body));
      if (res.source_faces) setSourceFaces(res.source_faces);
      if (res.source_faces_info) setSourceFacesInfo(res.source_faces_info);
    } catch (e) {
      // Surface failures (e.g. a 404 when the backend hasn't been restarted to
      // pick up a new endpoint) instead of silently doing nothing.
      notify(`${path.split('/').pop()} failed: ${e.message}. If this is a new feature, restart the app server.`, 'error');
    }
  };

  // Removing a source faceset shifts every later index down by one. The
  // person→source mapping stores raw indices, so without this it keeps pointing
  // at whatever slid into the removed slot (wrong face) or off the end of the
  // list (the backend then swaps in an empty faceset).
  const removeSource = async (i) => {
    await sourceAction('/api/source/remove', { index: i });
    setFaceMapping((prev) => {
      const next = {};
      for (const [pid, src] of Object.entries(prev || {})) {
        if (typeof src !== 'number' || src === i) continue;  // dropped → falls back to default
        next[pid] = src > i ? src - 1 : src;
      }
      return next;
    });
    const nextSel = selSource === i ? 0 : selSource > i ? selSource - 1 : selSource;
    if (nextSel !== selSource) {
      setSelSource(nextSel);
      try { await postJSON('/api/source/select', { index: nextSel }); } catch { /* selection is best-effort */ }
    }
  };

  const selectSource = async (i) => { setSelSource(i); await postJSON('/api/source/select', { index: i }); };

  const useFaceFromFrame = async () => {
    try {
      const res = await postJSON('/api/target/use_face', { index: selTarget, frame });
      setTargetFaces(res.target_faces);
      setTargetGroups(res.target_groups || []);
      setTargetNames(res.target_names || []);
      setTargetFacesInfo(res.target_faces_info || []);
      set('face_detection_mode', 'Selected face');
      notify(`Added ${res.count} target person(s)`);
    } catch (e) { notify(e.message, 'error'); }
  };

  // Spot-check the AI upscale on just the current preview frame — sends the
  // exact swapped image the preview shows to the backend, which upscales the
  // one frame and returns it for inspection in a modal.
  const upscaleThisFrame = async () => {
    if (!previewSrc) { notify('No preview to upscale', 'error'); return; }
    setUpscaling(true);
    try {
      const res = await postJSON('/api/preview_upscale', {
        image: previewSrc,
        subtype: p.upscale_model_after || 'esrganx2',
      });
      if (!res.image) throw new Error(res.message || 'upscale failed');
      setUpscaledSrc(res.image);
      setUpscaledDims(res.width && res.height ? { w: res.width, h: res.height } : null);
    } catch (e) {
      notify(e.message, 'error');
    } finally {
      setUpscaling(false);
    }
  };

  // Click a numbered face box in the live preview to capture just that person
  // as a NEW target face (face_index = its left-to-right box order).
  const addPersonFromBox = async (faceIndex) => {
    try {
      const res = await postJSON('/api/target/use_face', { index: selTarget, frame, face_index: faceIndex });
      if (!res.count) { notify('No face found for that box', 'error'); return; }
      setTargetFaces(res.target_faces);
      setTargetGroups(res.target_groups || []);
      setTargetNames(res.target_names || []);
      setTargetFacesInfo(res.target_faces_info || []);
      set('face_detection_mode', 'Selected face');
      notify(`Added Person ${(previewPersonIds[faceIndex] ?? faceIndex) + 1} to target faces`);
    } catch (e) { notify(e.message, 'error'); }
  };

  const setFrameMarkerVal = async (which, val) => {
    if (!val || isNaN(val)) return;
    try {
      const res = await postJSON('/api/target/set_frame', { which, frame: val });
      if (res.targets) setTargets(res.targets);
      notify(`Set ${which} frame = ${val}`);
    } catch (e) { notify(e.message, 'error'); }
  };

  // ── start / stop ──
  const start = async () => {
    try {
      await postJSON('/api/settings', p);            // persist CFG
      await postJSON('/api/swap', {
        ...p,
        enhancer: p.selected_enhancer, detection: p.face_detection_mode,
        output_method: p.output_method, video_method: p.video_swapping_method,
        upscale: p.subsample_upscale, mask_engine: p.mask_engine, clip_text: p.mask_clip_text,
        sam2_model_size: p.sam2_model_size, track_identities: p.track_identities,
        autorotate: p.autorotate_faces,
        face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
        num_swap_steps: num(p.num_swap_steps, 1),
        face_mapping: getFaceMappingArray(),
      });
      setStartTime(Date.now());
      // Ask for desktop-notification permission so we can ping on completion if
      // the tab is backgrounded (no-op if already decided).
      try { if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission(); } catch { /* ignore */ }
      // Optimistically flag processing so the old "Latest output" clears
      // immediately, before the first poll tick (~1s) confirms it.
      setProgress((pr) => ({ ...pr, processing: true, paused: false, progress: 0, desc: 'Starting…' }));
      notify('Processing started');
    } catch (e) { notify(e.message, 'error'); }
  };

  const stop = async () => { await postJSON('/api/stop', {}); notify('Stopping…', 'info'); };

  const pause = async () => {
    try {
      await postJSON('/api/pause', {});
      setProgress((pr) => ({ ...pr, paused: true, desc: 'Paused' }));
      notify('Paused', 'info');
    } catch (e) { notify(e.message, 'error'); }
  };

  const resume = async () => {
    try {
      await postJSON('/api/resume', {});
      setProgress((pr) => ({ ...pr, paused: false, desc: 'Resuming…' }));
      notify('Resumed');
    } catch (e) { notify(e.message, 'error'); }
  };

  // Hide the previous "Latest output" while a job is running so a new upload +
  // run never shows a stale result. The poll keeps reporting the old _last_output
  // until the new job finishes, so we gate on `processing` rather than clearing
  // progress.output (which the next poll tick would just restore).
  const out = progress.processing ? null : progress.output;
  const outUrl = out?.path ? `${API}/api/file?path=${encodeURIComponent(out.path)}&t=${progress.progress}` : '';
  const prog = progress.progress || 0;

  const elapsedMs = progress.processing && startTime ? Date.now() - startTime : 0;
  const etaMs = progress.processing && prog > 0.01 ? (elapsedMs * (1 - prog)) / prog : 0;
  // While actively scrubbing the timeline (or playing back) each frame is a
  // fresh server fetch, so request a lightweight downscaled frame — a full-res
  // HD/4K JPEG per frame makes dragging feel sluggish. Snap back to full
  // resolution the instant the user settles on a frame.
  const scrubbingNow = isScrubbing || isPlaying;
  const rawUrl = targets.length > 0
    ? `${API}/api/target/preview?index=${selTarget}&frame=${frame}${scrubbingNow ? '&width=960' : ''}`
    : '';

  const revealOutput = async () => {
    try { await postJSON('/api/reveal', { path: out?.path }); }
    catch (e) { notify(e.message, 'error'); }
  };

  // ── Buffered, video-style playback ──────────────────────────────────────
  // Each frame is a fresh server JPEG (a video seek + decode), which can't keep
  // up with real time — a naive setInterval(setFrame) makes the browser refetch
  // per tick and the video stutters through "cut" frames. Instead we prefetch a
  // rolling window of upcoming frames into decoded blob URLs and drive the
  // playhead off a requestAnimationFrame clock, only advancing onto frames that
  // are already buffered. The result plays back continuously like a real player.
  const clearPlayBuffer = () => {
    for (const url of playBufRef.current.values()) URL.revokeObjectURL(url);
    playBufRef.current.clear();
    playFetchRef.current.clear();
  };
  useEffect(() => clearPlayBuffer, []);  // revoke any buffered blobs on unmount
  useEffect(() => { targetsRef.current = targets; });
  // Detect a playhead move that did NOT come from the playback loop and queue it
  // as a seek. Previously the loop just overwrote it on the next tick, so
  // dragging the timeline or typing a frame during playback looked like nothing
  // happened at all.
  useEffect(() => {
    if (!isPlaying) { playSeekRef.current = null; return; }
    if (frame === playWroteRef.current) return;   // our own write, echoed back
    playSeekRef.current = frame;
  }, [frame, isPlaying]);

  useEffect(() => {
    if (!isPlaying) {
      clearPlayBuffer();
      playBufIdxRef.current = '';
      setBufferedSrc(null);
      stalledRef.current = false;
      setPlayStalled(false);
      return;
    }
    const idx = selTarget;
    const fps = targetsRef.current[idx]?.fps || 25;
    // The timeline is 1-based (a fresh target reports start_frame 0), so clamp
    // the loop/start point to 1 — otherwise looping jumps to "Frame 0" and
    // double-shows the first frame. Read live from the ref each tick so
    // dragging In/Out during playback takes effect without restarting the loop.
    // `|| maxFrames`, NOT `?? maxFrames`: the backend reports end_frame 0 for a
    // target whose trim has not been initialised yet (ProcessEntry starts at 0),
    // and ?? only substitutes null/undefined — so 0 survived as a real Out point,
    // making `end` 1 and every play press stop on the first tick. The timeline's
    // own In/Out display already used ||, so the bar showed the full range while
    // the player thought the clip was one frame long.
    const bounds = () => {
      const t = targetsRef.current[idx];
      const s = Math.max(1, t?.start_frame || 1);
      return { start: s, end: Math.max(s, t?.end_frame || maxFrames) };
    };
    let { start, end } = bounds();
    const frameDur = 1000 / (fps * (playbackRate || 1));
    // Lead is measured in TIME, not frames: a 60fps clip drains the buffer twice
    // as fast as a 30fps one, so a fixed 120-frame look-ahead is two seconds of
    // cover on one and one second on the other. Chunk size scales with it for
    // the same reason — the per-request overhead has to be amortised over enough
    // frames to keep up with the drain rate (raising CHUNK is the documented
    // lever for slow playback; parallelising requests is not, because
    // out-of-order arrivals break the decoder's sequential fast path).
    const AHEAD = Math.max(120, Math.round(fps * 2.5));
    const BEHIND = 8;    // frames retained behind before eviction
    const CHUNK = Math.max(16, Math.min(48, Math.round(fps / 2)));

    // Keep the buffered frames across speed/loop changes (they're still valid);
    // only discard and rebuild when the effect is (re)bound to a different video.
    const sig = `${idx}:${maxFrames}`;
    if (playBufIdxRef.current !== sig) {
      clearPlayBuffer();
      setBufferedSrc(null);
      playBufIdxRef.current = sig;
    }
    // Press play with the playhead parked on (or past) the out point and a real
    // player rewinds and plays again — it does not sit there. Without this the
    // loop stopped on its very first tick, so after a clip had played through
    // once the button appeared dead: it flipped to Pause and straight back.
    let cur = frame >= end ? start : Math.max(start, Math.min(frame, end));
    // The seek-detector runs first on this same commit (it is declared above and
    // shares the isPlaying dep) and will have queued the CURRENT playhead as an
    // external seek. That is the position we just decided to move away from, so
    // clear it — otherwise the first tick seeks straight back to the out point
    // and stops, undoing the rewind above.
    playSeekRef.current = null;
    let rendered = -1;
    let cancelled = false;
    let rafId = null;
    let lastTs = null;
    let acc = 0;

    // Prefetch stays SINGLE-FLIGHT and strictly ascending: the server decodes the
    // next in-order frame cheaply but re-seeks (seconds per frame for long-GOP
    // video) whenever a request breaks sequence, so overlapping requests would
    // arrive out of order and force a seek per frame.
    //
    // But one frame per request also means one HTTP round trip, one FastAPI
    // dispatch and one capture-lock acquire PER FRAME, and since the playhead
    // refuses to advance onto an unbuffered frame, that round trip — not the
    // ~1 ms decode — becomes the frame rate. The buffer can never build a lead
    // either: it fills at exactly the speed it drains. That is why playback ran
    // far below the clip's real fps however fast the machine is.
    // /api/target/preview_seq returns CHUNK consecutive frames in one
    // length-prefixed response, so the fixed per-request cost is paid once per
    // CHUNK frames while the decode stays strictly sequential.
    let inFlight = 0;

    const nextNeeded = () => {
      for (let f = cur; f <= Math.min(end, cur + AHEAD); f++) {
        if (!playBufRef.current.has(f) && !playFetchRef.current.has(f)) return f;
      }
      // When looping and the look-ahead overruns the out point, warm the
      // wrap-around frames near `start` so the loop seam doesn't stall.
      if (isLooping) {
        const overflow = Math.max(0, cur + AHEAD - end);
        for (let f = start; f <= Math.min(end, start + overflow); f++) {
          if (!playBufRef.current.has(f) && !playFetchRef.current.has(f)) return f;
        }
      }
      return null;
    };

    // Split the length-prefixed body: [4-byte BE length][JPEG] repeated.
    const splitChunk = (buf) => {
      const view = new DataView(buf);
      const out = [];
      let off = 0;
      while (off + 4 <= buf.byteLength) {
        const len = view.getUint32(off);
        off += 4;
        if (len <= 0 || off + len > buf.byteLength) break;
        out.push(new Blob([buf.slice(off, off + len)], { type: 'image/jpeg' }));
        off += len;
      }
      return out;
    };

    const fetchChunk = (fr) => {
      // Never run past the out point in one request — beyond it the frames are
      // outside the retained window and would be evicted the moment they land.
      const n = Math.max(1, Math.min(CHUNK, end - fr + 1));
      for (let i = 0; i < n; i++) playFetchRef.current.add(fr + i);
      inFlight++;
      const done = () => {
        for (let i = 0; i < n; i++) playFetchRef.current.delete(fr + i);
        inFlight--;
      };
      fetch(`${API}/api/target/preview_seq?index=${idx}&start=${fr}&count=${n}&width=960`)
        .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject()))
        .then((buf) => {
          if (cancelled) return;
          splitChunk(buf).forEach((blob, i) => {
            const f = fr + i;
            if (!playBufRef.current.has(f)) playBufRef.current.set(f, URL.createObjectURL(blob));
          });
        })
        .catch(() => {})
        .finally(done);
    };

    const pump = () => {
      // Evict frames outside the retained window (and outside the loop wrap set).
      const overflow = isLooping ? Math.max(0, cur + AHEAD - end) : 0;
      for (const k of [...playBufRef.current.keys()]) {
        const keep = (k >= cur - BEHIND && k <= cur + AHEAD) ||
                     (overflow > 0 && k >= start && k <= start + overflow);
        if (!keep) {
          URL.revokeObjectURL(playBufRef.current.get(k));
          playBufRef.current.delete(k);
        }
      }
      // Top up the single in-flight sequential request.
      if (inFlight < 1) {
        const fr = nextNeeded();
        if (fr !== null) fetchChunk(fr);
      }
    };

    const tick = (ts) => {
      if (cancelled) return;
      if (lastTs === null) lastTs = ts;
      acc += ts - lastTs;
      lastTs = ts;

      // Honour In/Out points moved during playback.
      ({ start, end } = bounds());

      // ── External seek ────────────────────────────────────────────────────
      // Someone moved the playhead from outside the loop (see playSeekRef).
      // Adopt the new position and drop the look-ahead — it is all frames the
      // playhead has just left, and keeping it would stall the prefetcher, which
      // only ever requests the lowest unbuffered frame ahead of `cur`.
      if (playSeekRef.current !== null) {
        const want = Math.max(start, Math.min(playSeekRef.current, end));
        playSeekRef.current = null;
        if (want !== cur) {
          cur = want;
          rendered = -1;
          acc = 0;
          clearPlayBuffer();
        }
      }

      pump();
      // Advance whole frames for the elapsed time, but never onto a frame that
      // isn't buffered yet — hold there (buffering) so playback never skips.
      let guard = 0;
      let stop = false;
      let waiting = false;
      while (acc >= frameDur && guard++ < 240) {
        let next;
        if (cur >= end) {
          if (isLooping) { next = start; }
          else { stop = true; break; }
        } else {
          next = cur + 1;
        }
        if (!playBufRef.current.has(next)) {
          acc = Math.min(acc, frameDur);
          waiting = true;   // held on an unbuffered frame — that's buffering
          break;
        }
        acc -= frameDur;
        cur = next;
      }
      // Surface the hold. Without it a slow buffer is indistinguishable from a
      // dead button: the playhead just sits there with the Pause icon showing.
      if (waiting !== stalledRef.current) {
        stalledRef.current = waiting;
        setPlayStalled(waiting);
      }
      // Paint the current frame before honouring a non-loop stop, so playback
      // ends showing the out point rather than one frame short of it.
      if (cur !== rendered) {
        const url = playBufRef.current.get(cur);
        // playWroteRef must be set BEFORE setFrame so the seek-detection effect
        // that runs on this commit already sees it and doesn't treat our own
        // advance as an external seek.
        if (url) { setBufferedSrc(url); playWroteRef.current = cur; setFrame(cur); rendered = cur; }
      }
      if (stop) { setIsPlaying(false); return; }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => { cancelled = true; if (rafId) cancelAnimationFrame(rafId); };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `frame`/`targets` are read through refs on purpose: the loop writes `frame` itself every tick, and re-binding on `targets` restarted playback on every In/Out drag.
  }, [isPlaying, isLooping, playbackRate, selTarget, maxFrames]);

  // Storyboard loading effect. The strip spans the VISIBLE range, so zooming in
  // re-samples it over the shorter span instead of leaving the same twelve
  // whole-clip stills stretched behind a narrow window. Debounced because a
  // wheel-zoom emits a burst of view updates and each strip is 12 backend
  // seeks; only the range you settle on gets fetched.
  useEffect(() => {
    if (targets.length === 0 || maxFrames <= 1) {
      setStoryboardThumbs([]);
      return;
    }
    const id = setTimeout(() => {
      // 12 across the strip: enough that the filmstrip reads as the shape of
      // the clip rather than eight stretched stills, still only 12 seeks.
      const numThumbs = 12;
      const lo = Math.max(1, view.start);
      const hi = Math.max(lo, Math.min(view.end, maxFrames));
      const step = (hi - lo) / Math.max(1, numThumbs - 1);
      const urls = [];
      for (let i = 0; i < numThumbs; i++) {
        const f = Math.max(1, Math.min(maxFrames, Math.round(lo + i * step)));
        urls.push(`${API}/api/target/preview?index=${selTarget}&frame=${f}&width=200`);
      }
      setStoryboardThumbs(urls);
    }, 220);
    return () => clearTimeout(id);
  }, [selTarget, maxFrames, targets.length, view.start, view.end]);

  // ── Frame <-> track-position mapping, through the view window ─────────────
  // `pct` is 0..1 across the visible track, NOT across the clip. With the view
  // at its default (whole clip) these are exactly the old full-range formulas.
  const frameAtPct = (pct) => {
    const { start, end } = viewRef.current;
    const span = Math.max(0, end - start);
    return Math.max(1, Math.min(Math.round(start + pct * span), maxFrames));
  };
  const pctOfFrame = (f) => {
    const { start, end } = viewRef.current;
    return (f - start) / Math.max(1, end - start);
  };
  const viewSpan = () => Math.max(1, viewRef.current.end - viewRef.current.start);

  // A new clip (or a target swap) starts fully zoomed out.
  useEffect(() => {
    setView({ start: 1, end: Math.max(1, maxFrames) });
  }, [selTarget, maxFrames]);

  // Scrubbing event handler setup
  const updateTimelinePos = (targetFrame, type) => {
    if (type === 'start') {
      const end = targets[selTarget]?.end_frame ?? maxFrames;
      const val = Math.max(1, Math.min(targetFrame, end));
      setTargets((prev) => {
        const copy = [...prev];
        if (copy[selTarget]) copy[selTarget] = { ...copy[selTarget], start_frame: val };
        return copy;
      });
    } else if (type === 'end') {
      const start = targets[selTarget]?.start_frame ?? 1;
      const val = Math.max(start, Math.min(targetFrame, maxFrames));
      setTargets((prev) => {
        const copy = [...prev];
        if (copy[selTarget]) copy[selTarget] = { ...copy[selTarget], end_frame: val };
        return copy;
      });
    } else {
      setFrame(Math.max(1, Math.min(targetFrame, maxFrames)));
    }
  };

  const handleTimelinePointerMove = (e) => {
    if (!timelineRef.current) return;
    const clientX = e.clientX ?? e.touches?.[0]?.clientX;
    if (clientX === undefined) return;
    timelinePendingRef.current = clientX;
    if (timelineRafRef.current) return;
    timelineRafRef.current = requestAnimationFrame(() => {
      timelineRafRef.current = null;
      if (!timelineRef.current || timelinePendingRef.current === null) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(timelinePendingRef.current - rect.left, rect.width));
      const f = frameAtPct(x / rect.width);
      setHoverFrame((prev) => (prev === f ? prev : f));
    });
  };

  const handleTimelinePointerLeave = () => {
    if (timelineRafRef.current) { cancelAnimationFrame(timelineRafRef.current); timelineRafRef.current = null; }
    timelinePendingRef.current = null;
    setHoverFrame(null);
  };
  useEffect(() => () => { if (timelineRafRef.current) cancelAnimationFrame(timelineRafRef.current); }, []);

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: scrub handlers bind latest markers via helpers, no re-subscribe wanted */
  useEffect(() => {
    if (!isScrubbing) return;

    // Coalesce scrub moves to one state update per animation frame so dragging
    // doesn't queue up more re-renders of this large component than the display
    // can paint.
    let rafId = null;
    let pendingX = null;
    const flush = () => {
      rafId = null;
      if (!timelineRef.current || pendingX === null) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const pct = Math.max(0, Math.min((pendingX - rect.left) / rect.width, 1));
      let targetFrame = frameAtPct(pct);
      // Magnetic snap: while dragging the playhead, pull it onto the In/Out
      // points and the clip ends when it lands within ~10px, so lining the
      // playhead up with a marker doesn't require pixel-perfect aim. The
      // tolerance is ~10px of the VISIBLE span, so zooming in makes the snap
      // proportionally finer instead of swallowing the precision it bought.
      if (dragType === 'playhead' && maxFrames > 1) {
        const snapFrames = Math.max(1, Math.round((10 / rect.width) * viewSpan()));
        const snapPoints = [1, maxFrames, targets[selTarget]?.start_frame ?? 1, targets[selTarget]?.end_frame ?? maxFrames];
        let best = null, bestDist = Infinity;
        for (const sp of snapPoints) {
          const d = Math.abs(sp - targetFrame);
          if (d <= snapFrames && d < bestDist) { best = sp; bestDist = d; }
        }
        if (best !== null) targetFrame = best;
      }
      updateTimelinePos(targetFrame, dragType);
    };
    const handlePointerMove = (e) => {
      const clientX = e.clientX ?? e.touches?.[0]?.clientX;
      if (clientX === undefined) return;
      pendingX = clientX;
      if (rafId === null) rafId = requestAnimationFrame(flush);
    };

    const handlePointerUp = () => {
      if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
      setIsScrubbing(false);
      if (dragType === 'start') {
        const val = targets[selTarget]?.start_frame ?? 1;
        setFrameMarkerVal('start', val);
      } else if (dragType === 'end') {
        const val = targets[selTarget]?.end_frame ?? maxFrames;
        setFrameMarkerVal('end', val);
      }
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [isScrubbing, dragType, selTarget, maxFrames, targets]);
  /* eslint-enable react-hooks/exhaustive-deps */

  const handleTimelinePointerDown = (e) => {
    if (!timelineRef.current) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const clientX = e.clientX ?? e.touches?.[0]?.clientX;
    if (clientX === undefined) return;
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(x / rect.width, 1));
    const targetFrame = frameAtPct(pct);

    const sPct = pctOfFrame(targets[selTarget]?.start_frame ?? 1);
    const ePct = pctOfFrame(targets[selTarget]?.end_frame ?? maxFrames);
    const cPct = pctOfFrame(frame);

    const distStart = Math.abs(pct - sPct);
    const distEnd = Math.abs(pct - ePct);
    const distCurrent = Math.abs(pct - cPct);

    let dragTarget = 'playhead';
    const tolerance = 0.04;

    if (distStart < tolerance && distStart < distEnd && distStart < distCurrent) {
      dragTarget = 'start';
    } else if (distEnd < tolerance && distEnd < distStart && distEnd < distCurrent) {
      dragTarget = 'end';
    } else {
      dragTarget = 'playhead';
    }

    setDragType(dragTarget);
    setIsScrubbing(true);
    updateTimelinePos(targetFrame, dragTarget);
  };

  const renderPreviewClip = async () => {
    if (targets.length === 0) return;
    const fps = targets[selTarget]?.fps || 25;
    const origStart = targets[selTarget]?.start_frame || 1;
    const origEnd = targets[selTarget]?.end_frame || maxFrames;
    setOrigStartEnd({ start: origStart, end: origEnd });
    setIsGeneratingPreviewClip(true);

    const previewStart = frame;
    // 5 seconds preview = frame + 5 * fps
    const previewEnd = Math.min(maxFrames, frame + Math.round(5 * fps));

    try {
      // Set temporary start/end frames in backend
      await postJSON('/api/target/set_frame', { which: 'start', frame: previewStart });
      await postJSON('/api/target/set_frame', { which: 'end', frame: previewEnd });

      // Start the swap with current settings
      await postJSON('/api/settings', p);
      await postJSON('/api/swap', {
        ...p,
        enhancer: p.selected_enhancer, detection: p.face_detection_mode,
        output_method: p.output_method, video_method: p.video_swapping_method,
        upscale: p.subsample_upscale, mask_engine: p.mask_engine, clip_text: p.mask_clip_text,
        sam2_model_size: p.sam2_model_size, track_identities: p.track_identities,
        autorotate: p.autorotate_faces,
        face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
        num_swap_steps: num(p.num_swap_steps, 1),
        face_mapping: getFaceMappingArray(),
        target_index: selTarget,
        // Quick 5s preview clip: skip the heavy post-swap AI upscale so the
        // preview stays fast (the real render still upscales).
        upscale_after_swap: false,
      });

      setStartTime(Date.now());
      notify('Generating 5-second preview clip...');
    } catch (e) {
      notify(e.message, 'error');
      setIsGeneratingPreviewClip(false);
      setOrigStartEnd(null);
    }
  };

  // Restoration effect when swapping finishes
  /* eslint-disable react-hooks/exhaustive-deps -- intentional: fires on processing-complete transition; notify is stable */
  useEffect(() => {
    if (isGeneratingPreviewClip && !progress.processing && origStartEnd) {
      const restore = async () => {
        try {
          await postJSON('/api/target/set_frame', { which: 'start', frame: origStartEnd.start });
          await postJSON('/api/target/set_frame', { which: 'end', frame: origStartEnd.end });
          
          // Sync targets list
          const res = await getJSON('/api/state');
          if (res.targets) setTargets(res.targets);
        } catch (e) {
          console.error("Failed to restore timeline range markers:", e);
        } finally {
          setIsGeneratingPreviewClip(false);
          setOrigStartEnd(null);
          notify('5-second preview clip generated successfully!');
        }
      };
      restore();
    }
  }, [progress.processing, isGeneratingPreviewClip, origStartEnd]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // Keyboard Escape, Shortcuts HUD, & Global Productivity Hotkeys
  /* eslint-disable react-hooks/exhaustive-deps -- intentional: hotkey handlers read latest callbacks; re-subscribing every render would thrash listeners */
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignore key events if the user is typing in form controls
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;

      // Ignore while a modal dialog is open (confirm/prompt, command palette),
      // so playback/timeline hotkeys don't fire on the page behind it.
      if (document.querySelector('[role="dialog"]')) return;

      // Toggle shortcuts HUD: '?' or 'h'
      if (e.key === '?' || e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        setShowShortcutHUD((prev) => !prev);
        return;
      }

      // Play/Pause: Space
      if (e.key === ' ') {
        e.preventDefault();
        if (maxFrames > 1 && setIsPlaying) {
          setIsPlaying((p) => !p);
        } else if (setCompare) {
          // Match the 'C' shortcut and the UI toggle: compare and the
          // enhancer grid are mutually exclusive.
          setCompare((c) => {
            const nextVal = !c;
            if (nextVal) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); }
            return nextVal;
          });
        }
        return;
      }

      // Step frames left/right: Left/Right Arrow
      if (e.key === 'ArrowLeft' && maxFrames > 1 && setFrame) {
        e.preventDefault();
        setFrame((f) => Math.max(1, f - (e.shiftKey ? 10 : 1)));
        return;
      }
      if (e.key === 'ArrowRight' && maxFrames > 1 && setFrame) {
        e.preventDefault();
        setFrame((f) => Math.min(maxFrames, f + (e.shiftKey ? 10 : 1)));
        return;
      }

      // Jump to start/end: Home / End
      if (e.key === 'Home' && maxFrames > 1 && setFrame) {
        e.preventDefault();
        setFrame(1);
        return;
      }
      if (e.key === 'End' && maxFrames > 1 && setFrame) {
        e.preventDefault();
        setFrame(maxFrames);
        return;
      }

      // Range Trim Markers: '[' to set start, ']' to set end, 'R' to reset range
      if (e.key === '[') {
        e.preventDefault();
        setFrameMarkerVal('start', frame);
        return;
      }
      if (e.key === ']') {
        e.preventDefault();
        setFrameMarkerVal('end', frame);
        return;
      }
      if (e.key.toLowerCase() === 'r' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        setFrameMarkerVal('start', 1);
        setFrameMarkerVal('end', maxFrames);
        return;
      }

      // Comparison modes: 'C' to compare, 'S' to toggle split view
      if (e.key.toLowerCase() === 'c') {
        e.preventDefault();
        setCompare((prev) => {
          const nextVal = !prev;
          if (nextVal) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); }
          return nextVal;
        });
        return;
      }
      if (e.key.toLowerCase() === 's') {
        e.preventDefault();
        setSplitView((prev) => !prev);
        return;
      }

      // Add to batch queue: 'Q'
      if (e.key.toLowerCase() === 'q') {
        e.preventDefault();
        addToQueue();
        return;
      }

      // Swapping execution: Ctrl + Enter
      if (e.key === 'Enter' && e.ctrlKey) {
        e.preventDefault();
        if (isQueueRunning) return;
        if (queue.length > 0) {
          startQueue();
        } else {
          start();
        }
        return;
      }
    };

    const handleEsc = (e) => {
      if (e.key === 'Escape') {
        setShowShortcutHUD(false);
        setPastedFiles(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keydown', handleEsc);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keydown', handleEsc);
    };
  }, [
    frame,
    maxFrames,
    isPlaying,
    compare,
    splitView,
    queue,
    isQueueRunning,
    setFrame,
    setIsPlaying,
    setCompare,
    setComparingEnhancers,
    setSplitView,
    setFrameMarkerVal,
    addToQueue,
    start,
    startQueue
  ]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // Command-palette action bus (dispatched from App via window 'roop:command').
  // A ref keeps the handler map fresh without re-subscribing every render.
  const cmdRef = useRef({});
  cmdRef.current = {
    start: () => { if (isQueueRunning) return; if (queue.length > 0) startQueue(); else start(); },
    stop,
    queue: addToQueue,
    compare: () => setCompare((v) => { const n = !v; if (n) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); } return n; }),
    split: () => setSplitView((v) => !v),
    preview: () => refreshPreview({ force: true }),   // explicit user action — same as 🔄 Refresh
    shortcuts: () => setShowShortcutHUD(true),
  };
  useEffect(() => {
    const h = (e) => { const fn = cmdRef.current[e.detail?.id]; if (fn) fn(); };
    window.addEventListener('roop:command', h);
    return () => window.removeEventListener('roop:command', h);
  }, []);

  const startFrame = targets[selTarget]?.start_frame || 1;
  const endFrame = targets[selTarget]?.end_frame || maxFrames;

  // ── Pre-run estimate (idle only) ──
  // Heuristic baseline, refined by the measured ms/frame the backend has learned
  // for the CURRENT settings from past completed runs (roop.runtime_calib).
  const [calibEst, setCalibEst] = useState(null);
  const estFrames = maxFrames > 1 ? Math.max(1, endFrame - startFrame + 1) : (targets.length ? 1 : 0);

  // Fetch the learned estimate (debounced) whenever the perf-relevant settings
  // or the frame span change, while idle.
  useEffect(() => {
    if (progress.processing || estFrames <= 1 || targets.length === 0) { setCalibEst(null); return; }
    let cancelled = false;
    const t = setTimeout(async () => {
      try {
        const res = await postJSON('/api/runtime_estimate', {
          frames: estFrames,
          face_count: previewFaces.length,   // density hint from the current frame
          swap_model: p.swap_model,
          selected_enhancer: p.selected_enhancer,
          face_detection_mode: p.face_detection_mode,
          face_detector_size: p.face_detector_size,
          detector_engine: p.detector_engine,
          num_swap_steps: num(p.num_swap_steps, 1),
          subsample_upscale: p.subsample_upscale,
          track_identities: p.track_identities,
          temporal_detection: p.temporal_detection,
          mask_engine: p.mask_engine,
          stabilize_face: p.stabilize_face,
          stabilize_enhancer: p.stabilize_enhancer,
        });
        if (!cancelled) setCalibEst(res || null);
      } catch { if (!cancelled) setCalibEst(null); }
    }, 500);
    return () => { cancelled = true; clearTimeout(t); };
  }, [progress.processing, estFrames, previewFaces.length, p.swap_model, p.selected_enhancer,
      p.face_detection_mode, p.face_detector_size, p.detector_engine, p.num_swap_steps,
      p.subsample_upscale, p.track_identities, p.temporal_detection, p.mask_engine,
      p.stabilize_face, p.stabilize_enhancer, targets.length]);

  const heuristicPerFrame = (() => {
    let ms = 45;
    if (p.selected_enhancer && p.selected_enhancer !== 'None') ms += 70;
    const det = parseInt(p.face_detector_size || '640', 10) || 640;
    ms += (det / 640) * 15;
    ms += (num(p.num_swap_steps, 1) - 1) * 25;
    if (p.track_identities) ms += 8;
    const parallel = Math.max(1.5, Math.min(4, (telemetry?.threads || 3) * 0.6));
    return ms / parallel;   // wall-clock ms/frame, comparable to the measured value
  })();

  // Prefer the measured ms/frame. Blend 50/50 with the heuristic when the data
  // is thin (single sample, or the cross-settings global fallback).
  const estPerFrame = (() => {
    if (calibEst && calibEst.ms_per_frame) {
      const thin = calibEst.source !== 'measured' || (calibEst.samples || 0) < 2;
      return thin ? (calibEst.ms_per_frame + heuristicPerFrame) / 2 : calibEst.ms_per_frame;
    }
    return heuristicPerFrame;
  })();
  const estTotalMs = estFrames * estPerFrame;
  const estLearned = !!(calibEst && calibEst.source === 'measured' && (calibEst.samples || 0) >= 1);

  const heavyVram = (p.selected_enhancer && p.selected_enhancer !== 'None') &&
    (parseInt(p.face_detector_size || '640', 10) >= 960);

  // ── Clip advisor: sample the target, get recommended settings, apply. ──
  const [advice, setAdvice] = useState(null);
  const [advisorBusy, setAdvisorBusy] = useState(false);
  const ADVISOR_LABELS = {
    temporal_detection: 'Temporal detection',
    detector_engine: 'Detector engine',
    rescue_small_faces: 'Rescue small faces',
    face_detector_size: 'Detection resolution',
    subsample_upscale: 'Subsample upscale',
    face_detection_mode: 'Face selection',
    track_identities: 'Track identities',
    face_detector_threshold: 'Detection threshold',
    stabilize_face: 'Stabilize face',
  };
  const fmtAdviceVal = (v) => (v === true ? 'On' : v === false ? 'Off' : String(v));
  const runAdvisor = async () => {
    if (!targets.length) { notify('Load a target first', 'error'); return; }
    setAdvisorBusy(true);
    setAdvice(null);
    try {
      const res = await postJSON('/api/advisor', { index: selTarget, settings: p });
      setAdvice(res);
      if (res.recommendations?.length === 0 && !res.message) notify('Settings already fit this clip ✓');
    } catch (e) { notify(e.message, 'error'); } finally { setAdvisorBusy(false); }
  };
  const applyAdvice = () => {
    if (!advice?.recommendations?.length) return;
    advice.recommendations.forEach((r) => set(r.key, r.value));
    notify(`Applied ${advice.recommendations.length} recommended setting${advice.recommendations.length === 1 ? '' : 's'}`);
    setAdvice(null);
  };

  // ── Live camera (webcam → live swap → optional OBS virtual camera) ──
  const [liveActive, setLiveActive] = useState(false);
  const [liveBusy, setLiveBusy] = useState(false);
  const [liveCamNum, setLiveCamNum] = useState(0);
  const [liveRes, setLiveRes] = useState('1280x720');
  const [liveObs, setLiveObs] = useState(false);
  const [liveTick, setLiveTick] = useState(0);
  useEffect(() => {
    if (!liveActive) return;
    const id = setInterval(() => setLiveTick((t) => t + 1), 200);   // ~5fps preview
    return () => clearInterval(id);
  }, [liveActive]);
  // If the tab remounts while a cam session is running, pick its state back up.
  useEffect(() => {
    getJSON('/api/livecam/status').then((st) => setLiveActive(!!st.active)).catch(() => {});
  }, []);
  const startLiveCam = async () => {
    setLiveBusy(true);
    try {
      await postJSON('/api/livecam/start', { cam_number: liveCamNum, resolution: liveRes, stream_obs: liveObs });
      // The device opens asynchronously in the capture thread — confirm it came up.
      setTimeout(async () => {
        try {
          const st = await getJSON('/api/livecam/status');
          setLiveActive(!!st.active);
          if (!st.active) notify(`Camera ${liveCamNum} could not be opened — check the index / close other apps using it`, 'error');
          else notify('Live camera running' + (liveObs ? ' → streaming to virtual camera' : ''));
        } catch { /* backend gone */ }
        setLiveBusy(false);
      }, 1500);
    } catch (e) { notify(e.message, 'error'); setLiveBusy(false); }
  };
  const stopLiveCam = async () => {
    setLiveBusy(true);
    try { await postJSON('/api/livecam/stop', {}); } catch { /* already down */ }
    setLiveActive(false);
    setLiveBusy(false);
    notify('Live camera stopped');
  };

  // Derived values for the estimation box.
  const estFps = targets[selTarget]?.fps || 0;
  const estDurationS = estFps ? estFrames / estFps : 0;
  const estSourceLabel = estLearned ? 'Learned'
    : (calibEst?.source === 'global' ? 'Global avg' : 'Heuristic');
  const estSourceClass = estLearned ? 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10'
    : (calibEst?.source === 'global' ? 'text-amber-400 border-amber-500/30 bg-amber-500/10'
      : 'text-white/45 border-white/10 bg-white/5');

  return (
    <div className="flex flex-col lg:flex-row gap-6 items-start w-full">

      {/* COLUMN 1: Settings & Controls — sticky sidebar on large viewports so it
          follows the scroll (and never leaves the lower-left area empty) while
          scrolling a taller workspace. Scrolls internally when taller than the
          viewport.

          Hidden while a job runs: none of it can be acted on mid-run (the
          settings are already baked into the running job), and hiding it gives
          the whole viewport to the diagnostics, which is the only thing there is
          to look at until the run ends. Comes back on Stop / completion. */}
      <div className={`w-full lg:w-[380px] 3xl:w-[440px] 4xl:w-[520px] shrink-0 pr-0 lg:pr-2 space-y-5 select-none ${progress.processing ? 'hidden' : ''}`}>
        <Section title="Presets">
          <div className="flex flex-wrap gap-2">
            {Object.keys(PRESETS).map((name) => (
              <Button key={name} size="sm"
                variant={activePreset === name ? 'primary' : 'secondary'}
                onClick={() => applyPreset(name)}>
                {name === 'Fast' ? '⚡ Fast' : name === 'Balanced' ? '⚖️ Balanced' : '💎 Quality'}
              </Button>
            ))}
            <Button size="sm" variant="secondary" onClick={saveAsDefault}
              title="Save the current Face Swap tab settings as your default. 'Reset defaults' will restore to this.">
              ⭐ Save as default
            </Button>
            <Button size="sm" variant="secondary" onClick={resetToDefaults}
              title={userDefaults
                ? 'Restore every Face Swap tab setting to your saved default'
                : 'Restore every Face Swap tab setting to the factory defaults'}>
              ↩️ {userDefaults ? 'Reset to my default' : 'Reset defaults'}
            </Button>
            {userDefaults && (
              <Button size="sm" variant="secondary" onClick={clearUserDefault}
                title="Forget your saved default and go back to the factory defaults">
                🗑️ Clear my default
              </Button>
            )}
          </div>
          <div className="text-xs text-[var(--text-muted)] mt-2">
            Sets detection resolution, upscale, enhancer & swap steps. Other settings unchanged.
            {userDefaults
              ? ' “Reset” restores all Face Swap tab settings to the default you saved.'
              : ' “Save as default” stores the current Face Swap tab settings so “Reset” restores to them.'}
          </div>
        </Section>

        <Section title="Clip advisor">
          <button type="button" disabled={advisorBusy || !targets.length || progress.processing} onClick={runAdvisor}
            title="Samples the selected target (face sizes, count, detection coverage, motion, lighting) and recommends settings tuned to it. Nothing changes until you apply."
            className="w-full py-2 rounded-lg text-[12px] font-bold bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-40 flex items-center justify-center gap-2">
            {advisorBusy
              ? (<><span className="h-3 w-3 rounded-full border-2 border-[var(--accent)]/40 border-t-[var(--accent)] animate-spin" /> Analyzing target…</>)
              : '🧭 Analyze target & recommend settings'}
          </button>
          {advice && (
            <div className="space-y-2 mt-2">
              <div className="text-[10px] text-white/40 leading-relaxed">
                {advice.stats.sampled_frames} frame{advice.stats.sampled_frames === 1 ? '' : 's'} sampled ·
                faces found on {advice.stats.detection_coverage}% ·
                face size {advice.stats.min_face_size_pct}–{advice.stats.max_face_size_pct}% ·
                brightness {advice.stats.brightness}
                {advice.is_video ? ` · motion ${advice.stats.motion}` : ''}
              </div>
              {advice.message && <div className="text-[11px] text-amber-300/80">{advice.message}</div>}
              {advice.recommendations.length === 0 && !advice.message ? (
                <div className="text-[11px] font-bold text-emerald-400">✓ Current settings already fit this clip</div>
              ) : advice.recommendations.length > 0 && (
                <>
                  {advice.recommendations.map((r) => (
                    <div key={r.key} className="rounded-lg bg-black/25 border border-white/5 px-2.5 py-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[11px] font-bold text-white/85">{ADVISOR_LABELS[r.key] || r.key}</span>
                        <span className="text-[11px] font-mono shrink-0">
                          <span className="text-white/35">{fmtAdviceVal(p[r.key] ?? '—')}</span>
                          <span className="text-white/30"> → </span>
                          <span className="text-[var(--accent)] font-bold">{fmtAdviceVal(r.value)}</span>
                        </span>
                      </div>
                      <div className="text-[10px] text-white/40 leading-snug mt-0.5">{r.reason}</div>
                    </div>
                  ))}
                  <div className="flex gap-2">
                    <button type="button" onClick={applyAdvice}
                      className="flex-1 py-1.5 rounded-lg text-[11px] font-bold bg-[var(--accent)] text-white hover:opacity-90 transition-opacity">
                      ✓ Apply all {advice.recommendations.length}
                    </button>
                    <button type="button" onClick={() => setAdvice(null)}
                      className="px-3 py-1.5 rounded-lg text-[11px] font-bold bg-white/[0.04] border border-white/10 text-white/60 hover:text-white/90 transition-colors">
                      Dismiss
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </Section>

        <Section title="Live camera" collapsible defaultOpen={false}>
          <div className="text-[10px] text-white/40 leading-relaxed -mt-1">
            Swap your webcam feed live using the loaded source face{liveObs ? ',' : ''} — optionally
            published as a system <b>virtual camera</b> for OBS / video calls.
          </div>
          <div className="flex gap-2">
            <TextInput label="Camera #" type="number" value={liveCamNum}
              onChange={(v) => setLiveCamNum(Math.max(0, parseInt(v, 10) || 0))} />
            <Select label="Resolution" value={liveRes} onChange={setLiveRes}
              options={['640x480', '1280x720', '1920x1080']} />
          </div>
          <Toggle label="Stream to virtual camera (OBS)" info="Publishes the swapped feed as a system camera device via pyvirtualcam — pick 'OBS Virtual Camera' in any app." checked={liveObs} onChange={setLiveObs} />
          {!liveActive ? (
            <button type="button" disabled={liveBusy || progress.processing} onClick={startLiveCam}
              className="w-full py-2 rounded-lg text-[12px] font-bold bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-40 flex items-center justify-center gap-2">
              {liveBusy
                ? (<><span className="h-3 w-3 rounded-full border-2 border-[var(--accent)]/40 border-t-[var(--accent)] animate-spin" /> Opening camera…</>)
                : '📷 Start live camera'}
            </button>
          ) : (
            <>
              <div className="relative rounded-xl overflow-hidden bg-black/50 border border-white/10 aspect-video">
                <img src={`${API}/api/livecam/frame?t=${liveTick}`} alt="Live camera"
                  className="w-full h-full object-contain" draggable={false} />
                <span className="absolute top-2 left-2 inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-black/70 text-[10px] font-bold text-white/90 border border-white/10">
                  <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" /> LIVE
                </span>
              </div>
              <button type="button" disabled={liveBusy} onClick={stopLiveCam}
                className="w-full py-2 rounded-lg text-[12px] font-bold bg-white/[0.04] border border-white/10 text-white/70 hover:text-white hover:border-white/25 transition-colors disabled:opacity-40">
                ⏹ Stop live camera
              </button>
            </>
          )}
          {sourceFaces.length === 0 && (
            <div className="text-[10px] text-amber-300/70">No source face loaded — the feed will pass through unswapped.</div>
          )}
        </Section>

        <div className="space-y-5">
          <Section title="Swap settings">
          <Select label="Swap model" info="inswapper 128 · reswapper/hyperswap(a/b/c)/ghost(1-3)/simswap/hififace 256 · simswap_512 (each downloads on first use; ghost/simswap/hififace use their own alignment + identity converter)" value={p.swap_model} onChange={(v) => set('swap_model', v)} options={meta.swap_models} />
          <Select label="Face selection" value={p.face_detection_mode} onChange={(v) => set('face_detection_mode', v)} options={meta.face_detection_modes} />
          <Select
            label="Detector engine"
            info="SCRFD (default) is fast and accurate on frontal faces. YOLOFace is often better on steep profiles and partially occluded faces. RetinaFace has the highest recall on hard poses/lighting (fewest missed detections → less swap blink), slightly slower. All engines reuse the same identity/landmark models; alternates download a small model on first use."
            value={p.detector_engine || 'scrfd'}
            onChange={(v) => set('detector_engine', v)}
            options={meta.detector_engines || ['scrfd', 'yoloface', 'retinaface', 'retinaface_r50', 'yunet']}
          />
          <Select
            label="Face detection resolution"
            info="Higher resolution improves detection of small/distant faces, but runs slower. 640px is standard."
            value={p.face_detector_size || '640'}
            onChange={(v) => {
              set('face_detector_size', v);
              set('default_det_size', v === '640' || v === '960' || v === '1280');
            }} 
            options={['320', '640', '960', '1280']} 
          />
          <Slider
            label="Face detection threshold"
            info="Lower values (e.g. 0.40) detect angled, profile, or hard-to-see faces. Higher values avoid false detections. Default 0.50"
            min={0.10}
            max={0.90}
            step={0.05}
            value={num(p.face_detector_threshold, 0.50)}
            onChange={(v) => set('face_detector_threshold', v)}
          />
          <Slider
            label="Overlap NMS threshold"
            info="Lower values (e.g. 0.30) suppress close duplicates. Higher values (e.g. 0.60) allow detecting overlapping faces close to each other. Default 0.40"
            min={0.10}
            max={0.90}
            step={0.05}
            value={num(p.face_detector_nms, 0.40)}
            onChange={(v) => set('face_detector_nms', v)}
          />
          <Toggle label="🎯 Refine alignment (68-pt)" info="Derives the alignment keypoints from the 68-point landmark model instead of the detector's raw 5 points — more stable alignment on angled faces, less residual swap wobble. Small per-face cost." checked={!!p.refine_landmarks} onChange={(v) => set('refine_landmarks', v)} />
          <Select label="↔️ Profile alignment (90° faces)" info="Near-profile faces (~70°+ yaw) only — frontal and mid-angle faces are bit-identical in every mode. At high yaw both eyes project to almost the same point, so the normal 5-point fit against a frontal template is ill-conditioned. 'stabilize' takes the crop rotation from the eye→mouth axis, which stops head NOD leaking into in-plane roll (±25° of nod at 90° yaw otherwise swings the crop ~30°); this is a TEMPORAL fix — it reduces wobble across frames, so judge it on video, not on a still. 'pose' goes further and replaces the template with a head projected at the estimated yaw, cutting the fit error ~60%; geometrically cleaner, but the swap models were trained on frontally-aligned crops, so verify it visually before trusting it." value={p.yaw_align || 'off'} onChange={(v) => set('yaw_align', v)} options={['off', 'stabilize', 'pose']} />
          <Toggle label="🔬 Rescue small faces" info="When a frame has no detected face, retries on a 2x upscale to catch tiny/distant faces — without raising the global detection resolution for every frame." checked={!!p.rescue_small_faces} onChange={(v) => set('rescue_small_faces', v)} />
          <Slider label="Swapping steps" info="more = more likeness" min={1} max={5} step={1} value={num(p.num_swap_steps, 1)} onChange={(v) => set('num_swap_steps', v)} />
          <Select label="Post-processing enhancer" value={p.selected_enhancer} onChange={(v) => set('selected_enhancer', v)} options={meta.enhancers} />
          <Slider label="Max face similarity" info="0=identical 1=any" min={0.01} max={1} step={0.01} value={num(p.max_face_distance, 0.85)} onChange={(v) => set('max_face_distance', v)} />
          <Select label="Subsample upscale" value={p.subsample_upscale} onChange={(v) => set('subsample_upscale', v)} options={meta.upscale} />
          <Toggle label="🔎 AI upscale (after swap)" info="Runs an AI upscaler as the final step of the swap pass — each frame is swapped & enhanced first, then upscaled, producing a single output file (no second pass). Full-frame upscaling is heavy: ×4 on video is slow and VRAM-hungry." checked={!!p.upscale_after_swap} onChange={(v) => set('upscale_after_swap', v)} />
          {p.upscale_after_swap && (
            <Select label="AI upscale model" value={p.upscale_model_after} onChange={(v) => set('upscale_model_after', v)} options={AI_UPSCALE_MODELS} />
          )}
          <Select label="🎞 Frame interpolation (after swap)" info="Raises the output frame rate with motion-interpolated in-between frames as the final pass (after any upscale). RIFE = AI motion interpolation (recommended, fast); minterpolate = classical ffmpeg motion estimation (no model, much slower). Duration is unchanged — frame count and fps are multiplied together, audio untouched." value={p.interp_after_swap || 'off'} onChange={(v) => set('interp_after_swap', v)}
            options={[{ value: 'off', label: 'Off' }, { value: 'rife_2x', label: 'RIFE ×2 fps' }, { value: 'rife_4x', label: 'RIFE ×4 fps' }, { value: 'minterpolate_2x', label: 'ffmpeg minterpolate ×2' }]} />
          <Select label="Color/lighting match" info="Matches the swapped face's skin tone & lighting to the original scene. RCT = per-channel (fast, default). LCT = corrects hue casts. MKL = fullest match. None = off." value={p.color_transfer_mode || 'rct'} onChange={(v) => set('color_transfer_mode', v)} options={meta.color_transfer_modes || ['none', 'rct', 'lct', 'mkl']} />
          <Slider label="Original/Enhanced blend" min={0} max={1} step={0.01} value={num(p.blend_ratio, 0.8)} onChange={(v) => set('blend_ratio', v)} />
          <Slider label="Skin detail transfer" info="Adds the ORIGINAL footage's real high-frequency texture (pores, stubble, grain) onto the swapped face. The generator smooths skin and the enhancer fakes flickery pores; this uses genuine detail from the scene instead. 0 = off. Start ~0.3–0.5; too high reintroduces the target's skin identity." min={0} max={1} step={0.05} value={num(p.detail_transfer_strength, 0)} onChange={(v) => set('detail_transfer_strength', v)} />
        </Section>

        <Section title="Masking parameters" collapsible defaultOpen={false}>
          <Select label="Masking engine" value={p.mask_engine} onChange={(v) => set('mask_engine', v)} options={meta.mask_engines} />
          {p.mask_engine === 'Clip2Seg' && (
            <TextInput label="Objects to mask & restore" value={p.mask_clip_text} onChange={(v) => set('mask_clip_text', v)} placeholder="cup,hands,hair" />
          )}
          {p.mask_engine === 'Segment Anything 2 (tracked)' && (
            <Select label="SAM2 checkpoint (speed ↔ quality)" value={p.sam2_model_size || 'tiny'} onChange={(v) => set('sam2_model_size', v)} options={meta.sam2_model_sizes || ['tiny', 'small', 'base_plus', 'large']} />
          )}
          <Toggle label="Show mask overlay in preview" checked={!!p.show_mask_offsets} onChange={(v) => set('show_mask_offsets', v)} />
          <Slider label="Offset face top" min={0} max={2} step={0.01} value={num(p.mask_top, 0)} onChange={(v) => set('mask_top', v)} />
          <Slider label="Offset face bottom" min={0} max={2} step={0.01} value={num(p.mask_bottom, 0)} onChange={(v) => set('mask_bottom', v)} />
          <Slider label="Offset face left" min={0} max={2} step={0.01} value={num(p.mask_left, 0)} onChange={(v) => set('mask_left', v)} />
          <Slider label="Offset face right" min={0} max={2} step={0.01} value={num(p.mask_right, 0)} onChange={(v) => set('mask_right', v)} />
          <Slider label="Face mask edge blend" min={0} max={200} step={1} value={num(p.face_mask_blend, 20)} onChange={(v) => set('face_mask_blend', v)} />
        </Section>

        <Section title="Mouth & Angle math" collapsible defaultOpen={false}>
          <Slider label="Mouth mask top" min={0} max={2} step={0.01} value={num(p.mouth_top_scale, 1)} onChange={(v) => set('mouth_top_scale', v)} />
          <Slider label="Mouth mask bottom" min={0} max={2} step={0.01} value={num(p.mouth_bottom_scale, 1)} onChange={(v) => set('mouth_bottom_scale', v)} />
          <Slider label="Mouth mask left" min={0} max={2} step={0.01} value={num(p.mouth_left_scale, 1)} onChange={(v) => set('mouth_left_scale', v)} />
          <Slider label="Mouth mask right" min={0} max={2} step={0.01} value={num(p.mouth_right_scale, 1)} onChange={(v) => set('mouth_right_scale', v)} />
          <Slider label="Mouth mask edge blend" min={0} max={200} step={1} value={num(p.mouth_mask_blend, 10)} onChange={(v) => set('mouth_mask_blend', v)} />
          <Toggle label="🧊 3D source pose matching" info="Only affects image-source swappers (BlendSwap / UniFace) — feeds them a pose-matched source crop. Has NO effect on inswapper/ghost/hyperswap/simswap (their identity vector is pose-invariant, so warping the source would only degrade it)." checked={!!p.use_3d_recon} onChange={(v) => set('use_3d_recon', v)} />
          <Toggle label="🎯 Multi-angle source bank" info="auto-pick best source per frame" checked={!!p.use_source_bank} onChange={(v) => set('use_source_bank', v)} />
          <Toggle label="↔️ Frontalize angled faces" info="Un-rotates steep profile/side (lateral) faces before swapping so they don't come out distorted/'alien', then restores the original angle." checked={!!p.use_frontalization} onChange={(v) => set('use_frontalization', v)} />
          {p.use_frontalization && (
            <Slider label="Frontalize above angle (°)" info="Frontalization kicks in when the face yaw/pitch exceeds this. Lower = frontalize more; higher = only the steepest." min={10} max={60} step={5} value={num(p.frontalization_threshold, 30)} onChange={(v) => set('frontalization_threshold', v)} />
          )}
          <Toggle label="🧬 Reshape jaw/chin to source" info="Identity swappers (inswapper/hyperswap/reswapper…) keep the TARGET's jaw & chin bone structure. This warps the swapped face's lower-silhouette toward your SOURCE person's jaw/chin shape after the swap (a smooth liquify — no re-swap, so swap quality is untouched). Best for moderate shape differences; very large changes can distort the neck/background near the jaw. Landmark jitter is smoothed when Temporal detection is on." checked={!!p.jaw_reshape} onChange={(v) => set('jaw_reshape', v)} />
          {p.jaw_reshape && (
            <Slider label="Jaw reshape strength" info="0 = off (target's jaw), 1 = full source jaw/chin shape. Start around 0.4–0.6 and back off if the chin looks distorted." min={0} max={1} step={0.05} value={num(p.jaw_reshape_strength, 0.5)} onChange={(v) => set('jaw_reshape_strength', v)} />
          )}
        </Section>

        <Section title="Video parameters">
          <Select label="Video method" value={p.video_swapping_method} onChange={(v) => set('video_swapping_method', v)} options={meta.video_methods} />
          <Select label="On no face detected" value={p.no_face_action} onChange={(v) => set('no_face_action', v)} options={meta.no_face_actions} />
          <Toggle label="🛡️ Temporal detection (anti-flicker)" info="Video (In-Memory method): one tracked detection pre-pass over the clip. Short detection misses (≤10 frames) are gap-filled by interpolating the face's position, so the swap can't blink out; with 'Stabilize face' also on, keypoints AND mask/mouth landmarks are smoothed per person. The swap pass then skips per-frame detection and stays multi-threaded. Includes identity locking when that toggle is on." checked={!!p.temporal_detection} onChange={(v) => set('temporal_detection', v)} />
          <Toggle label="VR mode" checked={!!p.vr_mode} onChange={(v) => set('vr_mode', v)} />
          <Toggle label="✨ Reduce enhancer flicker" info="Temporally blends the enhanced face. Runs multi-threaded (work-stealing) when the launcher's ROOP_STAB_PARALLEL is on (the Pinokio default) — otherwise it forces single-thread. Either way it costs some extra compute (blending + per-block warm-up), so it's somewhat slower, not free." checked={!!p.stabilize_enhancer} onChange={(v) => set('stabilize_enhancer', v)} />
          {p.stabilize_enhancer && (
            <Slider label="Flicker reduction strength" info="higher = smoother" min={0} max={1} step={0.05} value={num(p.stabilize_enhancer_strength, 0.5)} onChange={(v) => set('stabilize_enhancer_strength', v)} />
          )}
        </Section>

        <div>
          <Section title="System options" collapsible defaultOpen={false}>
            <Toggle label="Auto rotate horizontal faces" checked={!!p.autorotate_faces} onChange={(v) => set('autorotate_faces', v)} />
            <Toggle label="Skip audio" checked={!!p.skip_audio} onChange={(v) => set('skip_audio', v)} />
            <Toggle label="Keep frames (when extracting)" checked={!!p.keep_frames} onChange={(v) => set('keep_frames', v)} />
            <Toggle label="Wait before creating video" checked={!!p.wait_after_extraction} onChange={(v) => set('wait_after_extraction', v)} />
          </Section>
        </div>

        <div>
          <Section title="Saved Profiles" collapsible defaultOpen={false}>
            <div className="flex flex-wrap gap-2 items-end">
              <div className="flex-1 min-w-[120px]">
                <TextInput label="Save active settings as:" value={newProfileName} onChange={setNewProfileName} placeholder="My Preset Name" />
              </div>
              <Button size="sm" onClick={saveProfile}>💾 Save</Button>
            </div>
            {profiles.length > 0 && (
              <div className="space-y-2 mt-3">
                <span className="text-xs font-semibold text-white/50">Apply custom preset:</span>
                <div className="flex flex-wrap gap-2">
                  {profiles.map((pr) => (
                    <div key={pr.name} className="flex items-center gap-1 bg-white/5 hover:bg-white/10 border border-white/5 rounded-lg px-2.5 py-1 text-xs transition-colors">
                      <button onClick={() => loadProfile(pr.name)} className="text-white hover:text-[var(--accent)] font-semibold">{pr.name}</button>
                      <button onClick={() => deleteProfile(pr.name)} className="text-white/40 hover:text-red-400 font-bold ml-1.5" title="Delete preset">✕</button>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-white/5">
              <Button size="xs" variant="secondary" onClick={exportProfiles}>📤 Export Presets</Button>
              <label className="inline-flex items-center justify-center px-2.5 py-1.5 rounded-lg text-xs font-bold bg-white/5 border border-white/10 text-white/80 hover:bg-white/10 hover:text-white cursor-pointer transition-all active:scale-95">
                📥 Import Presets
                <input type="file" accept=".json" onChange={importProfiles} className="hidden" />
              </label>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              <Button size="xs" variant="secondary" onClick={exportRecipe} className="!text-[var(--accent)]">🔗 Share Recipe</Button>
              <label className="inline-flex items-center justify-center px-2.5 py-1.5 rounded-lg text-xs font-bold bg-white/5 border border-white/10 text-white/80 hover:bg-white/10 hover:text-white cursor-pointer transition-all active:scale-95">
                📂 Load Recipe
                <input type="file" accept=".json" onChange={importRecipe} className="hidden" />
              </label>
            </div>
            <p className="text-[10px] text-white/30 mt-1.5 leading-relaxed">A recipe captures every setting <span className="text-white/45">and</span> the person→source mapping, so anyone can reproduce this exact look.</p>
          </Section>
        </div>

        <div>
          <Section title="Live Telemetry & Diagnostics" collapsible defaultOpen={false}>
            {telemetry ? (
              <div className="space-y-4 text-xs font-mono">
                {/* GPU & VRAM */}
                <div className="bg-black/25 p-3 rounded-xl border border-white/5 space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-white/40 text-[10px] uppercase font-bold tracking-wider">GPU</span>
                    <span className="text-white font-semibold truncate max-w-[200px]">{telemetry.gpu}</span>
                  </div>
                  {telemetry.vram_total > 0 && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-[10px]">
                        <span className="text-white/40">VRAM Usage</span>
                        <span className="text-emerald-400 font-bold">{telemetry.vram_used} GB / {telemetry.vram_total} GB</span>
                      </div>
                      <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                        <div 
                          className="bg-emerald-500 h-full rounded-full transition-all duration-500" 
                          style={{ width: `${Math.min(100, (telemetry.vram_used / telemetry.vram_total) * 100)}%` }} 
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* CPU & Memory */}
                <div className="bg-black/25 p-3 rounded-xl border border-white/5 space-y-2.5">
                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-[10px]">
                      <span className="text-white/40 uppercase font-bold tracking-wider">CPU Utilization</span>
                      <span className="text-orange-400 font-bold">{telemetry.cpu_percent}%</span>
                    </div>
                    <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                      <div 
                        className="bg-orange-500 h-full rounded-full transition-all duration-500" 
                        style={{ width: `${Math.min(100, telemetry.cpu_percent)}%` }} 
                      />
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-[10px]">
                      <span className="text-white/40 uppercase font-bold tracking-wider">System RAM</span>
                      <span className="text-blue-300 font-bold">{telemetry.ram_used} GB / {telemetry.ram_total} GB</span>
                    </div>
                    {telemetry.ram_total > 0 && (
                      <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                        <div 
                          className="bg-blue-500 h-full rounded-full transition-all duration-500" 
                          style={{ width: `${Math.min(100, (telemetry.ram_used / telemetry.ram_total) * 100)}%` }} 
                        />
                      </div>
                    )}
                  </div>
                </div>

                {/* Active threads info */}
                <div className="bg-black/25 px-3 py-2 rounded-xl border border-white/5 flex items-center justify-between">
                  <span className="text-[10px] text-white/40 uppercase font-bold tracking-wider">Active Python Threads</span>
                  <span className="text-pink-400 font-bold text-xs bg-pink-500/10 px-2 py-0.5 rounded-md border border-pink-500/20">{telemetry.threads}</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <Skeleton className="h-16 w-full" />
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-9 w-full" />
                <div className="text-[10px] text-white/25 italic text-center">Connecting to hardware diagnostics…</div>
              </div>
            )}
            <div className="mt-3 flex justify-between items-center">
              <Button size="sm" variant="secondary" onClick={() => setShowShortcutHUD(true)}>⌨️ Keyboard Shortcuts Info</Button>
            </div>
          </Section>
        </div>
      </div>
    </div>

      {/* COLUMN 2 & 3 WRAPPER: Preview canvas leads (hero) with the media
          asset managers as a right rail (2xl:flex-row-reverse), so the live
          preview is the visual center instead of buried on the far right. */}
      <div className="flex-1 w-full min-w-0 space-y-6 flex flex-col 2xl:flex-row-reverse gap-6">

        {/* COLUMN 2: Media Asset Manager — right rail (hidden while running, as
            with column 1 — sources/targets are locked in for the current job). */}
        <div className={`w-full 2xl:w-[360px] 3xl:w-[440px] 4xl:w-[500px] shrink-0 space-y-6 select-none ${progress.processing ? 'hidden' : ''}`}>
          <Section title="Target faces">
            <PersonGroups
              targetFaces={targetFaces}
              targetGroups={targetGroups}
              targetNames={targetNames}
              targetFacesInfo={targetFacesInfo}
              selTargetFace={selTargetFace}
              setSelTargetFace={setSelTargetFace}
              sourceFaces={sourceFaces}
              faceMapping={faceMapping}
              setFaceMapping={setFaceMapping}
              frame={frame}
              selTarget={selTarget}
              setTargetFaces={setTargetFaces}
              setTargetGroups={setTargetGroups}
              setTargetNames={setTargetNames}
              setTargetFacesInfo={setTargetFacesInfo}
              notify={notify}
              clearPreviewCache={clearPreviewCache}
            />
          </Section>

          <Section title="Enhancements">
            <Toggle label="🔒 Lock face identities (video)" info="For 'Selected face' mode on video: tracks each person across the clip and keeps them on one source, so identities don't flip frame-to-frame when faces cross or turn. Adds a short tracking pre-pass; the swap stays multi-threaded." checked={!!p.track_identities} onChange={(v) => set('track_identities', v)} />
            <Toggle label="🎯 Stabilize face (video)" info="Temporal keypoint smoothing — reduces swap wobble. Runs at Max Threads (2-pass) unless Enhancer Flicker is on." checked={!!p.stabilize_face} onChange={(v) => set('stabilize_face', v)} />
            {p.stabilize_face && (
              <>
                <Select label="Smoothing method" info="One Euro = adaptive (best jitter-vs-lag). EMA = simpler fixed smoothing." value={p.stabilize_method || 'one_euro'} onChange={(v) => set('stabilize_method', v)} options={['one_euro', 'ema']} />
                {p.stabilize_method !== 'ema' && (
                  <>
                    <Slider label="Smoothing (min cutoff)" info="lower = smoother, more lag" min={0.01} max={0.3} step={0.01} value={num(p.stabilize_min_cutoff, 0.05)} onChange={(v) => set('stabilize_min_cutoff', v)} />
                    <Slider label="Reactivity (beta)" info="higher = less lag on fast motion" min={0} max={0.2} step={0.01} value={num(p.stabilize_beta, 0.02)} onChange={(v) => set('stabilize_beta', v)} />
                  </>
                )}
              </>
            )}
            <Toggle label="Restore original mouth area" checked={!!p.restore_original_mouth} onChange={(v) => set('restore_original_mouth', v)} />
            {p.selected_enhancer && p.selected_enhancer.toLowerCase() === 'codeformer' && (
              <Slider
                label="CodeFormer fidelity weight"
                info="Balances restoration quality vs original identity. 0.1 = maximum sharpness (may shift face geometry), 0.9 = maximum similarity to source (but less restoration detail)."
                min={0.1}
                max={0.9}
                step={0.05}
                value={num(p.codeformer_fidelity, 0.5)}
                onChange={(v) => set('codeformer_fidelity', v)}
              />
            )}
          </Section>

          <Section title="Target file(s)">
            <FileDrop accept="image/*,video/*,.webp" multiple label="Add target media" onFiles={onAddTarget} busy={uploadingTgt} hint="drop images or videos here" />
            {targets.length === 0 ? (
              <div className="h-24 flex items-center justify-center rounded-xl border border-dashed border-white/10 text-xs text-white/35 bg-black/10 select-none">No target media loaded</div>
            ) : (
              <div className="space-y-2 max-h-52 overflow-y-auto pr-1">
                {targets.map((t, i) => {
                  const job = queue.find(j => j.targetName === t.name);
                  let statusLabel = null;
                  let badgeColor = '';
                  if (job) {
                    if (job.status === 'Running') {
                      statusLabel = 'Running';
                      badgeColor = 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10 animate-pulse';
                    } else if (job.status === 'Finished') {
                      statusLabel = 'Finished';
                      badgeColor = 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
                    } else if (job.status === 'Failed') {
                      statusLabel = 'Failed';
                      badgeColor = 'text-red-400 border-red-500/30 bg-red-500/10';
                    }
                  }
                  const isVideo = t.frames > 1;
                  const duration = isVideo && t.fps ? (t.frames / t.fps).toFixed(1) : null;
                  const typeIcon = isVideo ? '🎥' : '🖼️';
                  return (
                    <div key={i}
                      className={`group flex items-center gap-3 px-3.5 py-3 rounded-xl text-sm border transition-all duration-200 cursor-pointer ${selTarget === i ? 'bg-[var(--accent)]/10 border-[var(--accent)]/40' : 'bg-white/[0.02] border-white/[0.06] hover:border-white/15 hover:bg-white/[0.04]'}`}
                      onClick={() => selectTarget(i)}>
                      {/* v=name busts the browser cache: the URL is index-based, so
                          after a removal the same index can point at another file. */}
                      <img
                        src={`${API}/api/target/preview?index=${i}&frame=1&width=96&v=${encodeURIComponent(t.name)}`}
                        alt=""
                        loading="lazy"
                        className="w-12 h-9 shrink-0 rounded-lg object-cover bg-black/40 border border-white/10"
                        onError={(e) => { e.target.style.display = 'none'; if (e.target.nextElementSibling) e.target.nextElementSibling.style.display = 'block'; }}
                      />
                      <div className="text-base shrink-0 opacity-75 hidden">{typeIcon}</div>
                      <div className="flex-1 min-w-0">
                        <span className="truncate block font-bold text-white/90 group-hover:text-white transition-colors">{t.name}</span>
                        <div className="flex items-center gap-2 mt-0.5 text-[10px] font-medium text-white/40">
                          {isVideo ? (
                            <span>{t.frames} frames · {t.fps} FPS{duration ? ` · ${duration}s` : ''}</span>
                          ) : (
                            <span>Static Image</span>
                          )}
                          {statusLabel && (
                            <span className={`text-[8px] uppercase tracking-wider px-1.5 py-0.5 rounded border font-semibold ${badgeColor}`}>
                              {statusLabel}
                            </span>
                          )}
                        </div>
                      </div>
                      <button type="button" title="Remove this target"
                        onClick={(e) => { e.stopPropagation(); removeTarget(i); }}
                        className="h-6 w-6 shrink-0 rounded-full bg-black/50 text-white/60 opacity-0 group-hover:opacity-100 hover:bg-[var(--accent-hover)] hover:text-white transition-all flex items-center justify-center">✕</button>
                    </div>
                  );
                })}
              </div>
            )}
            {targets.length > 0 && (
              <div className="pt-2 border-t border-white/5 flex justify-end">
                <Button size="sm" variant="stop" onClick={async () => { const r = await postJSON('/api/target/clear', {}); setTargets(r.targets); setTargetFaces([]); setTargetGroups([]); setTargetNames([]); setTargetFacesInfo([]); setFaceMapping({}); setPreviewSrc(''); }}>Clear targets</Button>
              </div>
            )}
          </Section>
          
          <Section title="Source images / facesets">
          <FileDrop accept="image/*,.fsz" multiple label="Add source faces" onFiles={onAddSource} busy={uploadingSrc} hint="drop images or .fsz here" />
          <FacesetLibrary
            canSave={sourceFaces.length > 0}
            onLoaded={(r) => { setSourceFaces(r.source_faces || []); if (r.source_faces_info) setSourceFacesInfo(r.source_faces_info); }}
            notify={notify} />
          <FaceGallery title="Input faces" faces={sourceFaces} selected={selSource} onSelect={selectSource} draggable={true} large={true}
            onRemove={(i) => removeSource(i)} empty="Upload a face image" info={sourceFacesInfo} />
          {sourceFaces.length > 0 && (
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/move', { index: selSource, direction: 'left' })}>⬅ Move</Button>
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/move', { index: selSource, direction: 'right' })}>Move ➡</Button>
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/remove', { index: selSource })}>❌ Remove</Button>
                <Button size="sm" variant="secondary" title="Set each tile to the most frontal face in its set" onClick={() => sourceAction('/api/source/refresh_thumbs', {})}>🙂 Frontal thumb</Button>
                <Button size="sm" variant="stop" onClick={() => sourceAction('/api/source/clear', {})}>Clear all</Button>
              </div>
              
              {sourceFacesInfo[selSource] && (
                <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 text-xs select-none">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-[10px] uppercase tracking-[0.14em] text-white/50">📁 Selected source details</span>
                    <span className="px-2 py-0.5 rounded-full bg-[var(--accent)]/10 text-[10px] text-[var(--accent)] font-bold border border-[var(--accent)]/20">
                      {sourceFacesInfo[selSource].count > 1 ? `${sourceFacesInfo[selSource].count} Reference Faces` : 'Single Face'}
                    </span>
                  </div>
                  
                  <div className="space-y-1.5 pt-1">
                    {sourceFacesInfo[selSource].count > 1 ? (
                      <>
                        <div className="text-[10px] font-bold text-white/40 mb-1">Pose Coverage Breakdown:</div>
                        <div className="flex flex-wrap gap-1.5">
                          {Object.entries(
                            sourceFacesInfo[selSource].poses.reduce((acc, p) => {
                              acc[p] = (acc[p] || 0) + 1;
                              return acc;
                            }, {})
                          ).map(([pose, cnt]) => (
                            <span key={pose} className="px-2 py-1 rounded-lg bg-white/[0.03] border border-white/5 text-[10px] text-white/70">
                              {pose} <span className="text-[var(--accent)] font-extrabold">({cnt})</span>
                            </span>
                          ))}
                        </div>
                      </>
                    ) : (
                      <div className="flex items-center justify-between text-white/60">
                        <span>Detected Pose:</span>
                        <span className="font-bold text-white">{sourceFacesInfo[selSource].poses[0] || 'Front'}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </Section>
      </div>

        {/* COLUMN 3: Active Canvas, Timeline & Outputs */}
        <div className="flex-1 min-w-0 space-y-6">
          {/* run bar */}
          <div className="sticky top-20 z-30 pb-3 bg-[#0c0e14]/90 backdrop-blur-md">
            {progress.processing ? (() => {
              const radius = 21;
              const circumference = radius * 2 * Math.PI;
              const strokeDashoffset = circumference - prog * circumference;
              return (
                <div className="relative overflow-hidden rounded-2xl glass-panel px-5 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4 shadow-xl border border-white/5 w-full">
                  {/* Left: Circular Progress Ring & Info */}
                  <div className="flex items-center gap-3.5 min-w-0">
                    <div className={`relative flex items-center justify-center h-14 w-14 select-none shrink-0 rounded-full transition-shadow duration-1000 ${!progress.paused ? 'shadow-[0_0_14px_var(--accent-glow)]' : ''}`}>
                      <svg className="transform -rotate-90 w-[52px] h-[52px]" viewBox="0 0 48 48">
                        <circle
                          stroke="rgba(255, 255, 255, 0.08)"
                          fill="transparent"
                          strokeWidth={3.5}
                          r={radius}
                          cx={24}
                          cy={24}
                        />
                        <circle
                          className="transition-all duration-500 ease-out"
                          stroke="var(--accent)"
                          fill="transparent"
                          strokeWidth={3.5}
                          strokeDasharray={`${circumference} ${circumference}`}
                          style={{ strokeDashoffset, filter: 'drop-shadow(0 0 4px var(--accent-glow))' }}
                          r={radius}
                          cx={24}
                          cy={24}
                          strokeLinecap="round"
                        />
                      </svg>
                      <AnimatedNumber value={prog * 100} decimals={0} suffix="%" className="absolute text-[13px] font-extrabold text-white tabular-nums" />
                    </div>

                    <div className="space-y-0.5 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className={`h-2 w-2 rounded-full ${progress.paused ? 'bg-amber-400' : 'bg-[var(--accent)] animate-ping'}`} />
                        <span className={`text-[11px] font-semibold uppercase tracking-[0.14em] ${progress.paused ? 'text-amber-400' : 'text-[var(--accent)]'}`}>
                          {progress.paused ? 'Paused' : 'Processing'}
                        </span>
                      </div>
                      <div className="text-sm font-bold text-white truncate max-w-[340px]">
                        {progress.desc || 'Swapping faces…'}
                      </div>
                      {progress.error && <div className="text-xs text-red-400 font-semibold">{progress.error}</div>}
                    </div>
                  </div>

                  {/* Elapsed / ETA compact readout (the full telemetry HUD now
                      lives in the Preview panel, so the run-bar stays a slim strip). */}
                  <div className="flex items-center gap-3 text-xs font-mono shrink-0">
                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-white/40 font-bold">Elapsed</span>
                      <span className="text-white font-bold tabular-nums whitespace-nowrap">{fmtTime(elapsedMs)}</span>
                    </div>
                    <div className="h-6 w-px bg-white/10" />
                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-white/40 font-bold">ETA</span>
                      <span className="text-emerald-400 font-bold tabular-nums whitespace-nowrap">{etaMs > 0 ? fmtTime(etaMs) : '--:--'}</span>
                    </div>
                  </div>

                  {/* Right: big icon action buttons */}
                  <div className="flex items-center gap-3 shrink-0">
                    {progress.paused ? (
                      <motion.button type="button" onClick={resume} title="Resume (Space)"
                        whileHover={{ y: -3, scale: 1.06 }} whileTap={{ scale: 0.92, y: 0 }} transition={spring.snappy}
                        className="group flex flex-col items-center gap-1.5 focus:outline-none">
                        <span className="h-11 w-11 rounded-xl flex items-center justify-center bg-emerald-500/15 border border-emerald-500/40 text-emerald-400 transition-colors duration-200 group-hover:bg-emerald-500/25">
                          <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor"><path d="M8 5.14v13.72a1 1 0 0 0 1.53.85l10.9-6.86a1 1 0 0 0 0-1.7L9.53 4.29A1 1 0 0 0 8 5.14z" /></svg>
                        </span>
                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/45 group-hover:text-emerald-400 transition-colors">Resume</span>
                      </motion.button>
                    ) : (
                      <motion.button type="button" onClick={pause} title="Pause (Space)"
                        whileHover={{ y: -3, scale: 1.06 }} whileTap={{ scale: 0.92, y: 0 }} transition={spring.snappy}
                        className="group flex flex-col items-center gap-1.5 focus:outline-none">
                        <span className="h-11 w-11 rounded-xl flex items-center justify-center bg-amber-500/15 border border-amber-500/40 text-amber-400 transition-colors duration-200 group-hover:bg-amber-500/25">
                          <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1.5" /><rect x="14" y="5" width="4" height="14" rx="1.5" /></svg>
                        </span>
                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/45 group-hover:text-amber-400 transition-colors">Pause</span>
                      </motion.button>
                    )}
                    <motion.button type="button" onClick={stop} title="Stop"
                      whileHover={{ y: -3, scale: 1.06 }} whileTap={{ scale: 0.92, y: 0 }} transition={spring.snappy}
                      className="group flex flex-col items-center gap-1.5 focus:outline-none">
                      <span className="h-11 w-11 rounded-xl flex items-center justify-center bg-red-500/15 border border-red-500/40 text-red-400 transition-colors duration-200 group-hover:bg-red-500/25">
                        <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2.5" /></svg>
                      </span>
                      <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/45 group-hover:text-red-400 transition-colors">Stop</span>
                    </motion.button>
                  </div>

                  {/* Smooth animated progress line along the bottom edge */}
                  <div className="absolute inset-x-0 bottom-0 h-1 bg-white/[0.04]">
                    <div
                      className={`h-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] transition-[width] duration-500 ease-out ${progress.paused ? '' : 'progress-bar-animated'}`}
                      style={{ width: `${Math.max(2, prog * 100)}%`, boxShadow: '0 0 10px var(--accent-glow)' }}
                    />
                  </div>
                </div>
              );
            })() : (
             <div className="w-full space-y-4">
              <div className="rounded-2xl glass-panel p-6 flex flex-col md:flex-row items-center justify-between gap-6 shadow-2xl border border-white/5 w-full">
                <div className="flex items-center gap-3 w-full md:w-auto">
                  <Button variant="primary" size="lg" onClick={start} disabled={targets.length === 0 || sourceFaces.length === 0} className="w-full md:w-auto justify-center">▶ Start Swapping</Button>
                  {maxFrames > 1 && (
                    <Button variant="secondary" size="lg" onClick={renderPreviewClip} disabled={targets.length === 0 || sourceFaces.length === 0 || isGeneratingPreviewClip} className="!text-orange-400 border border-orange-500/20 hover:bg-orange-500/10 w-full md:w-auto justify-center">
                      ⚡ Render 5s Preview
                    </Button>
                  )}
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2.5 text-sm font-semibold text-[var(--text-muted)] max-w-xs truncate text-right">
                    <span className={`h-2.5 w-2.5 rounded-full ${targets.length > 0 && sourceFaces.length > 0 ? 'bg-emerald-500 animate-pulse shadow-[0_0_8px_#10b981]' : 'bg-red-500/50'}`} />
                    {targets.length === 0 ? 'No target media selected' : sourceFaces.length === 0 ? 'No source faces loaded' : 'Ready to swap'}
                  </div>
                </div>
              </div>
             </div>
            )}
          </div>

          <Section title="Preview" tilt={false} glare={false} hover={false}>
            {/* While a job runs we no longer stream live swapped frames into the
                preview box (they thrashed the GPU and jittered). Instead we show a
                progress panel that mirrors the terminal: percent, frame X / Y,
                FPS (from progress.desc), elapsed and time-left. previewSrc is
                React-only state and is empty after a remount, so the processing
                branch is keyed off progress.processing rather than previewSrc. */}
            {progress.processing ? (
              /* Fills the viewport rather than a 16:9 slot: with the settings
                 sidebar, asset rail, timeline and preview controls all hidden
                 during a run, this panel IS the screen, and the diagnostics
                 want the height. Floored so it stays usable on a short window. */
              <div className="relative h-[calc(100vh-230px)] min-h-[620px] rounded-2xl overflow-hidden processing-stage flex flex-col items-center select-none px-4 sm:px-6 py-4">
                {/* h-full + min-h-0 so the console below takes ALL the leftover
                    height instead of the whole block floating in a tall box. */}
                <div className="relative h-full w-full max-w-[1900px] min-h-0 flex flex-col gap-3">

                  {/* ── Headline ──────────────────────────────────────────────
                      One line that answers "where is it, and when is it done".
                      The old panel spread that over a centred status line, a
                      percentage above the bar and an ETA below it, so reading
                      the run meant collecting three separate scraps. */}
                  <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={`text-[11px] font-semibold uppercase tracking-[0.18em] ${progress.paused ? 'text-amber-400' : 'text-[var(--accent)]'}`}>
                          {progress.paused ? 'Paused' : 'Processing'}
                        </span>
                        {!progress.paused && <span className="h-px w-8 bg-[var(--accent)]/40" />}
                      </div>
                      <div className="mt-1 flex items-baseline gap-2.5">
                        <AnimatedNumber value={prog * 100} decimals={1} suffix="%"
                                        className="font-mono text-[34px] leading-none font-bold tabular-nums text-white" />
                        <span className="text-sm font-medium text-white/55 truncate max-w-[46ch]">
                          {progress.desc || 'Swapping faces…'}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-stretch gap-5 font-mono">
                      <div className="text-right">
                        <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-white/30">Elapsed</div>
                        <div className="text-[17px] font-bold tabular-nums text-white/85">{fmtTime(elapsedMs)}</div>
                      </div>
                      <div className="w-px bg-white/10" />
                      <div className="text-right">
                        <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-white/30">Time left</div>
                        <div className="text-[17px] font-bold tabular-nums text-emerald-400">{etaMs > 0 ? fmtTime(etaMs) : '--:--'}</div>
                      </div>
                      <div className="w-px bg-white/10" />
                      <div className="text-right">
                        <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-white/30">Finishes</div>
                        <div className="text-[17px] font-bold tabular-nums text-white/85">
                          {etaMs > 0 ? new Date(Date.now() + etaMs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* ── Pipeline rail ─────────────────────────────────────────
                      The stages ARE the progress bar: one continuous track split
                      into named segments that fill as the run moves through them,
                      instead of a plain bar with a separate row of chips
                      restating the same thing. */}
                  {(() => {
                    const d = (progress.desc || '').toLowerCase();
                    const stages = [
                      { key: 'analyze', label: 'Analyze' },
                      { key: 'swap', label: 'Swap' },
                      ...(p.upscale_after_swap ? [{ key: 'upscale', label: 'Upscale' }] : []),
                      { key: 'combine', label: 'Combine' },
                    ];
                    let activeKey = 'swap';
                    if (/combin|finaliz|encod|audio|mux/.test(d)) activeKey = 'combine';
                    else if (/upscal/.test(d)) activeKey = 'upscale';
                    else if (/processing frame|swapp/.test(d)) activeKey = 'swap';
                    else if (/analy|track|extract|detect|start/.test(d)) activeKey = 'analyze';
                    let activeIdx = stages.findIndex((s) => s.key === activeKey);
                    if (activeIdx < 0) activeIdx = 1;
                    return (
                      <div className="w-full">
                        <div className="flex items-stretch gap-1">
                          {stages.map((s, i) => {
                            const state = i < activeIdx ? 'done' : i === activeIdx ? 'active' : 'pending';
                            return (
                              <div key={s.key} className="flex-1 min-w-0">
                                <div className={`h-1.5 rounded-full overflow-hidden ${state === 'pending' ? 'bg-white/[0.06]' : 'bg-white/[0.08]'}`}>
                                  {state === 'done' && <div className="h-full w-full rounded-full bg-emerald-500/70" />}
                                  {state === 'active' && (
                                    <div className={`h-full rounded-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] transition-[width] duration-500 ease-out ${progress.paused ? '' : 'progress-bar-animated'}`}
                                         style={{ width: `${Math.max(4, prog * 100)}%` }} />
                                  )}
                                </div>
                                <div className={`mt-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] truncate ${
                                  state === 'done' ? 'text-emerald-400/70'
                                  : state === 'active' ? 'text-white'
                                  : 'text-white/25'}`}>
                                  {state === 'done' && <span aria-hidden>✓</span>}
                                  <span className="truncate">{s.label}</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })()}

                  {/* Diagnostics — throughput trend, GPU/VRAM/CPU, per-stage cost
                      and the settings in force. Fills what used to be empty box. */}
                  <DiagnosticsPanel
                    desc={progress.desc}
                    telemetry={telemetry}
                    processing={progress.processing}
                    paused={progress.paused}
                    config={runConfigSummary}
                    elapsedMs={elapsedMs}
                    etaMs={etaMs}
                    prog={prog}
                  />

                  {/* Live terminal feed — mirrors what the real console prints
                      (stage changes, per-frame FPS, encode/combine, done/errors). */}
                  <ProcessingTerminal log={progress.log || []} paused={progress.paused} className="flex-1 min-h-0" bodyClass="h-full" />

                  {progress.error && <div className="text-xs text-red-400 font-semibold text-center">{progress.error}</div>}
                </div>
              </div>
            ) : previewSrc ? (
              comparingEnhancers ? (() => {
                const activeList = selectedGridEnhancers.filter(e => meta.enhancers?.includes(e));
                const gridColsClass = activeList.length === 1 ? 'grid-cols-1' : 'grid-cols-2';
                return (
                  <div className="space-y-4">
                    {/* Enhancer selector row */}
                    <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 select-none">
                      <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/40 block">📊 Compare Enhancers (Select up to 4)</span>
                      <div className="flex flex-wrap gap-2">
                        {meta.enhancers?.map((enh) => {
                          const isSelected = selectedGridEnhancers.includes(enh);
                          return (
                            <button
                              key={enh}
                              type="button"
                              onClick={() => {
                                if (isSelected) {
                                  if (selectedGridEnhancers.length > 1) {
                                    setSelectedGridEnhancers(prev => prev.filter(x => x !== enh));
                                  }
                                } else {
                                  if (selectedGridEnhancers.length >= 4) {
                                    notify('You can select a maximum of 4 enhancers for grid comparison.', 'warning');
                                  } else {
                                    setSelectedGridEnhancers(prev => [...prev, enh]);
                                  }
                                }
                              }}
                              className={`px-3 py-1.5 rounded-lg text-[11px] font-semibold border transition-all duration-200 ${isSelected ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white' : 'bg-white/[0.02] border-white/10 text-white/50 hover:border-white/20 hover:text-white/85'}`}
                            >
                              {enh}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <CompareGrid
                      items={activeList}
                      gridColsClass={gridColsClass}
                      previews={enhancerPreviews}
                      times={enhancerTimes}
                      timers={liveRenderingTimers}
                    />
                  </div>
                );
              })() : comparingMasks ? (() => {
                const activeMasks = selectedGridMasks.filter(m => meta.mask_engines?.includes(m));
                const gridColsClass = activeMasks.length === 1 ? 'grid-cols-1' : 'grid-cols-2';
                return (
                  <div className="space-y-4">
                    {/* Mask-engine selector row */}
                    <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 select-none">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/40 block">🎭 Compare Mask Engines (Select up to 4)</span>
                        <span className="text-[10px] text-white/30">Enhancer: <span className="text-white/55 font-semibold">{p.selected_enhancer || 'None'}</span></span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {meta.mask_engines?.map((mE) => {
                          const isSelected = selectedGridMasks.includes(mE);
                          return (
                            <button
                              key={mE}
                              type="button"
                              onClick={() => {
                                if (isSelected) {
                                  if (selectedGridMasks.length > 1) {
                                    setSelectedGridMasks(prev => prev.filter(x => x !== mE));
                                  }
                                } else {
                                  if (selectedGridMasks.length >= 4) {
                                    notify('You can select a maximum of 4 mask engines for grid comparison.', 'warning');
                                  } else {
                                    setSelectedGridMasks(prev => [...prev, mE]);
                                  }
                                }
                              }}
                              className={`px-3 py-1.5 rounded-lg text-[11px] font-semibold border transition-all duration-200 ${isSelected ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white' : 'bg-white/[0.02] border-white/10 text-white/50 hover:border-white/20 hover:text-white/85'}`}
                            >
                              {mE}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <CompareGrid
                      items={activeMasks}
                      gridColsClass={gridColsClass}
                      previews={maskPreviews}
                      times={maskTimes}
                      timers={maskRenderTimers}
                    />
                  </div>
                );
              })() : comparingSwappers ? (() => {
                const activeSwappers = selectedGridSwappers.filter(m => meta.swap_models?.includes(m));
                const gridColsClass = activeSwappers.length === 1 ? 'grid-cols-1' : 'grid-cols-2';
                return (
                  <div className="space-y-4">
                    {/* Swapper-model selector row */}
                    <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 select-none">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/40 block">🔀 Compare Swapper Models (Select up to 4)</span>
                        <span className="text-[10px] text-white/30">Enhancer: <span className="text-white/55 font-semibold">{p.selected_enhancer || 'None'}</span></span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {meta.swap_models?.map((sM) => {
                          const isSelected = selectedGridSwappers.includes(sM);
                          return (
                            <button
                              key={sM}
                              type="button"
                              onClick={() => {
                                if (isSelected) {
                                  if (selectedGridSwappers.length > 1) {
                                    setSelectedGridSwappers(prev => prev.filter(x => x !== sM));
                                  }
                                } else {
                                  if (selectedGridSwappers.length >= 4) {
                                    notify('You can select a maximum of 4 swapper models for grid comparison.', 'warning');
                                  } else {
                                    setSelectedGridSwappers(prev => [...prev, sM]);
                                  }
                                }
                              }}
                              className={`px-3 py-1.5 rounded-lg text-[11px] font-semibold border transition-all duration-200 ${isSelected ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white' : 'bg-white/[0.02] border-white/10 text-white/50 hover:border-white/20 hover:text-white/85'}`}
                            >
                              {sM}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <CompareGrid
                      items={activeSwappers}
                      gridColsClass={gridColsClass}
                      previews={swapperPreviews}
                      times={swapperTimes}
                      timers={swapperRenderTimers}
                    />
                  </div>
                );
              })() : comparingUpscalers ? (() => {
                const activeUpscalers = selectedGridUpscalers.filter(l => AI_UPSCALE_MODELS.some(m => m.label === l));
                const gridColsClass = activeUpscalers.length === 1 ? 'grid-cols-1' : 'grid-cols-2';
                return (
                  <div className="space-y-4">
                    {/* AI-upscaler selector row */}
                    <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 select-none">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/40 block">🔎 Compare AI Upscalers (Select up to 4)</span>
                        <span className="text-[10px] text-white/30">Swaps once, then upscales each</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {AI_UPSCALE_MODELS.map((m) => {
                          const isSelected = selectedGridUpscalers.includes(m.label);
                          return (
                            <button
                              key={m.value}
                              type="button"
                              onClick={() => {
                                if (isSelected) {
                                  if (selectedGridUpscalers.length > 1) {
                                    setSelectedGridUpscalers(prev => prev.filter(x => x !== m.label));
                                  }
                                } else {
                                  if (selectedGridUpscalers.length >= 4) {
                                    notify('You can select a maximum of 4 upscalers for grid comparison.', 'warning');
                                  } else {
                                    setSelectedGridUpscalers(prev => [...prev, m.label]);
                                  }
                                }
                              }}
                              className={`px-3 py-1.5 rounded-lg text-[11px] font-semibold border transition-all duration-200 ${isSelected ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white' : 'bg-white/[0.02] border-white/10 text-white/50 hover:border-white/20 hover:text-white/85'}`}
                            >
                              {m.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <CompareGrid
                      items={activeUpscalers}
                      gridColsClass={gridColsClass}
                      previews={upscalePreviews}
                      times={upscaleTimes}
                      timers={upscaleRenderTimers}
                    />
                  </div>
                );
              })() : (
                <InteractivePreview
                  beforeSrc={(isPlaying && bufferedSrc) ? bufferedSrc : rawUrl}
                  afterSrc={(!isScrubbing && !isPlaying) ? previewSrc : ((isPlaying && bufferedSrc) ? bufferedSrc : (getCachedPreview(selTarget, frame)?.image || rawUrl))}
                  scrubbing={scrubbingNow}
                  faces={previewFaces}
                  personIds={previewPersonIds}
                  onSelectPerson={addPersonFromBox}
                  splitView={splitView}
                  compare={compare}
                  onToggleCompare={() => setCompare((v) => { const n = !v; if (n) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); } return n; })}
                  frame={frame}
                  setFrame={setFrame}
                  maxFrames={maxFrames}
                  isPlaying={isPlaying}
                  previewing={previewing}
                  previewSecs={previewSecs}
                  setIsPlaying={setIsPlaying}
                />
              )
            ) : (
              <div className="relative aspect-video rounded-2xl overflow-hidden bg-gradient-to-br from-white/[0.03] to-black/20 border border-white/10 flex items-center justify-center select-none">
                <div className="absolute inset-0 pointer-events-none opacity-70" style={{ background: 'radial-gradient(circle at 50% 42%, var(--accent-glow), transparent 62%)' }} />
                {previewing ? (
                  <div className="relative flex flex-col items-center gap-3">
                    <div className="h-9 w-9 rounded-full border-2 border-white/10 border-t-[var(--accent)] animate-spin" />
                    <span className="text-sm font-medium text-white/50">Rendering preview…</span>
                  </div>
                ) : (() => {
                  const hasTarget = targets.length > 0;
                  const hasSource = sourceFaces.length > 0;
                  const ready = hasTarget && hasSource;
                  const steps = [
                    { done: hasTarget, label: 'Add a target image or video' },
                    { done: hasSource, label: 'Add a source face' },
                  ];
                  return (
                    <div className="relative flex flex-col items-center gap-4 text-center px-6 w-full max-w-sm">
                      <div className={`grid place-items-center h-14 w-14 rounded-2xl border text-2xl ${ready ? 'bg-emerald-500/10 border-emerald-500/25' : 'bg-[var(--accent)]/10 border-[var(--accent)]/20'}`}>
                        {ready ? '✨' : '🎭'}
                      </div>
                      <div>
                        <div className="text-sm font-semibold text-white/85">{ready ? 'Ready to preview' : 'No preview yet'}</div>
                        <div className="text-xs text-white/40 mt-1 leading-relaxed">
                          {ready
                            ? 'Scrub the timeline or press ▶ Start Swapping to render.'
                            : 'Two quick steps and the live swap shows up right here.'}
                        </div>
                      </div>
                      {!ready && (
                        <div className="w-full space-y-1.5">
                          {steps.map((s, i) => (
                            <div key={i} className={`flex items-center gap-2.5 px-3 py-2 rounded-lg border text-xs font-medium transition-colors ${s.done ? 'border-emerald-500/25 bg-emerald-500/[0.06] text-emerald-300/90' : 'border-white/10 bg-white/[0.02] text-white/55'}`}>
                              <span className={`grid place-items-center h-4 w-4 rounded-full text-[9px] font-bold shrink-0 ${s.done ? 'bg-emerald-500 text-black' : 'bg-white/10 text-white/50'}`}>{s.done ? '✓' : i + 1}</span>
                              <span className={s.done ? 'line-through opacity-70' : ''}>{s.label}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })()}
              </div>
            )}
            
            {/* Clip timeline — hidden while running: scrubbing a clip that is
                being written is meaningless, and the space belongs to the
                diagnostics. */}
            {maxFrames > 1 && !progress.processing && (
              <div className="pt-4 border-t border-[var(--border-color)]">
                <Timeline
                  fps={targets[selTarget]?.fps || 25}
                  maxFrames={maxFrames}
                  frame={frame}
                  setFrame={setFrame}
                  startFrame={startFrame}
                  endFrame={endFrame}
                  setFrameMarkerVal={setFrameMarkerVal}
                  timelineRef={timelineRef}
                  onPointerDown={handleTimelinePointerDown}
                  onPointerMove={handleTimelinePointerMove}
                  onPointerLeave={handleTimelinePointerLeave}
                  hoverFrame={hoverFrame}
                  isScrubbing={isScrubbing}
                  storyboardThumbs={storyboardThumbs}
                  view={view}
                  setView={setView}
                  isPlaying={isPlaying}
                  setIsPlaying={setIsPlaying}
                  buffering={playStalled}
                  isLooping={isLooping}
                  setIsLooping={setIsLooping}
                  playbackRate={playbackRate}
                  setPlaybackRate={setPlaybackRate}
                  thumbUrl={(f) => `${API}/api/target/preview?index=${selTarget}&frame=${f}&width=384`}
                  targetKey={targets[selTarget]?.name || String(selTarget)}
                />
              </div>
            )}

            {/* Preview controls — every one of these drives a preview render,
                which is exactly what must not happen mid-run, so they go away
                with everything else while a job is running. */}
            <div className={`flex items-center flex-wrap gap-3 ${maxFrames > 1 ? 'pt-3 border-t border-white/5' : ''} ${progress.processing ? 'hidden' : ''}`}>
              <Button size="sm" variant="secondary" title="Re-run the swap for this frame, ignoring the cached result" onClick={() => refreshPreview({ force: true })}>🔄 Refresh</Button>
              <Button size="sm" variant="primary" onClick={useFaceFromFrame}>Use face from frame</Button>
              {previewSrc && !comparingEnhancers && !comparingMasks && !comparingSwappers && !comparingUpscalers && (
                <Button size="sm" variant="secondary" disabled={upscaling} onClick={upscaleThisFrame}
                  title="AI-upscale just this frame to preview final quality">
                  {upscaling ? '🔎 Upscaling…' : `🔎 Upscale this frame (${AI_UPSCALE_MODELS.find(m => m.value === (p.upscale_model_after || 'esrganx2'))?.label || 'AI'})`}
                </Button>
              )}
            </div>

            <div className={`flex items-center flex-wrap gap-3 ${progress.processing ? 'hidden' : ''}`}>
              <Toggle label="✨ Live Swap" checked={fakePreview} onChange={setFakePreview} />
              <Toggle label="🔍 Compare" checked={compare} onChange={(v) => { setCompare(v); if (v) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); } }} />
              {compare && <Toggle label="Split View" checked={splitView} onChange={setSplitView} />}
              <Toggle label="📊 Enhancer Grid" checked={comparingEnhancers} onChange={(v) => { setComparingEnhancers(v); if (v) { setCompare(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); } }} />
              <Toggle label="🎭 Mask Grid" checked={comparingMasks} onChange={(v) => { setComparingMasks(v); if (v) { setCompare(false); setComparingEnhancers(false); setComparingSwappers(false); setComparingUpscalers(false); } }} />
              <Toggle label="🔀 Swapper Grid" checked={comparingSwappers} onChange={(v) => { setComparingSwappers(v); if (v) { setCompare(false); setComparingEnhancers(false); setComparingMasks(false); setComparingUpscalers(false); } }} />
              <Toggle label="🔎 Upscale Grid" checked={comparingUpscalers} onChange={(v) => { setComparingUpscalers(v); if (v) { setCompare(false); setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); } }} />
            </div>
          </Section>

          {/* Single-frame AI-upscale spot-check modal — portaled to <body> so
              transformed ancestors (TiltCard / motion) can't clip the fixed
              overlay. */}
          {upscaledSrc && createPortal((
            <div
              className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-6"
              onClick={() => setUpscaledSrc('')}
            >
              <div
                className="relative flex flex-col max-w-[95vw] max-h-[92vh] rounded-2xl border border-white/10 bg-[var(--card-bg)] shadow-2xl overflow-hidden"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-center justify-between gap-4 px-4 py-2.5 border-b border-white/10">
                  <span className="text-xs font-semibold text-white/80">
                    🔎 Upscaled frame{upscaledDims ? ` · ${upscaledDims.w}×${upscaledDims.h}` : ''}
                  </span>
                  <div className="flex items-center gap-2">
                    <a
                      href={upscaledSrc}
                      download={`upscaled_frame_${frame}.png`}
                      className="px-3 py-1.5 rounded-lg text-[11px] font-semibold border border-white/10 bg-white/[0.04] text-white/80 hover:border-white/25 hover:text-white transition-colors"
                    >
                      ⬇ Download
                    </a>
                    <button
                      type="button"
                      onClick={() => setUpscaledSrc('')}
                      className="px-3 py-1.5 rounded-lg text-[11px] font-semibold border border-white/10 bg-white/[0.04] text-white/80 hover:border-white/25 hover:text-white transition-colors"
                    >
                      ✕ Close
                    </button>
                  </div>
                </div>
                <div className="overflow-auto flex-1 min-h-0">
                  <img src={upscaledSrc} alt="Upscaled frame" className="block max-w-none" />
                </div>
              </div>
            </div>
          ), document.body)}

          {targets.length > 0 && sourceFaces.length > 0 && estFrames > 1 && (
            <TiltCard className="rounded-2xl w-full" max={6}>
            <div className="rounded-2xl glass-panel p-5 shadow-2xl border border-white/5 w-full">
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] uppercase tracking-[0.14em] text-white/40 font-semibold">Runtime estimation</span>
                <span className={`text-[10px] font-semibold uppercase tracking-wide px-2 py-0.5 rounded-full border ${estSourceClass}`}>{estSourceLabel}</span>
              </div>
              <div className="flex items-end gap-3 mb-4">
                <span className="text-3xl font-bold text-white/95 tabular-nums leading-none">~{fmtTime(estTotalMs)}</span>
                <span className="text-xs text-white/40 mb-0.5">{Math.round(estPerFrame)} ms/frame{heavyVram ? ' · high VRAM' : ''}</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-5 gap-y-2.5 text-[11px]">
                {[
                  ['Frames', estFrames.toLocaleString()],
                  ['Duration', estDurationS ? `${estDurationS.toFixed(1)}s @ ${estFps} fps` : '—'],
                  ['Faces / frame', `${previewFaces.length || '—'}${calibEst?.density_bucket ? ` (${calibEst.density_bucket})` : ''}`],
                  ['This combo', estLearned ? `${calibEst.samples} run${calibEst.samples > 1 ? 's' : ''}` : 'no data yet'],
                  ['GPU', calibEst?.gpu || '—'],
                  ['Threads', calibEst?.threads ?? '—'],
                  ['Precision', calibEst?.precision || '—'],
                  ['Learned combos', calibEst?.store?.entries ?? 0],
                  ['Total runs logged', calibEst?.store?.global_samples ?? 0],
                ].map(([k, v]) => (
                  <div key={k} className="flex flex-col leading-tight min-w-0">
                    <span className="text-white/35">{k}</span>
                    <span className="text-white/80 tabular-nums truncate" title={String(v)}>{v}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 pt-3 border-t border-white/5 text-[10px] text-white/35 leading-snug">
                {estLearned
                  ? 'Learned from your completed runs with these settings. Accuracy improves as you process more.'
                  : calibEst?.source === 'global'
                    ? 'No history for this exact settings + face-density combo yet — showing a blend of your overall average and the heuristic. Finish a run to calibrate it.'
                    : 'Heuristic estimate. Finish a run with these settings to start learning the real speed.'}
              </div>
            </div>
            </TiltCard>
          )}

          {/* Batch queue — hidden while running UNLESS a queue is what's running,
              because then this holds the only controls that stop the QUEUE
              rather than just the current job, and hiding them would let the
              next job start after a Stop. */}
          <div className={progress.processing && !isQueueRunning ? 'hidden' : ''}>
          <Section title="Batch Swapping Queue">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-3">
              <div className="text-xs text-white/50">
                {queue.length === 0
                  ? 'No jobs in queue. Configure settings & click "Add current to queue".'
                  : `${queue.length} jobs queued · ${queue.filter(j => j.status === 'Finished').length} finished`}
              </div>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="secondary" onClick={addToQueue} disabled={targets.length === 0 || sourceFaces.length === 0}>➕ Add current to queue</Button>
                {queue.length > 0 && (
                  <>
                    {isQueueRunning ? (
                      <>
                        <Button size="sm" variant="secondary" onClick={queuePaused ? resumeQueue : pauseQueue}>
                          {queuePaused ? '▶ Resume queue' : '⏸ Pause queue'}
                        </Button>
                        <Button size="sm" variant="stop" onClick={stopQueue}>⏹ Stop queue</Button>
                      </>
                    ) : (
                      <Button size="sm" variant="primary" onClick={startQueue}>▶ Start queue</Button>
                    )}
                    <Button size="sm" variant="ghost" className="text-red-400" onClick={clearQueue}>Clear</Button>
                  </>
                )}
              </div>
            </div>

            {queue.length > 0 && (
              <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
                {queue.map((job, idx) => {
                  const statusColors = {
                    Pending: 'text-white/60 bg-white/5 border-white/5',
                    Running: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30 animate-pulse shadow-[0_0_8px_rgba(234,179,8,0.2)]',
                    Paused: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
                    Finished: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
                    Failed: 'text-red-400 bg-red-500/10 border-red-500/30',
                  };
                  // Show the in-flight job as "Paused" while the queue is held.
                  const displayStatus = (queuePaused && job.status === 'Running') ? 'Paused' : job.status;
                  return (
                    <div key={job.id} className={`flex items-center justify-between px-3 py-2 rounded-xl text-xs border ${statusColors[displayStatus] || 'text-white bg-white/5'}`}>
                      <div className="flex-1 min-w-0 pr-3">
                        <span className="font-semibold text-white block truncate">{idx + 1}. {job.targetName}</span>
                        <span className="opacity-75 text-[10px] block truncate">Source: {job.sourceName} · Enhancer: {job.params.selected_enhancer || 'None'}</span>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <span className="font-bold uppercase text-[9px] tracking-wider px-2 py-0.5 rounded bg-black/30 border border-white/5">{displayStatus}</span>
                        {!isQueueRunning && (
                          <button onClick={() => removeFromQueue(job.id)} className="text-white/40 hover:text-red-400 font-bold" title="Remove job">✕</button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Section>
          </div>

          <Section title="Output settings & renders" className={progress.processing ? 'hidden' : ''}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
              <div className="space-y-4">
                <Select label="Output method" value={p.output_method} onChange={(v) => set('output_method', v)} options={meta.output_methods} />
              </div>
              {out?.path && (
                <div className="space-y-2">
                  <div className="text-xs text-[var(--text-muted)]">Latest output</div>
                  {out.kind === 'video'
                    ? <video src={outUrl} controls className="w-full rounded-xl border border-white/5" />
                    : <img src={outUrl} alt="output" className="w-full rounded-xl border border-white/5" />}
                  <div className="flex flex-wrap gap-2">
                    <a href={outUrl} download
                      className="inline-block px-3 py-1.5 rounded-xl text-sm bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white font-bold transition-colors">⬇ Download</a>
                    <Button size="sm" variant="secondary" onClick={revealOutput}>📂 Open folder</Button>
                  </div>
                  <QualityReport outputPath={out.path} notify={notify} />
                </div>
              )}
            </div>
          </Section>

        </div>
      </div>

      {/* Paste routing dialog */}
      {pastedFiles && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center animate-slide-up">
          <Card className="p-6 max-w-md w-full border border-white/10 shadow-2xl flex flex-col gap-4 text-center">
            <h3 className="text-lg font-bold text-white">📋 Clipboard/Dropped File</h3>
            <p className="text-sm text-white/60">Would you like to load <span className="font-semibold text-white">{pastedFiles[0]?.name}</span> as a Source Face or Target Media?</p>
            <div className="flex gap-3 justify-center mt-2">
              <Button variant="primary" onClick={() => { onAddSource(pastedFiles); setPastedFiles(null); }}>🎭 Source Face</Button>
              <Button variant="secondary" onClick={() => { onAddTarget(pastedFiles); setPastedFiles(null); }}>🎞️ Target Media</Button>
              <Button variant="stop" onClick={() => setPastedFiles(null)}>Cancel</Button>
            </div>
          </Card>
        </div>
      )}

      {/* Drag-n-Drop overlay removed (managed globally by App.jsx) */}

      {/* Keyboard Shortcuts HUD */}
      {showShortcutHUD && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center animate-slide-up" onClick={() => setShowShortcutHUD(false)}>
          <Card className="p-6 max-w-lg w-full border border-white/10 shadow-2xl flex flex-col gap-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">⌨️ Pro Keyboard Shortcuts</h3>
              <Button size="sm" variant="ghost" onClick={() => setShowShortcutHUD(false)}>✕</Button>
            </div>
            <div className="grid grid-cols-2 gap-x-8 gap-y-5 py-2 text-sm text-white/80">
              <div className="space-y-2.5">
                <h4 className="font-bold text-[var(--accent)] text-xs uppercase tracking-wider">Playback & Nav</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Play / Pause</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Space</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Prev / Next Frame</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">← / →</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Step 10 Frames</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Shift + ← / →</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Jump to Start/End</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Home / End</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-[var(--accent)] text-xs uppercase tracking-wider">Timeline Trimming</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Set Start Frame</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">[</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Set End Frame</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">]</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Reset Range</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">R</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Marker at playhead</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">M</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Zoom timeline</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Wheel on track</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Pan timeline</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Shift + Wheel</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-[var(--accent)] text-xs uppercase tracking-wider">Compare & Zoom</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Zoom In / Out</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">+ / -</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Toggle Comparison</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">C</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Toggle Split View</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">S</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-[var(--accent)] text-xs uppercase tracking-wider">Queue & Process</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Add to Batch Queue</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Q</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Run Swapper / Queue</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Ctrl + Enter</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Toggle Shortcuts HUD</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">?</kbd></div>
              </div>
            </div>
            <p className="text-xs text-white/40 text-center mt-2">Click anywhere outside this modal or press Esc to close.</p>
          </Card>
        </div>
      )}

    </div>
  );
}
