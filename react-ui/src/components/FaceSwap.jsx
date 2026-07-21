import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { getJSON, postJSON, postFiles, API } from '../api';
import { Section, Select, Slider, Toggle, TextInput, Button, FaceGallery, Card, AnimatedNumber, Skeleton } from './ui';
import PersonGroups from './PersonGroups';
import QualityReport from './QualityReport';
import FileDrop from './faceswap/FileDrop';
import CompareGrid from './faceswap/CompareGrid';
import InteractivePreview from './faceswap/InteractivePreview';
import FacesetLibrary from './faceswap/FacesetLibrary';
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

  const clearPreviewCache = () => {
    previewCacheRef.current = {};
  };

  const getCacheKey = (idx = selTarget, fr = frame) => {
    return `${idx}_${fr}_${previewKey}_${sourceFaces.length}_${targetFaces.length}_${selSource}_${selTargetFace}`;
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

  // Invalidate cache when source/target files or selections change
  useEffect(() => {
    clearPreviewCache();
  }, [sourceFaces.length, targetFaces.length, selSource, selTargetFace]);

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
  const [frameInput, setFrameInput] = useState(null); // non-null while typing a frame to jump to
  const timelineRef = useRef(null);
  // Coalesces timeline hover/scrub pointer-move work to one update per frame.
  // Pointer-move fires faster than this large component can re-render, so
  // without batching the events pile up and scrubbing/hovering feels sticky.
  const timelineRafRef = useRef(null);
  const timelinePendingRef = useRef(null);
  const playIntervalRef = useRef(null);
  const [isGeneratingPreviewClip, setIsGeneratingPreviewClip] = useState(false);
  const [origStartEnd, setOrigStartEnd] = useState(null);


  const previewBusyRef = useRef(false);   // a /api/preview call is in flight
  const previewPendingRef = useRef(null); // latest queued request while busy (coalesced)

  // p = the swap parameters, seeded from CFG (settings) and patched locally.
  const p = settings || {};
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));

  // While face-swap preview is on, auto-refresh when the swapped result would
  // change: new source faces, target faces, or any swap/mask parameter.
  const previewKey = JSON.stringify({
    fp: fakePreview,
    e: p.selected_enhancer, d: p.face_detection_mode, fd: p.max_face_distance,
    br: p.blend_ratio, me: p.mask_engine, ct: p.mask_clip_text, nfa: p.no_face_action,
    vr: p.vr_mode, ar: p.autorotate_faces, smo: p.show_mask_offsets,
    rom: p.restore_original_mouth, ns: p.num_swap_steps, up: p.subsample_upscale,
    r3: p.use_3d_recon, sb: p.use_source_bank, sm: p.swap_model,
    uf: p.use_frontalization, fth: p.frontalization_threshold,
    jr: p.jaw_reshape, jrs: p.jaw_reshape_strength,
    ctm: p.color_transfer_mode, s2: p.sam2_model_size,
    cf_fid: p.codeformer_fidelity,
    rl: p.refine_landmarks, rsf: p.rescue_small_faces, de: p.detector_engine,
    dds: p.default_det_size,
    fds: p.face_detector_size,
    fdt: p.face_detector_threshold,
    fdn: p.face_detector_nms,
    fm: faceMapping,
    mask_top: p.mask_top,
    mask_bottom: p.mask_bottom,
    mask_left: p.mask_left,
    mask_right: p.mask_right,
    face_mask_blend: p.face_mask_blend,
    mouth_mask_blend: p.mouth_mask_blend,
    mouth_top_scale: p.mouth_top_scale,
    mouth_bottom_scale: p.mouth_bottom_scale,
    mouth_left_scale: p.mouth_left_scale,
    mouth_right_scale: p.mouth_right_scale,
  });

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

    // Check client-side preview cache first
    const cached = getCachedPreview(idx, fr);
    if (cached) {
      setPreviewFaces(cached.faces);
      setPreviewPersonIds(cached.personIds || []);
      setPreviewSrc(cached.image);
      return;
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
      const res = await postJSON('/api/preview', {
        index: idx, frame: fr, fake_preview: fake,
        enhancer: p.selected_enhancer, codeformer_fidelity: num(p.codeformer_fidelity, 0.5),
        detection: p.face_detection_mode,
        face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
        mask_engine: p.mask_engine, clip_text: p.mask_clip_text,
        no_face_action: p.no_face_action, vr_mode: p.vr_mode, autorotate: p.autorotate_faces,
        show_mask_offsets: p.show_mask_offsets, restore_original_mouth: p.restore_original_mouth,
        num_swap_steps: num(p.num_swap_steps, 1), upscale: p.subsample_upscale,
        use_3d_recon: p.use_3d_recon, use_source_bank: p.use_source_bank,
        use_frontalization: p.use_frontalization, frontalization_threshold: num(p.frontalization_threshold, 30),
        jaw_reshape: p.jaw_reshape, jaw_reshape_strength: num(p.jaw_reshape_strength, 0.5),
        swap_model: p.swap_model, default_det_size: p.default_det_size,
        face_detector_size: p.face_detector_size, face_detector_threshold: p.face_detector_threshold,
        face_detector_nms: p.face_detector_nms,
        color_transfer_mode: p.color_transfer_mode, sam2_model_size: p.sam2_model_size,
        refine_landmarks: p.refine_landmarks, rescue_small_faces: p.rescue_small_faces,
        detector_engine: p.detector_engine,
        face_mapping: getFaceMappingArray(),
        mask_top: p.mask_top,
        mask_bottom: p.mask_bottom,
        mask_left: p.mask_left,
        mask_right: p.mask_right,
        face_mask_blend: p.face_mask_blend,
        mouth_mask_blend: p.mouth_mask_blend,
        mouth_top_scale: p.mouth_top_scale,
        mouth_bottom_scale: p.mouth_bottom_scale,
        mouth_left_scale: p.mouth_left_scale,
        mouth_right_scale: p.mouth_right_scale,
      }, { signal: ctrl.signal });
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
      const cacheKey = `${selTarget}_${frame}_${JSON.stringify({
        fp: fakePreview,
        e: localParams.selected_enhancer, d: localParams.face_detection_mode, fd: localParams.max_face_distance,
        br: localParams.blend_ratio, me: localParams.mask_engine, ct: localParams.mask_clip_text, nfa: localParams.no_face_action,
        vr: localParams.vr_mode, ar: localParams.autorotate_faces, smo: localParams.show_mask_offsets,
        rom: localParams.restore_original_mouth, ns: localParams.num_swap_steps, up: localParams.subsample_upscale,
        r3: localParams.use_3d_recon, sb: localParams.use_source_bank, sm: localParams.swap_model,
        uf: localParams.use_frontalization, fth: localParams.frontalization_threshold,
        jr: localParams.jaw_reshape, jrs: localParams.jaw_reshape_strength,
        ctm: localParams.color_transfer_mode,
        cf_fid: localParams.codeformer_fidelity,
        rl: localParams.refine_landmarks, rsf: localParams.rescue_small_faces, de: localParams.detector_engine,
        dds: localParams.default_det_size,
        fds: localParams.face_detector_size,
        fdt: localParams.face_detector_threshold,
        mask_top: localParams.mask_top,
        mask_bottom: localParams.mask_bottom,
        mask_left: localParams.mask_left,
        mask_right: localParams.mask_right,
        face_mask_blend: localParams.face_mask_blend,
        mouth_mask_blend: localParams.mouth_mask_blend,
        mouth_top_scale: localParams.mouth_top_scale,
        mouth_bottom_scale: localParams.mouth_bottom_scale,
        mouth_left_scale: localParams.mouth_left_scale,
        mouth_right_scale: localParams.mouth_right_scale,
      })}_${sourceFaces.length}_${targetFaces.length}_${selSource}_${selTargetFace}`;
      
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

        const res = await postJSON('/api/preview', {
          index: selTarget, frame: frame, fake_preview: fakePreview,
          enhancer: enh, codeformer_fidelity: num(p.codeformer_fidelity, 0.5),
          detection: p.face_detection_mode,
          face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
          mask_engine: p.mask_engine, clip_text: p.mask_clip_text,
          no_face_action: p.no_face_action, vr_mode: p.vr_mode, autorotate: p.autorotate_faces,
          show_mask_offsets: p.show_mask_offsets, restore_original_mouth: p.restore_original_mouth,
          num_swap_steps: num(p.num_swap_steps, 1), upscale: p.subsample_upscale,
          use_3d_recon: p.use_3d_recon, use_source_bank: p.use_source_bank,
          use_frontalization: p.use_frontalization, frontalization_threshold: num(p.frontalization_threshold, 30),
          jaw_reshape: p.jaw_reshape, jaw_reshape_strength: num(p.jaw_reshape_strength, 0.5),
          swap_model: p.swap_model, default_det_size: p.default_det_size,
          face_detector_size: p.face_detector_size, face_detector_threshold: p.face_detector_threshold,
          face_detector_nms: p.face_detector_nms,
          color_transfer_mode: p.color_transfer_mode, sam2_model_size: p.sam2_model_size,
          refine_landmarks: p.refine_landmarks, rescue_small_faces: p.rescue_small_faces,
          detector_engine: p.detector_engine,
          face_mapping: getFaceMappingArray(),
          mask_top: p.mask_top,
          mask_bottom: p.mask_bottom,
          mask_left: p.mask_left,
          mask_right: p.mask_right,
          face_mask_blend: p.face_mask_blend,
          mouth_mask_blend: p.mouth_mask_blend,
          mouth_top_scale: p.mouth_top_scale,
          mouth_bottom_scale: p.mouth_bottom_scale,
          mouth_left_scale: p.mouth_left_scale,
          mouth_right_scale: p.mouth_right_scale,
        });
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
  }, [comparingEnhancers, selectedGridEnhancers, frame, selTarget, targets.length, sourceFaces.length, targetFaces.length, selSource, selTargetFace, previewKey]);
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
      const cacheKey = `${selTarget}_${frame}_${JSON.stringify({
        fp: fakePreview,
        e: localParams.selected_enhancer, d: localParams.face_detection_mode, fd: localParams.max_face_distance,
        br: localParams.blend_ratio, me: localParams.mask_engine, ct: localParams.mask_clip_text, nfa: localParams.no_face_action,
        vr: localParams.vr_mode, ar: localParams.autorotate_faces, smo: localParams.show_mask_offsets,
        rom: localParams.restore_original_mouth, ns: localParams.num_swap_steps, up: localParams.subsample_upscale,
        r3: localParams.use_3d_recon, sb: localParams.use_source_bank, sm: localParams.swap_model,
        uf: localParams.use_frontalization, fth: localParams.frontalization_threshold,
        jr: localParams.jaw_reshape, jrs: localParams.jaw_reshape_strength,
        ctm: localParams.color_transfer_mode,
        cf_fid: localParams.codeformer_fidelity,
        rl: localParams.refine_landmarks, rsf: localParams.rescue_small_faces, de: localParams.detector_engine,
        dds: localParams.default_det_size,
        fds: localParams.face_detector_size,
        fdt: localParams.face_detector_threshold,
        mask_top: localParams.mask_top,
        mask_bottom: localParams.mask_bottom,
        mask_left: localParams.mask_left,
        mask_right: localParams.mask_right,
        face_mask_blend: localParams.face_mask_blend,
        mouth_mask_blend: localParams.mouth_mask_blend,
        mouth_top_scale: localParams.mouth_top_scale,
        mouth_bottom_scale: localParams.mouth_bottom_scale,
        mouth_left_scale: localParams.mouth_left_scale,
        mouth_right_scale: localParams.mouth_right_scale,
      })}_${sourceFaces.length}_${targetFaces.length}_${selSource}_${selTargetFace}`;

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

        const res = await postJSON('/api/preview', {
          index: selTarget, frame: frame, fake_preview: fakePreview,
          enhancer: p.selected_enhancer, codeformer_fidelity: num(p.codeformer_fidelity, 0.5),
          detection: p.face_detection_mode,
          face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
          mask_engine: me, clip_text: p.mask_clip_text,
          no_face_action: p.no_face_action, vr_mode: p.vr_mode, autorotate: p.autorotate_faces,
          show_mask_offsets: p.show_mask_offsets, restore_original_mouth: p.restore_original_mouth,
          num_swap_steps: num(p.num_swap_steps, 1), upscale: p.subsample_upscale,
          use_3d_recon: p.use_3d_recon, use_source_bank: p.use_source_bank,
          use_frontalization: p.use_frontalization, frontalization_threshold: num(p.frontalization_threshold, 30),
          jaw_reshape: p.jaw_reshape, jaw_reshape_strength: num(p.jaw_reshape_strength, 0.5),
          swap_model: p.swap_model, default_det_size: p.default_det_size,
          face_detector_size: p.face_detector_size, face_detector_threshold: p.face_detector_threshold,
          face_detector_nms: p.face_detector_nms,
          color_transfer_mode: p.color_transfer_mode, sam2_model_size: p.sam2_model_size,
          refine_landmarks: p.refine_landmarks, rescue_small_faces: p.rescue_small_faces,
          detector_engine: p.detector_engine,
          face_mapping: getFaceMappingArray(),
          mask_top: p.mask_top, mask_bottom: p.mask_bottom, mask_left: p.mask_left, mask_right: p.mask_right,
          face_mask_blend: p.face_mask_blend, mouth_mask_blend: p.mouth_mask_blend,
          mouth_top_scale: p.mouth_top_scale, mouth_bottom_scale: p.mouth_bottom_scale,
          mouth_left_scale: p.mouth_left_scale, mouth_right_scale: p.mouth_right_scale,
        });
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
  }, [comparingMasks, selectedGridMasks, frame, selTarget, targets.length, sourceFaces.length, targetFaces.length, selSource, selTargetFace, previewKey]);
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
      const cacheKey = `${selTarget}_${frame}_${JSON.stringify({
        fp: fakePreview,
        e: localParams.selected_enhancer, d: localParams.face_detection_mode, fd: localParams.max_face_distance,
        br: localParams.blend_ratio, me: localParams.mask_engine, ct: localParams.mask_clip_text, nfa: localParams.no_face_action,
        vr: localParams.vr_mode, ar: localParams.autorotate_faces, smo: localParams.show_mask_offsets,
        rom: localParams.restore_original_mouth, ns: localParams.num_swap_steps, up: localParams.subsample_upscale,
        r3: localParams.use_3d_recon, sb: localParams.use_source_bank, sm: localParams.swap_model,
        uf: localParams.use_frontalization, fth: localParams.frontalization_threshold,
        jr: localParams.jaw_reshape, jrs: localParams.jaw_reshape_strength,
        ctm: localParams.color_transfer_mode,
        cf_fid: localParams.codeformer_fidelity,
        rl: localParams.refine_landmarks, rsf: localParams.rescue_small_faces, de: localParams.detector_engine,
        dds: localParams.default_det_size,
        fds: localParams.face_detector_size,
        fdt: localParams.face_detector_threshold,
        mask_top: localParams.mask_top,
        mask_bottom: localParams.mask_bottom,
        mask_left: localParams.mask_left,
        mask_right: localParams.mask_right,
        face_mask_blend: localParams.face_mask_blend,
        mouth_mask_blend: localParams.mouth_mask_blend,
        mouth_top_scale: localParams.mouth_top_scale,
        mouth_bottom_scale: localParams.mouth_bottom_scale,
        mouth_left_scale: localParams.mouth_left_scale,
        mouth_right_scale: localParams.mouth_right_scale,
      })}_${sourceFaces.length}_${targetFaces.length}_${selSource}_${selTargetFace}`;

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

        const res = await postJSON('/api/preview', {
          index: selTarget, frame: frame, fake_preview: fakePreview,
          enhancer: p.selected_enhancer, codeformer_fidelity: num(p.codeformer_fidelity, 0.5),
          detection: p.face_detection_mode,
          face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
          mask_engine: p.mask_engine, clip_text: p.mask_clip_text,
          no_face_action: p.no_face_action, vr_mode: p.vr_mode, autorotate: p.autorotate_faces,
          show_mask_offsets: p.show_mask_offsets, restore_original_mouth: p.restore_original_mouth,
          num_swap_steps: num(p.num_swap_steps, 1), upscale: p.subsample_upscale,
          use_3d_recon: p.use_3d_recon, use_source_bank: p.use_source_bank,
          use_frontalization: p.use_frontalization, frontalization_threshold: num(p.frontalization_threshold, 30),
          jaw_reshape: p.jaw_reshape, jaw_reshape_strength: num(p.jaw_reshape_strength, 0.5),
          swap_model: sm, default_det_size: p.default_det_size,
          face_detector_size: p.face_detector_size, face_detector_threshold: p.face_detector_threshold,
          face_detector_nms: p.face_detector_nms,
          color_transfer_mode: p.color_transfer_mode, sam2_model_size: p.sam2_model_size,
          refine_landmarks: p.refine_landmarks, rescue_small_faces: p.rescue_small_faces,
          detector_engine: p.detector_engine,
          face_mapping: getFaceMappingArray(),
          mask_top: p.mask_top, mask_bottom: p.mask_bottom, mask_left: p.mask_left, mask_right: p.mask_right,
          face_mask_blend: p.face_mask_blend, mouth_mask_blend: p.mouth_mask_blend,
          mouth_top_scale: p.mouth_top_scale, mouth_bottom_scale: p.mouth_bottom_scale,
          mouth_left_scale: p.mouth_left_scale, mouth_right_scale: p.mouth_right_scale,
        });
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
  }, [comparingSwappers, selectedGridSwappers, frame, selTarget, targets.length, sourceFaces.length, targetFaces.length, selSource, selTargetFace, previewKey]);
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
      const baseRes = await postJSON('/api/preview', {
        index: selTarget, frame: frame, fake_preview: true,
        enhancer: p.selected_enhancer, codeformer_fidelity: num(p.codeformer_fidelity, 0.5),
        detection: p.face_detection_mode,
        face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
        mask_engine: p.mask_engine, clip_text: p.mask_clip_text,
        no_face_action: p.no_face_action, vr_mode: p.vr_mode, autorotate: p.autorotate_faces,
        show_mask_offsets: p.show_mask_offsets, restore_original_mouth: p.restore_original_mouth,
        num_swap_steps: num(p.num_swap_steps, 1), upscale: p.subsample_upscale,
        use_3d_recon: p.use_3d_recon, use_source_bank: p.use_source_bank,
        use_frontalization: p.use_frontalization, frontalization_threshold: num(p.frontalization_threshold, 30),
        jaw_reshape: p.jaw_reshape, jaw_reshape_strength: num(p.jaw_reshape_strength, 0.5),
        swap_model: p.swap_model, default_det_size: p.default_det_size,
        face_detector_size: p.face_detector_size, face_detector_threshold: p.face_detector_threshold,
        face_detector_nms: p.face_detector_nms,
        color_transfer_mode: p.color_transfer_mode, sam2_model_size: p.sam2_model_size,
        refine_landmarks: p.refine_landmarks, rescue_small_faces: p.rescue_small_faces,
        detector_engine: p.detector_engine,
        face_mapping: getFaceMappingArray(),
        mask_top: p.mask_top, mask_bottom: p.mask_bottom, mask_left: p.mask_left, mask_right: p.mask_right,
        face_mask_blend: p.face_mask_blend, mouth_mask_blend: p.mouth_mask_blend,
        mouth_top_scale: p.mouth_top_scale, mouth_bottom_scale: p.mouth_bottom_scale,
        mouth_left_scale: p.mouth_left_scale, mouth_right_scale: p.mouth_right_scale,
      });
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
  }, [comparingUpscalers, selectedGridUpscalers, frame, selTarget, targets.length, sourceFaces.length, targetFaces.length, selSource, selTargetFace, previewKey]);
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
    if (targets.length === 0 || progress.processing || isScrubbing || isPlaying) return;
    const t = setTimeout(() => refreshPreview(), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewKey, sourceFaces.length, targetFaces.length, isScrubbing, isPlaying]);

  // ── source / target file handling ──
  const onAddSource = async (files) => {
    if (!files || !files.length) return;
    const before = sourceFaces.length;
    setUploadingSrc(true);
    try {
      const res = await postFiles('/api/source/add', files);
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

  const sourceAction = async (path, body) => {
    try {
      const res = await postJSON(path, body);
      if (res.source_faces) setSourceFaces(res.source_faces);
      if (res.source_faces_info) setSourceFacesInfo(res.source_faces_info);
    } catch (e) {
      // Surface failures (e.g. a 404 when the backend hasn't been restarted to
      // pick up a new endpoint) instead of silently doing nothing.
      notify(`${path.split('/').pop()} failed: ${e.message}. If this is a new feature, restart the app server.`, 'error');
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
  const rawUrl = targets.length > 0 ? `${API}/api/target/preview?index=${selTarget}&frame=${frame}` : '';

  const revealOutput = async () => {
    try { await postJSON('/api/reveal', { path: out?.path }); }
    catch (e) { notify(e.message, 'error'); }
  };

  // Playback timer effect
  useEffect(() => {
    if (isPlaying) {
      const fps = targets[selTarget]?.fps || 25;
      const intervalMs = 1000 / (fps * (playbackRate || 1));
      playIntervalRef.current = setInterval(() => {
        setFrame((f) => {
          const start = targets[selTarget]?.start_frame ?? 1;
          const end = targets[selTarget]?.end_frame ?? maxFrames;
          let next = f + 1;
          if (next > end) {
            if (isLooping) {
              next = start;
            } else {
              setIsPlaying(false);
              return f;
            }
          }
          return next;
        });
      }, intervalMs);
    } else {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, isLooping, playbackRate, selTarget, maxFrames, targets]);

  // Storyboard loading effect
  useEffect(() => {
    if (targets.length === 0 || maxFrames <= 1) {
      setStoryboardThumbs([]);
      return;
    }
    const numThumbs = 8;
    const step = maxFrames > numThumbs ? (maxFrames - 1) / (numThumbs - 1) : 1;
    const urls = [];
    for (let i = 0; i < numThumbs; i++) {
      const f = Math.min(maxFrames, Math.round(1 + i * step));
      urls.push(`${API}/api/target/preview?index=${selTarget}&frame=${f}&width=200`);
    }
    setStoryboardThumbs(urls);
  }, [selTarget, maxFrames, targets.length]);

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
      const pct = x / rect.width;
      const f = Math.max(1, Math.min(Math.round(pct * (maxFrames - 1)) + 1, maxFrames));
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
      let targetFrame = Math.round(pct * (maxFrames - 1)) + 1;
      // Magnetic snap: while dragging the playhead, pull it onto the In/Out
      // points and the clip ends when it lands within ~10px, so lining the
      // playhead up with a marker doesn't require pixel-perfect aim.
      if (dragType === 'playhead' && maxFrames > 1) {
        const snapFrames = Math.max(1, Math.round((10 / rect.width) * (maxFrames - 1)));
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
    const targetFrame = Math.round(pct * (maxFrames - 1)) + 1;

    const sPct = ((targets[selTarget]?.start_frame ?? 1) - 1) / (maxFrames - 1);
    const ePct = ((targets[selTarget]?.end_frame ?? maxFrames) - 1) / (maxFrames - 1);
    const cPct = (frame - 1) / (maxFrames - 1);

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

  // Format a frame index as an m:ss timecode for the timeline ruler/readouts.
  const fmtTC = (f, fps) => {
    const s = Math.max(0, f) / (fps || 25);
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${String(sec).padStart(2, '0')}`;
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
    preview: () => refreshPreview(),
    shortcuts: () => setShowShortcutHUD(true),
  };
  useEffect(() => {
    const h = (e) => { const fn = cmdRef.current[e.detail?.id]; if (fn) fn(); };
    window.addEventListener('roop:command', h);
    return () => window.removeEventListener('roop:command', h);
  }, []);

  const startFrame = targets[selTarget]?.start_frame || 1;
  const endFrame = targets[selTarget]?.end_frame || maxFrames;
  const startPct = maxFrames > 1 ? ((startFrame - 1) / (maxFrames - 1)) * 100 : 0;
  const endPct = maxFrames > 1 ? ((endFrame - 1) / (maxFrames - 1)) * 100 : 100;
  const currentPct = maxFrames > 1 ? ((frame - 1) / (maxFrames - 1)) * 100 : 0;

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
          viewport. */}
      <div className="w-full lg:w-[380px] 3xl:w-[440px] 4xl:w-[520px] shrink-0 pr-0 lg:pr-2 space-y-5 select-none">
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

        {/* COLUMN 2: Media Asset Manager — right rail */}
        <div className="w-full 2xl:w-[360px] 3xl:w-[440px] 4xl:w-[500px] shrink-0 space-y-6 select-none">
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
            onRemove={(i) => sourceAction('/api/source/remove', { index: i })} empty="Upload a face image" info={sourceFacesInfo} />
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
              <div className="relative aspect-video rounded-2xl overflow-hidden bg-gradient-to-br from-white/[0.03] to-black/20 border border-white/10 flex flex-col items-center justify-center select-none px-6 sm:px-10">
                <div className="absolute inset-0 pointer-events-none opacity-60" style={{ background: 'radial-gradient(circle at 50% 38%, var(--accent-glow), transparent 60%)' }} />
                <div className="relative w-full max-w-xl flex flex-col items-center gap-4">
                  <div className="flex flex-col items-center gap-1.5">
                    <div className="flex items-center gap-2">
                      <span className={`h-2.5 w-2.5 rounded-full ${progress.paused ? 'bg-amber-400' : 'bg-[var(--accent)] animate-ping'}`} />
                      <span className={`text-xs font-bold uppercase tracking-[0.18em] ${progress.paused ? 'text-amber-400' : 'text-[var(--accent)]'}`}>
                        {progress.paused ? 'Paused' : 'Processing'}
                      </span>
                    </div>
                    {/* desc mirrors the terminal line: "Processing frame X / Y (Z FPS)" */}
                    <div className="text-sm font-semibold text-white/85 text-center tabular-nums break-words">
                      {progress.desc || 'Swapping faces…'}
                    </div>
                  </div>

                  <div className="w-full space-y-2">
                    <div className="flex items-center justify-between text-[11px] font-mono text-white/45">
                      <span className="text-white/85 font-bold tabular-nums">{Math.round(prog * 100)}%</span>
                      <span className="tabular-nums">{fmtTime(elapsedMs)} elapsed</span>
                    </div>
                    <div className="h-2.5 w-full rounded-full bg-white/[0.06] overflow-hidden">
                      <div
                        className={`h-full rounded-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] transition-[width] duration-500 ease-out ${progress.paused ? '' : 'progress-bar-animated'}`}
                        style={{ width: `${Math.max(2, prog * 100)}%`, boxShadow: '0 0 10px var(--accent-glow)' }}
                      />
                    </div>
                    <div className="flex items-center justify-between text-[11px] font-mono text-white/45">
                      <span className="tabular-nums text-emerald-400/90 font-semibold">{etaMs > 0 ? `${fmtTime(etaMs)} left` : '—'}</span>
                      <span className="tabular-nums">ETA {etaMs > 0 ? fmtTime(etaMs) : '--:--'}</span>
                    </div>
                  </div>

                  {/* Pipeline stage stepper — mirrors the terminal phases. */}
                  {(() => {
                    const d = (progress.desc || '').toLowerCase();
                    const stages = [
                      { key: 'analyze', label: 'Analyze', icon: '🔍' },
                      { key: 'swap', label: 'Swap', icon: '🎭' },
                      ...(p.upscale_after_swap ? [{ key: 'upscale', label: 'Upscale', icon: '🔎' }] : []),
                      { key: 'combine', label: 'Combine', icon: '🎬' },
                    ];
                    let activeKey = 'swap';
                    if (/combin|finaliz|encod|audio|mux/.test(d)) activeKey = 'combine';
                    else if (/upscal/.test(d)) activeKey = 'upscale';
                    else if (/processing frame|swapp/.test(d)) activeKey = 'swap';
                    else if (/analy|track|extract|detect|start/.test(d)) activeKey = 'analyze';
                    let activeIdx = stages.findIndex((s) => s.key === activeKey);
                    if (activeIdx < 0) activeIdx = 1;
                    return (
                      <div className="w-full flex items-center gap-1.5">
                        {stages.map((s, i) => {
                          const state = i < activeIdx ? 'done' : i === activeIdx ? 'active' : 'pending';
                          return (
                            <React.Fragment key={s.key}>
                              <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[10px] font-bold uppercase tracking-wide transition-colors duration-300 ${
                                state === 'done' ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                                : state === 'active' ? 'border-[var(--accent)]/50 bg-[var(--accent)]/15 text-white'
                                : 'border-white/10 bg-white/[0.02] text-white/35'}`}>
                                <span>{state === 'done' ? '✓' : s.icon}</span>
                                <span>{s.label}</span>
                                {state === 'active' && !progress.paused && <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)] animate-ping" />}
                              </div>
                              {i < stages.length - 1 && <div className={`h-px flex-1 min-w-[6px] transition-colors duration-300 ${i < activeIdx ? 'bg-emerald-500/40' : 'bg-white/10'}`} />}
                            </React.Fragment>
                          );
                        })}
                      </div>
                    );
                  })()}

                  {/* Live hardware telemetry — the HUD that used to live in the run-bar. */}
                  {telemetry && (
                    <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1.5 text-[10px] font-mono text-white/40">
                      <span>VRAM <span className="text-blue-300 font-semibold tabular-nums">{telemetry.vram_used} / {telemetry.vram_total} GB</span></span>
                      <span className="text-white/15">·</span>
                      <span>CPU <span className="text-white/70 font-semibold tabular-nums">{telemetry.cpu_percent}%</span></span>
                      <span className="text-white/15">·</span>
                      <span>Threads <span className="text-white/70 font-semibold tabular-nums">{telemetry.threads}</span></span>
                    </div>
                  )}

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
                  beforeSrc={rawUrl}
                  afterSrc={(!isScrubbing && !isPlaying) ? previewSrc : (getCachedPreview(selTarget, frame)?.image || rawUrl)}
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
            
            {/* CINEMATIC TIMELINE SLIDER */}
            {maxFrames > 1 && (
              <div className="space-y-2.5 pt-3 border-t border-[var(--border-color)] select-none">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="h-3 w-[3px] rounded-full bg-[var(--accent)]" />
                    <span className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">Cinematic Timeline</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[11px] font-mono text-[var(--text-muted)]">
                    <span className="px-2 py-0.5 rounded-md border border-[var(--border-color)] bg-[var(--surface-2)]">
                      In <span className="text-[var(--text-main)] font-semibold tabular-nums">{targets[selTarget]?.start_frame ?? 1}</span>
                    </span>
                    <span className="px-2 py-0.5 rounded-md border border-[var(--border-color)] bg-[var(--surface-2)]">
                      Out <span className="text-[var(--text-main)] font-semibold tabular-nums">{targets[selTarget]?.end_frame ?? maxFrames}</span>
                    </span>
                    {(() => {
                      const s = targets[selTarget]?.start_frame ?? 1;
                      const e = targets[selTarget]?.end_frame ?? maxFrames;
                      const len = Math.max(0, e - s + 1);
                      return (
                        <span className="px-2 py-0.5 rounded-md border border-[var(--accent)]/30 bg-[var(--accent)]/10 text-[var(--accent)]" title="Selected range length">
                          {len} f · {fmtTC(len, targets[selTarget]?.fps || 25)}
                        </span>
                      );
                    })()}
                  </div>
                </div>

                {/* Time ruler */}
                <div className="flex items-center justify-between px-0.5 text-[10px] font-mono tabular-nums text-[var(--text-muted)] opacity-70">
                  {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
                    <span key={i}>{fmtTC(Math.round(t * (maxFrames - 1)) + 1, targets[selTarget]?.fps || 25)}</span>
                  ))}
                </div>

                {/* Timeline Track with Storyboard Background */}
                <div className="relative">
                  {hoverFrame !== null && (
                    <div
                      className="absolute bottom-[76px] z-50 flex flex-col items-center gap-1.5 pointer-events-none select-none -translate-x-1/2"
                      style={{ left: `${maxFrames > 1 ? Math.min(92, Math.max(8, ((hoverFrame - 1) / (maxFrames - 1)) * 100)) : 0}%` }}
                    >
                      <div className="flex flex-col items-center gap-1.5 p-1.5 rounded-lg border border-[var(--border-color)] bg-[var(--card-bg)] backdrop-blur-xl shadow-[0_8px_24px_rgba(0,0,0,0.4)]">
                        <img
                          src={`${API}/api/target/preview?index=${selTarget}&frame=${hoverFrame}&width=384`}
                          alt="Hover Preview"
                          className="w-40 h-[90px] object-cover rounded-md border border-[var(--border-color)] bg-black/50"
                          onError={(e) => { e.currentTarget.style.visibility = 'hidden'; }}
                          onLoad={(e) => { e.currentTarget.style.visibility = 'visible'; }}
                        />
                        <span className="text-[10px] font-mono text-[var(--text-muted)] whitespace-nowrap">
                          Frame <span className="text-[var(--text-main)] font-semibold tabular-nums">{hoverFrame}</span> · {fmtTC(hoverFrame, targets[selTarget]?.fps || 25)}
                        </span>
                      </div>
                      {/* Caret pointer */}
                      <div className="w-2 h-2 rotate-45 -mt-[5px] border-b border-r border-[var(--border-color)] bg-[var(--card-bg)]" />
                    </div>
                  )}

                  <div
                    ref={timelineRef}
                    onPointerDown={handleTimelinePointerDown}
                    onPointerMove={handleTimelinePointerMove}
                    onPointerLeave={handleTimelinePointerLeave}
                    className="relative h-16 w-full rounded-lg bg-[var(--input-bg)] border border-[var(--border-color)] overflow-hidden cursor-ew-resize select-none timeline-ticks timeline-glow-track"
                  >
                  {/* Storyboard filmstrip */}
                  {storyboardThumbs.length > 0 && (
                    <div className="absolute inset-0 flex opacity-45 pointer-events-none">
                      {storyboardThumbs.map((url, i) => (
                        <img
                          key={i}
                          src={url}
                          alt="thumb"
                          className="flex-1 h-full object-cover border-r border-black/30 last:border-r-0"
                          loading="lazy"
                          onError={(e) => { e.currentTarget.style.opacity = '0'; }}
                        />
                      ))}
                    </div>
                  )}
                  {/* Legibility scrim over the filmstrip */}
                  <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-black/45 via-transparent to-black/15" />

                  {/* Out-of-range dimming (pro-editor trim convention) */}
                  <div className="absolute top-0 bottom-0 left-0 z-10 bg-black/55 pointer-events-none" style={{ width: `${startPct}%` }} />
                  <div className="absolute top-0 bottom-0 right-0 z-10 bg-black/55 pointer-events-none" style={{ left: `${endPct}%` }} />

                  {/* Active-range baseline */}
                  <div
                    className="absolute bottom-0 h-[2px] z-10 bg-[var(--accent)] opacity-80 pointer-events-none"
                    style={{ left: `${startPct}%`, width: `${endPct - startPct}%` }}
                  />

                  {/* In handle */}
                  <div
                    className="absolute top-0 bottom-0 w-[2px] -translate-x-1/2 z-20 bg-[var(--text-main)]/85"
                    style={{ left: `${startPct}%` }}
                  >
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-7 w-1.5 rounded-full bg-[var(--text-main)] shadow-[0_1px_3px_rgba(0,0,0,0.6)] pointer-events-none" />
                  </div>

                  {/* Out handle */}
                  <div
                    className="absolute top-0 bottom-0 w-[2px] -translate-x-1/2 z-20 bg-[var(--text-main)]/85"
                    style={{ left: `${endPct}%` }}
                  >
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-7 w-1.5 rounded-full bg-[var(--text-main)] shadow-[0_1px_3px_rgba(0,0,0,0.6)] pointer-events-none" />
                  </div>

                  {/* Hover scrub indicator — a faint line showing exactly where a
                      click will drop the playhead, tracking the cursor. */}
                  {hoverFrame !== null && (
                    <div
                      className="absolute top-0 bottom-0 w-px -translate-x-1/2 bg-white/35 z-20 pointer-events-none"
                      style={{ left: `${maxFrames > 1 ? ((hoverFrame - 1) / (maxFrames - 1)) * 100 : 0}%` }}
                    />
                  )}

                  {/* Playhead — white core stays legible over any thumbnail, an
                      accent pin-head reads as the grabbable cursor. Glides while
                      playing/stepping; tracks the pointer exactly while scrubbing. */}
                  <div
                    className={`absolute top-0 bottom-0 z-30 pointer-events-none ${isScrubbing ? '' : 'transition-[left] duration-100 ease-out'}`}
                    style={{ left: `${currentPct}%` }}
                  >
                    {/* vertical line */}
                    <div className="absolute inset-y-0 left-0 -translate-x-1/2 w-[2px] bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.4),0_0_8px_var(--accent-glow)]" />
                    {/* top pin-head knob (accent, brand identity) */}
                    <div className="absolute top-[3px] left-0 -translate-x-1/2 h-3.5 w-3.5 rounded-full bg-[var(--accent)] border-2 border-white shadow-[0_1px_5px_rgba(0,0,0,0.6)]" />
                    {/* bottom cap knob */}
                    <div className="absolute bottom-[3px] left-0 -translate-x-1/2 h-2.5 w-2.5 rounded-full bg-white shadow-[0_1px_4px_rgba(0,0,0,0.6)]" />
                  </div>
                </div>
              </div>

                {/* Timeline controls toolbar — flex-wrap so the control groups
                   reflow to any container width and re-adapt automatically when
                   the window is dragged between monitors of different sizes,
                   instead of overflowing/overlapping off the right edge. */}
                <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-2.5 rounded-lg border border-[var(--border-color)] bg-[var(--card-bg)] p-2.5">
                  {/* Left: timecode / editable frame readout (click to jump) */}
                  <div className="font-mono text-xs text-[var(--text-muted)] flex items-center gap-2.5 w-full sm:w-auto justify-between sm:justify-start">
                    <span className="tabular-nums text-sm font-semibold text-[var(--text-main)]">{fmtTC(frame, targets[selTarget]?.fps || 25)}</span>
                    <span className="opacity-30 hidden sm:inline">·</span>
                    <span className="tabular-nums flex items-center gap-1">
                      Frame
                      <input
                        type="number"
                        min={1}
                        max={maxFrames}
                        value={frameInput ?? frame}
                        onFocus={(e) => { setFrameInput(String(frame)); e.target.select(); }}
                        onChange={(e) => setFrameInput(e.target.value)}
                        onBlur={() => {
                          if (frameInput !== null) {
                            const v = Math.max(1, Math.min(maxFrames, parseInt(frameInput, 10) || frame));
                            setFrame(v);
                          }
                          setFrameInput(null);
                        }}
                        onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') { setFrameInput(null); e.currentTarget.blur(); } }}
                        className="w-14 text-center text-[var(--text-main)] bg-[var(--input-bg)] outline-none rounded border border-[var(--border-color)] py-0.5 focus:border-[var(--accent)] transition-colors tabular-nums"
                        title="Type a frame number and press Enter to jump"
                      />
                      <span className="opacity-40">/ {maxFrames}</span>
                    </span>
                  </div>

                  {/* Center: transport */}
                  <div className="spring-cluster flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-1">
                    <button
                      onClick={() => setFrame(targets[selTarget]?.start_frame || 1)}
                      className="p-1.5 rounded-md text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.05] active:scale-95 transition-colors"
                      title="Jump to In point"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5"/></svg>
                    </button>
                    <button
                      onClick={() => setFrame(f => Math.max(1, f - 1))}
                      className="p-1.5 rounded-md text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.05] active:scale-95 transition-colors"
                      title="Previous Frame (Left Arrow)"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="19 12 5 12"/><polyline points="12 19 5 12 12 5"/></svg>
                    </button>

                    <button
                      onClick={() => setIsPlaying(!isPlaying)}
                      className={`px-3.5 py-1.5 rounded-md font-semibold text-xs inline-flex items-center gap-1.5 active:scale-95 transition-colors ${isPlaying ? 'bg-[var(--surface-2)] text-[var(--text-main)] border border-[var(--border-strong)] hover:bg-white/[0.06]' : 'bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)]'}`}
                      title="Play/Pause (Spacebar)"
                    >
                      {isPlaying ? (
                        <>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                          Pause
                        </>
                      ) : (
                        <>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                          Play
                        </>
                      )}
                    </button>

                    <button
                      onClick={() => setFrame(f => Math.min(maxFrames, f + 1))}
                      className="p-1.5 rounded-md text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.05] active:scale-95 transition-colors"
                      title="Next Frame (Right Arrow)"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 12 19 12"/><polyline points="12 5 19 12 12 19"/></svg>
                    </button>
                    <button
                      onClick={() => setFrame(targets[selTarget]?.end_frame || maxFrames)}
                      className="p-1.5 rounded-md text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.05] active:scale-95 transition-colors"
                      title="Jump to Out point"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/></svg>
                    </button>
                  </div>

                  {/* Set In / Out / Reset */}
                  <div className="spring-cluster flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-1">
                    <button
                      onClick={() => setFrameMarkerVal('start', frame)}
                      className="px-2.5 py-1 rounded-md text-[11px] font-semibold text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.05] transition-colors"
                      title="Set In point to current frame"
                    >
                      Set In
                    </button>
                    <button
                      onClick={() => setFrameMarkerVal('end', frame)}
                      className="px-2.5 py-1 rounded-md text-[11px] font-semibold text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.05] transition-colors"
                      title="Set Out point to current frame"
                    >
                      Set Out
                    </button>
                    <button
                      onClick={async () => {
                        await setFrameMarkerVal('start', 1);
                        await setFrameMarkerVal('end', maxFrames);
                      }}
                      className="px-2.5 py-1 rounded-md text-[11px] font-semibold text-[var(--text-muted)]/70 hover:text-[var(--text-main)] hover:bg-white/[0.05] transition-colors"
                      title="Reset range to full video"
                    >
                      Reset
                    </button>
                  </div>

                  {/* Right: numeric range inputs & loop */}
                  <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                    <div className="flex items-center gap-1.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] px-2 py-1">
                      <span className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">Range</span>
                      <input
                        type="number"
                        value={targets[selTarget]?.start_frame || 1}
                        onChange={(e) => setFrameMarkerVal('start', parseInt(e.target.value, 10))}
                        className="w-11 text-center text-xs font-mono font-semibold text-[var(--text-main)] bg-[var(--input-bg)] outline-none rounded border border-[var(--border-color)] py-0.5 focus:border-[var(--accent)] transition-colors"
                        title="In frame"
                      />
                      <span className="text-[var(--text-muted)] text-xs">–</span>
                      <input
                        type="number"
                        value={targets[selTarget]?.end_frame || maxFrames}
                        onChange={(e) => setFrameMarkerVal('end', parseInt(e.target.value, 10))}
                        className="w-11 text-center text-xs font-mono font-semibold text-[var(--text-main)] bg-[var(--input-bg)] outline-none rounded border border-[var(--border-color)] py-0.5 focus:border-[var(--accent)] transition-colors"
                        title="Out frame"
                      />
                    </div>

                    {/* Playback speed */}
                    <div className="spring-cluster flex items-center rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5" title="Playback speed">
                      {[0.5, 1, 2, 4].map((r) => (
                        <button
                          key={r}
                          onClick={() => setPlaybackRate(r)}
                          className={`px-1.5 py-1 rounded-md text-[10px] font-bold tabular-nums transition-colors ${playbackRate === r ? 'bg-[var(--accent)] text-white' : 'text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.05]'}`}
                        >
                          {r}×
                        </button>
                      ))}
                    </div>

                    <button
                      onClick={() => setIsLooping(!isLooping)}
                      className={`p-1.5 rounded-md border transition-colors active:scale-95 ${isLooping ? 'bg-[var(--accent)]/12 text-[var(--accent)] border-[var(--accent)]/30' : 'text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.05] border-[var(--border-color)]'}`}
                      title="Toggle loop playback"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div className={`flex items-center flex-wrap gap-3 ${maxFrames > 1 ? 'pt-3 border-t border-white/5' : ''}`}>
              <Button size="sm" variant="secondary" onClick={() => refreshPreview()}>🔄 Refresh</Button>
              <Button size="sm" variant="primary" onClick={useFaceFromFrame}>Use face from frame</Button>
              {previewSrc && !comparingEnhancers && !comparingMasks && !comparingSwappers && !comparingUpscalers && (
                <Button size="sm" variant="secondary" disabled={upscaling} onClick={upscaleThisFrame}
                  title="AI-upscale just this frame to preview final quality">
                  {upscaling ? '🔎 Upscaling…' : `🔎 Upscale this frame (${AI_UPSCALE_MODELS.find(m => m.value === (p.upscale_model_after || 'esrganx2'))?.label || 'AI'})`}
                </Button>
              )}
            </div>

            <div className="flex items-center flex-wrap gap-3">
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

          <Section title="Output settings & renders">
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
