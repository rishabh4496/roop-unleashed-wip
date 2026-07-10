import React, { useEffect, useRef, useState } from 'react';
import { getJSON, postJSON, postFiles, API } from '../api';
import { Section, Select, Slider, Toggle, TextInput, Button, FaceGallery, Card } from './ui';
import PersonGroups from './PersonGroups';

const num = (v, d) => (v === undefined || v === null || v === '' ? d : Number(v));

const fmtTime = (ms) => {
  const s = Math.max(0, Math.round(ms / 1000));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${String(s % 60).padStart(2, '0')}s` : `${s}s`;
};

export default function FaceSwap({ meta, settings, setSettings, notify, registerFileListener }) {
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
  const [fakePreview, setFakePreview] = useState(true);
  const [uploadingSrc, setUploadingSrc] = useState(false);
  const [uploadingTgt, setUploadingTgt] = useState(false);
  const [progress, setProgress] = useState({ processing: false, progress: 0, desc: '', output: null });
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

  // Telemetry HUD State
  const [telemetry, setTelemetry] = useState(null);

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

  // Profile Management
  const [profiles, setProfiles] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('roop_profiles') || '[]');
    } catch {
      return [];
    }
  });
  const [newProfileName, setNewProfileName] = useState('');

  // Keyboard Shortcuts Modal visibility
  const [showShortcutHUD, setShowShortcutHUD] = useState(false);

  // Drag and drop overlay state removed (managed globally by App.jsx)

  // Batch Queue Manager
  const [queue, setQueue] = useState([]);
  const [currentQueueIndex, setCurrentQueueIndex] = useState(null);
  const [isQueueRunning, setIsQueueRunning] = useState(false);

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


  // Telemetry Polling Effect
  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const data = await getJSON('/api/system/telemetry');
        setTelemetry(data);
      } catch (err) {
        // quiet fail
      }
    };
    fetchTelemetry();
    const id = setInterval(fetchTelemetry, 3000);
    return () => clearInterval(id);
  }, []);

  // Save active parameters to a new profile
  // Persist presets both to localStorage (instant/offline) and the backend
  // (survives cache clears, shareable via profiles.json on disk).
  const persistProfiles = (updated) => {
    setProfiles(updated);
    localStorage.setItem('roop_profiles', JSON.stringify(updated));
    postJSON('/api/profiles', { profiles: updated }).catch(() => { /* offline-tolerant */ });
  };

  // On mount, prefer server-side presets if any exist (merge, server wins).
  useEffect(() => {
    getJSON('/api/profiles').then((res) => {
      const server = Array.isArray(res.profiles) ? res.profiles : [];
      if (server.length === 0) return;
      setProfiles((local) => {
        const map = new Map((local || []).map((pr) => [pr.name, pr]));
        server.forEach((pr) => { if (pr && pr.name) map.set(pr.name, pr); });
        const merged = Array.from(map.values());
        localStorage.setItem('roop_profiles', JSON.stringify(merged));
        return merged;
      });
    }).catch(() => { /* backend not ready — localStorage still works */ });
  }, []);

  const saveProfile = () => {
    if (!newProfileName.trim()) {
      notify('Enter a profile name first', 'error');
      return;
    }
    const profile = {
      name: newProfileName,
      settings: { ...p }
    };
    const updated = [...profiles.filter(pr => pr.name !== newProfileName), profile];
    persistProfiles(updated);
    setNewProfileName('');
    notify(`Saved profile: ${profile.name}`);
  };

  // Load a profile and apply it
  const loadProfile = (name) => {
    const profile = profiles.find(pr => pr.name === name);
    if (!profile) return;
    setSettings((s) => ({ ...s, ...profile.settings }));
    notify(`Loaded profile: ${name}`);
  };

  // Delete a profile
  const deleteProfile = (name) => {
    const updated = profiles.filter(pr => pr.name !== name);
    persistProfiles(updated);
    notify(`Deleted profile: ${name}`);
  };

  // Export profiles
  const exportProfiles = () => {
    try {
      const dataStr = localStorage.getItem('roop_profiles') || '[]';
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'roop_unleashed_presets.json';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      notify('Exported presets successfully!');
    } catch (e) {
      notify('Failed to export presets: ' + e.message, 'error');
    }
  };

  // Import profiles
  const importProfiles = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const imported = JSON.parse(event.target.result);
        if (!Array.isArray(imported)) {
          throw new Error('Presets file must be a JSON array of profiles.');
        }
        const existing = JSON.parse(localStorage.getItem('roop_profiles') || '[]');
        const existingMap = new Map(existing.map(p => [p.name, p]));
        imported.forEach(p => {
          if (p.name && p.settings) {
            existingMap.set(p.name, p);
          }
        });
        const merged = Array.from(existingMap.values());
        persistProfiles(merged);
        notify('Imported presets successfully!');
      } catch (err) {
        notify('Failed to import presets: ' + err.message, 'error');
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
    setCurrentQueueIndex(null);
  };

  const startQueue = async () => {
    if (queue.length === 0) {
      notify('Queue is empty', 'error');
      return;
    }
    setIsQueueRunning(true);
    setCurrentQueueIndex(0);
  };

  // Queue Runner State Machine
  useEffect(() => {
    if (!isQueueRunning || currentQueueIndex === null) return;
    
    if (currentQueueIndex >= queue.length) {
      setIsQueueRunning(false);
      setCurrentQueueIndex(null);
      notify('All batch queue jobs finished!', 'success');
      return;
    }

    const job = queue[currentQueueIndex];
    if (job.status !== 'Pending') {
      setCurrentQueueIndex((idx) => idx + 1);
      return;
    }

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

        startTimeRef.current = Date.now();
        setProgress({ processing: true, paused: false, progress: 0, desc: 'Starting queue job…', output: null });
        startPolling();
      } catch (e) {
        notify(`Job "${job.targetName}" failed to start: ${e.message}`, 'error');
        setQueue((prev) => prev.map(j => j.id === job.id ? { ...j, status: 'Failed' } : j));
        setCurrentQueueIndex((idx) => idx + 1);
      }
    };
    executeJob();
  }, [isQueueRunning, currentQueueIndex]);

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

  // Custom Timeline and Playback States
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(true);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [dragType, setDragType] = useState('playhead'); // 'playhead', 'start', 'end'
  const [storyboardThumbs, setStoryboardThumbs] = useState([]);
  const [hoverFrame, setHoverFrame] = useState(null);
  const timelineRef = useRef(null);
  const playIntervalRef = useRef(null);
  const [isGeneratingPreviewClip, setIsGeneratingPreviewClip] = useState(false);
  const [origStartEnd, setOrigStartEnd] = useState(null);

  const pollRef = useRef(null);
  const startTimeRef = useRef(null);
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
    ctm: p.color_transfer_mode, s2: p.sam2_model_size,
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
        if (!startTimeRef.current) startTimeRef.current = Date.now();
        startPolling();
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
        enhancer: p.selected_enhancer, detection: p.face_detection_mode,
        face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
        mask_engine: p.mask_engine, clip_text: p.mask_clip_text,
        no_face_action: p.no_face_action, vr_mode: p.vr_mode, autorotate: p.autorotate_faces,
        show_mask_offsets: p.show_mask_offsets, restore_original_mouth: p.restore_original_mouth,
        num_swap_steps: num(p.num_swap_steps, 1), upscale: p.subsample_upscale,
        use_3d_recon: p.use_3d_recon, use_source_bank: p.use_source_bank,
        use_frontalization: p.use_frontalization, frontalization_threshold: num(p.frontalization_threshold, 30),
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
      setPreviewSrc(res.image || '');
      if (res.image) {
        setCachedPreview(idx, fr, { faces: res.faces || [], image: res.image });
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
        ctm: localParams.color_transfer_mode,
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
          enhancer: enh, detection: p.face_detection_mode,
          face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
          mask_engine: p.mask_engine, clip_text: p.mask_clip_text,
          no_face_action: p.no_face_action, vr_mode: p.vr_mode, autorotate: p.autorotate_faces,
          show_mask_offsets: p.show_mask_offsets, restore_original_mouth: p.restore_original_mouth,
          num_swap_steps: num(p.num_swap_steps, 1), upscale: p.subsample_upscale,
          use_3d_recon: p.use_3d_recon, use_source_bank: p.use_source_bank,
          use_frontalization: p.use_frontalization, frontalization_threshold: num(p.frontalization_threshold, 30),
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
      } catch (err) {
        if (activeIntervalsRef.current[enh]) {
          clearInterval(activeIntervalsRef.current[enh]);
          delete activeIntervalsRef.current[enh];
        }
        setLiveRenderingTimers((prev) => ({ ...prev, [enh]: null }));
        // Fail silently
      }
    }
  };

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
    const res = await postJSON(path, body);
    if (res.source_faces) setSourceFaces(res.source_faces);
    if (res.source_faces_info) setSourceFacesInfo(res.source_faces_info);
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
      startTimeRef.current = Date.now();
      // Optimistically flag processing so the old "Latest output" clears
      // immediately, before the first poll tick (~1s) confirms it.
      setProgress((pr) => ({ ...pr, processing: true, paused: false, progress: 0, desc: 'Starting…' }));
      notify('Processing started');
      startPolling();
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

  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const pr = await getJSON('/api/progress');
        setProgress(pr);
        if (!pr.processing) { clearInterval(pollRef.current); pollRef.current = null; }
      } catch { /* ignore */ }
    }, 1000);
  };
  useEffect(() => () => pollRef.current && clearInterval(pollRef.current), []);

  // Hide the previous "Latest output" while a job is running so a new upload +
  // run never shows a stale result. The poll keeps reporting the old _last_output
  // until the new job finishes, so we gate on `processing` rather than clearing
  // progress.output (which the next poll tick would just restore).
  const out = progress.processing ? null : progress.output;
  const outUrl = out?.path ? `${API}/api/file?path=${encodeURIComponent(out.path)}&t=${progress.progress}` : '';
  const prog = progress.progress || 0;
  const elapsedMs = progress.processing && startTimeRef.current ? Date.now() - startTimeRef.current : 0;
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
      const intervalMs = 1000 / fps;
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
  }, [isPlaying, isLooping, selTarget, maxFrames, targets]);

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
    const rect = timelineRef.current.getBoundingClientRect();
    const clientX = e.clientX ?? e.touches?.[0]?.clientX;
    if (clientX === undefined) return;
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const pct = x / rect.width;
    const f = Math.max(1, Math.min(Math.round(pct * (maxFrames - 1)) + 1, maxFrames));
    setHoverFrame(f);
  };

  const handleTimelinePointerLeave = () => {
    setHoverFrame(null);
  };

  useEffect(() => {
    if (!isScrubbing) return;

    const handlePointerMove = (e) => {
      if (!timelineRef.current) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const clientX = e.clientX ?? e.touches?.[0]?.clientX;
      if (clientX === undefined) return;
      const x = clientX - rect.left;
      const pct = Math.max(0, Math.min(x / rect.width, 1));
      const targetFrame = Math.round(pct * (maxFrames - 1)) + 1;
      updateTimelinePos(targetFrame, dragType);
    };

    const handlePointerUp = () => {
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
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [isScrubbing, dragType, selTarget, maxFrames, targets]);

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
      });

      startTimeRef.current = Date.now();
      notify('Generating 5-second preview clip...');
      startPolling();
    } catch (e) {
      notify(e.message, 'error');
      setIsGeneratingPreviewClip(false);
      setOrigStartEnd(null);
    }
  };

  // Restoration effect when swapping finishes
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

  // Keyboard Escape, Shortcuts HUD, & Global Productivity Hotkeys
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
            if (nextVal) setComparingEnhancers(false);
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
          if (nextVal) setComparingEnhancers(false);
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

  const startFrame = targets[selTarget]?.start_frame || 1;
  const endFrame = targets[selTarget]?.end_frame || maxFrames;
  const startPct = maxFrames > 1 ? ((startFrame - 1) / (maxFrames - 1)) * 100 : 0;
  const endPct = maxFrames > 1 ? ((endFrame - 1) / (maxFrames - 1)) * 100 : 100;
  const currentPct = maxFrames > 1 ? ((frame - 1) / (maxFrames - 1)) * 100 : 0;

  return (
    <div className="flex flex-col lg:flex-row gap-6 items-start w-full">
      {/* COLUMN 1: Settings & Controls — sticky sidebar on large viewports so it
          follows the scroll (and never leaves the lower-left area empty) while
          scrolling a taller workspace. Scrolls internally when taller than the
          viewport. */}
      <div className="w-full lg:w-[400px] 3xl:w-[820px] 4xl:w-[900px] shrink-0 lg:sticky lg:top-20 lg:self-start lg:max-h-[calc(100vh-100px)] lg:overflow-y-auto pr-0 lg:pr-2 select-none">
        <Section title="Presets">
          <div className="flex flex-wrap gap-2">
            {Object.keys(PRESETS).map((name) => (
              <Button key={name} size="sm"
                variant={activePreset === name ? 'primary' : 'secondary'}
                onClick={() => applyPreset(name)}>
                {name === 'Fast' ? '⚡ Fast' : name === 'Balanced' ? '⚖️ Balanced' : '💎 Quality'}
              </Button>
            ))}
          </div>
          <div className="text-xs text-[var(--text-muted)] mt-2">
            Sets detection resolution, upscale, enhancer & swap steps. Other settings unchanged.
          </div>
        </Section>

        <div className="grid grid-cols-1 3xl:grid-cols-2 gap-6">
          <Section title="Swap settings">
          <Select label="Swap model" info="inswapper 128 · reswapper/hyperswap(a/b/c)/ghost(1-3)/simswap/hififace 256 · simswap_512 (each downloads on first use; ghost/simswap/hififace use their own alignment + identity converter)" value={p.swap_model} onChange={(v) => set('swap_model', v)} options={meta.swap_models} />
          <Select label="Face selection" value={p.face_detection_mode} onChange={(v) => set('face_detection_mode', v)} options={meta.face_detection_modes} />
          <Toggle label="🔒 Lock face identities (video)" info="For 'Selected face' mode on video: tracks each person across the clip and keeps them on one source, so identities don't flip frame-to-frame when faces cross or turn. Adds a short tracking pre-pass; the swap stays multi-threaded." checked={!!p.track_identities} onChange={(v) => set('track_identities', v)} />
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
          <Select label="Color/lighting match" info="Matches the swapped face's skin tone & lighting to the original scene. RCT = per-channel (fast, default). LCT = corrects hue casts. MKL = fullest match. None = off." value={p.color_transfer_mode || 'rct'} onChange={(v) => set('color_transfer_mode', v)} options={meta.color_transfer_modes || ['none', 'rct', 'lct', 'mkl']} />
          <Slider label="Original/Enhanced blend" min={0} max={1} step={0.01} value={num(p.blend_ratio, 0.8)} onChange={(v) => set('blend_ratio', v)} />
        </Section>

        <Section title="Masking parameters">
          <Select label="Masking engine" value={p.mask_engine} onChange={(v) => set('mask_engine', v)} options={meta.mask_engines} />
          {p.mask_engine === 'Clip2Seg' && (
            <TextInput label="Objects to mask & restore" value={p.mask_clip_text} onChange={(v) => set('mask_clip_text', v)} placeholder="cup,hands,hair" />
          )}
          {p.mask_engine === 'Segment Anything 2 (tracked)' && (
            <Select label="SAM2 checkpoint (speed ↔ quality)" value={p.sam2_model_size || 'tiny'} onChange={(v) => set('sam2_model_size', v)} options={meta.sam2_model_sizes || ['tiny', 'small', 'base_plus', 'large']} />
          )}
          <Toggle label="Show mask overlay in preview" checked={!!p.show_mask_offsets} onChange={(v) => set('show_mask_offsets', v)} />
          <Toggle label="Restore original mouth area" checked={!!p.restore_original_mouth} onChange={(v) => set('restore_original_mouth', v)} />
          <Slider label="Offset face top" min={0} max={2} step={0.01} value={num(p.mask_top, 0)} onChange={(v) => set('mask_top', v)} />
          <Slider label="Offset face bottom" min={0} max={2} step={0.01} value={num(p.mask_bottom, 0)} onChange={(v) => set('mask_bottom', v)} />
          <Slider label="Offset face left" min={0} max={2} step={0.01} value={num(p.mask_left, 0)} onChange={(v) => set('mask_left', v)} />
          <Slider label="Offset face right" min={0} max={2} step={0.01} value={num(p.mask_right, 0)} onChange={(v) => set('mask_right', v)} />
          <Slider label="Face mask edge blend" min={0} max={200} step={1} value={num(p.face_mask_blend, 20)} onChange={(v) => set('face_mask_blend', v)} />
        </Section>

        <Section title="Mouth & Angle math">
          <Slider label="Mouth mask top" min={0} max={2} step={0.01} value={num(p.mouth_top_scale, 1)} onChange={(v) => set('mouth_top_scale', v)} />
          <Slider label="Mouth mask bottom" min={0} max={2} step={0.01} value={num(p.mouth_bottom_scale, 1)} onChange={(v) => set('mouth_bottom_scale', v)} />
          <Slider label="Mouth mask left" min={0} max={2} step={0.01} value={num(p.mouth_left_scale, 1)} onChange={(v) => set('mouth_left_scale', v)} />
          <Slider label="Mouth mask right" min={0} max={2} step={0.01} value={num(p.mouth_right_scale, 1)} onChange={(v) => set('mouth_right_scale', v)} />
          <Slider label="Mouth mask edge blend" min={0} max={200} step={1} value={num(p.mouth_mask_blend, 10)} onChange={(v) => set('mouth_mask_blend', v)} />
          <Toggle label="🧊 3D source pose matching" info="experimental — improves angled swaps" checked={!!p.use_3d_recon} onChange={(v) => set('use_3d_recon', v)} />
          <Toggle label="🎯 Multi-angle source bank" info="auto-pick best source per frame" checked={!!p.use_source_bank} onChange={(v) => set('use_source_bank', v)} />
          <Toggle label="↔️ Frontalize angled faces" info="Un-rotates steep profile/side (lateral) faces before swapping so they don't come out distorted/'alien', then restores the original angle." checked={!!p.use_frontalization} onChange={(v) => set('use_frontalization', v)} />
          {p.use_frontalization && (
            <Slider label="Frontalize above angle (°)" info="Frontalization kicks in when the face yaw/pitch exceeds this. Lower = frontalize more; higher = only the steepest." min={10} max={60} step={5} value={num(p.frontalization_threshold, 30)} onChange={(v) => set('frontalization_threshold', v)} />
          )}
        </Section>

        <Section title="Video parameters">
          <Select label="Video method" value={p.video_swapping_method} onChange={(v) => set('video_swapping_method', v)} options={meta.video_methods} />
          <Select label="On no face detected" value={p.no_face_action} onChange={(v) => set('no_face_action', v)} options={meta.no_face_actions} />
          <Toggle label="🛡️ Temporal detection (anti-flicker)" info="Video (In-Memory method): one tracked detection pre-pass over the clip. Short detection misses (≤10 frames) are gap-filled by interpolating the face's position, so the swap can't blink out; with 'Stabilize face' also on, keypoints AND mask/mouth landmarks are smoothed per person. The swap pass then skips per-frame detection and stays multi-threaded. Includes identity locking when that toggle is on." checked={!!p.temporal_detection} onChange={(v) => set('temporal_detection', v)} />
          <Toggle label="VR mode" checked={!!p.vr_mode} onChange={(v) => set('vr_mode', v)} />
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
          <Toggle label="✨ Reduce enhancer flicker" info="Temporally blends the enhanced face. Runs multi-threaded (work-stealing) when the launcher's ROOP_STAB_PARALLEL is on (the Pinokio default) — otherwise it forces single-thread. Either way it costs some extra compute (blending + per-block warm-up), so it's somewhat slower, not free." checked={!!p.stabilize_enhancer} onChange={(v) => set('stabilize_enhancer', v)} />
          {p.stabilize_enhancer && (
            <Slider label="Flicker reduction strength" info="higher = smoother" min={0} max={1} step={0.05} value={num(p.stabilize_enhancer_strength, 0.5)} onChange={(v) => set('stabilize_enhancer_strength', v)} />
          )}
        </Section>

        <div className="3xl:col-span-2">
          <Section title="System options">
            <Toggle label="Auto rotate horizontal faces" checked={!!p.autorotate_faces} onChange={(v) => set('autorotate_faces', v)} />
            <Toggle label="Skip audio" checked={!!p.skip_audio} onChange={(v) => set('skip_audio', v)} />
            <Toggle label="Keep frames (when extracting)" checked={!!p.keep_frames} onChange={(v) => set('keep_frames', v)} />
            <Toggle label="Wait before creating video" checked={!!p.wait_after_extraction} onChange={(v) => set('wait_after_extraction', v)} />
          </Section>
        </div>

        <div className="3xl:col-span-2">
          <Section title="Saved Profiles">
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
          </Section>
        </div>

        <div className="3xl:col-span-2">
          <Section title="Live Telemetry & Diagnostics">
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
              <div className="text-xs text-white/30 italic">Connecting to hardware diagnostics…</div>
            )}
            <div className="mt-3 flex justify-between items-center">
              <Button size="sm" variant="secondary" onClick={() => setShowShortcutHUD(true)}>⌨️ Keyboard Shortcuts Info</Button>
            </div>
          </Section>
        </div>
      </div>
    </div>

      {/* COLUMN 2 & 3 WRAPPER: Media asset managers on left and large active workspace on right */}
      <div className="flex-1 w-full space-y-6 flex flex-col 2xl:flex-row gap-6">
        
        {/* COLUMN 2: Media Asset Manager */}
        <div className="w-full 2xl:w-[360px] 3xl:w-[380px] shrink-0 space-y-6 select-none">
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
                      className={`group flex items-center gap-3 px-3.5 py-3 rounded-xl text-sm border transition-all duration-300 cursor-pointer ${selTarget === i ? 'bg-[var(--accent)]/10 border-[var(--accent)]/40 shadow-[0_0_15px_var(--accent-glow)] scale-[0.99]' : 'bg-black/30 border-white/5 hover:border-white/15 hover:bg-white/[0.01]'}`}
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
                            <span className={`text-[8px] uppercase tracking-widest px-1.5 py-0.5 rounded border font-black ${badgeColor}`}>
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
          <FaceGallery title="Input faces" faces={sourceFaces} selected={selSource} onSelect={selectSource} draggable={true}
            onRemove={(i) => sourceAction('/api/source/remove', { index: i })} empty="Upload a face image" info={sourceFacesInfo} />
          {sourceFaces.length > 0 && (
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/move', { index: selSource, direction: 'left' })}>⬅ Move</Button>
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/move', { index: selSource, direction: 'right' })}>Move ➡</Button>
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/remove', { index: selSource })}>❌ Remove</Button>
                <Button size="sm" variant="stop" onClick={() => sourceAction('/api/source/clear', {})}>Clear all</Button>
              </div>
              
              {sourceFacesInfo[selSource] && (
                <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 text-xs select-none">
                  <div className="flex items-center justify-between">
                    <span className="font-extrabold text-[10px] uppercase tracking-widest text-white/50">📁 Selected Source Details</span>
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
                <div className="relative overflow-hidden rounded-2xl glass-panel px-6 py-5 flex flex-col md:flex-row items-center justify-between gap-6 shadow-2xl border border-white/5 w-full">
                  {/* Left: Circular Progress Ring & Info */}
                  <div className="flex items-center gap-5">
                    <div className={`relative flex items-center justify-center h-20 w-20 select-none shrink-0 rounded-full transition-shadow duration-1000 ${!progress.paused ? 'shadow-[0_0_18px_var(--accent-glow)]' : ''}`}>
                      <svg className="transform -rotate-90 w-[72px] h-[72px]" viewBox="0 0 48 48">
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
                      <span className="absolute text-base font-extrabold text-white tabular-nums">
                        {Math.round(prog * 100)}%
                      </span>
                    </div>

                    <div className="space-y-1">
                      <div className="flex items-center gap-1.5">
                        <span className={`h-2 w-2 rounded-full ${progress.paused ? 'bg-amber-400' : 'bg-[var(--accent)] animate-ping'}`} />
                        <span className={`text-xs font-black uppercase tracking-widest ${progress.paused ? 'text-amber-400' : 'text-[var(--accent)]'}`}>
                          {progress.paused ? 'Paused' : 'Processing'}
                        </span>
                      </div>
                      <div className="text-sm font-extrabold text-white truncate max-w-[360px]">
                        {progress.desc || 'Swapping faces…'}
                      </div>
                      {progress.error && <div className="text-xs text-red-400 font-semibold">{progress.error}</div>}
                    </div>
                  </div>

                  {/* Center: Live Telemetry HUD */}
                  <div className="flex flex-wrap items-center gap-4 text-xs font-mono bg-black/20 px-5 py-2.5 rounded-xl border border-white/5">
                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-white/40 font-bold">Elapsed</span>
                      <span className="text-white font-bold tabular-nums">{fmtTime(elapsedMs)}</span>
                    </div>
                    <div className="h-6 w-px bg-white/10" />
                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-white/40 font-bold">ETA</span>
                      <span className="text-emerald-400 font-bold tabular-nums">
                        {etaMs > 0 ? fmtTime(etaMs) : '--:--'}
                      </span>
                    </div>
                    {telemetry && (
                      <>
                        <div className="h-6 w-px bg-white/10" />
                        <div className="flex flex-col relative group/telemetry cursor-help">
                          <span className="text-[10px] uppercase tracking-wider text-white/40 font-bold">VRAM</span>
                          <span className="text-blue-300 font-bold tabular-nums">
                            {telemetry.vram_used} / {telemetry.vram_total} GB
                          </span>
                          
                          {/* Hardware diagnostics tooltip hover card */}
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 hidden group-hover/telemetry:block z-50 w-64 p-4 rounded-xl bg-black/95 backdrop-blur-lg border border-white/10 shadow-2xl text-left font-sans">
                            <h4 className="text-[9px] font-bold text-[var(--accent)] uppercase tracking-widest mb-2 border-b border-white/10 pb-1">System Telemetry</h4>
                            <div className="space-y-1.5 text-xs text-white/80">
                              <div className="flex justify-between gap-2"><span className="text-white/40">GPU:</span><span className="font-semibold text-white truncate max-w-[130px]" title={telemetry.gpu}>{telemetry.gpu}</span></div>
                              <div className="flex justify-between"><span className="text-white/40">RAM Used:</span><span className="font-semibold text-white">{telemetry.ram_used} / {telemetry.ram_total} GB</span></div>
                              <div className="flex justify-between"><span className="text-white/40">CPU Load:</span><span className="font-semibold text-white">{telemetry.cpu_percent}%</span></div>
                              <div className="flex justify-between"><span className="text-white/40">Active Threads:</span><span className="font-semibold text-white">{telemetry.threads}</span></div>
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Right: big icon action buttons */}
                  <div className="flex items-center gap-3 shrink-0">
                    {progress.paused ? (
                      <button type="button" onClick={resume} title="Resume (Space)"
                        className="group flex flex-col items-center gap-1.5 focus:outline-none">
                        <span className="h-14 w-14 rounded-2xl flex items-center justify-center bg-emerald-500/15 border border-emerald-500/40 text-emerald-400 shadow-[0_0_18px_rgba(16,185,129,0.25)] transition-all duration-200 group-hover:bg-emerald-500/25 group-hover:scale-105 group-active:scale-95">
                          <svg viewBox="0 0 24 24" className="w-7 h-7" fill="currentColor"><path d="M8 5.14v13.72a1 1 0 0 0 1.53.85l10.9-6.86a1 1 0 0 0 0-1.7L9.53 4.29A1 1 0 0 0 8 5.14z" /></svg>
                        </span>
                        <span className="text-[10px] font-black uppercase tracking-widest text-white/45 group-hover:text-emerald-400 transition-colors">Resume</span>
                      </button>
                    ) : (
                      <button type="button" onClick={pause} title="Pause (Space)"
                        className="group flex flex-col items-center gap-1.5 focus:outline-none">
                        <span className="h-14 w-14 rounded-2xl flex items-center justify-center bg-amber-500/15 border border-amber-500/40 text-amber-400 shadow-[0_0_18px_rgba(245,158,11,0.2)] transition-all duration-200 group-hover:bg-amber-500/25 group-hover:scale-105 group-active:scale-95">
                          <svg viewBox="0 0 24 24" className="w-7 h-7" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1.5" /><rect x="14" y="5" width="4" height="14" rx="1.5" /></svg>
                        </span>
                        <span className="text-[10px] font-black uppercase tracking-widest text-white/45 group-hover:text-amber-400 transition-colors">Pause</span>
                      </button>
                    )}
                    <button type="button" onClick={stop} title="Stop"
                      className="group flex flex-col items-center gap-1.5 focus:outline-none">
                      <span className="h-14 w-14 rounded-2xl flex items-center justify-center bg-red-500/15 border border-red-500/40 text-red-400 shadow-[0_0_18px_rgba(239,68,68,0.2)] transition-all duration-200 group-hover:bg-red-500/25 group-hover:scale-105 group-active:scale-95">
                        <svg viewBox="0 0 24 24" className="w-7 h-7" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2.5" /></svg>
                      </span>
                      <span className="text-[10px] font-black uppercase tracking-widest text-white/45 group-hover:text-red-400 transition-colors">Stop</span>
                    </button>
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
              <div className="rounded-2xl glass-panel p-6 flex flex-col md:flex-row items-center justify-between gap-6 shadow-2xl border border-white/5 w-full">
                <div className="flex items-center gap-3 w-full md:w-auto">
                  <Button variant="primary" size="lg" onClick={start} disabled={targets.length === 0 || sourceFaces.length === 0} className="w-full md:w-auto justify-center">▶ Start Swapping</Button>
                  {maxFrames > 1 && (
                    <Button variant="secondary" size="lg" onClick={renderPreviewClip} disabled={targets.length === 0 || sourceFaces.length === 0 || isGeneratingPreviewClip} className="!text-orange-400 border border-orange-500/20 hover:bg-orange-500/10 w-full md:w-auto justify-center">
                      ⚡ Render 5s Preview
                    </Button>
                  )}
                </div>
                <div className="flex items-center gap-2.5 text-sm font-semibold text-[var(--text-muted)] max-w-xs truncate text-right">
                  <span className={`h-2.5 w-2.5 rounded-full ${targets.length > 0 && sourceFaces.length > 0 ? 'bg-emerald-500 animate-pulse shadow-[0_0_8px_#10b981]' : 'bg-red-500/50'}`} />
                  {targets.length === 0 ? 'No target media selected' : sourceFaces.length === 0 ? 'No source faces loaded' : 'Ready to swap'}
                </div>
              </div>
            )}
          </div>

          <Section title="Preview">
            {previewSrc ? (
              comparingEnhancers ? (() => {
                const activeList = selectedGridEnhancers.filter(e => meta.enhancers?.includes(e));
                const gridColsClass = activeList.length === 1 ? 'grid-cols-1' : 'grid-cols-2';
                return (
                  <div className="space-y-4">
                    {/* Enhancer selector row */}
                    <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 select-none">
                      <span className="text-[10px] font-black uppercase tracking-widest text-white/40 block">📊 Compare Enhancers (Select up to 4)</span>
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
                              className={`px-3 py-1.5 rounded-xl text-[11px] font-bold border transition-all duration-300 ${isSelected ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white shadow-[0_0_12px_var(--accent-glow)]' : 'bg-white/[0.02] border-white/5 text-white/50 hover:border-white/20 hover:text-white/85'}`}
                            >
                              {enh}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <EnhancerCompareGrid
                      activeList={activeList}
                      gridColsClass={gridColsClass}
                      enhancerPreviews={enhancerPreviews}
                      enhancerTimes={enhancerTimes}
                      liveRenderingTimers={liveRenderingTimers}
                    />
                  </div>
                );
              })() : (
                <InteractivePreview 
                  beforeSrc={rawUrl} 
                  afterSrc={progress.processing && progress.live_frame ? progress.live_frame : ((!isScrubbing && !isPlaying) ? previewSrc : (getCachedPreview(selTarget, frame)?.image || rawUrl))} 
                  faces={previewFaces}
                  splitView={splitView}
                  compare={compare}
                  setCompare={setCompare}
                  frame={frame}
                  setFrame={setFrame}
                  maxFrames={maxFrames}
                  previewing={previewing}
                  previewSecs={previewSecs}
                  setIsPlaying={setIsPlaying}
                  processing={progress.processing}
                  liveFrame={progress.live_frame}
                  liveSeq={progress.live_seq || 0}
                />
              )
            ) : (
              <div className="relative aspect-video rounded-2xl overflow-hidden bg-black/40 border border-white/5 flex items-center justify-center">
                <span className="text-[var(--text-muted)] text-sm">Select a target to preview</span>
              </div>
            )}
            
            {/* CINEMATIC TIMELINE SLIDER */}
            {maxFrames > 1 && (
              <div className="space-y-3 pt-3 border-t border-white/5 select-none">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-white/40">Cinematic Timeline</span>
                  <div className="text-[11px] font-mono text-[var(--text-muted)] bg-black/30 px-2 py-0.5 rounded border border-white/5">
                    Start: <span className="text-[var(--text-main)] font-semibold">{targets[selTarget]?.start_frame ?? 1}</span> · 
                    End: <span className="text-[var(--text-main)] font-semibold">{targets[selTarget]?.end_frame ?? maxFrames}</span>
                  </div>
                </div>

                {/* Timeline Track with Storyboard Background */}
                <div className="relative">
                  {hoverFrame !== null && (
                    <div 
                      className="absolute bottom-[66px] z-50 hud-glass p-2 rounded-xl border border-white/10 flex flex-col items-center gap-1.5 shadow-2xl pointer-events-none transition-all duration-75 select-none -translate-x-1/2"
                      style={{ left: `${maxFrames > 1 ? ((hoverFrame - 1) / (maxFrames - 1)) * 100 : 0}%` }}
                    >
                      <img
                        src={`${API}/api/target/preview?index=${selTarget}&frame=${hoverFrame}&width=240`}
                        alt="Hover Preview"
                        className="w-24 h-13 object-cover rounded-lg border border-white/10 bg-black/50"
                      />
                      <span className="text-[10px] font-mono font-extrabold text-[#C5A880] tracking-wider whitespace-nowrap">
                        FRAME {hoverFrame} · {((hoverFrame) / (targets[selTarget]?.fps || 25)).toFixed(2)}s
                      </span>
                      {/* Triangle indicator anchor pointer */}
                      <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 w-2.5 h-2.5 rotate-45 hud-glass border-b border-r border-white/10" />
                    </div>
                  )}

                  <div 
                    ref={timelineRef}
                    onPointerDown={handleTimelinePointerDown}
                    onPointerMove={handleTimelinePointerMove}
                    onPointerLeave={handleTimelinePointerLeave}
                    className="relative h-14 w-full rounded-xl bg-black/75 border border-white/10 overflow-hidden cursor-ew-resize select-none timeline-ticks timeline-glow-track shadow-2xl"
                  >
                  {/* Storyboard Thumbnails strip */}
                  {storyboardThumbs.length > 0 && (
                    <div className="absolute inset-0 flex opacity-25 pointer-events-none">
                      {storyboardThumbs.map((url, i) => (
                        <img 
                          key={i} 
                          src={url} 
                          alt="thumb" 
                          className="flex-1 h-full object-cover border-r border-white/5 last:border-r-0"
                          loading="lazy"
                        />
                      ))}
                    </div>
                  )}

                  {/* Active Track Highlight */}
                  <div 
                    className="absolute top-0 bottom-0 bg-[var(--accent)]/8 border-l-2 border-r-2 border-[var(--accent)]/30 z-10 pointer-events-none shadow-[inset_0_0_10px_rgba(233,69,96,0.15)]"
                    style={{ left: `${startPct}%`, width: `${endPct - startPct}%` }}
                  />

                  {/* Start Marker Handle */}
                  <div 
                    className="absolute top-0 bottom-0 w-1 bg-gradient-to-b from-emerald-400 to-teal-500 hover:from-emerald-300 hover:to-teal-400 z-25 transition-all"
                    style={{ left: `${startPct}%` }}
                  >
                    <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full bg-white border-2 border-emerald-400 shadow-[0_2px_4px_rgba(0,0,0,0.5)] pointer-events-none" />
                    {/* Visual marker bracket */}
                    <div className="absolute top-1/2 left-0 w-2.5 h-5 -translate-y-1/2 border-t-2 border-b-2 border-r-2 border-emerald-400 rounded-r pointer-events-none" />
                  </div>

                  {/* End Marker Handle */}
                  <div 
                    className="absolute top-0 bottom-0 w-1 bg-gradient-to-b from-amber-400 to-orange-500 hover:from-amber-300 hover:to-orange-400 z-25 transition-all"
                    style={{ left: `${endPct}%` }}
                  >
                    <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full bg-white border-2 border-orange-400 shadow-[0_2px_4px_rgba(0,0,0,0.5)] pointer-events-none" />
                    {/* Visual marker bracket */}
                    <div className="absolute top-1/2 right-0 w-2.5 h-5 -translate-y-1/2 border-t-2 border-b-2 border-l-2 border-orange-400 rounded-l pointer-events-none" />
                  </div>

                  {/* Red Playhead Line & Top diamond */}
                  <div 
                    className="absolute top-0 bottom-0 w-0.5 bg-gradient-to-b from-[#ff3366] to-[#E94560] z-30 pointer-events-none shadow-[0_0_12px_rgba(233,69,96,1)]"
                    style={{ left: `${currentPct}%` }}
                  >
                    <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2.5 h-2.5 rotate-45 bg-gradient-to-br from-[#ff5588] to-[#E94560] border border-white/35 shadow-[0_2px_6px_rgba(233,69,96,0.6)]" />
                  </div>
                </div>
              </div>

                {/* Cinematic Timeline Controls Toolbar */}
                <div className="flex flex-col sm:flex-row items-center justify-between gap-3 bg-black/35 p-3 rounded-xl border border-white/5 shadow-lg">
                  {/* Left Side: Timecode / Frame Info */}
                  <div className="text-xs text-[var(--text-muted)] font-mono flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-start">
                    <div className="flex items-center gap-1.5">
                      <span className="font-bold text-[var(--text-main)] text-sm">Frame {frame}</span>
                      <span className="opacity-40">/</span>
                      <span className="opacity-60">{maxFrames}</span>
                    </div>
                    <span className="opacity-30 hidden sm:inline">·</span>
                    <span className="bg-white/5 px-2 py-0.5 rounded text-[11px]">{(frame / (targets[selTarget]?.fps || 25)).toFixed(2)}s</span>
                  </div>

                  {/* Center: Playback Control Buttons */}
                  <div className="flex items-center gap-1 bg-black/40 p-1 rounded-lg border border-white/5">
                    <button 
                      onClick={() => setFrame(targets[selTarget]?.start_frame || 1)} 
                      className="p-1.5 hover:bg-white/5 rounded text-white/70 hover:text-white active:scale-95 transition-all"
                      title="Jump to Start Range"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5"/></svg>
                    </button>
                    <button 
                      onClick={() => setFrame(f => Math.max(1, f - 1))} 
                      className="p-1.5 hover:bg-white/5 rounded text-white/70 hover:text-white active:scale-95 transition-all"
                      title="Previous Frame (Left Arrow)"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="19 12 5 12"/><polyline points="12 19 5 12 12 5"/></svg>
                    </button>
                    
                    <button 
                      onClick={() => setIsPlaying(!isPlaying)} 
                      className={`px-3 py-1.5 rounded-lg font-bold text-xs flex items-center gap-1 active:scale-95 transition-all ${isPlaying ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' : 'bg-[var(--accent)] text-white shadow-[0_2px_8px_var(--accent-glow)] hover:bg-[var(--accent-hover)]'}`}
                      title="Play/Pause (Spacebar)"
                    >
                      {isPlaying ? (
                        <>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                          PAUSE
                        </>
                      ) : (
                        <>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                          PLAY
                        </>
                      )}
                    </button>

                    <button 
                      onClick={() => setFrame(f => Math.min(maxFrames, f + 1))} 
                      className="p-1.5 hover:bg-white/5 rounded text-white/70 hover:text-white active:scale-95 transition-all"
                      title="Next Frame (Right Arrow)"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 12 19 12"/><polyline points="12 5 19 12 12 19"/></svg>
                    </button>
                    <button 
                      onClick={() => setFrame(targets[selTarget]?.end_frame || maxFrames)} 
                      className="p-1.5 hover:bg-white/5 rounded text-white/70 hover:text-white active:scale-95 transition-all"
                      title="Jump to End Range"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/></svg>
                    </button>
                  </div>

                  {/* Trim Shortcuts */}
                  <div className="flex items-center gap-1 bg-black/20 p-1 rounded-lg border border-white/5">
                    <button 
                      onClick={() => setFrameMarkerVal('start', frame)}
                      className="px-2 py-1 hover:bg-emerald-500/10 rounded text-emerald-400 text-[10px] font-bold transition-all"
                      title="Set Range Start to Current Frame"
                    >
                      [ Set Start
                    </button>
                    <button 
                      onClick={() => setFrameMarkerVal('end', frame)}
                      className="px-2 py-1 hover:bg-orange-500/10 rounded text-orange-400 text-[10px] font-bold transition-all"
                      title="Set Range End to Current Frame"
                    >
                      Set End ]
                    </button>
                    <button 
                      onClick={async () => {
                        await setFrameMarkerVal('start', 1);
                        await setFrameMarkerVal('end', maxFrames);
                      }}
                      className="px-2 py-1 hover:bg-white/5 rounded text-white/50 hover:text-white text-[10px] font-bold transition-all"
                      title="Reset Range to Full Video"
                    >
                      Reset
                    </button>
                  </div>

                  {/* Right Side: Loop toggle & manual start/end frame inputs */}
                  <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
                    {/* Manual start/end numeric inputs */}
                    <div className="flex items-center gap-1.5 bg-black/20 px-2 py-1 rounded-lg border border-white/5">
                      <span className="text-[10px] text-white/40 font-semibold uppercase">Range:</span>
                      <input 
                        type="number"
                        value={targets[selTarget]?.start_frame || 1}
                        onChange={(e) => setFrameMarkerVal('start', parseInt(e.target.value, 10))}
                        className="bg-black/30 w-12 text-center text-xs font-mono font-bold text-emerald-400 outline-none rounded border border-white/5 py-0.5 focus:border-emerald-500"
                        title="Start Frame"
                      />
                      <span className="text-white/20 text-xs">-</span>
                      <input 
                        type="number"
                        value={targets[selTarget]?.end_frame || maxFrames}
                        onChange={(e) => setFrameMarkerVal('end', parseInt(e.target.value, 10))}
                        className="bg-black/30 w-12 text-center text-xs font-mono font-bold text-orange-400 outline-none rounded border border-white/5 py-0.5 focus:border-orange-500"
                        title="End Frame"
                      />
                    </div>

                    <button 
                      onClick={() => setIsLooping(!isLooping)} 
                      className={`p-1.5 rounded transition-all active:scale-95 ${isLooping ? 'bg-[var(--accent)]/15 text-[var(--accent)] border border-[var(--accent)]/30' : 'text-white/40 hover:text-white/60 hover:bg-white/5'}`} 
                      title="Toggle Loop Playback"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div className={`flex items-center gap-3 ${maxFrames > 1 ? 'pt-3 border-t border-white/5' : ''}`}>
              <Button size="sm" variant="secondary" onClick={() => refreshPreview()}>🔄 Refresh</Button>
              <Button size="sm" variant="primary" onClick={useFaceFromFrame}>Use face from frame</Button>
            </div>

            <div className="flex items-center flex-wrap gap-3">
              <Toggle label="✨ Live Swap" checked={fakePreview} onChange={setFakePreview} />
              <Toggle label="🔍 Compare" checked={compare} onChange={(v) => { setCompare(v); if (v) setComparingEnhancers(false); }} />
              {compare && <Toggle label="Split View" checked={splitView} onChange={setSplitView} />}
              <Toggle label="📊 Enhancer Grid" checked={comparingEnhancers} onChange={(v) => { setComparingEnhancers(v); if (v) setCompare(false); }} />
            </div>
          </Section>

          <Section title="Batch Swapping Queue">
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs text-white/50">
                {queue.length === 0 
                  ? 'No jobs in queue. Configure settings & click "Add Current to Queue".' 
                  : `${queue.length} jobs queued · ${queue.filter(j => j.status === 'Finished').length} finished`}
              </div>
              <div className="flex gap-2">
                <Button size="sm" variant="secondary" onClick={addToQueue} disabled={targets.length === 0 || sourceFaces.length === 0}>➕ Add Current to Queue</Button>
                {queue.length > 0 && (
                  <>
                    <Button size="sm" variant={isQueueRunning ? 'stop' : 'primary'} onClick={isQueueRunning ? () => setIsQueueRunning(false) : startQueue}>
                      {isQueueRunning ? '⏹ Stop Queue' : '▶ Start Queue'}
                    </Button>
                    <Button size="sm" variant="ghost" className="text-red-400" onClick={clearQueue}>Clear</Button>
                  </>
                )}
              </div>
            </div>

            {queue.length > 0 && (
              <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                {queue.map((job, idx) => {
                  const statusColors = {
                    Pending: 'text-white/60 bg-white/5 border-white/5',
                    Running: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30 animate-pulse shadow-[0_0_8px_rgba(234,179,8,0.2)]',
                    Finished: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
                    Failed: 'text-red-400 bg-red-500/10 border-red-500/30',
                  };
                  return (
                    <div key={job.id} className={`flex items-center justify-between px-3 py-2 rounded-xl text-xs border ${statusColors[job.status] || 'text-white bg-white/5'}`}>
                      <div className="flex-1 min-w-0 pr-3">
                        <span className="font-semibold text-white block truncate">{idx + 1}. {job.targetName}</span>
                        <span className="opacity-75 text-[10px] block truncate">Source: {job.sourceName} · Enhancer: {job.params.selected_enhancer || 'None'}</span>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <span className="font-bold uppercase text-[9px] tracking-wider px-2 py-0.5 rounded bg-black/30 border border-white/5">{job.status}</span>
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

function FileDrop({ label, accept, multiple, onFiles, busy, hint }) {
  const [drag, setDrag] = useState(false);
  const onDrop = (e) => {
    e.preventDefault(); setDrag(false);
    // Mark the native event as consumed so App.jsx's global drop handler
    // doesn't ALSO route these files (which popped the Source/Target dialog
    // on top of a drop that this zone already handled).
    e.nativeEvent.roopConsumed = true;
    if (busy) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length) onFiles(e.dataTransfer.files);
  };
  return (
    <label
      onDragOver={(e) => { e.preventDefault(); if (!busy) setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      className={`block ${busy ? 'cursor-wait pointer-events-none' : 'cursor-pointer group'}`}
    >
      <div className={`px-4 py-3.5 rounded-xl border-2 border-dashed text-center transition-all duration-300 ${busy ? 'border-[var(--accent)]/60 bg-[var(--accent)]/[0.06]' : drag ? 'bg-[var(--accent)]/[0.1] scale-[0.98]' : 'border-white/15 hover:border-[var(--accent)]/50 hover:bg-white/[0.02]'}`}>
        {busy ? (
          <span className="inline-flex items-center gap-2 text-xs font-semibold text-white/80">
            <span className="h-3.5 w-3.5 rounded-full border-2 border-white/30 border-t-[var(--accent)] animate-spin" />
            Uploading & analysing…
          </span>
        ) : (
          <div className="flex items-center justify-center gap-3 pointer-events-none">
            <div className={`p-2 rounded-xl bg-black/20 ${drag ? 'scale-110 text-[var(--accent)] shadow-[0_0_12px_var(--accent-glow)]' : 'text-white/40 group-hover:text-white/70 group-hover:bg-white/5'} transition-all duration-300`}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <div className="text-left">
              <span className={`text-xs font-bold tracking-wide block ${drag ? 'text-[var(--accent)]' : 'text-white/80'}`}>{drag ? 'Drop files now' : label}</span>
              {!drag && hint && <span className="block text-[10px] text-white/40 mt-0.5">{hint}</span>}
            </div>
          </div>
        )}
      </div>
      <input type="file" accept={accept} multiple={multiple}
        onChange={(e) => { if (e.target.files.length) onFiles(e.target.files); e.target.value = ''; }}
        disabled={busy} className="hidden" />
    </label>
  );
}

function FloatingEmojis() {
  const emojis = ['✨', '🪄', '🎭', '🔄', '🌟', '🚀', '🔮'];
  const [activeEmojis, setActiveEmojis] = useState([]);

  useEffect(() => {
    const interval = setInterval(() => {
      const e = emojis[Math.floor(Math.random() * emojis.length)];
      const id = Date.now();
      const left = Math.random() * 80 + 10;
      setActiveEmojis((prev) => [...prev, { id, emoji: e, left }]);
      
      setTimeout(() => {
        setActiveEmojis((prev) => prev.filter(item => item.id !== id));
      }, 2000);
    }, 400);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden z-10">
      {activeEmojis.map(({ id, emoji, left }) => (
        <div key={id} className="absolute bottom-4 text-2xl emoji-float" style={{ left: `${left}%` }}>
          {emoji}
        </div>
      ))}
    </div>
  );
}

function AIScannerOverlay() {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden z-20">
      {/* Dynamic Laser Line Sweep */}
      <div className="absolute w-full h-0.5 bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent opacity-85 shadow-[0_0_12px_var(--accent)] animate-scanner-sweep" />

      {/* Grid Overlay */}
      <div className="absolute inset-0 opacity-[0.04] bg-[linear-gradient(rgba(255,255,255,0.15)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.15)_1px,transparent_1px)] bg-[size:20px_20px] animate-pulse" />

      {/* Corner HUD Brackets */}
      <div className="absolute top-4 left-4 w-4 h-4 border-t-2 border-l-2 border-[var(--accent)]/55 opacity-70 animate-pulse" />
      <div className="absolute top-4 right-4 w-4 h-4 border-t-2 border-r-2 border-[var(--accent)]/55 opacity-70 animate-pulse" />
      <div className="absolute bottom-4 left-4 w-4 h-4 border-b-2 border-l-2 border-[var(--accent)]/55 opacity-70 animate-pulse" />
      <div className="absolute bottom-4 right-4 w-4 h-4 border-b-2 border-r-2 border-[var(--accent)]/55 opacity-70 animate-pulse" />

      {/* Subtle Digital Telemetry Labels */}
      <div className="absolute top-4 left-10 text-[8px] font-mono text-[var(--accent)]/45 uppercase tracking-widest select-none font-bold animate-pulse">SYSTEM SWAP SEARCHING: TARGET</div>
      <div className="absolute bottom-4 right-10 text-[8px] font-mono text-[var(--accent)]/45 uppercase tracking-widest select-none font-bold animate-pulse">GRID OVERLAY: ACTIVE</div>
    </div>
  );
}

function EnhancerCompareGrid({ activeList, gridColsClass, enhancerPreviews, enhancerTimes, liveRenderingTimers }) {
  // Zoom/pan is shared by every cell, so the same region is magnified in all
  // enhancers at once — that's what makes fine differences readable.
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  useEffect(() => {
    const handleUp = () => setIsPanning(false);
    window.addEventListener('mouseup', handleUp);
    window.addEventListener('touchend', handleUp);
    return () => {
      window.removeEventListener('mouseup', handleUp);
      window.removeEventListener('touchend', handleUp);
    };
  }, []);

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomSpeed = 0.15;
    const newZoom = Math.min(Math.max(1, zoom + (e.deltaY < 0 ? zoomSpeed : -zoomSpeed)), 8);
    if (newZoom === 1) setPan({ x: 0, y: 0 });
    setZoom(newZoom);
  };

  const handlePointerDown = (e) => {
    if (zoom > 1) {
      setIsPanning(true);
      setStartPan({ x: (e.clientX ?? e.touches?.[0]?.clientX) - pan.x, y: (e.clientY ?? e.touches?.[0]?.clientY) - pan.y });
    }
  };

  const handlePointerMove = (e) => {
    if (!isPanning || zoom <= 1) return;
    const cx = e.clientX ?? e.touches?.[0]?.clientX;
    const cy = e.clientY ?? e.touches?.[0]?.clientY;
    setPan({ x: cx - startPan.x, y: cy - startPan.y });
  };

  // Double click toggles fit ↔ 2.5x centered near the clicked spot of that cell
  const handleDoubleClick = (e) => {
    if (zoom > 1) {
      setZoom(1);
      setPan({ x: 0, y: 0 });
    } else {
      setZoom(2.5);
      const cell = e.currentTarget;
      const rect = cell.getBoundingClientRect();
      const clickX = (e.clientX ?? e.touches?.[0]?.clientX) - rect.left;
      const clickY = (e.clientY ?? e.touches?.[0]?.clientY) - rect.top;
      setPan({
        x: (rect.width / 2 - clickX) * 1.5,
        y: (rect.height / 2 - clickY) * 1.5
      });
    }
  };

  const resetZoom = () => { setZoom(1); setPan({ x: 0, y: 0 }); };
  const transformStyle = { transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`, transformOrigin: 'center' };

  return (
    <div
      ref={containerRef}
      className={`relative grid ${gridColsClass} gap-3 aspect-video max-h-[45vh] rounded-2xl overflow-hidden bg-black/45 border border-white/5 p-2`}
      onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove}
      style={{ touchAction: 'none', cursor: zoom > 1 ? (isPanning ? 'grabbing' : 'grab') : 'zoom-in' }}
    >
      {activeList.map((enh) => (
        <div key={enh} className="relative rounded-xl overflow-hidden bg-black/50 border border-white/5 flex items-center justify-center"
             onDoubleClick={handleDoubleClick}>
          {enhancerPreviews[enh] ? (
            <div className="w-full h-full flex items-center justify-center transition-transform duration-75 select-none" style={transformStyle}>
              <img src={enhancerPreviews[enh]} alt={enh} className="max-w-full max-h-full object-contain pointer-events-none" draggable={false} />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center gap-2 text-white/40 text-xs">
              <span className="h-4 w-4 rounded-full border-2 border-white/20 border-t-[var(--accent)] animate-spin" />
              <span className="font-semibold">Rendering {enh}…</span>
              {liveRenderingTimers[enh] && (
                <span className="text-[10px] text-white/30 font-mono">
                  Elapsed: {liveRenderingTimers[enh]}
                </span>
              )}
            </div>
          )}
          <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur border border-white/10 text-[10px] font-bold text-white uppercase pointer-events-none">{enh}</span>
          {enhancerTimes[enh] && (
            <span className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur border border-white/10 text-[10px] font-bold text-white/60 font-mono pointer-events-none">
              ⏱️ {enhancerTimes[enh]}
            </span>
          )}
        </div>
      ))}
      {zoom > 1 && (
        <button
          type="button"
          onClick={resetZoom}
          className="absolute top-3 right-3 z-30 px-2.5 py-1 rounded-full bg-black/70 backdrop-blur border border-white/10 text-[11px] font-bold text-white/80 hover:text-white hover:border-white/30 transition-all"
          title="Reset zoom (or double-click)"
        >
          🔍 {zoom.toFixed(1)}× — Reset
        </button>
      )}
      {zoom === 1 && (
        <span className="absolute top-3 right-3 z-30 px-2.5 py-1 rounded-full bg-black/50 backdrop-blur text-[10px] font-semibold text-white/40 pointer-events-none select-none">
          Scroll or double-click to zoom all
        </span>
      )}
    </div>
  );
}

function InteractivePreview({
  beforeSrc, 
  afterSrc, 
  faces = [], 
  splitView = false,
  compare = false,
  setCompare,
  setFrame,
  maxFrames = 1,
  previewing = false,
  previewSecs = 0,
  setIsPlaying,
  processing = false,
  liveFrame = null,
  liveSeq = 0
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
        setZoom((z) => Math.min(z + 0.5, 5));
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

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomSpeed = 0.15;
    const newZoom = Math.min(Math.max(1, zoom + (e.deltaY < 0 ? zoomSpeed : -zoomSpeed)), 5);
    if (newZoom === 1) setPan({ x: 0, y: 0 });
    setZoom(newZoom);
  };

  const handlePointerDown = (e) => {
    if (zoom > 1 && !isDraggingSlider) {
      setIsPanning(true);
      setStartPan({ x: (e.clientX || e.touches?.[0]?.clientX) - pan.x, y: (e.clientY || e.touches?.[0]?.clientY) - pan.y });
    }
  };

  const handlePointerMove = (e) => {
    const cx = e.clientX ?? e.touches?.[0]?.clientX;
    const cy = e.clientY ?? e.touches?.[0]?.clientY;
    if (isDraggingSlider) {
      handleSliderMove(cx);
    } else if (isPanning && zoom > 1) {
      setPan({ x: cx - startPan.x, y: cy - startPan.y });
    }
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

  const renderFaces = () => {
    if (!faces.length || !imgDim || !showBoxes) return null;
    return faces.map((bbox, i) => {
      const [sx, sy, ex, ey] = bbox;
      const left = (sx / imgDim.w) * 100;
      const top = (sy / imgDim.h) * 100;
      const width = ((ex - sx) / imgDim.w) * 100;
      const height = ((ey - sy) / imgDim.h) * 100;
      return (
        <div key={i} className="absolute border-2 border-[var(--accent)] shadow-[0_0_10px_var(--accent-glow)] z-20 pointer-events-none"
             style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}>
          <span className="absolute -top-6 left-0 bg-[var(--accent)] text-white text-[10px] font-bold px-1.5 py-0.5 rounded whitespace-nowrap">
            Person {i}
          </span>
        </div>
      );
    });
  };

  const transformStyle = { transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`, transformOrigin: 'center' };
  const aspectStyle = { aspectRatio: imgDim ? `${imgDim.w}/${imgDim.h}` : '1', maxHeight: '100%', maxWidth: '100%', display: 'flex' };

  const triggerFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  // Split View comparisons
  if (compare && splitView) {
    return (
      <div 
        ref={containerRef}
        className={`relative w-full aspect-video max-h-[45vh] rounded-2xl overflow-hidden bg-black/40 border border-white/5 group shadow-xl ${isFullscreen ? 'h-screen w-screen' : ''}`}
        onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onDoubleClick={handleDoubleClick} style={{ touchAction: 'none' }}
      >
        <div className="flex w-full h-full transition-transform duration-75" style={transformStyle}>
          <div className="flex-1 relative border-r border-white/10 flex items-center justify-center overflow-hidden bg-black/50">
            <div className="relative" style={aspectStyle}>
              <img src={beforeSrc} alt="Before" className="w-full h-full object-contain pointer-events-none" 
                   onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })} />
              <div className="absolute inset-0 pointer-events-none">{renderFaces()}</div>
            </div>
            <span className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur text-[11px] font-bold text-white/80 uppercase">Before</span>
          </div>
          <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-black/50">
            <div className="relative" style={aspectStyle}>
              <img src={afterSrc} alt="After" className="w-full h-full object-contain pointer-events-none" />
            </div>
            <span className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-[var(--accent)]/80 backdrop-blur text-[11px] font-bold text-white uppercase">After</span>
          </div>
        </div>

        {/* HUD control bar overlays */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 p-1.5 rounded-xl hud-glass opacity-60 hover:opacity-100 transition-all duration-300 z-50">
          <button onClick={() => setZoom(z => Math.min(z + 0.5, 5))} className="p-2 text-xs font-bold font-mono rounded-lg hud-glass-button" title="Zoom In">+</button>
          <button onClick={() => setZoom(z => { const nz = Math.max(1, z - 0.5); if (nz === 1) setPan({x:0, y:0}); return nz; })} className="p-2 text-xs font-bold font-mono rounded-lg hud-glass-button" title="Zoom Out">-</button>
          <button onClick={() => { setZoom(1); setPan({x:0, y:0}); }} className="px-2 py-1 text-[10px] font-bold rounded-lg hud-glass-button" title="Reset view">FIT</button>
          <div className="w-px h-4 bg-white/10 mx-1" />
          <button onClick={() => setShowBoxes(b => !b)} className={`px-2 py-1 text-[10px] font-bold rounded-lg apple-transition ${showBoxes ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'}`}>BOXES</button>
          <button onClick={triggerFullscreen} className="p-2 rounded-lg hud-glass-button" title="Toggle Fullscreen">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
          </button>
        </div>
      </div>
    );
  }

  // Standard or Slide-Comparison View
  return (
    <div 
      className={`relative w-full aspect-video max-h-[45vh] rounded-2xl overflow-hidden bg-black/40 border border-white/5 select-none group shadow-xl ${isFullscreen ? 'h-screen w-screen' : ''} ${previewing ? 'preview-glow' : ''}`}
      ref={containerRef} onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onDoubleClick={handleDoubleClick} style={{ touchAction: 'none' }}
    >
      {previewing && <AIScannerOverlay />}
      {previewing && (
        <div className="absolute top-3 right-3 flex flex-col items-end gap-1.5 z-50">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/70 backdrop-blur-md text-xs font-bold text-white/95 tabular-nums border border-white/10 shadow-2xl">
            <span className="h-3 w-3 rounded-full border-2 border-white/30 border-t-[var(--accent)] animate-spin" />
            Rendering… {previewSecs}s
          </div>
        </div>
      )}
      {processing && liveFrame && (
        <div className="absolute top-3 right-3 flex flex-col items-end gap-1.5 z-50">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--accent)]/90 backdrop-blur-md text-[10px] font-black tracking-widest text-white border border-[var(--accent)]/30 shadow-2xl animate-pulse uppercase">
            <span className="h-2 w-2 rounded-full bg-white animate-ping" />
            Live Swapping{liveSeq > 0 ? ` · ${liveSeq} frames` : ''}
          </div>
        </div>
      )}

      {/* Frame navigation shortcuts popup guide (visible when video) */}
      {maxFrames > 1 && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 px-2.5 py-1 rounded-lg bg-black/50 backdrop-blur text-[10px] font-bold text-white/50 border border-white/5 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-30 flex items-center gap-2">
          <span>Shortcuts:</span>
          <span className="bg-white/10 px-1 py-0.5 rounded text-white font-mono">←</span>
          <span className="bg-white/10 px-1 py-0.5 rounded text-white font-mono">→</span>
          <span>Frame</span>
          <span className="bg-white/10 px-2 py-0.5 rounded text-white font-mono">Space</span>
          <span>Compare</span>
        </div>
      )}

      <div className="absolute inset-0 transition-transform duration-75 flex items-center justify-center" style={transformStyle}>
        
        {/* Before Image & Bounding Boxes Wrapper */}
        <div className="relative z-10" style={aspectStyle} ref={imageRef}>
          <img src={beforeSrc} alt="Before" className="w-full h-full object-contain pointer-events-none" 
               onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })} draggable={false} />
          <div className="absolute inset-0 pointer-events-none z-30">{renderFaces()}</div>
          
          {/* After Image Overlay with Clip-path */}
          <div className="absolute inset-0 pointer-events-none z-20" 
               style={{ clipPath: compare ? `polygon(${sliderPosition}% 0, 100% 0, 100% 100%, ${sliderPosition}% 100%)` : 'none' }}>
            <img src={afterSrc} alt="After" className="w-full h-full object-contain" draggable={false} />
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

      {compare && <span className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur text-[11px] font-bold tracking-wider text-white/80 uppercase shadow z-30 pointer-events-none">Before</span>}
      <span className="absolute top-3 right-3 px-2.5 py-1 rounded-full bg-[var(--accent)]/80 backdrop-blur text-[11px] font-bold tracking-wider text-white uppercase shadow z-30 transition-opacity duration-300 pointer-events-none"
            style={{ opacity: compare && sliderPosition > 85 ? 0 : 1 }}>{compare ? 'After' : 'Swapped'}</span>

      {/* HUD control bar overlays */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 p-1.5 rounded-xl hud-glass opacity-60 group-hover:opacity-100 hover:opacity-100 transition-all duration-300 z-50">
        <button onClick={() => setZoom(z => Math.min(z + 0.5, 5))} className="p-2 text-xs font-bold font-mono rounded-lg hud-glass-button" title="Zoom In">+</button>
        <button onClick={() => setZoom(z => { const nz = Math.max(1, z - 0.5); if (nz === 1) setPan({x:0, y:0}); return nz; })} className="p-2 text-xs font-bold font-mono rounded-lg hud-glass-button" title="Zoom Out">-</button>
        <button onClick={() => { setZoom(1); setPan({x:0, y:0}); }} className="px-2 py-1 text-[10px] font-bold rounded-lg hud-glass-button" title="Reset view">FIT</button>
        <div className="w-px h-4 bg-white/10 mx-1" />
        <button onClick={() => setShowBoxes(b => !b)} className={`px-2 py-1 text-[10px] font-bold rounded-lg apple-transition ${showBoxes ? 'text-[var(--accent)] bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'}`}>BOXES</button>
        <button onClick={triggerFullscreen} className="p-2 rounded-lg hud-glass-button" title="Toggle Fullscreen">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
        </button>
      </div>
    </div>
  );
}
