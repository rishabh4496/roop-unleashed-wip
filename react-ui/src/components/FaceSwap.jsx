import React, { useEffect, useRef, useState } from 'react';
import { getJSON, postJSON, postFiles, API } from '../api';
import { Section, Select, Slider, Toggle, TextInput, Button, FaceGallery } from './ui';

const num = (v, d) => (v === undefined || v === null || v === '' ? d : Number(v));

const fmtTime = (ms) => {
  const s = Math.max(0, Math.round(ms / 1000));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${String(s % 60).padStart(2, '0')}s` : `${s}s`;
};

export default function FaceSwap({ meta, settings, setSettings, notify }) {
  const [sourceFaces, setSourceFaces] = useState([]);
  const [targetFaces, setTargetFaces] = useState([]);
  const [targetGroups, setTargetGroups] = useState([]);
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
  
  // Custom Timeline and Playback States
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(true);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [dragType, setDragType] = useState('playhead'); // 'playhead', 'start', 'end'
  const [storyboardThumbs, setStoryboardThumbs] = useState([]);
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

  // ── initial rehydrate ──
  // Pinokio reloads the webview whenever you switch the RUN/DEV/FILES tabs,
  // which remounts this component and wipes its React state. The backend keeps
  // running, so we restore both the faces/targets AND the live job state.
  useEffect(() => {
    getJSON('/api/state').then((st) => {
      setSourceFaces(st.source_faces || []);
      setTargetFaces(st.target_faces || []);
      setTargetGroups(st.target_groups || []);
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
        use_frontalization: p.use_frontalization, frontalization_threshold: num(p.frontalization_threshold, 25),
        swap_model: p.swap_model, default_det_size: p.default_det_size,
      }, { signal: ctrl.signal });
      if (res.faces) setPreviewFaces(res.faces);
      setPreviewSrc(res.image || '');
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
    dds: p.default_det_size,
  });
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
      const res = await postFiles('/api/target/add', files);
      setTargets(res.targets);
      setSelTarget(res.selected_target_index || 0);
      const mf = res.targets[res.selected_target_index || 0]?.frames || 1;
      setMaxFrames(mf); setFrame(1);
      refreshPreview({ index: res.selected_target_index || 0, frame: 1 });
      notify(`Added ${res.targets.length} target(s)`);
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
  };

  const selectSource = async (i) => { setSelSource(i); await postJSON('/api/source/select', { index: i }); };

  const useFaceFromFrame = async () => {
    try {
      const res = await postJSON('/api/target/use_face', { index: selTarget, frame });
      setTargetFaces(res.target_faces);
      setTargetGroups(res.target_groups || []);
      set('face_detection_mode', 'Selected face');
      notify(`Added ${res.count} target person(s)`);
    } catch (e) { notify(e.message, 'error'); }
  };

  // Add another angle (profile/side/upside-down) of the SELECTED person, from
  // the current frame, so matching survives pose changes (less flicker).
  const addAngle = async () => {
    const person = targetGroups[selTargetFace] ?? 0;
    try {
      const res = await postJSON('/api/target/add_angle', { person, index: selTarget, frame });
      setTargetFaces(res.target_faces);
      setTargetGroups(res.target_groups || []);
      if (res.count) notify(`Added angle to Person ${person + 1} (match ${res.distance})`);
      else notify(res.message || 'No matching face in this frame', 'error');
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
        face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
        num_swap_steps: num(p.num_swap_steps, 1),
      });
      startTimeRef.current = Date.now();
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

  const out = progress.output;
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
      urls.push(`${API}/api/target/preview?index=${selTarget}&frame=${f}`);
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
        face_distance: num(p.max_face_distance, 0.85), blend_ratio: num(p.blend_ratio, 0.8),
        num_swap_steps: num(p.num_swap_steps, 1),
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

  const startFrame = targets[selTarget]?.start_frame || 1;
  const endFrame = targets[selTarget]?.end_frame || maxFrames;
  const startPct = maxFrames > 1 ? ((startFrame - 1) / (maxFrames - 1)) * 100 : 0;
  const endPct = maxFrames > 1 ? ((endFrame - 1) / (maxFrames - 1)) * 100 : 100;
  const currentPct = maxFrames > 1 ? ((frame - 1) / (maxFrames - 1)) * 100 : 0;

  return (
    <div className="flex flex-col lg:flex-row gap-6 items-start w-full">
      {/* COLUMN 1: Settings & Controls (Scrollable sidebar on large viewports) */}
      <div className="w-full lg:w-[400px] 3xl:w-[820px] 4xl:w-[900px] shrink-0 lg:max-h-[calc(100vh-140px)] lg:overflow-y-auto pr-0 lg:pr-2 select-none">
        <div className="grid grid-cols-1 3xl:grid-cols-2 gap-6">
          <Section title="Swap settings">
          <Select label="Swap model" info="inswapper 128 · reswapper 256 · hyperswap 256 (downloads on first use)" value={p.swap_model} onChange={(v) => set('swap_model', v)} options={meta.swap_models} />
          <Select label="Face selection" value={p.face_detection_mode} onChange={(v) => set('face_detection_mode', v)} options={meta.face_detection_modes} />
          <Toggle label="High-accuracy detection (640px)" info="On = 640px (accurate, default). Off = 320px: ~4× faster face detection, but may miss small or distant faces. Leave on for far/multi-face shots; turn off to speed up close-up swaps." checked={p.default_det_size !== false} onChange={(v) => set('default_det_size', v)} />
          <Slider label="Swapping steps" info="more = more likeness" min={1} max={5} step={1} value={num(p.num_swap_steps, 1)} onChange={(v) => set('num_swap_steps', v)} />
          <Select label="Post-processing enhancer" value={p.selected_enhancer} onChange={(v) => set('selected_enhancer', v)} options={meta.enhancers} />
          <Slider label="Max face similarity" info="0=identical 1=any" min={0.01} max={1} step={0.01} value={num(p.max_face_distance, 0.85)} onChange={(v) => set('max_face_distance', v)} />
          <Select label="Subsample upscale" value={p.subsample_upscale} onChange={(v) => set('subsample_upscale', v)} options={meta.upscale} />
          <Slider label="Original/Enhanced blend" min={0} max={1} step={0.01} value={num(p.blend_ratio, 0.8)} onChange={(v) => set('blend_ratio', v)} />
        </Section>

        <Section title="Masking parameters">
          <Select label="Masking engine" value={p.mask_engine} onChange={(v) => set('mask_engine', v)} options={meta.mask_engines} />
          {p.mask_engine === 'Clip2Seg' && (
            <TextInput label="Objects to mask & restore" value={p.mask_clip_text} onChange={(v) => set('mask_clip_text', v)} placeholder="cup,hands,hair" />
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
            <Slider label="Frontalize above angle (°)" info="Frontalization kicks in when the face yaw/pitch exceeds this. Lower = frontalize more; higher = only the steepest." min={10} max={60} step={5} value={num(p.frontalization_threshold, 25)} onChange={(v) => set('frontalization_threshold', v)} />
          )}
        </Section>

        <Section title="Video parameters">
          <Select label="Video method" value={p.video_swapping_method} onChange={(v) => set('video_swapping_method', v)} options={meta.video_methods} />
          <Select label="On no face detected" value={p.no_face_action} onChange={(v) => set('no_face_action', v)} options={meta.no_face_actions} />
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
          <Toggle label="✨ Reduce enhancer flicker" info="Temporally blends the enhanced face. NOTE: forces SINGLE-THREAD, so it's slower." checked={!!p.stabilize_enhancer} onChange={(v) => set('stabilize_enhancer', v)} />
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
      </div>
    </div>

      {/* COLUMN 2 & 3 WRAPPER: Media asset managers on left and large active workspace on right */}
      <div className="flex-1 w-full space-y-6 flex flex-col 2xl:flex-row gap-6">
        
        {/* COLUMN 2: Media Asset Manager */}
        <div className="w-full 2xl:w-[360px] 3xl:w-[380px] shrink-0 space-y-6 select-none">
          <Section title="Source images / facesets">
            <FileDrop accept="image/*,.fsz" multiple label="Add source faces" onFiles={onAddSource} busy={uploadingSrc} hint="drop images or .fsz here" />
            <FaceGallery title="Input faces" faces={sourceFaces} selected={selSource} onSelect={selectSource}
              onRemove={(i) => sourceAction('/api/source/remove', { index: i })} empty="Upload a face image" />
            {sourceFaces.length > 0 && (
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/move', { index: selSource, direction: 'left' })}>⬅ Move</Button>
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/move', { index: selSource, direction: 'right' })}>Move ➡</Button>
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/remove', { index: selSource })}>❌ Remove</Button>
                <Button size="sm" variant="stop" onClick={() => sourceAction('/api/source/clear', {})}>Clear all</Button>
              </div>
            )}
          </Section>

          <Section title="Target file(s)">
            <FileDrop accept="image/*,video/*,.webp" multiple label="Add target media" onFiles={onAddTarget} busy={uploadingTgt} hint="drop images or videos here" />
            {targets.length === 0 ? (
              <div className="h-24 flex items-center justify-center rounded-lg border border-dashed border-white/10 text-xs text-white/30">No targets yet</div>
            ) : (
              <div className="space-y-1.5 max-h-40 overflow-auto">
                {targets.map((t, i) => (
                  <div key={i}
                    className={`group flex items-center gap-2 px-3 py-2 rounded-xl text-sm border transition-colors cursor-pointer ${selTarget === i ? 'bg-[var(--accent)]/15 border-[var(--accent)]/50 shadow-[0_0_10px_var(--accent-glow)]' : 'bg-black/20 border-white/5 hover:border-white/20'}`}
                    onClick={() => selectTarget(i)}>
                    <div className="flex-1 min-w-0">
                      <span className="truncate block">{t.name}</span>
                      <span className="text-xs text-[var(--text-muted)]">{t.frames > 1 ? `${t.frames} frames` : 'image'}</span>
                    </div>
                    <button type="button" title="Remove this target"
                      onClick={(e) => { e.stopPropagation(); removeTarget(i); }}
                      className="h-6 w-6 shrink-0 rounded-full bg-black/40 text-white/60 opacity-0 group-hover:opacity-100 hover:bg-[var(--accent-hover)] hover:text-white transition-opacity flex items-center justify-center">✕</button>
                  </div>
                ))}
              </div>
            )}
            {targets.length > 0 && <Button size="sm" variant="stop" onClick={async () => { const r = await postJSON('/api/target/clear', {}); setTargets(r.targets); setTargetFaces([]); setTargetGroups([]); setPreviewSrc(''); }}>Clear targets</Button>}
          </Section>

          <Section title="Target faces">
            <FaceGallery title="" faces={targetFaces} selected={selTargetFace} onSelect={setSelTargetFace}
              groups={targetGroups} vertical={true}
              onRemove={async (i) => { const r = await postJSON('/api/target/remove_face', { index: i }); setTargetFaces(r.target_faces); setTargetGroups(r.target_groups || []); if (selTargetFace >= r.target_faces.length) setSelTargetFace(Math.max(0, r.target_faces.length - 1)); }}
              empty={targets.length === 0 ? 'No target loaded' : "Use 'face from frame'"} />
            {targetFaces.length > 0 && (
              <div className="mt-3 space-y-2">
                <Button size="sm" variant="primary" className="w-full justify-center" onClick={addAngle}>
                  ➕ Add angle to Person {(targetGroups[selTargetFace] ?? 0) + 1}
                </Button>
                <div className="flex flex-wrap gap-2">
                  <Button size="sm" variant="secondary" onClick={() => { const g = [...targetGroups]; g[selTargetFace] = Math.max(0, (g[selTargetFace] || 0) - 1); setTargetGroups(g); postJSON('/api/target/group', { groups: g }); }}>👥 Group -</Button>
                  <Button size="sm" variant="secondary" onClick={() => { const g = [...targetGroups]; g[selTargetFace] = (g[selTargetFace] || 0) + 1; setTargetGroups(g); postJSON('/api/target/group', { groups: g }); }}>Group +</Button>
                  <Button size="sm" variant="stop" onClick={() => { const g = targetGroups.map(() => 0); setTargetGroups(g); postJSON('/api/target/group', { groups: g }); }}>Reset</Button>
                </div>
                <p className="text-[11px] text-[var(--text-muted)] leading-snug">
                  Capture profile / side / upside-down views of the selected person across frames — matching uses the closest angle, so the swap doesn't drop out when the head turns. Each colored group = one person → one source faceset.
                </p>
              </div>
            )}
          </Section>
        </div>

        {/* COLUMN 3: Active Canvas, Timeline & Outputs */}
        <div className="flex-1 min-w-0 space-y-6">
          <Section title="Preview">
            {previewSrc ? (
              <InteractivePreview 
                beforeSrc={rawUrl} 
                afterSrc={(!isScrubbing && !isPlaying) ? previewSrc : rawUrl} 
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
              />
            ) : (
              <div className="relative aspect-video rounded-2xl overflow-hidden bg-black/40 border border-white/5 flex items-center justify-center">
                <span className="text-[var(--text-muted)] text-sm">Select a target to preview</span>
              </div>
            )}
            
            <div className="flex items-center flex-wrap gap-3">
              <Toggle label="✨ Live Swap" checked={fakePreview} onChange={setFakePreview} />
              <Toggle label="🔍 Compare" checked={compare} onChange={setCompare} />
              {compare && <Toggle label="Split View" checked={splitView} onChange={setSplitView} />}
              <Button size="sm" variant="secondary" onClick={() => refreshPreview()}>🔄 Refresh</Button>
              <Button size="sm" variant="primary" onClick={useFaceFromFrame}>Use face from frame</Button>
            </div>

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
                <div 
                  ref={timelineRef}
                  onPointerDown={handleTimelinePointerDown}
                  className="relative h-14 w-full rounded-xl bg-black/60 border border-white/10 overflow-hidden cursor-ew-resize select-none shadow-inner"
                >
                  {/* Storyboard Thumbnails strip */}
                  {storyboardThumbs.length > 0 && (
                    <div className="absolute inset-0 flex opacity-30 pointer-events-none">
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
                    className="absolute top-0 bottom-0 bg-[var(--accent)]/15 border-l border-r border-[var(--accent)]/30 z-10 pointer-events-none"
                    style={{ left: `${startPct}%`, width: `${endPct - startPct}%` }}
                  />

                  {/* Start Marker Handle */}
                  <div 
                    className="absolute top-0 bottom-0 w-1.5 bg-emerald-500 hover:bg-emerald-400 z-25 transition-colors"
                    style={{ left: `${startPct}%` }}
                  >
                    <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full bg-emerald-500 border border-white/20 shadow-md pointer-events-none" />
                    {/* Visual marker bracket */}
                    <div className="absolute top-1/2 left-0 w-2 h-4 -translate-y-1/2 border-t-2 border-b-2 border-r-2 border-emerald-500 rounded-r pointer-events-none" />
                  </div>

                  {/* End Marker Handle */}
                  <div 
                    className="absolute top-0 bottom-0 w-1.5 bg-orange-500 hover:bg-orange-400 z-25 transition-colors"
                    style={{ left: `${endPct}%` }}
                  >
                    <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full bg-orange-500 border border-white/20 shadow-md pointer-events-none" />
                    {/* Visual marker bracket */}
                    <div className="absolute top-1/2 right-0 w-2 h-4 -translate-y-1/2 border-t-2 border-b-2 border-l-2 border-orange-500 rounded-l pointer-events-none" />
                  </div>

                  {/* Red Playhead Line & Top diamond */}
                  <div 
                    className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-30 pointer-events-none shadow-[0_0_8px_rgba(239,68,68,0.8)]"
                    style={{ left: `${currentPct}%` }}
                  >
                    <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-3 h-3 rotate-45 bg-red-500 border border-white/20 shadow-md" />
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
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white font-bold transition-colors">⬇ Download</a>
                    <Button size="sm" variant="secondary" onClick={revealOutput}>📂 Open folder</Button>
                  </div>
                </div>
              )}
            </div>
          </Section>

          {/* run bar */}
          <div className="sticky bottom-0 py-3 z-30">
            <div className="rounded-2xl glass-panel p-4 flex items-center gap-4 shadow-2xl">
              {progress.processing ? (
                <div className="flex items-center gap-2">
                  {progress.paused ? (
                    <Button variant="primary" size="lg" onClick={resume}>▶ Resume</Button>
                  ) : (
                    <Button variant="secondary" size="lg" onClick={pause}>⏸ Pause</Button>
                  )}
                  <Button variant="stop" size="lg" onClick={stop}>⏹ Stop</Button>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Button variant="primary" size="lg" onClick={start} disabled={targets.length === 0 || sourceFaces.length === 0}>▶ Start Swapping</Button>
                  {maxFrames > 1 && (
                    <Button variant="secondary" size="lg" onClick={renderPreviewClip} disabled={targets.length === 0 || sourceFaces.length === 0 || isGeneratingPreviewClip} className="!text-orange-400 border border-orange-500/20 hover:bg-orange-500/10">
                      ⚡ Render 5s Preview
                    </Button>
                  )}
                </div>
              )}
              <div className="flex-1">
                <div className="flex justify-between text-xs text-[var(--text-muted)] mb-1">
                  <span>{progress.desc || (progress.processing ? 'Processing…' : 'Idle')}</span>
                  <span className="flex items-center gap-2 tabular-nums">
                    {progress.processing && (
                      <span className="text-[var(--text-muted)] opacity-70">⏱ {fmtTime(elapsedMs)}{etaMs > 0 ? ` · ETA ${fmtTime(etaMs)}` : ''}</span>
                    )}
                    <span className="font-bold text-[var(--text-main)]">{Math.round(prog * 100)}%</span>
                  </span>
                </div>
                <div className="h-2.5 rounded-full bg-black/40 overflow-hidden relative shadow-inner">
                  <div className={`absolute top-0 bottom-0 left-0 bg-[var(--accent)] transition-all duration-300 ${progress.processing ? 'progress-bar-animated shadow-[0_0_10px_var(--accent-glow)]' : ''}`} style={{ width: `${(progress.progress || 0) * 100}%` }} />
                </div>
                {progress.error && <div className="text-xs text-red-400 mt-1">{progress.error}</div>}
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

function FileDrop({ label, accept, multiple, onFiles, busy, hint }) {
  const [drag, setDrag] = useState(false);
  const onDrop = (e) => {
    e.preventDefault(); setDrag(false);
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
      <div className={`px-4 py-8 rounded-xl border-2 border-dashed text-center transition-all duration-300 ${busy ? 'border-[var(--accent)]/60 bg-[var(--accent)]/[0.06]' : drag ? 'border-[var(--accent)] bg-[var(--accent)]/[0.1] scale-[0.98]' : 'border-white/15 hover:border-[var(--accent)]/50 hover:bg-white/[0.02]'}`}>
        {busy ? (
          <span className="inline-flex items-center gap-2 text-sm font-medium text-white/80">
            <span className="h-4 w-4 rounded-full border-2 border-white/30 border-t-[var(--accent)] animate-spin" />
            Uploading & analysing…
          </span>
        ) : (
          <div className="flex flex-col items-center justify-center gap-3 pointer-events-none">
            <div className={`p-3.5 rounded-full bg-black/20 ${drag ? 'scale-110 text-[var(--accent)] shadow-[0_0_15px_var(--accent-glow)]' : 'text-white/40 group-hover:text-white/70 group-hover:bg-white/5'} transition-all duration-300`}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <div>
              <span className={`text-sm font-semibold tracking-wide ${drag ? 'text-[var(--accent)]' : 'text-white/80'}`}>{drag ? 'Drop files now' : label}</span>
              {!drag && hint && <span className="block text-xs text-white/40 mt-1">{hint}</span>}
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
  setIsPlaying
}) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef(null);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [showBoxes, setShowBoxes] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [imgDim, setImgDim] = useState(null);

  const handleSliderMove = (clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
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

  // Keyboard Navigation: arrow keys step frames, spacebar toggles play/compare
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignore if user is typing in input fields
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;

      if (e.key === 'ArrowLeft' && maxFrames > 1 && setFrame) {
        e.preventDefault();
        setFrame((f) => Math.max(1, f - 1));
      } else if (e.key === 'ArrowRight' && maxFrames > 1 && setFrame) {
        e.preventDefault();
        setFrame((f) => Math.min(maxFrames, f + 1));
      } else if (e.key === ' ') {
        e.preventDefault();
        if (maxFrames > 1 && setIsPlaying) {
          setIsPlaying((p) => !p);
        } else if (setCompare) {
          setCompare((c) => !c);
        }
      } else if (e.key === '=' || e.key === '+') {
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
  }, [maxFrames, setFrame, setCompare, setIsPlaying]);

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
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 p-1.5 rounded-xl bg-black/65 backdrop-blur-md border border-white/5 shadow-2xl opacity-40 hover:opacity-100 transition-opacity duration-300 z-50">
          <button onClick={() => setZoom(z => Math.min(z + 0.5, 5))} className="p-2 text-white/70 hover:text-white text-xs font-bold font-mono hover:bg-white/10 rounded-lg apple-transition" title="Zoom In">+</button>
          <button onClick={() => setZoom(z => { const nz = Math.max(1, z - 0.5); if (nz === 1) setPan({x:0, y:0}); return nz; })} className="p-2 text-white/70 hover:text-white text-xs font-bold font-mono hover:bg-white/10 rounded-lg apple-transition" title="Zoom Out">-</button>
          <button onClick={() => { setZoom(1); setPan({x:0, y:0}); }} className="px-2 py-1 text-white/70 hover:text-white text-[10px] font-bold hover:bg-white/10 rounded-lg apple-transition" title="Reset view">FIT</button>
          <div className="w-px h-4 bg-white/10 mx-1" />
          <button onClick={() => setShowBoxes(b => !b)} className={`px-2 py-1 text-[10px] font-bold rounded-lg apple-transition ${showBoxes ? 'text-[var(--accent)] bg-[var(--accent)]/10' : 'text-white/70 hover:text-white hover:bg-white/10'}`}>BOXES</button>
          <button onClick={triggerFullscreen} className="p-2 text-white/70 hover:text-white hover:bg-white/10 rounded-lg apple-transition" title="Toggle Fullscreen">
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
      {previewing && <FloatingEmojis />}
      {previewing && (
        <div className="absolute top-3 right-3 flex flex-col items-end gap-1.5 z-50">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/70 backdrop-blur-md text-xs font-bold text-white/95 tabular-nums border border-white/10 shadow-2xl">
            <span className="h-3 w-3 rounded-full border-2 border-white/30 border-t-[var(--accent)] animate-spin" />
            Rendering… {previewSecs}s
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
        <div className="relative z-10" style={aspectStyle}>
          <img src={beforeSrc} alt="Before" className="w-full h-full object-contain pointer-events-none" 
               onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })} draggable={false} />
          <div className="absolute inset-0 pointer-events-none z-30">{renderFaces()}</div>
          
          {/* After Image Overlay with Clip-path */}
          <div className="absolute inset-0 pointer-events-none z-20" 
               style={{ clipPath: compare ? `polygon(${sliderPosition}% 0, 100% 0, 100% 100%, ${sliderPosition}% 100%)` : 'none' }}>
            <img src={afterSrc} alt="After" className="w-full h-full object-contain" draggable={false} />
          </div>
        </div>

      </div>

      {compare && <span className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur text-[11px] font-bold tracking-wider text-white/80 uppercase shadow z-30 pointer-events-none">Before</span>}
      <span className="absolute top-3 right-3 px-2.5 py-1 rounded-full bg-[var(--accent)]/80 backdrop-blur text-[11px] font-bold tracking-wider text-white uppercase shadow z-30 transition-opacity duration-300 pointer-events-none"
            style={{ opacity: compare && sliderPosition > 85 ? 0 : 1 }}>{compare ? 'After' : 'Swapped'}</span>

      {/* Slider Line & Handle (only when compare) */}
      {compare && (
        <div className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.6)] z-40 pointer-events-none" style={{ left: `${sliderPosition}%` }}>
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-[0_4px_10px_rgba(0,0,0,0.4)] transition-transform duration-200 group-hover:scale-110 cursor-ew-resize pointer-events-auto"
               onPointerDown={(e) => { e.stopPropagation(); setIsDraggingSlider(true); handleSliderMove(e.clientX ?? e.touches?.[0]?.clientX); }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="mr-0.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="rotate-180 ml-0.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
          </div>
        </div>
      )}

      {/* HUD control bar overlays */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 p-1.5 rounded-xl bg-black/65 backdrop-blur-md border border-white/5 shadow-2xl opacity-40 group-hover:opacity-100 hover:opacity-100 transition-opacity duration-300 z-50">
        <button onClick={() => setZoom(z => Math.min(z + 0.5, 5))} className="p-2 text-white/70 hover:text-white text-xs font-bold font-mono hover:bg-white/10 rounded-lg apple-transition" title="Zoom In">+</button>
        <button onClick={() => setZoom(z => { const nz = Math.max(1, z - 0.5); if (nz === 1) setPan({x:0, y:0}); return nz; })} className="p-2 text-white/70 hover:text-white text-xs font-bold font-mono hover:bg-white/10 rounded-lg apple-transition" title="Zoom Out">-</button>
        <button onClick={() => { setZoom(1); setPan({x:0, y:0}); }} className="px-2 py-1 text-white/70 hover:text-white text-[10px] font-bold hover:bg-white/10 rounded-lg apple-transition" title="Reset view">FIT</button>
        <div className="w-px h-4 bg-white/10 mx-1" />
        <button onClick={() => setShowBoxes(b => !b)} className={`px-2 py-1 text-[10px] font-bold rounded-lg apple-transition ${showBoxes ? 'text-[var(--accent)] bg-[var(--accent)]/10' : 'text-white/70 hover:text-white hover:bg-white/10'}`}>BOXES</button>
        <button onClick={triggerFullscreen} className="p-2 text-white/70 hover:text-white hover:bg-white/10 rounded-lg apple-transition" title="Toggle Fullscreen">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
        </button>
      </div>
    </div>
  );
}
