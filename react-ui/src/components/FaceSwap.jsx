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
  const pollRef = useRef(null);
  const startTimeRef = useRef(null);
  const previewBusyRef = useRef(false);   // a /api/preview call is in flight
  const previewPendingRef = useRef(null); // latest queued request while busy (coalesced)

  // p = the swap parameters, seeded from CFG (settings) and patched locally.
  const p = settings || {};
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));

  // ── initial rehydrate ──
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
        swap_model: p.swap_model,
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
    if (targets.length === 0 || progress.processing) return;
    const t = setTimeout(() => refreshPreview(), 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selTarget, frame, targets.length]);

  // While face-swap preview is on, auto-refresh when the swapped result would
  // change: new source faces, target faces, or any swap/mask parameter.
  const previewKey = JSON.stringify({
    fp: fakePreview,
    e: p.selected_enhancer, d: p.face_detection_mode, fd: p.max_face_distance,
    br: p.blend_ratio, me: p.mask_engine, ct: p.mask_clip_text, nfa: p.no_face_action,
    vr: p.vr_mode, ar: p.autorotate_faces, smo: p.show_mask_offsets,
    rom: p.restore_original_mouth, ns: p.num_swap_steps, up: p.subsample_upscale,
    r3: p.use_3d_recon, sb: p.use_source_bank, sm: p.swap_model,
  });
  useEffect(() => {
    if (targets.length === 0 || progress.processing) return;
    const t = setTimeout(() => refreshPreview(), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewKey, sourceFaces.length, targetFaces.length]);

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

  const setFrameMarker = async (which) => {
    try {
      const res = await postJSON('/api/target/set_frame', { which, frame });
      if (res.targets) setTargets(res.targets);
      notify(`Set ${which} frame = ${frame}`);
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

  return (
    <div className="space-y-6">
      {/* uploads */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
                  className={`group flex items-center gap-2 px-3 py-2 rounded-lg text-sm border transition-colors cursor-pointer ${selTarget === i ? 'bg-[#E94560]/15 border-[#E94560]/50' : 'bg-black/20 border-white/5 hover:border-white/20'}`}
                  onClick={() => selectTarget(i)}>
                  <div className="flex-1 min-w-0">
                    <span className="truncate block">{t.name}</span>
                    <span className="text-xs text-white/40">{t.frames > 1 ? `${t.frames} frames` : 'image'}</span>
                  </div>
                  <button type="button" title="Remove this target"
                    onClick={(e) => { e.stopPropagation(); removeTarget(i); }}
                    className="h-6 w-6 shrink-0 rounded-full bg-black/40 text-white/60 opacity-0 group-hover:opacity-100 hover:bg-[#E94560] hover:text-white transition-opacity flex items-center justify-center">✕</button>
                </div>
              ))}
            </div>
          )}
          {targets.length > 0 && <Button size="sm" variant="stop" onClick={async () => { const r = await postJSON('/api/target/clear', {}); setTargets(r.targets); setTargetFaces([]); setTargetGroups([]); setPreviewSrc(''); }}>Clear targets</Button>}
        </Section>
      </div>

      {/* preview + target faces */}
      <Section title="Preview">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2 space-y-3">
            {compare && previewSrc && rawUrl ? (
              <InteractivePreview 
                beforeSrc={rawUrl} 
                afterSrc={previewSrc} 
                faces={previewFaces}
                splitView={splitView}
              />
            ) : (
              <div className={`relative aspect-video rounded-lg overflow-hidden bg-black/40 border border-white/10 flex items-center justify-center transition-all duration-300 ${previewing ? 'preview-glow' : ''}`}>
                {previewSrc ? <img src={previewSrc} alt="preview" className="max-w-full max-h-full object-contain" />
                  : <span className="text-white/30 text-sm">Select a target to preview</span>}
                {previewing && <FloatingEmojis />}
                {previewing && (
                  <div className="absolute top-2 right-2 flex flex-col items-end gap-1 z-20">
                    <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur text-xs text-white/80 tabular-nums border border-white/10 shadow-lg">
                      <span className="h-3 w-3 rounded-full border-2 border-white/30 border-t-[#E94560] animate-spin" />
                      Rendering… {previewSecs}s
                    </div>
                    {previewSecs >= 12 && (
                      <div className="px-2.5 py-1 rounded-lg bg-black/60 backdrop-blur text-[10px] text-white/50 max-w-[220px] text-right border border-white/10 shadow-lg">
                        First run of a model downloads it &amp; builds the engine — this can take a few minutes.
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            <div className="flex items-center flex-wrap gap-3">
              <Toggle label="✨ Live Swap" checked={fakePreview} onChange={setFakePreview} />
              <Toggle label="🔍 Compare" checked={compare} onChange={setCompare} />
              {compare && <Toggle label="Split View" checked={splitView} onChange={setSplitView} />}
              <Button size="sm" variant="secondary" onClick={() => refreshPreview()}>🔄 Refresh</Button>
              <Button size="sm" variant="primary" onClick={useFaceFromFrame}>Use face from frame</Button>
            </div>
            {maxFrames > 1 && (
              <>
                <div className="pt-2 border-t border-white/5">
                  <Slider label="Frame" info={`${frame} / ${maxFrames}`} min={1} max={maxFrames} step={1} value={frame}
                    onChange={(v) => setFrame(v)} />
                </div>
                <div className="flex flex-wrap items-center gap-2 bg-black/20 p-2 rounded-lg border border-white/5">
                  <Button size="sm" variant="secondary" onClick={() => refreshPreview()} className="!py-1.5 flex-1 sm:flex-none justify-center">Show frame</Button>
                  <div className="w-px h-5 bg-white/10 mx-1 hidden sm:block"></div>
                  
                  <div className="flex items-center gap-1 bg-black/30 rounded px-2 py-1">
                    <span className="text-xs opacity-60">Start:</span>
                    <input type="number" 
                      className="bg-transparent w-16 text-right font-mono text-white outline-none text-sm focus:ring-1 focus:ring-[#E94560] rounded"
                      key={`start-${selTarget}-${targets[selTarget]?.start_frame}`}
                      defaultValue={targets[selTarget]?.start_frame ?? 1}
                      onBlur={(e) => setFrameMarkerVal('start', parseInt(e.target.value, 10))}
                      onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                    />
                  </div>
                  <Button size="sm" variant="ghost" onClick={() => setFrameMarker('start')} title="Set to current slider frame" className="!py-1.5 !px-2">⬅ Set</Button>
                  
                  <div className="w-px h-5 bg-white/10 mx-1 hidden sm:block"></div>
                  
                  <div className="flex items-center gap-1 bg-black/30 rounded px-2 py-1">
                    <span className="text-xs opacity-60">End:</span>
                    <input type="number" 
                      className="bg-transparent w-16 text-right font-mono text-white outline-none text-sm focus:ring-1 focus:ring-[#E94560] rounded"
                      key={`end-${selTarget}-${targets[selTarget]?.end_frame}`}
                      defaultValue={targets[selTarget]?.end_frame ?? maxFrames}
                      onBlur={(e) => setFrameMarkerVal('end', parseInt(e.target.value, 10))}
                      onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                    />
                  </div>
                  <Button size="sm" variant="ghost" onClick={() => setFrameMarker('end')} title="Set to current slider frame" className="!py-1.5 !px-2">Set ➡</Button>
                </div>
              </>
            )}
          </div>
          <div>
            <FaceGallery title="Target faces" faces={targetFaces} selected={selTargetFace} onSelect={setSelTargetFace}
              groups={targetGroups} vertical={true}
              onRemove={async (i) => { const r = await postJSON('/api/target/remove_face', { index: i }); setTargetFaces(r.target_faces); setTargetGroups(r.target_groups || []); if (selTargetFace >= r.target_faces.length) setSelTargetFace(Math.max(0, r.target_faces.length - 1)); }}
              empty="Use 'face from frame'" />
            {targetFaces.length > 0 && (
              <div className="mt-2 space-y-1.5">
                <Button size="sm" variant="primary" className="w-full" onClick={addAngle}>
                  ➕ Add angle to Person {(targetGroups[selTargetFace] ?? 0) + 1}
                </Button>
                <p className="text-[11px] text-white/40 leading-snug">
                  Capture profile / side / upside-down views of the selected person across frames — matching uses the closest angle, so the swap doesn't drop out when the head turns. Each colored group = one person → one source faceset.
                </p>
              </div>
            )}
          </div>
        </div>
      </Section>

      {/* core controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Section title="Swap">
          <Select label="Swap model" info="inswapper 128 · reswapper 256 · hyperswap 256 (downloads on first use)" value={p.swap_model} onChange={(v) => set('swap_model', v)} options={meta.swap_models} />
          <Select label="Face selection" value={p.face_detection_mode} onChange={(v) => set('face_detection_mode', v)} options={meta.face_detection_modes} />
          <Slider label="Swapping steps" info="more = more likeness" min={1} max={5} step={1} value={num(p.num_swap_steps, 1)} onChange={(v) => set('num_swap_steps', v)} />
          <Select label="Post-processing enhancer" value={p.selected_enhancer} onChange={(v) => set('selected_enhancer', v)} options={meta.enhancers} />
          <Slider label="Max face similarity" info="0=identical 1=any" min={0.01} max={1} step={0.01} value={num(p.max_face_distance, 0.85)} onChange={(v) => set('max_face_distance', v)} />
          <Select label="Subsample upscale" value={p.subsample_upscale} onChange={(v) => set('subsample_upscale', v)} options={meta.upscale} />
          <Slider label="Original/Enhanced blend" min={0} max={1} step={0.01} value={num(p.blend_ratio, 0.8)} onChange={(v) => set('blend_ratio', v)} />
        </Section>

        <Section title="Masking">
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

        <Section title="Mouth mask">
          <Slider label="Mouth mask top" min={0} max={2} step={0.01} value={num(p.mouth_top_scale, 1)} onChange={(v) => set('mouth_top_scale', v)} />
          <Slider label="Mouth mask bottom" min={0} max={2} step={0.01} value={num(p.mouth_bottom_scale, 1)} onChange={(v) => set('mouth_bottom_scale', v)} />
          <Slider label="Mouth mask left" min={0} max={2} step={0.01} value={num(p.mouth_left_scale, 1)} onChange={(v) => set('mouth_left_scale', v)} />
          <Slider label="Mouth mask right" min={0} max={2} step={0.01} value={num(p.mouth_right_scale, 1)} onChange={(v) => set('mouth_right_scale', v)} />
          <Slider label="Mouth mask edge blend" min={0} max={200} step={1} value={num(p.mouth_mask_blend, 10)} onChange={(v) => set('mouth_mask_blend', v)} />
          <Toggle label="🧊 3D source pose matching" info="experimental — improves angled swaps" checked={!!p.use_3d_recon} onChange={(v) => set('use_3d_recon', v)} />
          <Toggle label="🎯 Multi-angle source bank" info="auto-pick best source per frame" checked={!!p.use_source_bank} onChange={(v) => set('use_source_bank', v)} />
        </Section>
      </div>

      {/* video + output */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Section title="Video processing">
          <Select label="Video method" value={p.video_swapping_method} onChange={(v) => set('video_swapping_method', v)} options={meta.video_methods} />
          <Select label="On no face detected" value={p.no_face_action} onChange={(v) => set('no_face_action', v)} options={meta.no_face_actions} />
          <Toggle label="VR mode" checked={!!p.vr_mode} onChange={(v) => set('vr_mode', v)} />
          <Toggle label="🎯 Stabilize face (video)" info="Smoothing reduces swap wobble (uses In-Memory method; runs single-thread)" checked={!!p.stabilize_face} onChange={(v) => set('stabilize_face', v)} />
          {p.stabilize_face && (
            <>
              <Select label="Stabilization Method" value={p.stabilize_method || 'one_euro'} onChange={(v) => set('stabilize_method', v)} options={['one_euro', 'ema']} />
              {p.stabilize_method !== 'ema' && (
                <>
                  <Slider label="Smoothing (min cutoff)" info="lower = smoother, more lag" min={0.01} max={0.3} step={0.01} value={num(p.stabilize_min_cutoff, 0.05)} onChange={(v) => set('stabilize_min_cutoff', v)} />
                  <Slider label="Reactivity (beta)" info="higher = less lag on fast motion" min={0} max={0.2} step={0.01} value={num(p.stabilize_beta, 0.02)} onChange={(v) => set('stabilize_beta', v)} />
                </>
              )}
            </>
          )}
          <Toggle label="✨ Reduce enhancer flicker (video)" info="temporally blends the enhanced face to kill GFPGAN/GPEN shimmer (uses In-Memory method; single-thread)" checked={!!p.stabilize_enhancer} onChange={(v) => set('stabilize_enhancer', v)} />
          {p.stabilize_enhancer && (
            <Slider label="Flicker reduction strength" info="higher = smoother (more ghosting risk on motion)" min={0} max={1} step={0.05} value={num(p.stabilize_enhancer_strength, 0.5)} onChange={(v) => set('stabilize_enhancer_strength', v)} />
          )}
        </Section>
        <Section title="Options">
          <Toggle label="Auto rotate horizontal faces" checked={!!p.autorotate_faces} onChange={(v) => set('autorotate_faces', v)} />
          <Toggle label="Skip audio" checked={!!p.skip_audio} onChange={(v) => set('skip_audio', v)} />
          <Toggle label="Keep frames (when extracting)" checked={!!p.keep_frames} onChange={(v) => set('keep_frames', v)} />
          <Toggle label="Wait before creating video" checked={!!p.wait_after_extraction} onChange={(v) => set('wait_after_extraction', v)} />
        </Section>
        <Section title="Output">
          <Select label="Output method" value={p.output_method} onChange={(v) => set('output_method', v)} options={meta.output_methods} />
          {out?.path && (
            <div className="space-y-2">
              <div className="text-xs text-white/50">Latest output</div>
              {out.kind === 'video'
                ? <video src={outUrl} controls className="w-full rounded-lg border border-white/10" />
                : <img src={outUrl} alt="output" className="w-full rounded-lg border border-white/10" />}
              <div className="flex flex-wrap gap-2">
                <a href={outUrl} download
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-[#E94560] hover:bg-[#d83a52] text-white transition-colors">⬇ Download</a>
                <Button size="sm" variant="secondary" onClick={revealOutput}>📂 Open folder</Button>
              </div>
            </div>
          )}
        </Section>
      </div>

      {/* run bar */}
      <div className="sticky bottom-0 -mx-2 px-2 py-3">
        <div className="rounded-xl bg-[#16213E]/80 backdrop-blur-md border border-white/10 p-4 flex items-center gap-4">
          {progress.processing ? (
            <Button variant="stop" size="lg" onClick={stop}>⏹ Stop</Button>
          ) : (
            <Button variant="primary" size="lg" onClick={start} disabled={targets.length === 0 || sourceFaces.length === 0}>▶ Start Swapping</Button>
          )}
          <div className="flex-1">
            <div className="flex justify-between text-xs text-white/50 mb-1">
              <span>{progress.desc || (progress.processing ? 'Processing…' : 'Idle')}</span>
              <span className="flex items-center gap-2 tabular-nums">
                {progress.processing && (
                  <span className="text-white/40">⏱ {fmtTime(elapsedMs)}{etaMs > 0 ? ` · ETA ${fmtTime(etaMs)}` : ''}</span>
                )}
                <span>{Math.round(prog * 100)}%</span>
              </span>
            </div>
            <div className="h-2.5 rounded-full bg-black/40 overflow-hidden relative shadow-inner">
              <div className={`absolute top-0 bottom-0 left-0 bg-[#E94560] transition-all duration-300 ${progress.processing ? 'progress-bar-animated shadow-[0_0_10px_rgba(233,69,96,0.6)]' : ''}`} style={{ width: `${(progress.progress || 0) * 100}%` }} />
            </div>
            {progress.error && <div className="text-xs text-red-400 mt-1">{progress.error}</div>}
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
      <div className={`px-4 py-8 rounded-xl border-2 border-dashed text-center transition-all duration-300 ${busy ? 'border-[#E94560]/60 bg-[#E94560]/[0.06]' : drag ? 'border-[#E94560] bg-[#E94560]/[0.1] scale-[0.98]' : 'border-white/15 hover:border-[#E94560]/50 hover:bg-white/[0.02]'}`}>
        {busy ? (
          <span className="inline-flex items-center gap-2 text-sm font-medium text-white/80">
            <span className="h-4 w-4 rounded-full border-2 border-white/30 border-t-[#E94560] animate-spin" />
            Uploading & analysing…
          </span>
        ) : (
          <div className="flex flex-col items-center justify-center gap-3 pointer-events-none">
            <div className={`p-3.5 rounded-full bg-black/20 ${drag ? 'scale-110 text-[#E94560] shadow-[0_0_15px_rgba(233,69,96,0.3)]' : 'text-white/40 group-hover:text-white/70 group-hover:bg-white/5'} transition-all duration-300`}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <div>
              <span className={`text-sm font-semibold tracking-wide ${drag ? 'text-[#E94560]' : 'text-white/80'}`}>{drag ? 'Drop files now' : label}</span>
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

function InteractivePreview({ beforeSrc, afterSrc, faces = [], splitView = false }) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef(null);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });

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

  const renderFaces = () => {
    if (!faces.length || !imgDim) return null;
    return faces.map((bbox, i) => {
      const [sx, sy, ex, ey] = bbox;
      const left = (sx / imgDim.w) * 100;
      const top = (sy / imgDim.h) * 100;
      const width = ((ex - sx) / imgDim.w) * 100;
      const height = ((ey - sy) / imgDim.h) * 100;
      return (
        <div key={i} className="absolute border-2 border-[#E94560] shadow-[0_0_10px_rgba(233,69,96,0.5)] z-20 pointer-events-none"
             style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}>
          <span className="absolute -top-6 left-0 bg-[#E94560] text-white text-[10px] font-bold px-1.5 py-0.5 rounded whitespace-nowrap">
            Person {i}
          </span>
        </div>
      );
    });
  };

  const transformStyle = { transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`, transformOrigin: 'center' };
  const aspectStyle = { aspectRatio: imgDim ? `${imgDim.w}/${imgDim.h}` : '1', maxHeight: '100%', maxWidth: '100%', display: 'flex' };

  if (splitView) {
    return (
      <div className="relative w-full aspect-video rounded-lg overflow-hidden bg-black/40 border border-white/10 group shadow-lg"
           onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} style={{ touchAction: 'none' }}>
        <div className="flex w-full h-full transition-transform duration-75" style={transformStyle}>
          <div className="flex-1 relative border-r border-white/20 flex items-center justify-center overflow-hidden bg-black/50">
            <div className="relative" style={aspectStyle}>
              <img src={beforeSrc} alt="Before" className="w-full h-full object-contain pointer-events-none" 
                   onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })} />
              <div className="absolute inset-0 pointer-events-none">{renderFaces()}</div>
            </div>
            <span className="absolute top-2 left-2 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur text-[11px] font-bold text-white/80 uppercase">Before</span>
          </div>
          <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-black/50">
            <div className="relative" style={aspectStyle}>
              <img src={afterSrc} alt="After" className="w-full h-full object-contain pointer-events-none" />
            </div>
            <span className="absolute top-2 left-2 px-2.5 py-1 rounded-full bg-[#E94560]/80 backdrop-blur text-[11px] font-bold text-white uppercase">After</span>
          </div>
        </div>
        {zoom > 1 && (
          <button onClick={() => { setZoom(1); setPan({x:0, y:0}); }} className="absolute bottom-4 right-4 bg-black/80 px-3 py-1.5 rounded-lg text-xs font-medium hover:bg-white/20 transition-colors z-50 shadow-xl border border-white/10 pointer-events-auto">Reset Zoom</button>
        )}
      </div>
    );
  }

  return (
    <div className="relative w-full aspect-video rounded-lg overflow-hidden bg-black/40 border border-white/10 select-none group shadow-lg"
         ref={containerRef} onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} style={{ touchAction: 'none' }}>
      
      <div className="absolute inset-0 transition-transform duration-75 flex items-center justify-center" style={transformStyle}>
        
        {/* Before Image & Bounding Boxes Wrapper */}
        <div className="relative z-10" style={aspectStyle}>
          <img src={beforeSrc} alt="Before" className="w-full h-full object-contain pointer-events-none" 
               onLoad={(e) => setImgDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })} draggable={false} />
          <div className="absolute inset-0 pointer-events-none">{renderFaces()}</div>
          
          {/* After Image Overlay with Clip-path */}
          <div className="absolute inset-0 pointer-events-none z-20" 
               style={{ clipPath: `polygon(${sliderPosition}% 0, 100% 0, 100% 100%, ${sliderPosition}% 100%)` }}>
            <img src={afterSrc} alt="After" className="w-full h-full object-contain" draggable={false} />
          </div>
        </div>

      </div>

      <span className="absolute top-2 left-2 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur text-[11px] font-bold tracking-wider text-white/80 uppercase shadow z-30 pointer-events-none">Before</span>
      <span className="absolute top-2 right-2 px-2.5 py-1 rounded-full bg-[#E94560]/80 backdrop-blur text-[11px] font-bold tracking-wider text-white uppercase shadow z-30 transition-opacity duration-300 pointer-events-none"
            style={{ opacity: sliderPosition > 85 ? 0 : 1 }}>After</span>

      {/* Slider Line & Handle */}
      <div className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.6)] z-40 pointer-events-none" style={{ left: `${sliderPosition}%` }}>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-[0_4px_10px_rgba(0,0,0,0.4)] transition-transform duration-200 group-hover:scale-110 cursor-ew-resize pointer-events-auto"
             onPointerDown={(e) => { e.stopPropagation(); setIsDraggingSlider(true); handleSliderMove(e.clientX ?? e.touches?.[0]?.clientX); }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#E94560" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="mr-0.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#E94560" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="rotate-180 ml-0.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
        </div>
      </div>

      {zoom > 1 && (
        <button onClick={(e) => { e.stopPropagation(); setZoom(1); setPan({x:0, y:0}); }} className="absolute bottom-4 right-4 bg-black/80 px-3 py-1.5 rounded-lg text-xs font-medium hover:bg-white/20 transition-colors z-50 shadow-xl border border-white/10 pointer-events-auto cursor-pointer">Reset Zoom</button>
      )}
    </div>
  );
}
