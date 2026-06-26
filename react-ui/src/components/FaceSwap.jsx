import React, { useEffect, useRef, useState } from 'react';
import { getJSON, postJSON, postFiles, API } from '../api';
import { Section, Select, Slider, Toggle, TextInput, Button, FaceGallery } from './ui';

const num = (v, d) => (v === undefined || v === null || v === '' ? d : Number(v));

export default function FaceSwap({ meta, settings, setSettings, notify }) {
  const [sourceFaces, setSourceFaces] = useState([]);
  const [targetFaces, setTargetFaces] = useState([]);
  const [targets, setTargets] = useState([]);
  const [selSource, setSelSource] = useState(0);
  const [selTarget, setSelTarget] = useState(0);
  const [selTargetFace, setSelTargetFace] = useState(0);
  const [frame, setFrame] = useState(1);
  const [maxFrames, setMaxFrames] = useState(1);
  const [fakePreview, setFakePreview] = useState(false);
  const [previewSrc, setPreviewSrc] = useState('');
  const [uploadingSrc, setUploadingSrc] = useState(false);
  const [uploadingTgt, setUploadingTgt] = useState(false);
  const [progress, setProgress] = useState({ processing: false, progress: 0, desc: '', output: null });
  const [previewing, setPreviewing] = useState(false);
  const pollRef = useRef(null);
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
      setTargets(st.targets || []);
    }).catch(() => {});
  }, []);

  const refreshPreview = async (opts = {}) => {
    if (targets.length === 0) { setPreviewSrc(''); return; }
    // Single-flight: the backend's live_swap shares one (non-thread-safe)
    // ProcessMgr on the GPU. Two overlapping /api/preview calls corrupt/hang
    // TensorRT/CUDA. So never run two at once — queue the latest request and
    // run it once the current one finishes.
    if (previewBusyRef.current) { previewPendingRef.current = opts; return; }
    previewBusyRef.current = true;
    setPreviewing(true);
    const idx = opts.index ?? selTarget;
    const fr = opts.frame ?? frame;
    const fake = opts.fake ?? fakePreview;
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
      });
      setPreviewSrc(res.image || '');
    } catch (e) { notify(e.message, 'error'); }
    finally {
      previewBusyRef.current = false;
      setPreviewing(false);
      if (previewPendingRef.current) {
        const next = previewPendingRef.current;
        previewPendingRef.current = null;
        refreshPreview(next);
      }
    }
  };

  // Auto-refresh preview when the selected target or frame changes.
  useEffect(() => {
    if (targets.length === 0 || progress.processing) return;
    const t = setTimeout(() => refreshPreview(), 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selTarget, frame]);

  // While face-swap preview is on, auto-refresh when the swapped result would
  // change: new source faces, target faces, or any swap/mask parameter.
  const previewKey = JSON.stringify({
    e: p.selected_enhancer, d: p.face_detection_mode, fd: p.max_face_distance,
    br: p.blend_ratio, me: p.mask_engine, ct: p.mask_clip_text, nfa: p.no_face_action,
    vr: p.vr_mode, ar: p.autorotate_faces, smo: p.show_mask_offsets,
    rom: p.restore_original_mouth, ns: p.num_swap_steps, up: p.subsample_upscale,
    r3: p.use_3d_recon, sb: p.use_source_bank, sm: p.swap_model,
  });
  useEffect(() => {
    if (!fakePreview || targets.length === 0 || progress.processing) return;
    const t = setTimeout(() => refreshPreview(), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fakePreview, previewKey, sourceFaces.length, targetFaces.length]);

  // ── source / target file handling ──
  const onAddSource = async (e) => {
    if (!e.target.files.length) return;
    const before = sourceFaces.length;
    const fileEl = e.target;
    setUploadingSrc(true);
    try {
      const res = await postFiles('/api/source/add', fileEl.files);
      setSourceFaces(res.source_faces);
      const added = res.source_faces.length - before;
      if (added > 0) notify(`Loaded ${added} face(s) — ${res.faceset_count} faceset(s) total`);
      else notify('No face detected in the uploaded file(s)', 'error');
    } catch (err) { notify(err.message, 'error'); }
    finally { setUploadingSrc(false); fileEl.value = ''; }
  };

  const onAddTarget = async (e) => {
    if (!e.target.files.length) return;
    const fileEl = e.target;
    setUploadingTgt(true);
    try {
      const res = await postFiles('/api/target/add', fileEl.files);
      setTargets(res.targets);
      setSelTarget(res.selected_target_index || 0);
      const mf = res.targets[res.selected_target_index || 0]?.frames || 1;
      setMaxFrames(mf); setFrame(1);
      refreshPreview({ index: res.selected_target_index || 0, frame: 1 });
      notify(`Added ${res.targets.length} target(s)`);
    } catch (err) { notify(err.message, 'error'); }
    finally { setUploadingTgt(false); fileEl.value = ''; }
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
      set('face_detection_mode', 'Selected face');
      notify(`Added ${res.count} target face(s)`);
    } catch (e) { notify(e.message, 'error'); }
  };

  const setFrameMarker = async (which) => {
    await postJSON('/api/target/set_frame', { which, frame });
    notify(`Set ${which} frame = ${frame}`);
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

  return (
    <div className="space-y-6">
      {/* uploads */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Section title="Source images / facesets">
          <FileDrop accept="image/*,.fsz" multiple label="Add source faces" onChange={onAddSource} busy={uploadingSrc} />
          <FaceGallery title="Input faces" faces={sourceFaces} selected={selSource} onSelect={selectSource} empty="Upload a face image" />
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
          <FileDrop accept="image/*,video/*,.webp" multiple label="Add target media" onChange={onAddTarget} busy={uploadingTgt} />
          {targets.length === 0 ? (
            <div className="h-24 flex items-center justify-center rounded-lg border border-dashed border-white/10 text-xs text-white/30">No targets yet</div>
          ) : (
            <div className="space-y-1.5 max-h-40 overflow-auto">
              {targets.map((t, i) => (
                <button key={i} onClick={() => selectTarget(i)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm border transition-colors ${selTarget === i ? 'bg-[#E94560]/15 border-[#E94560]/50' : 'bg-black/20 border-white/5 hover:border-white/20'}`}>
                  <span className="truncate block">{t.name}</span>
                  <span className="text-xs text-white/40">{t.frames > 1 ? `${t.frames} frames` : 'image'}</span>
                </button>
              ))}
            </div>
          )}
          {targets.length > 0 && <Button size="sm" variant="stop" onClick={async () => { const r = await postJSON('/api/target/clear', {}); setTargets(r.targets); setPreviewSrc(''); }}>Clear targets</Button>}
        </Section>
      </div>

      {/* preview + target faces */}
      <Section title="Preview">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2 space-y-3">
            <div className="relative aspect-video rounded-lg overflow-hidden bg-black/40 border border-white/10 flex items-center justify-center">
              {previewSrc ? <img src={previewSrc} alt="preview" className="max-w-full max-h-full object-contain" />
                : <span className="text-white/30 text-sm">Select a target to preview</span>}
              {previewing && (
                <div className="absolute top-2 right-2 inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur text-xs text-white/80">
                  <span className="h-3 w-3 rounded-full border-2 border-white/30 border-t-[#E94560] animate-spin" />
                  Rendering…
                </div>
              )}
            </div>
            <div className="flex items-center gap-3">
              <Toggle label="Face-swap preview" checked={fakePreview} onChange={(v) => { setFakePreview(v); refreshPreview({ fake: v }); }} />
              <Button size="sm" variant="secondary" onClick={() => refreshPreview()}>🔄 Refresh</Button>
              <Button size="sm" variant="primary" onClick={useFaceFromFrame}>Use face from frame</Button>
            </div>
            {maxFrames > 1 && (
              <>
                <Slider label="Frame" info={`${frame} / ${maxFrames}`} min={1} max={maxFrames} step={1} value={frame}
                  onChange={(v) => setFrame(v)} />
                <div className="flex gap-2">
                  <Button size="sm" variant="secondary" onClick={() => refreshPreview()}>Show frame</Button>
                  <Button size="sm" variant="ghost" onClick={() => setFrameMarker('start')}>⬅ Set start</Button>
                  <Button size="sm" variant="ghost" onClick={() => setFrameMarker('end')}>Set end ➡</Button>
                </div>
              </>
            )}
          </div>
          <div>
            <FaceGallery title="Target faces" faces={targetFaces} selected={selTargetFace} onSelect={setSelTargetFace} empty="Use 'face from frame'" />
            {targetFaces.length > 0 && (
              <div className="mt-2 flex gap-2">
                <Button size="sm" variant="secondary" onClick={async () => { const r = await postJSON('/api/target/remove_face', { index: selTargetFace }); setTargetFaces(r.target_faces); }}>❌ Remove</Button>
              </div>
            )}
          </div>
        </div>
      </Section>

      {/* core controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Section title="Swap">
          <Select label="Swap model" value={p.swap_model} onChange={(v) => set('swap_model', v)} options={meta.swap_models} />
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
              <span>{Math.round((progress.progress || 0) * 100)}%</span>
            </div>
            <div className="h-2 rounded-full bg-black/40 overflow-hidden">
              <div className="h-full bg-[#E94560] transition-all" style={{ width: `${(progress.progress || 0) * 100}%` }} />
            </div>
            {progress.error && <div className="text-xs text-red-400 mt-1">{progress.error}</div>}
          </div>
        </div>
      </div>
    </div>
  );
}

function FileDrop({ label, accept, multiple, onChange, busy }) {
  return (
    <label className={`block ${busy ? 'cursor-wait pointer-events-none' : 'cursor-pointer'}`}>
      <div className={`px-4 py-6 rounded-lg border-2 border-dashed text-center transition-colors ${busy ? 'border-[#E94560]/60 bg-[#E94560]/[0.06]' : 'border-white/15 hover:border-[#E94560]/50 hover:bg-white/[0.02]'}`}>
        {busy ? (
          <span className="inline-flex items-center gap-2 text-sm text-white/80">
            <span className="h-4 w-4 rounded-full border-2 border-white/30 border-t-[#E94560] animate-spin" />
            Uploading & analysing…
          </span>
        ) : (
          <span className="text-sm text-white/60">📁 {label}</span>
        )}
      </div>
      <input type="file" accept={accept} multiple={multiple} onChange={onChange} disabled={busy} className="hidden" />
    </label>
  );
}
