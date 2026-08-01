import React, { useState } from 'react';
import { postJSON } from '../api';
import { Section, Select, Slider, Toggle, TextInput } from './ui';
import ThemeGallery from './ThemeGallery';
import { Icon } from '../icons';

// A Section that participates in the settings search: with a query active it
// keeps only the controls whose label/info match (or the whole section when the
// section title itself matches), and hides itself when nothing matches. With no
// query it behaves exactly like a plain Section.
function FilterSection({ title, query, children, ...rest }) {
  const q = (query || '').trim().toLowerCase();
  if (!q) return <Section title={title} {...rest}>{children}</Section>;
  const titleMatch = title.toLowerCase().includes(q);
  const kids = React.Children.toArray(children).filter((c) => {
    if (titleMatch) return true;
    const pr = (c && c.props) || {};
    return [pr.label, pr.info].some((v) => typeof v === 'string' && v.toLowerCase().includes(q));
  });
  if (kids.length === 0) return null;
  return <Section title={title} {...rest}>{kids}</Section>;
}

export default function Settings({ meta, settings, setSettings, notify }) {
  const p = settings || {};
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));
  const [query, setQuery] = useState('');

  const apply = async () => {
    try {
      await postJSON('/api/settings', p);
      notify('Settings applied');
    } catch (e) { notify(e.message, 'error'); }
  };

  const cleanTemp = async () => {
    try { await postJSON('/api/source/clear', {}); await postJSON('/api/target/clear', {}); notify('Cleared loaded media'); }
    catch (e) { notify(e.message, 'error'); }
  };

  return (
    <div className="space-y-6">
      <div className="relative max-w-md">
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30 pointer-events-none"><Icon.search size={14} /></span>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search settings… (e.g. thread, nvenc, codec)"
          aria-label="Search settings"
          className="w-full pl-9 pr-3 py-2.5 rounded-xl glass-input text-white text-compact focus:outline-none placeholder:text-white/25"
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 4xl:grid-cols-4 gap-6">
        <FilterSection title="Server" query={query}>
          <Toggle label="Public server (share)" checked={!!p.server_share} onChange={(v) => set('server_share', v)} />
          <Toggle label="Clear output folder before each run" checked={!!p.clear_output} onChange={(v) => set('clear_output', v)} />
          <TextInput label="Server name" info="blank = local" value={p.server_name} onChange={(v) => set('server_name', v)} placeholder="127.0.0.1" />
          <TextInput label="Server port" info="0 = default" type="number" value={p.server_port} onChange={(v) => set('server_port', v)} />
          <TextInput label="Filename output template" info="{file} {time} {date} {i} {timestamp}" value={p.output_template} onChange={(v) => set('output_template', v)} placeholder="{file}_{timestamp}" />
          <TextInput label="Faceset library folder" info="Where saved facesets live. Blank = app/facesets. Point at a cloud folder (OneDrive/Dropbox/Google Drive) to sync facesets across devices." value={p.faceset_library_path || ''} onChange={(v) => set('faceset_library_path', v)} placeholder="e.g. C:\Users\you\OneDrive\roop-facesets" />
        </FilterSection>

        {/* "Theme" belongs in the title: the search matches on a child's label
            or info, and this section's children are a bare heading and the
            gallery — neither carries one, so under the old title searching for
            the most obvious word for this control hid it. */}
        <FilterSection title="Appearance & Theme" query={query}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-white/70">Interface Theme</span>
            <span className="text-micro text-[var(--accent)] font-bold">{p.selected_theme || 'Default'}</span>
          </div>
          <ThemeGallery value={p.selected_theme} onChange={(v) => { set('selected_theme', v); postJSON('/api/settings', { selected_theme: v }).catch(() => {}); }} />
        </FilterSection>

        <FilterSection title="Performance" query={query}>
          <Select label="Provider" info="Inference sessions are built at startup — provider and precision changes take effect after restarting the app." value={p.provider} onChange={(v) => set('provider', v)} options={meta.providers} />
          {p.provider === 'tensorrt' && (
            <Select label="Precision mode (TensorRT)" info="mixed = recommended; fp16 = fastest; fp32 = most accurate. Applies after app restart." value={p.trt_precision ?? 'mixed'} onChange={(v) => set('trt_precision', v)} options={meta.trt_precisions ?? ['fp32', 'fp16', 'mixed']} />
          )}
          <Toggle label="Force CPU for face analyser" checked={!!p.force_cpu} onChange={(v) => set('force_cpu', v)} />

          <Slider 
            label="Face detection threshold" 
            min={0.10} 
            max={0.90} 
            step={0.05} 
            value={p.face_detector_threshold ?? 0.60} 
            onChange={(v) => set('face_detector_threshold', v)} 
          />
          <Slider 
            label="Overlap NMS threshold" 
            min={0.10} 
            max={0.90} 
            step={0.05} 
            value={p.face_detector_nms ?? 0.40} 
            onChange={(v) => set('face_detector_nms', v)} 
          />
          <Slider label="Max threads" info="default 3" min={1} max={32} step={1} value={p.max_threads ?? 3} onChange={(v) => set('max_threads', v)} />
          <Slider label="Max memory (GB)" info="0 = no limit" min={0} max={128} step={1} value={p.memory_limit ?? 0} onChange={(v) => set('memory_limit', v)} />
        </FilterSection>

        <FilterSection title="Advanced performance (restart to apply)" query={query}>
          <p className="text-xs text-white/40 -mt-2">These override the launcher env and the VRAM auto-tuner. Leave on "auto" unless you know what you're tuning. Changes take effect after restarting the app.</p>
          <Select label="Swapper TRT pool" info="ROOP_TRT_POOL — TensorRT contexts for the SWAPPER only; it does not affect face detection. 'auto' selects by VRAM: <7GB = 0 (disabled), 7-11.5GB = 2, 11.5-15.5GB = 4, 15.5GB+ = 8. Lower this first if you need to free VRAM for another pool." value={p.perf_trt_pool || 'auto'} onChange={(v) => set('perf_trt_pool', v)} options={meta.pool_sizes || ['auto', '1', '2', '3', '4', '5', '6', '7', '8']} />
          <Select label="Detect/Mask pool" info="ROOP_DETMASK_POOL — TensorRT contexts for face detection and masking, and the width of 'Analyzing faces'. LOWERING it slows that stage close to proportionally. DO NOT just raise it to match Max threads. Each instance carries its own model set plus a copy of the detector (retinaface_r50 is ~104MB), and on a 12GB card 8 does not fit alongside the swapper pool: measured, it ran out of VRAM and thrashed from 11.8 fps down to 0.5 and still falling, at 95% VRAM. The auto tier (12GB = 4) is chosen to leave that headroom. If you raise it, go one step at a time and watch VRAM — 5 or 6 may fit, 8 does not. Raising it only helps when the stage is DETECTION-bound. Check STAGE TIMING (ROOP_PROFILE=1): if track_decode per frame exceeds track_detect divided by this pool size, the stage is waiting on the video decoder instead and more instances buy nothing but VRAM." value={p.perf_detmask_pool || 'auto'} onChange={(v) => set('perf_detmask_pool', v)} options={meta.pool_sizes || ['auto', '1', '2', '3', '4', '5', '6', '7', '8']} />
          <Select label="Expression pool" info="ROOP_EXPR_POOL — TensorRT contexts for the LivePortrait expression restorer, the most expensive per-face stage there is (a full re-render: 5 models, one of them a 421MB generator). Only allocated when expression restore is actually on. 'auto' is VRAM-tiered: below 11.5GB = 0 (single context), above = 2, which was measured +28% on the stage. Raise to 3 only if STAGE TIMING shows 'expression' total/wall-clock exceeding the slot count — i.e. threads queueing for a slot. Each slot is ~537MB of weights, the largest of any pool here." value={p.perf_expr_pool || 'auto'} onChange={(v) => set('perf_expr_pool', v)} options={meta.pool_sizes || ['auto', '1', '2', '3', '4', '5', '6', '7', '8']} />
          <Select label="Encoder preset" info="ROOP_ENCODER_PRESET — Encoding speed preset. 'auto' selects: 'faster' for CPU encoders (libx264/libx265), and 'p5' (VBR HQ) for NVENC GPU encoders." value={p.perf_encoder_preset || 'auto'} onChange={(v) => set('perf_encoder_preset', v)} options={meta.encoder_presets || ['auto', 'faster', 'fast', 'medium']} />
          <Select label="GPU video decode (NVDEC)" info="ROOP_NVDEC — Decode the source video on the GPU's dedicated NVDEC engine (ffmpeg -hwaccel cuda) instead of CPU cv2, speeding up the analysis pre-pass and the swap pass decode. 'auto'/'on' = enabled behind a per-file probe with automatic CPU fallback; 'off' = always CPU." value={p.perf_nvdec || 'auto'} onChange={(v) => set('perf_nvdec', v)} options={meta.tristate || ['auto', 'on', 'off']} />
          <Select label="Batched swap" info="ROOP_BATCH_SWAP — Groups face tiles to process them in a single batched GPU pass. 'auto' defaults to 'on'." value={p.perf_batch_swap || 'auto'} onChange={(v) => set('perf_batch_swap', v)} options={meta.tristate || ['auto', 'on', 'off']} />
          <Select label="Stage profiling (terminal)" info="ROOP_PROFILE — Prints a detailed performance execution breakdown in the terminal window. 'auto' defaults to 'on'." value={p.perf_profile || 'auto'} onChange={(v) => set('perf_profile', v)} options={meta.tristate || ['auto', 'on', 'off']} />
        </FilterSection>

        <FilterSection title="Output" query={query}>
          <Select label="Image format" value={p.output_image_format} onChange={(v) => set('output_image_format', v)} options={meta.image_formats} />
          <Select label="Video format" value={p.output_video_format} onChange={(v) => set('output_video_format', v)} options={meta.video_formats} />
          <Select label="Video codec" value={p.output_video_codec} onChange={(v) => set('output_video_codec', v)} options={meta.video_codecs} />
          {(() => {
            // NVENC codecs use -cq (a different scale than libx264/265 -crf); the
            // same number produces a much bigger, near-lossless file on GPU encoders.
            // Make the label/help reflect which rate control is actually in effect.
            const codec = p.output_video_codec || '';
            const isNvenc = /_nvenc$/.test(codec);
            // Encoder-accepted range: x264/x265 and NVENC -cq stop at 51, the
            // VP9/AV1 family at 63. The slider used to go to 100; past the limit
            // libx265 (the default) and both NVENC encoders fail the render
            // outright, while libx264 quietly clamps. Measured, not assumed.
            const qMax = /libvpx|vp9|aom|av1/i.test(codec) ? 63 : 51;
            const qLabel = isNvenc ? 'Video quality (cq)' : 'Video quality (crf)';
            const qInfo = isNvenc
              ? `NVENC uses -cq, not CRF: LOWER = bigger/near-lossless (14 is huge). Try ~23 for a normal-size file. Max ${qMax}.`
              : `default 14 (libx264/265 CRF: lower = bigger). Try ~23 for a normal-size file. Max ${qMax}.`;
            return (
              <Slider label={qLabel} info={qInfo} min={0} max={qMax} step={1} value={Math.min(p.video_quality ?? 14, qMax)} onChange={(v) => set('video_quality', v)} />
            );
          })()}
          <Toggle label="Use OS temp folder" checked={!!p.use_os_temp_folder} onChange={(v) => set('use_os_temp_folder', v)} />
          <Toggle label="Show video in browser (re-encodes)" checked={!!p.output_show_video} onChange={(v) => set('output_show_video', v)} />
        </FilterSection>
      </div>

      {/* Floating Sticky Action Dock — the ONLY place these two actions live.
          The dock follows the page, so a second static copy at the bottom (and
          a third in the header) just gave the same button three times. */}
      <div className="sticky bottom-6 right-6 z-40 flex justify-end pointer-events-none">
        <div className="pointer-events-auto flex items-center gap-2 p-2 rounded-2xl bg-black/80 backdrop-blur-xl border border-white/15 shadow-2xl">
          <button
            type="button"
            onClick={apply}
            className="px-4 py-2 rounded-xl bg-[var(--accent)] hover:brightness-110 text-white font-bold text-xs shadow-lg transition-transform active:scale-95 flex items-center gap-1.5"
          >
            Apply Settings
          </button>
          <button
            type="button"
            onClick={cleanTemp}
            className="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white/80 font-bold text-xs transition-colors"
            title="Clean loaded media"
          >
            Clean
          </button>
        </div>
      </div>
    </div>
  );
}
