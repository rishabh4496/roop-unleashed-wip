import React, { useState } from 'react';
import { postFile, fileUrl } from '../api';
import { Section, Select, Slider, Button } from './ui';

const RESOLUTIONS = ['Original', '3840x', '2560x', '1920x', '1280x', '1024x', '640x'];
const ROTATIONS = ['None', '90° Clockwise', '90° Counter-Clockwise', '180°'];

export default function Extras({ notify }) {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [opts, setOpts] = useState({ resolution: 'Original', rotation: 'None', fps: 30, crop_left: 0, crop_right: 0, crop_top: 0, crop_bottom: 0 });
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const set = (k, v) => setOpts((o) => ({ ...o, [k]: v }));

  const onPick = (e) => { const f = e.target.files[0]; setFile(f); setFileName(f?.name || ''); setResult(null); };

  const apply = async () => {
    if (!file) { notify('Pick a file first', 'error'); return; }
    setBusy(true);
    try {
      const res = await postFile('/api/extras/apply', file, opts);
      setResult(res); notify('Done');
    } catch (e) { notify(e.message, 'error'); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-6">
      <Section>
        <h2 className="text-lg font-bold">Media editor</h2>
        <p className="text-sm text-white/50">Resize, rotate, crop and re-time images or videos.</p>
      </Section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Section title="Input & transform">
          <label className="block cursor-pointer">
            <div className="px-4 py-5 rounded-lg border-2 border-dashed border-white/15 hover:border-[#E94560]/50 text-center text-sm text-white/60">
              📁 {fileName || 'Pick an image or video'}
            </div>
            <input type="file" accept="image/*,video/*" onChange={onPick} className="hidden" />
          </label>
          <Select label="Resize width" value={opts.resolution} onChange={(v) => set('resolution', v)} options={RESOLUTIONS} />
          <Select label="Rotation" value={opts.rotation} onChange={(v) => set('rotation', v)} options={ROTATIONS} />
          <Slider label="Output FPS (video)" min={1} max={120} step={1} value={opts.fps} onChange={(v) => set('fps', v)} />
        </Section>

        <Section title="Crop (%)">
          <Slider label="Left" min={0} max={49} step={1} value={opts.crop_left} onChange={(v) => set('crop_left', v)} />
          <Slider label="Right" min={0} max={49} step={1} value={opts.crop_right} onChange={(v) => set('crop_right', v)} />
          <Slider label="Top" min={0} max={49} step={1} value={opts.crop_top} onChange={(v) => set('crop_top', v)} />
          <Slider label="Bottom" min={0} max={49} step={1} value={opts.crop_bottom} onChange={(v) => set('crop_bottom', v)} />
          <Button variant="primary" onClick={apply} disabled={busy}>{busy ? 'Processing…' : 'Apply'}</Button>
        </Section>
      </div>

      {result?.path && (
        <Section title="Output">
          {result.kind === 'video'
            ? <video src={fileUrl(result.path)} controls className="w-full rounded-lg border border-white/10" />
            : <img src={fileUrl(result.path)} alt="output" className="max-w-full rounded-lg border border-white/10" />}
          <a href={fileUrl(result.path)} download className="inline-block mt-2 text-sm text-[#E94560] underline">⬇ Download</a>
        </Section>
      )}

      <Section>
        <p className="text-xs text-white/40">
          The Gradio "Frame Editor" (per-frame canvas painting, tracked re-swap, MP4/GIF compile) is not yet ported to React.
          Use the legacy Gradio UI for that workflow.
        </p>
      </Section>
    </div>
  );
}
