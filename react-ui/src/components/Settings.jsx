import React from 'react';
import { postJSON } from '../api';
import { Section, Select, Slider, Toggle, TextInput, Button } from './ui';

export default function Settings({ meta, settings, setSettings, notify }) {
  const p = settings || {};
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));

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
      <div className="grid grid-cols-1 lg:grid-cols-3 4xl:grid-cols-4 gap-6">
        <Section title="Server">
          <Toggle label="Public server (share)" checked={!!p.server_share} onChange={(v) => set('server_share', v)} />
          <Toggle label="Clear output folder before each run" checked={!!p.clear_output} onChange={(v) => set('clear_output', v)} />
          <TextInput label="Server name" info="blank = local" value={p.server_name} onChange={(v) => set('server_name', v)} placeholder="127.0.0.1" />
          <TextInput label="Server port" info="0 = default" type="number" value={p.server_port} onChange={(v) => set('server_port', v)} />
          <TextInput label="Filename output template" info="{file} {time} {index} {timestamp}" value={p.output_template} onChange={(v) => set('output_template', v)} placeholder="{file}_{timestamp}" />
        </Section>

        <Section title="Appearance">
          <Select 
            label="Interface Theme" 
            value={p.selected_theme || 'Default'} 
            onChange={(v) => set('selected_theme', v)} 
            options={['Default', 'Glass Light', 'Cyberpunk Dark', 'Cyberpunk Light', 'Emerald Dark', 'Emerald Light', 'Nordic Dark', 'Nordic Light']} 
          />
        </Section>

        <Section title="Performance">
          <Select label="Provider" value={p.provider} onChange={(v) => set('provider', v)} options={meta.providers} />
          {p.provider === 'tensorrt' && (
            <Select label="Precision mode (TensorRT)" info="mixed = recommended; fp16 = fastest; fp32 = most accurate" value={p.trt_precision ?? 'mixed'} onChange={(v) => set('trt_precision', v)} options={meta.trt_precisions ?? ['fp32', 'fp16', 'mixed']} />
          )}
          <Toggle label="Force CPU for face analyser" checked={!!p.force_cpu} onChange={(v) => set('force_cpu', v)} />
          <Toggle label="Use default Det-Size" checked={p.default_det_size !== false} onChange={(v) => set('default_det_size', v)} />
          <Slider label="Max threads" info="default 3" min={1} max={32} step={1} value={p.max_threads ?? 3} onChange={(v) => set('max_threads', v)} />
          <Slider label="Max memory (GB)" info="0 = no limit" min={0} max={128} step={1} value={p.memory_limit ?? 0} onChange={(v) => set('memory_limit', v)} />
        </Section>

        <Section title="Output">
          <Select label="Image format" value={p.output_image_format} onChange={(v) => set('output_image_format', v)} options={meta.image_formats} />
          <Select label="Video format" value={p.output_video_format} onChange={(v) => set('output_video_format', v)} options={meta.video_formats} />
          <Select label="Video codec" value={p.output_video_codec} onChange={(v) => set('output_video_codec', v)} options={meta.video_codecs} />
          <Slider label="Video quality (crf)" info="default 14" min={0} max={100} step={1} value={p.video_quality ?? 14} onChange={(v) => set('video_quality', v)} />
          <Toggle label="Use OS temp folder" checked={!!p.use_os_temp_folder} onChange={(v) => set('use_os_temp_folder', v)} />
          <Toggle label="Show video in browser (re-encodes)" checked={!!p.output_show_video} onChange={(v) => set('output_show_video', v)} />
        </Section>
      </div>

      <div className="flex flex-wrap gap-3">
        <Button variant="primary" onClick={apply}>💾 Apply Settings</Button>
        <Button variant="secondary" onClick={cleanTemp}>🧹 Clean loaded media</Button>
      </div>
    </div>
  );
}
