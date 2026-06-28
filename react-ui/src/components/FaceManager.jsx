import React, { useState } from 'react';
import { postJSON, postFiles, postFile, fileUrl, API } from '../api';
import { Section, Slider, Button, FaceGallery } from './ui';

export default function FaceManager({ notify }) {
  const [faces, setFaces] = useState([]);
  const [sel, setSel] = useState(0);
  const [video, setVideo] = useState(null);
  const [frame, setFrame] = useState(1);
  const [maxFrames, setMaxFrames] = useState(1);
  const [built, setBuilt] = useState(null);

  const onAddFiles = async (e) => {
    if (!e.target.files.length) return;
    try {
      const res = await postFiles('/api/facemgr/add', e.target.files);
      setFaces(res.faces);
      if (res.video) { setVideo(res.video); setMaxFrames(res.frames || 1); setFrame(1); }
      notify('Loaded faces');
    } catch (err) { notify(err.message, 'error'); }
    e.target.value = '';
  };

  const onLoadFaceset = async (e) => {
    if (!e.target.files.length) return;
    try { const res = await postFile('/api/facemgr/faceset', e.target.files[0]); setFaces(res.faces); notify('Faceset loaded'); }
    catch (err) { notify(err.message, 'error'); }
    e.target.value = '';
  };

  const cut = async () => { const res = await postJSON('/api/facemgr/cut', { frame }); setFaces(res.faces); notify('Faces cut from frame'); };
  const remove = async () => { const res = await postJSON('/api/facemgr/remove', { index: sel }); setFaces(res.faces); };
  const clear = async () => { await postJSON('/api/facemgr/clear', {}); setFaces([]); setVideo(null); setBuilt(null); };
  const build = async () => {
    try { const res = await postJSON('/api/facemgr/build', {}); setBuilt(res); notify('Faceset file created'); }
    catch (e) { notify(e.message, 'error'); }
  };

  return (
    <div className="space-y-6">
      <Section>
        <h2 className="text-lg font-bold">Create blending facesets</h2>
        <p className="text-sm text-white/50">Add multiple reference images of the same person into a single faceset (.fsz) file for better likeness.</p>
      </Section>

      <div className="grid grid-cols-1 lg:grid-cols-2 3xl:grid-cols-3 gap-6">
        <Section title="Add faces">
          <label className="block cursor-pointer">
            <div className="px-4 py-5 rounded-lg border-2 border-dashed border-white/15 hover:border-[var(--accent)]/50 text-center text-sm text-white/60">📁 Add images / videos</div>
            <input type="file" accept="image/*,video/*" multiple onChange={onAddFiles} className="hidden" />
          </label>
          <label className="block cursor-pointer">
            <div className="px-4 py-3 rounded-lg border border-white/10 hover:border-white/30 text-center text-sm text-white/50">Load existing faceset (.fsz)</div>
            <input type="file" accept=".fsz" onChange={onLoadFaceset} className="hidden" />
          </label>
          {video && (
            <div className="space-y-2">
              <div className="rounded-lg overflow-hidden bg-black/40 border border-white/10">
                <img src={`${API}/api/facemgr/frame?frame=${frame}&t=${frame}`} alt="frame" className="w-full" />
              </div>
              <Slider label="Frame" info={`${frame} / ${maxFrames}`} min={1} max={maxFrames} step={1} value={frame} onChange={setFrame} />
              <Button size="sm" variant="secondary" onClick={cut}>Use faces from this frame</Button>
            </div>
          )}
        </Section>

        <Section title="Faces in faceset">
          <FaceGallery title="" faces={faces} selected={sel} onSelect={setSel} empty="No faces yet" />
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="secondary" onClick={remove}>Remove selected</Button>
            <Button size="sm" variant="primary" onClick={build}>Create / update faceset</Button>
            <Button size="sm" variant="stop" onClick={clear}>Clear all</Button>
          </div>
          {built && (
            <a href={fileUrl(built.path)} download={built.name}
              className="inline-block mt-2 text-sm text-[var(--accent)] underline">⬇ Download {built.name}</a>
          )}
        </Section>
      </div>
    </div>
  );
}
