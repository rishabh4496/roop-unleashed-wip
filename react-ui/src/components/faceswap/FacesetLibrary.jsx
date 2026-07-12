import React, { useCallback, useEffect, useRef, useState } from 'react';
import { getJSON, postJSON, postFile, fileUrl } from '../../api';
import { Button } from '../ui';

// Persistent, named .fsz facesets on disk. Save the selected source faceset here
// once and reload it any time without re-uploading. Point the library folder
// (Settings → "Faceset library folder") at OneDrive/Dropbox/Google Drive to sync
// facesets across devices.
export default function FacesetLibrary({ canSave, onLoaded, notify }) {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState([]);
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState(null); // filename being renamed
  const [renameVal, setRenameVal] = useState('');
  const [filter, setFilter] = useState('');
  const importRef = useRef(null);

  const q = filter.trim().toLowerCase();
  const shown = q ? entries.filter((e) => e.name.toLowerCase().includes(q)) : entries;

  const refresh = useCallback(async () => {
    try {
      const r = await getJSON('/api/faceset/library');
      setEntries(r.entries || []);
    } catch (e) { notify?.(e.message, 'error'); }
  }, [notify]);

  useEffect(() => { if (open) refresh(); }, [open, refresh]);

  const saveCurrent = async () => {
    const name = window.prompt('Name this faceset', '');
    if (name == null) return;
    setBusy(true);
    try {
      const r = await postJSON('/api/faceset/library/save', { name });
      setEntries(r.entries || []);
      notify?.(`Saved “${r.saved}” to library`);
    } catch (e) { notify?.(e.message, 'error'); } finally { setBusy(false); }
  };

  const load = async (filename) => {
    setBusy(true);
    try {
      const r = await postJSON('/api/faceset/library/load', { filename });
      onLoaded?.(r);
      notify?.('Faceset loaded into source faces');
    } catch (e) { notify?.(e.message, 'error'); } finally { setBusy(false); }
  };

  const del = async (filename) => {
    if (!window.confirm(`Delete “${filename.replace(/\.fsz$/i, '')}” from the library? This removes the file on disk.`)) return;
    try {
      const r = await postJSON('/api/faceset/library/delete', { filename });
      setEntries(r.entries || []);
    } catch (e) { notify?.(e.message, 'error'); }
  };

  const beginRename = (entry) => { setRenaming(entry.filename); setRenameVal(entry.name); };
  const commitRename = async (filename) => {
    try {
      const r = await postJSON('/api/faceset/library/rename', { filename, name: renameVal });
      setEntries(r.entries || []);
    } catch (e) { notify?.(e.message, 'error'); } finally { setRenaming(null); }
  };

  const onImport = async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = '';
    setBusy(true);
    try {
      let r;
      for (const f of files) r = await postFile('/api/faceset/library/import', f);
      if (r) setEntries(r.entries || []);
      if (files.length) notify?.(`Imported ${files.length} faceset(s)`);
    } catch (err) { notify?.(err.message, 'error'); } finally { setBusy(false); }
  };

  const openFolder = async () => {
    try { await postJSON('/api/faceset/library/open', {}); } catch (e) { notify?.(e.message, 'error'); }
  };

  return (
    <div className="rounded-xl bg-black/45 border border-white/5 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 text-left select-none hover:bg-white/[0.02] transition-colors"
      >
        <span className="font-semibold text-[10px] uppercase tracking-[0.14em] text-white/50">
          📚 Faceset library
        </span>
        <span className="flex items-center gap-2">
          {entries.length > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-white/[0.04] text-[10px] text-white/50 border border-white/5">
              {entries.length}
            </span>
          )}
          <span className="text-white/40 text-xs">{open ? '▲' : '▼'}</span>
        </span>
      </button>

      {open && (
        <div className="px-3.5 pb-3.5 space-y-3">
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="primary" disabled={!canSave || busy} onClick={saveCurrent}>
              💾 Save selected face
            </Button>
            <Button size="sm" variant="secondary" onClick={() => importRef.current?.click()}>⬆ Import .fsz</Button>
            <Button size="sm" variant="secondary" onClick={openFolder}>📂 Open folder</Button>
            <Button size="sm" variant="secondary" onClick={refresh}>↻</Button>
            <input ref={importRef} type="file" accept=".fsz" multiple className="hidden" onChange={onImport} />
          </div>

          {entries.length === 0 ? (
            <p className="text-[11px] text-white/35 leading-relaxed">
              No saved facesets yet. Select a source face above and hit
              <span className="text-white/55"> Save selected face</span> to keep it here — it survives
              restarts, so you never re-upload. Set the folder to a cloud drive in Settings to sync across devices.
            </p>
          ) : (
            <>
            {entries.length > 6 && (
              <input
                value={filter}
                onChange={(ev) => setFilter(ev.target.value)}
                placeholder={`Search ${entries.length} facesets…`}
                className="w-full bg-black/40 border border-white/10 focus:border-[var(--accent)]/40 rounded-lg px-2.5 py-1.5 text-[11px] text-white/80 placeholder-white/25 outline-none"
              />
            )}
            <div className="flex flex-col gap-1 max-h-64 overflow-y-auto pr-0.5 -mr-1 [scrollbar-width:thin]">
              {shown.length === 0 ? (
                <p className="text-[11px] text-white/30 py-2 text-center">No match for “{filter}”.</p>
              ) : shown.map((e) => (
                <div
                  key={e.filename}
                  className="group flex items-center gap-2.5 rounded-lg bg-white/[0.02] border border-white/5 hover:border-white/15 hover:bg-white/[0.04] transition-colors pr-2"
                >
                  <button
                    type="button"
                    onClick={() => load(e.filename)}
                    disabled={busy}
                    title="Load into source faces"
                    className="flex items-center gap-2.5 flex-1 min-w-0 py-1.5 pl-1.5 text-left"
                  >
                    <span className="shrink-0 w-9 h-9 rounded-md overflow-hidden bg-black/40 border border-white/10">
                      {e.thumb
                        ? <img src={e.thumb} alt={e.name} className="w-full h-full object-cover" draggable={false} />
                        : <span className="flex items-center justify-center w-full h-full text-white/20 text-sm">🧑</span>}
                    </span>
                    {renaming === e.filename ? (
                      <input
                        autoFocus
                        value={renameVal}
                        onClick={(ev) => ev.stopPropagation()}
                        onChange={(ev) => setRenameVal(ev.target.value)}
                        onBlur={() => commitRename(e.filename)}
                        onKeyDown={(ev) => { if (ev.key === 'Enter') commitRename(e.filename); if (ev.key === 'Escape') setRenaming(null); }}
                        className="flex-1 min-w-0 bg-black/50 border border-[var(--accent)]/40 rounded px-1.5 py-0.5 text-[11px] text-white/90 outline-none"
                      />
                    ) : (
                      <span className="flex-1 min-w-0 truncate text-[11px] text-white/75" title={e.name}>
                        {e.name}
                        {e.faces > 1 && <span className="text-white/35"> · {e.faces} faces</span>}
                      </span>
                    )}
                  </button>

                  <div className="flex items-center gap-1.5 text-[10px] text-white/35 shrink-0 opacity-60 group-hover:opacity-100 transition-opacity">
                    <button type="button" className="hover:text-white/80 transition-colors" title="Rename" onClick={() => beginRename(e)}>✏️</button>
                    <a className="hover:text-white/80 transition-colors" title="Export .fsz" href={fileUrl(e.path)} download={e.filename}>⬇</a>
                    <button type="button" className="hover:text-[var(--accent)] transition-colors" title="Delete" onClick={() => del(e.filename)}>🗑</button>
                  </div>
                </div>
              ))}
            </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
