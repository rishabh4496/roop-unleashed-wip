import React, { useEffect, useState, useRef } from 'react';
import { getJSON, postJSON, postFiles, fileUrl } from '../api';
import { Button, Card } from './ui';

export default function Gallery({ notify }) {
  const [files, setFiles] = useState([]);
  const [outputPath, setOutputPath] = useState('');
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('all'); // 'all', 'video', 'image'
  const [sortBy, setSortBy] = useState('new'); // 'new' | 'old' | 'big' | 'name'
  const [busyFile, setBusyFile] = useState(''); // tracking loading reuse actions

  const fetchOutputs = async () => {
    setLoading(true);
    try {
      const res = await getJSON('/api/output');
      setFiles(res.files || []);
      setOutputPath(res.output_path || '');
    } catch (e) {
      notify(e.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOutputs();
  }, []);

  const revealFolder = async () => {
    try {
      await postJSON('/api/reveal', {});
      notify('Opened output directory');
    } catch (e) {
      notify(e.message, 'error');
    }
  };

  const revealFile = async (name) => {
    try {
      const fullPath = `${outputPath}/${name}`;
      await postJSON('/api/reveal', { path: fullPath });
      notify(`Revealed ${name}`);
    } catch (e) {
      notify(e.message, 'error');
    }
  };

  const deleteFile = async (name) => {
    if (!window.confirm(`Are you sure you want to delete ${name}?`)) return;
    try {
      await postJSON('/api/output/delete', { name });
      notify(`Deleted ${name}`);
      setFiles((prev) => prev.filter((f) => f.name !== name));
    } catch (e) {
      notify(e.message, 'error');
    }
  };

  const reuseAsTarget = async (name) => {
    setBusyFile(name);
    notify(`Preparing to load ${name} as target...`, 'info');
    try {
      const fullPath = `${outputPath}/${name}`;
      const url = fileUrl(fullPath);
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], name, { type: blob.type });
      
      await postFiles('/api/target/add', [file]);
      notify(`Loaded ${name} into face swap targets queue!`);
    } catch (e) {
      notify(`Failed to reuse as target: ${e.message}`, 'error');
    } finally {
      setBusyFile('');
    }
  };

  const reuseAsSource = async (name) => {
    setBusyFile(name);
    notify(`Extracting faces from ${name}...`, 'info');
    try {
      const fullPath = `${outputPath}/${name}`;
      const url = fileUrl(fullPath);
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], name, { type: blob.type });
      
      const result = await postFiles('/api/source/add', [file]);
      if (result.source_faces && result.source_faces.length > 0) {
        notify(`Successfully extracted ${result.source_faces.length} faces!`);
      } else {
        notify('No faces detected in this output file.', 'error');
      }
    } catch (e) {
      notify(`Failed to reuse as source: ${e.message}`, 'error');
    } finally {
      setBusyFile('');
    }
  };

  // Filter & Search files
  const filteredFiles = files.filter((f) => {
    const matchesSearch = f.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter =
      filterType === 'all' ||
      (filterType === 'video' && f.kind === 'video') ||
      (filterType === 'image' && f.kind === 'image');
    return matchesSearch && matchesFilter;
  }).sort((a, b) => {
    if (sortBy === 'old') return (a.mtime || 0) - (b.mtime || 0);
    if (sortBy === 'big') return (b.size || 0) - (a.size || 0);
    if (sortBy === 'name') return a.name.localeCompare(b.name);
    return (b.mtime || 0) - (a.mtime || 0); // 'new'
  });

  const totalSize = filteredFiles.reduce((acc, f) => acc + (f.size || 0), 0);

  const getFormatDate = (timestamp) => {
    try {
      return new Date(timestamp * 1000).toLocaleString();
    } catch {
      return '';
    }
  };

  const fmtSize = (bytes) => {
    if (!bytes && bytes !== 0) return '';
    if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 ** 3)).toFixed(2)} GB`;
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 ** 2)).toFixed(1)} MB`;
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  };

  return (
    <div className="space-y-6">
      {/* Upper header action bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-black text-white/90">📂 OUTPUTS GALLERY</h2>
          <p className="text-sm text-white/50">Browse, manage, and reuse files from the output folder.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={fetchOutputs} disabled={loading}>
            🔄 Refresh
          </Button>
          <Button variant="primary" onClick={revealFolder}>
            📁 Reveal Folder
          </Button>
        </div>
      </div>

      {/* Filter and Search capsule */}
      <Card className="p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="w-full md:w-auto flex-1 max-w-md">
          <input
            type="text"
            placeholder="Search outputs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-3 py-2 rounded-xl glass-input text-white text-sm focus:outline-none"
          />
        </div>
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5 bg-black/20 p-1 rounded-xl border border-white/5">
            {['all', 'video', 'image'].map((t) => (
              <button
                key={t}
                onClick={() => setFilterType(t)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${
                  filterType === t
                    ? 'bg-[#E94560] text-white shadow-[0_2px_8px_rgba(233,69,96,0.3)]'
                    : 'text-white/60 hover:text-white'
                }`}
              >
                {t}s
              </button>
            ))}
          </div>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            title="Sort outputs"
            className="px-3 py-2 rounded-xl glass-input text-white text-xs font-bold focus:outline-none cursor-pointer"
          >
            <option value="new" className="bg-[#121420]">Newest first</option>
            <option value="old" className="bg-[#121420]">Oldest first</option>
            <option value="big" className="bg-[#121420]">Largest first</option>
            <option value="name" className="bg-[#121420]">Name (A–Z)</option>
          </select>
        </div>
      </Card>

      {!loading && files.length > 0 && (
        <div className="flex items-center gap-2 text-[11px] text-white/40 -mt-2 px-1">
          <span className="font-bold text-white/60">{filteredFiles.length}</span> shown
          {filteredFiles.length !== files.length && <span>of {files.length}</span>}
          <span className="text-white/20">·</span>
          <span className="font-bold text-white/60">{fmtSize(totalSize)}</span> total
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 gap-3">
          <div className="h-8 w-8 rounded-full border-4 border-white/10 border-t-[#E94560] animate-spin" />
          <span className="text-white/40 text-sm font-medium">Scanning outputs folder...</span>
        </div>
      )}

      {/* Grid view */}
      {!loading && filteredFiles.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 border-2 border-dashed border-white/10 rounded-2xl">
          <span className="text-4xl mb-2">📁</span>
          <span className="text-white/50 text-sm">No files found matching the criteria.</span>
        </div>
      )}

      {!loading && filteredFiles.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredFiles.map((file) => {
            const absolutePath = `${outputPath}/${file.name}`;
            const srcUrl = fileUrl(absolutePath);

            return (
              <VideoHoverCard
                key={file.name}
                file={file}
                srcUrl={srcUrl}
                dateStr={getFormatDate(file.mtime)}
                sizeStr={fmtSize(file.size)}
                onDelete={() => deleteFile(file.name)}
                onReveal={() => revealFile(file.name)}
                onReuseTarget={() => reuseAsTarget(file.name)}
                onReuseSource={() => reuseAsSource(file.name)}
                isBusy={busyFile === file.name}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

function VideoHoverCard({ file, srcUrl, dateStr, sizeStr, onDelete, onReveal, onReuseTarget, onReuseSource, isBusy }) {
  const [hovered, setHovered] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    if (!videoRef.current) return;
    if (hovered) {
      videoRef.current.play().catch(() => {});
    } else {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  }, [hovered]);

  const isVideo = file.kind === 'video';

  return (
    <Card
      className="tap overflow-hidden border border-white/5 flex flex-col group/card hover:border-[#E94560]/40 hover:shadow-[0_12px_36px_rgba(233,69,96,0.15)]"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* File preview box */}
      <div className="relative aspect-video bg-black/45 flex items-center justify-center overflow-hidden border-b border-white/5 shrink-0 select-none">
        {isVideo ? (
          <video
            ref={videoRef}
            src={srcUrl}
            muted
            loop
            playsInline
            preload="metadata"
            className="w-full h-full object-contain pointer-events-none transition-transform duration-500 group-hover/card:scale-102"
          />
        ) : (
          <img
            src={srcUrl}
            alt={file.name}
            loading="lazy"
            className="w-full h-full object-contain pointer-events-none transition-transform duration-500 group-hover/card:scale-105"
          />
        )}

        {/* Video badge */}
        {isVideo && (
          <span className="absolute top-2 right-2 bg-black/70 px-2 py-0.5 rounded text-[10px] font-bold text-white tracking-widest pointer-events-none z-10 flex items-center gap-1 border border-white/5">
            🎬 VIDEO
          </span>
        )}

        {/* Action hover mask overlay */}
        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover/card:opacity-100 transition-opacity flex flex-col items-center justify-center gap-2 p-4">
          <div className="flex gap-2 w-full max-w-[200px]">
            <Button
              size="sm"
              variant="primary"
              onClick={onReuseTarget}
              disabled={isBusy}
              className="flex-1 justify-center whitespace-nowrap"
            >
              🎯 Set Target
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={onReuseSource}
              disabled={isBusy}
              className="flex-1 justify-center whitespace-nowrap"
            >
              👥 Extract Faces
            </Button>
          </div>
          <div className="flex gap-2 w-full max-w-[200px]">
            <Button
              size="sm"
              variant="secondary"
              onClick={onReveal}
              disabled={isBusy}
              className="flex-1 justify-center"
            >
              📁 Reveal
            </Button>
            <a
              href={srcUrl}
              download={file.name}
              className="flex-1 rounded-xl font-bold border border-white/5 bg-white/10 hover:bg-white/15 text-white backdrop-blur-md px-3 py-1.5 text-xs flex items-center justify-center"
            >
              ⬇ Get
            </a>
          </div>
        </div>
      </div>

      {/* File details panel */}
      <div className="p-4 flex-1 flex flex-col justify-between gap-2 min-w-0">
        <div className="min-w-0">
          <h4
            className="text-sm font-bold text-white/90 truncate cursor-pointer hover:text-[#E94560]"
            title={file.name}
            onClick={onReveal}
          >
            {file.name}
          </h4>
          <span className="text-[10px] font-mono text-white/40 block mt-0.5">{dateStr}{sizeStr ? ` · ${sizeStr}` : ''}</span>
        </div>
        <div className="flex justify-between items-center shrink-0">
          <span className="text-[10px] uppercase font-bold tracking-widest text-[#E94560]/80">
            {file.kind}
          </span>
          <button
            type="button"
            onClick={onDelete}
            disabled={isBusy}
            className="text-[11px] font-bold text-white/30 hover:text-red-400 cursor-pointer transition-colors"
          >
            🗑️ Delete
          </button>
        </div>
      </div>
    </Card>
  );
}
