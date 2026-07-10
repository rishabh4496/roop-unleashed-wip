import React, { useState, useRef } from 'react';
import { postJSON } from '../api';
import { PERSON_COLORS } from './ui';

// Coarse pose buckets we consider "primary coverage" for a person. Anything the
// backend labels (e.g. "Left Profile + Up Tilt") is matched against these by
// substring so combos still count toward the base angle.
const PRIMARY_POSES = ['Front', 'Left Profile', 'Right Profile', 'Up Tilt', 'Down Tilt'];
const POSE_ICON = {
  'Front': '😐', 'Left Profile': '◧', 'Right Profile': '◨', 'Up Tilt': '⬆', 'Down Tilt': '⬇',
};

// Group target-face indices by their person rank, preserving rank order.
function groupByPerson(groups) {
  const map = new Map();
  groups.forEach((rank, i) => {
    if (!map.has(rank)) map.set(rank, []);
    map.get(rank).push(i);
  });
  return Array.from(map.entries()).sort((a, b) => a[0] - b[0]); // [rank, [indices]]
}

export default function PersonGroups({
  targetFaces, targetGroups, targetNames, targetFacesInfo,
  selTargetFace, setSelTargetFace,
  sourceFaces, faceMapping, setFaceMapping,
  frame, selTarget,
  setTargetFaces, setTargetGroups, setTargetNames, setTargetFacesInfo,
  notify, clearPreviewCache,
}) {
  const [expanded, setExpanded] = useState({});      // rank -> bool override
  const [editingRank, setEditingRank] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [dropTarget, setDropTarget] = useState(null); // rank being hovered in a drag
  const [busy, setBusy] = useState(false);
  const containerRef = useRef(null);

  const people = groupByPerson(targetGroups.slice(0, targetFaces.length));
  const selRank = targetGroups[selTargetFace];

  // Push the four parallel arrays back to the parent from an API payload.
  const applyPayload = (res) => {
    if (res.target_faces) setTargetFaces(res.target_faces);
    if (res.target_groups) setTargetGroups(res.target_groups);
    if (res.target_names !== undefined && setTargetNames) setTargetNames(res.target_names || []);
    if (res.target_faces_info !== undefined && setTargetFacesInfo) setTargetFacesInfo(res.target_faces_info || []);
    clearPreviewCache && clearPreviewCache();
  };

  const isExpanded = (rank) => (rank in expanded ? expanded[rank] : rank === selRank);
  const toggleExpand = (rank) => setExpanded((e) => ({ ...e, [rank]: !isExpanded(rank) }));

  const nameFor = (rank) => (targetNames && targetNames[rank]) || '';
  const labelFor = (rank) => nameFor(rank) || `Person ${rank + 1}`;

  const call = async (path, body, okMsg) => {
    setBusy(true);
    try {
      const res = await postJSON(path, body);
      applyPayload(res);
      if (okMsg) notify(okMsg);
      return res;
    } catch (e) {
      notify(e.message, 'error');
    } finally {
      setBusy(false);
    }
  };

  const removeAngle = async (i) => {
    const res = await call('/api/target/remove_face', { index: i });
    if (res && selTargetFace >= (res.target_faces?.length || 0)) {
      setSelTargetFace(Math.max(0, (res.target_faces?.length || 1) - 1));
    }
  };

  const addAngle = (rank) => call('/api/target/add_angle', { person: rank, index: selTarget, frame },
    `Captured a new angle for ${labelFor(rank)}`);

  const autoCluster = async () => {
    const res = await call('/api/target/autocluster', {});
    if (res) { setExpanded({}); notify(`Grouped into ${res.people} ${res.people === 1 ? 'person' : 'people'}`); }
  };

  const resetGroups = () => {
    const g = targetGroups.map((_, i) => i); // each face its own person
    setTargetGroups(g);
    setExpanded({});
    postJSON('/api/target/group', { groups: g }).then(applyPayload).catch((e) => notify(e.message, 'error'));
  };

  // Move a single angle to another person (or a brand-new one via `newPerson`).
  const reassign = (faceIdx, targetRank) => {
    const g = [...targetGroups];
    g[faceIdx] = targetRank;
    setTargetGroups(g);
    postJSON('/api/target/group', { groups: g }).then(applyPayload).catch((e) => notify(e.message, 'error'));
  };

  const commitName = (rank) => {
    setEditingRank(null);
    const name = editValue.trim();
    if (name === nameFor(rank)) return;
    call('/api/target/name', { person: rank, name });
  };

  const setMapping = (rank, val) => {
    setFaceMapping((prev) => ({ ...prev, [rank]: val }));
    clearPreviewCache && clearPreviewCache();
  };

  // Scoped arrow-key nav: move the selected person up/down. stopPropagation so
  // the global handler (which uses arrows to step video frames) never fires.
  const onKeyDown = (e) => {
    if (e.target !== containerRef.current) return;
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    e.stopPropagation();
    const ranks = people.map(([r]) => r);
    const cur = ranks.indexOf(selRank);
    const nextRank = ranks[Math.min(ranks.length - 1, Math.max(0, cur + (e.key === 'ArrowDown' ? 1 : -1)))];
    const firstFace = targetGroups.indexOf(nextRank);
    if (firstFace >= 0) setSelTargetFace(firstFace);
  };

  const otherRanks = people.map(([r]) => r);
  const nextNewRank = (Math.max(-1, ...otherRanks) + 1);

  if (targetFaces.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-white/10 bg-black/10 p-4 text-center space-y-1.5 select-none">
        <div className="text-2xl opacity-40">🧑‍🤝‍🧑</div>
        <div className="text-xs text-white/50 font-semibold">No people captured yet</div>
        <div className="text-[11px] text-white/30 leading-relaxed">
          Load a target below, scrub to a clear frame, then use <span className="text-white/50 font-bold">“Face from frame”</span> to capture people. Add more angles per person for steadier video swaps.
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onKeyDown={onKeyDown}
      className="space-y-3 outline-none focus:ring-1 focus:ring-[var(--accent)]/30 rounded-xl"
    >
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] font-black uppercase tracking-widest text-white/40">
          {people.length} {people.length === 1 ? 'person' : 'people'} · {targetFaces.length} {targetFaces.length === 1 ? 'angle' : 'angles'}
        </span>
        <div className="flex gap-1.5">
          <button type="button" disabled={busy} onClick={autoCluster}
            title="Group every captured face by identity automatically"
            className="px-2 py-1 rounded-lg text-[10px] font-bold bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-40">
            🧩 Auto-group
          </button>
          <button type="button" disabled={busy} onClick={resetGroups}
            title="Split every angle back into its own person"
            className="px-2 py-1 rounded-lg text-[10px] font-bold bg-white/[0.03] border border-white/10 text-white/50 hover:text-white/80 hover:border-white/20 transition-colors disabled:opacity-40">
            Reset
          </button>
        </div>
      </div>

      {sourceFaces.length > 0 && (
        <div className="text-[10px] text-white/30 flex items-center gap-1.5 select-none">
          <span className="h-1 w-1 rounded-full bg-[var(--accent)]" />
          Drag a source face onto a person, or use the dropdown, to choose who they become.
        </div>
      )}

      {people.map(([rank, indices]) => {
        const color = PERSON_COLORS[rank % PERSON_COLORS.length];
        const open = isExpanded(rank);
        const isSel = rank === selRank;
        const currentMap = faceMapping[rank] !== undefined ? faceMapping[rank] : rank;
        const mapValid = currentMap >= 0 && currentMap < sourceFaces.length;

        // Pose coverage for this person.
        const poses = indices.map((i) => (targetFacesInfo && targetFacesInfo[i]?.pose) || 'Front');
        const covered = new Set();
        poses.forEach((p) => PRIMARY_POSES.forEach((pp) => { if (p.includes(pp)) covered.add(pp); }));
        const missing = ['Front', 'Left Profile', 'Right Profile'].filter((pp) => !covered.has(pp));

        return (
          <div
            key={rank}
            onDragOver={sourceFaces.length ? (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'link'; setDropTarget(rank); } : undefined}
            onDragLeave={() => setDropTarget((d) => (d === rank ? null : d))}
            onDrop={sourceFaces.length ? (e) => {
              e.preventDefault();
              setDropTarget(null);
              const raw = e.dataTransfer.getData('text/roop-source');
              if (raw !== '') setMapping(rank, parseInt(raw, 10));
            } : undefined}
            className={`rounded-xl border transition-all overflow-hidden ${dropTarget === rank ? 'border-[var(--accent)] shadow-[0_0_0_2px_var(--accent-glow)]' : isSel ? 'border-white/20 bg-white/[0.02]' : 'border-white/5 bg-black/25 hover:border-white/10'}`}
            style={isSel ? { boxShadow: `inset 3px 0 0 ${color}` } : { boxShadow: `inset 3px 0 0 ${color}55` }}
          >
            {/* Header */}
            <div className="flex items-center gap-2 px-3 py-2.5 cursor-pointer" onClick={() => { setSelTargetFace(indices[0]); }}>
              <button type="button" onClick={(e) => { e.stopPropagation(); toggleExpand(rank); }}
                className="text-white/40 hover:text-white/80 transition-transform shrink-0" style={{ transform: open ? 'rotate(90deg)' : 'none' }}>▶</button>
              <img src={targetFaces[indices[0]]} alt="" className="w-8 h-8 rounded-lg object-cover shrink-0 border" style={{ borderColor: color }} />
              <div className="flex-1 min-w-0" onClick={(e) => e.stopPropagation()}>
                {editingRank === rank ? (
                  <input
                    autoFocus
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onBlur={() => commitName(rank)}
                    onKeyDown={(e) => { if (e.key === 'Enter') commitName(rank); if (e.key === 'Escape') setEditingRank(null); }}
                    placeholder={`Person ${rank + 1}`}
                    className="w-full px-2 py-1 rounded-md glass-input text-white text-xs font-bold focus:outline-none"
                  />
                ) : (
                  <div className="flex items-center gap-1.5">
                    <span className="font-extrabold text-sm truncate" style={{ color }}>{labelFor(rank)}</span>
                    <button type="button" title="Rename person"
                      onClick={() => { setEditingRank(rank); setEditValue(nameFor(rank)); }}
                      className="text-white/25 hover:text-white/70 text-[11px] shrink-0">✎</button>
                  </div>
                )}
                <div className="text-[10px] text-white/35 font-medium">{indices.length} {indices.length === 1 ? 'angle' : 'angles'}</div>
              </div>

              {/* Mapping dropdown */}
              {sourceFaces.length > 0 && (
                <select
                  onClick={(e) => e.stopPropagation()}
                  value={currentMap}
                  onChange={(e) => setMapping(rank, parseInt(e.target.value, 10))}
                  title="Which source face this person becomes"
                  className={`px-2 py-1 rounded-lg glass-input text-white text-[11px] font-bold focus:outline-none cursor-pointer max-w-[120px] shrink-0 ${mapValid ? '' : 'text-white/50'}`}
                >
                  <option value={-1} className="bg-[#121420]">❌ Skip</option>
                  {sourceFaces.map((_, sfIdx) => (
                    <option key={sfIdx} value={sfIdx} className="bg-[#121420]">🎭 Face {sfIdx + 1}</option>
                  ))}
                </select>
              )}
            </div>

            {/* Body */}
            {open && (
              <div className="px-3 pb-3 space-y-3 border-t border-white/5 pt-3">
                {/* Angle strip */}
                <div className="flex flex-wrap gap-2">
                  {indices.map((i) => {
                    const pose = (targetFacesInfo && targetFacesInfo[i]?.pose) || 'Front';
                    const sel = i === selTargetFace;
                    return (
                      <div key={i} className="relative group/angle">
                        <button type="button" onClick={() => setSelTargetFace(i)}
                          className={`block rounded-lg overflow-hidden border-2 transition-all ${sel ? 'scale-105' : 'opacity-80 hover:opacity-100'}`}
                          style={{ borderColor: sel ? color : 'transparent' }}>
                          <img src={targetFaces[i]} alt={pose} className="w-14 h-14 object-cover" />
                        </button>
                        {/* hover-enlarge popover */}
                        <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover/angle:block z-40">
                          <div className="p-1.5 rounded-xl bg-black/95 border border-white/10 shadow-2xl flex flex-col items-center gap-1">
                            <img src={targetFaces[i]} alt="" className="w-28 h-28 object-cover rounded-lg" />
                            <span className="text-[10px] font-bold text-[var(--accent)] whitespace-nowrap">{pose}</span>
                          </div>
                          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 w-2 h-2 rotate-45 bg-black/95 border-b border-r border-white/10" />
                        </div>
                        {/* pose tag */}
                        <span className="absolute bottom-0.5 left-0.5 right-0.5 text-center text-[7px] font-black text-white bg-black/70 rounded px-0.5 truncate leading-tight pointer-events-none">{pose}</span>
                        {/* per-angle delete */}
                        <button type="button" title="Remove this angle"
                          onClick={(e) => { e.stopPropagation(); removeAngle(i); }}
                          className="absolute -top-1.5 -right-1.5 h-5 w-5 rounded-full bg-black/80 text-white/70 text-[10px] opacity-0 group-hover/angle:opacity-100 hover:bg-[var(--accent-hover)] hover:text-white transition-all flex items-center justify-center border border-white/10">✕</button>
                        {/* reassign to another person */}
                        {people.length > 1 && (
                          <select
                            value={rank}
                            onChange={(e) => reassign(i, parseInt(e.target.value, 10))}
                            title="Move this angle to another person"
                            className="mt-1 w-14 px-1 py-0.5 rounded-md glass-input text-white/70 text-[8px] focus:outline-none cursor-pointer">
                            {otherRanks.map((r) => (
                              <option key={r} value={r} className="bg-[#121420]">→ {labelFor(r)}</option>
                            ))}
                            <option value={nextNewRank} className="bg-[#121420]">→ New</option>
                          </select>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Coverage meter */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[9px] font-black uppercase tracking-widest text-white/30 mr-0.5">Coverage</span>
                  {PRIMARY_POSES.filter((pp) => covered.has(pp)).map((pp) => (
                    <span key={pp} className="px-1.5 py-0.5 rounded-md text-[9px] font-bold border" style={{ color, borderColor: `${color}55`, background: `${color}18` }}>
                      {POSE_ICON[pp]} {pp.replace(' Profile', '').replace(' Tilt', '')}
                    </span>
                  ))}
                  {missing.map((pp) => (
                    <span key={pp} title={`No ${pp} angle captured yet — add one for steadier swaps`}
                      className="px-1.5 py-0.5 rounded-md text-[9px] font-semibold border border-dashed border-white/15 text-white/30">
                      + {pp.replace(' Profile', '')}
                    </span>
                  ))}
                </div>

                {/* Grab angle at current frame */}
                <button type="button" disabled={busy} onClick={() => addAngle(rank)}
                  className="w-full py-1.5 rounded-lg text-[11px] font-bold bg-white/[0.03] border border-white/10 text-white/70 hover:border-[var(--accent)]/40 hover:text-white transition-colors disabled:opacity-40">
                  ➕ Capture this person’s angle at frame {frame}
                </button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
