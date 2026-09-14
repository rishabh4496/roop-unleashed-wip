import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { getJSON, postJSON, postFiles, API } from '../api';
import { Card, Section, Button, InfoBadge, MotionIcon } from './ui';
import { Icon } from '../icons';
import useQueue from './faceswap/useQueue';
import QueuePanel from './faceswap/QueuePanel';
import FacesetLibrary from './faceswap/FacesetLibrary';
import { FACESWAP_DEFAULTS } from './faceswap/defaults';
import { heuristicMsPerFrame } from './faceswap/useRuntimeEstimate';

// Helper to convert index to target preview URL
const targetPreviewUrl = (idx) => `${API}/api/target/preview?index=${idx}&frame=1`;

// Fallback only — the real list is `meta.enhancers`, straight from the backend
// (see the `enhancerOptions` memo below).
//
// This used to be a hand-written list, and it had drifted into two names the
// backend does not have: 'CodeFormer' (it is 'Codeformer') and 'KEEP' (it is
// 'KEEP (sidecar)'). get_processing_plugins matches the enhancer by exact
// string and falls through to no enhancer at all when nothing matches, so
// picking either of those in a batch silently rendered the whole job with
// enhancement OFF — no error, no warning, just a worse output than the one
// asked for. Deriving the list from meta is what stops that recurring.
const ENHANCER_FALLBACK = [
  'GPEN Realistic', 'UltraMax', 'GPEN Ultimate', 'Restore Ultra', 'Restoreformer++', 'Codeformer',
  'GFPGAN', 'GPEN', 'DMDNet', 'None',
];

// The handful offered as one-click "set every file to this" shortcuts. Filtered
// against what the backend actually supports before being shown.
const BULK_ENHANCERS = ['GPEN Realistic', 'UltraMax', 'GPEN Ultimate', 'Restore Ultra', 'Restoreformer++', 'None'];

// Fixing the list above does not fix the Quick Slots and exported presets that
// were saved WHILE it was wrong — those still carry the bad names, and would
// keep rendering with no enhancer. Translated on the way into a payload, which
// is the one place every path (slots, presets, matrix rows, the mode-1 select)
// converges.
const LEGACY_ENHANCER_ALIASES = { CodeFormer: 'Codeformer', KEEP: 'KEEP (sidecar)' };
const normalizeEnhancer = (name) => LEGACY_ENHANCER_ALIASES[name] || name;

// Same rule as the enhancers: the LIST is meta.face_detection_modes, and only
// the parenthetical hints live here. The four spelled out in this panel were a
// subset of the backend's six — "All female" and "All male" simply could not be
// picked for a batch, for no reason anyone recorded.
const DETECTION_MODE_HINTS = {
  'Selected face': 'By Person Rank',
  'All input faces': 'Gallery Order',
  'All faces': 'Swap every detected face',
  'First found': 'First detected face',
};
const DETECTION_MODE_FALLBACK = ['Selected face', 'All input faces', 'All faces', 'First found'];

// A job's frame range is in the BACKEND's coordinates, not the timeline's.
// `frame_start` is a 0-based INCLUSIVE index and `frame_end` is EXCLUSIVE:
// read_frames_thread seeks to frame_start and then reads exactly
// (frame_end - frame_start) frames (ProcessMgr.run_batch_inmem). A fresh target
// therefore reports start_frame 0 / end_frame = total, which renders the whole
// clip.
//
// The timeline on the Face Swap tab DISPLAYS 1-based frame numbers (see the
// note in usePlaybackBuffer), and that display convention had leaked in here as
// `target.start_frame || 1` -- which turns the 0 a fresh target reports into 1
// and silently drops frame 0 from every batch job. Measured on a 100-frame
// clip: 99 frames rendered, frame 0 missing.
const wholeFileRange = (t) => ({
  frame_start: Math.max(0, Number(t?.start_frame) || 0),
  frame_end: Number(t?.end_frame) || Number(t?.frames) || 1,
});

export default function BatchSwap({ meta, settings = {}, notify }) {
  const enhancerOptions = useMemo(
    () => (Array.isArray(meta?.enhancers) && meta.enhancers.length ? meta.enhancers : ENHANCER_FALLBACK),
    [meta],
  );
  const bulkEnhancers = useMemo(
    () => BULK_ENHANCERS.filter((e) => enhancerOptions.includes(e)),
    [enhancerOptions],
  );
  const detectionModes = useMemo(
    () => (Array.isArray(meta?.face_detection_modes) && meta.face_detection_modes.length
      ? meta.face_detection_modes
      : DETECTION_MODE_FALLBACK),
    [meta],
  );
  // ── Server State ────────────────────────────────────────────────────────
  const [targets, setTargets] = useState([]);
  const [sourceFaces, setSourceFaces] = useState([]);
  const [sourceFacesInfo, setSourceFacesInfo] = useState([]);
  const [targetFaces, setTargetFaces] = useState([]);
  const [targetGroups, setTargetGroups] = useState([]);
  const [targetNames, setTargetNames] = useState([]);
  const [loadingState, setLoadingState] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [showFacesetLib, setShowFacesetLib] = useState(false);

  // Queue handle
  const queue = useQueue({ notify });

  // Staged jobs (ready to be sent to queue)
  const [stagedJobs, setStagedJobs] = useState([]);

  // Active Batch Strategy: 'one_to_many' | 'grouped' | 'matrix' | 'recipes'
  const [batchMode, setBatchMode] = useState('one_to_many');

  // Quick Slots & Health Inspector Modal State
  const [activeQuickSlot, setActiveQuickSlot] = useState(null);
  const [showHealthModal, setShowHealthModal] = useState(false);
  const [autoFallbackEnabled, setAutoFallbackEnabled] = useState(true);

  // ── Strategy 1: One-to-Many state ──
  const [mode1Mappings, setMode1Mappings] = useState([{ personRank: 0, sourceIdx: 0 }]);
  const [mode1SwapMode, setMode1SwapMode] = useState('Selected face');
  const [mode1SelectedTargets, setMode1SelectedTargets] = useState([]); // indices
  const [mode1Enhancer, setMode1Enhancer] = useState('Restoreformer++');
  const [mode1EnhancerBlend, setMode1EnhancerBlend] = useState(0.85);
  const [mode1FaceDistance, setMode1FaceDistance] = useState(0.75);

  // ── Strategy 2: Grouped Batch state ──
  const [groups, setGroups] = useState([
    {
      id: 1,
      label: 'Batch Group A',
      targetIndices: [],
      mappings: [{ personRank: 0, sourceIdx: 0 }],
      swapMode: 'Selected face',
      enhancer: 'Restoreformer++',
      faceDistance: 0.75,
    },
  ]);

  // ── Strategy 3: Per-File Matrix state ──
  const [matrixConfig, setMatrixConfig] = useState({});

  // Matrix Filter / Search Query
  const [matrixSearch, setMatrixSearch] = useState('');
  const [matrixFilterStatus, setMatrixFilterStatus] = useState('all'); // 'all' | 'enabled' | 'disabled'

  // Segment Splitter State
  const [splitTargetIdx, setSplitTargetIdx] = useState(0);
  const [splitSegmentCount, setSplitSegmentCount] = useState(4);

  // ── Rehydrate State from Backend ────────────────────────────────────────
  // `settings` is read here only to seed a NEW matrix row's defaults, so it is
  // read through a ref rather than being a dependency. As a dependency it made
  // this callback a new function on every settings edit, and the mount effect
  // below re-ran with it: every change on the Settings tab silently re-fetched
  // /api/state and flipped this whole panel back to its loading state.
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  // Every configuration below addresses a target file by its INDEX -- the keys
  // of matrixConfig, the mode-1 selection, a group's targetIndices -- and
  // /api/target/remove renumbers every file after the one it drops. Nothing
  // re-pointed those indices, so removing a target silently shifted every
  // configuration onto its neighbour: the matrix row that said "file B ->
  // faceset 3, trim 200-400" now described file C, and the batch rendered it
  // that way without a word. The prune effect below only dropped indices past
  // the end of the list, which conceals the shift rather than correcting it.
  //
  // Names are the stable handle -- they are what the queue resolves a job by at
  // dispatch time (routes_queue._run_one), for the same reason.
  const targetNamesRef = useRef([]);

  const refreshBackendState = useCallback(async () => {
    const cfg = settingsRef.current || {};
    try {
      setLoadingState(true);
      const st = await getJSON('/api/state');
      const sFaces = st.source_faces || [];
      const sInfo = st.source_faces_info || [];
      const tg = st.targets || [];
      setSourceFaces(sFaces);
      setSourceFacesInfo(sInfo);
      setTargetFaces(st.target_faces || []);
      setTargetGroups(st.target_groups || []);
      setTargetNames(st.target_names || []);
      setTargets(tg);

      // Re-point index-addressed config at the files it was made for, before
      // anything below fills in defaults for the new numbering.
      const prevNames = targetNamesRef.current;
      const names = tg.map((t) => t.name);
      targetNamesRef.current = names;
      const shifted = prevNames.length > 0
        && (prevNames.length !== names.length
            || prevNames.some((n, i) => n !== names[i]));
      if (shifted) {
        const remap = new Map();
        prevNames.forEach((name, oldIdx) => {
          const at = names.indexOf(name);
          if (at !== -1) remap.set(oldIdx, at);
        });
        const move = (list) => list
          .map((i) => remap.get(i))
          .filter((i) => i !== undefined);
        setMatrixConfig((prev) => {
          const moved = {};
          remap.forEach((newIdx, oldIdx) => {
            if (prev[oldIdx]) moved[newIdx] = prev[oldIdx];
          });
          return moved;
        });
        setMode1SelectedTargets(move);
        setGroups((prev) => prev.map((g) => ({
          ...g, targetIndices: move(g.targetIndices),
        })));
      }

      // Initialize default selected targets for Mode 1 if empty
      setMode1SelectedTargets((prev) => (prev.length === 0 ? tg.map((_, i) => i) : prev));

      // Initialize Matrix Config for all target files
      setMatrixConfig((prev) => {
        const updated = { ...prev };
        tg.forEach((t, idx) => {
          if (!updated[idx]) {
            updated[idx] = {
              mappings: [{ personRank: 0, sourceIdx: 0 }],
              swapMode: 'Selected face',
              enabled: true,
              enhancer: cfg.selected_enhancer || 'Restoreformer++',
              faceDistance: parseFloat(cfg.max_face_distance || 0.75),
              frameStart: wholeFileRange(t).frame_start,
              frameEnd: wholeFileRange(t).frame_end,
            };
          }
        });
        return updated;
      });
    } catch (e) {
      notify?.('Failed to load server state: ' + e.message, 'error');
    } finally {
      setLoadingState(false);
    }
  }, [notify]);

  useEffect(() => {
    refreshBackendState();
  }, [refreshBackendState]);

  // Prune Stale Selected Target / Source Indices on list change
  useEffect(() => {
    if (targets.length === 0) return;
    setMode1SelectedTargets((prev) => prev.filter((i) => i < targets.length));
    setGroups((prev) =>
      prev.map((g) => ({
        ...g,
        targetIndices: g.targetIndices.filter((i) => i < targets.length),
      }))
    );
  }, [targets.length]);

  // ── Quick-Slot Preset Management ─────────────────────────────────────
  const saveQuickSlot = (slotNum) => {
    try {
      const slotData = {
        batchMode,
        mode1Mappings,
        mode1SwapMode,
        mode1Enhancer,
        mode1EnhancerBlend,
        mode1FaceDistance,
        groups,
        matrixConfig,
        savedAt: new Date().toISOString(),
      };
      localStorage.setItem(`roop_batch_slot_${slotNum}`, JSON.stringify(slotData));
      setActiveQuickSlot(slotNum);
      notify?.(`Saved configuration into Quick Slot #${slotNum}!`);
    } catch (e) {
      notify?.('Failed to save slot: ' + e.message, 'error');
    }
  };

  const loadQuickSlot = (slotNum) => {
    try {
      const raw = localStorage.getItem(`roop_batch_slot_${slotNum}`);
      if (!raw) {
        notify?.(`Quick Slot #${slotNum} is empty`, 'info');
        return;
      }
      const data = JSON.parse(raw);
      if (data.batchMode) setBatchMode(data.batchMode);
      if (Array.isArray(data.mode1Mappings)) setMode1Mappings(data.mode1Mappings);
      if (data.mode1SwapMode) setMode1SwapMode(data.mode1SwapMode);
      if (data.mode1Enhancer) setMode1Enhancer(data.mode1Enhancer);
      if (data.mode1EnhancerBlend != null) setMode1EnhancerBlend(data.mode1EnhancerBlend);
      if (data.mode1FaceDistance != null) setMode1FaceDistance(data.mode1FaceDistance);
      if (Array.isArray(data.groups)) setGroups(data.groups);
      if (data.matrixConfig) setMatrixConfig(data.matrixConfig);
      setActiveQuickSlot(slotNum);
      notify?.(`Loaded configuration from Quick Slot #${slotNum}!`);
    } catch (e) {
      notify?.('Failed to load slot: ' + e.message, 'error');
    }
  };

  // ── File Upload Handlers ───────────────────────────────────────────────
  const handleUploadTargets = async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    try {
      setUploading(true);
      notify?.(`Uploading ${files.length} target file(s)...`, 'info');
      await postFiles('/api/target/add', files);
      notify?.(`Successfully added ${files.length} target file(s)`);
      await refreshBackendState();
    } catch (err) {
      notify?.('Failed to upload target files: ' + err.message, 'error');
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const handleUploadSources = async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    try {
      setUploading(true);
      notify?.(`Uploading ${files.length} source faceset(s)...`, 'info');
      await postFiles('/api/source/add', files);
      notify?.(`Successfully added ${files.length} source face(s)`);
      await refreshBackendState();
    } catch (err) {
      notify?.('Failed to upload source faces: ' + err.message, 'error');
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  // ── Source & Target Removal Handlers ─────────────────────────
  const removeSourceFaceset = async (e, idx) => {
    e?.stopPropagation();
    try {
      await postJSON('/api/source/remove', { index: idx });
      notify?.(`Removed source faceset #${idx + 1}`);
      await refreshBackendState();
    } catch (err) {
      notify?.('Failed to remove faceset: ' + err.message, 'error');
    }
  };

  const clearAllSourceFacesets = async () => {
    try {
      await postJSON('/api/source/clear', {});
      notify?.('Cleared all source facesets');
      await refreshBackendState();
    } catch (err) {
      notify?.('Failed to clear sources: ' + err.message, 'error');
    }
  };

  const removeTargetFile = async (e, idx) => {
    e?.stopPropagation();
    try {
      await postJSON('/api/target/remove', { index: idx });
      notify?.(`Removed target file #${idx + 1}`);
      await refreshBackendState();
    } catch (err) {
      notify?.('Failed to remove target: ' + err.message, 'error');
    }
  };

  const clearAllTargetFiles = async () => {
    try {
      await postJSON('/api/target/clear', {});
      notify?.('Cleared all target media');
      await refreshBackendState();
    } catch (err) {
      notify?.('Failed to clear targets: ' + err.message, 'error');
    }
  };

  // ── Multi-Face Payload Builder (Dense & Robust) ────────────────────────
  const createJobPayload = useCallback(
    (mappings = [], swapMode = 'Selected face', overrides = {}) => {
      const base = { ...FACESWAP_DEFAULTS, ...settings };
      
      // Build dense face_mapping array (fill gaps with 0 to prevent nulls)
      let maxRank = 0;
      mappings.forEach((m) => {
        const r = Math.max(0, parseInt(m.personRank, 10) || 0);
        if (r > maxRank) maxRank = r;
      });

      const faceMapping = new Array(maxRank + 1).fill(0);
      mappings.forEach((m) => {
        const rank = Math.max(0, parseInt(m.personRank, 10) || 0);
        const srcIdx = Math.max(0, parseInt(m.sourceIdx, 10) || 0);
        faceMapping[rank] = srcIdx;
      });

      const primarySourceIdx = mappings[0]?.sourceIdx || 0;

      return {
        payload: {
          ...base,
          enhancer: normalizeEnhancer(overrides.enhancer || base.selected_enhancer || 'Restoreformer++'),
          enhancer_type: overrides.enhancer_type || normalizeEnhancer(overrides.enhancer || base.selected_enhancer || 'Restoreformer++'),
          enhancer_blend: parseFloat(overrides.enhancer_blend ?? base.enhancer_blend ?? base.blend_ratio ?? 0.85),
          detection: swapMode || 'Selected face',
          output_method: base.output_method || 'Images & Video',
          video_method: base.video_swapping_method || 'In-Memory processing',
          upscale: base.subsample_upscale || '128px',
          mask_engine: base.mask_engine || 'DFL XSeg',
          mask_engine_2: base.mask_engine_2 || 'None',
          clip_text: base.mask_clip_text || '',
          sam2_model_size: base.sam2_model_size || 'tiny',
          track_identities: !!base.track_identities,
          autorotate: !!base.autorotate_faces,
          face_distance: parseFloat(overrides.faceDistance ?? base.max_face_distance ?? 0.75),
          blend_ratio: parseFloat(overrides.blend_ratio ?? base.blend_ratio ?? base.enhancer_blend ?? 0.85),
          num_swap_steps: parseInt(base.num_swap_steps || 1, 10),
          auto_fallback: autoFallbackEnabled,
          face_mapping: faceMapping,
        },
        primarySourceIdx,
        mappings,
      };
    },
    [settings, autoFallbackEnabled],
  );

  // ── Portable Preset Export & Import ─────────────────────────────────────
  const exportBatchPreset = () => {
    try {
      const preset = {
        type: 'roop-batch-preset',
        version: 3,
        exported_at: new Date().toISOString(),
        batchMode,
        mode1Mappings,
        mode1SwapMode,
        mode1SelectedTargets,
        mode1Enhancer,
        mode1EnhancerBlend,
        mode1FaceDistance,
        groups,
        matrixConfig,
        sourceNames: sourceFacesInfo.map((s) => s.name || ''),
        targetNames: targets.map((t) => t.name || ''),
        stagedJobs,
      };
      const blob = new Blob([JSON.stringify(preset, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `batch_matrix_preset_${Date.now()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      notify?.('Exported Pro Batch Matrix Preset .json');
    } catch (e) {
      notify?.('Failed to export preset: ' + e.message, 'error');
    }
  };

  const importBatchPreset = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const p = JSON.parse(event.target.result);
        if (p.type !== 'roop-batch-preset') {
          throw new Error('File is not a valid Batch Matrix preset JSON');
        }
        if (p.batchMode) setBatchMode(p.batchMode);
        if (Array.isArray(p.mode1Mappings)) setMode1Mappings(p.mode1Mappings);
        if (p.mode1SwapMode) setMode1SwapMode(p.mode1SwapMode);
        if (Array.isArray(p.mode1SelectedTargets)) setMode1SelectedTargets(p.mode1SelectedTargets);
        if (p.mode1Enhancer) setMode1Enhancer(p.mode1Enhancer);
        if (p.mode1EnhancerBlend != null) setMode1EnhancerBlend(p.mode1EnhancerBlend);
        if (p.mode1FaceDistance != null) setMode1FaceDistance(p.mode1FaceDistance);
        if (Array.isArray(p.groups)) setGroups(p.groups);
        if (p.matrixConfig) setMatrixConfig(p.matrixConfig);
        if (Array.isArray(p.stagedJobs)) setStagedJobs(p.stagedJobs);
        notify?.('Successfully imported Pro Batch Matrix Preset!');
      } catch (err) {
        notify?.('Failed to import preset: ' + err.message, 'error');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  // ── Smart Auto-Matching ───────────────────────────────────────────────
  const autoMatchFacesetsToTargets = () => {
    if (targets.length === 0 || sourceFacesInfo.length === 0) {
      notify?.('Requires loaded target files and source facesets to auto-match', 'error');
      return;
    }

    let matchCount = 0;
    const updatedMatrix = { ...matrixConfig };

    targets.forEach((target, tIdx) => {
      const targetClean = target.name.toLowerCase().replace(/[^a-z0-9]/g, ' ');
      let bestSourceIdx = -1;

      sourceFacesInfo.forEach((srcInfo, sIdx) => {
        const srcName = (srcInfo.name || `face_${sIdx}`).toLowerCase().replace(/[^a-z0-9]/g, ' ');
        const tokens = srcName.split(/\s+/).filter((t) => t.length >= 3);
        const isMatch = tokens.some((token) => targetClean.includes(token));
        if (isMatch && bestSourceIdx === -1) {
          bestSourceIdx = sIdx;
        }
      });

      if (bestSourceIdx !== -1) {
        updatedMatrix[tIdx] = {
          ...(updatedMatrix[tIdx] || {}),
          mappings: [{ personRank: 0, sourceIdx: bestSourceIdx }],
          enabled: true,
        };
        matchCount++;
      }
    });

    setMatrixConfig(updatedMatrix);
    if (matchCount > 0) {
      notify?.(`Smart Auto-Matched ${matchCount} target file(s) with source facesets!`);
    } else {
      notify?.('No matching filename patterns found between target media and source facesets', 'info');
    }
  };

  // ── Pro Recipe Generators ─────────────────────────────────────────────
  const recipeCartesianProduct = () => {
    if (targets.length === 0 || sourceFaces.length === 0) {
      notify?.('Requires target files and source facesets', 'error');
      return;
    }
    const newJobs = [];
    targets.forEach((target, tIdx) => {
      sourceFaces.forEach((_, sIdx) => {
        const { payload, primarySourceIdx, mappings } = createJobPayload(
          [{ personRank: 0, sourceIdx: sIdx }],
          'Selected face'
        );
        const sName = sourceFacesInfo[sIdx]?.name || `Faceset #${sIdx + 1}`;
        newJobs.push({
          id: `staged_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
          target_name: target.name,
          target_index: tIdx,
          source_index: primarySourceIdx,
          source_name: sName,
          mappings,
          ...wholeFileRange(target),
          total_frames: target.frames || 1,
          label: `NxM Combinatorial | ${target.name} ➔ ${sName}`,
          payload,
        });
      });
    });
    setStagedJobs((prev) => [...prev, ...newJobs]);
    notify?.(`Generated ${newJobs.length} Cartesian Combination Job(s) (${targets.length} targets × ${sourceFaces.length} sources)`);
  };

  const recipeSequentialMatch = () => {
    if (targets.length === 0 || sourceFaces.length === 0) {
      notify?.('Requires target files and source facesets', 'error');
      return;
    }
    const newJobs = [];
    targets.forEach((target, tIdx) => {
      const sIdx = tIdx % sourceFaces.length;
      const sName = sourceFacesInfo[sIdx]?.name || `Faceset #${sIdx + 1}`;
      const { payload, primarySourceIdx, mappings } = createJobPayload(
        [{ personRank: 0, sourceIdx: sIdx }],
        'Selected face'
      );
      newJobs.push({
        id: `staged_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        target_name: target.name,
        target_index: tIdx,
        source_index: primarySourceIdx,
        source_name: sName,
        mappings,
        ...wholeFileRange(target),
        total_frames: target.frames || 1,
        label: `Sequential | ${target.name} ➔ ${sName}`,
        payload,
      });
    });
    setStagedJobs((prev) => [...prev, ...newJobs]);
    notify?.(`Generated ${newJobs.length} Sequential Match Job(s)`);
  };

  // ── Segment Splitter ────────────────────────────────────────────────
  const splitTargetIntoSegments = () => {
    const target = targets[splitTargetIdx];
    if (!target) {
      notify?.('Selected target file does not exist', 'error');
      return;
    }
    const totalFrames = target.frames || 1;
    if (totalFrames <= 1) {
      notify?.('Target file is a single image or has no frames to split', 'error');
      return;
    }
    const segs = Math.max(2, Math.min(splitSegmentCount, 32));
    const step = Math.ceil(totalFrames / segs);
    const newJobs = [];

    // 0-based, half-open [fs, fe) -- see wholeFileRange. This was written
    // 1-based and inclusive (`i * step + 1` .. `(i + 1) * step`), which left a
    // one-frame hole at EVERY segment boundary as well as dropping frame 0:
    // measured on a 100-frame clip split four ways, 96 frames rendered with
    // 0, 25, 50 and 75 missing. Joining those segments back together is the
    // whole point of the splitter, and the join then has a visible hitch at
    // each seam. Dropping the +1 makes the segments exactly contiguous.
    for (let i = 0; i < segs; i++) {
      const fs = i * step;
      const fe = Math.min((i + 1) * step, totalFrames);
      if (fs >= totalFrames) break;
      const span = fe - fs;

      const { payload, primarySourceIdx, mappings } = createJobPayload(
        [{ personRank: 0, sourceIdx: 0 }],
        'Selected face'
      );

      newJobs.push({
        id: `staged_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        target_name: target.name,
        target_index: splitTargetIdx,
        source_index: primarySourceIdx,
        source_name: sourceFacesInfo[primarySourceIdx]?.name || `Faceset #${primarySourceIdx + 1}`,
        mappings,
        frame_start: fs,
        frame_end: fe,
        total_frames: span,
        // Labelled 1-based inclusive, to read the same as the timeline.
        label: `Segment ${i + 1}/${segs} (${fs + 1}-${fe}) | ${target.name}`,
        payload,
      });
    }

    setStagedJobs((prev) => [...prev, ...newJobs]);
    notify?.(`Split "${target.name}" into ${newJobs.length} segment jobs for parallel rendering`);
  };

  // ── Strategy 1 Handlers ────────────────────────────────────────────────
  const addMode1Mapping = () => {
    setMode1Mappings((prev) => [...prev, { personRank: prev.length, sourceIdx: 0 }]);
  };
  const removeMode1Mapping = (idx) => {
    setMode1Mappings((prev) => prev.filter((_, i) => i !== idx));
  };
  const updateMode1Mapping = (idx, patch) => {
    setMode1Mappings((prev) => prev.map((m, i) => (i === idx ? { ...m, ...patch } : m)));
  };

  const generateMode1Jobs = () => {
    if (mode1SelectedTargets.length === 0) {
      notify?.('Select at least one target file first', 'error');
      return;
    }
    if (sourceFaces.length === 0) {
      notify?.('Add a source faceset first', 'error');
      return;
    }

    const newJobs = mode1SelectedTargets.map((tIdx) => {
      const target = targets[tIdx];
      const targetName = target?.name || `Target ${tIdx + 1}`;
      const { payload, primarySourceIdx, mappings } = createJobPayload(mode1Mappings, mode1SwapMode, {
        enhancer: mode1Enhancer,
        enhancer_type: mode1Enhancer,
        enhancer_blend: mode1EnhancerBlend,
        blend_ratio: mode1EnhancerBlend,
        faceDistance: mode1FaceDistance,
      });

      const mapDesc = mappings.map((m) => `P#${m.personRank + 1}➔F#${m.sourceIdx + 1}`).join(', ');

      return {
        id: `staged_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        target_name: targetName,
        target_index: tIdx,
        source_index: primarySourceIdx,
        source_name: sourceFacesInfo[primarySourceIdx]?.name || `Faceset ${primarySourceIdx + 1}`,
        mappings,
        ...wholeFileRange(target),
        total_frames: target?.frames || 1,
        label: `1:M | ${targetName} (${mapDesc})`,
        payload,
      };
    });

    setStagedJobs((prev) => [...prev, ...newJobs]);
    notify?.(`Generated ${newJobs.length} multi-face batch job(s) for review`);
  };

  // ── Strategy 2 Handlers ────────────────────────────────────────────────
  const addGroup = () => {
    const nextId = groups.length + 1;
    setGroups((prev) => [
      ...prev,
      {
        id: Date.now(),
        label: `Batch Group ${String.fromCharCode(64 + nextId)}`,
        targetIndices: [],
        mappings: [{ personRank: 0, sourceIdx: 0 }],
        swapMode: 'Selected face',
        enhancer: 'Restoreformer++',
        faceDistance: 0.75,
      },
    ]);
  };

  const removeGroup = (groupId) => {
    setGroups((prev) => prev.filter((g) => g.id !== groupId));
  };

  const updateGroup = (groupId, patch) => {
    setGroups((prev) => prev.map((g) => (g.id === groupId ? { ...g, ...patch } : g)));
  };

  const addGroupMapping = (groupId) => {
    setGroups((prev) =>
      prev.map((g) =>
        g.id === groupId ? { ...g, mappings: [...g.mappings, { personRank: g.mappings.length, sourceIdx: 0 }] } : g
      )
    );
  };
  const removeGroupMapping = (groupId, mapIdx) => {
    setGroups((prev) =>
      prev.map((g) =>
        g.id === groupId ? { ...g, mappings: g.mappings.filter((_, i) => i !== mapIdx) } : g
      )
    );
  };
  const updateGroupMapping = (groupId, mapIdx, patch) => {
    setGroups((prev) =>
      prev.map((g) =>
        g.id === groupId
          ? { ...g, mappings: g.mappings.map((m, i) => (i === mapIdx ? { ...m, ...patch } : m)) }
          : g
      )
    );
  };

  const generateGroupJobs = () => {
    if (sourceFaces.length === 0) {
      notify?.('Add a source faceset first', 'error');
      return;
    }

    let totalGen = 0;
    const newJobs = [];

    groups.forEach((grp) => {
      if (grp.targetIndices.length === 0) return;
      grp.targetIndices.forEach((tIdx) => {
        const target = targets[tIdx];
        if (!target) return;
        const targetName = target.name;
        const { payload, primarySourceIdx, mappings } = createJobPayload(grp.mappings, grp.swapMode, {
          enhancer: grp.enhancer,
          faceDistance: grp.faceDistance,
        });

        const mapDesc = mappings.map((m) => `P#${m.personRank + 1}➔F#${m.sourceIdx + 1}`).join(', ');

        newJobs.push({
          id: `staged_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
          target_name: targetName,
          target_index: tIdx,
          source_index: primarySourceIdx,
          source_name: sourceFacesInfo[primarySourceIdx]?.name || `Faceset ${primarySourceIdx + 1}`,
          mappings,
          ...wholeFileRange(target),
          total_frames: target.frames || 1,
          label: `${grp.label} | ${targetName} (${mapDesc})`,
          payload,
        });
        totalGen++;
      });
    });

    if (totalGen === 0) {
      notify?.('No target files assigned to any group', 'error');
      return;
    }

    setStagedJobs((prev) => [...prev, ...newJobs]);
    notify?.(`Generated ${totalGen} group batch job(s)`);
  };

  // ── Strategy 3 Handlers ────────────────────────────────────────────────
  const addMatrixMapping = (tIdx) => {
    setMatrixConfig((prev) => {
      const cfg = prev[tIdx] || { mappings: [] };
      const updatedMaps = [...(cfg.mappings || []), { personRank: cfg.mappings?.length || 0, sourceIdx: 0 }];
      return { ...prev, [tIdx]: { ...cfg, mappings: updatedMaps } };
    });
  };

  const removeMatrixMapping = (tIdx, mapIdx) => {
    setMatrixConfig((prev) => {
      const cfg = prev[tIdx];
      if (!cfg) return prev;
      const updatedMaps = (cfg.mappings || []).filter((_, i) => i !== mapIdx);
      return { ...prev, [tIdx]: { ...cfg, mappings: updatedMaps } };
    });
  };

  const updateMatrixMapping = (tIdx, mapIdx, patch) => {
    setMatrixConfig((prev) => {
      const cfg = prev[tIdx];
      if (!cfg) return prev;
      const updatedMaps = (cfg.mappings || []).map((m, i) => (i === mapIdx ? { ...m, ...patch } : m));
      return { ...prev, [tIdx]: { ...cfg, mappings: updatedMaps } };
    });
  };

  const generateMatrixJobs = () => {
    if (targets.length === 0) {
      notify?.('No target files available', 'error');
      return;
    }
    if (sourceFaces.length === 0) {
      notify?.('Add a source faceset first', 'error');
      return;
    }

    const newJobs = [];
    targets.forEach((target, tIdx) => {
      const cfg = matrixConfig[tIdx];
      if (!cfg || !cfg.enabled) return;

      const { payload, primarySourceIdx, mappings } = createJobPayload(cfg.mappings || [], cfg.swapMode, {
        enhancer: cfg.enhancer,
        faceDistance: cfg.faceDistance,
      });

      const whole = wholeFileRange(target);
      const fs = cfg.frameStart != null ? cfg.frameStart : whole.frame_start;
      const fe = cfg.frameEnd != null ? cfg.frameEnd : whole.frame_end;
      const spanFrames = Math.max(1, fe - fs);
      const mapDesc = mappings.map((m) => `P#${m.personRank + 1}➔F#${m.sourceIdx + 1}`).join(', ');

      newJobs.push({
        id: `staged_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        target_name: target.name,
        target_index: tIdx,
        source_index: primarySourceIdx,
        source_name: sourceFacesInfo[primarySourceIdx]?.name || `Faceset ${primarySourceIdx + 1}`,
        mappings,
        frame_start: fs,
        frame_end: fe,
        total_frames: spanFrames,
        label: `Matrix | ${target.name} (${mapDesc})`,
        payload,
      });
    });

    if (newJobs.length === 0) {
      notify?.('No files enabled for matrix batch', 'error');
      return;
    }

    setStagedJobs((prev) => [...prev, ...newJobs]);
    notify?.(`Generated ${newJobs.length} matrix batch job(s)`);
  };

  // Bulk Matrix Actions
  const bulkSetMatrixSource = (srcIdx) => {
    setMatrixConfig((prev) => {
      const copy = { ...prev };
      Object.keys(copy).forEach((k) => {
        copy[k] = { ...copy[k], mappings: [{ personRank: 0, sourceIdx: srcIdx }] };
      });
      return copy;
    });
    notify?.(`Set all target files to Faceset #${srcIdx + 1}`);
  };

  const bulkSetMatrixEnhancer = (enhancerName) => {
    setMatrixConfig((prev) => {
      const copy = { ...prev };
      Object.keys(copy).forEach((k) => {
        copy[k] = { ...copy[k], enhancer: enhancerName };
      });
      return copy;
    });
    notify?.(`Set all matrix items enhancer to ${enhancerName}`);
  };

  const bulkToggleMatrixEnable = (enableState) => {
    setMatrixConfig((prev) => {
      const copy = { ...prev };
      Object.keys(copy).forEach((k) => {
        copy[k] = { ...copy[k], enabled: enableState };
      });
      return copy;
    });
    notify?.(enableState ? 'Enabled all matrix target files' : 'Disabled all matrix target files');
  };

  const bulkAutoIncrementTargetPersons = () => {
    setMatrixConfig((prev) => {
      const copy = { ...prev };
      Object.keys(copy).forEach((k, idx) => {
        copy[k] = { ...copy[k], mappings: [{ personRank: idx, sourceIdx: 0 }] };
      });
      return copy;
    });
    notify?.('Auto-incremented target person ranks (P#1, P#2, P#3...) across files');
  };

  // Filtered Matrix targets
  const filteredMatrixTargets = useMemo(() => {
    return targets
      .map((target, originalIndex) => ({ target, originalIndex }))
      .filter(({ target, originalIndex }) => {
        const cfg = matrixConfig[originalIndex] || {};
        if (matrixFilterStatus === 'enabled' && !cfg.enabled) return false;
        if (matrixFilterStatus === 'disabled' && cfg.enabled) return false;
        if (!matrixSearch.trim()) return true;
        const q = matrixSearch.toLowerCase();
        return target.name.toLowerCase().includes(q);
      });
  }, [targets, matrixConfig, matrixFilterStatus, matrixSearch]);

  // ── Smart Queue Reordering / Sorting ─────────────────────────────────────
  const sortStagedJobs = (criteria) => {
    setStagedJobs((prev) => {
      const copy = [...prev];
      if (criteria === 'shortest') {
        copy.sort((a, b) => (a.total_frames || 0) - (b.total_frames || 0));
        notify?.('Sorted staged jobs by shortest frame count first');
      } else if (criteria === 'longest') {
        copy.sort((a, b) => (b.total_frames || 0) - (a.total_frames || 0));
        notify?.('Sorted staged jobs by longest frame count first');
      } else if (criteria === 'name') {
        copy.sort((a, b) => a.target_name.localeCompare(b.target_name));
        notify?.('Sorted staged jobs alphabetically by file name');
      } else if (criteria === 'source') {
        copy.sort((a, b) => (a.source_index || 0) - (b.source_index || 0));
        notify?.('Grouped staged jobs by source faceset');
      }
      return copy;
    });
  };

  // ── Batch Time & Frame Estimator ─────────────────────────────────────────
  // Costed per job from that job's OWN payload, through the same heuristic the
  // Face Swap tab's pre-run estimate uses. Staged jobs can differ from each
  // other in every lever that matters (enhancer, detection resolution, swap
  // steps, tracking), so a single flat ms/frame for the whole batch — which is
  // what this was — could not be right for more than one of them at a time.
  const stagedStats = useMemo(() => {
    let totalFrames = 0;
    let estSeconds = 0;

    stagedJobs.forEach((job) => {
      const frames = job.total_frames || job.payload?.end_frame || 1;
      totalFrames += frames;
      const pl = job.payload || {};
      // The override lands on `enhancer`; `selected_enhancer` is the base
      // config it was built from. The heuristic reads the latter name.
      const perFrameMs = heuristicMsPerFrame(
        { ...pl, selected_enhancer: pl.enhancer ?? pl.selected_enhancer },
        undefined,   // no telemetry on this tab — the heuristic's own default
      );
      estSeconds += (frames * perFrameMs) / 1000;
    });

    const mins = Math.floor(estSeconds / 60);
    const secs = Math.round(estSeconds % 60);
    const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;

    return { totalFrames, timeStr };
  }, [stagedJobs]);

  // ── Pre-flight checks ────────────────────────────────────────────────────
  // These are what the health modal reports. It used to print three fixed
  // lines — two green ticks claiming the target names and the source indices
  // had been resolved, and a shield claiming auto-fallback was on — none of
  // which looked at anything. A pre-flight that always passes is worse than
  // none, because the button under it starts the batch.
  //
  // The queue resolves a job by target NAME at dispatch time and by source
  // INDEX into the loaded facesets, so those are exactly the two things that
  // can be stale here: a target removed after staging, or a mapping pointing
  // past the end of a faceset list that has since shrunk.
  const preflight = useMemo(() => {
    const names = new Set(targets.map((t) => t.name));
    const missingTargets = [...new Set(
      stagedJobs.filter((j) => !names.has(j.target_name)).map((j) => j.target_name),
    )];
    const badSources = stagedJobs.filter((j) => {
      const idxs = (j.mappings || []).map((m) => Number(m.sourceIdx));
      idxs.push(Number(j.source_index));
      return idxs.some((i) => !Number.isInteger(i) || i < 0 || i >= sourceFaces.length);
    });
    return {
      missingTargets,
      badSourceCount: badSources.length,
      ok: missingTargets.length === 0 && badSources.length === 0,
    };
  }, [stagedJobs, targets, sourceFaces.length]);

  // ── Staged Jobs Commit Handlers (Atomic Batch Add) ──────────────────────
  const enqueueStagedJobs = async (autoStart = false) => {
    if (stagedJobs.length === 0) {
      notify?.('No staged jobs to enqueue', 'error');
      return;
    }

    try {
      const jobsToAdd = stagedJobs.map((j) => ({
        target_name: j.target_name,
        source_index: j.source_index,
        source_name: j.source_name,
        payload: j.payload,
        frame_start: j.frame_start,
        frame_end: j.frame_end,
        label: j.label,
      }));

      // Uses atomic /api/queue/add_batch via useQueue hook.
      //
      // The result has to be CHECKED. useQueue's `act` swallows the failure --
      // it notifies, re-reads the queue and returns null, it does not throw --
      // so the catch below never fires on a rejected request. This used to
      // announce "Enqueued N job(s)" and then clear stagedJobs regardless, so a
      // backend that refused the batch left the user with a success toast and
      // an empty staging list: the whole configuration gone, with nothing
      // queued. Keeping the staged jobs on failure is the point.
      const snap = await queue.addMany(jobsToAdd);
      if (!snap) return;          // `act` has already surfaced the reason
      notify?.(`Enqueued ${jobsToAdd.length} job(s) to server queue`);
      setStagedJobs([]);

      if (autoStart) {
        // Same again: /api/queue/start answers 409 when a single run is already
        // processing, or when the hardware benchmark holds the GPU.
        if (await queue.start()) notify?.('Started batch processing queue!');
      }
    } catch (e) {
      notify?.('Failed to enqueue jobs: ' + e.message, 'error');
    }
  };

  const removeStagedJob = (jobId) => {
    setStagedJobs((prev) => prev.filter((j) => j.id !== jobId));
  };

  const clearStagedJobs = () => {
    setStagedJobs([]);
  };

  // Target Person Ranks list options
  const targetPersonOptions = useMemo(() => {
    const count = Math.max(
      1,
      targetGroups.length,
      targetFaces.length,
      targetNames.length,
      5
    );
    return Array.from({ length: count }, (_, i) => ({
      rank: i,
      label: targetNames[i] ? `Person #${i + 1} (${targetNames[i]})` : `Person #${i + 1}`,
    }));
  }, [targetGroups, targetFaces, targetNames]);

  return (
    <div className="space-y-6 pb-12 relative">
      {/* ── Pre-flight Batch Health Inspector Modal ── */}
      {showHealthModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#121216] border border-white/10 rounded-2xl max-w-xl w-full p-6 space-y-4 shadow-2xl animate-scale-in text-white">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="text-lg font-bold flex items-center gap-2.5 text-yellow-400">
                <MotionIcon icon={Icon.wand} size="md" variant="amber" animate="pulse" />
                Pre-flight Batch Health Breakdown
              </h3>
              <button
                type="button"
                onClick={() => setShowHealthModal(false)}
                aria-label="Close health breakdown modal"
                className="text-white/40 hover:text-white font-bold"
              >
                ✕
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="p-3 bg-white/5 rounded-xl border border-white/5">
                <span className="text-nano text-white/40 block">Staged Jobs</span>
                <span className="text-lg font-bold text-white">{stagedJobs.length}</span>
              </div>
              <div className="p-3 bg-white/5 rounded-xl border border-white/5">
                <span className="text-nano text-white/40 block">Total Frames</span>
                <span className="text-lg font-bold text-emerald-400">
                  {stagedStats.totalFrames.toLocaleString()}
                </span>
              </div>
              <div className="p-3 bg-white/5 rounded-xl border border-white/5">
                <span className="text-nano text-white/40 block">Est. Runtime</span>
                <span className="text-lg font-bold text-amber-400">{stagedStats.timeStr}</span>
              </div>
              <div className="p-3 bg-white/5 rounded-xl border border-white/5">
                <span className="text-nano text-white/40 block">Est. Disk Space</span>
                <span className="text-lg font-bold text-cyan-400">
                  ~{Math.round(stagedStats.totalFrames * 0.25)} MB
                </span>
              </div>
            </div>

            <div className="space-y-2 bg-black/50 p-4 rounded-xl border border-white/5 text-xs">
              <span className="font-semibold text-white/70 block uppercase tracking-wider text-nano">
                System Diagnostics & Verification
              </span>
              <div className="space-y-1.5 text-white/80">
                {preflight.missingTargets.length === 0 ? (
                  <div className="flex items-center gap-2 text-emerald-400">
                    <span>✓</span>
                    <span>
                      All {stagedJobs.length} target name{stagedJobs.length === 1 ? '' : 's'} resolve
                      against the {targets.length} loaded file{targets.length === 1 ? '' : 's'}.
                    </span>
                  </div>
                ) : (
                  <div className="flex items-start gap-2 text-red-400">
                    <span>✕</span>
                    <span>
                      {preflight.missingTargets.length} target{preflight.missingTargets.length === 1 ? '' : 's'} no
                      longer loaded — these jobs will fail at dispatch:{' '}
                      <span className="font-mono text-red-300">{preflight.missingTargets.join(', ')}</span>
                    </span>
                  </div>
                )}

                {preflight.badSourceCount === 0 ? (
                  <div className="flex items-center gap-2 text-emerald-400">
                    <span>✓</span>
                    <span>
                      Every source mapping points inside the {sourceFaces.length} loaded
                      faceset{sourceFaces.length === 1 ? '' : 's'}.
                    </span>
                  </div>
                ) : (
                  <div className="flex items-start gap-2 text-red-400">
                    <span>✕</span>
                    <span>
                      {preflight.badSourceCount} job{preflight.badSourceCount === 1 ? '' : 's'} map to a faceset
                      index that no longer exists — they would swap the wrong face or none at all.
                      Re-generate them after reloading the sources.
                    </span>
                  </div>
                )}

                <div className={`flex items-center gap-2 ${autoFallbackEnabled ? 'text-yellow-300' : 'text-white/45'}`}>
                  <span>{autoFallbackEnabled ? '🛡️' : '○'}</span>
                  <span>
                    {autoFallbackEnabled
                      ? 'Auto-fallback on: if GPU memory spikes, a job retries with enhancer=None.'
                      : 'Auto-fallback off: a job that runs out of GPU memory fails instead of retrying.'}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex justify-end gap-3 pt-2">
              <Button size="sm" variant="secondary" onClick={() => setShowHealthModal(false)}>
                Cancel
              </Button>
              <Button
                size="sm"
                variant="primary"
                onClick={() => {
                  setShowHealthModal(false);
                  enqueueStagedJobs(true);
                }}
              >
                {preflight.ok
                  ? `🚀 Confirm & Launch Batch (${stagedJobs.length} Jobs)`
                  : `⚠ Launch Anyway (${stagedJobs.length} Jobs)`}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* ── Top Header Banner ── */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-6 rounded-2xl">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Icon.batch className="text-[var(--accent)]" size={24} />
            Batch Matrix & Multi-File Workbench
          </h2>
          <p className="text-xs text-white/50 mt-1 max-w-2xl">
            Configure multi-file batch swaps with flexible combinations: single faceset across multiple files,
            custom file-group rules, distinct facesets per target file, or automated Pro Recipes.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <label className="cursor-pointer">
            <input type="file" multiple accept="video/*,image/*" onChange={handleUploadTargets} className="sr-only" />
            <Button size="sm" variant="secondary" className="pointer-events-none" disabled={uploading}>
              <Icon.upload size={14} className="mr-1.5" />
              + Add Targets
            </Button>
          </label>

          <label className="cursor-pointer">
            <input type="file" multiple accept="image/*,.fsz" onChange={handleUploadSources} className="sr-only" />
            <Button size="sm" variant="secondary" className="pointer-events-none" disabled={uploading}>
              <Icon.upload size={14} className="mr-1.5" />
              + Add Sources
            </Button>
          </label>

          {/* Persistent Faceset Library Toggle Button */}
          <Button
            size="sm"
            variant={showFacesetLib ? 'primary' : 'secondary'}
            onClick={() => setShowFacesetLib((v) => !v)}
            title="Open persistent Faceset Library"
          >
            <Icon.faces size={14} className="mr-1" />
            {showFacesetLib ? 'Hide Library' : 'Faceset Library'}
          </Button>

          {/* Preset Export / Import */}
          <Button size="sm" variant="secondary" onClick={exportBatchPreset} title="Export Batch Preset JSON">
            <Icon.download size={14} className="mr-1" />
            Export Preset
          </Button>

          <label className="cursor-pointer">
            <input type="file" accept=".json" onChange={importBatchPreset} className="sr-only" />
            <Button size="sm" variant="secondary" className="pointer-events-none" title="Import Batch Preset JSON">
              <Icon.upload size={14} className="mr-1" />
              Import Preset
            </Button>
          </label>

          <Button size="sm" variant="ghost" onClick={refreshBackendState} title="Refresh media library">
            <Icon.refresh size={14} className={loadingState ? 'animate-spin' : ''} />
          </Button>
        </div>
      </div>

      {/* ── Quick-Slot Preset Toolbar & Settings ── */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 bg-white/5 rounded-2xl border border-white/10 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-bold text-white flex items-center gap-1.5">
            <Icon.settings size={14} className="text-[var(--accent)]" />
            Quick Slots (1–5):
          </span>
          {[1, 2, 3, 4, 5].map((slot) => {
            const hasSaved = !!localStorage.getItem(`roop_batch_slot_${slot}`);
            const isActive = activeQuickSlot === slot;
            return (
              <div key={slot} className="flex items-center gap-1 bg-black/40 p-1 rounded-xl border border-white/5">
                <button
                  type="button"
                  onClick={() => loadQuickSlot(slot)}
                  className={`px-2 py-0.5 rounded-lg text-micro font-semibold transition-all ${
                    isActive
                      ? 'bg-[var(--accent)] text-black font-bold'
                      : hasSaved
                      ? 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30'
                      : 'bg-white/5 text-white/40 hover:bg-white/10'
                  }`}
                  title={hasSaved ? `Load Quick Slot #${slot}` : `Slot #${slot} is empty`}
                >
                  Slot {slot}
                </button>
                <button
                  type="button"
                  onClick={() => saveQuickSlot(slot)}
                  className="text-nano text-white/40 hover:text-[var(--accent)] font-bold px-1"
                  title={`Save current setup into Quick Slot #${slot}`}
                  aria-label={`Save Quick Slot #${slot}`}
                >
                  💾
                </button>
              </div>
            );
          })}
        </div>

        {/* Auto-Fallback Toggle */}
        <label className="flex items-center gap-2 text-micro cursor-pointer text-white/70 hover:text-white">
          <input
            type="checkbox"
            checked={autoFallbackEnabled}
            onChange={(e) => setAutoFallbackEnabled(e.target.checked)}
            className="rounded border-white/20 bg-black/40 text-[var(--accent)] focus:ring-0"
          />
          <span>🛡️ Auto-Fallback Enhancer (Retry with 'None' if GPU Out-Of-Memory Error)</span>
        </label>
      </div>

      {/* ── Persistent Faceset Library Panel (Full-Width) ── */}
      {showFacesetLib && (
        <div className="animate-fade-in mb-3">
          <FacesetLibrary
            canSave={sourceFaces.length > 0}
            onLoaded={(r) => {
              setSourceFaces(r.source_faces || []);
              if (r.source_faces_info) setSourceFacesInfo(r.source_faces_info);
              refreshBackendState();
            }}
            notify={notify}
          />
        </div>
      )}

      {/* ── Summary Library Strip ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Source Facesets Summary */}
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-white/60 flex items-center gap-1.5">
              <Icon.faces size={14} className="text-[var(--accent)]" />
              Loaded Source Facesets ({sourceFaces.length})
            </span>

            <div className="flex items-center gap-2.5">
              {sourceFaces.length > 0 && (
                <button
                  type="button"
                  onClick={clearAllSourceFacesets}
                  className="text-micro font-medium text-red-400/80 hover:text-red-400 hover:underline"
                  title="Clear all loaded source facesets"
                >
                  Clear All
                </button>
              )}
              <button
                type="button"
                onClick={() => setShowFacesetLib((v) => !v)}
                className="text-micro font-semibold text-[var(--accent)] hover:underline flex items-center gap-1"
              >
                <Icon.faces size={12} />
                {showFacesetLib ? 'Hide Library' : 'Browse Library'}
              </button>
            </div>
          </div>
          {sourceFaces.length === 0 ? (
            <div className="text-xs text-white/40 italic p-3 text-center border border-dashed border-white/10 rounded-xl space-y-1.5">
              <p>No source facesets loaded in active workspace.</p>
              <button
                type="button"
                onClick={() => setShowFacesetLib(true)}
                className="text-micro text-[var(--accent)] font-semibold hover:underline block mx-auto"
              >
                + Browse & Load from Faceset Library
              </button>
            </div>
          ) : (
            <div className="flex flex-wrap gap-2.5 max-h-36 overflow-y-auto pr-1">
              {sourceFaces.map((thumb, idx) => (
                <div
                  key={idx}
                  className={`relative group/src flex items-center gap-2 p-1.5 pr-7 rounded-xl border transition-all ${
                    mode1Mappings[0]?.sourceIdx === idx
                      ? 'bg-[var(--accent)]/15 border-[var(--accent)] text-white'
                      : 'bg-white/5 border-white/5 text-white/70 hover:bg-white/10'
                  }`}
                >
                  <img
                    src={thumb}
                    alt={`Faceset ${idx + 1}`}
                    className="w-8 h-8 rounded-lg object-cover bg-black/40"
                  />
                  <div className="text-micro">
                    <span className="font-semibold block truncate max-w-[90px]">
                      {sourceFacesInfo[idx]?.name || `Faceset #${idx + 1}`}
                    </span>
                    <span className="text-white/40 block">Idx {idx}</span>
                  </div>

                  {/* Remove Individual Faceset Button */}
                  <button
                    type="button"
                    onClick={(e) => removeSourceFaceset(e, idx)}
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-black/60 hover:bg-red-500/90 text-white/50 hover:text-white flex items-center justify-center text-nano font-bold opacity-0 group-hover/src:opacity-100 transition-all"
                    title={`Remove Faceset #${idx + 1}`}
                    aria-label={`Remove Faceset #${idx + 1}`}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Target Files Summary */}
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-white/60 flex items-center gap-1.5">
              <Icon.outputs size={14} className="text-[var(--accent)]" />
              Loaded Target Media ({targets.length})
            </span>

            {targets.length > 0 && (
              <button
                type="button"
                onClick={clearAllTargetFiles}
                className="text-micro font-medium text-red-400/80 hover:text-red-400 hover:underline"
                title="Clear all loaded target media"
              >
                Clear All
              </button>
            )}
          </div>
          {targets.length === 0 ? (
            <div className="text-xs text-white/40 italic p-3 text-center border border-dashed border-white/10 rounded-xl">
              No target media loaded. Click "+ Add Targets" above to upload videos or images.
            </div>
          ) : (
            <div className="flex flex-wrap gap-2 max-h-36 overflow-y-auto pr-1">
              {targets.map((target, idx) => (
                <div
                  key={idx}
                  className="relative group/tgt flex items-center gap-2 p-1.5 pr-7 rounded-xl bg-white/5 border border-white/5 text-white/80 text-micro"
                >
                  <img
                    src={targetPreviewUrl(idx)}
                    alt={target.name}
                    className="w-7 h-7 rounded-lg object-cover bg-black/50"
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                  <span className="truncate max-w-[110px] font-medium" title={target.name}>
                    {idx + 1}. {target.name}
                  </span>
                  <span className="text-white/30 text-nano">({target.frames || 1}f)</span>

                  {/* Remove Individual Target File Button */}
                  <button
                    type="button"
                    onClick={(e) => removeTargetFile(e, idx)}
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-black/60 hover:bg-red-500/90 text-white/50 hover:text-white flex items-center justify-center text-nano font-bold opacity-0 group-hover/tgt:opacity-100 transition-all"
                    title={`Remove ${target.name}`}
                    aria-label={`Remove ${target.name}`}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* ── Strategy Selection Tabs ── */}
      <Section title="Select Batch Strategy & Pro Generators">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <button
            type="button"
            onClick={() => setBatchMode('one_to_many')}
            className={`p-4 rounded-xl text-left border transition-all ${
              batchMode === 'one_to_many'
                ? 'bg-[var(--accent)]/15 border-[var(--accent)] shadow-[0_0_15px_rgba(234,179,8,0.15)]'
                : 'bg-white/5 border-white/5 hover:bg-white/10'
            }`}
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="font-bold text-sm text-white flex items-center gap-2">
                <Icon.compare size={16} className="text-[var(--accent)]" />
                1 Faceset ➔ Multi Files
              </span>
              {batchMode === 'one_to_many' && <Icon.done size={14} className="text-[var(--accent)]" />}
            </div>
            <p className="text-xs text-white/50 leading-relaxed">
              Apply selected source faceset and target mappings across multiple selected target files.
            </p>
          </button>

          <button
            type="button"
            onClick={() => setBatchMode('grouped')}
            className={`p-4 rounded-xl text-left border transition-all ${
              batchMode === 'grouped'
                ? 'bg-[var(--accent)]/15 border-[var(--accent)] shadow-[0_0_15px_rgba(234,179,8,0.15)]'
                : 'bg-white/5 border-white/5 hover:bg-white/10'
            }`}
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="font-bold text-sm text-white flex items-center gap-2">
                <Icon.split size={16} className="text-[var(--accent)]" />
                Grouped Multi-Batch
              </span>
              {batchMode === 'grouped' && <Icon.done size={14} className="text-[var(--accent)]" />}
            </div>
            <p className="text-xs text-white/50 leading-relaxed">
              Assign distinct target file groups to different facesets and target face ranks.
            </p>
          </button>

          <button
            type="button"
            onClick={() => setBatchMode('matrix')}
            className={`p-4 rounded-xl text-left border transition-all ${
              batchMode === 'matrix'
                ? 'bg-[var(--accent)]/15 border-[var(--accent)] shadow-[0_0_15px_rgba(234,179,8,0.15)]'
                : 'bg-white/5 border-white/5 hover:bg-white/10'
            }`}
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="font-bold text-sm text-white flex items-center gap-2">
                <Icon.layout size={16} className="text-[var(--accent)]" />
                Per-File Matrix
              </span>
              {batchMode === 'matrix' && <Icon.done size={14} className="text-[var(--accent)]" />}
            </div>
            <p className="text-xs text-white/50 leading-relaxed">
              Full matrix grid: explicitly map distinct facesets and target faces per file individually.
            </p>
          </button>

          <button
            type="button"
            onClick={() => setBatchMode('recipes')}
            className={`p-4 rounded-xl text-left border transition-all ${
              batchMode === 'recipes'
                ? 'bg-[var(--accent)]/15 border-[var(--accent)] shadow-[0_0_15px_rgba(234,179,8,0.15)]'
                : 'bg-white/5 border-white/5 hover:bg-white/10'
            }`}
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="font-bold text-sm text-white flex items-center gap-2">
                <Icon.wand size={16} className="text-[var(--accent)]" />
                Pro Recipes & Splitter
              </span>
              {batchMode === 'recipes' && <Icon.done size={14} className="text-[var(--accent)]" />}
            </div>
            <p className="text-xs text-white/50 leading-relaxed">
              1-Click Combinatorial Matrix, Sequential Pairs & Multi-Segment Video Splitter.
            </p>
          </button>
        </div>
      </Section>

      {/* ── Mode 1 Panel: 1 Faceset + 1 Target Face -> Multi Files ── */}
      {batchMode === 'one_to_many' && (
        <Section title="Strategy 1: Single Faceset + Target Mapping ➔ Multi Files">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Controls */}
            <div className="space-y-4 md:col-span-1">
              {/* Multi-Face Mapping Builder */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-white/70 flex items-center gap-1.5">
                    Target Face Mappings ({mode1Mappings.length})
                    <InfoBadge info="Map target person ranks in destination media to source facesets." />
                  </label>
                  <button
                    type="button"
                    onClick={addMode1Mapping}
                    className="text-micro text-[var(--accent)] hover:underline font-medium"
                  >
                    + Add Person Mapping
                  </button>
                </div>

                {mode1Mappings.map((m, mapIdx) => (
                  <div key={mapIdx} className="p-2.5 rounded-xl bg-black/40 border border-white/10 space-y-2">
                    <div className="flex items-center justify-between text-micro text-white/50">
                      <span>Mapping #{mapIdx + 1}</span>
                      {mode1Mappings.length > 1 && (
                        <button
                          type="button"
                          onClick={() => removeMode1Mapping(mapIdx)}
                          className="hover:text-red-400 font-bold"
                          aria-label={`Remove mapping ${mapIdx + 1}`}
                        >
                          ✕
                        </button>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <span className="text-nano text-white/40 block mb-0.5">Target Person</span>
                        <select
                          value={m.personRank}
                          onChange={(e) => updateMode1Mapping(mapIdx, { personRank: parseInt(e.target.value, 10) })}
                          className="w-full bg-black/60 border border-white/10 rounded-lg px-2 py-1 text-micro text-white"
                        >
                          {targetPersonOptions.map((opt) => (
                            <option key={opt.rank} value={opt.rank}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <span className="text-nano text-white/40 block mb-0.5">Source Faceset</span>
                        <select
                          value={m.sourceIdx}
                          onChange={(e) => updateMode1Mapping(mapIdx, { sourceIdx: parseInt(e.target.value, 10) })}
                          className="w-full bg-black/60 border border-white/10 rounded-lg px-2 py-1 text-micro text-white"
                        >
                          {sourceFaces.map((_, idx) => (
                            <option key={idx} value={idx}>
                              {sourceFacesInfo[idx]?.name || `Faceset #${idx + 1}`}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div>
                <label className="block text-xs font-semibold text-white/70 mb-1.5">Detection / Swap Mode</label>
                <select
                  value={mode1SwapMode}
                  onChange={(e) => setMode1SwapMode(e.target.value)}
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-3 py-2 text-xs text-white"
                >
                  {detectionModes.map((m) => (
                    <option key={m} value={m}>
                      {DETECTION_MODE_HINTS[m] ? `${m} (${DETECTION_MODE_HINTS[m]})` : m}
                    </option>
                  ))}
                </select>
              </div>

              {/* Quality Overrides */}
              <div className="pt-2 border-t border-white/10 space-y-3">
                <span className="text-micro font-semibold uppercase tracking-wider text-white/50 block">
                  Quality Overrides
                </span>
                <div>
                  <label className="block text-nano font-medium text-white/60 mb-1">Face Enhancer</label>
                  <select
                    value={mode1Enhancer}
                    onChange={(e) => setMode1Enhancer(e.target.value)}
                    className="w-full bg-black/50 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-white"
                  >
                    {enhancerOptions.map((e) => (
                      <option key={e} value={e}>
                        {e}
                      </option>
                    ))}
                  </select>
                </div>
                {mode1Enhancer && mode1Enhancer !== 'None' && mode1Enhancer !== 'none' && (
                  <div>
                    <label className="block text-nano font-medium text-white/60 mb-1">
                      Enhancer Blend / Opacity ({Number(mode1EnhancerBlend ?? 0.85).toFixed(2)})
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={mode1EnhancerBlend ?? 0.85}
                      onChange={(e) => setMode1EnhancerBlend(parseFloat(e.target.value))}
                      className="w-full accent-[var(--accent)]"
                    />
                  </div>
                )}
                <div>
                  <label className="block text-nano font-medium text-white/60 mb-1">
                    Max Face Distance ({mode1FaceDistance})
                  </label>
                  <input
                    type="range"
                    min="0.2"
                    max="1.2"
                    step="0.05"
                    value={mode1FaceDistance}
                    onChange={(e) => setMode1FaceDistance(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)]"
                  />
                </div>
              </div>

              <Button
                variant="primary"
                className="w-full justify-center py-2.5 mt-2"
                onClick={generateMode1Jobs}
                disabled={targets.length === 0 || sourceFaces.length === 0}
              >
                <Icon.add size={16} className="mr-1.5" />
                Generate Batch Jobs ({mode1SelectedTargets.length} Files)
              </Button>
            </div>

            {/* Target Media Picker */}
            <div className="md:col-span-2 space-y-2">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-semibold text-white/70">
                  Select Target Files ({mode1SelectedTargets.length} / {targets.length} selected)
                </span>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setMode1SelectedTargets(targets.map((_, i) => i))}
                    className="text-micro text-[var(--accent)] hover:underline"
                  >
                    Select All
                  </button>
                  <span className="text-white/20">|</span>
                  <button
                    type="button"
                    onClick={() => setMode1SelectedTargets([])}
                    className="text-micro text-white/40 hover:text-white"
                  >
                    Clear
                  </button>
                </div>
              </div>

              {targets.length === 0 ? (
                <div className="p-8 text-center border border-dashed border-white/10 rounded-xl text-xs text-white/40">
                  No target files available. Add files using the button in the header.
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-h-72 overflow-y-auto pr-1">
                  {targets.map((target, idx) => {
                    const isSel = mode1SelectedTargets.includes(idx);
                    return (
                      <div
                        key={idx}
                        onClick={() => {
                          setMode1SelectedTargets((prev) =>
                            isSel ? prev.filter((i) => i !== idx) : [...prev, idx]
                          );
                        }}
                        className={`flex items-center gap-3 p-2.5 rounded-xl border cursor-pointer transition-all ${
                          isSel
                            ? 'bg-[var(--accent)]/15 border-[var(--accent)]/60 text-white'
                            : 'bg-white/5 border-white/5 text-white/60 hover:bg-white/10'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isSel}
                          onChange={() => {}} // handled by parent onClick
                          className="rounded border-white/20 bg-black/40 text-[var(--accent)] focus:ring-0"
                        />
                        <img
                          src={targetPreviewUrl(idx)}
                          alt={target.name}
                          className="w-10 h-10 rounded-lg object-cover bg-black/50 shrink-0"
                          onError={(e) => {
                            e.target.style.display = 'none';
                          }}
                        />
                        <div className="min-w-0 flex-1">
                          <span className="text-xs font-medium text-white block truncate">{target.name}</span>
                          <span className="text-micro text-white/40 block">
                            {target.frames || 1} frames · {target.fps ? `${Math.round(target.fps)} fps` : 'image'}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </Section>
      )}

      {/* ── Mode 2 Panel: Grouped Multi-Batch ── */}
      {batchMode === 'grouped' && (
        <Section
          title="Strategy 2: Grouped Multi-Batch (File Groups ➔ Distinct Facesets & Target Faces)"
          action={
            <Button size="sm" variant="secondary" onClick={addGroup}>
              <Icon.add size={14} className="mr-1" />
              Add Group
            </Button>
          }
        >
          <div className="space-y-4">
            {groups.map((grp) => (
              <Card key={grp.id} className="p-4 space-y-3 bg-black/40">
                <div className="flex items-center justify-between pb-2 border-b border-white/10">
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full bg-[var(--accent)]" />
                    <input
                      type="text"
                      value={grp.label}
                      onChange={(e) => updateGroup(grp.id, { label: e.target.value })}
                      className="bg-transparent text-sm font-bold text-white border-b border-transparent hover:border-white/20 focus:border-[var(--accent)] outline-none px-1"
                    />
                  </div>
                  {groups.length > 1 && (
                    <button
                      type="button"
                      onClick={() => removeGroup(grp.id)}
                      className="text-white/40 hover:text-red-400 text-xs font-medium"
                    >
                      Remove Group
                    </button>
                  )}
                </div>

                {/* Multi-Face Mappings for Group */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-micro text-white/60">
                    <span className="font-semibold">Target Person Mappings</span>
                    <button
                      type="button"
                      onClick={() => addGroupMapping(grp.id)}
                      className="text-[var(--accent)] hover:underline"
                    >
                      + Add Mapping
                    </button>
                  </div>
                  {grp.mappings?.map((m, mapIdx) => (
                    <div key={mapIdx} className="grid grid-cols-1 md:grid-cols-2 gap-2 p-2 bg-black/60 rounded-xl border border-white/5 relative group/map">
                      <div>
                        <div className="flex items-center justify-between mb-0.5">
                          <span className="text-nano text-white/40">Target Person</span>
                          {grp.mappings.length > 1 && (
                            <button
                              type="button"
                              onClick={() => removeGroupMapping(grp.id, mapIdx)}
                              className="text-micro text-white/40 hover:text-red-400 font-bold"
                              title="Remove mapping"
                              aria-label={`Remove mapping ${mapIdx + 1}`}
                            >
                              ✕
                            </button>
                          )}
                        </div>
                        <select
                          value={m.personRank}
                          onChange={(e) => updateGroupMapping(grp.id, mapIdx, { personRank: parseInt(e.target.value, 10) })}
                          className="w-full bg-black/60 border border-white/10 rounded-lg px-2 py-1 text-micro text-white"
                        >
                          {targetPersonOptions.map((opt) => (
                            <option key={opt.rank} value={opt.rank}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <span className="text-nano text-white/40 block mb-0.5">Source Faceset</span>
                        <select
                          value={m.sourceIdx}
                          onChange={(e) => updateGroupMapping(grp.id, mapIdx, { sourceIdx: parseInt(e.target.value, 10) })}
                          className="w-full bg-black/60 border border-white/10 rounded-lg px-2 py-1 text-micro text-white"
                        >
                          {sourceFaces.map((_, idx) => (
                            <option key={idx} value={idx}>
                              {sourceFacesInfo[idx]?.name || `Faceset #${idx + 1}`}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Target File Selector for Group */}
                <div>
                  <label className="block text-micro font-semibold text-white/60 mb-1.5">
                    Assigned Target Files ({grp.targetIndices.length} assigned)
                  </label>
                  <div className="flex flex-wrap gap-2 max-h-36 overflow-y-auto p-2 bg-black/30 rounded-xl border border-white/5">
                    {targets.map((target, idx) => {
                      const isAssigned = grp.targetIndices.includes(idx);
                      return (
                        <button
                          key={idx}
                          type="button"
                          onClick={() => {
                            const updated = isAssigned
                              ? grp.targetIndices.filter((i) => i !== idx)
                              : [...grp.targetIndices, idx];
                            updateGroup(grp.id, { targetIndices: updated });
                          }}
                          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-micro transition-all ${
                            isAssigned
                              ? 'bg-[var(--accent)] text-black font-bold'
                              : 'bg-white/5 text-white/60 hover:bg-white/10'
                          }`}
                        >
                          {isAssigned ? '✓ ' : '+ '}
                          {target.name}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </Card>
            ))}

            <Button variant="primary" className="w-full justify-center py-2.5" onClick={generateGroupJobs}>
              <Icon.add size={16} className="mr-1.5" />
              Generate Grouped Batch Jobs
            </Button>
          </div>
        </Section>
      )}

      {/* ── Mode 3 Panel: Per-File Matrix Grid ── */}
      {batchMode === 'matrix' && (
        <Section title="Strategy 3: Per-File Custom Mapping Matrix">
          {/* Toolbar: Search, Filter, Bulk Operations */}
          <div className="p-3.5 bg-white/5 rounded-2xl border border-white/10 space-y-3 mb-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              {/* Search & Filter */}
              <div className="flex flex-wrap items-center gap-2 flex-1">
                <input
                  type="text"
                  placeholder="🔍 Search target files..."
                  value={matrixSearch}
                  onChange={(e) => setMatrixSearch(e.target.value)}
                  className="bg-black/60 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-white placeholder-white/40 focus:border-[var(--accent)] outline-none min-w-[200px]"
                />
                <select
                  value={matrixFilterStatus}
                  onChange={(e) => setMatrixFilterStatus(e.target.value)}
                  className="bg-black/60 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-white"
                >
                  <option value="all">All Targets ({targets.length})</option>
                  <option value="enabled">Enabled Only</option>
                  <option value="disabled">Disabled Only</option>
                </select>
              </div>

              {/* Smart Bulk Tools */}
              <div className="flex flex-wrap gap-2 shrink-0">
                <Button size="sm" variant="primary" onClick={autoMatchFacesetsToTargets}>
                  <Icon.wand className="mr-1" size={13} />
                  Smart Auto-Match Names
                </Button>
                {sourceFaces.length > 0 && (
                  <Button size="sm" variant="secondary" onClick={() => bulkSetMatrixSource(0)}>
                    Set All ➔ Faceset #1
                  </Button>
                )}
                <Button size="sm" variant="secondary" onClick={bulkAutoIncrementTargetPersons}>
                  Auto-Inc Person Ranks
                </Button>
              </div>
            </div>

            {/* Quick Bulk Toggles */}
            <div className="flex items-center justify-between border-t border-white/10 pt-2 text-micro">
              <div className="flex items-center gap-2">
                <span className="text-white/40 font-semibold">Bulk Select:</span>
                <button
                  type="button"
                  onClick={() => bulkToggleMatrixEnable(true)}
                  className="text-[var(--accent)] hover:underline font-medium"
                >
                  Enable All
                </button>
                <span className="text-white/20">|</span>
                <button
                  type="button"
                  onClick={() => bulkToggleMatrixEnable(false)}
                  className="text-white/40 hover:text-white"
                >
                  Disable All
                </button>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-white/40 font-semibold">Bulk Enhancer:</span>
                {bulkEnhancers.map((enh) => (
                  <button
                    key={enh}
                    type="button"
                    onClick={() => bulkSetMatrixEnhancer(enh)}
                    className="text-white/60 hover:text-[var(--accent)] hover:underline"
                  >
                    {enh}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Matrix Table */}
          {filteredMatrixTargets.length === 0 ? (
            <div className="p-8 text-center border border-dashed border-white/10 rounded-xl text-xs text-white/40">
              No target files match the current search or filter criteria.
            </div>
          ) : (
            <div className="space-y-3 max-h-[550px] overflow-y-auto pr-1">
              {filteredMatrixTargets.map(({ target, originalIndex: tIdx }) => {
                const cfg = matrixConfig[tIdx] || {
                  mappings: [{ personRank: 0, sourceIdx: 0 }],
                  swapMode: 'Selected face',
                  enabled: true,
                  enhancer: 'Restoreformer++',
                  faceDistance: 0.75,
                  frameStart: wholeFileRange(target).frame_start,
                  frameEnd: wholeFileRange(target).frame_end,
                };

                return (
                  <div
                    key={tIdx}
                    className={`flex flex-col gap-3 p-3.5 rounded-xl border transition-all ${
                      cfg.enabled
                        ? 'bg-black/40 border-white/10 text-white'
                        : 'bg-white/5 border-white/5 opacity-50 text-white/40'
                    }`}
                  >
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-white/5 pb-2">
                      <div className="flex items-center gap-3 min-w-0">
                        <input
                          type="checkbox"
                          checked={cfg.enabled}
                          onChange={(e) =>
                            setMatrixConfig((prev) => ({
                              ...prev,
                              [tIdx]: { ...cfg, enabled: e.target.checked },
                            }))
                          }
                          className="rounded border-white/20 bg-black/40 text-[var(--accent)] focus:ring-0"
                        />
                        <img
                          src={targetPreviewUrl(tIdx)}
                          alt={target.name}
                          className="w-9 h-9 rounded-lg object-cover bg-black/50 shrink-0"
                          onError={(e) => {
                            e.target.style.display = 'none';
                          }}
                        />
                        <div className="min-w-0">
                          <span className="text-xs font-semibold block truncate" title={target.name}>
                            {tIdx + 1}. {target.name}
                          </span>
                          <span className="text-micro text-white/40 block">
                            {target.frames || 1} frames · {target.fps ? `${Math.round(target.fps)} fps` : 'image'}
                          </span>
                        </div>
                      </div>

                      {/* Trim Segment Range Controls */}
                      <div className="flex items-center gap-2 text-micro shrink-0">
                        <span className="text-white/40 font-medium">Trim:</span>
                        {/* Shown and typed 1-based inclusive, the way the
                            timeline reads. cfg holds the backend's 0-based
                            half-open pair, so only the start shifts: a 1-based
                            inclusive end is already the exclusive end. */}
                        <input
                          type="number"
                          min="1"
                          max={target.frames || 999999}
                          value={(cfg.frameStart ?? 0) + 1}
                          disabled={!cfg.enabled}
                          onChange={(e) =>
                            setMatrixConfig((prev) => ({
                              ...prev,
                              [tIdx]: {
                                ...cfg,
                                frameStart: Math.max(0, (parseInt(e.target.value, 10) || 1) - 1),
                              },
                            }))
                          }
                          className="w-16 bg-black/60 border border-white/10 rounded px-1.5 py-0.5 text-center text-white"
                        />
                        <span className="text-white/30">to</span>
                        <input
                          type="number"
                          min="1"
                          max={target.frames || 999999}
                          value={cfg.frameEnd ?? target.frames ?? 1}
                          disabled={!cfg.enabled}
                          onChange={(e) =>
                            setMatrixConfig((prev) => ({
                              ...prev,
                              [tIdx]: { ...cfg, frameEnd: Math.max(1, parseInt(e.target.value, 10) || 1) },
                            }))
                          }
                          className="w-16 bg-black/60 border border-white/10 rounded px-1.5 py-0.5 text-center text-white"
                        />

                        <button
                          type="button"
                          onClick={() => addMatrixMapping(tIdx)}
                          className="text-micro text-[var(--accent)] hover:underline font-medium ml-2"
                        >
                          + Add Target Person Pair
                        </button>
                      </div>
                    </div>

                    {/* Mappings Grid */}
                    <div className="space-y-2">
                      {cfg.mappings?.map((m, mapIdx) => (
                        <div key={mapIdx} className="grid grid-cols-1 sm:grid-cols-2 gap-2 bg-black/60 p-2 rounded-xl border border-white/5 relative">
                          <div>
                            <div className="flex items-center justify-between mb-0.5">
                              <span className="text-nano text-white/40">Target Person</span>
                              {cfg.mappings.length > 1 && (
                                <button
                                  type="button"
                                  onClick={() => removeMatrixMapping(tIdx, mapIdx)}
                                  className="text-micro text-white/40 hover:text-red-400 font-bold"
                                  title="Remove mapping"
                                  aria-label={`Remove mapping ${mapIdx + 1}`}
                                >
                                  ✕
                                </button>
                              )}
                            </div>
                            <select
                              value={m.personRank}
                              disabled={!cfg.enabled}
                              onChange={(e) => updateMatrixMapping(tIdx, mapIdx, { personRank: parseInt(e.target.value, 10) })}
                              className="w-full bg-black/60 border border-white/10 rounded-lg px-2 py-1 text-micro text-white"
                            >
                              {targetPersonOptions.map((opt) => (
                                <option key={opt.rank} value={opt.rank}>
                                  {opt.label}
                                </option>
                              ))}
                            </select>
                          </div>
                          <div>
                            <span className="text-nano text-white/40 block mb-0.5">Assigned Faceset</span>
                            <select
                              value={m.sourceIdx}
                              disabled={!cfg.enabled}
                              onChange={(e) => updateMatrixMapping(tIdx, mapIdx, { sourceIdx: parseInt(e.target.value, 10) })}
                              className="w-full bg-black/60 border border-white/10 rounded-lg px-2 py-1 text-micro text-white"
                            >
                              {sourceFaces.map((_, sIdx) => (
                                <option key={sIdx} value={sIdx}>
                                  {sourceFacesInfo[sIdx]?.name || `Faceset #${sIdx + 1}`}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          <Button variant="primary" className="w-full justify-center py-2.5 mt-4" onClick={generateMatrixJobs}>
            <Icon.add size={16} className="mr-1.5" />
            Generate Per-File Matrix Batch Jobs
          </Button>
        </Section>
      )}

      {/* ── Mode 4 Panel: Pro Recipes & Video Segment Splitter ── */}
      {batchMode === 'recipes' && (
        <Section title="Strategy 4: Automated Pro Recipes & Video Segment Splitter">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Cartesian Generator */}
            <Card className="p-4 space-y-3 bg-black/40 flex flex-col justify-between">
              <div>
                <h3 className="font-bold text-sm text-white flex items-center gap-2 mb-1">
                  <Icon.batch size={18} className="text-[var(--accent)]" />
                  Cartesian Combinatorial Matrix
                </h3>
                <p className="text-xs text-white/50">
                  Generate every combination of loaded target files and source facesets ({targets.length} targets × {sourceFaces.length} sources = {targets.length * sourceFaces.length} jobs).
                </p>
              </div>

              <Button variant="primary" className="w-full justify-center py-2 text-xs" onClick={recipeCartesianProduct} disabled={targets.length === 0 || sourceFaces.length === 0}>
                ⚡ Generate All Combinations ({targets.length * sourceFaces.length} Jobs)
              </Button>
            </Card>

            {/* Sequential Match Generator */}
            <Card className="p-4 space-y-3 bg-black/40 flex flex-col justify-between">
              <div>
                <h3 className="font-bold text-sm text-white flex items-center gap-2 mb-1">
                  <Icon.split size={18} className="text-[var(--accent)]" />
                  1-to-1 Sequential Pairer
                </h3>
                <p className="text-xs text-white/50">
                  Match Target #1 to Source #1, Target #2 to Source #2... sequentially down the file list.
                </p>
              </div>

              <Button variant="secondary" className="w-full justify-center py-2 text-xs" onClick={recipeSequentialMatch} disabled={targets.length === 0 || sourceFaces.length === 0}>
                🔗 Generate Sequential Pairs ({Math.min(targets.length, sourceFaces.length)} Jobs)
              </Button>
            </Card>

            {/* Multi-Segment Video Splitter */}
            <Card className="p-4 space-y-3 bg-black/40 flex flex-col justify-between">
              <div className="space-y-2">
                <h3 className="font-bold text-sm text-white flex items-center gap-2">
                  <Icon.split size={18} className="text-[var(--accent)]" />
                  Multi-Segment Video Splitter
                </h3>
                <p className="text-xs text-white/50">
                  Split long video renders into N chunk jobs for parallel processing, then auto-join with 1 click.
                </p>

                <div className="space-y-1.5 pt-1">
                  <div>
                    <span className="text-micro text-white/60 block">Target Video</span>
                    <select
                      value={splitTargetIdx}
                      onChange={(e) => setSplitTargetIdx(parseInt(e.target.value, 10))}
                      className="w-full bg-black/60 border border-white/10 rounded-lg px-2 py-1 text-micro text-white"
                    >
                      {targets.map((t, i) => (
                        <option key={i} value={i}>
                          {t.name} ({t.frames || 1} frames)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <span className="text-micro text-white/60 block">Number of Equal Segments</span>
                    <input
                      type="number"
                      min="2"
                      max="32"
                      value={splitSegmentCount}
                      onChange={(e) => setSplitSegmentCount(parseInt(e.target.value, 10) || 2)}
                      className="w-full bg-black/60 border border-white/10 rounded-lg px-2 py-1 text-micro text-white text-center"
                    />
                  </div>
                </div>
              </div>

              <Button variant="primary" className="w-full justify-center py-2 text-xs" onClick={splitTargetIntoSegments} disabled={targets.length === 0}>
                ✂ Split Video into {splitSegmentCount} Segments
              </Button>
            </Card>
          </div>
        </Section>
      )}

      {/* ── Staged Jobs Workbench ── */}
      {stagedJobs.length > 0 && (
        <Section
          title={`Staged Jobs Workbench (${stagedJobs.length} Ready)`}
          action={
            <Button size="sm" variant="ghost" className="text-red-400" onClick={clearStagedJobs}>
              Clear Staged
            </Button>
          }
        >
          <div className="space-y-4">
            {/* Header + Stats + Reorder Toolbar */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 bg-white/5 p-4 rounded-xl border border-white/10">
              <div className="flex flex-col gap-0.5">
                <span className="text-xs font-semibold text-white">
                  Ready to Enqueue ({stagedJobs.length} Jobs)
                </span>
                <span className="text-micro text-emerald-400/90 font-mono">
                  📊 Total {stagedStats.totalFrames.toLocaleString()} frames · Est. ~{stagedStats.timeStr} runtime
                </span>
              </div>

              {/* Sort Dropdown & Enqueue Buttons */}
              <div className="flex flex-wrap items-center gap-2">
                <select
                  onChange={(e) => sortStagedJobs(e.target.value)}
                  className="bg-black/60 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-white"
                  defaultValue=""
                >
                  <option value="" disabled>
                    ⇅ Sort Staged Queue...
                  </option>
                  <option value="shortest">Shortest clips first</option>
                  <option value="longest">Longest clips first</option>
                  <option value="name">Alphabetical by filename</option>
                  <option value="source">Group by source faceset</option>
                </select>

                <Button size="sm" variant="secondary" onClick={() => setShowHealthModal(true)}>
                  🔍 Pre-flight Health Check
                </Button>

                <Button size="sm" variant="secondary" onClick={() => enqueueStagedJobs(false)}>
                  Enqueue to Queue
                </Button>
                <Button size="sm" variant="primary" onClick={() => enqueueStagedJobs(true)}>
                  ▶ Enqueue & Start Batch
                </Button>
              </div>
            </div>

            {/* Visual Pair Cards List */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-80 overflow-y-auto pr-1">
              {stagedJobs.map((job, idx) => (
                <div
                  key={job.id}
                  className="flex items-center gap-3 p-3 rounded-2xl bg-black/60 border border-white/10 relative group hover:border-[var(--accent)]/50 transition-all"
                >
                  {/* Left: Target Media Preview */}
                  <div className="relative shrink-0">
                    <img
                      src={targetPreviewUrl(job.target_index)}
                      alt={job.target_name}
                      className="w-14 h-14 rounded-xl object-cover bg-black/50 border border-white/10"
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                    <span className="absolute -top-1.5 -left-1.5 bg-[var(--accent)] text-black text-nano font-bold px-1.5 py-0.5 rounded-md shadow">
                      #{idx + 1}
                    </span>
                  </div>

                  {/* Center: Pair Arrow */}
                  <div className="flex flex-col items-center justify-center shrink-0 text-white/40">
                    <span className="text-xs font-bold text-[var(--accent)]">➔</span>
                    <span className="text-nano font-mono text-white/30">{job.total_frames || 1}f</span>
                  </div>

                  {/* Right: Source Faceset Preview */}
                  <div className="shrink-0">
                    <img
                      src={sourceFaces[job.source_index]}
                      alt={job.source_name}
                      className="w-14 h-14 rounded-xl object-cover bg-black/50 border border-white/10"
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                  </div>

                  {/* Details */}
                  <div className="min-w-0 flex-1 pl-1">
                    <span className="text-xs font-semibold text-white block truncate" title={job.target_name}>
                      {job.target_name}
                    </span>
                    <span className="text-micro text-white/50 block truncate">
                      Source: {job.source_name}
                    </span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {job.mappings?.map((m, i) => (
                        <span key={i} className="text-nano px-1.5 py-0.5 rounded bg-white/10 text-white/70 font-mono">
                          P#{m.personRank + 1}➔F#{m.sourceIdx + 1}
                        </span>
                      ))}
                      {job.payload?.enhancer && job.payload.enhancer !== 'None' && (
                        <span className="text-nano px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono">
                          {job.payload.enhancer}
                        </span>
                      )}
                      {(job.frame_start > 1 || job.frame_end < job.total_frames) && (
                        <span className="text-nano px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono">
                          f:{job.frame_start}-{job.frame_end}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Delete Button */}
                  <button
                    type="button"
                    onClick={() => removeStagedJob(job.id)}
                    className="text-white/30 hover:text-red-400 p-1 font-bold shrink-0 self-start"
                    title="Remove job"
                    aria-label={`Remove job ${job.label || job.id}`}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        </Section>
      )}

      {/* ── Integrated Server Batch Queue Panel ── */}
      <QueuePanel
        q={queue}
        canAdd={false}
        notify={notify}
        onLoadJobSettings={(job) => {
          notify?.(`Loaded settings from "${job.target_name}"`);
        }}
      />
    </div>
  );
}
