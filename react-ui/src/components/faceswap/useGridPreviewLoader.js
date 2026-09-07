import { useEffect } from 'react';
import { postJSON } from '../../api';
import { runExclusive } from './previewGate';

// ── One comparison grid's preview loader ──────────────────────────────────
// Renders one preview per selected variant, holding every OTHER setting fixed,
// so the grid isolates the effect of the one thing being compared.
//
// There were three of these — enhancers, mask engines, swapper models — and
// they were the same seventy lines three times over. Their own comments said
// so ("Identical shape to loadMaskPreviews/loadEnhancerPreviews"). The only
// real differences were which settings key gets varied, which list of values is
// legal, and which set of state setters to write into.
//
// The AI-upscale grid is deliberately NOT built on this. It is a genuinely
// different shape: it swaps the frame ONCE and then upscales that single result
// with each model (one swap + N upscales, against a different endpoint, with no
// per-cell preview cache), rather than re-running the whole swap per cell.
// Forcing it through here would mean a parameter that switches the body wholesale.
//
// ── Cancellation ──────────────────────────────────────────────────────────
// Cells render one at a time, in sequence, and each is a full swap — so a grid
// of four can be in flight for many seconds. `activeCheck` is consulted before
// and after every await: when the effect is torn down (grid closed, frame
// scrubbed, a setting changed) the in-flight cell finishes but its result is
// dropped rather than written into the state of a grid that has moved on.
//
// ── Serialisation ─────────────────────────────────────────────────────────
// Rendering cells in sequence is not on its own enough, because this loader is
// not the only thing calling /api/preview. Every request goes through
// previewGate so a cell can never overlap the main preview or another surface —
// see previewGate.js for what overlapping actually costs.

const TIMER_TICK_MS = 100;

export default function useGridPreviewLoader({
  enabled,            // the grid's `comparing` flag
  selection,          // the chosen variants, in display order
  allowed,            // values the backend actually supports (undefined = allow all)
  paramKey,           // the settings key this grid varies
  setPreviews, setTimes, setTimers, setErrors, intervalsRef,   // from useCompareGrid
  settings,           // current params; each cell overrides `paramKey` on a copy
  fakePreview,
  selTarget, frame, targetCount,
  buildPreviewPayload, previewSignature, previewCacheRef,
  cacheSuffix,        // the source/target/selection part of the cache key
  reloadKey,          // previewKey — any preview-relevant setting change
}) {
  const allowedKey = Array.isArray(allowed) ? allowed.join('\u0001') : '';
  /* eslint-disable react-hooks/exhaustive-deps -- intentional: `load` is
     rebuilt every render and closes over current values on purpose; the effect
     is keyed on the things that should actually re-render the grid. Depending
     on the closure itself would re-run every render and never settle. */
  useEffect(() => {
    if (!enabled || targetCount === 0) return undefined;
    let active = true;
    const activeCheck = () => active;

    const load = async () => {
      const available = allowed ? selection.filter((v) => allowed.includes(v)) : selection;

      const cacheKeyFor = (value) =>
        `${selTarget}_${frame}_${previewSignature({ ...settings, [paramKey]: value }, fakePreview)}_${cacheSuffix}`;

      // Which cells on screen are already a render of the CURRENT settings.
      // Everything else is dropped, and that distinction is the whole point:
      // deselecting one variant must not blank the others back to a spinner
      // (they are still correct), but a SETTINGS change invalidates all of
      // them — and keeping those on screen left the previous settings' pictures
      // up, with no spinner, while the new ones rendered one at a time behind
      // them. The grid then reads as "nothing happened when I changed
      // anything", which is exactly the failure a comparison grid must not have.
      const fresh = new Set(available.filter((v) => previewCacheRef.current[cacheKeyFor(v)]));
      const keepFresh = (prev) => {
        const reset = {};
        for (const v of available) if (fresh.has(v) && prev[v]) reset[v] = prev[v];
        return reset;
      };
      setPreviews(keepFresh);
      setTimes(keepFresh);
      setTimers(keepFresh);
      setErrors(keepFresh);

      for (const value of available) {
        if (!activeCheck()) return;
        const localParams = { ...settings, [paramKey]: value };
        const cacheKey = cacheKeyFor(value);

        if (previewCacheRef.current[cacheKey]) {
          if (!activeCheck()) return;
          setPreviews((prev) => ({ ...prev, [value]: previewCacheRef.current[cacheKey].image }));
          setTimes((prev) => ({ ...prev, [value]: 'Cached' }));
          setErrors((prev) => ({ ...prev, [value]: null }));
          continue;
        }

        const stopTimer = () => {
          if (intervalsRef.current[value]) {
            clearInterval(intervalsRef.current[value]);
            delete intervalsRef.current[value];
          }
        };

        try {
          const start = Date.now();
          setErrors((prev) => ({ ...prev, [value]: null }));
          setTimers((prev) => ({ ...prev, [value]: '0.0s' }));
          intervalsRef.current[value] = setInterval(() => {
            setTimers((prev) => ({ ...prev, [value]: `${((Date.now() - start) / 1000).toFixed(1)}s` }));
          }, TIMER_TICK_MS);

          const res = await runExclusive(() => postJSON('/api/preview', buildPreviewPayload(localParams, {
            index: selTarget, frame, fake: fakePreview,
          })));
          const duration = ((Date.now() - start) / 1000).toFixed(2);
          stopTimer();
          if (!activeCheck()) return;
          if (res.image) {
            setPreviews((prev) => ({ ...prev, [value]: res.image }));
            setTimes((prev) => ({ ...prev, [value]: `${duration}s` }));
            setTimers((prev) => ({ ...prev, [value]: null }));
            setErrors((prev) => ({ ...prev, [value]: null }));
            previewCacheRef.current[cacheKey] = { faces: res.faces || [], image: res.image };
          } else {
            setTimers((prev) => ({ ...prev, [value]: null }));
            setErrors((prev) => ({
              ...prev,
              [value]: String(res.error || 'No preview image returned'),
            }));
          }
        } catch (error) {
          stopTimer();
          if (!activeCheck()) return;
          setTimers((prev) => ({ ...prev, [value]: null }));
          // Keep the other cells rendering, but expose the failed cell. A
          // swapper model can fail to download on first use, and SAM2-tracked
          // masking may skip a frame; a permanent spinner made both look hung.
          setErrors((prev) => ({
            ...prev,
            [value]: String(error?.message || 'Preview failed for this variant'),
          }));
        }
      }
    };

    load();
    return () => {
      active = false;
      // Timers outlive the request they were measuring if the grid is torn down
      // mid-cell, so clear the whole set rather than the current one.
      if (intervalsRef.current) {
        Object.values(intervalsRef.current).forEach(clearInterval);
        intervalsRef.current = {};
      }
    };
  }, [enabled, selection, frame, selTarget, targetCount, cacheSuffix, reloadKey, allowedKey]);
  /* eslint-enable react-hooks/exhaustive-deps */
}
