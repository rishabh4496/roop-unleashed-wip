import { useEffect, useRef, useState } from 'react';

// ── One comparison grid's state ───────────────────────────────────────────
// The Face Swap tab can put up to four variants of the same frame side by side
// for four different things: enhancers, mask engines, swapper models and AI
// upscalers. Each of those carried its own copy of the identical eight
// declarations — a `comparing` flag, the chosen model list with its
// localStorage round-trip and validation, a previews map, a times map, a
// per-cell render timer map, and a ref of live intervals.
//
// Four copies of one idea is four places to fix a bug in and four chances for
// them to drift, which had already started: the enhancer grid's timer state is
// still named `liveRenderingTimers` rather than following the other three.
//
// The differences between them turned out to be exactly two — the storage key
// and what counts as a valid selection — so both are parameters and everything
// else is shared.
//
// Returns generic names (`comparing`, `selected`, `previews`…). Call sites
// rename them on destructure to whatever they already called them, which is
// what keeps this a pure move rather than a rewrite of the thousands of lines
// downstream that use those names.

const isString = (x) => typeof x === 'string';

const normalizeSelection = (values, defaults, isValid, allowed) => {
  const allowedSet = Array.isArray(allowed) ? new Set(allowed) : null;
  const seen = new Set();
  const valid = (Array.isArray(values) ? values : [])
    .filter((value) => isValid(value) && (!allowedSet || allowedSet.has(value)))
    .filter((value) => {
      if (seen.has(value)) return false;
      seen.add(value);
      return true;
    })
    .slice(0, 4);
  if (valid.length > 0) return valid;

  const fallback = (Array.isArray(defaults) ? defaults : [])
    .find((value) => isValid(value) && (!allowedSet || allowedSet.has(value)));
  if (fallback !== undefined) return [fallback];

  const catalogFallback = Array.isArray(allowed)
    ? allowed.find((value) => isValid(value))
    : undefined;
  if (catalogFallback !== undefined) return [catalogFallback];

  // A catalog can be empty while metadata is loading. Keep the selection empty
  // in that case; the next catalog update will normalize it again.
  if (allowedSet) return [];
  return (Array.isArray(defaults) ? defaults : [])
    .filter(isValid)
    .filter((value, index, values) => values.indexOf(value) === index)
    .slice(0, 4);
};

/**
 * @param {string}   storageKey  localStorage key holding the chosen list.
 * @param {string[]} defaults    used when nothing valid is stored.
 * @param {(x:any)=>boolean} [isValid]  per-item check; defaults to "a string".
 * @param {string[]} [allowed]         current backend catalog for this grid.
 */
export default function useCompareGrid({ storageKey, defaults, isValid = isString, allowed }) {
  const [comparing, setComparing] = useState(false);
  // Keep the catalog effect stable even when the parent creates metadata
  // arrays while rendering, and rerun it when metadata arrives asynchronously.
  const allowedKey = Array.isArray(allowed) ? allowed.join('\u0001') : '';

  const [selected, setSelected] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
      // 1..4 cells: the grid lays out at most four, and an empty list would
      // render a comparison with nothing in it.
      if (Array.isArray(saved) && saved.length >= 1 && saved.length <= 4 && saved.every(isValid)) {
        return normalizeSelection(saved, defaults, isValid, allowed);
      }
    } catch { /* fall through to default */ }
    return normalizeSelection(defaults, defaults, isValid, allowed);
  });

  // localStorage can outlive a backend catalog after a model is renamed or
  // removed. Previously those stale entries counted toward the four-cell limit
  // while the grid silently filtered them out.
  useEffect(() => {
    if (!Array.isArray(allowed)) return;
    setSelected((prev) => {
      const next = normalizeSelection(prev, defaults, isValid, allowed);
      if (next.length === prev.length && next.every((value, index) => value === prev[index])) return prev;
      return next;
    });
    // The joined catalog is the intentional dependency; `allowed` may be a
    // freshly-created array on every parent render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allowedKey]);

  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify(selected));
  }, [storageKey, selected]);

  // Rendered cell images, their measured cost, and the ticking "rendering for
  // N s" counters — plus the interval handles those counters run on, which are
  // a ref because they are cleared imperatively when a cell resolves.
  const [previews, setPreviews] = useState({});
  const [times, setTimes] = useState({});
  const [timers, setTimers] = useState({});
  const [errors, setErrors] = useState({});
  const intervalsRef = useRef({});

  return {
    comparing, setComparing,
    selected, setSelected,
    previews, setPreviews,
    times, setTimes,
    timers, setTimers,
    errors, setErrors,
    intervalsRef,
  };
}
