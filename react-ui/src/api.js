// Thin client for the FastAPI backend (app/api.py) proxied via Vite.
export const API = window.location.origin;

async function handle(res) {
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).message || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res.text();
}

// Opt-in request deadline. Fetch has no default timeout, so a backend that
// accepts the socket and then stalls (mid-GPU-stall, or killed between accept
// and response) leaves the promise pending forever — a polled request that
// never settles silently stops the poll loop. Callers that must notice a dead
// server pass `timeout`; long-running endpoints (start, auto-capture, upscale)
// deliberately pass nothing and wait as long as it takes.
//
// An explicit `signal` still wins: aborting it aborts the request, and we clear
// our timer either way so no stray abort fires after the response lands.
const withDeadline = (opts = {}) => {
  if (!opts.timeout) return { signal: opts.signal, done: () => {} };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new Error('Request timed out')), opts.timeout);
  opts.signal?.addEventListener('abort', () => ctrl.abort(opts.signal.reason), { once: true });
  return { signal: ctrl.signal, done: () => clearTimeout(timer) };
};

export const getJSON = (path, opts = {}) => {
  const d = withDeadline(opts);
  return fetch(`${API}${path}`, { signal: d.signal }).then(handle).finally(d.done);
};

export const postJSON = (path, body, opts = {}) => {
  const d = withDeadline(opts);
  return fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
    signal: d.signal,
    // keepalive lets the request outlive a page teardown (e.g. Pinokio's
    // Run<->Dev webview reload) so a last-moment flush still reaches the server.
    keepalive: opts.keepalive,
  }).then(handle).finally(d.done);
};

export const postFiles = (path, files, fields) => {
  const fd = new FormData();
  const list = files instanceof FileList ? Array.from(files) : [].concat(files);
  list.forEach((f) => fd.append('files', f));
  if (fields) Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
  return fetch(`${API}${path}`, { method: 'POST', body: fd }).then(handle);
};

export const postFile = (path, file, fields) => {
  const fd = new FormData();
  fd.append('file', file);
  if (fields) Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
  return fetch(`${API}${path}`, { method: 'POST', body: fd }).then(handle);
};

export const fileUrl = (p) => `${API}/api/file?path=${encodeURIComponent(p)}`;
