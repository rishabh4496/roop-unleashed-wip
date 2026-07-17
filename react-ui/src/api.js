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

export const getJSON = (path) => fetch(`${API}${path}`).then(handle);

export const postJSON = (path, body, opts = {}) =>
  fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
    signal: opts.signal,
    // keepalive lets the request outlive a page teardown (e.g. Pinokio's
    // Run<->Dev webview reload) so a last-moment flush still reaches the server.
    keepalive: opts.keepalive,
  }).then(handle);

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
