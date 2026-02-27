/**
 * Voice Deepfake Detector & Generator — Frontend JS
 *
 * Architecture:
 *   - Static site on GitHub Pages (no backend required to load)
 *   - API base URL stored in localStorage; user configures via modal
 *   - "Demo mode" when no backend configured (UI still fully navigable)
 *   - /health checked on load; status bar reflects live state
 *
 * API contract (see backend/app.py):
 *   POST /detect  : FormData{ audio: File } → { prediction, confidence, model_used, notes }
 *   POST /generate: FormData{ audio: File, text: str } → { audio_base64, method_used, notes }
 *   GET  /health  : → { status, model_name, ... }
 */

'use strict';

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const LS_KEY_API = 'deepfake_api_url';
const RESULTS_PATH = '../models/results.json';   // relative from frontend/
const HEALTH_TIMEOUT_MS = 5000;

// ─── STATE ───────────────────────────────────────────────────────────────────
let apiBaseUrl = (localStorage.getItem(LS_KEY_API) || '').replace(/\/$/, '');

// ─── DOM HELPERS ─────────────────────────────────────────────────────────────
const $  = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function showLoading(msg = 'Processing, please wait…') {
  $('#loading-msg').textContent = msg;
  $('#loading-overlay').classList.remove('hidden');
}
function hideLoading() {
  $('#loading-overlay').classList.add('hidden');
}

function setResult(containerId, html, type = 'info') {
  const el = $(`#${containerId}`);
  el.className = `result-box result-${type}`;
  el.innerHTML = html;
  el.classList.remove('hidden');
}

function clearResult(containerId) {
  const el = $(`#${containerId}`);
  el.className = 'result-box hidden';
  el.innerHTML = '';
}

// ─── TAB NAVIGATION ──────────────────────────────────────────────────────────
function activateTab(name) {
  $$('.tab').forEach(t => {
    const active = t.dataset.tab === name;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', active);
  });
  $$('.tab-content').forEach(s => {
    s.classList.toggle('hidden', s.id !== `tab-${name}`);
  });
  window.location.hash = name;
}

$$('.tab').forEach(btn => {
  btn.addEventListener('click', () => activateTab(btn.dataset.tab));
});

// Deep-link support via hash
function handleHash() {
  const hash = window.location.hash.replace('#', '');
  const valid = ['detect', 'generate', 'results', 'about'];
  if (valid.includes(hash)) activateTab(hash);
}
window.addEventListener('hashchange', handleHash);
handleHash();

// In-page tab links
document.addEventListener('click', e => {
  const link = e.target.closest('[data-tab-link]');
  if (link) { e.preventDefault(); activateTab(link.dataset.tabLink); }
});

// ─── ETHICS BANNER ───────────────────────────────────────────────────────────
$('#ethics-dismiss').addEventListener('click', () => {
  $('#ethics-banner').style.display = 'none';
  localStorage.setItem('ethics_dismissed', '1');
});
if (localStorage.getItem('ethics_dismissed')) {
  $('#ethics-banner').style.display = 'none';
}

// ─── API STATUS ───────────────────────────────────────────────────────────────
async function checkApiHealth() {
  const dot  = $('#api-status-indicator');
  const text = $('#api-status-text');

  if (!apiBaseUrl) {
    dot.className = 'status-dot status-demo';
    text.textContent = 'Demo Mode — no backend configured. Click ⚙ to add API URL.';
    updateDemoNotices(true);
    return;
  }

  text.textContent = `Checking ${apiBaseUrl}/health …`;
  dot.className = 'status-dot status-unknown';

  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), HEALTH_TIMEOUT_MS);
    const res = await fetch(`${apiBaseUrl}/health`, { signal: ctrl.signal });
    clearTimeout(timer);

    if (res.ok) {
      const data = await res.json();
      dot.className = 'status-dot status-ok';
      text.textContent = `Backend OK — model: ${data.model_name || 'unknown'}, uptime: ${data.uptime_seconds}s`;
      updateDemoNotices(false);
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    dot.className = 'status-dot status-error';
    text.textContent = `Backend unreachable (${err.message}). Showing Demo Mode.`;
    updateDemoNotices(true);
  }
}

function updateDemoNotices(show) {
  ['detect-demo-notice', 'generate-demo-notice'].forEach(id => {
    const el = $(`#${id}`);
    if (el) el.hidden = !show;
  });
}

// ─── API CONFIG MODAL ─────────────────────────────────────────────────────────
function openConfigModal() {
  const modal = $('#api-config-modal');
  $('#api-url-input').value = apiBaseUrl;
  $('#api-test-result').classList.add('hidden');
  modal.hidden = false;
}
function closeConfigModal() {
  $('#api-config-modal').hidden = true;
}

$('#api-config-btn').addEventListener('click', openConfigModal);
$('#api-cancel-btn').addEventListener('click', closeConfigModal);
$('#api-config-modal').addEventListener('click', e => {
  if (e.target === e.currentTarget) closeConfigModal();
});

// Open config from demo notices
['detect-open-config', 'generate-open-config'].forEach(id => {
  const el = $(`#${id}`);
  if (el) el.addEventListener('click', e => { e.preventDefault(); openConfigModal(); });
});

$('#api-save-btn').addEventListener('click', async () => {
  const raw = $('#api-url-input').value.trim().replace(/\/$/, '');
  const resultEl = $('#api-test-result');
  resultEl.classList.remove('hidden');
  resultEl.textContent = 'Testing connection…';

  if (!raw) {
    localStorage.removeItem(LS_KEY_API);
    apiBaseUrl = '';
    resultEl.textContent = 'Cleared. Running in Demo Mode.';
    setTimeout(() => { closeConfigModal(); checkApiHealth(); }, 1000);
    return;
  }

  try {
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), HEALTH_TIMEOUT_MS);
    const res = await fetch(`${raw}/health`, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    resultEl.textContent = `✓ Connected!\n${JSON.stringify(data, null, 2)}`;
    apiBaseUrl = raw;
    localStorage.setItem(LS_KEY_API, raw);
    setTimeout(() => { closeConfigModal(); checkApiHealth(); }, 1500);
  } catch (err) {
    resultEl.textContent = `✗ Could not connect: ${err.message}\n\nMake sure the backend is running:\n  cd backend\n  uvicorn app:app --reload`;
  }
});

$('#api-clear-btn').addEventListener('click', () => {
  localStorage.removeItem(LS_KEY_API);
  apiBaseUrl = '';
  $('#api-url-input').value = '';
  $('#api-test-result').textContent = 'Cleared. Running in Demo Mode.';
  $('#api-test-result').classList.remove('hidden');
  setTimeout(() => { closeConfigModal(); checkApiHealth(); }, 800);
});

// ─── FILE UPLOAD HELPERS ──────────────────────────────────────────────────────
function setupUploadArea(dropZoneId, fileInputId, filenameId, previewId) {
  const zone  = $(`#${dropZoneId}`);
  const input = $(`#${fileInputId}`);
  const label = $(`#${filenameId}`);
  const preview = previewId ? $(`#${previewId}`) : null;

  function handleFile(file) {
    if (!file) return;
    label.textContent = file.name;
    label.closest('.upload-label').classList.add('has-file');
    if (preview) {
      const url = URL.createObjectURL(file);
      const audio = preview.querySelector('audio');
      if (audio) { audio.src = url; }
      preview.classList.remove('hidden');
    }
  }

  input.addEventListener('change', () => handleFile(input.files[0]));

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
      // Transfer to input
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      handleFile(file);
    }
  });
}

setupUploadArea('detect-drop-zone', 'detect-file', 'detect-filename', 'detect-preview');
setupUploadArea('generate-drop-zone', 'generate-file', 'generate-filename', null);

// ─── DETECT FORM ──────────────────────────────────────────────────────────────
$('#detect-form').addEventListener('submit', async e => {
  e.preventDefault();
  clearResult('detect-result');

  if (!apiBaseUrl) {
    setResult('detect-result',
      '<div class="result-label">Demo Mode</div>' +
      '<p>No backend configured. <a href="#" id="from-detect-config">Click here to add API URL</a> ' +
      'or see the <a href="#results" data-tab-link="results">Model Results</a> tab for pre-computed metrics.</p>',
      'info');
    const link = $('#from-detect-config');
    if (link) link.addEventListener('click', e => { e.preventDefault(); openConfigModal(); });
    return;
  }

  const file = $('#detect-file').files[0];
  if (!file) { alert('Please select a WAV file.'); return; }

  const fd = new FormData();
  fd.append('audio', file, file.name);

  showLoading('Analysing audio…');
  try {
    const res = await fetch(`${apiBaseUrl}/detect`, { method: 'POST', body: fd });
    const data = await res.json();
    hideLoading();

    if (!res.ok) {
      setResult('detect-result',
        `<div class="result-label">Error</div><p>${data.detail || data.error || 'Unknown error'}</p>`,
        'error');
      return;
    }

    const isFake = data.prediction === 'fake' || data.prediction === '1' || data.prediction === 1;
    const conf   = data.confidence != null ? `${(data.confidence * 100).toFixed(1)}%` : 'N/A';
    const type   = isFake ? 'fake' : 'real';
    const label  = isFake ? '⚠ Deepfake Detected' : '✓ Real Voice';

    setResult('detect-result', `
      <div class="result-label">${label}</div>
      <div class="result-confidence">Confidence: <strong>${conf}</strong></div>
      <div class="result-meta">
        Model: ${data.model_used || 'unknown'} &nbsp;|&nbsp;
        Features: ${data.feature_type || 'N/A'} &nbsp;|&nbsp;
        Inference: ${data.inference_time_s != null ? data.inference_time_s + 's' : 'N/A'}
      </div>
      ${data.notes ? `<p style="margin-top:.5rem;font-size:.85rem;">${data.notes}</p>` : ''}
    `, type);

  } catch (err) {
    hideLoading();
    setResult('detect-result',
      `<div class="result-label">Network Error</div><p>${err.message}</p>
       <p class="result-meta">Is the backend running at <code>${apiBaseUrl}</code>?</p>`,
      'error');
  }
});

// ─── GENERATE FORM ────────────────────────────────────────────────────────────
$('#generate-text').addEventListener('input', function() {
  $('#char-count').textContent = `${this.value.length} / 500`;
});

$('#generate-form').addEventListener('submit', async e => {
  e.preventDefault();
  clearResult('generate-result');

  if (!apiBaseUrl) {
    setResult('generate-result',
      '<div class="result-label">Demo Mode</div>' +
      '<p>No backend configured. <a href="#" id="from-gen-config">Click to add API URL</a>.</p>',
      'info');
    const link = $('#from-gen-config');
    if (link) link.addEventListener('click', ev => { ev.preventDefault(); openConfigModal(); });
    return;
  }

  const file = $('#generate-file').files[0];
  const text = $('#generate-text').value.trim();
  const consent = $('#consent-checkbox').checked;

  if (!file) { alert('Please select a speaker WAV file.'); return; }
  if (!text)  { alert('Please enter text to synthesise.'); return; }
  if (!consent) { alert('You must confirm consent before generating.'); return; }

  const fd = new FormData();
  fd.append('audio', file, file.name);
  fd.append('text', text);

  showLoading('Generating voice clone… (may take 30–120 s on CPU)');
  try {
    const res = await fetch(`${apiBaseUrl}/generate`, { method: 'POST', body: fd });
    const data = await res.json();
    hideLoading();

    if (!res.ok) {
      setResult('generate-result',
        `<div class="result-label">Error</div><p>${data.detail || data.error || 'Unknown'}</p>`,
        'error');
      return;
    }

    if (data.audio_base64) {
      const blob = b64ToBlob(data.audio_base64, data.audio_mime || 'audio/wav');
      const url  = URL.createObjectURL(blob);
      const isFallback = data.method_used === 'gtts_fallback';

      setResult('generate-result', `
        <div class="result-label">${isFallback ? '⚠ Fallback TTS (gTTS)' : '✓ Voice Cloned'}</div>
        ${data.notes ? `<p style="font-size:.85rem;margin-bottom:.6rem;">${data.notes}</p>` : ''}
        <div class="generated-audio-box">
          <audio controls src="${url}"></audio>
        </div>
        <div class="result-meta">
          Method: ${data.method_used} &nbsp;|&nbsp;
          Time: ${data.generation_time_s != null ? data.generation_time_s + 's' : 'N/A'}
        </div>
        <a href="${url}" download="generated_voice.wav" class="btn-secondary"
           style="display:inline-flex;margin-top:.6rem;font-size:.8rem;">
          ⬇ Download WAV
        </a>
      `, isFallback ? 'error' : 'real');
    } else {
      setResult('generate-result',
        `<div class="result-label">Error</div><p>No audio returned from server.</p>`,
        'error');
    }

  } catch (err) {
    hideLoading();
    setResult('generate-result',
      `<div class="result-label">Network Error</div><p>${err.message}</p>`,
      'error');
  }
});

function b64ToBlob(b64, mime) {
  const bytes = atob(b64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

// ─── RESULTS TAB ─────────────────────────────────────────────────────────────
async function loadResults() {
  const loading = $('#results-loading');
  const tableWrap = $('#results-table-container');
  const fallback  = $('#results-fallback');
  const tbody     = $('#results-tbody');

  // Try fetching results.json; path is relative to root when deployed
  const paths = [
    './models/results.json',
    '../models/results.json',
    '/Voice-Deepfake-Vishing-Detector-Generator/models/results.json',
  ];

  let data = null;
  for (const p of paths) {
    try {
      const r = await fetch(p);
      if (r.ok) { data = await r.json(); break; }
    } catch (_) { /* try next */ }
  }

  loading.style.display = 'none';

  if (!data) {
    fallback.classList.remove('hidden');
    return;
  }

  const FEATURE_DIMS = { mfcc: 13, fft: 6, hybrid: 19, mfcc_legacy: 18, unknown: '?' };
  let bestF1 = -1;
  Object.values(data).forEach(m => { if (m.f1 > bestF1) bestF1 = m.f1; });

  tbody.innerHTML = '';
  Object.entries(data).forEach(([ft, m]) => {
    const isBest = m.f1 === bestF1;
    const badge = (v, good = .85, warn = .70) =>
      `<span class="metric-badge ${v >= good ? 'badge-green' : v >= warn ? 'badge-yellow' : 'badge-red'}">${(v*100).toFixed(1)}%</span>`;

    const tr = document.createElement('tr');
    if (isBest) tr.classList.add('best-row');
    tr.innerHTML = `
      <td>${isBest ? '⭐ ' : ''}${ft.toUpperCase()}</td>
      <td>${FEATURE_DIMS[ft] || '?'}</td>
      <td>${badge(m.accuracy)}</td>
      <td>${badge(m.precision)}</td>
      <td>${badge(m.recall)}</td>
      <td>${badge(m.f1)}</td>
      <td>${m.inference_1k_ms != null ? m.inference_1k_ms + ' ms' : 'N/A'}</td>
    `;
    tbody.appendChild(tr);
  });

  tableWrap.classList.remove('hidden');
}

// Load results when tab activated or on page load
const origActivate = activateTab;
window._activateTabOrig = activateTab;
function activateTabWithHooks(name) {
  window._activateTabOrig(name);
  if (name === 'results') loadResults();
}
// Rebind tabs
$$('.tab').forEach(btn => {
  btn.replaceWith(btn.cloneNode(true));
});
$$('.tab').forEach(btn => {
  btn.addEventListener('click', () => activateTabWithHooks(btn.dataset.tab));
});

// ─── INIT ─────────────────────────────────────────────────────────────────────
(async () => {
  await checkApiHealth();
  // If results tab active on load
  if (window.location.hash === '#results') loadResults();
})();
