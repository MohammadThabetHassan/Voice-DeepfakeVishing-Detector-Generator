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
const LS_KEY_THEME = 'deepfake_theme';
const LS_KEY_HISTORY = 'deepfake_history';
const MAX_HISTORY_ITEMS = 10;
const RESULTS_PATH = '../models/results.json';   // relative from frontend/
const HEALTH_TIMEOUT_MS = 5000;

// ─── STATE ───────────────────────────────────────────────────────────────────
let apiBaseUrl = (localStorage.getItem(LS_KEY_API) || '').replace(/\/$/, '');

// Recording state
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = 0;
let recordingTimerInterval = null;
let isRecording = false;
let audioContext = null;

// Focus management state
let lastFocusedElement = null;
let modalFocusTrap = null;

// ─── ACCESSIBILITY UTILITIES ─────────────────────────────────────────────────

/**
 * Announce a message to screen readers via the aria-live region
 */
function announceToScreenReader(message, priority = 'polite') {
  const announcer = document.getElementById('sr-announcer');
  if (!announcer) return;

  // Clear previous content to ensure announcement
  announcer.setAttribute('aria-live', 'off');
  announcer.textContent = '';

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      announcer.setAttribute('aria-live', priority);
      announcer.textContent = message;
    });
  });
}

/**
 * Save the currently focused element before opening a modal
 */
function saveFocus() {
  lastFocusedElement = document.activeElement;
}

/**
 * Restore focus to the element that triggered the modal
 */
function restoreFocus() {
  if (lastFocusedElement && lastFocusedElement.focus) {
    lastFocusedElement.focus();
    lastFocusedElement = null;
  }
}

/**
 * Get all focusable elements within a container
 */
function getFocusableElements(container) {
  const selector = [
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    'a[href]',
    '[tabindex]:not([tabindex="-1"])',
    '[contenteditable]'
  ].join(', ');

  return [...container.querySelectorAll(selector)].filter(el => {
    return el.offsetParent !== null && !el.hasAttribute('hidden');
  });
}

/**
 * Trap focus within a modal/container
 */
function trapFocus(container) {
  const focusableElements = getFocusableElements(container);
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];

  function handleTabKey(e) {
    if (e.key !== 'Tab') return;

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      }
    } else {
      if (document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    }
  }

  container.addEventListener('keydown', handleTabKey);

  // Focus first element
  if (firstElement) {
    firstElement.focus();
  }

  return () => container.removeEventListener('keydown', handleTabKey);
}

/**
 * Handle Escape key to close modals
 */
function handleEscapeKey(e) {
  if (e.key === 'Escape') {
    const modal = document.querySelector('.modal:not([hidden])');
    const overlay = document.querySelector('.overlay:not(.hidden)');

    if (modal) {
      e.preventDefault();
      closeConfigModal();
    } else if (overlay && overlay.id === 'loading-overlay') {
      // Don't close loading overlay with Escape
      return;
    }
  }
}

/**
 * Handle keyboard shortcuts
 */
function handleKeyboardShortcuts(e) {
  // Only handle if not in an input/textarea
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
    return;
  }

  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  const ctrlOrCmd = isMac ? e.metaKey : e.ctrlKey;

  if (ctrlOrCmd) {
    switch (e.key.toLowerCase()) {
      case 'd':
        e.preventDefault();
        activateTab('detect');
        $('#tab-detect')?.focus();
        announceToScreenReader('Switched to Detect Deepfake tab');
        break;
      case 'g':
        e.preventDefault();
        activateTab('generate');
        $('#tab-generate')?.focus();
        announceToScreenReader('Switched to Generate Clone tab');
        break;
      case 'r':
        e.preventDefault();
        activateTab('results');
        $('#tab-results')?.focus();
        announceToScreenReader('Switched to Model Results tab');
        break;
      case 'u':
        e.preventDefault();
        const activeTab = document.querySelector('.tab-content:not(.hidden)');
        const fileInput = activeTab?.querySelector('input[type="file"]');
        if (fileInput) {
          fileInput.click();
          announceToScreenReader('File upload dialog opened');
        }
        break;
    }
  }
}

// ─── DOM HELPERS ─────────────────────────────────────────────────────────────
const $  = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function showLoading(msg = 'Processing, please wait…') {
  $('#loading-msg').textContent = msg;
  const overlay = $('#loading-overlay');
  overlay.classList.remove('hidden');
  overlay.setAttribute('aria-hidden', 'false');
  saveFocus();

  // Update progressbar
  const spinner = overlay.querySelector('[role="progressbar"]');
  if (spinner) {
    spinner.setAttribute('aria-valuenow', '50');
  }

  announceToScreenReader(msg, 'polite');
}
function hideLoading() {
  const overlay = $('#loading-overlay');
  overlay.classList.add('hidden');
  overlay.setAttribute('aria-hidden', 'true');

  // Reset progressbar
  const spinner = overlay.querySelector('[role="progressbar"]');
  if (spinner) {
    spinner.setAttribute('aria-valuenow', '0');
  }

  restoreFocus();
}

function setResult(containerId, html, type = 'info') {
  const el = $(`#${containerId}`);
  el.className = `result-box result-${type}`;
  el.innerHTML = html;
  el.classList.remove('hidden');

  // Move focus to result for screen readers
  el.setAttribute('tabindex', '-1');
  el.focus();

  // Announce to screen reader
  const typeLabels = {
    'info': 'Information',
    'success': 'Success',
    'error': 'Error',
    'real': 'Real voice detected',
    'fake': 'Deepfake detected',
    'uncertain': 'Detection uncertain'
  };
  const label = typeLabels[type] || 'Result';

  // Extract text content for announcement
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = html;
  const textContent = tempDiv.textContent || tempDiv.innerText || '';
  announceToScreenReader(`${label}: ${textContent.substring(0, 200)}`, 'polite');
}

function clearResult(containerId) {
  const el = $(`#${containerId}`);
  el.className = 'result-box hidden';
  el.innerHTML = '';
  el.removeAttribute('tabindex');
}

// ─── TAB NAVIGATION ──────────────────────────────────────────────────────────
function activateTab(name) {
  const tabs = $$('.tab');
  const panels = $$('.tab-content');

  tabs.forEach(t => {
    const active = t.dataset.tab === name;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', active);
    t.setAttribute('tabindex', active ? '0' : '-1');
  });

  panels.forEach(s => {
    const isActive = s.id === `tab-${name}`;
    s.classList.toggle('hidden', !isActive);
    s.setAttribute('aria-hidden', !isActive);
  });

  window.location.hash = name;
}

// Tab keyboard navigation
$$('.tab').forEach((btn, index, allTabs) => {
  btn.addEventListener('click', () => activateTab(btn.dataset.tab));

  btn.addEventListener('keydown', (e) => {
    let newIndex = index;

    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        e.preventDefault();
        newIndex = (index + 1) % allTabs.length;
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        e.preventDefault();
        newIndex = (index - 1 + allTabs.length) % allTabs.length;
        break;
      case 'Home':
        e.preventDefault();
        newIndex = 0;
        break;
      case 'End':
        e.preventDefault();
        newIndex = allTabs.length - 1;
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        activateTab(btn.dataset.tab);
        return;
    }

    if (newIndex !== index) {
      allTabs[newIndex].focus();
      activateTab(allTabs[newIndex].dataset.tab);
    }
  });
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
    _updateEngineLabel(null);
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
      const engineLabel = data.tts_engine === 'indextts2' ? 'IndexTTS2'
        : data.tts_engine === 'gtts_fallback' ? 'gTTS (fallback)'
        : 'none';
      text.textContent = (
        `Backend OK — detector: ${data.model_name || 'none'} · ` +
        `TTS: ${engineLabel} · uptime: ${data.uptime_seconds}s`
      );
      updateDemoNotices(false);
      _updateEngineLabel(data);
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    dot.className = 'status-dot status-error';
    text.textContent = `Backend unreachable (${err.message}). Showing Demo Mode.`;
    updateDemoNotices(true);
    _updateEngineLabel(null);
  }
}

function _updateEngineLabel(health) {
  const label = $('#tts-engine-label');
  if (!label) return;
  if (!health) {
    label.textContent = 'No backend connected';
    label.style.color = 'var(--text-muted)';
    return;
  }
  if (health.indextts2_available) {
    label.textContent = '✓ IndexTTS2 (zero-shot voice cloning)';
    label.style.color = 'var(--brand-green)';
  } else if (health.gtts_available) {
    label.textContent = '⚠ gTTS fallback (generic voice, NOT a clone)';
    label.style.color = 'var(--brand-yellow)';
  } else {
    label.textContent = '✗ No TTS engine available';
    label.style.color = 'var(--brand-red)';
  }
}

function updateDemoNotices(show) {
  ['detect-demo-notice', 'generate-demo-notice'].forEach(id => {
    const el = $(`#${id}`);
    if (el) el.hidden = !show;
  });
}

// ─── API CONFIG MODAL ─────────────────────────────────────────────────────────
let modalCleanup = null;

function openConfigModal() {
  saveFocus();
  const modal = $('#api-config-modal');
  $('#api-url-input').value = apiBaseUrl;
  $('#api-test-result').classList.add('hidden');
  modal.hidden = false;
  modal.setAttribute('aria-hidden', 'false');

  // Trap focus in modal
  modalCleanup = trapFocus(modal);

  announceToScreenReader('Backend configuration dialog opened');
}

function closeConfigModal() {
  const modal = $('#api-config-modal');
  modal.hidden = true;
  modal.setAttribute('aria-hidden', 'true');

  // Remove focus trap
  if (modalCleanup) {
    modalCleanup();
    modalCleanup = null;
  }

  restoreFocus();
  announceToScreenReader('Configuration dialog closed');
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

// ─── WAVEFORM VISUALIZATION ───────────────────────────────────────────────────
function drawWaveform(audioBuffer, canvas) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width = canvas.clientWidth;
  const height = canvas.height = canvas.clientHeight;
  const data = audioBuffer.getChannelData(0);
  const step = Math.ceil(data.length / width);
  const amp = height / 2;

  ctx.clearRect(0, 0, width, height);
  ctx.beginPath();
  ctx.strokeStyle = '#4f46e5';
  ctx.lineWidth = 2;

  for (let x = 0; x < width; x++) {
    const start = x * step;
    const end = Math.min(start + step, data.length);
    let min = 1, max = -1;
    for (let i = start; i < end; i++) {
      const val = data[i];
      if (val < min) min = val;
      if (val > max) max = val;
    }
    const y1 = amp + min * amp;
    const y2 = amp + max * amp;
    if (x === 0) {
      ctx.moveTo(x, y1);
      ctx.lineTo(x, y2);
    } else {
      ctx.lineTo(x, y1);
      ctx.lineTo(x, y2);
    }
  }
  ctx.stroke();
}

async function loadAndDrawWaveform(file) {
  const canvas = $('#waveform-canvas');
  if (!canvas) return;

  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await ac.decodeAudioData(arrayBuffer);
    drawWaveform(audioBuffer, canvas);
    ac.close();
  } catch (err) {
    console.error('Failed to draw waveform:', err);
  }
}

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
      preview.setAttribute('aria-hidden', 'false');
      loadAndDrawWaveform(file);
    }
    announceToScreenReader(`File selected: ${file.name}`);
  }

  input.addEventListener('change', () => {
    handleFile(input.files[0]);
  });

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

  // Keyboard support for drop zone
  zone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      input.click();
    }
  });

  // Make zone focusable
  zone.setAttribute('tabindex', '0');
}

setupUploadArea('detect-drop-zone', 'detect-file', 'detect-filename', 'detect-preview');
setupUploadArea('generate-drop-zone', 'generate-file', 'generate-filename', null);

// ─── MICROPHONE RECORDING ─────────────────────────────────────────────────────
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
  const secs = (seconds % 60).toString().padStart(2, '0');
  return `${mins}:${secs}`;
}

function updateRecordingTimer() {
  const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
  const timerEl = $('#recording-timer');
  if (timerEl) timerEl.textContent = formatTime(elapsed);
}

function updateRecordingUI(recording) {
  const recordBtn = $('#record-btn');
  const stopBtn = $('#stop-record-btn');
  const statusEl = $('#recording-status');
  const timerEl = $('#recording-timer');

  if (recording) {
    if (recordBtn) recordBtn.classList.add('hidden');
    if (stopBtn) stopBtn.classList.remove('hidden');
    if (statusEl) statusEl.classList.remove('hidden');
    if (timerEl) timerEl.classList.remove('hidden');
  } else {
    if (recordBtn) recordBtn.classList.remove('hidden');
    if (stopBtn) stopBtn.classList.add('hidden');
    if (statusEl) statusEl.classList.add('hidden');
    if (timerEl) {
      timerEl.classList.add('hidden');
      timerEl.textContent = '00:00';
    }
  }
}

async function startRecording() {
  try {
    announceToScreenReader('Starting audio recording. Please speak now.', 'assertive');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Try to use audio/webm codec, fallback to default
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : 'audio/ogg';

    mediaRecorder = new MediaRecorder(stream, { mimeType });
    recordedChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      announceToScreenReader('Recording stopped. Processing audio...', 'polite');
      const audioBlob = new Blob(recordedChunks, { type: mimeType });
      await processRecordedAudio(audioBlob);

      // Stop all tracks to release microphone
      stream.getTracks().forEach(track => track.stop());
    };

    mediaRecorder.onerror = (e) => {
      console.error('MediaRecorder error:', e);
      announceToScreenReader('Recording error occurred. Please try again.', 'assertive');
      alert('Recording error occurred. Please try again.');
      stopRecording();
    };

    mediaRecorder.start(100); // Collect data every 100ms
    isRecording = true;
    recordingStartTime = Date.now();
    recordingTimerInterval = setInterval(updateRecordingTimer, 1000);
    updateRecordingUI(true);

    announceToScreenReader('Recording started. Press stop recording button when finished.', 'polite');

  } catch (err) {
    console.error('Error accessing microphone:', err);
    let errorMsg = '';
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      errorMsg = 'Microphone permission denied. Please allow access to record audio.';
    } else if (err.name === 'NotFoundError') {
      errorMsg = 'No microphone found. Please connect a microphone and try again.';
    } else {
      errorMsg = `Could not access microphone: ${err.message}`;
    }
    announceToScreenReader(errorMsg, 'assertive');
    alert(errorMsg);
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    clearInterval(recordingTimerInterval);
    updateRecordingUI(false);
  }
}

async function processRecordedAudio(audioBlob) {
  showLoading('Converting to WAV…');
  try {
    const wavBlob = await convertToWav(audioBlob);

    // Create a File object from the Blob
    const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });

    // Populate the file input
    const fileInput = $('#detect-file');
    const dt = new DataTransfer();
    dt.items.add(wavFile);
    fileInput.files = dt.files;

    // Update UI to show the file
    const label = $('#detect-filename');
    label.textContent = wavFile.name;
    label.closest('.upload-label').classList.add('has-file');

    // Update audio preview
    const preview = $('#detect-preview');
    const audioPlayer = preview.querySelector('audio');
    if (audioPlayer) {
      const url = URL.createObjectURL(wavFile);
      audioPlayer.src = url;
      preview.classList.remove('hidden');
    }

    // Draw waveform for recorded audio
    loadAndDrawWaveform(wavFile);

    hideLoading();
  } catch (err) {
    hideLoading();
    console.error('Error converting audio:', err);
    alert(`Failed to convert audio: ${err.message}`);
  }
}

async function convertToWav(audioBlob) {
  // Create audio context
  audioContext = new (window.AudioContext || window.webkitAudioContext)();

  // Decode the audio data
  const arrayBuffer = await audioBlob.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  // Resample to 16kHz mono
  const targetSampleRate = 16000;
  const numberOfChannels = 1;
  const offlineContext = new OfflineAudioContext(
    numberOfChannels,
    audioBuffer.duration * targetSampleRate,
    targetSampleRate
  );

  const source = offlineContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineContext.destination);
  source.start(0);

  const resampledBuffer = await offlineContext.startRendering();

  // Convert to 16-bit PCM WAV
  return audioBufferToWav(resampledBuffer);
}

function audioBufferToWav(audioBuffer) {
  const numberOfChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;

  const samples = audioBuffer.getChannelData(0); // Mono
  const dataLength = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  // Write WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  // RIFF chunk
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, 'WAVE');

  // fmt chunk
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numberOfChannels * bytesPerSample, true);
  view.setUint16(32, numberOfChannels * bytesPerSample, true);
  view.setUint16(34, bitDepth, true);

  // data chunk
  writeString(36, 'data');
  view.setUint32(40, dataLength, true);

  // Write samples (16-bit PCM)
  const offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset + i * 2, sample * 0x7FFF, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

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
  if (!file) {
    announceToScreenReader('Please select a WAV file before submitting.', 'assertive');
    alert('Please select a WAV file.');
    $('#detect-file').focus();
    return;
  }

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

    const prediction = String(data.prediction ?? '').toLowerCase();
    const basePrediction = String(data.base_prediction ?? data.prediction ?? '').toLowerCase();
    const isUncertain = Boolean(data.is_uncertain) || prediction === 'uncertain';
    const isFake = !isUncertain && (basePrediction === 'fake' || basePrediction === '1');
    const conf = data.confidence != null ? `${(data.confidence * 100).toFixed(1)}%` : 'N/A';
    const fakeProb = data.fake_probability != null ? `${(data.fake_probability * 100).toFixed(1)}%` : 'N/A';
    const threshold = data.threshold != null ? `${(data.threshold * 100).toFixed(1)}%` : 'N/A';
    const qualityScore = data.quality_score != null ? `${(data.quality_score * 100).toFixed(1)}%` : 'N/A';
    const qualityWarnings = Array.isArray(data.quality_warnings) ? data.quality_warnings : [];

    let type = 'real';
    let label = '✓ Real Voice';
    if (isUncertain) {
      type = 'uncertain';
      label = '⚖ Uncertain Result';
    } else if (isFake) {
      type = 'fake';
      label = '⚠ Deepfake Detected';
    }

    setResult('detect-result', `
      <div class="result-label">${label}</div>
      <div class="result-confidence">Confidence: <strong>${conf}</strong></div>
      <div class="result-meta">
        Decision: ${isUncertain ? 'uncertain' : (isFake ? 'fake' : 'real')} &nbsp;|&nbsp;
        Fake probability: ${fakeProb} &nbsp;|&nbsp;
        Threshold: ${threshold}${data.threshold_profile ? ` (${data.threshold_profile})` : ''}
      </div>
      <div class="result-meta">
        Model: ${data.model_used || 'unknown'} &nbsp;|&nbsp;
        Features: ${data.feature_type || 'N/A'} &nbsp;|&nbsp;
        Windows: ${data.windows_analyzed ?? 'N/A'} &nbsp;|&nbsp;
        Inference: ${data.inference_time_s != null ? data.inference_time_s + 's' : 'N/A'}
      </div>
      <div class="result-meta">
        Quality score: ${qualityScore}${qualityWarnings.length ? ` &nbsp;|&nbsp; Warnings: ${qualityWarnings.join(', ')}` : ''}
      </div>
      ${data.notes ? `<p style="margin-top:.5rem;font-size:.85rem;">${data.notes}</p>` : ''}
    `, type);

    saveDetectionResult(data, file.name);

    // Additional screen reader announcement for the result
    const resultMsg = isUncertain
      ? `Detection is uncertain with ${conf} confidence. Manual review recommended.`
      : isFake
        ? `Deepfake detected with ${conf} confidence`
        : `Real voice detected with ${conf} confidence`;
    announceToScreenReader(resultMsg, 'assertive');

  } catch (err) {
    hideLoading();
    setResult('detect-result',
      `<div class="result-label">Network Error</div><p>${err.message}</p>
       <p class="result-meta">Is the backend running at <code>${apiBaseUrl}</code>?</p>`,
      'error');
    announceToScreenReader('Network error. Please check your connection and try again.', 'assertive');
  }
});

// ─── GENERATE FORM ────────────────────────────────────────────────────────────
$('#generate-text').addEventListener('input', function() {
  const count = this.value.length;
  $('#char-count').textContent = `${count} / 500`;
  // Announce when approaching limit
  if (count === 490) {
    announceToScreenReader('10 characters remaining', 'polite');
  } else if (count === 500) {
    announceToScreenReader('Maximum character limit reached', 'polite');
  }
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

  if (!file) {
    announceToScreenReader('Please select a speaker WAV file.', 'assertive');
    alert('Please select a speaker WAV file.');
    $('#generate-file').focus();
    return;
  }
  if (!text) {
    announceToScreenReader('Please enter text to synthesise.', 'assertive');
    alert('Please enter text to synthesise.');
    $('#generate-text').focus();
    return;
  }
  if (!consent) {
    announceToScreenReader('You must confirm consent before generating.', 'assertive');
    alert('You must confirm consent before generating.');
    $('#consent-checkbox').focus();
    return;
  }

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
           style="display:inline-flex;margin-top:.6rem;font-size:.8rem;"
           aria-label="Download generated voice as WAV file">
          ⬇ Download WAV
        </a>
      `, isFallback ? 'error' : 'real');

      // Announce result
      const resultMsg = isFallback
        ? 'Voice generation complete using fallback TTS. Note: This is not a voice clone.'
        : 'Voice clone generated successfully';
      announceToScreenReader(resultMsg, 'assertive');
    } else {
      setResult('generate-result',
        `<div class="result-label">Error</div><p>No audio returned from server.</p>`,
        'error');
      announceToScreenReader('Error: No audio returned from server.', 'assertive');
    }

  } catch (err) {
    hideLoading();
    setResult('generate-result',
      `<div class="result-label">Network Error</div><p>${err.message}</p>`,
      'error');
    announceToScreenReader('Network error during voice generation. Please try again.', 'assertive');
  }
});

function b64ToBlob(b64, mime) {
  const bytes = atob(b64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

// ─── HISTORY MANAGEMENT ──────────────────────────────────────────────────────
function formatTimestamp(date) {
  const d = new Date(date);
  const now = new Date();
  const isToday = d.toDateString() === now.toDateString();
  const timeStr = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

  if (isToday) {
    return timeStr;
  }
  return `${d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} ${timeStr}`;
}

function saveDetectionResult(result, filename) {
  const history = loadHistory();
  const prediction = String(result.prediction ?? '').toLowerCase();
  const basePrediction = String(result.base_prediction ?? result.prediction ?? '').toLowerCase();
  const isUncertain = Boolean(result.is_uncertain) || prediction === 'uncertain';
  const entry = {
    timestamp: Date.now(),
    filename: filename || 'Unknown',
    prediction: isUncertain ? 'uncertain' : basePrediction,
    base_prediction: basePrediction,
    is_uncertain: isUncertain,
    confidence: result.confidence,
    fake_probability: result.fake_probability,
    threshold: result.threshold,
    quality_score: result.quality_score,
    model_used: result.model_used || 'unknown'
  };

  history.unshift(entry);

  if (history.length > MAX_HISTORY_ITEMS) {
    history.pop();
  }

  localStorage.setItem(LS_KEY_HISTORY, JSON.stringify(history));
  renderHistory();
}

function loadHistory() {
  try {
    const data = localStorage.getItem(LS_KEY_HISTORY);
    return data ? JSON.parse(data) : [];
  } catch (e) {
    console.error('Failed to load history:', e);
    return [];
  }
}

function clearHistory() {
  localStorage.removeItem(LS_KEY_HISTORY);
  renderHistory();
}

function renderHistory() {
  const history = loadHistory();
  const container = $('#history-list');

  if (!container) return;

  if (history.length === 0) {
    container.innerHTML = '<p class="history-empty">No detections yet. Upload an audio file to get started.</p>';
    return;
  }

  container.innerHTML = history.map(item => {
    const pred = String(item.prediction ?? '').toLowerCase();
    const isUncertain = Boolean(item.is_uncertain) || pred === 'uncertain';
    const isFake = !isUncertain && (pred === 'fake' || pred === '1');
    const badgeClass = isUncertain ? 'uncertain' : (isFake ? 'fake' : 'real');
    const badgeText = isUncertain ? 'Uncertain' : (isFake ? 'Deepfake' : 'Real');
    const confidence = item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : 'N/A';
    const fakeProb = item.fake_probability != null ? `${(item.fake_probability * 100).toFixed(1)}%` : null;
    const threshold = item.threshold != null ? `${(item.threshold * 100).toFixed(1)}%` : null;
    const scoreText = fakeProb && threshold ? `${fakeProb} / th ${threshold}` : confidence;

    return `
      <div class="history-item">
        <span class="history-timestamp">${formatTimestamp(item.timestamp)}</span>
        <span class="history-filename" title="${item.filename}">${item.filename}</span>
        <span class="history-badge ${badgeClass}">${badgeText}</span>
        <span class="history-confidence">${scoreText}</span>
        <span class="history-model">${item.model_used}</span>
      </div>
    `;
  }).join('');
}

// ─── RESULTS TAB ─────────────────────────────────────────────────────────────
async function loadResults() {
  const loading = $('#results-loading');
  const tableWrap = $('#results-table-container');
  const fallback  = $('#results-fallback');
  const tbody     = $('#results-tbody');

  // Try fetching results.json; path is relative to root when deployed
  const paths = [
    ...(apiBaseUrl ? [`${apiBaseUrl}/model-results`] : []),
    './models/results.json',
    '../models/results.json',
    '/Voice-Deepfake-Vishing-Detector-Generator/models/results.json',
    'https://mohammadthabethassan.github.io/Voice-Deepfake-Vishing-Detector-Generator/models/results.json',
  ];

  let data = null;
  let lastError = null;
  for (const p of paths) {
    try {
      const r = await fetch(p);
      if (r.ok) { 
        data = await r.json(); 
        console.log('Results loaded from:', p);
        break; 
      }
    } catch (e) { 
      lastError = e;
      console.log('Failed to load from:', p, e.message);
    }
  }
  if (!data && lastError) {
    console.error('All paths failed to load results.json');
  }

  loading.style.display = 'none';

  if (!data) {
    fallback.classList.remove('hidden');
    return;
  }

  const FEATURE_DIMS = {
    mfcc: 13,
    fft: 6,
    hybrid: 19,
    enhanced: 75,
    mfcc_legacy: 18,
    ensemble: 'Varies',
    unknown: '?'
  };
  const inferBaseFeatureType = (modelKey) => {
    const k = String(modelKey || '').toLowerCase();
    if (k.startsWith('mfcc_legacy')) return 'mfcc_legacy';
    if (k.startsWith('mfcc')) return 'mfcc';
    if (k.startsWith('fft')) return 'fft';
    if (k.startsWith('hybrid') || k.startsWith('mfcc_hybrid')) return 'hybrid';
    if (k.startsWith('enhanced')) return 'enhanced';
    if (k.startsWith('ensemble')) return 'ensemble';
    return 'unknown';
  };
  let bestF1 = -1;
  Object.values(data).forEach(m => { if (m.f1 > bestF1) bestF1 = m.f1; });

  tbody.innerHTML = '';
  Object.entries(data).forEach(([ft, m]) => {
    const baseFt = inferBaseFeatureType(ft);
    const dims = FEATURE_DIMS[baseFt] ?? '?';
    const isBest = m.f1 === bestF1;
    const badge = (v, good = .85, warn = .70) =>
      `<span class="metric-badge ${v >= good ? 'badge-green' : v >= warn ? 'badge-yellow' : 'badge-red'}">${(v*100).toFixed(1)}%</span>`;

    const tr = document.createElement('tr');
    if (isBest) tr.classList.add('best-row');
    tr.innerHTML = `
      <td>${isBest ? '⭐ ' : ''}${ft.toUpperCase()}</td>
      <td>${dims}</td>
      <td>${badge(m.accuracy)}</td>
      <td>${badge(m.precision)}</td>
      <td>${badge(m.recall)}</td>
      <td>${badge(m.f1)}</td>
      <td>${m.inference_1k_ms != null ? m.inference_1k_ms + ' ms' : 'N/A'}</td>
    `;
    tbody.appendChild(tr);
  });

  tableWrap.classList.remove('hidden');

  // Render confusion matrices
  const cmContainer = $('#confusion-matrices-container');
  const cmGrid = $('#confusion-grid');
  if (cmContainer && cmGrid) {
    cmGrid.innerHTML = '';
    Object.entries(data).forEach(([ft, m]) => {
      if (!m.confusion_matrix) return;
      const isBest = m.f1 === bestF1;
      const cm = m.confusion_matrix;
      // cm format: [[TN, FP], [FN, TP]]
      const [[tn, fp], [fn, tp]] = cm;
      const maxCell = Math.max(tn, fp, fn, tp, 1);
      const realTotal = Math.max(1, tn + fp);
      const fakeTotal = Math.max(1, fn + tp);
      const pct = (value, total) => `${((value / total) * 100).toFixed(1)}%`;
      const cellBg = (value, correct) => {
        const intensity = 0.12 + 0.78 * (value / maxCell);
        const rgb = correct ? '22,163,74' : '220,38,38';
        return `rgba(${rgb}, ${Math.min(0.9, intensity).toFixed(3)})`;
      };
      const cell = (klass, title, value, total, correct) => `
        <td class="${klass} heat-cell" title="${title}" style="background:${cellBg(value, correct)}">
          <span class="cm-count">${value}</span>
          <span class="cm-rate">${pct(value, total)}</span>
        </td>
      `;

      const div = document.createElement('div');
      div.className = 'confusion-matrix';
      div.innerHTML = `
        <h4>${ft.toUpperCase()}${isBest ? ' <span class="best-badge">⭐ Best</span>' : ''}</h4>
        <table class="confusion-table" role="table" aria-label="Confusion matrix for ${ft} model">
          <thead>
            <tr>
              <th></th>
              <th scope="col">Predicted Real</th>
              <th scope="col">Predicted Fake</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row" class="label-cell">Actual Real</th>
              ${cell('tn', 'True Negative', tn, realTotal, true)}
              ${cell('fp', 'False Positive', fp, realTotal, false)}
            </tr>
            <tr>
              <th scope="row" class="label-cell">Actual Fake</th>
              ${cell('fn', 'False Negative', fn, fakeTotal, false)}
              ${cell('tp', 'True Positive', tp, fakeTotal, true)}
            </tr>
          </tbody>
        </table>
        <div class="confusion-legend">
          <span><span class="dot correct"></span> Correct</span>
          <span><span class="dot incorrect"></span> Incorrect</span>
        </div>
        <div class="heat-scale">
          <span>Low</span>
          <span class="heat-scale-bar"></span>
          <span>High</span>
        </div>
      `;
      cmGrid.appendChild(div);
    });
    cmContainer.classList.remove('hidden');
  }
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

// ─── RECORDING BUTTON EVENT LISTENERS ─────────────────────────────────────────
$('#record-btn')?.addEventListener('click', startRecording);
$('#stop-record-btn')?.addEventListener('click', stopRecording);

// ─── THEME MANAGEMENT ─────────────────────────────────────────────────────────
function initTheme() {
  const savedTheme = localStorage.getItem(LS_KEY_THEME);
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = savedTheme || (prefersDark ? 'dark' : 'light');
  setTheme(theme);
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(LS_KEY_THEME, theme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  setTheme(newTheme);
}

// ─── KEYBOARD SHORTCUTS & ACCESSIBILITY INIT ───────────────────────────────────
document.addEventListener('keydown', handleEscapeKey);
document.addEventListener('keydown', handleKeyboardShortcuts);

// Make history items keyboard accessible
function enhanceHistoryAccessibility() {
  const historyItems = document.querySelectorAll('.history-item');
  historyItems.forEach((item, index) => {
    item.setAttribute('tabindex', '0');
    item.setAttribute('role', 'listitem');

    // Allow keyboard navigation between history items
    item.addEventListener('keydown', (e) => {
      const items = [...document.querySelectorAll('.history-item')];
      let newIndex = index;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          newIndex = Math.min(index + 1, items.length - 1);
          break;
        case 'ArrowUp':
          e.preventDefault();
          newIndex = Math.max(index - 1, 0);
          break;
        case 'Home':
          e.preventDefault();
          newIndex = 0;
          break;
        case 'End':
          e.preventDefault();
          newIndex = items.length - 1;
          break;
      }

      if (newIndex !== index && items[newIndex]) {
        items[newIndex].focus();
      }
    });
  });
}

// Override renderHistory to add accessibility enhancements
const originalRenderHistory = renderHistory;
window.renderHistory = function() {
  originalRenderHistory();
  enhanceHistoryAccessibility();
};

// ─── INIT ─────────────────────────────────────────────────────────────────────
(async () => {
  initTheme();
  renderHistory();
  await checkApiHealth();
  // If results tab active on load
  if (window.location.hash === '#results') loadResults();

  // Set initial aria-hidden states
  $$('.tab-content').forEach(panel => {
    panel.setAttribute('aria-hidden', panel.classList.contains('hidden'));
  });

  // Announce page load
  announceToScreenReader('Voice Deepfake Detector and Generator loaded. Use Ctrl or Command plus D, G, R to navigate tabs.', 'polite');
})();

// History clear button event listener
$('#clear-history-btn')?.addEventListener('click', () => {
  if (confirm('Clear all detection history?')) {
    clearHistory();
    announceToScreenReader('Detection history cleared', 'polite');
  }
});

// Theme toggle event listener
$('#theme-toggle')?.addEventListener('click', () => {
  toggleTheme();
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  announceToScreenReader(`Switched to ${isDark ? 'dark' : 'light'} mode`, 'polite');
});
