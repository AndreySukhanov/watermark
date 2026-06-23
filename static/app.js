'use strict';

const state = {
  path: null,
  name: null,
  width: 0,
  height: 0,
  duration: 0,
  fps: 0,
  jobId: null,
  ws: null,
  status: 'idle',
  device: 'cuda',
  startTime: null,
  timerInterval: null,
};

const STEPS = ['step-upload', 'step-ready', 'step-processing', 'step-done', 'step-error'];

function showStep(id) {
  STEPS.forEach(s => {
    document.getElementById(s).style.display = s === id ? '' : 'none';
  });
}

function formatDuration(seconds) {
  if (!seconds || seconds <= 0) return '0 sec';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m > 0) return `${m} min ${s} sec`;
  return `${s} sec`;
}

function formatTimer(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function friendlyStatus(logMsg) {
  const m = logMsg.toLowerCase();
  if (m.includes('ocr mask') || m.includes('easyocr') || m.includes('median'))
    return 'Распознаём водяной знак...';
  if (m.includes('template match'))
    return 'Ищем все копии водяного знака...';
  if (m.includes('glyph') || m.includes('mask coverage') || m.includes('mask:'))
    return 'Строим маску удаления...';
  if (m.includes('extract'))
    return 'Извлекаем кадры из видео...';
  if (m.includes('iopaint') && m.includes('done'))
    return 'Инпейнтинг завершён';
  if (m.includes('iopaint') || m.includes('inpaint') || m.includes('batch'))
    return 'Удаляем водяной знак с кадров...';
  if (m.includes('gfpgan') && m.includes('done'))
    return 'Лица восстановлены';
  if (m.includes('восстановление лиц') || m.includes('face_restore') || m.includes('gfpgan'))
    return 'Восстанавливаем лица...';
  if (m.includes('reassembl'))
    return 'Собираем финальное видео...';
  if (m.includes('output'))
    return 'Почти готово...';
  return null;
}

// ── File handling ──

function onDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('video/'));
  if (files.length > 0) uploadFile(files[0]);
}

function onFileSelected(files) {
  const arr = Array.from(files);
  if (arr.length > 0) uploadFile(arr[0]);
}

async function uploadFile(file) {
  document.getElementById('upload-progress').style.display = '';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Upload failed');
    const info = await res.json();
    applyVideoInfo(info);
  } catch (e) {
    alert('Ошибка загрузки: ' + e.message);
    document.getElementById('upload-progress').style.display = 'none';
  }
}

async function loadFromPath() {
  const path = document.getElementById('local-path').value.trim();
  if (!path) return;
  try {
    const res = await fetch(`/api/info?path=${encodeURIComponent(path)}`);
    if (!res.ok) { const d = await res.json(); alert(d.error || 'Error'); return; }
    applyVideoInfo(await res.json());
  } catch (e) {
    alert('Ошибка: ' + e.message);
  }
}

function applyVideoInfo(info) {
  state.path = info.path;
  state.name = info.name;
  state.width = info.width;
  state.height = info.height;
  state.duration = info.duration;
  state.fps = info.fps;

  const meta = `${info.width}x${info.height}  ·  ${formatDuration(info.duration)}`;

  document.getElementById('video-name').textContent = info.name;
  document.getElementById('video-meta').textContent = meta;

  const thumb = document.getElementById('video-thumb');
  const t = Math.min(2, info.duration * 0.25);
  thumb.innerHTML = `<img src="/api/frame?path=${encodeURIComponent(info.path)}&time=${t}&_=${Date.now()}" alt="">`;

  document.getElementById('upload-progress').style.display = 'none';
  showStep('step-ready');
}

// ── Processing ──

function startProcessing() {
  if (!state.path) return;

  document.getElementById('proc-video-name').textContent = state.name;
  document.getElementById('proc-video-meta').textContent =
    `${state.width}x${state.height}  ·  ${formatDuration(state.duration)}`;

  setProgress(0);
  setStatus('Подготовка...');
  document.getElementById('log').textContent = '';
  showStep('step-processing');

  state.startTime = Date.now();
  startTimer();

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws/process`);
  state.ws = ws;
  state.status = 'processing';

  ws.onopen = () => {
    ws.send(JSON.stringify({
      path: state.path,
      regions: [],
      duration: state.duration,
      fps: state.fps,
      width: state.width,
      height: state.height,
      mode: 'ai',
      device: state.device,
      engine: 'lama_fast',
      engine_options: {
        mask_shape: 'ocr',
        mask_dilate: 3,
        mask_padding: 2,
      },
    }));
  };

  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    switch (msg.type) {
      case 'job_id':
        state.jobId = msg.data;
        break;
      case 'log':
        appendLog(msg.data);
        const friendly = friendlyStatus(msg.data);
        if (friendly) setStatus(friendly);
        break;
      case 'progress':
        setProgress(msg.data);
        break;
      case 'done':
        onDone(msg);
        break;
      case 'error':
        onError(msg.data);
        break;
    }
  };

  ws.onerror = () => onError('Потеряно соединение с сервером');
  ws.onclose = () => {
    if (state.status === 'processing') onError('Соединение прервано');
  };
}

function onDone(msg) {
  stopTimer();
  state.status = msg.success ? 'done' : 'error';
  state.ws = null;

  if (msg.success) {
    const elapsed = state.startTime ? Math.round((Date.now() - state.startTime) / 1000) : 0;
    document.getElementById('done-subtitle').textContent =
      `Водяной знак удалён за ${formatTimer(elapsed)}`;
    document.getElementById('btn-download').href = msg.download_url;
    showStep('step-done');
  } else {
    onError(msg.message || 'Неизвестная ошибка');
  }
}

function onError(text) {
  stopTimer();
  state.status = 'error';
  state.ws = null;
  document.getElementById('error-text').textContent = text;
  showStep('step-error');
}

async function cancelProcessing() {
  if (state.ws) state.ws.close();
  if (state.jobId) {
    await fetch(`/api/cancel/${state.jobId}`, { method: 'POST' });
    state.jobId = null;
  }
  stopTimer();
  state.status = 'cancelled';
  resetToReady();
}

// ── Progress helpers ──

function setProgress(pct) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-pct').textContent = Math.round(pct) + '%';
}

function setStatus(text) {
  document.getElementById('progress-status').textContent = text;
}

function appendLog(line) {
  const el = document.getElementById('log');
  el.textContent += (el.textContent ? '\n' : '') + line;
  el.scrollTop = el.scrollHeight;
}

function startTimer() {
  stopTimer();
  state.timerInterval = setInterval(() => {
    if (!state.startTime) return;
    const elapsed = Math.round((Date.now() - state.startTime) / 1000);
    document.getElementById('progress-time').textContent = formatTimer(elapsed);
  }, 1000);
}

function stopTimer() {
  if (state.timerInterval) {
    clearInterval(state.timerInterval);
    state.timerInterval = null;
  }
}

// ── Navigation ──

function resetToUpload() {
  state.path = null;
  state.name = null;
  state.jobId = null;
  document.getElementById('local-path').value = '';
  document.getElementById('file-input').value = '';
  showStep('step-upload');
}

function resetToReady() {
  if (state.path) {
    showStep('step-ready');
  } else {
    resetToUpload();
  }
}

// ── Device toggle ──

function setDevice(d) {
  state.device = d;
  document.getElementById('btn-cuda').classList.toggle('active', d === 'cuda');
  document.getElementById('btn-cpu').classList.toggle('active', d === 'cpu');
}

// ── Init ──
showStep('step-upload');
