'use strict';

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  path: null,
  name: null,
  width: 0,
  height: 0,
  duration: 0,
  fps: 0,
  jobId: null,
  ws: null,
  status: 'idle',         // idle | processing | done | error | cancelled
  downloadUrl: null,
  pendingFiles: [],       // Array of File objects selected by user
  pendingUploads: new Map(),
  engine: 'lama_fast',
  engines: [],
  qualityMaskProfile: 'hybrid_segmenter',
  qualityCropProfile: 'balanced',
  qualityAnalysis: null,
};

// ── Canvas ───────────────────────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

let frameImg = null;
let scaleX = 1, scaleY = 1;   // canvas px → video px

// Drag state
let dragging = false, dx0 = 0, dy0 = 0, dx1 = 0, dy1 = 0;

// Multi-region state
let regions = [];    // [{x, y, w, h}, ...]
let activeIdx = -1;  // index of selected region in list (-1 = none)
let mode = 'delogo'; // 'delogo' | 'ai'
let device = 'cpu';  // 'cpu' | 'cuda'
let showMaskPreview = false;

const MASK_PADDING = 8;
const MASK_DILATE = 4;
const MASK_EXPAND = MASK_PADDING + MASK_DILATE;
const FALLBACK_AI_ENGINES = [
  {
    key: 'lama_fast',
    label: 'LaMa Fast',
    family: 'lama',
    description: 'Максимально быстрый режим для длинных роликов.',
    estimate_multiplier: 4.9,
    skip: 4,
    refine_mask: false,
  },
  {
    key: 'propainter_quality',
    label: 'ProPainter',
    family: 'propainter',
    description: 'Video-aware quality mode с лучшей temporal consistency.',
    estimate_multiplier: 19.0,
    skip: 1,
    refine_mask: true,
  },
];

const QUALITY_MASK_PRESETS = {
  auto: {
    label: 'Glyph / Auto',
    hint: 'Текущий glyph-based refined mask без HF-сегментации.',
    options: {},
  },
  hf_segmenter: {
    label: 'HF Universal',
    hint: 'Прямая HF-сегментация universal-весами для полупрозрачного текста.',
    options: {
      mask_shape: 'hf_segmenter',
      segmenter_weights: 'segmenter_universal.pth',
      segmenter_threshold: 0.45,
    },
  },
  temporal_hf_segmenter: {
    label: 'Temporal + Universal HF',
    hint: 'Локально самый перспективный профиль: объединяет temporal hits и HF universal, чтобы убирать грубые полосы glyph-mask.',
    options: {
      mask_shape: 'temporal_hf_segmenter',
      segmenter_weights: 'segmenter_universal.pth',
      segmenter_threshold: 0.25,
      temporal_mask_samples: 6,
      temporal_mask_min_hits: 2,
    },
  },
  hybrid_segmenter: {
    label: 'Hybrid + Universal HF',
    hint: 'Рекомендуемый quality path: glyph-mask + HF universal, с ограничением по регионам.',
    options: {
      mask_shape: 'hybrid_segmenter',
      segmenter_weights: 'segmenter_universal.pth',
      segmenter_threshold: 0.42,
    },
  },
};

const QUALITY_CROP_PRESETS = {
  balanced: {
    label: 'Balanced',
    hint: 'Крупнее crop-группы, меньше запусков ProPainter. Быстрее, но больше риск зацепить лишний фон.',
    options: {},
  },
  detail: {
    label: 'Detail',
    hint: 'Меньше и плотнее crop-группы вокруг watermark. Медленнее, но это самый перспективный quality path сейчас.',
    options: {
      propainter_crop_padding: 24,
      propainter_crop_merge_gap: 8,
      propainter_crop_max_width: 720,
      propainter_crop_max_height: 320,
    },
  },
};

function fileKey(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

function getMaskRect(reg) {
  const maxWidth = state.width || (frameImg ? frameImg.naturalWidth : 0);
  const maxHeight = state.height || (frameImg ? frameImg.naturalHeight : 0);
  const x = Math.max(0, reg.x - MASK_EXPAND);
  const y = Math.max(0, reg.y - MASK_EXPAND);
  const right = maxWidth > 0 ? Math.min(maxWidth, reg.x + reg.w + MASK_EXPAND) : reg.x + reg.w + MASK_EXPAND;
  const bottom = maxHeight > 0 ? Math.min(maxHeight, reg.y + reg.h + MASK_EXPAND) : reg.y + reg.h + MASK_EXPAND;
  return { x, y, w: Math.max(0, right - x), h: Math.max(0, bottom - y) };
}

function getMaskRects() {
  return regions.map(getMaskRect).filter(r => r.w > 0 && r.h > 0);
}

function calculateUnionArea(rects) {
  if (!rects.length) return 0;
  const xs = [...new Set(rects.flatMap(r => [r.x, r.x + r.w]))].sort((a, b) => a - b);
  let area = 0;
  for (let i = 0; i < xs.length - 1; i++) {
    const x0 = xs[i];
    const x1 = xs[i + 1];
    if (x1 <= x0) continue;
    const intervals = rects
      .filter(r => r.x < x1 && r.x + r.w > x0)
      .map(r => [r.y, r.y + r.h])
      .sort((a, b) => a[0] - b[0]);
    if (!intervals.length) continue;
    let coveredY = 0;
    let [start, end] = intervals[0];
    for (let j = 1; j < intervals.length; j++) {
      const [nextStart, nextEnd] = intervals[j];
      if (nextStart <= end) {
        end = Math.max(end, nextEnd);
      } else {
        coveredY += end - start;
        [start, end] = [nextStart, nextEnd];
      }
    }
    coveredY += end - start;
    area += coveredY * (x1 - x0);
  }
  return area;
}

function updateMaskMeta() {
  const button = document.getElementById('btn-mask-preview');
  const meta = document.getElementById('mask-meta');
  const rects = getMaskRects();
  button.textContent = showMaskPreview ? 'Скрыть маску' : 'Показать маску';
  button.classList.toggle('is-active', showMaskPreview);

  if (!rects.length) {
    meta.textContent = showMaskPreview ? 'Маска включена · 0 регионов' : 'Нет регионов';
    return;
  }

  const frameArea = Math.max(1, (state.width || 0) * (state.height || 0));
  const coveragePct = frameArea > 1 ? Math.min(100, calculateUnionArea(rects) / frameArea * 100) : 0;
  const pctLabel = coveragePct >= 10 ? coveragePct.toFixed(0) : coveragePct.toFixed(1);
  meta.textContent = `${rects.length} рег. · ~${pctLabel}% кадра · +${MASK_EXPAND}px`;
}

function invalidateQualityAnalysis() {
  state.qualityAnalysis = null;
  renderQualityAnalysis();
}

canvas.addEventListener('mousedown', e => {
  if (!frameImg) return;
  const r = canvas.getBoundingClientRect();
  dx0 = dx1 = (e.clientX - r.left) * (canvas.width  / r.width);
  dy0 = dy1 = (e.clientY - r.top)  * (canvas.height / r.height);
  dragging = true;
});

canvas.addEventListener('mousemove', e => {
  if (!dragging) return;
  const r = canvas.getBoundingClientRect();
  dx1 = (e.clientX - r.left) * (canvas.width  / r.width);
  dy1 = (e.clientY - r.top)  * (canvas.height / r.height);
  redraw();
});

canvas.addEventListener('mouseup', e => {
  if (!dragging) return;
  dragging = false;
  const r = canvas.getBoundingClientRect();
  dx1 = (e.clientX - r.left) * (canvas.width  / r.width);
  dy1 = (e.clientY - r.top)  * (canvas.height / r.height);
  commitSelection();
  redraw();
});

function commitSelection() {
  const x = Math.max(0, Math.round(Math.min(dx0, dx1) * scaleX));
  const y = Math.max(0, Math.round(Math.min(dy0, dy1) * scaleY));
  let w = Math.round(Math.abs(dx1 - dx0) * scaleX);
  let h = Math.round(Math.abs(dy1 - dy0) * scaleY);
  w = Math.min(w, state.width  - x);
  h = Math.min(h, state.height - y);
  if (w < 2 || h < 2) return; // ignore tiny accidental clicks
  regions.push({ x, y, w, h });
  activeIdx = regions.length - 1;
  invalidateQualityAnalysis();
  syncCoordsToInputs();
  renderRegionsList();
}

function redraw() {
  if (!frameImg) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(frameImg, 0, 0, canvas.width, canvas.height);

  if (showMaskPreview) {
    ctx.save();
    getMaskRects().forEach(reg => {
      const cx = reg.x / scaleX;
      const cy = reg.y / scaleY;
      const cw = reg.w / scaleX;
      const ch = reg.h / scaleY;
      ctx.fillStyle = 'rgba(184, 48, 32, 0.18)';
      ctx.fillRect(cx, cy, cw, ch);
      ctx.strokeStyle = 'rgba(224, 144, 32, 0.75)';
      ctx.lineWidth = 1;
      ctx.strokeRect(cx, cy, cw, ch);
    });
    ctx.restore();
  }

  if (dragging) {
    const x = Math.min(dx0, dx1), y = Math.min(dy0, dy1);
    const w = Math.abs(dx1 - dx0),  h = Math.abs(dy1 - dy0);
    ctx.save();
    ctx.strokeStyle = '#ff5050';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }

  // Draw all committed regions
  regions.forEach((reg, i) => {
    const cx = reg.x / scaleX, cy = reg.y / scaleY;
    const cw = reg.w / scaleX, ch = reg.h / scaleY;
    const isActive = i === activeIdx;

    ctx.save();
    ctx.strokeStyle = isActive ? '#ff5050' : '#ff8800';
    ctx.lineWidth = isActive ? 2 : 1.5;
    ctx.setLineDash([]);
    ctx.strokeRect(cx, cy, cw, ch);

    // Number label
    const label = String(i + 1);
    ctx.font = 'bold 12px monospace';
    const lx = cx + 3;
    const ly = cy > 18 ? cy - 4 : cy + ch + 14;
    ctx.fillStyle = isActive ? 'rgba(200,20,20,0.85)' : 'rgba(200,100,0,0.85)';
    ctx.fillRect(lx - 2, ly - 12, 16, 16);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, lx, ly);
    ctx.restore();
  });
}

function renderRegionsList() {
  const el = document.getElementById('regions-list');
  if (regions.length === 0) {
    el.innerHTML = '<div class="regions-empty">Нет регионов — нарисуйте прямоугольники на превью</div>';
    updateMaskMeta();
    return;
  }
  el.innerHTML = regions.map((r, i) => `
    <div class="region-row${i === activeIdx ? ' active' : ''}" onclick="selectRegion(${i})">
      <span class="region-num">${i + 1}</span>
      <span class="region-coords">x=${r.x} y=${r.y} w=${r.w} h=${r.h}</span>
      <button class="btn btn-sm btn-ghost region-del" onclick="deleteRegion(${i}); event.stopPropagation()">✕</button>
    </div>
  `).join('');
  updateMaskMeta();
}

function selectRegion(i) {
  activeIdx = i;
  syncCoordsToInputs();
  renderRegionsList();
  redraw();
}

function deleteRegion(i) {
  regions.splice(i, 1);
  if (activeIdx >= regions.length) activeIdx = regions.length - 1;
  invalidateQualityAnalysis();
  syncCoordsToInputs();
  renderRegionsList();
  redraw();
}

// ── Coordinate controls ───────────────────────────────────────────────────────

function syncCoordsToInputs() {
  if (activeIdx >= 0 && activeIdx < regions.length) {
    const r = regions[activeIdx];
    document.getElementById('inp-x').value = r.x;
    document.getElementById('inp-y').value = r.y;
    document.getElementById('inp-w').value = r.w;
    document.getElementById('inp-h').value = r.h;
  } else {
    document.getElementById('inp-x').value = 0;
    document.getElementById('inp-y').value = 0;
    document.getElementById('inp-w').value = 0;
    document.getElementById('inp-h').value = 0;
  }
}

function applyCoords() {
  const x = parseInt(document.getElementById('inp-x').value) || 0;
  const y = parseInt(document.getElementById('inp-y').value) || 0;
  const w = parseInt(document.getElementById('inp-w').value) || 0;
  const h = parseInt(document.getElementById('inp-h').value) || 0;
  if (activeIdx >= 0 && activeIdx < regions.length) {
    regions[activeIdx] = { x, y, w, h };
  } else {
    if (w < 1 || h < 1) return;
    regions.push({ x, y, w, h });
    activeIdx = regions.length - 1;
  }
  invalidateQualityAnalysis();
  renderRegionsList();
  redraw();
}

function resetSelection() {
  regions = [];
  activeIdx = -1;
  invalidateQualityAnalysis();
  syncCoordsToInputs();
  renderRegionsList();
  if (frameImg) redraw();
}

function toggleMaskPreview() {
  showMaskPreview = !showMaskPreview;
  updateMaskMeta();
  if (frameImg) redraw();
}

function formatDurationLabel(seconds) {
  if (!seconds || seconds <= 0) return null;
  if (seconds >= 3600) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.round((seconds % 3600) / 60);
    return `${hours} ч ${minutes} мин`;
  }
  if (seconds >= 60) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${minutes} мин ${secs} с`;
  }
  return `${Math.round(seconds)} с`;
}

function getSelectedEngine() {
  return state.engines.find(engine => engine.key === state.engine) || state.engines[0] || FALLBACK_AI_ENGINES[0];
}

function getSelectedQualityMaskProfile() {
  return QUALITY_MASK_PRESETS[state.qualityMaskProfile] || QUALITY_MASK_PRESETS.hybrid_segmenter;
}

function getSelectedQualityCropProfile() {
  return QUALITY_CROP_PRESETS[state.qualityCropProfile] || QUALITY_CROP_PRESETS.balanced;
}

function formatMaskShapeLabel(maskShape) {
  if (maskShape === 'hybrid_segmenter') return 'Hybrid + Universal HF';
  if (maskShape === 'hf_segmenter') return 'HF Universal';
  if (maskShape === 'temporal_hf_segmenter') return 'Temporal + Universal HF';
  return 'Glyph / Auto';
}

function getEngineOptions() {
  const engine = getSelectedEngine();
  if (!engine || engine.family !== 'propainter') return {};
  return {
    ...getSelectedQualityMaskProfile().options,
    ...getSelectedQualityCropProfile().options,
  };
}

function renderQualityMaskControls() {
  const row = document.getElementById('quality-mask-row');
  const select = document.getElementById('quality-mask-shape');
  const meta = document.getElementById('quality-mask-meta');
  const cropRow = document.getElementById('quality-crop-row');
  const cropSelect = document.getElementById('quality-crop-profile');
  const cropMeta = document.getElementById('quality-crop-meta');
  const engine = getSelectedEngine();
  const visible = mode === 'ai' && engine?.family === 'propainter';
  row.style.display = visible ? '' : 'none';
  cropRow.style.display = visible ? '' : 'none';
  if (!visible) return;

  if (!QUALITY_MASK_PRESETS[state.qualityMaskProfile]) {
    state.qualityMaskProfile = 'hybrid_segmenter';
  }
  if (!QUALITY_CROP_PRESETS[state.qualityCropProfile]) {
    state.qualityCropProfile = 'balanced';
  }
  select.value = state.qualityMaskProfile;
  meta.textContent = getSelectedQualityMaskProfile().hint;
  cropSelect.value = state.qualityCropProfile;
  cropMeta.textContent = getSelectedQualityCropProfile().hint;
}

function renderEngineList() {
  const el = document.getElementById('engine-list');
  if (!state.engines.length) {
    el.innerHTML = '<div class="regions-empty">Профили AI недоступны</div>';
    return;
  }
  el.innerHTML = state.engines.map(engine => `
    <button class="btn engine-btn${engine.key === state.engine ? ' is-active' : ''}" onclick="setEngine('${engine.key}')">
      <span class="engine-copy">
        <span class="engine-name">${engine.label}</span>
        <span class="engine-desc">${engine.description}</span>
      </span>
      <span class="engine-chip">~${engine.estimate_multiplier}x</span>
    </button>
  `).join('');
}

function updateAiModeMeta() {
  const warning = document.getElementById('ai-warning');
  const engineRow = document.getElementById('engine-row');
  const deviceRow = document.getElementById('device-row');
  const meta = document.getElementById('engine-meta');
  const engine = getSelectedEngine();
  const maskProfile = getSelectedQualityMaskProfile();
  const cropProfile = getSelectedQualityCropProfile();

  engineRow.style.display = mode === 'ai' ? '' : 'none';
  warning.style.display = mode === 'ai' ? '' : 'none';
  renderQualityMaskControls();

  if (mode !== 'ai') {
    deviceRow.style.display = 'none';
    return;
  }

  const needsCuda = engine.family === 'propainter';
  if (needsCuda) {
    setDevice('cuda');
  }
  deviceRow.style.display = needsCuda ? 'none' : '';

  const estimate = state.duration > 0 ? formatDurationLabel(state.duration * engine.estimate_multiplier) : null;
  const skipPart = engine.skip > 1 ? `Skip ${engine.skip}` : 'Без skip';
  const refinePart = engine.family === 'propainter'
    ? maskProfile.label
    : (engine.refine_mask ? 'refined mask' : 'простая mask');
  const cropPart = engine.family === 'propainter' ? cropProfile.label : null;
  warning.textContent = estimate
    ? `${engine.label}: ориентир ~${estimate} · ${skipPart} · ${refinePart}${cropPart ? ` · ${cropPart}` : ''}.`
    : `${engine.label}: ${engine.description}`;
  meta.textContent = `${engine.description} Режим: ${skipPart}. Маска: ${refinePart}.${cropPart ? ` Crop: ${cropPart}.` : ''}`;
}

function setEngine(key) {
  state.engine = key;
  invalidateQualityAnalysis();
  renderEngineList();
  updateAiModeMeta();
}

function setQualityMaskProfile(value) {
  if (!QUALITY_MASK_PRESETS[value]) value = 'hybrid_segmenter';
  state.qualityMaskProfile = value;
  invalidateQualityAnalysis();
  renderQualityMaskControls();
  updateAiModeMeta();
}

function setQualityCropProfile(value) {
  if (!QUALITY_CROP_PRESETS[value]) value = 'balanced';
  state.qualityCropProfile = value;
  invalidateQualityAnalysis();
  renderQualityMaskControls();
  updateAiModeMeta();
}

async function loadAiEngines() {
  try {
    const res = await fetch('/api/ai/engines');
    if (!res.ok) throw new Error('engine api');
    state.engines = await res.json();
  } catch (_) {
    state.engines = FALLBACK_AI_ENGINES;
  }
  if (!state.engines.some(engine => engine.key === state.engine)) {
    state.engine = state.engines[0]?.key || 'lama_fast';
  }
  renderEngineList();
  updateAiModeMeta();
  renderQualityMaskControls();
}

function renderQualityAnalysis() {
  const meta = document.getElementById('quality-meta');
  const grid = document.getElementById('quality-preview-grid');
  const ref = document.getElementById('quality-reference');
  const mask = document.getElementById('quality-mask');
  const cropsWrap = document.getElementById('quality-crops-wrap');
  const crops = document.getElementById('quality-crops');
  const cropList = document.getElementById('quality-crop-list');
  if (!state.qualityAnalysis) {
    meta.textContent = 'Для анализа нужен хотя бы один регион и загруженный кадр.';
    grid.style.display = 'none';
    ref.removeAttribute('src');
    mask.removeAttribute('src');
    crops.removeAttribute('src');
    cropsWrap.style.display = 'none';
    cropList.style.display = 'none';
    cropList.innerHTML = '';
    return;
  }
  const data = state.qualityAnalysis;
  const coverage = typeof data.mask_coverage === 'number' ? data.mask_coverage.toFixed(3) : '0.000';
  const bbox = data.mask_bbox || {};
  const maskShape = formatMaskShapeLabel(data.engine?.mask_shape || 'auto');
  const cropGroups = Array.isArray(data.crop_groups) ? data.crop_groups : [];
  const maxCropArea = cropGroups.reduce((best, item) => Math.max(best, (item.w || 0) * (item.h || 0)), 0);
  const largestCrop = cropGroups.find(item => ((item.w || 0) * (item.h || 0)) === maxCropArea);
  meta.textContent =
    `Engine: ${data.engine?.label || state.engine} · reference ${data.reference_time}s · ` +
    `регионов ${data.merged_region_count} · автонайдено ${data.suggested_region_count} · ` +
    `mask ${coverage}% · ${maskShape} · bbox ${bbox.w || 0}×${bbox.h || 0}` +
    (cropGroups.length ? ` · crop groups ${cropGroups.length} · max ${largestCrop?.w || 0}×${largestCrop?.h || 0}` : '');
  ref.src = data.reference_url + `?_=${Date.now()}`;
  mask.src = data.mask_preview_url + `?_=${Date.now()}`;
  if (data.crop_preview_url && cropGroups.length) {
    crops.src = data.crop_preview_url + `?_=${Date.now()}`;
    cropsWrap.style.display = '';
    cropList.innerHTML = cropGroups
      .map(group => `<span class="quality-crop-chip">#${group.index} · ${group.region_count} reg · ${group.w}×${group.h}</span>`)
      .join('');
    cropList.style.display = '';
  } else {
    crops.removeAttribute('src');
    cropsWrap.style.display = 'none';
    cropList.style.display = 'none';
    cropList.innerHTML = '';
  }
  grid.style.display = '';
}

function updateQualityCardVisibility() {
  const card = document.getElementById('quality-card');
  card.style.display = mode === 'ai' ? '' : 'none';
}

async function analyzeQuality(autodetect = false) {
  if (!state.path) {
    alert('Сначала загрузите видеофайл.');
    return;
  }
  if (regions.length === 0) {
    alert('Нужен хотя бы один регион для quality analyze.');
    return;
  }

  const btnAnalyze = document.getElementById('btn-quality-analyze');
  const btnDetect = document.getElementById('btn-quality-detect');
  const engineOptions = getEngineOptions();
  btnAnalyze.disabled = true;
  btnDetect.disabled = true;

  try {
    const res = await fetch('/api/quality/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        path: state.path,
        duration: state.duration,
        width: state.width,
        height: state.height,
        regions,
        engine: state.engine,
        engine_options: engineOptions,
        autodetect,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'Quality analyze failed');
    }
    state.qualityAnalysis = data;
    if (autodetect && Array.isArray(data.merged_regions) && data.merged_regions.length >= regions.length) {
      regions = data.merged_regions;
      activeIdx = regions.length ? regions.length - 1 : -1;
      syncCoordsToInputs();
      renderRegionsList();
      redraw();
      appendLog(`+ Auto-detect: найдено ${data.suggested_region_count} дополнительных регионов`);
    } else {
      appendLog(`+ Quality analyze: mask ${data.mask_coverage.toFixed(3)}%`);
    }
    renderQualityAnalysis();
  } catch (e) {
    appendLog(`✗ Quality analyze: ${e.message}`);
    alert('Ошибка quality analyze: ' + e.message);
  } finally {
    btnAnalyze.disabled = false;
    btnDetect.disabled = false;
  }
}

// ── Mode toggle ───────────────────────────────────────────────────────────────

function setMode(m) {
  mode = m;
  document.getElementById('btn-mode-delogo').classList.toggle('active', m === 'delogo');
  document.getElementById('btn-mode-ai').classList.toggle('active', m === 'ai');
  updateAiModeMeta();
  updateQualityCardVisibility();
}

function setDevice(d) {
  device = d;
  document.getElementById('btn-cpu').classList.toggle('active',  d === 'cpu');
  document.getElementById('btn-cuda').classList.toggle('active', d === 'cuda');
}

// ── File loading ─────────────────────────────────────────────────────────────

function switchTab(tabId) {
  document.getElementById('mode-path').style.display   = tabId === 'path'   ? '' : 'none';
  document.getElementById('mode-upload').style.display = tabId === 'upload' ? '' : 'none';
  document.getElementById('tab-path').classList.toggle('active',   tabId === 'path');
  document.getElementById('tab-upload').classList.toggle('active', tabId === 'upload');
}

async function loadFromPath() {
  const path = document.getElementById('local-path').value.trim();
  if (!path) return;
  try {
    const res = await fetch(`/api/info?path=${encodeURIComponent(path)}`);
    if (!res.ok) { const d = await res.json(); alert(d.error); return; }
    const info = await res.json();
    applyFileInfo(info);
    await loadFrame();
  } catch (e) {
    alert('Ошибка: ' + e.message);
  }
}

function onDrop(e) {
  e.preventDefault();
  const files = Array.from(e.dataTransfer.files);
  if (files.length > 0) onFileSelected(files);
}

function onFileSelected(files) {
  const fileArray = Array.from(files);
  if (fileArray.length > 0) {
    state.pendingFiles = fileArray;
    state.pendingUploads = new Map();
    renderPendingFiles();
    // Use the first file to set up the preview and regions
    uploadFile(fileArray[0], true);
  }
}

function renderPendingFiles() {
  const el = document.getElementById('pending-files-list');
  const btnAll = document.getElementById('btn-add-all-queue');
  if (state.pendingFiles.length > 1) {
    el.innerHTML = `Выбрано файлов: ${state.pendingFiles.length}. Первый (${state.pendingFiles[0].name}) открыт для настройки регионов.`;
    btnAll.style.display = 'inline-block';
    btnAll.textContent = `+ Добавить ВСЕ (${state.pendingFiles.length})`;
  } else {
    el.innerHTML = '';
    btnAll.style.display = 'none';
  }
}

async function uploadFile(file, isPreview = false) {
  if (isPreview) {
    setBadge('processing', 'Загрузка превью...');
  }
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!res.ok) { alert('Ошибка загрузки ' + file.name); return null; }
    const info = await res.json();
    state.pendingUploads.set(fileKey(file), info);
    if (isPreview) {
      applyFileInfo(info);
      await loadFrame();
    }
    return info;
  } catch (e) {
    alert('Ошибка загрузки: ' + e.message);
    return null;
  } finally {
    if (isPreview) setBadge('idle', 'Готов');
  }
}

function applyFileInfo(info) {
  state.path     = info.path;
  state.name     = info.name;
  state.width    = info.width;
  state.height   = info.height;
  state.duration = info.duration;
  state.fps      = info.fps;
  invalidateQualityAnalysis();

  const dur = info.duration;
  const mins = Math.floor(dur / 60);
  const secs = Math.floor(dur % 60);
  const durStr = mins > 0 ? `${mins} мин ${secs} с` : `${secs} с`;

  document.getElementById('metadata').textContent =
    `${info.name}  ·  ${info.width}×${info.height}  ·  ${durStr}`;

  document.getElementById('file-meta').style.display = '';
  document.getElementById('frame-counter').textContent = '▸ ' + info.name;
  updateMaskMeta();
  updateAiModeMeta();
}

// ── Frame extraction ──────────────────────────────────────────────────────────

async function loadFrame() {
  if (!state.path) return;
  const t = parseFloat(document.getElementById('preview-time').value) || 1.0;
  const url = `/api/frame?path=${encodeURIComponent(state.path)}&time=${t}&_=${Date.now()}`;

  const img = new Image();
  img.onload = () => {
    frameImg = img;

    // Size canvas to fit container while preserving aspect ratio
    const wrap = document.getElementById('canvas-wrap');
    const maxW = wrap.clientWidth  - 4;
    const maxH = wrap.clientHeight - 4;
    const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
    canvas.width  = Math.round(img.naturalWidth  * scale);
    canvas.height = Math.round(img.naturalHeight * scale);

    scaleX = img.naturalWidth  / canvas.width;
    scaleY = img.naturalHeight / canvas.height;

    canvas.style.display = 'block';
    document.getElementById('canvas-empty').style.display = 'none';

    // Restore existing regions
    if (regions.length > 0) redraw();
    else ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.onerror = () => appendLog('Не удалось загрузить кадр. Проверьте FFmpeg и путь к файлу.');
  img.src = url;
}

// ── Processing ────────────────────────────────────────────────────────────────

function startProcessing() {
  if (!state.path)          { alert('Сначала загрузите видеофайл.'); return; }
  if (regions.length === 0) { alert('Выделите хотя бы один регион водяного знака на превью.'); return; }

  clearLog();
  setProgress(0);
  setBadge('processing', 'Обработка...');
  document.getElementById('btn-start').disabled         = true;
  document.getElementById('btn-cancel').disabled        = false;
  document.getElementById('btn-download').style.display = 'none';
  document.getElementById('progress-wrap').style.display = '';
  document.getElementById('status-card').style.display  = '';

  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${window.location.host}/ws/process`);
  state.ws = ws;
  const engineOptions = getEngineOptions();

  ws.onopen = () => {
    ws.send(JSON.stringify({
      path:     state.path,
      regions,
      duration: state.duration,
      fps:      state.fps,
      width:    state.width,
      height:   state.height,
      mode,
      device,
      engine: state.engine,
      engine_options: engineOptions,
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

  ws.onerror = () => onError('WebSocket ошибка соединения');
  ws.onclose = () => {
    if (state.status === 'processing') onError('Соединение прервано');
  };
  state.status = 'processing';
}

function onDone(msg) {
  state.status = msg.success ? 'done' : 'error';
  document.getElementById('btn-start').disabled  = false;
  document.getElementById('btn-cancel').disabled = true;

  if (msg.success) {
    setProgress(100);
    setBadge('done', 'Готово');
    const btn = document.getElementById('btn-download');
    btn.href = msg.download_url;
    btn.style.display = '';
    appendLog('✓ Обработка завершена. Файл готов к скачиванию.');
  } else {
    setBadge('error', 'Ошибка');
    appendLog('✗ Ошибка: ' + (msg.message || 'неизвестная ошибка'));
  }
  state.ws = null;
}

function onError(text) {
  state.status = 'error';
  setBadge('error', 'Ошибка');
  document.getElementById('status-card').style.display = '';
  appendLog('✗ ' + text);
  document.getElementById('btn-start').disabled  = false;
  document.getElementById('btn-cancel').disabled = true;
  state.ws = null;
}

async function cancelProcessing() {
  if (state.ws) state.ws.close();
  if (state.jobId) {
    await fetch(`/api/cancel/${state.jobId}`, { method: 'POST' });
    state.jobId = null;
  }
  state.status = 'cancelled';
  setBadge('cancelled', 'Отменено');
  document.getElementById('btn-start').disabled  = false;
  document.getElementById('btn-cancel').disabled = true;
  appendLog('— Отменено пользователем');
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function setBadge(cls, text) {
  const b = document.getElementById('status-badge');
  b.className = 'badge ' + cls;
  b.textContent = text;
}

function setProgress(pct) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-label').textContent = pct + '%';
}

function appendLog(line) {
  const el = document.getElementById('log');
  el.textContent += (el.textContent ? '\n' : '') + line;
  el.scrollTop = el.scrollHeight;
}

function clearLog() {
  document.getElementById('log').textContent = '';
}

// ── Batch queue ───────────────────────────────────────────────────────────────

async function addToQueue() {
  if (!state.path)          { alert('Сначала загрузите видеофайл.'); return; }
  if (regions.length === 0) { alert('Выделите хотя бы один регион.'); return; }

  const body = {
    path:     state.path,
    name:     state.name,
    regions,
    duration: state.duration,
    fps:      state.fps,
    width:    state.width,
    height:   state.height,
    mode,
    device,
    engine:   state.engine,
    engine_options: getEngineOptions(),
  };
  const res = await fetch('/api/queue', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) { alert('Ошибка добавления в очередь'); return; }
  const data = await res.json();
  appendLog(`+ В очередь добавлено задание ${data.job_id.slice(0, 8)}…`);
}

async function addAllToQueue() {
  if (state.pendingFiles.length === 0) { alert('Сначала загрузите несколько файлов.'); return; }
  if (regions.length === 0) { alert('Выделите хотя бы один регион на превью.'); return; }
  
  const btnAll = document.getElementById('btn-add-all-queue');
  btnAll.disabled = true;
  btnAll.textContent = 'Загрузка...';

  for (let i = 0; i < state.pendingFiles.length; i++) {
    const file = state.pendingFiles[i];
    appendLog(`Загрузка ${file.name} (${i+1}/${state.pendingFiles.length})...`);

    const cachedInfo = state.pendingUploads.get(fileKey(file));
    const info = cachedInfo || await uploadFile(file, false);
    if (!info) continue;
    if (cachedInfo) {
      appendLog(`Используется уже загруженный preview для ${file.name}`);
    }
    
    const body = {
      path:     info.path,
      name:     info.name,
      regions,
      duration: info.duration,
      fps:      info.fps,
      width:    info.width,
      height:   info.height,
      mode,
      device,
      engine:   state.engine,
      engine_options: getEngineOptions(),
    };
    
    try {
      const res = await fetch('/api/queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) { 
        appendLog(`✗ Ошибка добавления в очередь: ${file.name}`);
      } else {
        const data = await res.json();
        appendLog(`+ Добавлено в очередь: ${file.name} (job ${data.job_id.slice(0, 8)})`);
      }
    } catch (e) {
      appendLog(`✗ Ошибка добавления: ${e.message}`);
    }
  }

  btnAll.textContent = '+ Добавить ВСЕ';
  btnAll.disabled = false;
  state.pendingFiles = [];
  state.pendingUploads = new Map();
  renderPendingFiles();
}

async function removeFromQueue(jobId) {
  await fetch(`/api/queue/${jobId}`, { method: 'DELETE' });
}

function renderQueueList(jobs) {
  const el = document.getElementById('queue-list');
  if (!jobs.length) {
    el.innerHTML = '<div class="regions-empty">Очередь пуста</div>';
    return;
  }
  const statusLabel = { queued: 'В очереди', processing: 'Обработка', done: 'Готово', error: 'Ошибка', cancelled: 'Отменено' };
  const statusCls   = { queued: '', processing: 'processing', done: 'done', error: 'error', cancelled: 'cancelled' };
  el.innerHTML = jobs.map(j => `
    <div class="queue-job">
      <div class="queue-job-top">
        <span class="queue-job-name">${j.name || j.job_id.slice(0, 8) + '…'}</span>
        <span class="badge ${statusCls[j.status] || ''}">${statusLabel[j.status] || j.status}</span>
        <button class="btn btn-sm btn-ghost" onclick="removeFromQueue('${j.job_id}')">✕</button>
      </div>
      ${j.status === 'processing' ? `
        <div class="progress-track" style="margin-top:4px">
          <div class="progress-fill" style="width:${j.progress}%"></div>
        </div>` : ''}
      ${j.download_url ? `<a class="btn btn-sm btn-success" href="${j.download_url}" download style="margin-top:4px">↓ Скачать</a>` : ''}
      ${j.error ? `<div class="warn-text" style="margin-top:2px">${j.error}</div>` : ''}
    </div>
  `).join('');
}

// Poll queue every 2 seconds
setInterval(async () => {
  try {
    const res = await fetch('/api/queue');
    if (res.ok) renderQueueList(await res.json());
  } catch (_) {}
}, 2000);

updateMaskMeta();
updateQualityCardVisibility();
renderQualityAnalysis();
loadAiEngines();
