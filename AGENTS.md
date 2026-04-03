# AGENTS.md

Полный контекст проекта для передачи AI-агентам (Codex, Claude Code, etc).

## Что это за проект

**Watermark Remover** — веб-сервис для удаления статичных водяных знаков из видео.

**Web server** (FastAPI) — единственная актуальная версия проекта. Деплоится на RunPod GPU. AI-инпеинтинг через IOPaint/LAMA. Desktop/Qt-код удалён из репозитория.

## Текущая стадия

**Web-версия активно тестируется на RunPod GPU (RTX A5000, 24GB VRAM).**

Что уже работает:
- Загрузка видео (upload или путь на сервере)
- Превью кадра на canvas с рисованием множественных регионов
- Два метода удаления: быстрый (FFmpeg delogo) и AI (IOPaint LAMA)
- AI обработка: батчевая (по 500 кадров), параллельная (4 IOPaint процесса), с frame skip
- WebSocket для real-time прогресса
- Скачивание результата
- Health endpoint с детекцией GPU
- Idle watchdog для автоостановки RunPod пода

## Актуальные проблемы (КРИТИЧНЫЕ)

### 1. Качество AI-обработки: водяные знаки НЕ удаляются полностью
**Причина**: тестовое видео (`Араб.mp4`) содержит тайловый водяной знак "artInflext.com", повторяющийся диагонально ~11 раз по кадру 1920x1080. При тестировании через Playwright регионы задавались неточно (промахивались мимо реальных позиций водяных знаков).

**Что нужно**:
- Правильно определять позиции водяных знаков и покрывать ВСЕ экземпляры
- Возможно: автодетекция повторяющихся водяных знаков
- Тестирование с корректными регионами для валидации качества LAMA inpainting

### 2. Скорость обработки: 6x вместо целевого 5x
- Видео 3:10 (190с) обрабатывается ~19 мин (6x)
- Цель: не более 5x от длительности видео
- Текущие оптимизации: 4 параллельных IOPaint + frame_skip=3
- Возможные дальнейшие оптимизации: прямой batch inference LAMA без CLI overhead, resize strategy

### 3. Playwright тест: добавление множественных регионов через UI
- Функция `applyCoords()` в `app.js` обновляет активный регион вместо добавления нового
- Нужно сбрасывать `activeIdx = -1` перед каждым добавлением через `page.evaluate()`
- Иначе все 11 регионов перезаписывают один и тот же

## Архитектура web-версии

```
server.py                  # FastAPI + WebSocket, точка входа
  /api/upload              # POST, загрузка видео
  /api/info?path=          # GET, метаданные видео
  /api/frame?path=&time=   # GET, извлечение кадра (превью)
  /ws/process              # WebSocket, запуск обработки
  /api/download/{file}     # GET, скачивание результата
  /api/queue               # GET/POST, пакетная очередь
  /api/cancel/{job_id}     # POST, отмена задачи
  /health                  # GET, статус + GPU info

services/
  iopaint_runner.py        # Генерация маски, извлечение кадров, запуск IOPaint,
                           # параллельная обработка (run_iopaint_parallel),
                           # прореживание кадров (thin_frames),
                           # заполнение пропусков (fill_skipped_frames),
                           # сборка видео (reassemble_video)
  video_info.py            # ffprobe/ffmpeg -i → resolution/duration/fps

static/
  index.html               # UI: Darkroom — preview canvas, regions, mode toggle, progress
  app.js                   # Canvas drawing, WebSocket client, file upload, queue polling
  style.css                # Dark theme

Dockerfile                 # pytorch + ffmpeg + iopaint, порт 8000
.dockerignore
deploy_runpod.sh           # Setup script for RunPod pods
```

## Ключевые параметры обработки

```python
BATCH_SIZE = 500   # кадров на батч (ограничение диска)
FRAME_SKIP = 3     # обработка каждого 3-го кадра (остальные дублируются)
IOPAINT_WORKERS = 4  # параллельных процессов IOPaint
```

## Поток AI-обработки (_process_ai в server.py)

1. Клиент подключается по WebSocket, отправляет: `{path, regions, duration, fps, width, height, mode:"ai", device:"cuda"}`
2. Генерируется маска (PIL) по координатам регионов
3. Цикл по батчам:
   a. `extract_frames_range()` — FFmpeg извлекает 500 кадров с правильной нумерацией
   b. `thin_frames()` — удаляет кадры не на границе skip (оставляет каждый 3-й)
   c. `run_iopaint_parallel()` — разбивает на 4 sub-батча, запускает IOPaint параллельно
   d. Перемещает результаты в `all_inpainted/`
4. `fill_skipped_frames()` — копирует ближайший обработанный кадр в пропуски
5. `reassemble_video()` — FFmpeg собирает PNG → MP4 с аудио из оригинала

## Деплой на RunPod

- **GPU**: RTX A5000 24GB VRAM (или аналог)
- **Порты**: 8000 (HTTP сервис), 22 (SSH)
- **SSH**: `ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519`
- **Proxy URL**: `https://<POD_ID>-8000.proxy.runpod.net/`
- Файлы в `/workspace/` (persistent volume)
- FFmpeg устанавливается через apt, iopaint через pip
- LAMA модель (~196MB) скачивается при первом запуске

## Деплой вручную

```bash
# На RunPod pod:
apt-get update && apt-get install -y ffmpeg
pip install fastapi uvicorn[standard] python-multipart aiofiles Pillow iopaint

# Скопировать файлы:
scp -P <PORT> -i ~/.ssh/id_ed25519 server.py services/ static/ root@<IP>:/workspace/

# Запуск:
cd /workspace && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## Тестирование

```bash
# Playwright тест (автоматически загружает видео, рисует регионы, запускает AI обработку):
python test_ai_processing.py

# Проверка здоровья сервера:
curl https://<POD_ID>-8000.proxy.runpod.net/health
```

## Зависимости (web)

`requirements_web.txt`:
- fastapi, uvicorn[standard], python-multipart, aiofiles, Pillow, iopaint
- Имплицитно: torch (из базового образа RunPod), ffmpeg (apt)

## Что НЕ делать

- Не возвращать desktop/Qt-версию (PySide6, `app.py`, `main_window.py`, `widgets/`) — репозиторий полностью web-only
- Не добавлять Docker/docker-compose — деплой напрямую на RunPod pod
- Не использовать chunking/splitting видео — все кадры обрабатываются как PNG
- Не менять формат WebSocket протокола без обновления app.js
- Не удалять FRAME_SKIP — без него обработка займёт 2+ часа на 3-мин видео

## Контакты заказчика

- Заказчик одобрил бюджет на GPU (RunPod On-Demand)
- Фокус: web-версия с AI качеством
- Требование: скорость обработки не более 5x от длительности видео
