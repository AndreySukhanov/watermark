import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from playwright.sync_api import sync_playwright

VIDEO_PATH = r"C:\Users\Пользователь\Desktop\watermark\Араб.mp4"
BASE_URL   = "http://127.0.0.1:8000"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")

    # ── Screenshot 1: начальное состояние ──
    page.screenshot(path="/tmp/wm_01_initial.png", full_page=True)
    print("✓ Страница загружена")

    # ── Ввести путь к файлу ──
    page.fill("#local-path", VIDEO_PATH)
    page.click("button:has-text('Открыть')")
    page.wait_for_timeout(2000)
    page.screenshot(path="/tmp/wm_02_meta.png", full_page=True)
    meta = page.text_content("#metadata")
    print(f"✓ Метаданные: {meta.strip()}")

    # ── Загрузить кадр ──
    page.fill("#preview-time", "5")
    page.click("button:has-text('Загрузить кадр')")
    page.wait_for_timeout(3000)
    page.screenshot(path="/tmp/wm_03_frame.png", full_page=True)
    print("✓ Кадр загружен")

    # ── Нарисовать прямоугольник на canvas (симуляция мыши) ──
    canvas = page.locator("#canvas")
    box    = canvas.bounding_box()
    print(f"  Canvas: {int(box['width'])}×{int(box['height'])}px")

    # Watermark region: верхний правый угол ~75-95% x, 2-12% y
    x1 = box["x"] + box["width"]  * 0.75
    y1 = box["y"] + box["height"] * 0.02
    x2 = box["x"] + box["width"]  * 0.97
    y2 = box["y"] + box["height"] * 0.14

    page.mouse.move(x1, y1)
    page.mouse.down()
    page.mouse.move(x2, y2, steps=20)
    page.mouse.up()
    page.wait_for_timeout(300)
    page.screenshot(path="/tmp/wm_04_selection.png", full_page=True)

    sel_x = page.input_value("#inp-x")
    sel_y = page.input_value("#inp-y")
    sel_w = page.input_value("#inp-w")
    sel_h = page.input_value("#inp-h")
    print(f"✓ Выделение: x={sel_x} y={sel_y} w={sel_w} h={sel_h}")

    # ── Запустить обработку ──
    page.click("#btn-start")
    page.wait_for_timeout(1500)
    page.screenshot(path="/tmp/wm_05_processing.png", full_page=True)
    badge = page.text_content("#status-badge")
    print(f"✓ Статус после запуска: {badge.strip()}")

    # ── Ждать завершения (max 300s) ──
    try:
        page.wait_for_selector("#btn-download:visible", timeout=300_000)
        page.screenshot(path="/tmp/wm_06_done.png", full_page=True)
        badge = page.text_content("#status-badge")
        print(f"✓ Финальный статус: {badge.strip()}")
        download_href = page.get_attribute("#btn-download", "href")
        print(f"✓ Ссылка для скачивания: {download_href}")
    except Exception as e:
        badge = page.text_content("#status-badge")
        log   = page.text_content("#log")
        page.screenshot(path="/tmp/wm_06_timeout.png", full_page=True)
        print(f"✗ Timeout/ошибка. Статус: {badge}. Лог: {log[-300:]}")

    browser.close()
    print("\nСкриншоты: /tmp/wm_01..06_*.png")
