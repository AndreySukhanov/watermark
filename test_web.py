import io
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

from test_helpers import BASE_URL, open_video_in_ui, save_shot, wait_for_preview

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        save_shot(page, "web_01_initial.png")
        print("✓ Страница загружена")

        source_info = open_video_in_ui(page)
        wait_for_preview(page)
        page.wait_for_timeout(2000)
        save_shot(page, "web_02_loaded.png")
        meta = (page.text_content("#metadata") or "").strip()
        print(f"✓ Видео открыто через {source_info['source']}: {meta}")

        page.locator("#preview-time").fill("5")
        page.locator("button", has_text="Обновить").click()
        page.wait_for_timeout(2500)
        save_shot(page, "web_03_frame.png")
        print("✓ Кадр обновлён")

        canvas = page.locator("#canvas")
        box = canvas.bounding_box()
        print(f"  Canvas: {int(box['width'])}×{int(box['height'])}px")

        x1 = box["x"] + box["width"] * 0.75
        y1 = box["y"] + box["height"] * 0.02
        x2 = box["x"] + box["width"] * 0.97
        y2 = box["y"] + box["height"] * 0.14

        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move(x2, y2, steps=20)
        page.mouse.up()
        page.wait_for_timeout(300)
        save_shot(page, "web_04_selection.png")

        sel_x = page.input_value("#inp-x")
        sel_y = page.input_value("#inp-y")
        sel_w = page.input_value("#inp-w")
        sel_h = page.input_value("#inp-h")
        print(f"✓ Выделение: x={sel_x} y={sel_y} w={sel_w} h={sel_h}")

        page.locator("#btn-mask-preview").click()
        page.wait_for_timeout(200)
        save_shot(page, "web_05_mask.png")
        print("✓ Preview маски включён")

        page.locator("#btn-start").click()
        page.wait_for_timeout(1500)
        save_shot(page, "web_06_processing.png")
        badge = (page.text_content("#status-badge") or "").strip()
        print(f"✓ Статус после запуска: {badge}")

        try:
            page.wait_for_selector("#btn-download:visible", timeout=300_000)
            save_shot(page, "web_07_done.png")
            badge = (page.text_content("#status-badge") or "").strip()
            download_href = page.get_attribute("#btn-download", "href")
            print(f"✓ Финальный статус: {badge}")
            print(f"✓ Ссылка для скачивания: {download_href}")
        except Exception:
            badge = page.text_content("#status-badge") or ""
            log = page.text_content("#log") or ""
            save_shot(page, "web_07_timeout.png")
            print(f"✗ Timeout/ошибка. Статус: {badge}. Лог: {log[-300:]}")
            raise
        finally:
            browser.close()

        print(f"\nСкриншоты: {Path('output/playwright').resolve()}")


if __name__ == "__main__":
    main()
