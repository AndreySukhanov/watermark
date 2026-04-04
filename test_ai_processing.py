r"""
Playwright test: open video, add 11 watermark regions, AI+GPU, process.

Supports either:
- SERVER_VIDEO=/workspace/... for server-side path mode
- VIDEO_FILE=C:\path\to\video.mp4 for upload mode
"""
import io
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from test_helpers import BASE_URL, open_video_in_ui, save_shot, wait_for_preview

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


WATERMARK_REGIONS = json.loads(
    (Path(__file__).parent / "assets" / "arab_watermark_regions.json").read_text(encoding="utf-8")
)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})

    print("[1] Opening app...")
    page.goto(BASE_URL, timeout=30000)
    page.wait_for_load_state("networkidle")
    print("    OK")

    source_info = open_video_in_ui(page)
    print(f"[2] Loading video via {source_info['source']}: {source_info['value']}")
    wait_for_preview(page)
    page.wait_for_timeout(2000)
    meta = page.text_content("#metadata") or ""
    print(f"    Meta: {meta}")
    save_shot(page, "ai_01_loaded.png")

    print(f"[3] Adding {len(WATERMARK_REGIONS)} regions...")
    for r in WATERMARK_REGIONS:
        page.evaluate("activeIdx = -1")
        page.fill("#inp-x", str(r["x"]))
        page.fill("#inp-y", str(r["y"]))
        page.fill("#inp-w", str(r["w"]))
        page.fill("#inp-h", str(r["h"]))
        page.click("button:has-text('Применить')")
        page.wait_for_timeout(120)

    page.locator("#btn-mask-preview").click()
    page.wait_for_timeout(200)
    save_shot(page, "ai_02_regions.png")
    count = page.evaluate("regions.length")
    print(f"    Created {count} regions")

    if count != len(WATERMARK_REGIONS):
        print(f"    WARNING: expected {len(WATERMARK_REGIONS)}, got {count}")

    print("[4] AI + CUDA...")
    page.click("#btn-mode-ai")
    page.wait_for_timeout(200)
    page.click("#btn-cuda")
    page.wait_for_timeout(200)

    print("[5] Start processing...")
    page.click("#btn-start")
    page.wait_for_timeout(2000)
    save_shot(page, "ai_03_started.png")

    print("[6] Monitoring...")
    t0 = time.time()
    max_seconds = 60 * 60
    last_pct = ""
    shot_idx = 4
    last_shot = time.time()

    while time.time() - t0 < max_seconds:
        page.wait_for_timeout(5000)
        badge = page.text_content("#status-badge") or ""
        pct = page.text_content("#progress-label") or ""

        if pct != last_pct:
            elapsed = int(time.time() - t0)
            print(f"    [{elapsed}s] {pct} | {badge}")
            last_pct = pct

        if time.time() - last_shot > 60:
            save_shot(page, f"ai_{shot_idx:02d}.png")
            shot_idx += 1
            last_shot = time.time()

        if any(s in badge for s in ["Готово", "Ошибка", "Отменено"]):
            print(f"    >>> FINAL: {badge}")
            break

        if page.locator("#btn-download").is_visible():
            print("    >>> Download available!")
            break

    save_shot(page, "ai_final.png")
    log = page.text_content("#log") or ""
    print(f"\n=== LOG ===\n{log}\n")

    total_time = int(time.time() - t0)
    print(f"Processing: {total_time}s ({total_time/60:.1f} min)")
    print(f"Video: 190s | Ratio: {total_time/190:.1f}x")

    if page.locator("#btn-download").is_visible():
        href = page.locator("#btn-download").get_attribute("href") or ""
        print(f"DOWNLOAD: {BASE_URL}{href}")

    browser.close()
