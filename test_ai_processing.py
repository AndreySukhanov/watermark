"""
Playwright test: use server-side file path, add 11 watermark regions, AI+GPU, process.
"""
import sys
import time
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from playwright.sync_api import sync_playwright

URL = "https://953sjwh97442po-8000.proxy.runpod.net/"
# Use already-uploaded file on server to avoid slow re-upload
SERVER_VIDEO = "/workspace/temp_web/803ef888-b49d-4bf3-87b5-0d72e95f780a.mp4"
SHOTS = r"C:\Users\Пользователь\Desktop\watermark"

# All "artInflext.com" watermark positions on 1920x1080 frame
WATERMARK_REGIONS = [
    {"x": 0,    "y": 80,  "w": 320, "h": 110},
    {"x": 850,  "y": 40,  "w": 320, "h": 110},
    {"x": 1660, "y": 10,  "w": 260, "h": 110},
    {"x": 400,  "y": 260, "w": 320, "h": 110},
    {"x": 1250, "y": 230, "w": 320, "h": 110},
    {"x": 0,    "y": 400, "w": 320, "h": 110},
    {"x": 510,  "y": 430, "w": 320, "h": 110},
    {"x": 1020, "y": 410, "w": 320, "h": 110},
    {"x": 80,   "y": 580, "w": 320, "h": 110},
    {"x": 820,  "y": 610, "w": 320, "h": 110},
    {"x": 1440, "y": 590, "w": 320, "h": 110},
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})

    # 1. Open app
    print("[1] Opening app...")
    page.goto(URL, timeout=30000)
    page.wait_for_load_state("networkidle")
    print("    OK")

    # 2. Load video via server path (no upload needed)
    print(f"[2] Loading via path: {SERVER_VIDEO}")
    page.fill("#local-path", SERVER_VIDEO)
    page.click("button:has-text('Открыть')")

    # Wait for preview
    page.wait_for_function(
        "() => document.getElementById('canvas').style.display === 'block'",
        timeout=30000
    )
    page.wait_for_timeout(2000)
    meta = page.text_content("#metadata") or ""
    print(f"    Meta: {meta}")
    page.screenshot(path=f"{SHOTS}/shot_01_loaded.png", full_page=True)

    # 3. Add all 11 watermark regions
    print(f"[3] Adding {len(WATERMARK_REGIONS)} regions...")
    for i, r in enumerate(WATERMARK_REGIONS):
        page.evaluate("activeIdx = -1")  # Force ADD mode (not update)
        page.fill("#inp-x", str(r["x"]))
        page.fill("#inp-y", str(r["y"]))
        page.fill("#inp-w", str(r["w"]))
        page.fill("#inp-h", str(r["h"]))
        page.click("button:has-text('Применить')")
        page.wait_for_timeout(150)

    page.screenshot(path=f"{SHOTS}/shot_02_regions.png", full_page=True)
    count = page.evaluate("regions.length")
    print(f"    Created {count} regions")

    if count != len(WATERMARK_REGIONS):
        print(f"    WARNING: expected {len(WATERMARK_REGIONS)}, got {count}")

    # 4. AI mode + GPU
    print("[4] AI + CUDA...")
    page.click("#btn-mode-ai")
    page.wait_for_timeout(200)
    page.click("#btn-cuda")
    page.wait_for_timeout(200)

    # 5. Start
    print("[5] Start processing...")
    page.click("#btn-start")
    page.wait_for_timeout(2000)
    page.screenshot(path=f"{SHOTS}/shot_03_started.png", full_page=True)

    # 6. Monitor (max 60 min)
    print("[6] Monitoring...")
    t0 = time.time()
    MAX = 60 * 60
    last_pct = ""
    sn = 4
    last_shot = time.time()

    while time.time() - t0 < MAX:
        page.wait_for_timeout(5000)
        badge = page.text_content("#status-badge") or ""
        pct = page.text_content("#progress-label") or ""

        if pct != last_pct:
            elapsed = int(time.time() - t0)
            print(f"    [{elapsed}s] {pct} | {badge}")
            last_pct = pct

        if time.time() - last_shot > 60:
            sn += 1
            page.screenshot(path=f"{SHOTS}/shot_{sn:02d}.png")
            last_shot = time.time()

        if any(s in badge for s in ["Готово", "Ошибка", "Отменено"]):
            print(f"    >>> FINAL: {badge}")
            break

        if page.locator("#btn-download").is_visible():
            print("    >>> Download available!")
            break

    page.screenshot(path=f"{SHOTS}/shot_final.png", full_page=True)
    log = page.text_content("#log") or ""
    print(f"\n=== LOG ===\n{log}\n")

    total_time = int(time.time() - t0)
    print(f"Processing: {total_time}s ({total_time/60:.1f} min)")
    print(f"Video: 190s | Ratio: {total_time/190:.1f}x")

    if page.locator("#btn-download").is_visible():
        href = page.locator("#btn-download").get_attribute("href") or ""
        print(f"DOWNLOAD: {URL.rstrip('/')}{href}")

    browser.close()
