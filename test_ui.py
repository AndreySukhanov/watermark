"""
UI tests for multi-region + mode toggle changes.
"""
import sys
from playwright.sync_api import sync_playwright

ERRORS = []
PASSED = []

def check(name, cond, detail=""):
    if cond:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        ERRORS.append(name)
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://127.0.0.1:8000")
    page.wait_for_load_state("networkidle")

    print("\n=== 1. Page structure ===")
    check("regions-list present", page.locator("#regions-list").count() == 1)
    check("regions-list shows empty hint",
          "Нет регионов" in (page.locator("#regions-list").inner_text() or ""))
    check("coord inputs present",
          all(page.locator(f"#inp-{k}").count() == 1 for k in ["x","y","w","h"]))
    check("Apply / Reset buttons present",
          page.locator("button", has_text="Применить").count() >= 1 and
          page.locator("button", has_text="Очистить все").count() >= 1)
    check("mask preview toggle present", page.locator("#btn-mask-preview").count() == 1)
    check("mask meta present", page.locator("#mask-meta").count() == 1)

    print("\n=== 2. Mode toggle ===")
    check("delogo btn present", page.locator("#btn-mode-delogo").count() == 1)
    check("ai btn present",     page.locator("#btn-mode-ai").count() == 1)
    check("delogo btn active by default",
          "active" in (page.locator("#btn-mode-delogo").get_attribute("class") or ""))
    check("ai warning hidden by default",
          page.locator("#ai-warning").is_hidden())

    # Click AI mode
    page.locator("#btn-mode-ai").click()
    page.wait_for_timeout(200)
    check("ai btn becomes active after click",
          "active" in (page.locator("#btn-mode-ai").get_attribute("class") or ""))
    check("ai warning shown after click",
          page.locator("#ai-warning").is_visible())
    check("delogo btn loses active",
          "active" not in (page.locator("#btn-mode-delogo").get_attribute("class") or ""))

    # Switch back
    page.locator("#btn-mode-delogo").click()
    page.wait_for_timeout(200)
    check("delogo active again", "active" in (page.locator("#btn-mode-delogo").get_attribute("class") or ""))
    check("ai warning hidden again", page.locator("#ai-warning").is_hidden())

    print("\n=== 3. Load video via path ===")
    video_path = r"C:\Users\Пользователь\Desktop\watermark\Араб.mp4"
    page.locator("#local-path").fill(video_path)
    page.locator("button", has_text="Открыть").click()
    page.wait_for_timeout(3000)
    meta = page.locator("#metadata").inner_text()
    check("metadata populated", meta != "—", meta)
    check("canvas visible", page.locator("#canvas").is_visible())

    print("\n=== 4. Draw regions via JS ===")
    # Use JS to simulate 2 regions being committed directly
    page.evaluate("""() => {
        regions.push({x:50, y:20, w:120, h:30});
        regions.push({x:300, y:20, w:120, h:30});
        activeIdx = 1;
        syncCoordsToInputs();
        renderRegionsList();
        redraw();
    }""")
    page.wait_for_timeout(300)
    list_html = page.locator("#regions-list").inner_html()
    check("region 1 appears in list", "region-row" in list_html)
    check("two region rows", list_html.count("region-row") >= 2)
    check("active region syncs X to inputs",
          page.locator("#inp-x").input_value() == "300")
    check("mask meta shows region count",
          "рег." in (page.locator("#mask-meta").inner_text() or ""))

    print("\n=== 4.1 Mask preview toggle ===")
    page.locator("#btn-mask-preview").click()
    page.wait_for_timeout(200)
    mask_btn_class = page.locator("#btn-mask-preview").get_attribute("class") or ""
    mask_btn_text = page.locator("#btn-mask-preview").inner_text()
    check("mask preview button becomes active", "is-active" in mask_btn_class)
    check("mask preview button text switches to hide", "Скрыть маску" in mask_btn_text)

    page.locator("#btn-mask-preview").click()
    page.wait_for_timeout(200)
    mask_btn_class = page.locator("#btn-mask-preview").get_attribute("class") or ""
    check("mask preview button can be turned off", "is-active" not in mask_btn_class)

    print("\n=== 5. Select / delete region ===")
    # Select region 0
    page.evaluate("() => selectRegion(0)")
    page.wait_for_timeout(200)
    check("selecting region 0 syncs X=50",
          page.locator("#inp-x").input_value() == "50")

    # Delete region 0
    page.evaluate("() => deleteRegion(0)")
    page.wait_for_timeout(200)
    list_html2 = page.locator("#regions-list").inner_html()
    check("one region remains after delete", list_html2.count("region-row") == 1)

    print("\n=== 6. applyCoords adds new region when none active ===")
    page.evaluate("() => { regions=[]; activeIdx=-1; renderRegionsList(); }")
    page.locator("#inp-x").fill("10")
    page.locator("#inp-y").fill("10")
    page.locator("#inp-w").fill("80")
    page.locator("#inp-h").fill("40")
    page.locator("button", has_text="Применить").click()
    page.wait_for_timeout(200)
    count = page.evaluate("() => regions.length")
    check("applyCoords creates region when list empty", count == 1)

    print("\n=== 7. resetSelection clears all ===")
    page.locator("button", has_text="Очистить все").click()
    page.wait_for_timeout(200)
    check("regions cleared", page.evaluate("() => regions.length") == 0)
    check("empty hint shown again", "Нет регионов" in page.locator("#regions-list").inner_text())

    print("\n=== 8. Start validation — no regions ===")
    page.locator("#btn-start").click()
    page.wait_for_timeout(300)
    # Should show alert (dialog)
    dialog_fired = {"v": False}
    def on_dialog(d):
        dialog_fired["v"] = True
        d.dismiss()
    page.on("dialog", on_dialog)
    page.evaluate("() => { regions=[]; startProcessing(); }")
    page.wait_for_timeout(500)
    check("alert shown when no regions", dialog_fired["v"])

    page.screenshot(path=r"C:\Users\Пользователь\Desktop\watermark\temp_web\test_result.png", full_page=True)
    browser.close()

print(f"\n{'='*40}")
print(f"PASSED: {len(PASSED)}   FAILED: {len(ERRORS)}")
if ERRORS:
    print("Failed:", ERRORS)
    sys.exit(1)
