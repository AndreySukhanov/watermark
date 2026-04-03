from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto("https://l2ethv50lv9nql-8000.proxy.runpod.net/", timeout=30000)
    page.wait_for_load_state("networkidle")
    page.screenshot(path="C:/Users/Пользователь/Desktop/watermark/screenshot_app.png", full_page=True)
    print("Screenshot saved")
    browser.close()
