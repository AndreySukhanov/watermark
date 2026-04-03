from playwright.sync_api import sync_playwright
from test_helpers import BASE_URL, save_shot

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(BASE_URL, timeout=30000)
    page.wait_for_load_state("networkidle")
    path = save_shot(page, "app_screenshot.png")
    print(f"Screenshot saved: {path}")
    browser.close()
