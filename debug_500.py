import sys
from playwright.sync_api import sync_playwright

errors = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.on("response", lambda r: errors.append((r.status, r.url)) if r.status >= 400 else None)

    page.goto("http://127.0.0.1:8000")
    page.wait_for_load_state("networkidle")

    page.locator("#local-path").fill(r"C:\Users\Пользователь\Desktop\watermark\Араб.mp4")
    page.locator("button", has_text="Открыть").click()
    page.wait_for_timeout(4000)

    sys.stdout.buffer.write(b"=== HTTP errors ===\n")
    for status, url in errors:
        sys.stdout.buffer.write(f"  {status}  {url}\n".encode("utf-8"))
    if not errors:
        sys.stdout.buffer.write(b"  none\n")

    browser.close()
