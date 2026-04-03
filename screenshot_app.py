from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto("http://127.0.0.1:8000")
    page.wait_for_load_state("networkidle")

    # Load video
    page.locator("#local-path").fill(r"C:\Users\Пользователь\Desktop\watermark\Араб.mp4")
    page.locator("button", has_text="Открыть").click()
    page.wait_for_timeout(3000)

    # Add 6 regions covering the watermarks
    page.evaluate("""() => {
        const rs = [
            {x:30,  y:10,  w:150, h:35},
            {x:450, y:10,  w:150, h:35},
            {x:870, y:10,  w:150, h:35},
            {x:30,  y:390, w:150, h:35},
            {x:450, y:390, w:150, h:35},
            {x:870, y:390, w:150, h:35},
        ];
        rs.forEach(r => regions.push(r));
        activeIdx = 0;
        syncCoordsToInputs();
        renderRegionsList();
        redraw();
    }""")
    page.wait_for_timeout(400)

    page.screenshot(
        path=r"C:\Users\Пользователь\Desktop\watermark\temp_web\app_demo.png",
        full_page=True,
    )
    browser.close()
