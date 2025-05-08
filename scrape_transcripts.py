import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import os

async def scrape_transcript(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(url, timeout=60000)
        await page.wait_for_timeout(2000)

        try:
            await page.click("text='Show Transcript'", timeout=3000)
            await page.wait_for_timeout(1000)
        except:
            pass

        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")

        transcript = ""
        possible_blocks = soup.find_all("div")
        for block in possible_blocks:
            text = block.get_text(strip=True)
            if text.startswith("One of the most important skills") or "AJ Wilcox" in text:
                transcript = block.get_text(separator="\n").strip()
                break

        await browser.close()
        return transcript

async def main():
    base_url = "https://b2linked.com/blog/ep{}"
    os.makedirs("transcripts", exist_ok=True)

    for ep in range(1, 10):  # Episodes 01 to 09
        ep_str = f"{ep:02}"  # Pads to "01", "02", ...
        url = base_url.format(ep_str)
        print(f"Scraping episode {ep_str}...")

        try:
            transcript = await scrape_transcript(url)
            if transcript:
                with open(f"transcripts/ep{ep_str}.txt", "w", encoding="utf-8") as f:
                    f.write(transcript)
            else:
                with open(f"transcripts/ep{ep_str}.txt", "w", encoding="utf-8") as f:
                    f.write("Transcript not found")
        except Exception as e:
            with open(f"transcripts/ep{ep_str}.txt", "w", encoding="utf-8") as f:
                f.write(f"Error: {e}")

asyncio.run(main())
