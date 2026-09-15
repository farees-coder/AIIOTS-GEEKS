import asyncio
from scraper.scraper import Scraper

async def main():
    try:
        s = Scraper('https://example.com', max_pages=1, max_depth=1)
        res = await s.scrape_and_clean()
        print("Success:", res.keys())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
