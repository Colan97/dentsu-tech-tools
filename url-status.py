import streamlit as st
import pandas as pd
import asyncio
import aiohttp
import orjson
import nest_asyncio
import requests
import logging
import time

from typing import List, Dict, Set, Optional
from urllib.parse import urlparse, urljoin, urlunparse
from bs4 import BeautifulSoup
from datetime import datetime

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_TIMEOUT = 15
DEFAULT_MAX_URLS = 25000
MAX_REDIRECTS = 5
DEFAULT_USER_AGENT = "custom_adidas_seo_x3423/1.0"

USER_AGENTS = {
    "Googlebot Desktop": "...",
    "Googlebot Mobile": "...",
    "Chrome Desktop": "...",
    "Chrome Mobile": "...",
    "Custom Adidas SEO Bot": DEFAULT_USER_AGENT,
}

def parse_sitemap(url: str) -> List[str]:
    out = []
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            for loc_tag in root.findall(".//{*}loc"):
                if loc_tag.text:
                    out.append(loc_tag.text.strip())
    except:
        pass
    return out

def normalize_url(url: str) -> str:
    from urllib.parse import urlparse, urlunparse
    url = url.strip()
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)

def in_scope(base_url: str, test_url: str) -> bool:
    # Simple check: same domain only. Extend as needed for subfolders, subdomains, etc.
    base_parsed = urlparse(base_url)
    test_parsed = urlparse(test_url)
    return base_parsed.netloc.lower() == test_parsed.netloc.lower()

class URLChecker:
    def __init__(self, user_agent: str, concurrency: int = 5, timeout: int = 15):
        self.user_agent = user_agent
        self.concurrency = concurrency
        self.timeout = timeout
        self.session = None
        self.robots_cache = {}

    async def setup(self):
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        client_timeout = aiohttp.ClientTimeout(total=None, connect=self.timeout, sock_read=self.timeout)
        self.session = aiohttp.ClientSession(connector=connector, timeout=client_timeout, json_serialize=orjson.dumps)

    async def close(self):
        if self.session:
            await self.session.close()

    async def fetch_and_parse(self, url: str) -> Dict:
        data = {
            "Original_URL": url,
            "Final_URL": "",
            "Initial_Status_Code": "",
            "Final_Status_Code": "",
            "Title": "",
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        headers = {"User-Agent": self.user_agent}
        try:
            async with self.session.get(url, headers=headers, ssl=False, allow_redirects=False) as resp:
                data["Initial_Status_Code"] = str(resp.status)
                data["Final_URL"] = str(resp.url)
                if resp.status == 200 and resp.content_type and resp.content_type.startswith("text/html"):
                    html = await resp.text(errors="replace")
                    soup = BeautifulSoup(html, "lxml")
                    title_tag = soup.find("title")
                    if title_tag:
                        data["Title"] = title_tag.get_text(strip=True)
                else:
                    data["Final_Status_Code"] = str(resp.status)
                return data
        except asyncio.TimeoutError:
            data["Initial_Status_Code"] = "Timeout"
            data["Final_Status_Code"] = "Timeout"
            return data
        except Exception as e:
            data["Initial_Status_Code"] = "Error"
            data["Final_Status_Code"] = str(e)
            return data

async def discover_links(url: str, session: aiohttp.ClientSession, user_agent: str) -> List[str]:
    out = []
    headers = {"User-Agent": user_agent}
    try:
        async with session.get(url, headers=headers, ssl=False, allow_redirects=False) as resp:
            if resp.status == 200 and resp.content_type and resp.content_type.startswith("text/html"):
                html = await resp.text(errors="replace")
                soup = BeautifulSoup(html, "lxml")
                for a in soup.find_all("a", href=True):
                    abs_link = urljoin(url, a["href"])
                    out.append(abs_link)
    except:
        pass
    return out

async def bfs_crawl(
    seed_urls: List[str],
    checker: URLChecker,
    max_urls: int = DEFAULT_MAX_URLS,
    show_partial_callback=None
) -> List[Dict]:
    visited: Set[str] = set()
    queue = set(normalize_url(u) for u in seed_urls if u.strip())
    results = []
    retry_urls = []  # We'll store timeouts/errors here for a second pass

    await checker.setup()

    while queue and len(visited) < max_urls:
        current_layer = list(queue)
        queue.clear()
        tasks = [checker.fetch_and_parse(u) for u in current_layer]
        layer_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res in enumerate(layer_results):
            # If exception, skip
            if not isinstance(res, dict):
                continue
            results.append(res)
            url = current_layer[i]
            visited.add(url)

            # If it timed out, store for retry
            if res.get("Final_Status_Code") == "Timeout" or res.get("Initial_Status_Code") == "Timeout":
                retry_urls.append(url)
                continue

            # If status=200, discover new links
            # (You can refine to parse only if HTML was received.)
            if res.get("Initial_Status_Code") == "200":
                discovered = await discover_links(res["Final_URL"], checker.session, checker.user_agent)
                for link in discovered:
                    link_n = normalize_url(link)
                    if link_n not in visited and len(visited) + len(queue) < max_urls:
                        # e.g. check in_scope if you only want same domain
                        if in_scope(seed_urls[0], link_n):
                            queue.add(link_n)

        # Optional small pause to avoid overwhelming server
        # await asyncio.sleep(0.5)  # if you want further throttling

        if show_partial_callback:
            show_partial_callback(results, len(visited), max_urls)

    await checker.close()

    # 2nd pass: recrawl any timeouts with chunk-based approach or a single pass
    # We'll do a quick re-setup, possibly with smaller concurrency or same
    if retry_urls:
        await checker.setup()
        # or set checker.concurrency = 2 if you want slower approach on retry
        tasks2 = [checker.fetch_and_parse(u) for u in retry_urls]
        retry_results = await asyncio.gather(*tasks2, return_exceptions=True)
        await checker.close()

        # Merge updated results for timeouts
        for res2 in retry_results:
            if isinstance(res2, dict):
                # Find old dict with same "Original_URL"
                original = res2["Original_URL"]
                # Overwrite or append. We'll just append a new row or update.
                # If you prefer updating in place:
                for r in results:
                    if r["Original_URL"] == original:
                        r.update(res2)
                        break
                else:
                    results.append(res2)

    return results

def main():
    st.title("Smooth BFS Crawler with Timeout Recrawl")

    concurrency = st.slider("Concurrency (requests in parallel)", 1, 20, 5)
    ua_choice = st.selectbox("User Agent", list(USER_AGENTS.keys()))
    user_agent = USER_AGENTS[ua_choice]

    st.write("**Seed URLs**")
    seed_text = st.text_area("Enter seed URLs (one per line). Optionally include sitemaps below.")
    seeds = [u.strip() for u in seed_text.splitlines() if u.strip()]

    st.write("**Sitemap URLs** (optional)")
    sitemap_text = st.text_area("Enter sitemap URLs (one per line, merges with BFS seeds)")
    if sitemap_text.strip():
        lines = [x.strip() for x in sitemap_text.splitlines() if x.strip()]
        for sm in lines:
            seeds.extend(parse_sitemap(sm))

    if st.button("Start BFS Crawl"):
        if not seeds:
            st.warning("No seeds found.")
            return

        progress_ph = st.empty()
        progress_bar = st.progress(0.0)
        table_ph = st.empty()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def show_partial(results_list, visited_count, max_count):
            ratio = visited_count / max_count if max_count else 1.0
            progress_bar.progress(ratio)
            remain = max_count - visited_count
            pct = ratio * 100
            progress_ph.write(
                f"Visited/Crawled: {visited_count} of {max_count} "
                f"({pct:.2f}%) → {remain} remaining"
            )
            df_temp = pd.DataFrame(results_list)
            table_ph.dataframe(df_temp.tail(10), use_container_width=True)

        checker = URLChecker(user_agent=user_agent, concurrency=concurrency, timeout=DEFAULT_TIMEOUT)

        results = loop.run_until_complete(
            bfs_crawl(
                seed_urls=seeds,
                checker=checker,
                max_urls=DEFAULT_MAX_URLS,
                show_partial_callback=show_partial
            )
        )
        loop.close()

        if not results:
            st.warning("No results from BFS.")
            return

        df = pd.DataFrame(results)
        st.subheader("Final Results")
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"bfs_results_{now_str}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
