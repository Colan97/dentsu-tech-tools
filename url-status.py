import streamlit as st
import pandas as pd
import re
import asyncio
import aiohttp
import orjson
import nest_asyncio
import requests
import logging

from typing import List, Dict, Set, Optional
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from datetime import datetime

nest_asyncio.apply()

# -----------------------------
# Logging (optional)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='url_checker.log'
)

# -----------------------------
# Constants
# -----------------------------
DEFAULT_TIMEOUT = 15
DEFAULT_MAX_URLS = 25000
MAX_REDIRECTS = 5
DEFAULT_USER_AGENT = "custom_adidas_seo_x3423/1.0"

USER_AGENTS = {
    "Googlebot Desktop": (
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    ),
    "Googlebot Mobile": (
        "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Mobile Safari/537.36 "
        "(compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    ),
    "Chrome Desktop": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36"
    ),
    "Chrome Mobile": (
        "Mozilla/5.0 (Linux; Android 10; Pixel 3) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Mobile Safari/537.36"
    ),
    "Custom Adidas SEO Bot": DEFAULT_USER_AGENT,
}

# -----------------------------
# Helper Functions
# -----------------------------
def normalize_url(url: str) -> str:
    url = url.strip()
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    # remove fragment
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)

def parse_sitemap(url: str) -> List[str]:
    """
    Simple synchronous parse of <loc> in a sitemap.
    You can adapt this for asynchronous usage if desired.
    """
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

def in_scope(base_url: str, test_url: str, scope_mode: str) -> bool:
    """
    Scope logic for BFS link discovery:
      - "Exact URL Only"
      - "In Subfolder"
      - "Same Subdomain"
      - "All Subdomains"
    """
    from urllib.parse import urlparse
    base_parsed = urlparse(base_url)
    test_parsed = urlparse(test_url)

    if test_parsed.scheme != base_parsed.scheme:
        return False

    base_netloc = base_parsed.netloc.lower()
    test_netloc = test_parsed.netloc.lower()

    if scope_mode == "Exact URL Only":
        return (test_url == base_url)

    elif scope_mode == "In Subfolder":
        if test_netloc != base_netloc:
            return False
        return test_parsed.path.startswith(base_parsed.path)

    elif scope_mode == "Same Subdomain":
        return (test_netloc == base_netloc)

    elif scope_mode == "All Subdomains":
        parts = base_netloc.split('.')
        if len(parts) <= 1:
            return (test_netloc == base_netloc)
        root_domain = '.'.join(parts[-2:])
        return test_netloc.endswith(root_domain)

    return False

def compile_filters(include_pattern: str, exclude_pattern: str):
    inc = re.compile(include_pattern) if include_pattern else None
    exc = re.compile(exclude_pattern) if exclude_pattern else None
    return inc, exc

def regex_filter(url: str, inc, exc) -> bool:
    if inc:
        if not inc.search(url):
            return False
    if exc:
        if exc.search(url):
            return False
    return True

# -----------------------------
# Main URL Checker
# -----------------------------
class URLChecker:
    def __init__(self, user_agent: str, concurrency: int, timeout: int, respect_robots: bool):
        self.user_agent = user_agent
        self.concurrency = concurrency
        self.timeout_duration = timeout
        self.respect_robots = respect_robots
        self.robots_cache = {}
        self.session = None

    async def setup(self):
        connector = aiohttp.TCPConnector(
            limit=self.concurrency,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=False
        )
        client_timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self.timeout_duration,
            sock_read=self.timeout_duration
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=client_timeout,
            json_serialize=orjson.dumps
        )

    async def close(self):
        if self.session:
            await self.session.close()

    async def check_robots(self, url: str) -> bool:
        """
        Return True if allowed, or if not respecting robots.
        """
        if not self.respect_robots:
            return True
        from urllib.parse import urlparse
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        path_lower = parsed.path.lower()

        if base not in self.robots_cache:
            # fetch
            r_url = base + "/robots.txt"
            try:
                headers = {"User-Agent": self.user_agent}
                async with self.session.get(r_url, ssl=False, headers=headers) as resp:
                    if resp.status == 200:
                        txt = await resp.text()
                        self.robots_cache[base] = txt
                    else:
                        self.robots_cache[base] = None
            except:
                self.robots_cache[base] = None

        content = self.robots_cache[base]
        if not content:
            return True

        return self.parse_robots_txt(content, path_lower)

    def parse_robots_txt(self, robots_text: str, path_lower: str) -> bool:
        agent_lower = self.user_agent.lower()
        active = False
        lines = robots_text.splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(':', 1)
            if len(parts) < 2:
                continue
            key, val = parts[0].lower(), parts[1].strip().lower()
            if key == "user-agent":
                if val == '*' or agent_lower in val:
                    active = True
                else:
                    active = False
            elif key == "disallow" and active:
                if val and path_lower.startswith(val):
                    return False
        return True

    def status_label(self, code: int) -> str:
        codes = {
            200: "OK",
            301: "Permanent Redirect",
            302: "Temporary Redirect",
            307: "Temporary Redirect",
            308: "Permanent Redirect",
            404: "Not Found",
            403: "Forbidden",
            500: "Server Error",
            503: "Service Unavailable"
        }
        return codes.get(code, f"Status {code}")

    async def fetch_and_parse(self, url: str) -> Dict:
        # CAUSE OF ARROWINVALID ERROR:
        # Storing codes as str to allow "Timeout"/"Error" etc.
        data = {
            "Original_URL": url,
            "Initial_Status_Code": "",
            "Initial_Status_Type": "",
            "Final_URL": "",
            "Final_Status_Code": "",
            "Final_Status_Type": "",
            "Title": "",
            "Meta_Description": "",
            "H1_Text": "",
            "H1_Count": 0,
            "Canonical_URL": "",
            "Meta_Robots": "",
            "X_Robots_Tag": "",
            "HTML_Lang": "",
            "Is_Blocked_by_Robots": "",
            "Robots_Block_Rule": "",
            "Is_Indexable": "No",
            "Indexability_Reason": "",
            "HTTP_Last_Modified": "",
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # 1) Check robots
        allowed = await self.check_robots(url)
        data["Is_Blocked_by_Robots"] = "No" if allowed else "Yes"
        if not allowed:
            data["Robots_Block_Rule"] = "Disallow"
            data["Indexability_Reason"] = "Blocked by robots.txt"
            data["Final_URL"] = url
            data["Final_Status_Code"] = "N/A"
            data["Final_Status_Type"] = "Robots Block"
            return data

        headers = {"User-Agent": self.user_agent}
        try:
            async with self.session.get(url, headers=headers, ssl=False, allow_redirects=False) as resp:
                init_str = str(resp.status)
                data["Initial_Status_Code"] = init_str
                data["Initial_Status_Type"] = self.status_label(resp.status)
                data["Final_URL"] = str(resp.url)

                if resp.status in (301, 302, 307, 308):
                    loc = resp.headers.get("Location")
                    if loc:
                        return await self.follow_redirect_chain(url, loc, data, headers)
                    else:
                        data["Final_Status_Code"] = init_str
                        data["Final_Status_Type"] = data["Initial_Status_Type"]
                        data["Indexability_Reason"] = "Redirect w/o Location"
                        return data
                else:
                    if resp.status == 200 and resp.content_type and resp.content_type.startswith("text/html"):
                        html = await resp.text(errors='replace')
                        return self.parse_html_content(data, html, resp.headers, resp.status, True)
                    else:
                        data["Final_Status_Code"] = init_str
                        data["Final_Status_Type"] = data["Initial_Status_Type"]
                        data["Indexability_Reason"] = "Non-200 or non-HTML"
                        return data
        except asyncio.TimeoutError:
            data["Initial_Status_Code"] = "Timeout"
            data["Initial_Status_Type"] = "Request Timeout"
            data["Final_URL"] = url
            data["Final_Status_Code"] = "Timeout"
            data["Final_Status_Type"] = "Request Timeout"
            data["Indexability_Reason"] = "Timeout"
            return data
        except Exception as e:
            data["Initial_Status_Code"] = "Error"
            data["Initial_Status_Type"] = str(e)
            data["Final_URL"] = url
            data["Final_Status_Code"] = "Error"
            data["Final_Status_Type"] = str(e)
            data["Indexability_Reason"] = "Exception"
            return data

    async def follow_redirect_chain(self, orig_url: str, location: str, data: Dict, headers: Dict) -> Dict:
        current_url = orig_url
        for _ in range(MAX_REDIRECTS):
            next_url = urljoin(current_url, location)
            next_url = normalize_url(next_url)
            try:
                async with self.session.get(next_url, headers=headers, ssl=False, allow_redirects=False) as r2:
                    code_str = str(r2.status)
                    data["Final_URL"] = str(r2.url)
                    data["Final_Status_Code"] = code_str
                    data["Final_Status_Type"] = self.status_label(r2.status)

                    if r2.status in (301, 302, 307, 308):
                        loc2 = r2.headers.get("Location")
                        if not loc2:
                            data["Indexability_Reason"] = "Redirect w/o Location"
                            return data
                        current_url = next_url
                        location = loc2
                        continue
                    else:
                        if r2.status == 200 and r2.content_type and r2.content_type.startswith("text/html"):
                            html = await r2.text(errors='replace')
                            return self.parse_html_content(data, html, r2.headers, r2.status, True)
                        else:
                            data["Indexability_Reason"] = "Non-200 or non-HTML after redirect"
                            return data
            except asyncio.TimeoutError:
                data["Final_Status_Code"] = "Timeout"
                data["Final_Status_Type"] = "Request Timeout"
                data["Indexability_Reason"] = "Timeout in redirect chain"
                return data
            except Exception as e:
                data["Final_Status_Code"] = "Error"
                data["Final_Status_Type"] = str(e)
                data["Indexability_Reason"] = "Exception in redirect"
                return data

        data["Indexability_Reason"] = "Redirect Loop / Exceeded"
        data["Final_Status_Code"] = "Redirect Loop"
        data["Final_Status_Type"] = "Redirect Loop"
        return data

    def parse_html_content(self, data: Dict, html: str, headers: Dict, status: int, is_allowed: bool) -> Dict:
        soup = BeautifulSoup(html, "lxml")

        title = soup.find("title")
        data["Title"] = title.get_text(strip=True) if title else ""

        desc = soup.find("meta", attrs={"name": "description"})
        if desc and desc.has_attr("content"):
            data["Meta_Description"] = desc["content"]

        h1s = soup.find_all("h1")
        data["H1_Count"] = len(h1s)
        data["H1_Text"] = h1s[0].get_text(strip=True) if h1s else ""

        canon = soup.find("link", attrs={"rel": "canonical"})
        if canon and canon.has_attr("href"):
            data["Canonical_URL"] = canon["href"]

        meta_robots = soup.find("meta", attrs={"name": "robots"})
        if meta_robots and meta_robots.has_attr("content"):
            data["Meta_Robots"] = meta_robots["content"]
        x_robots = headers.get("X-Robots-Tag", "")
        data["X_Robots_Tag"] = x_robots

        html_tag = soup.find("html")
        if html_tag and html_tag.has_attr("lang"):
            data["HTML_Lang"] = html_tag["lang"]

        data["HTTP_Last_Modified"] = headers.get("Last-Modified", "")

        # Evaluate indexability
        combined = f"{data['Meta_Robots'].lower()} {x_robots.lower()}"
        if "noindex" in combined:
            data["Is_Indexable"] = "No"
            data["Indexability_Reason"] = "Noindex directive"
        elif status != 200:
            data["Is_Indexable"] = "No"
            data["Indexability_Reason"] = f"Status {status}"
        elif not is_allowed:
            data["Is_Indexable"] = "No"
            data["Indexability_Reason"] = "Blocked by robots.txt"
        else:
            data["Is_Indexable"] = "Yes"
            data["Indexability_Reason"] = "Page is indexable"

        return data

# -----------------------------
# BFS (Layer-Based)
# -----------------------------
async def layer_bfs(
    seeds: List[str],
    checker: URLChecker,
    scope_mode: str,
    include_regex: Optional[str],
    exclude_regex: Optional[str],
    show_partial_callback=None
) -> List[Dict]:
    # unlimited depth => stop only if no new pages or we hit default max
    visited: Set[str] = set()
    current_layer = set(normalize_url(u) for u in seeds if u.strip())
    results = []

    inc, exc = compile_filters(include_regex, exclude_regex)

    await checker.setup()

    while current_layer and len(visited) < DEFAULT_MAX_URLS:
        layer_list = list(current_layer)
        current_layer.clear()

        tasks = [checker.fetch_and_parse(u) for u in layer_list]
        layer_results = await asyncio.gather(*tasks, return_exceptions=True)

        valid = [r for r in layer_results if isinstance(r, dict)]
        results.extend(valid)
        for u in layer_list:
            visited.add(u)

        # discover next layer
        next_layer = set()
        for row in valid:
            final_url = row.get("Final_URL") or row["Original_URL"]
            # gather links
            discovered = await discover_links(final_url, checker.session, checker.user_agent)
            # filter scope, regex
            base_seed = seeds[0]
            for link in discovered:
                link_n = normalize_url(link)
                if not in_scope(base_seed, link_n, scope_mode):
                    continue
                if not regex_filter(link_n, inc, exc):
                    continue
                if link_n not in visited and len(visited) + len(next_layer) < DEFAULT_MAX_URLS:
                    next_layer.add(link_n)

        current_layer = next_layer
        if show_partial_callback:
            show_partial_callback(results, len(visited), DEFAULT_MAX_URLS)

    await checker.close()
    return results

async def discover_links(url: str, session: aiohttp.ClientSession, user_agent: str) -> List[str]:
    out = []
    headers = {"User-Agent": user_agent}
    try:
        async with session.get(url, headers=headers, ssl=False, allow_redirects=False) as resp:
            if resp.status == 200 and resp.content_type and resp.content_type.startswith("text/html"):
                text = await resp.text(errors='replace')
                soup = BeautifulSoup(text, "lxml")
                for a in soup.find_all("a", href=True):
                    abs_link = urljoin(url, a["href"])
                    out.append(abs_link)
    except:
        pass
    return out

# -----------------------------
# Chunk Mode (No BFS)
# -----------------------------
async def chunk_process(urls: List[str], checker: URLChecker, show_partial_callback=None) -> List[Dict]:
    results = []
    visited = set()
    final_list = []
    for u in urls:
        nu = normalize_url(u)
        if nu not in visited:
            visited.add(nu)
            final_list.append(nu)

    await checker.setup()

    chunk_size = 100
    processed = 0
    total = len(final_list)

    for i in range(0, total, chunk_size):
        chunk = final_list[i : i + chunk_size]
        tasks = [checker.fetch_and_parse(u) for u in chunk]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in chunk_results if isinstance(r, dict)]
        results.extend(valid)
        processed += len(chunk)
        if show_partial_callback:
            show_partial_callback(results, processed, total)

    await checker.close()
    return results

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Three-Mode Crawler: Spider, List, or Sitemap")

    # Sidebar: concurrency, user agent, respect robots, scope
    st.sidebar.header("Configuration")
    concurrency = st.sidebar.slider("Concurrency", 1, 50, 10)
    ua_choice = st.sidebar.selectbox("User Agent", list(USER_AGENTS.keys()))
    user_agent = USER_AGENTS[ua_choice]
    respect_robots = st.sidebar.checkbox("Respect robots.txt", value=True)

    scope_mode = st.sidebar.radio(
        "Crawl Scope",
        ["Exact URL Only", "In Subfolder", "Same Subdomain", "All Subdomains"],
        index=2
    )

    # Top-level radio: Spider (BFS), List, Sitemaps
    mode = st.radio("Select Mode", ["Spider (BFS)", "List", "Sitemap"], horizontal=True)

    st.write("----")

    # Container for user inputs
    user_urls = []
    user_sitemaps = []

    if mode == "Spider (BFS)":
        st.subheader("Spider (BFS) Mode")
        st.write("Enter **seed** URLs, plus optionally you can include sitemaps.")
        text_input = st.text_area("Seed URLs (one per line)", "")
        if text_input.strip():
            user_urls = [line.strip() for line in text_input.splitlines() if line.strip()]

        # Checkbox to include sitemaps in BFS
        include_sitemaps = st.checkbox("Include Sitemaps? (Multiple lines allowed)")
        if include_sitemaps:
            st.write("Enter one or multiple sitemap URLs (one per line).")
            sitemaps_text = st.text_area("Sitemap URLs", "")
            if sitemaps_text.strip():
                raw_sitemaps = [s.strip() for s in sitemaps_text.splitlines() if s.strip()]
                # Parse each
                for sm in raw_sitemaps:
                    # You can fetch the sitemaps now or do it in BFS seeds:
                    # We'll just parse them now in chunk mode:
                    # or we can treat them as BFS seeds directly.
                    # If you want BFS to BFS from the sitemap's URLs, you'd parse them now
                    # or just treat them as seeds:
                    # For simplicity, let's parse them to get actual URLs:
                    parsed = parse_sitemap(sm)
                    user_sitemaps.extend(parsed)
                st.write(f"Collected {len(user_sitemaps)} URLs from the sitemaps.")
        
        # "Advanced" button (expander) for BFS filters
        with st.expander("Advanced Filters (Include/Exclude Regex)"):
            st.write("Regex patterns to filter discovered URLs during BFS.")
            include_pattern = st.text_input("Include Regex", "")
            exclude_pattern = st.text_input("Exclude Regex", "")

        if st.button("Start BFS Spider"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            seeds = user_urls + user_sitemaps  # BFS seeds
            if not seeds:
                st.warning("No seeds provided for BFS.")
                return

            progress_ph = st.empty()
            table_ph = st.empty()

            def show_partial(res_list, done_count, total_count):
                pct = int((done_count / total_count) * 100) if total_count else 100
                progress_ph.progress(min(pct, 100))
                df_temp = pd.DataFrame(res_list)
                table_ph.dataframe(df_temp.tail(10), use_container_width=True)

            checker = URLChecker(
                user_agent=user_agent,
                concurrency=concurrency,
                timeout=DEFAULT_TIMEOUT,
                respect_robots=respect_robots
            )

            results = loop.run_until_complete(
                layer_bfs(
                    seeds=seeds,
                    checker=checker,
                    scope_mode=scope_mode,
                    include_regex=include_pattern,
                    exclude_regex=exclude_pattern,
                    show_partial_callback=show_partial
                )
            )
            loop.close()
            progress_ph.empty()

            if not results:
                st.warning("No results from BFS.")
                return

            df = pd.DataFrame(results)
            st.subheader("BFS Results")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"bfs_{now_str}.csv",
                mime="text/csv"
            )
            show_summary(df)

    elif mode == "List":
        st.subheader("List Mode")
        st.write("Paste URLs to crawl, with **no BFS** link discovery.")
        text_input = st.text_area("List of URLs", "")
        if st.button("Start Crawl"):
            if not text_input.strip():
                st.warning("No URLs provided.")
                return

            user_urls = [line.strip() for line in text_input.splitlines() if line.strip()]

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            progress_ph = st.empty()
            table_ph = st.empty()

            def show_partial(res_list, done_count, total_count):
                pct = int((done_count / total_count) * 100) if total_count else 100
                progress_ph.progress(min(pct, 100))
                df_temp = pd.DataFrame(res_list)
                table_ph.dataframe(df_temp.tail(10), use_container_width=True)

            checker = URLChecker(
                user_agent=user_agent,
                concurrency=concurrency,
                timeout=DEFAULT_TIMEOUT,
                respect_robots=respect_robots
            )
            results = loop.run_until_complete(
                chunk_process(user_urls, checker, show_partial_callback=show_partial)
            )
            loop.close()
            progress_ph.empty()

            if not results:
                st.warning("No results.")
                return

            df = pd.DataFrame(results)
            st.subheader("List Mode Results")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"list_results_{now_str}.csv",
                mime="text/csv"
            )
            show_summary(df)

    else:  # mode == "Sitemap"
        st.subheader("Sitemap Mode")
        st.write("Enter one or multiple sitemap URLs, parse them, and crawl those URLs (no BFS).")

        text_input = st.text_area("Sitemap URLs (one per line)")
        if st.button("Fetch & Crawl Sitemaps"):
            if not text_input.strip():
                st.warning("No sitemap URLs provided.")
                return

            lines = [line.strip() for line in text_input.splitlines() if line.strip()]
            all_sitemap_urls = []
            for sm in lines:
                parsed = parse_sitemap(sm)
                all_sitemap_urls.extend(parsed)
            st.write(f"Collected total {len(all_sitemap_urls)} URLs from all sitemaps.")

            if not all_sitemap_urls:
                st.warning("No URLs found in these sitemaps.")
                return

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            progress_ph = st.empty()
            table_ph = st.empty()

            def show_partial(res_list, done_count, total_count):
                pct = int((done_count / total_count) * 100) if total_count else 100
                progress_ph.progress(min(pct, 100))
                df_temp = pd.DataFrame(res_list)
                table_ph.dataframe(df_temp.tail(10), use_container_width=True)

            checker = URLChecker(
                user_agent=user_agent,
                concurrency=concurrency,
                timeout=DEFAULT_TIMEOUT,
                respect_robots=respect_robots
            )
            results = loop.run_until_complete(
                chunk_process(all_sitemap_urls, checker, show_partial_callback=show_partial)
            )
            loop.close()
            progress_ph.empty()

            if not results:
                st.warning("No results from sitemaps.")
                return

            df = pd.DataFrame(results)
            st.subheader("Sitemap Results")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"sitemap_results_{now_str}.csv",
                mime="text/csv"
            )
            show_summary(df)


def show_summary(df: pd.DataFrame):
    st.subheader("Summary")
    code_counts_init = df["Initial_Status_Code"].value_counts(dropna=False)
    st.write("**Initial Status Code Distribution**")
    for code, cnt in code_counts_init.items():
        st.write(f"{code}: {cnt}")

    code_counts_final = df["Final_Status_Code"].value_counts(dropna=False)
    st.write("**Final Status Code Distribution**")
    for code, cnt in code_counts_final.items():
        st.write(f"{code}: {cnt}")

    block_counts = df["Is_Blocked_by_Robots"].value_counts(dropna=False)
    st.write("**Blocked by Robots.txt?**")
    for val, cnt in block_counts.items():
        st.write(f"{val}: {cnt}")

    index_counts = df["Is_Indexable"].value_counts(dropna=False)
    st.write("**Indexable?**")
    for val, cnt in index_counts.items():
        st.write(f"{val}: {cnt}")


if __name__ == "__main__":
    main()
