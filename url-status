import streamlit as st
import pandas as pd
import re
import asyncio
import nest_asyncio
from datetime import datetime
from typing import List, Dict
import gc
import logging

# 3rd-party packages used by URLChecker
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from tenacity import retry, stop_after_attempt, wait_exponential
from collections import deque
import orjson

# Apply nest_asyncio so we can run async in Streamlit
nest_asyncio.apply()

# Configure logging (optional, logs to 'url_checker.log' file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='url_checker.log'
)

# --------------------------
# Global constants
# --------------------------
MAX_CONCURRENT_REQUESTS = 200
TIMEOUT_DURATION = 15
CHUNK_SIZE = 250
USER_AGENT = 'custom_adidas_seo_x3423/1.0'
RETRY_TIMEOUT_DURATION = 30  # For second-pass timeouts

class URLChecker:
    """
    A class that checks various SEO and technical aspects of URLs.
    """

    def __init__(self):
        self.results = deque()
        self.session = None
        self.semaphore = None
        self.robots_cache = {}

    async def setup(self, timeout_duration=TIMEOUT_DURATION):
        """
        Create an aiohttp session with concurrency limits and timeouts.
        """
        self.connector = aiohttp.TCPConnector(
            limit=MAX_CONCURRENT_REQUESTS,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=False
        )

        # Connect & read timeouts
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=timeout_duration,
            sock_read=timeout_duration
        )

        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            json_serialize=orjson.dumps
        )

        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def check_robots_txt(self, url: str) -> bool:
        """
        Fetch robots.txt if not cached, then check if the URL path is allowed.
        """
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = f"{base_url}/robots.txt"

        if base_url in self.robots_cache:
            return self.robots_cache[base_url]

        try:
            headers = {'User-Agent': USER_AGENT}
            async with self.session.get(robots_url, ssl=False, headers=headers) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    is_allowed = self.check_robots_rules(robots_content, parsed_url.path)
                    self.robots_cache[base_url] = is_allowed
                    return is_allowed
                # If non-200, assume not blocked
                return True
        except Exception as e:
            logging.warning(f"Could not fetch robots.txt from {robots_url}: {str(e)}")
            return True

    def check_robots_rules(self, robots_content: str, path: str) -> bool:
        """
        Very basic logic to see if path is disallowed under 'User-agent: *' or
        our custom user agent in the robots.txt.
        """
        user_agent_section = False
        for line in robots_content.split('\n'):
            line = line.strip().lower()
            if not line or line.startswith('#'):
                continue
            if line.startswith('user-agent:'):
                agent = line.split(':', 1)[1].strip()
                # if agent is * or specifically our user agent
                user_agent_section = (agent == '*') or (USER_AGENT.lower() in agent)
            elif user_agent_section and line.startswith('disallow:'):
                disallow_path = line.split(':', 1)[1].strip()
                if disallow_path and path.startswith(disallow_path):
                    return False
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def check_url(self, url: str) -> Dict:
        """
        Check a single URL:
        - robots.txt allowance
        - final status code, redirection chain
        - parse SEO metrics
        """
        headers = {'User-Agent': USER_AGENT}

        async with self.semaphore:
            try:
                # 1. robots.txt
                is_allowed = await self.check_robots_txt(url)

                # 2. Make request (no redirect)
                async with self.session.get(url, headers=headers, ssl=False, allow_redirects=False) as response:
                    status_code = response.status
                    location = response.headers.get('Location')

                    if status_code in [301, 302, 307, 308] and location:
                        # Follow redirect chain up to 5 times
                        final_url, final_status, html_content, final_headers = await self.follow_redirect_chain(
                            url, headers, max_redirects=5
                        )
                    else:
                        final_url = str(response.url)
                        final_status = status_code
                        final_headers = response.headers
                        html_content = None

                        # If it's HTML, read it
                        if (
                            response.content_type
                            and response.content_type.startswith(('text/html', 'application/xhtml+xml'))
                        ):
                            html_content = await response.text(encoding='utf-8')

                    if html_content and final_status == 200:
                        # parse the page
                        soup = BeautifulSoup(html_content, 'lxml')
                        title = self.get_title(soup)
                        meta_desc = self.get_meta_description(soup)
                        h1_count = self.count_h1_tags(soup)
                        html_lang = self.get_html_lang(soup)
                        x_robots_tag = final_headers.get('X-Robots-Tag', '')

                        canonical_url = self.get_canonical_url(soup)
                        robots_meta = self.get_robots_meta(soup)

                        # Check for noindex
                        meta_robots_all = f"{robots_meta.lower()} {x_robots_tag.lower()}"
                        has_noindex = 'noindex' in meta_robots_all

                        # Evaluate indexability
                        is_indexable = (
                            final_status == 200
                            and not has_noindex
                            and (not canonical_url or canonical_url == url)
                            and is_allowed
                        )

                        # Build reason
                        reason_parts = []
                        if final_status != 200:
                            reason_parts.append("Non-200 status code")
                        if not is_allowed:
                            reason_parts.append("Blocked by robots.txt")
                        if has_noindex:
                            reason_parts.append("Noindex directive")
                        if canonical_url and canonical_url != url:
                            reason_parts.append("Non-self canonical")
                        if not reason_parts:
                            reason_parts.append("Page is indexable")
                        index_reason = "; ".join(reason_parts)

                        return {
                            "Original_URL": url,
                            "Final_Status_Code": final_status,
                            "Status_Type": self.get_status_type(final_status),
                            "Resolved_URL": final_url,
                            "First_Redirect_URL": (
                                urljoin(url, location)
                                if status_code in [301, 302, 307, 308] and location
                                else None
                            ),
                            "Canonical_URL": canonical_url,
                            "Canonical_Matches_URL": ("Yes" if (not canonical_url or canonical_url == url) else "No"),
                            "Blocked_by_robots_txt": ("No" if is_allowed else "Yes"),
                            "X_Robots_Tag": x_robots_tag,
                            "Meta_Robots": robots_meta,
                            "Title": title,
                            "Meta_Description": meta_desc,
                            "H1_Count": h1_count,
                            "HTML_Lang": html_lang,
                            "Is_Indexable": ("Yes" if is_indexable else "No"),
                            "Indexability_Reason": index_reason,
                            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                    else:
                        # final status is not 200, or not HTML => not indexable
                        x_robots_tag = final_headers.get('X-Robots-Tag', '') if final_headers else ''
                        robots_meta = "N/A"
                        canonical_url = None

                        reason_parts = []
                        if final_status != 200:
                            reason_parts.append("Non-200 status code")
                        if not is_allowed:
                            reason_parts.append("Blocked by robots.txt")
                        if not reason_parts:
                            reason_parts.append("Page is indexable")

                        index_reason = "; ".join(reason_parts)

                        return {
                            "Original_URL": url,
                            "Final_Status_Code": final_status,
                            "Status_Type": self.get_status_type(final_status),
                            "Resolved_URL": final_url,
                            "First_Redirect_URL": (
                                urljoin(url, location)
                                if status_code in [301, 302, 307, 308] and location
                                else None
                            ),
                            "Canonical_URL": canonical_url,
                            "Canonical_Matches_URL": "N/A",
                            "Blocked_by_robots_txt": ("No" if is_allowed else "Yes"),
                            "X_Robots_Tag": x_robots_tag,
                            "Meta_Robots": robots_meta,
                            "Title": "N/A",
                            "Meta_Description": "N/A",
                            "H1_Count": 0,
                            "HTML_Lang": "N/A",
                            "Is_Indexable": "No",
                            "Indexability_Reason": index_reason,
                            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }

            except asyncio.TimeoutError:
                logging.error(f"Timeout for {url}")
                return self.create_error_response(url, 'Timeout', 'Request timed out', False)
            except Exception as e:
                logging.error(f"Error for {url}: {str(e)}")
                return self.create_error_response(url, 'Error', str(e), False)

    async def follow_redirect_chain(self, start_url: str, headers: Dict, max_redirects: int = 5):
        """
        Follow up to `max_redirects` 3xx redirects, returning final details.
        """
        current_url = start_url
        html_content = None
        final_headers = None

        for _ in range(max_redirects):
            async with self.session.get(
                current_url,
                headers=headers,
                ssl=False,
                allow_redirects=False
            ) as resp:
                status = resp.status
                final_headers = resp.headers

                if (resp.content_type and
                    resp.content_type.startswith(('text/html', 'application/xhtml+xml'))):
                    html_content = await resp.text(encoding='utf-8')
                else:
                    html_content = None

                if status in [301, 302, 307, 308]:
                    loc = resp.headers.get('Location')
                    if not loc:
                        return current_url, status, html_content, final_headers
                    current_url = urljoin(current_url, loc)
                else:
                    return current_url, status, html_content, final_headers

        # If we exit the loop, it's a redirect loop
        return current_url, 'Redirect Loop', html_content, final_headers

    def create_error_response(self, url: str, code: str, message: str, is_allowed: bool) -> Dict:
        """
        Handle error or timeout responses with a consistent schema.
        """
        reason_parts = []
        if code != "200":
            reason_parts.append("Non-200 status code")
        if not is_allowed:
            reason_parts.append("Blocked by robots.txt")
        if not reason_parts:
            reason_parts.append("Page is indexable")

        return {
            "Original_URL": url,
            "Final_Status_Code": code,
            "Status_Type": message,
            "Resolved_URL": "N/A",
            "First_Redirect_URL": "N/A",
            "Canonical_URL": "N/A",
            "Canonical_Matches_URL": "N/A",
            "Blocked_by_robots_txt": ("No" if is_allowed else "Yes"),
            "X_Robots_Tag": "N/A",
            "Meta_Robots": "N/A",
            "Title": "N/A",
            "Meta_Description": "N/A",
            "H1_Count": 0,
            "HTML_Lang": "N/A",
            "Is_Indexable": "No",
            "Indexability_Reason": "; ".join(reason_parts),
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    @staticmethod
    def get_canonical_url(soup: BeautifulSoup) -> str:
        canonical = soup.find('link', {'rel': 'canonical'})
        return canonical['href'] if canonical else None

    @staticmethod
    def get_robots_meta(soup: BeautifulSoup) -> str:
        robots_meta = soup.find('meta', {'name': 'robots'})
        return robots_meta['content'] if robots_meta else ''

    @staticmethod
    def get_title(soup: BeautifulSoup) -> str:
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else ''

    @staticmethod
    def get_meta_description(soup: BeautifulSoup) -> str:
        desc_tag = soup.find('meta', {'name': 'description'})
        return desc_tag['content'] if desc_tag and desc_tag.has_attr('content') else ''

    @staticmethod
    def count_h1_tags(soup: BeautifulSoup) -> int:
        return len(soup.find_all('h1'))

    @staticmethod
    def get_html_lang(soup: BeautifulSoup) -> str:
        html_tag = soup.find('html')
        if html_tag and html_tag.has_attr('lang'):
            return html_tag['lang']
        return ''

    @staticmethod
    def get_status_type(status) -> str:
        status_codes = {
            200: "OK",
            301: "Permanent Redirect",
            302: "Temporary Redirect",
            307: "Temporary Redirect",
            308: "Permanent Redirect",
            404: "Not Found",
            403: "Forbidden",
            500: "Internal Server Error",
            503: "Service Unavailable"
        }
        return status_codes.get(status, f"Status Code {status}")


async def run_url_checks(urls: List[str]) -> pd.DataFrame:
    """
    Create a URLChecker, process all URLs (including a second pass for timeouts),
    and return the final results as a DataFrame.
    """
    checker = URLChecker()

    # 1. Setup default session
    await checker.setup(timeout_duration=TIMEOUT_DURATION)

    # 2. Break URLs into chunks
    chunks = [urls[i : i + CHUNK_SIZE] for i in range(0, len(urls), CHUNK_SIZE)]
    results = []

    # We'll track progress in total URLs
    total_urls = len(urls)
    processed_count = 0

    # Create a Streamlit progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process chunks
    for chunk in chunks:
        tasks = [checker.check_url(u) for u in chunk]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in chunk_results if isinstance(r, dict)]
        results.extend(valid_results)

        processed_count += len(chunk)
        progress_pct = int((processed_count / total_urls) * 100)
        progress_bar.progress(progress_pct)
        status_text.text(f"Processed {processed_count} of {total_urls} URLs")

        gc.collect()

    # Close session
    await checker.session.close()

    # 3. Identify timeouts, do second pass if needed
    timeout_urls = [r["Original_URL"] for r in results if r["Final_Status_Code"] == "Timeout"]
    if timeout_urls:
        st.write(f"Retrying {len(timeout_urls)} timeouts with a longer timeout...")
        await checker.setup(timeout_duration=RETRY_TIMEOUT_DURATION)

        timeout_chunks = [
            timeout_urls[i : i + CHUNK_SIZE] for i in range(0, len(timeout_urls), CHUNK_SIZE)
        ]
        retried_count = 0

        for chunk in timeout_chunks:
            tasks = [checker.check_url(u) for u in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_results = [r for r in chunk_results if isinstance(r, dict)]
            # Remove old timeout results from 'results' before adding the updated results
            results = [r for r in results if r["Original_URL"] not in chunk]
            results.extend(valid_results)
            retried_count += len(chunk)

            progress_pct = int((retried_count / len(timeout_urls)) * 100)
            progress_bar.progress(progress_pct)
            status_text.text(f"Retried {retried_count} of {len(timeout_urls)} timeouts")

        await checker.session.close()

    # 4. Convert to DataFrame
    df = pd.DataFrame(results)

    # Reset progress bar
    progress_bar.empty()
    status_text.text("All done!")
    return df


def main():
    st.title("Async URL Checker (Streamlit Edition)")
    st.write(
        "This app checks various technical and SEO aspects of URLs. "
        "Paste or upload a list of URLs, then click **Run Checks**."
    )

    # Let user input URLs in two ways:
    input_option = st.selectbox("How would you like to enter URLs?", ["Paste text", "Upload file"])

    urls = []

    if input_option == "Paste text":
        text_input = st.text_area("Paste URLs here, separated by whitespace or newlines:")
        if text_input.strip():
            # Split on whitespace
            urls = re.split(r"\s+", text_input.strip())

    else:  # "Upload file"
        uploaded_file = st.file_uploader("Upload a text file with URLs (one per line, or space-separated)", type=["txt", "csv"])
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            urls = re.split(r"\s+", content.strip())

    # Remove duplicates & empty strings
    urls = [u.strip() for u in urls if u.strip()]
    urls = list(dict.fromkeys(urls))  # preserve order but remove duplicates

    st.write(f"Number of URLs: {len(urls)}")

    # Button to start the checks
    if st.button("Run Checks"):
        if not urls:
            st.warning("No URLs provided!")
            return

        # Run the async checks
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df = loop.run_until_complete(run_url_checks(urls))
        loop.close()

        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        # Provide a CSV download
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"url_check_results_{datetime.now().strftime('%Y_%m_%d')}.csv",
            mime="text/csv"
        )

        # Optional summary
        show_summary(df)


def show_summary(df: pd.DataFrame):
    """
    Display a quick summary of final status codes, robots.txt blocking, and indexability.
    """
    st.subheader("Summary")

    # Final status code distribution
    status_counts = df["Final_Status_Code"].value_counts()
    st.write("**Final Status Code Distribution**")
    for code, count in status_counts.items():
        st.write(f"- {code}: {count} URLs")

    # Robots.txt analysis
    blocked_counts = df["Blocked_by_robots_txt"].value_counts()
    allowed = blocked_counts.get("No", 0)
    blocked = blocked_counts.get("Yes", 0)
    st.write("**Robots.txt Analysis**")
    st.write(f"- Allowed: {allowed}")
    st.write(f"- Blocked: {blocked}")

    # How many 200
    df_200 = df[df["Final_Status_Code"] == 200]
    st.write(f"**HTTP 200 URLs**: {len(df_200)}")

    if not df_200.empty:
        # Indexable
        indexable_count = df_200["Is_Indexable"].value_counts().get("Yes", 0)
        st.write(f"- Indexable: {indexable_count}")

        # Noindex
        noindex_count = df_200[df_200["Indexability_Reason"].str.contains("Noindex directive", case=False, na=False)].shape[0]
        st.write(f"- Noindex: {noindex_count}")

        # Blocked by robots
        blocked_count = df_200[df_200["Indexability_Reason"].str.contains("Blocked by robots.txt", case=False, na=False)].shape[0]
        st.write(f"- Blocked by robots.txt: {blocked_count}")

        # Non-self canonical
        canonical_no_count = df_200[df_200["Canonical_Matches_URL"] == "No"].shape[0]
        st.write(f"- Non-self canonical: {canonical_no_count}")


if __name__ == "__main__":
    main()
