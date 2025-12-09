from brave_search import BraveSearchClient

client = BraveSearchClient(api_key="BSAUMU3kFTYvyOjoYUbdl0Bt8LVnw-I")

res = client.web_search("cricket score of ashes",count=3,country="IN",extra_params={"result_filter":"web"})
# print(res.json())

def get_urls_from_resp(response):
    # print(response["web"].keys())
    return [result["url"] for result in response["web"]["results"]]

# print(get_urls_from_resp(res))


import requests
from bs4 import BeautifulSoup
import re

def fetch_and_clean(url, *, ca_bundle_path=None, verify=True, timeout=10):
    """
    Fetch a URL and return cleaned text content.
    - ca_bundle_path: path to corporate CA bundle PEM file
    - verify: True/False or path to CA bundle
    """
    verify_option = ca_bundle_path if ca_bundle_path else verify

    try:
        resp = requests.get(url, timeout=timeout, verify=verify_option,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script/style/noscript tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Extract visible text
    text = soup.get_text(separator=" ")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def scrape_urls(urls):
    results = {}
    for url in urls:
        print(f"Scraping: {url}")
        content = fetch_and_clean(url, verify=False)  # fallback if corp SSL blocks
        results[url] = content + "..." if content else "No content"
    return results

# Example usage:
urls = get_urls_from_resp(res)
scraped = scrape_urls(urls)

for u, txt in scraped.items():
    print(f"\n--- {u} ---\n{txt}\n")