import os
import warnings
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class BraveSearchClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.search.brave.com/res/v1",
        timeout: float = 10.0,
        proxies: Optional[Dict[str, str]] = None,
        ca_bundle_path: Optional[str] = None,   # e.g., r"C:\path\to\corp_root_ca.pem"
        verify: Optional[bool] = None,          # None -> infer from ca_bundle_path; True/False explicit
        max_retries: int = 3,
        backoff_factor: float = 0.3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Session setup with retries
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            backoff_factor=backoff_factor,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Proxies (inherit from env if not provided)
        self.session.proxies = proxies or {
            "http": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
            "https": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
        }

        # SSL verification configuration
        self.ca_bundle_path = ca_bundle_path
        if verify is None:
            # If a CA bundle path is provided, use it; else default True
            self.verify = ca_bundle_path if ca_bundle_path else True
        else:
            self.verify = verify

        if self.verify is False:
            warnings.warn(
                "SSL verification is disabled. This should be a temporary fallback in controlled environments.",
                RuntimeWarning,
            )

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
            # Optional: brave supports language/region via headers or query
            # "Accept-Language": "en-US,en;q=0.9",
        }

    def web_search(
        self,
        query: str,
        *,
        count: int = 3,
        country: Optional[str] = None,
        safesearch: Optional[str] = None,  # "off", "moderate", "strict"
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a web search.
        Returns the JSON response as a dict.
        """
        params = {"q": query, "count": count}
        if country:
            params["country"] = country
        if safesearch:
            params["safesearch"] = safesearch
        if extra_params:
            params.update(extra_params)

        url = f"{self.base_url}/web/search"
        resp = self.session.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
            verify=False,
        )
        # print(resp.status_code)
        # print(resp.text)
        resp.raise_for_status()
        return resp.json()

    def news_search(self, query: str, *, count: int = 10, extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = {"q": query, "count": count}
        if extra_params:
            params.update(extra_params)
        url = f"{self.base_url}/news"
        resp = self.session.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
            verify=self.verify,
        )
        resp.raise_for_status()
        return resp.json()

    def images_search(self, query: str, *, count: int = 10, extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = {"q": query, "count": count}
        if extra_params:
            params.update(extra_params)
        url = f"{self.base_url}/images"
        resp = self.session.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
            verify=self.verify,
        )
        resp.raise_for_status()
        return resp.json()