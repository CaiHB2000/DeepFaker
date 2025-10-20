import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- 可通过环境变量调参 ----
DEFAULT_TOTAL   = int(os.getenv("P2P_HTTP_RETRIES", "5"))
DEFAULT_BACKOFF = float(os.getenv("P2P_HTTP_BACKOFF", "0.6"))
STATUS_FORCELIST = tuple(int(x) for x in os.getenv(
    "P2P_HTTP_RETRY_STATUS", "429,500,502,503,504"
).split(","))

DEFAULT_UA = os.getenv(
    "P2P_HTTP_UA",
    "P2P-Risk/0.1 (+https://example) DocSentinel seed/meta fetcher"
)

def make_session(total: int = DEFAULT_TOTAL, backoff: float = DEFAULT_BACKOFF) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total, connect=total, read=total, status=total,
        backoff_factor=backoff,
        status_forcelist=STATUS_FORCELIST,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.trust_env = True  # 读取 HTTP(S)_PROXY / ALL_PROXY

    # 统一默认头，避免 keep-alive 在部分代理链路上导致 EOF
    s.headers.update({
        "User-Agent": DEFAULT_UA,
        "Accept": "application/json, text/plain, */*",
        "Connection": "close",
    })
    return s

SESSION = make_session()

def http_get(url: str, params=None, timeout: float = 12.0, **kw):
    return SESSION.get(url, params=params, timeout=timeout, **kw)

def http_head(url: str, timeout: float = 8.0, allow_redirects: bool = True, **kw):
    return SESSION.head(url, timeout=timeout, allow_redirects=allow_redirects, **kw)
