#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch recent Reddit post URLs (image/video only, non-NSFW) and write to seed_urls.txt.

Usage examples:
  python scripts/fetch_reddit_seeds.py --subs pics,news,technology --days 7 --limit 40 --out tmp/seed_urls.txt
  # with OAuth (recommended):
  export REDDIT_CLIENT_ID=xxxx
  export REDDIT_CLIENT_SECRET=yyyy
  python scripts/fetch_reddit_seeds.py --subs all --query "deepfake OR AI video" --days 3 --limit 30 --min-score 10

Notes:
- If OAuth env vars are set, uses https://oauth.reddit.com (more reliable).
- Otherwise falls back to https://www.reddit.com/{...}.json endpoints.
- Respects HTTP(S) proxy env vars.
"""

import os
import sys
import time
import json
import math
import argparse
from datetime import datetime, timedelta, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
UA = "DocSentinelSeed/1.0 by u/Traditional_Ball2838 (academic)"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
VIDEO_HOSTS = ("v.redd.it", "redgifs.com", "gfycat.com", "streamable.com", "youtube.com", "youtu.be")
IMAGE_HOSTS = ("i.redd.it", "i.imgur.com", "imgur.com")
MEDIA_HINTS = {"image", "hosted:video", "rich:video"}

def get_proxies_from_env():
    # requests 自动读环境变量，无需额外传；为了可控，这里显式取用
    p = {}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        v = os.environ.get(k)
        if v:
            if k.lower().startswith("http"):
                p["http"] = v
            if k.lower().startswith("https"):
                p["https"] = v
    return p or None

def get_oauth_token_password(cid, secret, username, password, proxies=None):
    import requests
    auth = requests.auth.HTTPBasicAuth(cid, secret)
    headers = {"User-Agent": UA}
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "scope": "read"   # 读取帖子就够用；不需要 identity 等
    }
    r = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=auth, data=data, headers=headers, timeout=25, proxies=proxies
    )
    r.raise_for_status()
    j = r.json()
    return j["access_token"], j.get("token_type", "bearer")

def http_get_json(url, headers=None, params=None, proxies=None, max_retries=4, backoff=1.25):
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=25, proxies=proxies)
            # Reddit 有时以 429/5xx 节流；做指数退避
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code} {r.text[:120]}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep((backoff ** i) + 0.3 * i)
    raise last_err

def is_media_post(post: dict) -> bool:
    # 基于常见信号判断是否带图/视频
    if post.get("over_18"):
        return False
    if post.get("removed_by_category") or post.get("removed") or post.get("banned_at_utc"):
        return False

    hint = post.get("post_hint") or ""
    url = (post.get("url_overridden_by_dest") or post.get("url") or "").lower()
    domain = (post.get("domain") or "").lower()
    is_gallery = bool(post.get("is_gallery"))
    has_preview = "preview" in post
    has_media = post.get("media") is not None

    if hint in MEDIA_HINTS:
        return True
    if is_gallery:
        return True
    if has_media or has_preview:
        return True
    if url.endswith(IMG_EXTS):
        return True
    if any(h in domain for h in IMAGE_HOSTS + VIDEO_HOSTS):
        return True
    return False

def build_post_url(post: dict) -> str:
    perm = (post.get("permalink") or "").strip()
    url  = (post.get("url_overridden_by_dest") or post.get("url") or "").strip()

    # 优先用 permalink（最标准）
    if perm:
        base = "https://www.reddit.com"
        return base.rstrip("/") + "/" + perm.lstrip("/")

    # 次选：如果 url 本身就是 /r/xxx 开头
    if url.startswith("/r/"):
        return "https://www.reddit.com" + url

    # 已是完整 http(s) 链接则直接返回
    if url.startswith("http://") or url.startswith("https://"):
        return url

    return ""


def created_within_days(post: dict, days: int) -> bool:
    t = post.get("created_utc")
    if t is None:
        return True
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    return dt >= (datetime.now(timezone.utc) - timedelta(days=days))

def fetch_from_subreddit_oauth(sub: str, limit: int, days: int, query: str, min_score: int, token: str, proxies):
    base = "https://oauth.reddit.com"
    headers = {"Authorization": f"bearer {token}", "User-Agent": UA}
    out = []
    after = None
    # 选：优先 new 流，若有 query 则用 search
    use_search = bool(query and query.strip() and sub != "all")
    for _ in range(30):  # 安全上限分页轮数
        if use_search:
            url = f"{base}/r/{sub}/search"
            params = {
                "q": query,
                "restrict_sr": "on",
                "sort": "new",
                "t": "week",
                "limit": 100,
                "type": "link",
                "after": after
            }
        else:
            # 对 r/all 用 search 更稳妥（支持 after）；但 new 也可以
            url = f"{base}/r/{sub}/new"
            params = {"limit": 100, "after": after}

        data = http_get_json(url, headers=headers, params=params, proxies=proxies)
        children = (data.get("data") or {}).get("children") or []
        if not children:
            break
        for c in children:
            p = c.get("data") or {}
            if not created_within_days(p, days):
                continue
            if p.get("score", 0) < min_score:
                continue
            if is_media_post(p):
                out.append(p)
            if len(out) >= limit:
                return out
        after = (data.get("data") or {}).get("after")
        if not after:
            break
        time.sleep(0.6)  # 轻限速
    return out

def fetch_from_subreddit_public(sub: str, limit: int, days: int, query: str, min_score: int, proxies):
    base = "https://www.reddit.com"
    headers = {"User-Agent": UA}
    out = []
    after = None
    use_search = bool(query and query.strip() and sub != "all")

    for _ in range(30):
        if use_search:
            url = f"{base}/r/{sub}/search.json"
            params = {
                "q": query,
                "restrict_sr": "on",
                "sort": "new",
                "t": "week",
                "limit": 100,
                "after": after,
                "type": "link",
                "include_over_18": "off",
            }
        else:
            url = f"{base}/r/{sub}/new.json"
            params = {"limit": 100, "after": after}

        data = http_get_json(url, headers=headers, params=params, proxies=proxies)
        children = (data.get("data") or {}).get("children") or []
        if not children:
            break
        for c in children:
            p = c.get("data") or {}
            if not created_within_days(p, days):
                continue
            if p.get("score", 0) < min_score:
                continue
            if is_media_post(p):
                out.append(p)
            if len(out) >= limit:
                return out
        after = (data.get("data") or {}).get("after")
        if not after:
            break
        time.sleep(0.8)
    return out

def make_session(proxies=None):
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Connection": "close",     # 避免某些代理对 keep-alive 的奇怪行为
    })
    if proxies:
        s.proxies = proxies
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=0.6,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def get_json_public(path, params, proxies):
    """
    依次尝试：
    1) https://www.reddit.com{path}.json
    2) https://old.reddit.com{path}.json
    3) https://r.jina.ai/http://www.reddit.com{path}.json
    """
    s = make_session(proxies)
    bases = [
        "https://www.reddit.com",
        "https://old.reddit.com",
        "https://r.jina.ai/http://www.reddit.com",
    ]
    last = None
    for base in bases:
        url = f"{base}{path}"
        try:
            r = s.get(url, params=params, timeout=25)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
    raise last

def fetch_from_subreddit_public(sub: str, limit: int, days: int, query: str, min_score: int, proxies):
    out, after = [], None
    use_search = bool(query and query.strip())
    for _ in range(30):
        if use_search:
            path = f"/search.json"
            params = {
                "q": query, "sort": "new", "t": "week", "limit": 100, "after": after,
                "type": "link", "include_over_18": "off", "restrict_sr": "off" if sub=="all" else "on",
            }
            if sub != "all":
                # 指定子版块
                path = f"/r/{sub}/search.json"
        else:
            path = f"/r/{sub}/new.json"
            params = {"limit": 100, "after": after}

        data = get_json_public(path, params, proxies)
        children = (data.get("data") or {}).get("children") or []
        if not children:
            break
        for c in children:
            p = c.get("data") or {}
            if not created_within_days(p, days):  # 你原本的过滤函数
                continue
            if p.get("score", 0) < min_score:
                continue
            if is_media_post(p):                  # 你原本的媒体判定
                out.append(p)
            if len(out) >= limit:
                return out
        after = (data.get("data") or {}).get("after")
        if not after:
            break
        time.sleep(0.6)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", type=str, default="pics,news,worldnews", help="Comma-separated subreddits, or 'all'")
    ap.add_argument("--days", type=int, default=7, help="Only include posts created within last N days")
    ap.add_argument("--limit", type=int, default=40, help="Total number of posts to collect across all subs")
    ap.add_argument("--query", type=str, default="", help="Optional search query (Reddit search syntax).")
    ap.add_argument("--min-score", type=int, default=0, help="Filter by minimum post score (upvotes).")
    ap.add_argument("--include-nsfw", action="store_true", help="Include NSFW posts (default false). Not recommended.")
    ap.add_argument("--out", type=str, default="tmp/seed_urls.txt", help="Output file path")
    args = ap.parse_args()

    subs = [s.strip() for s in args.subs.split(",") if s.strip()]
    total_limit = max(1, args.limit)

    proxies = get_proxies_from_env()
    print("[info] proxies:", proxies or "None")
    cid = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user = os.environ.get("REDDIT_USERNAME")
    pwd = os.environ.get("REDDIT_PASSWORD")
    token = None
    if cid and secret and user and pwd:
        token, _ = get_oauth_token_password(cid, secret, user, pwd, proxies=proxies)

    per_sub = max(5, math.ceil(total_limit / max(1, len(subs))))  # 粗略分配
    seen_ids = set()
    posts = []

    for sub in subs:
        try:
            if token:
                res = fetch_from_subreddit_oauth(sub=sub, limit=per_sub, days=args.days,
                                                 query=args.query, min_score=args.min_score,
                                                 token=token, proxies=proxies)
            else:
                res = fetch_from_subreddit_public(sub=sub, limit=per_sub, days=args.days,
                                                  query=args.query, min_score=args.min_score,
                                                  proxies=proxies)
        except Exception as e:
            print(f"[warn] fetch {sub} failed: {e}")
            res = []
        kept = 0
        for p in res:
            if (not args.include_nsfw) and p.get("over_18"):
                continue
            pid = p.get("id")
            if not pid or pid in seen_ids:
                continue
            url = build_post_url(p)
            if not url:
                continue
            posts.append({"id": pid, "url": url})
            seen_ids.add(pid)
            kept += 1
            if len(posts) >= total_limit:
                break
        print(f"[info] {sub}: added {kept} posts.")
        if len(posts) >= total_limit:
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in posts:
            f.write(p["url"] + "\n")

    print(f"[done] wrote {len(posts)} urls -> {args.out}")
    if len(posts) == 0:
        print("[hint] Try lowering --min-score, increasing --days, adding more subs, or enabling OAuth.")

if __name__ == "__main__":
    main()
