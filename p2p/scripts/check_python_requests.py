import requests

targets = {
    "RDAP": "https://rdap.org/domain/google.com",
    "Wayback": "https://web.archive.org/cdx/search/cdx?url=example.com&output=json&limit=1",
    "Graph": "https://graph.facebook.com",
    "TikTok": "https://open.tiktokapis.com/",
    "YouTube": "https://www.googleapis.com/youtube/v3/videos",
    "RedditPublic": "https://www.reddit.com/.json",
    "RedditOAuth": "https://oauth.reddit.com",
    "TelegramBotAPI": "https://api.telegram.org/bot000000:dummy/getMe",
    "GDELT": "https://api.gdeltproject.org/api/v2/doc/doc?query=facebook&mode=ArtList&format=json",
}

ok = 0
fail = 0
for name, url in targets.items():
    try:
        r = requests.get(url, timeout=10)
        code = r.status_code
        if code:  # 任意状态码都视为“可达”，401/403/400 代表网络OK但未鉴权
            print(f"[PASS] {name} {code} {url}")
            ok += 1
        else:
            print(f"[FAIL] {name} NO_STATUS {url}")
            fail += 1
    except Exception as e:
        print(f"[FAIL] {name} EXC {e} {url}")
        fail += 1

print(f"\nSummary: PASS={ok} FAIL={fail}")
