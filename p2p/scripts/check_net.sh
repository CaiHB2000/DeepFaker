#!/usr/bin/env bash
set -u

PASS=0
FAIL=0

have() { command -v "$1" >/dev/null 2>&1; }
say() { printf "%-8s %s\n" "$1" "$2"; }

request_any() {
  local url="$1" name="$2"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$url")
  if [[ "$code" != "000" ]]; then
    say "[PASS]" "$name ($code) $url"
    PASS=$((PASS+1))
  else
    say "[FAIL]" "$name (NO RESPONSE) $url"
    FAIL=$((FAIL+1))
  fi
}

echo "=== Environment ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Host: $(hostname -f 2>/dev/null || hostname)"
echo "Shell: $SHELL"
echo "Proxy: HTTP_PROXY=${HTTP_PROXY:-} HTTPS_PROXY=${HTTPS_PROXY:-} ALL_PROXY=${ALL_PROXY:-}"
echo

echo "=== Basic DNS / Outbound IP ==="
if have getent; then
  getent hosts rdap.org >/dev/null && say "[PASS]" "DNS getent rdap.org" || { say "[FAIL]" "DNS getent rdap.org"; FAIL=$((FAIL+1)); }
elif have nslookup; then
  nslookup rdap.org >/dev/null 2>&1 && say "[PASS]" "DNS nslookup rdap.org" || { say "[FAIL]" "DNS nslookup rdap.org"; FAIL=$((FAIL+1)); }
else
  say "[SKIP]" "No getent/nslookup; skip DNS test"
fi

IP=$(curl -s --max-time 10 https://ifconfig.me || true)
if [[ -n "$IP" ]]; then say "[PASS]" "Outbound IP: $IP"; PASS=$((PASS+1)); else say "[FAIL]" "Outbound IP check"; FAIL=$((FAIL+1)); fi
echo

echo "=== HTTP/HTTPS Reachability ==="
request_any "https://example.com"                          "Generic HTTPS"
request_any "http://example.com"                           "Generic HTTP"

echo
echo "=== Provenance data sources ==="
request_any "https://rdap.org/domain/google.com"           "RDAP"
request_any "https://web.archive.org/cdx/search/cdx?url=example.com&output=json&limit=1" "Wayback CDX"

echo
echo "=== Platform / API endpoints (可达性，不含鉴权) ==="
request_any "https://graph.facebook.com"                   "Meta Graph API"
request_any "https://open.tiktokapis.com/"                 "TikTok API"
request_any "https://www.googleapis.com/youtube/v3/videos" "YouTube Data API"
request_any "https://www.reddit.com/.json"                 "Reddit Public"
request_any "https://oauth.reddit.com"                     "Reddit OAuth Gateway"
request_any "https://api.telegram.org/bot000000:dummy/getMe" "Telegram Bot API"
request_any "https://api.gdeltproject.org/api/v2/doc/doc?query=facebook&mode=ArtList&format=json" "GDELT v2 Doc API"

echo
echo "=== TLS Handshake (OpenSSL) ==="
if have openssl; then
  TLS_HOSTS=("reddit.com" "www.youtube.com" "graph.facebook.com" "open.tiktokapis.com" "api.telegram.org")
  # 检测 openssl 是否支持 -proxy 选项（OpenSSL 3.x 常见）
  if openssl s_client -help 2>&1 | grep -q -- '-proxy '; then
    OPENSSL_HAS_PROXY=1
  else
    OPENSSL_HAS_PROXY=0
  fi

  # 从环境变量解析代理（仅用于 -proxy）
  # 优先 HTTPS_PROXY -> HTTP_PROXY -> ALL_PROXY
  PROXY_URL="${HTTPS_PROXY:-${HTTP_PROXY:-${ALL_PROXY:-}}}"
  # 只取 host:port
  if [[ -n "$PROXY_URL" ]]; then
    PROXY_HP="${PROXY_URL#*://}"
    PROXY_HP="${PROXY_HP%/}"
  fi

  for host in "${TLS_HOSTS[@]}"; do
    if [[ -n "$PROXY_URL" && $OPENSSL_HAS_PROXY -eq 1 ]]; then
      # 经代理做 TLS 握手
      if timeout 8 bash -c "echo | openssl s_client -proxy '${PROXY_HP}' -connect '${host}:443' -servername '${host}' -brief >/dev/null 2>&1"; then
        say "[PASS]" "TLS (via proxy) ${host}:443"
        PASS=$((PASS+1))
      else
        say "[FAIL]" "TLS (via proxy) ${host}:443"
        FAIL=$((FAIL+1))
      fi
    elif [[ -n "$PROXY_URL" && $OPENSSL_HAS_PROXY -eq 0 ]]; then
      # 代理已启用但 openssl 不支持 -proxy：跳过直连，改用 curl 间接校验
      if curl -s -I --max-time 10 "${HTTP_PROXY:+--proxy $HTTP_PROXY}" "https://${host}" >/dev/null; then
        say "[PASS]" "TLS (curl via proxy) ${host}:443"
        PASS=$((PASS+1))
      else
        say "[FAIL]" "TLS (curl via proxy) ${host}:443"
        FAIL=$((FAIL+1))
      fi
    else
      # 无代理环境：直连 TLS，强制 timeout 防挂
      if timeout 8 bash -c "echo | openssl s_client -connect '${host}:443' -servername '${host}' -brief >/dev/null 2>&1"; then
        say "[PASS]" "TLS ${host}:443"
        PASS=$((PASS+1))
      else
        say "[FAIL]" "TLS ${host}:443"
        FAIL=$((FAIL+1))
      fi
    fi
  done
else
  say "[SKIP]" "OpenSSL not found; skip TLS handshakes"
fi

echo
echo "=== Summary ==="
echo "PASS: $PASS"
echo "FAIL: $FAIL"
if [[ $FAIL -eq 0 ]]; then
  echo "All good ✅"
  exit 0
else
  echo "Some checks FAILED ❌ 见上方具体项。"
  exit 1
fi
