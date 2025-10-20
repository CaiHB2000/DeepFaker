# p2p/datasource/webmeta_tables.py
# -*- coding: utf-8 -*-
"""
将 run_webmeta 生成的 JSONL 标准化为 CSV，并据此拆出三张“骨架表”：
  1) post_table.csv      : 每条种子URL对应的平台与帖子ID（posting_id=platform:post_id）
  2) media_manifest.csv  : 从 webmeta.csv 中筛出的“可直接下载的媒体URL”清单（image/*, video/*）
  3) post_map.csv        : URL → posting_id 的映射（为后续 content_id 对齐作准备）

兼容以下 WebArchive 命名：
  - 规范：first_seen_ts / latest_seen_ts / snapshots
  - 旧式：first_ts / last_ts / count
"""

from __future__ import annotations
import csv
import json
import re
from typing import Dict, Any, Iterable, Tuple, Optional
from pathlib import Path

# ========= 工具函数 =========

def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # 跳过坏行，但尽量健壮
                continue

def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def _as_int(x, default=None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default

def _starts_with_media(ct: Optional[str]) -> bool:
    if not isinstance(ct, str):
        return False
    ct = ct.lower().strip()
    return ct.startswith("image/") or ct.startswith("video/")

def _parse_platform_post_id(url: str) -> Tuple[str, str]:
    """从URL中尽力解析 (platform, post_id)。若无法解析，回退到 md5 前缀；但这里保持纯可复用，不引入hash，返回 ('unknown', url) 以便上游去重。"""
    # YouTube
    m = re.search(r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{6,})', url)
    if m:
        return "youtube", m.group(1)
    # Reddit (r/sub/comments/<id>/...)
    m = re.search(r'(?:https?://)?(?:www\.)?reddit\.com/(?:r/[^/]+/comments|comments)/([a-z0-9]{5,10})', url)
    if m:
        return "reddit", m.group(1)
    # Telegram (t.me/<chan>/123 或 t.me/c/<id>/123)
    m = re.search(r'(?:https?://)?t\.me/(?:c/)?[^/]+/(\d+)', url)
    if m:
        return "telegram", m.group(1)
    # 兜底
    return "unknown", url

# ========= 核心：JSONL → CSV =========

WEBMETA_FIELDS = [
    # 基本
    "url",
    "domain",
    "ts_collected",
    "scheme",
    # HTTP
    "http_url_requested",
    "http_final_url",
    "status_code",
    "content_type",
    "content_length",
    "server",
    "via",
    # TLS
    "tls_issuer",
    "tls_subject",
    "tls_not_before",
    "tls_not_after",
    "tls_serial_number",
    "tls_sig_hash_alg",
    # RDAP
    "rdap_registry",
    "rdap_domain_status",   # 以“|”拼接列表
    "rdap_registration",
    "rdap_expiration",
    # Wayback
    "wayback_first_seen_ts",
    "wayback_latest_seen_ts",
    "wayback_snapshots",
    # 错误
    "error_count",
]

def webmeta_jsonl_to_csv(jsonl_path: str, csv_path: str) -> None:
    """
    将 run_webmeta 产出的 JSONL 标准化为 CSV（列见 WEBMETA_FIELDS）。
    """
    src = Path(jsonl_path)
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=WEBMETA_FIELDS)
        w.writeheader()

        for j in _read_jsonl(src):
            url = j.get("url")
            domain = j.get("domain")
            ts_collected = j.get("ts_collected")

            http = j.get("http") or {}
            tls  = j.get("tls") or {}
            rdap = j.get("rdap") or {}
            way  = j.get("wayback") or {}
            errs = j.get("errors") or []

            # HTTP
            http_url_requested = http.get("url_requested") or http.get("request_url") or url
            http_final_url     = http.get("final_url") or http.get("response_url") or url
            status_code        = _as_int(http.get("status_code"))
            content_type       = http.get("content_type")
            content_length     = _as_int(http.get("content_length"))
            server             = http.get("server")
            via                = http.get("via")

            # TLS（沿用 dataclass 字段命名）
            tls_issuer         = tls.get("issuer")
            tls_subject        = tls.get("subject")
            tls_not_before     = tls.get("not_before")
            tls_not_after      = tls.get("not_after")
            tls_serial_number  = tls.get("serial_number")
            tls_sig_hash_alg   = tls.get("sig_hash_alg")

            # RDAP
            rdap_registry      = rdap.get("registry")
            rdap_status_list   = rdap.get("domain_status") or []
            rdap_domain_status = "|".join(rdap_status_list) if isinstance(rdap_status_list, list) else str(rdap_status_list)
            events             = rdap.get("events") or {}
            rdap_registration  = events.get("registration")
            rdap_expiration    = events.get("expiration")

            # Wayback（兼容旧字段）
            wayback_first_seen_ts = _coalesce(way.get("first_seen_ts"), way.get("first_ts"))
            wayback_latest_seen_ts = _coalesce(way.get("latest_seen_ts"), way.get("last_ts"))
            wayback_snapshots     = _as_int(_coalesce(way.get("snapshots"), way.get("count")))

            scheme = None
            if isinstance(url, str) and ":" in url:
                scheme = url.split(":", 1)[0]

            row = {
                "url": url,
                "domain": domain,
                "ts_collected": ts_collected,
                "scheme": scheme,
                "http_url_requested": http_url_requested,
                "http_final_url": http_final_url,
                "status_code": status_code,
                "content_type": content_type,
                "content_length": content_length,
                "server": server,
                "via": via,
                "tls_issuer": tls_issuer,
                "tls_subject": tls_subject,
                "tls_not_before": tls_not_before,
                "tls_not_after": tls_not_after,
                "tls_serial_number": tls_serial_number,
                "tls_sig_hash_alg": tls_sig_hash_alg,
                "rdap_registry": rdap_registry,
                "rdap_domain_status": rdap_domain_status,
                "rdap_registration": rdap_registration,
                "rdap_expiration": rdap_expiration,
                "wayback_first_seen_ts": wayback_first_seen_ts,
                "wayback_latest_seen_ts": wayback_latest_seen_ts,
                "wayback_snapshots": wayback_snapshots,
                "error_count": len(errs),
            }
            w.writerow(row)

# ========= 从 webmeta.csv 拆三张“骨架表” =========

POST_TABLE_FIELDS = ["posting_id", "platform", "post_id", "url"]
MEDIA_MANIFEST_FIELDS = ["media_url", "content_type", "local_path"]
POST_MAP_FIELDS = ["canonical_url", "posting_id"]

def build_tables_from_webmeta_csv(
    webmeta_csv_path: str,
    post_table_csv_path: str,
    media_manifest_csv_path: str,
    post_map_csv_path: str,
) -> None:
    """
    输入 webmeta.csv，输出三张“骨架表”：
      - post_table.csv
      - media_manifest.csv（仅包含 content_type 为 image/* 或 video/* 且状态码为 200 的 URL）
      - post_map.csv
    """
    src = Path(webmeta_csv_path)
    out_post = Path(post_table_csv_path); out_post.parent.mkdir(parents=True, exist_ok=True)
    out_media = Path(media_manifest_csv_path); out_media.parent.mkdir(parents=True, exist_ok=True)
    out_pmap = Path(post_map_csv_path); out_pmap.parent.mkdir(parents=True, exist_ok=True)

    # 读取 webmeta.csv
    rows = []
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # 1) post_table
    # 使用 http_final_url 优先作为判定来源，回退 url
    seen_posting_ids = set()
    post_rows = []
    for r in rows:
        u = r.get("http_final_url") or r.get("url") or ""
        plat, pid = _parse_platform_post_id(u)
        posting_id = f"{plat}:{pid}"
        if posting_id not in seen_posting_ids:
            post_rows.append({
                "posting_id": posting_id,
                "platform": plat,
                "post_id": pid,
                "url": u,
            })
            seen_posting_ids.add(posting_id)

    with out_post.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POST_TABLE_FIELDS)
        w.writeheader()
        for r in post_rows:
            w.writerow(r)

    # 2) media_manifest：筛直接媒体
    media_rows = []
    for r in rows:
        ct = r.get("content_type")
        sc = r.get("status_code")
        try:
            sc = int(sc) if sc is not None else None
        except Exception:
            sc = None
        u = r.get("http_final_url") or r.get("url")
        if _starts_with_media(ct) and sc == 200 and u:
            media_rows.append({
                "media_url": u,
                "content_type": ct,
                "local_path": "",  # 后续下载再回填
            })

    # 去重（按 media_url）
    seen_media = set()
    media_rows_dedup = []
    for m in media_rows:
        key = m["media_url"]
        if key in seen_media:
            continue
        seen_media.add(key)
        media_rows_dedup.append(m)

    with out_media.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MEDIA_MANIFEST_FIELDS)
        w.writeheader()
        for r in media_rows_dedup:
            w.writerow(r)

    # 3) post_map（URL → posting_id）
    pmap_rows = []
    for r in rows:
        u = r.get("http_final_url") or r.get("url")
        plat, pid = _parse_platform_post_id(u or "")
        posting_id = f"{plat}:{pid}"
        if u:
            pmap_rows.append({"canonical_url": u, "posting_id": posting_id})

    # 去重（(canonical_url, posting_id)）
    seen_map = set()
    pmap_rows_dedup = []
    for m in pmap_rows:
        key = (m["canonical_url"], m["posting_id"])
        if key in seen_map:
            continue
        seen_map.add(key)
        pmap_rows_dedup.append(m)

    with out_pmap.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POST_MAP_FIELDS)
        w.writeheader()
        for r in pmap_rows_dedup:
            w.writerow(r)

# ========= 便捷一条龙 =========

def jsonl_to_tables(
    webmeta_jsonl: str,
    out_dir: str,
    webmeta_csv_name: str = "webmeta.csv",
    post_table_name: str = "post_table.csv",
    media_manifest_name: str = "media_manifest.csv",
    post_map_name: str = "post_map.csv",
) -> Dict[str, str]:
    """
    从 JSONL 到三张表的一条龙封装，返回各输出路径字典。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    webmeta_csv = str(out / webmeta_csv_name)
    post_table_csv = str(out / post_table_name)
    media_manifest_csv = str(out / media_manifest_name)
    post_map_csv = str(out / post_map_name)

    webmeta_jsonl_to_csv(webmeta_jsonl, webmeta_csv)
    build_tables_from_webmeta_csv(webmeta_csv, post_table_csv, media_manifest_csv, post_map_csv)

    return {
        "webmeta_csv": webmeta_csv,
        "post_table_csv": post_table_csv,
        "media_manifest_csv": media_manifest_csv,
        "post_map_csv": post_map_csv,
    }
