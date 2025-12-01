# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Iterable
import re
import random

import praw
import prawcore
from praw.models import Submission

MEDIA_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
MEDIA_GIF_EXT = (".gif", ".gifv")
MEDIA_DIRECT_EXT = MEDIA_IMG_EXT + MEDIA_GIF_EXT + (".mp4",)

@dataclass
class MediaItem:
    media_url: str
    kind: str                 # image | gif | video_mp4 | video_dash | video_hls | thumb | ext_image
    content_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    source: Optional[str] = None        # i.redd.it / v.redd.it / imgur / youtube / ...
    downloadable: bool = True           # 是否可直接下载（直链）

@dataclass
class PostRecord:
    # 关键信息（统一，便于后续入库/打分/映射）
    platform: str = "reddit"
    subreddit: Optional[str] = None
    post_id: str = ""
    url: str = ""
    permalink: str = ""
    created_utc: float = 0.0
    author_id: Optional[str] = None
    author_name: Optional[str] = None
    title: Optional[str] = None
    is_nsfw: bool = False
    is_video: bool = False
    is_gallery: bool = False
    over_18: bool = False
    stickied: bool = False
    crosspost_parent: Optional[str] = None

    # 参与度
    score: Optional[int] = None
    ups: Optional[int] = None
    upvote_ratio: Optional[float] = None
    num_comments: Optional[int] = None
    gilded: Optional[int] = None

    # 外链与类型
    domain: Optional[str] = None
    url_overridden_by_dest: Optional[str] = None
    post_hint: Optional[str] = None

    # 媒体清单
    media: List[MediaItem] = field(default_factory=list)

    # 额外元数据
    selftext: Optional[str] = None
    selftext_html: Optional[str] = None
    link_flair_text: Optional[str] = None
    link_flair_richtext: List[Dict[str, Any]] = field(default_factory=list)
    flair_template_id: Optional[str] = None
    author_flair_text: Optional[str] = None
    author_flair_css_class: Optional[str] = None
    author_flair_richtext: List[Dict[str, Any]] = field(default_factory=list)
    author_is_blocked: Optional[bool] = None
    author_premium: Optional[bool] = None
    spoiler: bool = False
    locked: bool = False
    archived: bool = False
    pinned: bool = False
    distinguished: Optional[str] = None
    removed_by_category: Optional[str] = None
    num_crossposts: Optional[int] = None
    is_original_content: bool = False
    is_self: bool = False
    view_count: Optional[int] = None
    total_awards_received: Optional[int] = None
    whitelist_status: Optional[str] = None
    media_only: Optional[bool] = None
    edited_ts: Optional[float] = None
    subreddit_subscribers: Optional[int] = None
    thumbnail_url: Optional[str] = None
    thumbnail_width: Optional[int] = None
    thumbnail_height: Optional[int] = None

    def posting_id(self) -> str:
        return f"{self.platform}:{self.post_id}"

def _safe_author(subm: Submission):
    try:
        a = subm.author
        if a is None:
            return None, None
        # fullname 可能为 t2_xxx
        auth_id = getattr(subm, "author_fullname", None)
        name = getattr(a, "name", None)
        return auth_id, name
    except Exception:
        return None, None

def _guess_ct_from_url(u: str) -> Optional[str]:
    ul = u.lower()
    if ul.endswith(MEDIA_IMG_EXT):
        ext = ul.rsplit(".", 1)[-1]
        return f"image/{'jpeg' if ext=='jpg' else ext}"
    if ul.endswith(".mp4"):
        return "video/mp4"
    if ul.endswith(".webm"):
        return "video/webm"
    if ul.endswith((".m3u8",)):
        return "application/vnd.apple.mpegurl"
    if ul.endswith((".mpd",)):
        return "application/dash+xml"
    if ul.endswith(".gif") or ul.endswith(".gifv"):
        return "image/gif"
    return None

def _add_media(lst: List[MediaItem], item: MediaItem):
    # 去重（按 URL）
    if any(m.media_url == item.media_url for m in lst):
        return
    lst.append(item)

def extract_media(subm: Submission) -> List[MediaItem]:
    """
    尽量覆盖 Reddit 的所有常见媒体形态：
      - 单图：i.redd.it / 外站直链
      - 相册：gallery_data + media_metadata
      - 视频：v.redd.it（fallback MP4 + DASH/HLS）
      - 预览图：preview
      - 外链缩略图：oembed 不下载，仅给 thumb
    """
    items: List[MediaItem] = []
    url = subm.url or ""

    # 1) gallery（相册）
    try:
        if getattr(subm, "is_gallery", False) and hasattr(subm, "gallery_data") and hasattr(subm, "media_metadata"):
            for g in (subm.gallery_data or {}).get("items", []):
                mid = g.get("media_id")
                if not mid:
                    continue
                meta = (subm.media_metadata or {}).get(mid) or {}
                # 取最大图变体
                p = meta.get("p") or []
                if p:
                    last = p[-1]
                    u = last.get("u")
                    w = last.get("x"); h = last.get("y")
                else:
                    u = (meta.get("s") or {}).get("u")
                    w = (meta.get("s") or {}).get("x"); h = (meta.get("s") or {}).get("y")
                if u:
                    _add_media(items, MediaItem(
                        media_url=u.replace("&amp;", "&"),
                        kind="image",
                        content_type=_guess_ct_from_url(u),
                        width=w, height=h,
                        source="i.redd.it",
                        downloadable=True
                    ))
    except Exception:
        pass

    # 2) 原生视频（v.redd.it）
    try:
        if subm.is_video:
            # PRAW 提供 submission.media / secure_media 字段
            m = getattr(subm, "media", None) or getattr(subm, "secure_media", None)
            rv = None
            if isinstance(m, dict):
                rv = m.get("reddit_video") or (m.get("oembed") or {}).get("reddit_video")
            if rv is None and hasattr(subm, "secure_media") and subm.secure_media:
                rv = subm.secure_media.get("reddit_video")
            if rv and isinstance(rv, dict):
                # fallback MP4（带音频轨的“合流 MP4”较少，通常需 DASH 组合；这里先落盘 mp4）
                fb = rv.get("fallback_url")
                if fb:
                    _add_media(items, MediaItem(
                        media_url=fb, kind="video_mp4",
                        content_type="video/mp4", source="v.redd.it", downloadable=True
                    ))
                # DASH/HLS 留给后处理（下载器决定是否抓取分片/清单）
                dash = rv.get("dash_url") or rv.get("dashManifest")
                if dash:
                    _add_media(items, MediaItem(
                        media_url=dash, kind="video_dash",
                        content_type="application/dash+xml", source="v.redd.it", downloadable=False
                    ))
                hls = rv.get("hls_url")
                if hls:
                    _add_media(items, MediaItem(
                        media_url=hls, kind="video_hls",
                        content_type="application/vnd.apple.mpegurl", source="v.redd.it", downloadable=False
                    ))
    except Exception:
        pass

    # 3) 直链单图/动图/外站直链
    try:
        if url.lower().endswith(MEDIA_DIRECT_EXT):
            kind = "image"
            if url.lower().endswith(".mp4"):
                kind = "video_mp4"
            elif url.lower().endswith((".gif", ".gifv")):
                kind = "gif"
            _add_media(items, MediaItem(
                media_url=url, kind=kind,
                content_type=_guess_ct_from_url(url),
                source=re.sub(r"^https?://(www\.)?", "", subm.domain or "").strip("/"),
                downloadable=True
            ))
    except Exception:
        pass

    # 4) i.redd.it 单图（非直链扩展，但给了 preview）
    try:
        prev = getattr(subm, "preview", None) or {}
        images = prev.get("images") or []
        for im in images:
            # 最大分辨率
            src = (im.get("source") or {})
            if src.get("url"):
                u = src["url"].replace("&amp;", "&")
                _add_media(items, MediaItem(
                    media_url=u, kind="image",
                    content_type=_guess_ct_from_url(u),
                    width=src.get("width"), height=src.get("height"),
                    source="i.redd.it", downloadable=True
                ))
    except Exception:
        pass

    return items

def normalize_submission(subm: Submission) -> PostRecord:
    author_id, author_name = _safe_author(subm)
    rec = PostRecord(
        subreddit=getattr(subm, "subreddit", None).display_name if getattr(subm, "subreddit", None) else None,
        post_id=subm.id,
        url=subm.url or "",
        permalink="https://www.reddit.com" + (getattr(subm, "permalink", "") or ""),
        created_utc=float(getattr(subm, "created_utc", 0) or 0),
        author_id=author_id,
        author_name=author_name,
        title=getattr(subm, "title", None),
        is_nsfw=bool(getattr(subm, "over_18", False)),
        over_18=bool(getattr(subm, "over_18", False)),
        is_video=bool(getattr(subm, "is_video", False)),
        is_gallery=bool(getattr(subm, "is_gallery", False)),
        stickied=bool(getattr(subm, "stickied", False)),
        crosspost_parent=getattr(subm, "crosspost_parent", None),
        score=getattr(subm, "score", None),
        ups=getattr(subm, "ups", None),
        upvote_ratio=getattr(subm, "upvote_ratio", None),
        num_comments=getattr(subm, "num_comments", None),
        gilded=getattr(subm, "gilded", None),
        domain=getattr(subm, "domain", None),
        url_overridden_by_dest=getattr(subm, "url_overridden_by_dest", None),
        post_hint=getattr(subm, "post_hint", None),
    )
    rec.selftext = getattr(subm, "selftext", None)
    rec.selftext_html = getattr(subm, "selftext_html", None)
    rec.link_flair_text = getattr(subm, "link_flair_text", None)
    rec.link_flair_richtext = list(getattr(subm, "link_flair_richtext", []) or [])
    rec.flair_template_id = getattr(subm, "link_flair_template_id", None)
    rec.author_flair_text = getattr(subm, "author_flair_text", None)
    rec.author_flair_css_class = getattr(subm, "author_flair_css_class", None)
    rec.author_flair_richtext = list(getattr(subm, "author_flair_richtext", []) or [])
    rec.author_is_blocked = getattr(subm, "author_is_blocked", None)
    rec.author_premium = getattr(subm, "author_premium", None)
    rec.spoiler = bool(getattr(subm, "spoiler", False))
    rec.locked = bool(getattr(subm, "locked", False))
    rec.archived = bool(getattr(subm, "archived", False))
    rec.pinned = bool(getattr(subm, "pinned", False))
    rec.distinguished = getattr(subm, "distinguished", None)
    rec.removed_by_category = getattr(subm, "removed_by_category", None)
    rec.num_crossposts = getattr(subm, "num_crossposts", None)
    rec.is_original_content = bool(getattr(subm, "is_original_content", False))
    rec.is_self = bool(getattr(subm, "is_self", False))
    rec.view_count = getattr(subm, "view_count", None)
    rec.total_awards_received = getattr(subm, "total_awards_received", None)
    rec.whitelist_status = getattr(subm, "whitelist_status", None)
    rec.media_only = getattr(subm, "media_only", None)
    edited_val = getattr(subm, "edited", None)
    if isinstance(edited_val, bool):
        rec.edited_ts = None if not edited_val else float(getattr(subm, "created_utc", 0) or 0)
    else:
        try:
            rec.edited_ts = float(edited_val) if edited_val is not None else None
        except (TypeError, ValueError):
            rec.edited_ts = None
    rec.subreddit_subscribers = getattr(subm, "subreddit_subscribers", None)
    thumb = getattr(subm, "thumbnail", None)
    rec.thumbnail_url = thumb if isinstance(thumb, str) and thumb else None
    rec.thumbnail_width = getattr(subm, "thumbnail_width", None)
    rec.thumbnail_height = getattr(subm, "thumbnail_height", None)

    rec.media = extract_media(subm)
    return rec

def iter_posts(reddit, subs: List[str], query: str, days: int, min_score: int, per_sub_limit: int) -> Iterable[PostRecord]:
    """
    对每个 subreddit：
      - 有 query：search(sort=new, time_filter='week'/'month')
      - 无 query：new() 流
    逐条规范化，并做最小过滤（时间、分数）。
    """
    now = time.time()
    MIN_DELAY, MAX_DELAY = 2.0, 5.0  # 控制平均 1 req/s 左右
    MAX_BACKOFF = 300               # 429 时的最长等待（单位：秒）

    for sub in subs:
        backoff = 10  # 初始退避
        while True:
            try:
                sr = reddit.subreddit(sub)
                # Reddit 搜索的 time_filter 只有固定粒度；days>7 时用 'month' 较稳妥
                if query:
                    tf = 'week' if days <= 7 else ('month' if days <= 31 else 'year')
                    stream = sr.search(query=query, sort='new', time_filter=tf, limit=None)
                else:
                    stream = sr.new(limit=None)

                count = 0
                for s in stream:
                    # 时间过滤
                    if days > 0 and (now - float(getattr(s, "created_utc", now))) > days * 86400:
                        continue
                    # 分数过滤
                    if min_score and int(getattr(s, "score", 0) or 0) < min_score:
                        continue
                    rec = normalize_submission(s)
                    yield rec
                    count += 1
                    if per_sub_limit and count >= per_sub_limit:
                        break

                # 子版块处理成功，跳出重试循环
                break
            except prawcore.exceptions.TooManyRequests as e:
                wait = getattr(e, "sleep_time", None)
                if wait is None:
                    wait = backoff
                    backoff = min(backoff * 2, MAX_BACKOFF)
                else:
                    wait = max(10, min(int(wait), MAX_BACKOFF))
                print(f"[rate limit] subreddit {sub} hit 429, sleeping {wait}s...")
                time.sleep(wait)
                continue
            except Exception as e:
                print(f"[warn] subreddit {sub} fetch error: {e}")
                break
        # 子版块之间加入抖动延时，避免瞬时突发
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
