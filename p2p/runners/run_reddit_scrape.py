import argparse
import os
import praw
from datetime import datetime, timedelta
from typing import List


def build_argparser():
    ap = argparse.ArgumentParser(description="抓取 Reddit 帖子及其媒体内容")
    ap.add_argument("--subs", type=str, default="pics,news,worldnews", help="Comma-separated subreddits, or 'all'")
    ap.add_argument("--days", type=int, default=7, help="Only include posts created within last N days")
    ap.add_argument("--limit", type=int, default=40, help="Total number of posts to collect across all subs")
    ap.add_argument("--query", type=str, default="", help="Optional search query (Reddit search syntax).")
    ap.add_argument("--min-score", type=int, default=0, help="Filter by minimum post score (upvotes).")
    ap.add_argument("--include-nsfw", action="store_true", help="Include NSFW posts (default false). Not recommended.")
    ap.add_argument("--out", type=str, default="tmp/seed_urls.txt", help="Output file path")
    return ap


def created_within_days(post: praw.models.Submission, days: int) -> bool:
    """
    Check if the post was created within the last `days` days.
    :param post: PRAW Submission object
    :param days: Number of days to check
    :return: True if the post was created within `days` days, else False
    """
    t = post.created_utc
    if t is None:
        return True
    dt = datetime.utcfromtimestamp(t)
    return dt >= (datetime.utcnow() - timedelta(days=days))


def fetch_from_subreddit_praw(sub: str, limit: int, days: int, query: str, min_score: int, reddit) -> List[dict]:
    """
    Fetch posts from a subreddit using PRAW, including media URLs and detailed metadata.
    :param sub: Subreddit name
    :param limit: Maximum number of posts to fetch
    :param days: Fetch posts created within the last `days` days
    :param query: Search query (optional)
    :param min_score: Minimum post score to include
    :param reddit: PRAW Reddit instance
    :return: List of posts with metadata and media URLs
    """
    out = []
    try:
        # Get the subreddit
        subreddit = reddit.subreddit(sub)

        # Search or get the latest posts
        if query:
            posts = subreddit.search(query, sort='new', time_filter='week', limit=limit)
        else:
            posts = subreddit.new(limit=limit)

        for post in posts:
            # Check if the post was created within the specified number of days
            if created_within_days(post, days) and post.score >= min_score:
                post_data = {
                    "id": post.id,
                    "url": post.url,
                    "title": post.title,
                    "author": str(post.author) if post.author else "unknown",
                    "created_utc": post.created_utc,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "subreddit": post.subreddit.display_name,
                    "is_nsfw": post.over_18,  # NSFW flag
                    "media_urls": []
                }

                # Check if the post has media content (image/video)
                if post.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                    post_data["media_urls"].append(post.url)
                if 'reddit_video' in post.__dict__:
                    post_data["media_urls"].append(post.media['reddit_video']['fallback_url'])

                out.append(post_data)

            # Stop if we reach the limit
            if len(out) >= limit:
                break
    except Exception as e:
        print(f"[warn] Error fetching from subreddit {sub}: {e}")
    return out


def main():
    ap = build_argparser()
    args = ap.parse_args()

    # Reddit API credentials
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = "your_app_name"  # Replace with your app's name

    if not client_id or not client_secret:
        print("Please ensure REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables are set.")
        return

    # Initialize Reddit API with PRAW
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)

    # List of subreddits
    subs = [s.strip() for s in args.subs.split(",") if s.strip()]
    total_limit = max(1, args.limit)

    seen_ids = set()
    posts = []

    for sub in subs:
        try:
            # Fetch posts from the subreddit
            res = fetch_from_subreddit_praw(sub=sub, limit=total_limit, days=args.days,
                                            query=args.query, min_score=args.min_score,
                                            reddit=reddit)
        except Exception as e:
            print(f"[warn] Fetch {sub} failed: {e}")
            res = []

        for p in res:
            pid = p["id"]
            if pid not in seen_ids:
                posts.append({
                    "id": pid,
                    "url": p["url"],
                    "title": p["title"],
                    "author": p["author"],
                    "created_utc": p["created_utc"],
                    "score": p["score"],
                    "num_comments": p["num_comments"],
                    "subreddit": p["subreddit"],
                    "is_nsfw": p["is_nsfw"],
                    "media_urls": p["media_urls"]
                })
                seen_ids.add(pid)

            if len(posts) >= total_limit:
                break
        print(f"[info] {sub}: added {len(posts)} posts.")
        if len(posts) >= total_limit:
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in posts:
            f.write(p["url"] + "\n")

    print(f"[done] Wrote {len(posts)} URLs -> {args.out}")
    if len(posts) == 0:
        print("[hint] Try lowering --min-score, increasing --days, adding more subs.")


if __name__ == "__main__":
    main()
