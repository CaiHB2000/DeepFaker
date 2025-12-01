# p2p/datasource/connectors/reddit_connector.py
import praw
import requests
from typing import List, Dict, Optional
from p2p.datasource.schema import ProvenanceExternalMeta
from praw.exceptions import RedditAPIException  # 导入 Reddit API 异常


class RedditConnector:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        初始化 RedditConnector，配置 Reddit API 的凭证
        :param client_id: Reddit API 的 client_id
        :param client_secret: Reddit API 的 client_secret
        :param user_agent: Reddit API 的 user_agent
        """
        self.reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  user_agent=user_agent)

    def fetch_submission(self, submission_id: str) -> Optional[ProvenanceExternalMeta]:
        """
        获取指定 Reddit 帖子的内容和相关媒体
        :param submission_id: Reddit 帖子的 ID
        :return: ProvenanceExternalMeta 对象，包含媒体信息
        """
        try:
            # 获取 Reddit 帖子
            submission = self.reddit.submission(id=submission_id)

            # 如果帖子被删除或不可访问，Reddit 会抛出异常
            media_urls = self.extract_media(submission)

            # 创建 ProvenanceExternalMeta 对象
            meta = ProvenanceExternalMeta(url=submission.url,
                                          domain="reddit.com",
                                          ts_collected=submission.created_utc)

            # 如果媒体内容存在，加入 meta
            if media_urls:
                meta.http = {"media_urls": media_urls}

            return meta
        except RedditAPIException as e:
            print(f"Error fetching submission {submission_id}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching submission {submission_id}: {e}")
            return None

    def extract_media(self, submission) -> List[str]:
        """
        从 Reddit 帖子中提取媒体 URL（图片、视频）
        :param submission: Reddit 帖子对象
        :return: 包含媒体 URL 的列表
        """
        media_urls = []

        # 如果帖子包含图片
        if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
            media_urls.append(submission.url)

        # 如果帖子包含视频
        if hasattr(submission, 'media') and submission.media and 'reddit_video' in submission.media:
            video_url = submission.media['reddit_video']['fallback_url']
            media_urls.append(video_url)

        # 返回媒体资源
        return media_urls

    def fetch_multiple_submissions(self, submission_ids: List[str]) -> List[ProvenanceExternalMeta]:
        """
        批量获取多个 Reddit 帖子的内容和相关媒体
        :param submission_ids: Reddit 帖子的 ID 列表
        :return: 包含多个 ProvenanceExternalMeta 对象的列表
        """
        results = []
        for submission_id in submission_ids:
            result = self.fetch_submission(submission_id)
            if result:
                results.append(result)
            else:
                print(f"未能抓取帖子 {submission_id}")
        return results
