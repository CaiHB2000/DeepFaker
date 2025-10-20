# p2p/content/base.py
from abc import ABC, abstractmethod
from typing import Iterable, List, Dict

class IContentDetector(ABC):
    @abstractmethod
    def run(self, csv_path: str) -> List[Dict]:
        """
        输入：
            csv_path: DeepFakeBench 导出的逐样本 csv（至少包含 image、prob，可选 label）
        输出：
            List[Dict] 形如：
            [
              {"image": <str>, "content_score": <float>, "label": <int 或 -1>},
              ...
            ]
        """
        ...
