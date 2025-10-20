# p2p/aggregator/base.py
from abc import ABC, abstractmethod
from typing import Iterable, List, Dict

class IRiskAggregator(ABC):
    @abstractmethod
    def run(self, items: Iterable[Dict]) -> List[Dict]:
        """
        输入：[{ 'image','label?','content_score','cred_score','early_risk' }, ...]
        输出：在记录上新增 'p2p_risk' 字段
        """
        ...
