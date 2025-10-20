# p2p/provenance/base.py
from abc import ABC, abstractmethod
from typing import Iterable, List, Dict

class IProvenanceProbe(ABC):
    @abstractmethod
    def run(self, items: Iterable[Dict]) -> List[Dict]:
        """
        输入：若干条记录（至少包含 'image' 主键，通常来自 content 输出）
        输出：[{ 'image': str, 'cred_score': float }, ...]
        """
        ...
