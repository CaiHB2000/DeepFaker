# p2p/propagation/base.py
from abc import ABC, abstractmethod
from typing import Iterable, List, Dict

class IPropagationSim(ABC):
    @abstractmethod
    def run(self, items: Iterable[Dict]) -> List[Dict]:
        """
        输入：[{ 'image': str, 'content_score': float, 'cred_score': float }, ...]
        输出：[{ 'image': str, 'early_risk': float }, ...]
        """
        ...
