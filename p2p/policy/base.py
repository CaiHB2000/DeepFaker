# p2p/policy/base.py
from abc import ABC, abstractmethod
from typing import Dict

class IPolicy(ABC):
    @abstractmethod
    def apply(self, row: Dict) -> Dict:
        """
        输入：一条包含 'p2p_risk' 的记录
        输出：{'policy': 'none'|'label'|'limit', 'factor': float}
        """
        ...
