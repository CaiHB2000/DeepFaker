# p2p/policy/simple_threshold.py
from typing import Dict
from p2p.policy.base import IPolicy
from p2p.core.registries import register, POLICY_REG

@register(POLICY_REG, "simple_threshold")
class SimpleThreshold(IPolicy):
    """
    简单阈值策略：
      p2p > hi  → limit (factor=hi_f)
      p2p > mid → label (factor=mid_f)
      else      → none  (factor=1.0)
    """
    def __init__(self, hi=0.8, mid=0.6, hi_f=0.3, mid_f=0.6):
        self.hi, self.mid, self.hi_f, self.mid_f = hi, mid, hi_f, mid_f

    def apply(self, row: Dict) -> Dict:
        p = float(row["p2p_risk"])
        if p > self.hi:
            return {"policy": "limit", "factor": self.hi_f}
        if p > self.mid:
            return {"policy": "label", "factor": self.mid_f}
        return {"policy": "none", "factor": 1.0}
