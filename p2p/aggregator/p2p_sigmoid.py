# p2p/aggregator/p2p_sigmoid.py
import math
from typing import Iterable, List, Dict
from p2p.aggregator.base import IRiskAggregator
from p2p.core.registries import register, AGGR_REG
import numpy as np
def _sigmoid_k(x, k=5.0):
    return 1.0 / (1.0 + math.exp(-k * (x - 0.5)))

@register(AGGR_REG, "p2p_sigmoid")
class P2PSigmoid(IRiskAggregator):
    """
    P2P = σ( k * ( α*(1-cred) + β*content + γ*early ) )

    rescale:
      - "none"    → 不拉伸，直接用 sigmoid 输出
      - "linear"  → 全局线性映射到 [expand_lo, expand_hi]
      - "quantile"→ 分位三段线性映射（低/中/高三段，各自线性映射，保持排序但让三段都可见）
    """
    def __init__(self,
                 alpha=0.33, beta=0.33, gamma=0.34, k=5.0,
                 rescale: str = "none",
                 # linear 模式参数
                 expand_to=None,              # 例如传 (0.7, 0.9)
                 # quantile 模式参数
                 q_low: float = 0.20, q_high: float = 0.80,
                 tgt_low=(0.05, 0.35),
                 tgt_mid=(0.35, 0.75),
                 tgt_high=(0.75, 0.98)):
        self.alpha, self.beta, self.gamma, self.k = alpha, beta, gamma, k
        self.rescale = (rescale or "none").lower()
        self.expand_to = expand_to
        self.q_low, self.q_high = q_low, q_high
        self.tgt_low, self.tgt_mid, self.tgt_high = tgt_low, tgt_mid, tgt_high

    def _sigmoid(self, x: float) -> float:
        return _sigmoid_k(x, self.k)

    def _linmap(self, x, a1, a2, b1, b2):
        if abs(a2 - a1) < 1e-12:
            return (b1 + b2) / 2.0
        t = (x - a1) / (a2 - a1)
        return b1 + t * (b2 - b1)

    def _rescale_linear(self, vals, lo, hi):
        vmin, vmax = min(vals), max(vals)
        if vmax <= vmin + 1e-12:
            mid = (lo + hi) / 2.0
            return [mid for _ in vals]
        return [max(0.0, min(1.0, lo + (v - vmin) * (hi - lo) / (vmax - vmin))) for v in vals]

    def _rescale_quantile(self, vals):
        import numpy as np
        v = np.asarray(vals, dtype=np.float32)
        vmin, vmax = float(np.min(v)), float(np.max(v))
        ql = float(np.quantile(v, self.q_low))
        qh = float(np.quantile(v, self.q_high))
        L1, L2 = self.tgt_low
        M1, M2 = self.tgt_mid
        H1, H2 = self.tgt_high

        out = np.empty_like(v)
        # 低段
        mask = v <= ql
        out[mask] = self._linmap(v[mask], vmin, ql, L1, L2) if np.any(mask) else np.array([], dtype=v.dtype)
        # 中段
        mask = (v > ql) & (v <= qh)
        out[mask] = self._linmap(v[mask], ql, qh, M1, M2) if np.any(mask) else np.array([], dtype=v.dtype)
        # 高段
        mask = v > qh
        out[mask] = self._linmap(v[mask], qh, vmax, H1, H2) if np.any(mask) else np.array([], dtype=v.dtype)

        return np.clip(out, 0.0, 1.0).tolist()

    def run(self, items: Iterable[Dict]) -> List[Dict]:
        cache, vals = [], []
        for r in items:
            cred  = float(r["cred_score"])
            cont  = float(r["content_score"])
            early = float(r["early_risk"])
            x = self.alpha * (1.0 - cred) + self.beta * cont + self.gamma * early
            p2p_raw = self._sigmoid(x)  # ∈(0,1)
            cache.append((r, p2p_raw))
            vals.append(p2p_raw)

        # 默认不拉伸
        scaled = [v for v in vals]

        # 线性或分位三段拉伸
        if self.rescale == "linear" and self.expand_to and len(vals) > 1:
            lo, hi = self.expand_to
            scaled = self._rescale_linear(vals, lo, hi)
        elif self.rescale == "quantile" and len(vals) > 3:
            scaled = self._rescale_quantile(vals)

        out: List[Dict] = []
        for (r, p_raw), p in zip(cache, scaled):
            r2 = dict(r)
            r2["p2p_risk_raw"] = float(p_raw)         # 原始（用于真实评测/阈值）
            r2["p2p_risk"]     = float(p)             # 重标（用于可视化/策略调参）
            out.append(r2)
        return out
