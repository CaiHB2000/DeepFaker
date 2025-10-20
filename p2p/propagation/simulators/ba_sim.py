# p2p/propagation/simulators/ba_sim.py
import random, numpy as np, networkx as nx
from typing import Iterable, List, Dict
from p2p.propagation.base import IPropagationSim
from p2p.core.registries import register, PROP_REG


@register(PROP_REG, "ba_sim")
class BASimulator(IPropagationSim):
    """
    Barabási–Albert 无标度图上做早期扩散模拟：
      基础版: p_base = base + coef * content_score * (1 - cred_score)
      连续化: p = clip( p_base + noise_coef * Beta(a,b), 0, 1 )
    """
    def __init__(self, n: int = 300, steps: int = 6, seed: int = 1337,
                 base: float = 0.03, coef: float = 0.30,
                 noise_coef: float = 0.15, beta_a: float = 2.0, beta_b: float = 5.0):
        self.n, self.steps, self.seed = n, steps, seed
        self.base, self.coef = base, coef
        self.noise_coef, self.beta_a, self.beta_b = noise_coef, beta_a, beta_b

        # ✅ 在构造时固定随机源和图，避免每条样本重置
        self._rs = np.random.RandomState(self.seed)
        self._py_random = random.Random(self.seed)
        self._G = nx.barabasi_albert_graph(self.n, 3, seed=self.seed)

    def run(self, items: Iterable[Dict]) -> List[Dict]:
        out: List[Dict] = []
        for r in items:
            cs, cr = float(r["content_score"]), float(r["cred_score"])

            # 基础传播概率
            p_base = self.base + self.coef * cs * (1.0 - cr)

            # ✅ 连续化噪声（Beta）——每条样本不同，但可控
            noise = self._rs.beta(self.beta_a, self.beta_b)
            p = float(np.clip(p_base + self.noise_coef * noise, 0.0, 1.0))

            # 早期感染仿真（IC风格，固定同一张图）
            infected = {self._rs.randint(0, self.n)}
            for _ in range(self.steps):
                new = set()
                for u in infected:
                    for v in self._G.neighbors(u):
                        if v in infected:
                            continue
                        # 用 self._py_random，不要用全局 random.random()
                        if self._py_random.random() < p:
                            new.add(v)
                infected |= new

            early = float(len(infected) / self.n)
            out.append({"image": r["image"], "early_risk": early})
        return out
