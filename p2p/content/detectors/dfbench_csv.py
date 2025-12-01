# p2p/content/detectors/dfbench_csv.py
import csv
from typing import List, Dict
from p2p.core.utils import to_float
from p2p.core.registries import register, CONTENT_REG
from p2p.content.base import IContentDetector

@register(CONTENT_REG, "dfbench_csv")
class DFBenchCSV(IContentDetector):
    """
    读取 DeepFakeBench 导出的逐样本 csv：
      必需列：image, prob
      可选列：label（没有则填 -1）
    输出统一字段：image, content_score, label
    """
    def run(self, csv_path: str) -> List[Dict]:
        rows: List[Dict] = []
        with open(csv_path, "r") as f:
            it = csv.DictReader(f)
            # 兼容不同列名（prob/score）
            prob_col = "prob"
            if "prob" not in it.fieldnames and "score" in it.fieldnames:
                prob_col = "score"

            for r in it:
                rows.append({
                    "image": r["image"],
                    "content_score": to_float(r.get(prob_col, 0.0), 0.0),
                    "label": int(r.get("label", -1)) if r.get("label", None) not in (None, "") else -1,
                })
        return rows
