# p2p/provenance/probes/phash_stability.py
import cv2, numpy as np, imagehash
from PIL import Image
from typing import Iterable, List, Dict
from p2p.provenance.base import IProvenanceProbe
from p2p.core.registries import register, PROV_REG
from p2p.datasource.image_store import ImageStore

def _phash_stability(img_bgr, jpeg_q=70) -> float:
    if img_bgr is None:
        return 0.5
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h1 = imagehash.phash(Image.fromarray(img_rgb))
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
    if not ok:
        return 0.5
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        return 0.5
    h2 = imagehash.phash(Image.fromarray(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)))
    dist = (h1 - h2)  # 0~64
    return float(max(0.0, min(1.0, 1.0 - dist / 64.0)))

@register(PROV_REG, "phash_stability")
class PHashStability(IProvenanceProbe):
    """
    精简版：只负责“如何从图像计算 cred_score”。
    图像读取由 ImageStore 负责（LMDB/索引/兜底）。
    """
    def __init__(self, lmdb_root: str, jpeg_q: int = 70, build_index: bool = True, rgb_fallback: bool = True):
        self.store = ImageStore(
            lmdb_root=lmdb_root,
            build_index=build_index,
            rgb_fallback=rgb_fallback,
            # anchors 可按需扩展
            normalizer_anchors=["Celeb-DF-v2", "FaceForensics++"],
        )
        self.q = jpeg_q

    def run(self, items: Iterable[Dict]) -> List[Dict]:
        out: List[Dict] = []
        miss = 0
        for r in items:
            key = r["image"]  # 绝对路径或任意 key
            img = self.store.get_bgr(key)
            if img is None:
                miss += 1
            s = _phash_stability(img, self.q)
            out.append({"image": key, "cred_score": s})
        if miss:
            print(f"[PHashStability] WARN: {miss} items failed to read (returned 0.5).")
        return out
