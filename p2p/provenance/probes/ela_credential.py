import cv2, numpy as np
from typing import Iterable, List, Dict
from p2p.provenance.base import IProvenanceProbe
from p2p.core.registries import register, PROV_REG
from p2p.datasource.image_store import ImageStore

def _ela_score(img_bgr, jpeg_q=90):
    if img_bgr is None: return 0.5
    # 转为 JPEG(Q=jpeg_q) 再读回
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
    if not ok: return 0.5
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None: return 0.5
    # 残差 & 归一化
    diff = cv2.absdiff(img_bgr, dec).astype(np.float32)
    # 用能量/均值作为残差指标（也可换成 SSIM 反向）
    energy = float(np.mean(diff))  # 0..255
    # 映射为可信度：能量越小越“稳”，做一个简单的线性归一化
    # 经验：把 0..30 映射到 1..0；可按数据微调阈值 30
    t = 30.0
    cred = 1.0 - np.clip(energy / t, 0.0, 1.0)
    return float(cred)

@register(PROV_REG, "ela_credential")
class ELACredential(IProvenanceProbe):
    """
    ELA 残差能量 → 可信度（越小残差→越可信）
    参数：
      jpeg_q: 保存质量（默认90）
      thresh: 内部能量归一化阈值（默认30，用于 0..t 线性缩放）
    """
    def __init__(self, lmdb_root: str, jpeg_q: int = 90, build_index: bool = True, rgb_fallback: bool = True, thresh: float = 30.0):
        self.store = ImageStore(lmdb_root=lmdb_root, build_index=build_index, rgb_fallback=rgb_fallback,
                                normalizer_anchors=["Celeb-DF-v2","FaceForensics++"])
        self.jpeg_q = jpeg_q
        self.thresh = thresh

    def run(self, items: Iterable[Dict]) -> List[Dict]:
        out = []
        miss = 0
        for r in items:
            key = r["image"]
            img = self.store.get_bgr(key)
            if img is None: miss += 1
            # 复用局部函数，但把阈值注入映射
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_q)]) if img is not None else (False, None)
            if not ok or enc is None:
                s = 0.5
            else:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is None:
                    s = 0.5
                else:
                    diff = cv2.absdiff(img, dec).astype(np.float32)
                    energy = float(np.mean(diff))
                    s = float(1.0 - np.clip(energy / self.thresh, 0.0, 1.0))
            out.append({"image": key, "cred_score": s})
        if miss:
            print(f"[ELACredential] WARN: {miss} items failed to read (0.5 fallback).")
        return out
