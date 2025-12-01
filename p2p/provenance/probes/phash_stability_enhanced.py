import cv2, numpy as np, imagehash
from PIL import Image
from typing import Iterable, List, Dict
from p2p.provenance.base import IProvenanceProbe
from p2p.core.registries import register, PROV_REG
from p2p.datasource.image_store import ImageStore

def _rand_crop_resize(img, max_ratio=0.04):
    h, w = img.shape[:2]
    ch = int(h * np.random.uniform(0.0, max_ratio))
    cw = int(w * np.random.uniform(0.0, max_ratio))
    y0 = np.random.randint(0, max(1, ch+1)) if ch>0 else 0
    x0 = np.random.randint(0, max(1, cw+1)) if cw>0 else 0
    cropped = img[y0:h-(ch-y0) if (h-(ch-y0))>y0 else h, x0:w-(cw-x0) if (w-(cw-x0))>x0 else w]
    if cropped.size == 0: cropped = img
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def _light_perturb(img_bgr, jpeg_q=30, crop_ratio=0.04, blur_sigma=0.8, jitter=0.05):
    x = img_bgr.copy()
    if crop_ratio > 0:
        x = _rand_crop_resize(x, max_ratio=crop_ratio)
    if blur_sigma > 0 and np.random.rand() < 0.7:
        x = cv2.GaussianBlur(x, (3,3), sigmaX=blur_sigma)
    if jitter > 0 and np.random.rand() < 0.7:
        alpha = 1.0 + np.random.uniform(-jitter, jitter)  # contrast
        beta  = np.random.uniform(-8, 8)                  # brightness
        x = cv2.convertScaleAbs(x, alpha=alpha, beta=beta)
    ok, enc = cv2.imencode(".jpg", x, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
    if not ok: return img_bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img_bgr

def _phash_multi(img_bgr, trials=5, jpeg_q=30, crop_ratio=0.04, blur_sigma=0.8, jitter=0.05):
    if img_bgr is None: return 0.5
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h0 = imagehash.phash(Image.fromarray(img_rgb))
    sims = []
    for _ in range(int(trials)):
        pert = _light_perturb(img_bgr, jpeg_q=jpeg_q, crop_ratio=crop_ratio, blur_sigma=blur_sigma, jitter=jitter)
        h1 = imagehash.phash(Image.fromarray(cv2.cvtColor(pert, cv2.COLOR_BGR2RGB)))
        dist = (h0 - h1)  # 0..64
        sims.append(1.0 - dist/64.0)
    return float(np.clip(np.mean(sims), 0.0, 1.0))

@register(PROV_REG, "phash_stability_enhanced")
class PHashStabilityEnhanced(IProvenanceProbe):
    """
    多次轻扰动 + 强压缩的 pHash 稳定性：更有区分度。
    参数：
      trials: 采样次数（默认5）
      jpeg_q: 重压缩质量（默认30）
      crop_ratio: 随机裁剪幅度（默认0.04=4%）
      blur_sigma: 轻模糊 sigma（默认0.8）
      jitter: 亮度/对比抖动幅度（默认0.05）
    """
    def __init__(self, lmdb_root: str, trials: int = 5, jpeg_q: int = 30,
                 crop_ratio: float = 0.04, blur_sigma: float = 0.8, jitter: float = 0.05,
                 build_index: bool = True, rgb_fallback: bool = True):
        self.store = ImageStore(lmdb_root=lmdb_root, build_index=build_index, rgb_fallback=rgb_fallback,
                                normalizer_anchors=["Celeb-DF-v2","FaceForensics++"])
        self.trials = trials
        self.jpeg_q = jpeg_q
        self.crop_ratio = crop_ratio
        self.blur_sigma = blur_sigma
        self.jitter = jitter

    def run(self, items: Iterable[Dict]) -> List[Dict]:
        out = []
        miss = 0
        for r in items:
            key = r["image"]
            img = self.store.get_bgr(key)
            if img is None: miss += 1
            s = _phash_multi(img, trials=self.trials, jpeg_q=self.jpeg_q,
                             crop_ratio=self.crop_ratio, blur_sigma=self.blur_sigma, jitter=self.jitter)
            out.append({"image": key, "cred_score": s})
        if miss:
            print(f"[PHashStabilityEnhanced] WARN: {miss} items failed to read (0.5 fallback).")
        return out
