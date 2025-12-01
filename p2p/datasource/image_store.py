# p2p/datasource/image_store.py
import os, io, pickle
from typing import Optional, Dict, List
import numpy as np
import cv2
from PIL import Image

try:
    import lmdb
except Exception:
    lmdb = None  # 允许无 lmdb 环境

# --------- 通用工具 ---------
def _decode_value_to_bgr(v: bytes):
    if v is None:
        return None
    # JPEG/PNG
    if v[:3] == b'\xff\xd8\xff' or v[:8] == b'\x89PNG\r\n\x1a\n':
        arr = np.frombuffer(v, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # NumPy .npy
    if v[:6] == b'\x93NUMPY':
        try:
            arr = np.load(io.BytesIO(v), allow_pickle=True)
            if isinstance(arr, np.ndarray):
                if arr.ndim == 3 and arr.shape[2] in (1,3,4):
                    if arr.shape[2] == 3:
                        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    if arr.shape[2] == 4:
                        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        except Exception:
            pass
    # Pickle
    if v[:2] == b'\x80\x04':
        try:
            obj = pickle.loads(v)
            if isinstance(obj, dict) and 'image' in obj:
                im = obj['image']
                if isinstance(im, bytes):
                    arr = np.frombuffer(im, dtype=np.uint8)
                    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if isinstance(im, np.ndarray):
                    if im.ndim == 3 and im.shape[2] == 3:
                        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    # PIL fallback
    try:
        im = Image.open(io.BytesIO(v)); im.load()
        rgb = np.array(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def _normpath(p: str) -> str:
    return os.path.normpath(p).lstrip("/")

# --------- 键规范化策略 ---------
def default_key_normalizer(abs_path_or_key: str, anchors: Optional[List[str]] = None) -> str:
    """
    给定 CSV 中的绝对路径或任意 key，提取出 LMDB 使用的“相对键”。
    规则：
      - 若包含任一锚点（如 'Celeb-DF-v2'），返回其后的子路径；
      - 否则取最后4级路径作为兜底（<class>/frames/.../file.png）。
    """
    s = _normpath(abs_path_or_key)
    anchors = anchors or ["Celeb-DF-v2", "FaceForensics++"]
    for a in anchors:
        token = os.sep + a + os.sep
        pos = s.find(token)
        if pos != -1:
            return s[pos + len(token):].lstrip(os.sep)
    parts = s.split(os.sep)
    return os.path.join(*parts[-4:]) if len(parts) >= 4 else parts[-1]

# --------- ImageStore ----------
class ImageStore:
    """
    统一的图像读取器：
      - 可选 LMDB（带键索引）
      - 可选 RGB 绝对路径兜底
      - 可自定义 key 规范化策略
    """
    def __init__(
        self,
        lmdb_root: Optional[str] = None,
        build_index: bool = True,
        rgb_fallback: bool = True,
        key_normalizer=default_key_normalizer,
        normalizer_anchors: Optional[List[str]] = None,
    ):
        self.rgb_fallback = rgb_fallback
        self.key_normalizer = key_normalizer
        self.normalizer_anchors = normalizer_anchors or ["Celeb-DF-v2", "FaceForensics++"]

        self.env = None
        self.key_index: Dict[str, str] = {}
        if lmdb_root and lmdb:
            self.env = lmdb.open(lmdb_root, readonly=True, lock=False, readahead=False, max_readers=2048)
            if build_index:
                self._build_index()

    def _build_index(self, max_keys: int = 200000):
        if not self.env:
            return
        with self.env.begin(write=False) as txn:
            cur = txn.cursor()
            for i, (k, _) in enumerate(cur.iternext()):
                ks = k.decode("utf-8", errors="ignore")
                norm = _normpath(ks)
                self.key_index.setdefault(norm, ks)
                if i + 1 >= max_keys:
                    break

    def get_bgr(self, image_path_or_key: str):
        """
        读取一张图（返回 BGR），优先从 LMDB 查；失败时可回退到磁盘路径。
        """
        # 1) 直接 LMDB 按原值取
        if self.env:
            with self.env.begin(write=False) as txn:
                buf = txn.get(image_path_or_key.encode("utf-8"))
            if buf:
                img = _decode_value_to_bgr(buf)
                if img is not None:
                    return img

            # 2) 规范化后去索引中查真实 key（相对路径）
            qnorm = self.key_normalizer(image_path_or_key, anchors=self.normalizer_anchors)
            real = self.key_index.get(qnorm)
            if real:
                with self.env.begin(write=False) as txn:
                    buf = txn.get(real.encode("utf-8"))
                if buf:
                    img = _decode_value_to_bgr(buf)
                    if img is not None:
                        return img

        # 3) 兜底：按磁盘路径读（如果存在）
        if self.rgb_fallback and os.path.exists(image_path_or_key):
            img = cv2.imread(image_path_or_key, cv2.IMREAD_COLOR)
            if img is not None:
                return img

        return None
