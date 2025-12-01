import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Iterable, List, Optional, Dict, Any

from p2p.datasource.schema import PDQResult, VPDQResult, VPDQFrame, FileFingerprints

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import pdqhash as pdqhash_py  # type: ignore
except ImportError:  # pragma: no cover
    pdqhash_py = None

def _which(*names: str) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None

# ---------- PDQ (image) ----------
def _pdq_vector_to_hex(vec: Iterable[int]) -> Optional[str]:
    bits: List[int] = []
    for v in vec:
        # pdqhash returns +/-1 or booleans; treat >0 as 1 bit
        bits.append(1 if v > 0 else 0)
    if not bits:
        return None
    if len(bits) % 4 != 0:
        # pad trailing zeros to align nibbles (should not happen for PDQ)
        pad = 4 - (len(bits) % 4)
        bits.extend([0] * pad)
    hex_chars: List[str] = []
    for i in range(0, len(bits), 4):
        nibble = (bits[i] << 3) | (bits[i + 1] << 2) | (bits[i + 2] << 1) | bits[i + 3]
        hex_chars.append(f"{nibble:x}")
    return "".join(hex_chars)

def pdq_for_image(path: str) -> PDQResult:
    if pdqhash_py and cv2:
        try:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                return PDQResult(path=path, available=False, error="cv2_read_failed")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hash_vec, quality = pdqhash_py.compute(image)
            hash_hex = _pdq_vector_to_hex(hash_vec) if hash_vec is not None else None
            return PDQResult(
                path=path,
                available=True,
                hash_hex=hash_hex,
                quality=float(quality) if quality is not None else None,
                raw_text=None,
                error=None,
            )
        except Exception as e:  # pragma: no cover
            return PDQResult(path=path, available=False, error=f"pdq_python_exception:{e}")

    exe = _which("pdqhash", "pdq-photo-hasher", "pdq")
    if not exe:
        return PDQResult(path=path, available=False, error="pdq_cli_not_found")

    try:
        p = subprocess.run([exe, path], capture_output=True, text=True, timeout=30)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        if p.returncode != 0:
            return PDQResult(path=path, available=False, raw_text=out.strip(), error=f"pdq_cli_exit_{p.returncode}")
        m = re.search(r"\b([0-9a-fA-F]{64})\b", out)
        hx = m.group(1).lower() if m else None
        return PDQResult(path=path, available=True, hash_hex=hx, raw_text=out.strip())
    except subprocess.TimeoutExpired:
        return PDQResult(path=path, available=False, error="pdq_cli_timeout")
    except Exception as e:
        return PDQResult(path=path, available=False, error=f"pdq_cli_exception:{e}")

# ---------- vPDQ (video) ----------
def _parse_vpdq_text(text: str) -> List[VPDQFrame]:
    """
    解析 vpdq-hash-video 生成的文本格式。
    我们做“宽松解析”：每行抓一个 64位十六进制 hash，尽量从同一行找两个数字作为 ts/quality。
    """
    frames: List[VPDQFrame] = []
    for line in text.splitlines():
        m_hash = re.search(r"\b([0-9a-fA-F]{64})\b", line)
        if not m_hash:
            continue
        hx = m_hash.group(1).lower()
        nums = re.findall(r"\d+\.\d+|\d+", line)
        ts = float(nums[0]) if nums else 0.0
        q = float(nums[-1]) if len(nums) >= 2 else 0.0
        frames.append(VPDQFrame(ts=ts, hash_hex=hx, quality=q))
    return frames

def vpdq_for_video(path: str) -> VPDQResult:
    exe = _which("vpdq-hash-video")
    if not exe:
        return VPDQResult(path=path, available=False, error="vpdq_cli_not_found")

    try:
        with tempfile.TemporaryDirectory() as td:
            out_txt = os.path.join(td, "hashes.txt")
            # -r 1.0: 每 1 秒取一个哈希
            cmd = [exe, "-i", path, "-o", out_txt, "-r", "1.0"]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if p.returncode != 0 and not os.path.exists(out_txt):
                return VPDQResult(path=path, available=False, error=f"vpdq_cli_exit_{p.returncode}")

            raw_text = ""
            if os.path.exists(out_txt):
                try:
                    # ✅ 用 with 确保文件句柄关闭，消除 ResourceWarning
                    with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                        raw_text = f.read()
                except Exception:
                    raw_text = ""

            frames = _parse_vpdq_text(raw_text) if raw_text else []
            return VPDQResult(
                path=path,
                available=True,
                frames=frames,
                raw_json={
                    "raw_text": (raw_text[:2000] if raw_text else ""),
                    "stdout": (p.stdout or "")[:1000],
                    "stderr": (p.stderr or "")[:1000],
                },
                error=None if (frames or raw_text) else "vpdq_empty_output",
            )
    except subprocess.TimeoutExpired:
        return VPDQResult(path=path, available=False, error="vpdq_cli_timeout")
    except Exception as e:
        return VPDQResult(path=path, available=False, error=f"vpdq_cli_exception:{e}")



# ---------- 汇总 ----------
def fingerprints_for_path(path: str) -> FileFingerprints:
    fp = FileFingerprints(path=path)
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"):
            fp.pdq = pdq_for_image(path)
            if fp.pdq and not fp.pdq.available:
                fp.errors.append(fp.pdq.error or "pdq_unknown_error")
        elif ext in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".ogv"):
            fp.vpdq = vpdq_for_video(path)
            if fp.vpdq and not fp.vpdq.available:
                fp.errors.append(fp.vpdq.error or "vpdq_unknown_error")
        else:
            fp.errors.append("unsupported_extension")
    except Exception as e:
        fp.errors.append(f"fingerprint_exception:{e}")
    return fp
