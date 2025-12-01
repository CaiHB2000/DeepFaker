import json
import shutil
import subprocess
from typing import Optional

from p2p.datasource.schema import C2PAReport

def _which_c2patool() -> str:
    exe = shutil.which("c2patool")
    return exe or ""

def analyze_one(path: str) -> C2PAReport:
    exe = _which_c2patool()
    if not exe:
        return C2PAReport(path=path, present=False, error="c2patool_not_found")

    # 尝试多种调用方式：某些版本支持 -j，有的不支持
    candidates = [
        [exe, path, "-j"],
        [exe, path, "info", "-j"],
        [exe, path],
    ]

    last_stdout = ""
    last_stderr = ""
    last_code = None

    for cmd in candidates:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            last_stdout, last_stderr, last_code = p.stdout or "", p.stderr or "", p.returncode
            # 直接成功
            if p.returncode == 0:
                # 尝试 JSON 解析
                data = {}
                try:
                    if last_stdout.strip().startswith("{") or last_stdout.strip().startswith("["):
                        data = json.loads(last_stdout)
                except Exception:
                    # 非 JSON 输出也可，保留原文
                    data = {"raw_text": last_stdout.strip()}
                # 是否能看出有效验证结果（可缺省）
                vendor = None
                claim_gen = None
                assertions = None
                valid = None
                try:
                    mfs = data.get("manifests", []) if isinstance(data, dict) else []
                    if mfs:
                        mf0 = mfs[0]
                        claim_gen = mf0.get("claimGenerator")
                        assertions = len(mf0.get("assertions", []))
                    ver = data.get("verification", data.get("verifications")) if isinstance(data, dict) else None
                    if isinstance(ver, dict):
                        valid = (ver.get("overall") in ("OK", "VALID")) or bool(ver.get("isValid"))
                    vendor = (data.get("vendor") if isinstance(data, dict) else None) or (mfs and mfs[0].get("vendor"))
                except Exception:
                    pass
                return C2PAReport(
                    path=path, present=True, valid=valid, vendor=vendor,
                    claim_generator=claim_gen, assertions_count=assertions, raw=data
                )
            # 失败但可能是“无 Claim”
            else:
                if "No claim found" in (last_stderr or ""):
                    return C2PAReport(path=path, present=False, error="no_claim")
                # 有的版本会把提示写到 stdout
                if "No claim found" in (last_stdout or ""):
                    return C2PAReport(path=path, present=False, error="no_claim")
        except subprocess.TimeoutExpired:
            return C2PAReport(path=path, present=False, error="c2patool_timeout")
        except Exception as e:
            last_stderr = f"c2patool_exception:{e}"
            break

    # 兜底：返回最后一次尝试的信息
    err = (last_stderr.strip() or "c2patool_failed")
    return C2PAReport(path=path, present=False, error=err, raw={"raw_text": last_stdout.strip()} if last_stdout else {})
