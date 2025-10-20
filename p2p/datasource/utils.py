import json
from typing import Iterable, Dict, Any, Optional
from pathlib import Path

def write_jsonl(path: str, records: Iterable[Dict[str, Any]], ensure_dir: bool = True) -> int:
    p = Path(path)
    if ensure_dir:
        p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def append_jsonl(path: str, record: Dict[str, Any], ensure_dir: bool = True) -> None:
    p = Path(path)
    if ensure_dir:
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def read_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s

def normalize_header_dict(headers) -> Dict[str, str]:
    # requests.structures.CaseInsensitiveDict -> plain dict[str,str]
    return {str(k): str(v) for k, v in headers.items()}

def extract_host_from_url(url: str) -> Optional[str]:
    from urllib.parse import urlparse
    try:
        return urlparse(url).hostname
    except Exception:
        return None
