import os, csv, json
from typing import Iterable, Dict, List

def ensure_dir(path: str):
    """确保目录存在"""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def write_csv(path: str, rows: Iterable[Dict], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

def read_csv_as_dict(path: str, key="image"):
    """以 image 为主键读 CSV，返回 {key: row_dict}"""
    out = {}
    with open(path, "r") as f:
        it = csv.DictReader(f)
        for r in it:
            out[r[key]] = r
    return out

def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
