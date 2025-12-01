from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from torch.utils.data import Dataset


@dataclass
class WeFENDSample:
    guid: str
    text: str
    image_path: Optional[Path]
    label: int
    event_id: Optional[int] = None
    event_size: int = 0


_ACCOUNT_CACHE: Dict[Path, Tuple[Dict[str, int], Dict[str, str]]] = {}


def _load_account_mapping(root: Path) -> Tuple[Dict[str, int], Dict[str, str]]:
    if root in _ACCOUNT_CACHE:
        return _ACCOUNT_CACHE[root]

    metadata_dir = root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    mapping_path = metadata_dir / "guid_to_account.json"

    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as handle:
            guid_to_account = json.load(handle)
    else:
        guid_to_account: Dict[str, str] = {}
        for split in ("train/news.csv", "test/news.csv"):
            csv_path = root / split
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                guid = str(row.get("News Url", "")).strip()
                account = str(row.get("Ofiicial Account Name", "")).strip()
                if guid and account:
                    guid_to_account[guid] = account
        with mapping_path.open("w", encoding="utf-8") as handle:
            json.dump(guid_to_account, handle)

    account_to_index: Dict[str, int] = {
        account: idx for idx, account in enumerate(sorted(set(guid_to_account.values())))
    }
    _ACCOUNT_CACHE[root] = (account_to_index, guid_to_account)
    return _ACCOUNT_CACHE[root]


class WeFENDDataset(Dataset):
    def __init__(
        self,
        root: Path,
        csv_file: str,
        image_dir: str,
        limit: Optional[int] = None,
        split: str = "train",
        **_: Dict,
    ) -> None:
        root = Path(root).expanduser().resolve()
        csv_file = (root / csv_file).expanduser().resolve()
        image_dir = (root / image_dir).expanduser().resolve()
        if not csv_file.exists():
            raise FileNotFoundError(f"WeFEND CSV missing: {csv_file}")
        if not image_dir.exists():
            raise FileNotFoundError(f"WeFEND image directory missing: {image_dir}")

        df = pd.read_csv(csv_file)
        samples: List[WeFENDSample] = []
        for _, row in df.iterrows():
            guid = str(row["guid"])
            text = str(row["text"]) if pd.notna(row["text"]) else ""
            if not text.strip():
                text = "No text provided"
            label = int(row["label"])
            digest = hashlib.sha256(guid.encode("utf-8")).hexdigest()
            hashed_path = image_dir / f"{digest}.jpg"
            if hashed_path.exists():
                img_path = hashed_path
            else:
                raw_path = image_dir / f"{guid}.jpg"
                if raw_path.exists():
                    img_path = raw_path
                else:
                    continue
            samples.append(WeFENDSample(guid=guid, text=text, image_path=img_path, label=label))
            if limit is not None and len(samples) >= limit:
                break

        account_to_index, guid_to_account = _load_account_mapping(root)
        for sample in samples:
            account = guid_to_account.get(sample.guid)
            if account is not None:
                sample.event_id = account_to_index.get(account)
        event_counts = Counter(sample.event_id for sample in samples if sample.event_id is not None)
        for sample in samples:
            if sample.event_id is not None:
                sample.event_size = event_counts.get(sample.event_id, 0)
            else:
                sample.event_size = 0

        if not samples:
            raise RuntimeError(f"No valid samples found in {csv_file} using images at {image_dir}.")

        self.samples = samples
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return {
            "id": sample.guid,
            "text": sample.text,
            "image_path": sample.image_path,
            "label": sample.label,
            "event_id": sample.event_id,
             "event_size": sample.event_size,
            "split": self.split,
        }
