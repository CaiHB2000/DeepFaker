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
class FakedditSample:
    guid: str
    text: str
    image_path: Optional[Path]
    label: int
    event_id: Optional[int] = None
    event_size: int = 0


def hashed_image_path(image_dir: Path, guid: str, ext: str = ".jpg") -> Path:
    digest = hashlib.sha256(guid.encode("utf-8")).hexdigest()
    return image_dir / f"{digest}{ext}"


_SUBREDDIT_CACHE: Dict[Path, Tuple[Dict[str, int], Dict[str, str]]] = {}


def _load_subreddit_mapping(root: Path) -> Tuple[Dict[str, int], Dict[str, str]]:
    if root in _SUBREDDIT_CACHE:
        return _SUBREDDIT_CACHE[root]

    metadata_dir = root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    mapping_path = metadata_dir / "guid_to_subreddit.json"

    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as handle:
            guid_to_subreddit = json.load(handle)
    else:
        guid_to_subreddit: Dict[str, str] = {}
        raw_dir = root / "raw" / "multimodal_only_samples"
        for split in ("multimodal_train.tsv", "multimodal_validate.tsv", "multimodal_test_public.tsv"):
            tsv_path = raw_dir / split
            if not tsv_path.exists():
                continue
            df = pd.read_csv(tsv_path, sep="\t")
            for _, row in df.iterrows():
                guid = str(row["id"])
                subreddit = str(row.get("subreddit", "")).strip()
                if not subreddit:
                    continue
                guid_to_subreddit[guid] = subreddit
        with mapping_path.open("w", encoding="utf-8") as handle:
            json.dump(guid_to_subreddit, handle)

    subreddit_to_index: Dict[str, int] = {
        subreddit: idx for idx, subreddit in enumerate(sorted(set(guid_to_subreddit.values())))
    }

    _SUBREDDIT_CACHE[root] = (subreddit_to_index, guid_to_subreddit)
    return _SUBREDDIT_CACHE[root]


class FakedditDataset(Dataset):
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
            raise FileNotFoundError(f"Fakeddit CSV missing: {csv_file}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Fakeddit image directory missing: {image_dir}")

        df = pd.read_csv(csv_file)
        samples: List[FakedditSample] = []
        for _, row in df.iterrows():
            guid = str(row["guid"])
            text = str(row["text"]) if pd.notna(row["text"]) else ""
            label = int(row["label"])
            img_path = hashed_image_path(image_dir, guid)
            if not img_path.exists():
                continue
            samples.append(FakedditSample(guid=guid, text=text, image_path=img_path, label=label))
            if limit is not None and len(samples) >= limit:
                break

        subreddit_to_index, guid_to_subreddit = _load_subreddit_mapping(root)
        for sample in samples:
            subreddit = guid_to_subreddit.get(sample.guid)
            if subreddit is not None:
                sample.event_id = subreddit_to_index.get(subreddit)
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
