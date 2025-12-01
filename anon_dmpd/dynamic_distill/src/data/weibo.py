from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from torch.utils.data import Dataset


@dataclass
class WeiboSample:
    guid: str
    text: str
    image_path: Optional[Path]
    label: int
    event_id: Optional[int] = None


def _parse_weibo_file(path: Path, label: int, image_root: Path) -> Dict[str, WeiboSample]:
    samples: Dict[str, WeiboSample] = {}
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    idx = 0
    total = len(lines)
    while idx < total:
        meta_line = lines[idx].strip()
        idx += 1
        if not meta_line:
            continue

        image_line = lines[idx].strip() if idx < total else ""
        idx += 1
        text_line = lines[idx].strip() if idx < total else ""
        idx += 1

        parts = meta_line.split("|")
        guid = parts[0]

        candidate_paths: List[Path] = []
        for token in image_line.split("|"):
            token = token.strip()
            if not token or token.lower() == "null":
                continue
            filename = token.split("/")[-1]
            candidate = image_root / filename
            if candidate.exists():
                candidate_paths.append(candidate)

        image_path = candidate_paths[0] if candidate_paths else None
        samples[guid] = WeiboSample(
            guid=guid,
            text=text_line,
            image_path=image_path,
            label=label,
        )
    return samples


class WeiboMultimodalDataset(Dataset):
    """Dataset wrapper for the Weibo multimodal fake news corpus."""

    def __init__(
        self,
        root: Path,
        split: str,
        id_file: Optional[str],
        text_sources: Iterable[Dict],
        limit: Optional[int] = None,
    ) -> None:
        self.root = root
        self.split = split
        self.samples: List[WeiboSample] = []

        records: Dict[str, WeiboSample] = {}
        for source in text_sources:
            rel_path = Path(source["path"])
            label = int(source.get("label", 1))
            image_dir = source.get("image_dir")
            if image_dir is None:
                image_dir = "rumor_images" if label == 1 else "nonrumor_images"
            file_path = root / rel_path
            image_root = root / image_dir
            if not file_path.exists():
                raise FileNotFoundError(f"Weibo text file missing: {file_path}")
            records.update(_parse_weibo_file(file_path, label, image_root=image_root))

        if id_file:
            id_path = root / id_file
            with id_path.open("rb") as handle:
                id_mapping = pickle.load(handle)
            for guid, event_id in id_mapping.items():
                sample = records.get(guid)
                if sample is None:
                    continue
                sample.event_id = event_id
                self.samples.append(sample)
        else:
            self.samples = list(records.values())

        if limit is not None:
            self.samples = self.samples[:limit]

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
            "split": self.split,
        }
