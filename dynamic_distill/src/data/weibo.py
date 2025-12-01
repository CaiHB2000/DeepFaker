from __future__ import annotations

import pickle
import random
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
        label_noise: float = 0.0,
        event_noise_rate: float = 0.0,
        noise_seed: int = 1234,
    ) -> None:
        self.root = root
        self.split = split
        self.samples: List[WeiboSample] = []
        self.corrupted_events: List[int] = []
        self.label_noise = float(label_noise)
        self.event_noise_rate = float(event_noise_rate)
        self.noise_seed = int(noise_seed)

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

        # Apply synthetic label noise for analysis experiments (train split only).
        if self.split == "train" and (self.label_noise > 0 or self.event_noise_rate > 0):
            rng = random.Random(self.noise_seed)

            # Event-level corruption: flip labels for a fraction of events.
            if self.event_noise_rate > 0:
                event_ids = [s.event_id for s in self.samples if s.event_id is not None]
                unique_events = sorted(set(event_ids))
                num_to_corrupt = int(len(unique_events) * self.event_noise_rate + 1e-6)
                corrupt_events = set(rng.sample(unique_events, k=num_to_corrupt)) if num_to_corrupt > 0 else set()
                self.corrupted_events = sorted(corrupt_events)
                for sample in self.samples:
                    if sample.event_id in corrupt_events:
                        sample.label = 1 - int(sample.label)

            # Instance-level symmetric noise.
            if self.label_noise > 0:
                for sample in self.samples:
                    if rng.random() < self.label_noise:
                        sample.label = 1 - int(sample.label)

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
