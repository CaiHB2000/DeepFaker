from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset


@dataclass
class GenericSample:
    guid: str
    text: str
    image_path: Optional[Path]
    label: int
    event_id: Optional[int] = None
    event_size: int = 0


class GenericDataset(Dataset):
    """
    Minimal CSV-based multimodal dataset.
    Expects columns: image_path, text, label, (optional) event_id.
    image_path is relative to the provided image_dir/root.
    """

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
        csv_path = (root / csv_file).expanduser().resolve()
        image_root = (root / image_dir).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Generic CSV missing: {csv_path}")
        if not image_root.exists():
            raise FileNotFoundError(f"Generic image directory missing: {image_root}")

        df = pd.read_csv(csv_path)
        samples: List[GenericSample] = []
        for _, row in df.iterrows():
            guid = str(row.get("guid", len(samples)))
            text = str(row["text"]) if pd.notna(row["text"]) else ""
            label = int(row["label"])
            rel_path = str(row["image_path"])
            img_path = (image_root / rel_path).resolve()
            if not img_path.exists():
                continue
            event_id = int(row["event_id"]) if "event_id" in row and pd.notna(row["event_id"]) else None
            samples.append(GenericSample(guid=guid, text=text, image_path=img_path, label=label, event_id=event_id))
            if limit is not None and len(samples) >= limit:
                break

        if not samples:
            raise RuntimeError(f"No valid samples found in {csv_path} using images at {image_root}.")

        self.samples = samples
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {
            "id": s.guid,
            "text": s.text,
            "image_path": s.image_path,
            "label": s.label,
            "event_id": s.event_id,
            "event_size": s.event_size,
            "split": self.split,
        }
