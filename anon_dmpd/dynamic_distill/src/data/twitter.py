from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from torch.utils.data import Dataset


LABEL_MAP = {"fake": 1, "real": 0}


@dataclass
class TwitterSample:
    guid: str
    text: str
    image_id: Optional[str]
    image_path: Optional[Path]
    label: int


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # Collapse consecutive whitespace to reduce spurious differences
    return " ".join(text.split())


def _build_content_groups(df: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """Connect samples that share text or image content to prevent leakage."""

    n_rows = len(df)
    parent = list(range(n_rows))
    rank = [0] * n_rows

    def find_root(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        root_a = find_root(a)
        root_b = find_root(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    text_to_indices: Dict[str, List[int]] = defaultdict(list)
    image_to_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, row in df.iterrows():
        text_key = _normalize_text(row.get("post_text", ""))
        image_value = row.get("image_id")
        image_key = image_value.strip() if isinstance(image_value, str) else ""

        if text_key:
            text_to_indices[text_key].append(idx)
        if image_key:
            image_to_indices[image_key].append(idx)

    for indices in text_to_indices.values():
        anchor = indices[0]
        for other in indices[1:]:
            union(anchor, other)

    for indices in image_to_indices.values():
        anchor = indices[0]
        for other in indices[1:]:
            union(anchor, other)

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n_rows):
        groups[find_root(idx)].append(idx)

    label_by_group: Dict[int, int] = {}
    for group_id, members in groups.items():
        labels = df.loc[members, "label_id"].unique()
        if len(labels) != 1:
            raise ValueError("Mixed labels detected within a Twitter content group; check preprocessing.")
        label_by_group[group_id] = int(labels[0])

    return groups, label_by_group


def _select_validation_groups(
    groups: Dict[int, List[int]],
    label_by_group: Dict[int, int],
    label_counts: Dict[int, int],
    val_fraction: float,
    seed: int,
) -> Set[int]:
    target_counts = {label: int(round(count * val_fraction)) for label, count in label_counts.items()}
    rng = random.Random(seed)
    group_sizes = {group_id: len(members) for group_id, members in groups.items()}

    val_group_ids: Set[int] = set()

    for label in sorted(target_counts):
        desired = target_counts[label]
        if desired <= 0:
            continue
        label_groups: List[Tuple[int, int]] = [
            (group_id, group_sizes[group_id])
            for group_id, group_label in label_by_group.items()
            if group_label == label
        ]
        if not label_groups:
            continue

        rng.shuffle(label_groups)
        selected: List[Tuple[int, int]] = []
        selected_ids: Set[int] = set()
        current = 0

        for group_id, size in label_groups:
            if current >= desired:
                break
            selected.append((group_id, size))
            selected_ids.add(group_id)
            current += size

        remaining = [(group_id, size) for group_id, size in label_groups if group_id not in selected_ids]

        improved = True
        while improved and selected:
            improved = False
            for idx, (group_id, size) in enumerate(list(selected)):
                alternative = current - size
                if abs(alternative - desired) < abs(current - desired):
                    selected.pop(idx)
                    selected_ids.remove(group_id)
                    current = alternative
                    improved = True
                    break

        for group_id, size in remaining:
            alternative = current + size
            if abs(alternative - desired) < abs(current - desired):
                selected.append((group_id, size))
                selected_ids.add(group_id)
                current = alternative

        if not selected_ids:
            group_id, _ = label_groups[0]
            selected_ids.add(group_id)

        val_group_ids.update(selected_ids)

    return val_group_ids


class TwitterMultimodalDataset(Dataset):
    """Dataset wrapper for the Twitter (MediaEval) multimodal fake news corpus."""

    _DEFAULT_TEXT = "No text provided"
    _MIN_WORDS = 5

    def __init__(
        self,
        root: Path,
        split: str,
        csv_file: str,
        image_dir: str,
        val_fraction: float = 0.1,
        seed: int = 42,
        limit: Optional[int] = None,
    ) -> None:
        self.root = root
        self.split = split
        csv_path = root / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Twitter CSV missing: {csv_path}")
        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            raise ValueError(f"Twitter CSV {csv_path} lacks 'label' column.")

        df = df[df["label"].isin(LABEL_MAP)].copy().reset_index(drop=True)
        df["label_id"] = df["label"].map(LABEL_MAP)

        if split in {"train", "val"} and val_fraction > 0:
            groups, label_by_group = _build_content_groups(df)
            val_group_ids = _select_validation_groups(
                groups=groups,
                label_by_group=label_by_group,
                label_counts=df["label_id"].value_counts().to_dict(),
                val_fraction=val_fraction,
                seed=seed,
            )
            if split == "val":
                selected_indices = sorted(idx for group_id in val_group_ids for idx in groups[group_id])
            else:
                selected_indices = sorted(
                    idx for group_id, members in groups.items() if group_id not in val_group_ids for idx in members
                )
            df_split = df.iloc[selected_indices].reset_index(drop=True)
        else:
            df_split = df

        image_root = root / image_dir
        samples: List[TwitterSample] = []
        for _, row in df_split.iterrows():
            guid = str(row["post_id"])
            image_id = row.get("image_id")
            image_path = None
            if isinstance(image_id, str):
                candidate = image_root / f"{image_id}.jpg"
                if not candidate.exists():
                    candidate = image_root / image_id
                if candidate.exists():
                    image_path = candidate
            samples.append(
                TwitterSample(
                    guid=guid,
                    text=row.get("post_text", ""),
                    image_id=image_id if isinstance(image_id, str) else None,
                    image_path=image_path,
                    label=int(row["label_id"]),
                )
            )

        if limit is not None:
            samples = samples[:limit]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        text = sample.text if isinstance(sample.text, str) else ""
        if not text or self._word_count(text) < self._MIN_WORDS:
            text = self._DEFAULT_TEXT
        return {
            "id": sample.guid,
            "text": text,
            "image_path": sample.image_path,
            "label": sample.label,
            "image_id": sample.image_id,
        }

    @staticmethod
    def _word_count(text: str) -> int:
        return len([token for token in text.strip().split() if token])
