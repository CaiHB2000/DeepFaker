from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .twitter import TwitterMultimodalDataset
from .fakeddit import FakedditDataset
from .wefend import WeFENDDataset
from .weibo import WeiboMultimodalDataset
from .generic import GenericDataset


@dataclass
class DataModule:
    train: DataLoader
    val: Optional[DataLoader]
    test: DataLoader


def build_datasets(
    config: Dict,
    collate_fn,
    generator: Optional["torch.Generator"] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> DataModule:
    data_cfg = config["data"]
    root = Path(data_cfg["root"]).expanduser().resolve()
    dataset_type = data_cfg["type"].lower()

    if dataset_type == "weibo":
        dataset_cls = WeiboMultimodalDataset
    elif dataset_type == "twitter":
        dataset_cls = TwitterMultimodalDataset
    elif dataset_type == "fakeddit":
        dataset_cls = FakedditDataset
    elif dataset_type == "wefend":
        dataset_cls = WeFENDDataset
    elif dataset_type == "generic":
        dataset_cls = GenericDataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    splits: Dict[str, Dataset] = {}

    for split in ("train", "val", "test"):
        split_cfg = data_cfg.get(split)
        if split_cfg is None:
            if split == "val":
                splits[split] = None  # type: ignore
                continue
            raise ValueError(f"Missing configuration for data split '{split}'.")
        splits[split] = dataset_cls(
            root=root,
            split=split,
            **split_cfg,
        )

    batch_size = config["training"]["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)

    train_sampler: Optional[WeightedRandomSampler] = None
    sampler_cfg = data_cfg.get("train_sampler")
    if sampler_cfg:
        sampler_cfg_lower = str(sampler_cfg).lower()
        if sampler_cfg_lower in {"balanced", "class_weighted"}:
            train_dataset = splits["train"]
            if hasattr(train_dataset, "samples"):
                labels = [getattr(sample, "label", None) for sample in getattr(train_dataset, "samples")]
            else:
                labels = []
                for idx in range(len(train_dataset)):
                    sample = train_dataset[idx]
                    labels.append(int(sample["label"]))
            if not labels:
                raise RuntimeError("Unable to derive labels for balanced train sampler.")
            class_counts = Counter(labels)
            weights = [1.0 / class_counts[label] for label in labels]
            train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
        else:
            raise ValueError(f"Unsupported train_sampler option: {sampler_cfg}")

    def make_loader(ds: Dataset, shuffle: bool, sampler: Optional[WeightedRandomSampler] = None) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )

    train_loader = make_loader(splits["train"], shuffle=True, sampler=train_sampler)
    val_dataset = splits["val"]
    val_loader = make_loader(val_dataset, shuffle=False) if val_dataset is not None else None
    test_loader = make_loader(splits["test"], shuffle=False)

    return DataModule(train=train_loader, val=val_loader, test=test_loader)
