#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, get_scheduler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install pyyaml to use the training script: `pip install pyyaml`.") from exc

from dynamic_distill.src.data import build_datasets
from dynamic_distill.src.models import EANNModel, TextEncoder, VisionEncoder
from dynamic_distill.src.utils import expected_calibration_error

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EANN baseline with gradient reversal.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--progress", action="store_true", help="Display tqdm progress bars.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def determine_fallback_image_size(size_cfg: Any) -> int:
    if isinstance(size_cfg, int):
        return size_cfg
    if isinstance(size_cfg, dict):
        for key in ("height", "width", "shortest_edge"):
            if key in size_cfg:
                return int(size_cfg[key])
    return 224


def build_collate_fn(
    tokenizer: AutoTokenizer,
    image_processor: AutoImageProcessor,
    max_length: int,
    fallback_size: int,
):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [sample.get("text", "") or "" for sample in batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        images: List[Image.Image] = []
        event_ids: List[int] = []
        for sample in batch:
            path = sample.get("image_path")
            if path is not None and Path(path).exists():
                try:
                    with Image.open(path) as img:
                        image = img.convert("RGB")
                        width, height = image.size
                        target_w = max(width, 64)
                        target_h = max(height, 64)
                        if target_w > width or target_h > height:
                            padded = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
                            offset = ((target_w - width) // 2, (target_h - height) // 2)
                            padded.paste(image, offset)
                            image = padded
                except (OSError, ValueError):
                    image = Image.new("RGB", (fallback_size, fallback_size), color=(0, 0, 0))
            else:
                image = Image.new("RGB", (fallback_size, fallback_size), color=(0, 0, 0))
            images.append(image)

            event_id = sample.get("event_id")
            if event_id is None:
                event_ids.append(-1)
            else:
                try:
                    event_ids.append(int(event_id))
                except (TypeError, ValueError):
                    event_ids.append(-1)

        vision_inputs = image_processor(images, return_tensors="pt")
        vision_inputs = {"pixel_values": vision_inputs["pixel_values"]}
        for image in images:
            image.close()

        labels = torch.tensor([int(sample["label"]) for sample in batch], dtype=torch.long)
        meta = {
            "ids": [sample.get("id") for sample in batch],
            "event_ids": torch.tensor(event_ids, dtype=torch.long),
        }

        return {
            "text": {k: v for k, v in tokenized.items()},
            "vision": {k: v for k, v in vision_inputs.items()},
            "labels": labels,
            "meta": meta,
        }

    return collate


def gather_event_mapping(dataset) -> Dict[int, int]:
    events = set()
    for sample in getattr(dataset, "samples", []):
        event_id = getattr(sample, "event_id", None)
        if event_id is None:
            continue
        events.add(int(event_id))
    mapping = {event: idx for idx, event in enumerate(sorted(events))}
    return mapping


@torch.no_grad()
def evaluate(
    model: EANNModel,
    loader: Optional[DataLoader],
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if loader is None:
        return {}, []

    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    correct = 0
    num_classes = model.num_classes
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    records: List[Dict[str, Any]] = []

    model.eval()
    for batch in loader:
        text = {k: v.to(device) for k, v in batch["text"].items()}
        vision = {k: v.to(device) for k, v in batch["vision"].items()}
        labels = batch["labels"].to(device)

        outputs = model(text_batch=text, vision_batch=vision)
        logits = outputs.logits
        loss = ce(logits, labels)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item()
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        for label, pred in zip(labels.view(-1), preds.view(-1)):
            confusion[label.long(), pred.long()] += 1

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

        confidences, _ = probs.max(dim=-1)
        ids = batch.get("meta", {}).get("ids", [None] * labels.size(0))
        for idx in range(labels.size(0)):
            records.append(
                {
                    "id": ids[idx] if idx < len(ids) else None,
                    "label": int(labels[idx].item()),
                    "prediction": int(preds[idx].item()),
                    "confidence": float(confidences[idx].item()),
                }
            )

    probs_cat = torch.cat(all_probs, dim=0) if all_probs else torch.zeros(total, num_classes)
    labels_cat = torch.cat(all_labels, dim=0) if all_labels else torch.zeros(total, dtype=torch.long)

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for entry in records:
        confusion[entry["label"], entry["prediction"]] += 1

    confusion_float = confusion.float()
    tp = torch.diag(confusion_float)
    precision = tp / confusion_float.sum(dim=0).clamp_min(1.0)
    recall = tp / confusion_float.sum(dim=1).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    f1_macro = f1.mean().item()
    pos_f1 = f1[1].item() if num_classes > 1 else f1_macro

    ece = expected_calibration_error(probs_cat, labels_cat) if total > 0 else 0.0

    metrics = {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "f1_macro": f1_macro,
        "f1_pos": pos_f1,
        "ece": ece,
    }
    model.train()
    return metrics, records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer_cfg = config["tokenizer"]
    image_cfg = config["vision_processor"]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_cfg["name"],
        local_files_only=tokenizer_cfg.get("local_files_only", False),
    )
    image_processor = AutoImageProcessor.from_pretrained(
        image_cfg["name"],
        local_files_only=image_cfg.get("local_files_only", False),
    )

    fallback_size = determine_fallback_image_size(image_cfg.get("image_size", 224))
    collate_fn = build_collate_fn(
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=tokenizer_cfg.get("max_length", 128),
        fallback_size=fallback_size,
    )

    generator = None
    worker_init_fn = None
    if args.seed is not None:
        generator = torch.Generator().manual_seed(args.seed)

        def seed_worker(worker_id: int) -> None:
            worker_seed = args.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        worker_init_fn = seed_worker

    data_module = build_datasets(
        config,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    train_dataset = getattr(data_module.train, "dataset", None)
    event_mapping = gather_event_mapping(train_dataset) if train_dataset is not None else {}

    model_cfg = config["model"]
    text_encoder = TextEncoder(
        model_name=model_cfg.get("text_model"),
        projection_dim=model_cfg.get("encoder_dim", 768),
        local_files_only=model_cfg.get("local_files_only", False),
    )
    vision_encoder = VisionEncoder(
        model_name=model_cfg.get("vision_model"),
        projection_dim=model_cfg.get("encoder_dim", 768),
        local_files_only=model_cfg.get("local_files_only", False),
    )

    model = EANNModel(
        num_classes=model_cfg.get("num_classes", 2),
        num_domains=len(event_mapping),
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        encoder_dim=model_cfg.get("encoder_dim", 768),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        dropout=model_cfg.get("dropout", 0.1),
        grl_lambda=model_cfg.get("grl_lambda", 1.0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if config["training"].get("allow_data_parallel", False) and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optim"]["lr"],
        weight_decay=config["optim"].get("weight_decay", 0.01),
    )

    scheduler_cfg = config.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type")
    scheduler = None
    if scheduler_type:
        try:
            steps_per_epoch = len(data_module.train)
        except TypeError:
            steps_per_epoch = None
        if scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_cfg.get("mode", "max"),
                factor=scheduler_cfg.get("factor", 0.5),
                patience=scheduler_cfg.get("patience", 2),
                threshold=scheduler_cfg.get("threshold", 1e-4),
                threshold_mode=scheduler_cfg.get("threshold_mode", "rel"),
                cooldown=scheduler_cfg.get("cooldown", 0),
                min_lr=scheduler_cfg.get("min_lr", 0.0),
            )
        elif steps_per_epoch is not None:
            total_steps = steps_per_epoch * config["training"]["epochs"]
            scheduler = get_scheduler(
                scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=scheduler_cfg.get("warmup_steps", 0),
                num_training_steps=total_steps,
            )

    logging_cfg = config.get("logging", {})
    output_root = Path(logging_cfg.get("output_dir", "runs"))
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = logging_cfg.get("run_name")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not run_name:
        run_name = f"{config['data']['type']}_eann_{timestamp}"
    if args.seed is not None:
        run_name = f"{run_name}_seed{args.seed:02d}"
    run_dir = output_root / run_name
    if run_dir.exists() and not logging_cfg.get("overwrite", False):
        run_dir = output_root / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] logging to {run_dir}")
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as cfg_out:
        yaml.safe_dump(config, cfg_out)

    ce = torch.nn.CrossEntropyLoss()
    lambda_domain = config["loss"].get("lambda_domain", 0.1)

    def map_event_ids(raw_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if not event_mapping:
            return None
        mapped = []
        for eid in raw_ids.tolist():
            mapped.append(event_mapping.get(eid, -1))
        mapped_tensor = torch.tensor(mapped, dtype=torch.long, device=device)
        mask = mapped_tensor >= 0
        if mask.any():
            return mapped_tensor, mask
        return None

    num_epochs = config["training"]["epochs"]
    log_every = max(1, config["training"].get("log_every", 50))
    patience = config["training"].get("early_stopping_patience", 3)
    best_val_f1 = -float("inf")
    best_epoch = -1
    best_checkpoint = run_dir / "model_best.pt"
    history_train: List[Dict[str, float]] = []
    history_val: List[Dict[str, float]] = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_domain_loss = 0.0
        total_batches = 0
        total_samples = 0

        train_loader = data_module.train
        train_iterable: Iterable = train_loader
        if args.progress:
            try:
                total_batches = len(train_loader)
            except TypeError:
                total_batches = None
            train_iterable = tqdm(
                train_loader,
                total=total_batches,
                desc=f"epoch {epoch + 1}",
                leave=False,
                dynamic_ncols=False,
                mininterval=0.5,
            )

        for batch_idx, batch in enumerate(train_iterable, start=1):
            text = {k: v.to(device) for k, v in batch["text"].items()}
            vision = {k: v.to(device) for k, v in batch["vision"].items()}
            labels = batch["labels"].to(device)
            event_ids = batch["meta"].get("event_ids").to(device)

            outputs = model(text_batch=text, vision_batch=vision)
            logits = outputs.logits

            loss_cls = ce(logits, labels)
            loss_domain = torch.tensor(0.0, device=device)
            if outputs.domain_logits is not None and lambda_domain > 0:
                mapped = map_event_ids(event_ids)
                if mapped is not None:
                    mapped_ids, mask = mapped
                    if mask.any():
                        domain_logits = outputs.domain_logits[mask]
                        loss_domain = ce(domain_logits, mapped_ids[mask])

            loss = loss_cls + lambda_domain * loss_domain

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_type != "reduce_on_plateau":
                scheduler.step()

            epoch_loss += loss.item() * labels.size(0)
            epoch_cls_loss += loss_cls.item() * labels.size(0)
            epoch_domain_loss += loss_domain.item() * labels.size(0)
            total_samples += labels.size(0)
            total_batches += 1

            if batch_idx % log_every == 0:
                print(
                    f"[epoch {epoch+1}/{num_epochs}] step={batch_idx} "
                    f"loss={loss.item():.4f} cls={loss_cls.item():.4f} dom={loss_domain.item():.4f}",
                    flush=True,
                )

        train_metrics = {
            "epoch": epoch + 1,
            "loss_total": epoch_loss / max(total_samples, 1),
            "loss_cls": epoch_cls_loss / max(total_samples, 1),
            "loss_domain": epoch_domain_loss / max(total_samples, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_train.append(train_metrics)

        val_metrics, _ = evaluate(model, data_module.val, device)
        history_val.append({"epoch": epoch + 1, **val_metrics})
        print(
            f"[epoch {epoch+1}] val_acc={val_metrics.get('acc', 0):.4f} "
            f"val_f1={val_metrics.get('f1_macro', 0):.4f} val_ece={val_metrics.get('ece', 0):.4f}",
            flush=True,
        )

        if scheduler is not None and scheduler_type == "reduce_on_plateau":
            scheduler.step(val_metrics.get("f1_macro", 0.0))

        if val_metrics.get("f1_macro", -float("inf")) > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), best_checkpoint)

        if patience is not None and epoch - best_epoch >= patience:
            print("[info] Early stopping triggered.")
            break

    torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), run_dir / "model_last.pt")

    # Load best model for evaluation
    if best_checkpoint.exists():
        state = torch.load(best_checkpoint, map_location=device)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)

    test_metrics, test_records = evaluate(model, data_module.test, device)
    print(
        f"[test] acc={test_metrics.get('acc', 0):.4f} "
        f"f1={test_metrics.get('f1_macro', 0):.4f} ece={test_metrics.get('ece', 0):.4f}",
        flush=True,
    )

    # Persist metrics
    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        keys = rows[0].keys()
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)  # type: ignore[name-defined]
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    import csv  # delayed import to satisfy static analysers

    write_csv(run_dir / "train_metrics.csv", history_train)
    write_csv(run_dir / "val_metrics.csv", history_val)
    if test_records:
        keys = test_records[0].keys()
        with (run_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in test_records:
                writer.writerow(row)

    summary = {
        "config_path": str(args.config),
        "seed": args.seed,
        "train_history": history_train,
        "val_history": history_val,
        "test_metrics": test_metrics,
        "best_epoch": best_epoch + 1,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
