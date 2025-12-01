#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import warnings
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image, ImageEnhance, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import IterableDataset
from transformers import AutoImageProcessor, AutoTokenizer, get_scheduler

try:
    import yaml
except ImportError as exc:  # pragma: no cover - optional dependency.
    raise ImportError("Install pyyaml to use the training script: `pip install pyyaml`.") from exc

from dynamic_distill.src.data import build_datasets
from dynamic_distill.src.models import (
    DynamicModalDistillationModel,
    TextEncoder,
    VisionEncoder,
)
from dynamic_distill.src.models.encoders import EncoderOutput
from dynamic_distill.src.training.trainer import DynamicDistillationTrainer, TrainerConfig
from dynamic_distill.src.utils import expected_calibration_error
from dynamic_distill.src.losses import compute_dynamic_distillation
import inspect
import hashlib


class SyntheticMultimodalStream(IterableDataset):
    """Produces synthetic batches for smoke-testing the training loop."""

    def __init__(self, seq_len: int, image_size: int, batch_size: int, num_classes: int, steps: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.steps = steps

    def __iter__(self):
        for _ in range(self.steps):
            text_inputs = {
                "input_ids": torch.randint(0, 1000, (self.batch_size, self.seq_len)),
                "attention_mask": torch.ones(self.batch_size, self.seq_len),
            }
            vision_inputs = {
                "pixel_values": torch.randn(self.batch_size, 3, self.image_size, self.image_size),
            }
            labels = torch.randint(0, self.num_classes, (self.batch_size,))
            yield {
                "text": text_inputs,
                "vision": vision_inputs,
                "labels": labels,
            }


def build_synthetic_loader(config: Dict) -> torch.utils.data.DataLoader:
    data_cfg = config["data"]["synthetic"]
    dataset = SyntheticMultimodalStream(
        seq_len=data_cfg.get("seq_len", 128),
        image_size=data_cfg.get("image_size", 224),
        batch_size=config["training"]["batch_size"],
        num_classes=config["model"]["num_classes"],
        steps=data_cfg.get("steps_per_epoch", 10),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=None)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dynamic modal priority distillation MVP.")
    parser.add_argument("--config", type=Path, default=Path("dynamic_distill/configs/default_mvp.yaml"))
    parser.add_argument("--synthetic", action="store_true", help="Run a synthetic smoke test regardless of config.")
    parser.add_argument("--disable-distill", action="store_true", help="Turn off dynamic distillation (baseline).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override logging output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--progress", action="store_true", help="Display per-epoch tqdm progress bar.")
    parser.add_argument(
        "--trace-loader",
        action="store_true",
        help="Print detailed timing for data loading (first few batches each epoch).",
    )
    return parser.parse_args()


def determine_fallback_image_size(size_cfg: Any) -> int:
    if isinstance(size_cfg, int):
        return size_cfg
    if isinstance(size_cfg, dict):
        for key in ("height", "width", "shortest_edge"):
            if key in size_cfg:
                return int(size_cfg[key])
    return 224


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_collate_fn(
    tokenizer: AutoTokenizer,
    image_processor: Optional[AutoImageProcessor],
    max_length: int,
    fallback_size: int,
    augment_cfg: Optional[Dict[str, Any]] = None,
    text_dropout: float = 0.0,
    torchvision_resnet: bool = False,
):
    augment_cfg = augment_cfg or {}
    augment_enabled = bool(augment_cfg.get("enabled", augment_cfg))
    flip_prob = float(augment_cfg.get("flip_prob", 0.0))
    color_jitter = float(augment_cfg.get("color_jitter", 0.0))
    random_rescale = float(augment_cfg.get("random_rescale", 0.0))

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        is_training = bool(batch and batch[0].get("split") == "train")
        texts: List[str] = []
        for sample in batch:
            text = sample.get("text", "") or ""
            if is_training and text_dropout > 0:
                tokens = text.split()
                if tokens:
                    kept = [tok for tok in tokens if random.random() > text_dropout]
                    if not kept:
                        kept = tokens[:1]
                    text = " ".join(kept)
            texts.append(text)

        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        vision_inputs = {}
        event_ids: List[int] = []
        event_sizes: List[int] = []
        if image_processor is not None or torchvision_resnet:
            images: List[Image.Image] = []
            for sample in batch:
                path = sample.get("image_path")
                image: Image.Image
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

                if augment_enabled and is_training:
                    if flip_prob > 0 and random.random() < flip_prob:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    if random_rescale > 0:
                        width, height = image.size
                        scale = random.uniform(1.0 - random_rescale, 1.0)
                        crop_w = max(int(width * scale), 1)
                        crop_h = max(int(height * scale), 1)
                        if crop_w < width or crop_h < height:
                            left = random.randint(0, width - crop_w)
                            upper = random.randint(0, height - crop_h)
                            image = image.crop((left, upper, left + crop_w, upper + crop_h)).resize(
                                (width, height), Image.BILINEAR
                            )
                    if color_jitter > 0:
                        factor = random.uniform(1.0 - color_jitter, 1.0 + color_jitter)
                        image = ImageEnhance.Brightness(image).enhance(factor)

                images.append(image)
                event_id = sample.get("event_id")
                event_size = sample.get("event_size", 0)
                if event_id is None:
                    event_ids.append(-1)
                else:
                    try:
                        event_ids.append(int(event_id))
                    except (TypeError, ValueError):
                        event_ids.append(-1)
                try:
                    event_sizes.append(int(event_size))
                except (TypeError, ValueError):
                    event_sizes.append(0)

            if image_processor is not None:
                vision_inputs = image_processor(images, return_tensors="pt")
            else:
                # torchvision resnet expects (B,3,H,W) float tensor [0,1]
                import torchvision.transforms as T

                trans = T.Compose(
                    [
                        T.Resize((fallback_size, fallback_size)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )
                tensor_imgs = torch.stack([trans(img) for img in images])
                vision_inputs = {"pixel_values": tensor_imgs}
            vision_inputs = {"pixel_values": vision_inputs["pixel_values"]}
            for image in images:
                image.close()
        else:
            # text-only mode: fill placeholders
            event_ids = [int(sample.get("event_id") or -1) for sample in batch]
            event_sizes = [int(sample.get("event_size") or 0) for sample in batch]
            # create zero images to carry batch dimension
            bsz = len(batch)
            vision_inputs = {"pixel_values": torch.zeros((bsz, 3, fallback_size, fallback_size))}

        labels = torch.tensor([int(sample["label"]) for sample in batch], dtype=torch.long)
        meta = {
            "ids": [sample.get("id") for sample in batch],
            "event_ids": torch.tensor(event_ids, dtype=torch.long),
            "event_sizes": torch.tensor(event_sizes, dtype=torch.long),
        }

        return {
            "text": {k: v for k, v in tokenized.items()},
            "vision": {k: v for k, v in vision_inputs.items()},
            "labels": labels,
            "meta": meta,
        }

    return collate


@torch.no_grad()
def evaluate(
    trainer: DynamicDistillationTrainer,
    loader: torch.utils.data.DataLoader,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    model = trainer.model
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    correct = 0
    num_classes = trainer.num_classes
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    records: List[Dict[str, Any]] = []

    model.eval()
    for batch in loader:
        moved = trainer._move_batch(batch)  # type: ignore[attr-defined]
        labels = moved["labels"]
        outputs = model(text_batch=moved["text"], vision_batch=moved["vision"])
        logits = outputs["fusion_logits"]
        total_loss += ce(logits, labels).item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            for p, l in zip(preds.view(-1), labels.view(-1)):
                confusion[l.long(), p.long()] += 1

            confidences, _ = probs.max(dim=-1)
            ids = moved.get("meta", {}).get("ids", [None] * labels.size(0))
            for idx in range(labels.size(0)):
                record = {
                    "id": ids[idx] if idx < len(ids) else None,
                    "label": int(labels[idx].item()),
                    "prediction": int(preds[idx].item()),
                    "confidence": float(confidences[idx].item()),
                }
                # add per-class probabilities for reliability plots
                for k in range(num_classes):
                    record[f"prob_{k}"] = float(probs[idx, k].item())
                record["correct"] = int(record["label"] == record["prediction"])
                records.append(record)

    model.train()
    total = max(total, 1)
    probs_cat = torch.cat(all_probs, dim=0) if all_probs else torch.zeros(total, num_classes)
    labels_cat = torch.cat(all_labels, dim=0) if all_labels else torch.zeros(total, dtype=torch.long)
    # Manual metrics
    confusion_float = confusion.float()
    tp = torch.diag(confusion_float)
    precision = tp / confusion_float.sum(dim=0).clamp_min(1.0)
    recall = tp / confusion_float.sum(dim=1).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    f1_macro = f1.mean().item()
    pos_f1 = f1[1].item() if num_classes > 1 else f1_macro

    ece = expected_calibration_error(probs_cat, labels_cat) if total > 0 else 0.0

    metrics = {
        "loss": total_loss / total,
        "acc": correct / total,
        "f1_macro": f1_macro,
        "f1_pos": pos_f1,
        "ece": ece,
    }
    return metrics, records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.disable_distill:
        config.setdefault("loss", {})["gamma"] = 0.0
        print("[info] Distillation disabled (baseline mode).")

    logging_cfg = config.get("logging", {})
    output_root = Path(args.output_dir) if args.output_dir else Path(logging_cfg.get("output_dir", "runs"))
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = logging_cfg.get("run_name")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not run_name:
        run_name = f"{config['data']['type']}_{timestamp}"
    if args.seed is not None:
        run_name = f"{run_name}_seed{args.seed:02d}"
    run_dir = output_root / run_name
    if run_dir.exists() and not logging_cfg.get("overwrite", False):
        run_dir = output_root / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] logging to {run_dir}")
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as cfg_out:
        yaml.safe_dump(config, cfg_out)

    generator = None
    worker_init_fn = None
    if args.seed is not None:
        set_seed_everywhere(args.seed)
        generator = torch.Generator()
        generator.manual_seed(args.seed)

        def seed_worker(worker_id: int) -> None:
            worker_seed = args.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        worker_init_fn = seed_worker

    if args.synthetic or config["data"]["type"].lower() == "synthetic":
        loader = build_synthetic_loader(config)
        val_loader = None
        test_loader = None
        tokenizer = None
        image_processor = None
    else:
        tokenizer_cfg = config["tokenizer"]
        image_cfg = config["vision_processor"]

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_cfg["name"],
            local_files_only=tokenizer_cfg.get("local_files_only", False),
        )
        image_processor = None
        if config["model"].get("vision_model") not in (None, "", "none", "null"):
            vis_name = image_cfg["name"]
            if str(vis_name).lower() not in {
                "resnet50",
                "resnet-50",
                "torchvision/resnet50",
                "resnet101",
                "resnet-101",
                "torchvision/resnet101",
                "microsoft/resnet-101",
            }:
                image_processor = AutoImageProcessor.from_pretrained(
                    vis_name,
                    local_files_only=image_cfg.get("local_files_only", False),
                )

        fallback_size = determine_fallback_image_size(image_cfg.get("image_size", 224))
        augment_cfg = image_cfg.get("augment")
        if isinstance(augment_cfg, bool):
            augment_cfg = {"enabled": augment_cfg}
        text_dropout = float(tokenizer_cfg.get("word_dropout", 0.0))
        collate_fn = build_collate_fn(
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=tokenizer_cfg.get("max_length", 128),
            fallback_size=fallback_size,
            augment_cfg=augment_cfg,
            text_dropout=text_dropout,
            torchvision_resnet=image_processor is None
            and str(image_cfg["name"]).lower()
            in {
                "resnet50",
                "resnet-50",
                "torchvision/resnet50",
                "resnet101",
                "resnet-101",
                "torchvision/resnet101",
                "microsoft/resnet-101",
            },
        )

        data_module = build_datasets(
            config,
            collate_fn=collate_fn,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
        loader = data_module.train
        val_loader = data_module.val
        test_loader = data_module.test
        if loader is not None:
            train_size = None
            try:
                train_size = len(loader.dataset)  # type: ignore[attr-defined]
            except Exception:
                train_size = None
            if train_size is not None:
                print(f"[info] train dataset size: {train_size}")
            try:
                est_batches = len(loader)  # type: ignore[arg-type]
                print(f"[info] estimated train batches per epoch: {est_batches}")
            except TypeError:
                print("[info] train loader has no static length (iterable dataset).")

    # log distillation function signature for debugging reproducibility
    try:
        distill_src = inspect.getsource(compute_dynamic_distillation)
        digest = hashlib.sha1(distill_src.encode("utf-8")).hexdigest()
        print(
            "[debug] compute_dynamic_distillation "
            f"path={inspect.getfile(compute_dynamic_distillation)} sha1={digest[:12]} len={len(distill_src)}"
        )
    except (OSError, TypeError):
        print("[debug] compute_dynamic_distillation source unavailable", flush=True)

    text_model_name = config["model"].get("text_model")
    vision_model_name = config["model"].get("vision_model")
    vision_disabled = vision_model_name in (None, "", "none", "null")
    text_disabled = text_model_name in (None, "", "none", "null")

    if text_disabled:
        class ZeroTextEncoder(torch.nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.dim = dim
                self.register_buffer("zero", torch.zeros(1))

            def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
                batch = input_ids.shape[0] if input_ids is not None else (attention_mask.shape[0] if attention_mask is not None else 1)
                device = None
                for t in (input_ids, attention_mask, token_type_ids):
                    if t is not None:
                        device = t.device
                        break
                if device is None:
                    device = self.zero.device
                zeros = torch.zeros((batch, self.dim), device=device)
                return EncoderOutput(representation=zeros, sequence=None, extras={})

        text_encoder = ZeroTextEncoder(config["model"]["encoder_dim"])
    else:
        text_encoder = TextEncoder(
            model_name=text_model_name,
            projection_dim=config["model"]["encoder_dim"],
            local_files_only=config["model"].get("local_files_only", False),
        )
    if vision_disabled:
        class ZeroVisionEncoder(torch.nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.dim = dim
                self.register_buffer("zero", torch.zeros(1))

            def forward(self, pixel_values=None):
                batch = 1 if pixel_values is None else pixel_values.shape[0]
                device = pixel_values.device if pixel_values is not None else self.zero.device
                zeros = torch.zeros((batch, self.dim), device=device)
                return EncoderOutput(representation=zeros, sequence=None, extras={})

        vision_encoder = ZeroVisionEncoder(config["model"]["encoder_dim"])
    else:
        vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            projection_dim=config["model"]["encoder_dim"],
            local_files_only=config["model"].get("local_files_only", False),
        )

    model = DynamicModalDistillationModel(
        num_classes=config["model"]["num_classes"],
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        encoder_dim=config["model"]["encoder_dim"],
        classifier_hidden=config["model"].get("classifier_hidden"),
        dropout=config["model"].get("dropout", 0.1),
    )
    init_checkpoint = config["model"].get("init_checkpoint")
    if init_checkpoint:
        state = torch.load(Path(init_checkpoint), map_location="cpu")
        model.load_state_dict(state, strict=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optim"]["lr"],
        weight_decay=config["optim"].get("weight_decay", 0.01),
    )

    scheduler_cfg = config.get("scheduler", {})
    scheduler = None
    scheduler_type = scheduler_cfg.get("type")
    epochs = config["training"]["epochs"]
    max_steps = config["training"].get("max_steps_per_epoch")
    steps_per_epoch: Optional[int] = None
    try:
        steps_per_epoch = len(loader)  # type: ignore[arg-type]
    except TypeError:
        steps_per_epoch = None
    if max_steps is not None:
        steps_per_epoch = max_steps if steps_per_epoch is None else min(steps_per_epoch, max_steps)
    if steps_per_epoch is None:
        steps_per_epoch = scheduler_cfg.get("steps_per_epoch")
    if scheduler_type:
        scheduler_type_lower = scheduler_type.lower()
    else:
        scheduler_type_lower = None

    if scheduler_type_lower == "reduce_on_plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "max"),
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 3),
            threshold=scheduler_cfg.get("threshold", 1e-4),
            threshold_mode=scheduler_cfg.get("threshold_mode", "rel"),
            cooldown=scheduler_cfg.get("cooldown", 0),
            min_lr=scheduler_cfg.get("min_lr", 0.0),
            eps=scheduler_cfg.get("eps", 1e-08),
        )
    elif scheduler_type and steps_per_epoch is not None:
        total_steps = steps_per_epoch * epochs
        warmup_steps = scheduler_cfg.get("warmup_steps", 0)
        scheduler = get_scheduler(
            scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    elif scheduler_type:
        warnings.warn("Unable to determine steps_per_epoch; scheduler disabled.")

    teacher_cfg = config.get("teacher", {})
    teacher_config_path = teacher_cfg.get("config") or teacher_cfg.get("teacher_config")
    teacher_checkpoint_path = teacher_cfg.get("checkpoint") or teacher_cfg.get("teacher_checkpoint")
    class_weights_cfg = config.get("loss", {}).get("class_weights")
    class_weights_tensor = None
    if class_weights_cfg is not None:
        class_weights_tensor = torch.tensor(class_weights_cfg, dtype=torch.float32)
    distill_cfg = config.get("distillation", {})
    event_filter_cfg = distill_cfg.get("event_filter", {})
    positive_event_cfg = distill_cfg.get("positive_event_gate", {})
    uncertainty_weight_cfg = distill_cfg.get("uncertainty_weight", {})
    delta_schedule_cfg = distill_cfg.get("delta_schedule", {})
    student_focus_cfg = distill_cfg.get("student_focus", {})
    event_layer_cfg = distill_cfg.get("event_layer", {})
    consistency_cfg = distill_cfg.get("consistency", {})
    weak_teacher_cfg = distill_cfg.get("weak_teacher", {})
    weak_teacher_dyn_cfg = weak_teacher_cfg.get("dynamic_scale", {})
    event_reweight_cfg = config.get("loss", {}).get("event_reweight", {})
    fallback_cfg = distill_cfg.get("fallback", {})
    soft_quota_cfg = distill_cfg.get("soft_quota", {})
    event_calib_cfg = distill_cfg.get("event_calibration", {})
    consensus_gate_cfg = distill_cfg.get("consensus_gate", {})
    force_distill_cfg = distill_cfg.get("force_distill", {})
    reliability_cfg = config.get("reliability", {})

    trainer = DynamicDistillationTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=TrainerConfig(
            alpha=config.get("loss", {}).get("alpha", 1.0),
            beta=config.get("loss", {}).get("beta", 0.0),
            gamma=config.get("loss", {}).get("gamma", 0.0),
            temperature=distill_cfg.get("temperature", 2.0),
            lambda_feat=distill_cfg.get("lambda_feat", 0.0),
            lambda_kl=distill_cfg.get("lambda_kl", 1.0),
            delta=distill_cfg.get("delta", 0.05),
            evidence_anneal_steps=config.get("loss", {}).get("evidence_anneal_steps", 0),
            distill_warmup_steps=distill_cfg.get("warmup_steps", 0),
            use_ema_teacher=teacher_cfg.get("use_ema", False),
            ema_decay=teacher_cfg.get("ema_decay", 0.999),
            teacher_checkpoint=teacher_checkpoint_path,
            teacher_config=teacher_config_path,
            distill_start_fraction=distill_cfg.get("start_fraction", 0.0),
            distill_end_fraction=distill_cfg.get("end_fraction", 1.0),
            adaptive_temperature=distill_cfg.get("adaptive_temperature", {}).get("enabled", False),
            temperature_base=distill_cfg.get("adaptive_temperature", {}).get(
                "base", distill_cfg.get("temperature", 2.0)
            ),
            temperature_coeff=distill_cfg.get("adaptive_temperature", {}).get("coeff", 0.0),
            use_confidence_gate=distill_cfg.get("confidence_gate", {}).get("enabled", False),
            confidence_margin=distill_cfg.get("confidence_gate", {}).get("margin", 0.0),
            use_uncertainty_ema=distill_cfg.get("uncertainty_ema", {}).get("enabled", False),
            uncertainty_ema=distill_cfg.get("uncertainty_ema", {}).get("momentum", 0.9),
            trace_batches=5 if args.trace_loader else 0,
            allow_data_parallel=config["training"].get("allow_data_parallel", False),
            lambda_fusion_to_text=distill_cfg.get("lambda_fusion_to_text", 0.0),
            lambda_fusion_to_vision=distill_cfg.get("lambda_fusion_to_vision", 0.0),
            fusion_confidence_margin=distill_cfg.get("fusion_confidence", 0.0),
            scheduler_type=scheduler_type_lower,
            class_weights=class_weights_tensor,
            min_event_size_for_distill=event_filter_cfg.get("min_size", 0),
            event_filter_min_teacher_acc=event_filter_cfg.get("teacher_min_acc", 0.0),
            event_filter_min_teacher_conf=event_filter_cfg.get("teacher_min_conf", 0.0),
            event_filter_warmup_steps=event_filter_cfg.get("warmup_steps", 0),
            distill_require_student_mistake=distill_cfg.get("require_student_mistake", True),
            agreement_confidence_gap=distill_cfg.get("agreement_confidence_gap", 0.0),
            positive_distill_boost=distill_cfg.get("positive_distill_boost", 1.0),
            positive_student_conf_margin=distill_cfg.get("positive_student_conf_margin", 0.0),
            positive_stage_start_fraction=distill_cfg.get("positive_stage_start_fraction"),
            positive_stage_boost=distill_cfg.get("positive_stage_boost"),
            positive_stage_conf_margin=distill_cfg.get("positive_stage_conf_margin"),
            positive_event_gate_enabled=positive_event_cfg.get("enabled", False),
            positive_event_teacher_conf=positive_event_cfg.get("teacher_conf", 0.0),
            positive_event_student_conf=positive_event_cfg.get("student_conf", 1.0),
            positive_event_min_size=positive_event_cfg.get("min_size", 0),
            positive_event_only=positive_event_cfg.get("only", False),
            fusion_teacher_must_match=distill_cfg.get("fusion_teacher_must_match", False),
            uncertainty_weight_enabled=uncertainty_weight_cfg.get("enabled", False),
            uncertainty_weight_scale=uncertainty_weight_cfg.get("scale", 0.0),
            uncertainty_weight_power=uncertainty_weight_cfg.get("power", 1.0),
            uncertainty_weight_clip=uncertainty_weight_cfg.get("clip"),
            delta_schedule_start_fraction=delta_schedule_cfg.get("start_fraction"),
            delta_schedule_end_fraction=delta_schedule_cfg.get("end_fraction"),
            delta_schedule_end_value=delta_schedule_cfg.get("end_value"),
            event_student_focus_enabled=student_focus_cfg.get("enabled", False),
            event_student_focus_warmup=student_focus_cfg.get("warmup", 0),
            event_student_focus_threshold=student_focus_cfg.get("threshold", 1.0),
            event_student_focus_mode=student_focus_cfg.get("mode", "acc_below"),
            event_reweight_enabled=event_reweight_cfg.get("enabled", False),
            event_reweight_warmup_steps=event_reweight_cfg.get("warmup_steps", 0),
            event_reweight_min_size=event_reweight_cfg.get("min_size", 0),
            event_reweight_scale=event_reweight_cfg.get("scale", 0.0),
            event_reweight_power=event_reweight_cfg.get("power", 1.0),
            event_reweight_focus_positive=event_reweight_cfg.get("focus_positive", False),
            event_reweight_clip=event_reweight_cfg.get("clip"),
            fallback_distill_enabled=fallback_cfg.get("enabled", False),
            fallback_min_pairs_fraction=fallback_cfg.get("min_pairs_fraction", 0.0),
            fallback_teacher_conf=fallback_cfg.get("teacher_conf", 0.0),
            fallback_student_conf=fallback_cfg.get("student_conf", 1.0),
            fallback_positive_only=fallback_cfg.get("positive_only", False),
            fallback_require_student_mistake=fallback_cfg.get("require_student_mistake", False),
            soft_quota_enabled=soft_quota_cfg.get("enabled", False),
            soft_quota_min_frac=soft_quota_cfg.get("min_frac", 0.0),
            soft_quota_max_frac=soft_quota_cfg.get("max_frac", 0.5),
            soft_quota_score_temp_coeff=soft_quota_cfg.get("score_temp_coeff", 0.0),
            soft_quota_score_kl_scale=soft_quota_cfg.get("score_kl_scale", 0.0),
            soft_quota_per_event_cap_frac=soft_quota_cfg.get("per_event_cap_frac", 1.0),
            soft_quota_min_reliability=soft_quota_cfg.get("min_reliability", 0.0),
            event_calibration_enabled=event_calib_cfg.get("enabled", False),
            event_calibration_min_seen=event_calib_cfg.get("min_seen", 5),
            event_calibration_default_reliability=event_calib_cfg.get("default_reliability", 0.7),
            event_calibration_temp_scale=event_calib_cfg.get("temp_scale", 1.0),
            event_calibration_conf_scale=event_calib_cfg.get("conf_scale", 0.1),
            event_calibration_alpha0=event_calib_cfg.get("alpha0", 1.0),
            event_calibration_beta0=event_calib_cfg.get("beta0", 1.0),
            event_calibration_min_temp=event_calib_cfg.get("min_temp", 0.7),
            event_calibration_max_temp=event_calib_cfg.get("max_temp", 1.3),
            event_layer_enabled=event_layer_cfg.get("enabled", False),
            event_layer_metric=event_layer_cfg.get("metric", "min"),
            event_layer_min_seen=event_layer_cfg.get("min_seen", 5),
            event_layer_high_threshold=event_layer_cfg.get("high_threshold", 0.92),
            event_layer_mid_threshold=event_layer_cfg.get("mid_threshold", 0.86),
            event_layer_weight_high=event_layer_cfg.get("weight_high", 1.0),
            event_layer_weight_mid=event_layer_cfg.get("weight_mid", 0.7),
            event_layer_weight_low=event_layer_cfg.get("weight_low", 0.4),
            event_layer_default_weight=event_layer_cfg.get("default_weight", 1.0),
            consistency_enabled=consistency_cfg.get("enabled", False),
            consistency_lambda=consistency_cfg.get("lambda", 0.0),
            consistency_metric=consistency_cfg.get("metric", "min"),
            consistency_min_seen=consistency_cfg.get("min_seen", 3),
            consistency_max_score=consistency_cfg.get("max_score", 0.85),
            consistency_apply_positive=consistency_cfg.get("apply_positive", True),
            consistency_apply_negative=consistency_cfg.get("apply_negative", False),
            consistency_default_score=consistency_cfg.get("default_score", 1.0),
            weak_teacher_enabled=weak_teacher_cfg.get("enabled", False),
            weak_teacher_lambda=weak_teacher_cfg.get("lambda", 0.0),
            weak_teacher_metric=weak_teacher_cfg.get("metric", "min"),
            weak_teacher_min_seen=weak_teacher_cfg.get("min_seen", 3),
            weak_teacher_threshold=weak_teacher_cfg.get("threshold", 0.85),
            weak_teacher_apply_positive=weak_teacher_cfg.get("apply_positive", True),
            weak_teacher_apply_negative=weak_teacher_cfg.get("apply_negative", False),
            weak_teacher_temperature=weak_teacher_cfg.get("temperature", 2.2),
            weak_teacher_default_score=weak_teacher_cfg.get("default_score", 0.9),
            weak_teacher_use_ema=weak_teacher_cfg.get("use_ema", False),
            weak_teacher_mix_ratio=weak_teacher_cfg.get("mix_ratio", 0.5),
            weak_teacher_event_ema_enabled=weak_teacher_cfg.get("event_ema_enabled", False),
            weak_teacher_event_ema_decay=weak_teacher_cfg.get("event_ema_decay", 0.6),
            weak_teacher_event_mix=weak_teacher_cfg.get("event_mix", 0.5),
            weak_teacher_dynamic_scale=weak_teacher_dyn_cfg.get("enabled", False),
            weak_teacher_dynamic_min_score=weak_teacher_dyn_cfg.get("min_score", 0.0),
            weak_teacher_dynamic_max_score=weak_teacher_dyn_cfg.get("max_score", 1.0),
            weak_teacher_dynamic_power=weak_teacher_dyn_cfg.get("power", 1.0),
            consensus_gate_enabled=consensus_gate_cfg.get("enabled", False),
            consensus_gate_min_conf=consensus_gate_cfg.get("min_conf", 0.9),
            consensus_gate_require_agreement=consensus_gate_cfg.get("require_agreement", True),
            force_distill_min_frac=force_distill_cfg.get("min_frac", 0.0),
            force_distill_max_frac=force_distill_cfg.get("max_frac", force_distill_cfg.get("min_frac", 0.0)),
            disable_uncertainty_mask=distill_cfg.get("disable_uncertainty_mask", False),
            distill_bootstrap_fraction=distill_cfg.get("bootstrap_fraction", 0.0),
            reliability_enabled=reliability_cfg.get("enabled", False),
            reliability_lambda_mix=reliability_cfg.get("lambda_mix", 0.0),
            reliability_lambda_gate=reliability_cfg.get("lambda_gate", 0.0),
            reliability_lambda_rel=reliability_cfg.get("lambda_rel", 0.0),
            reliability_event_dim=reliability_cfg.get("event_dim", 32),
            reliability_hidden_dim=reliability_cfg.get("hidden_dim", 64),
            reliability_extra_dim=reliability_cfg.get("extra_dim", 2),
            reliability_num_events=reliability_cfg.get("num_events", 5000),
        ),
    )
    if trainer.reliability_module is not None:
        optimizer.add_param_group({"params": trainer.reliability_module.parameters()})

    log_every = max(1, config["training"].get("log_every", 10))

    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []
    best_val_f1 = -float("inf")
    best_state: Optional[str] = None
    best_checkpoint_path: Optional[Path] = None
    best_epoch = -1
    early_stopping_patience = config["training"].get("early_stopping_patience")

    for epoch in range(epochs):
        trainer.update_epoch(epoch, epochs)
        epoch_stats: List[Dict[str, float]] = []
        train_iterable: Iterable = loader
        total_batches = None
        if args.progress:
            try:
                total_batches = len(loader)
            except TypeError:
                total_batches = config["training"].get("max_steps_per_epoch")
            train_iterable = tqdm(
                loader,
                total=total_batches,
                desc=f"epoch {epoch + 1}",
                leave=False,
                dynamic_ncols=False,
                ncols=110,
                mininterval=0.5,
            )
        else:
            train_iterable = loader

        load_start_time = time.perf_counter()
        for batch_idx, batch in enumerate(train_iterable, start=1):
            if args.trace_loader and batch_idx <= 5:
                elapsed = time.perf_counter() - load_start_time
                print(
                    f"[loader] epoch {epoch+1} batch {batch_idx} fetched in {elapsed:.2f}s "
                    f"(max_steps={max_steps})"
                )
            compute_start = time.perf_counter()
            stats = trainer.train_step(batch)
            if args.trace_loader and batch_idx <= 5:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                compute_elapsed = time.perf_counter() - compute_start
                print(
                    f"[compute] epoch {epoch+1} batch {batch_idx} step={trainer.global_step} "
                    f"took {compute_elapsed:.2f}s "
                    f"loss={stats['loss_total']:.4f} distill_pairs={stats['num_distill_pairs']}"
                )
            if batch_idx % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[epoch {epoch+1}/{epochs}] step={trainer.global_step} "
                    f"loss={stats['loss_total']:.4f} "
                    f"fusion={stats['loss_fusion']:.4f} "
                    f"distill={stats['loss_distill']:.4f} "
                    f"pairs={stats['num_distill_pairs']} "
                    f"lr={lr:.2e}"
                )
            epoch_stats.append(stats)
            if max_steps is not None and batch_idx >= max_steps:
                break
            load_start_time = time.perf_counter()

        if epoch_stats:
            summary: Dict[str, float] = {"epoch": epoch + 1}
            for key in epoch_stats[0].keys():
                value = sum(stat[key] for stat in epoch_stats) / len(epoch_stats)
                summary[key] = float(value)
            summary["lr"] = optimizer.param_groups[0]["lr"]
            train_history.append(summary)

        if val_loader is not None:
            val_metrics, _ = evaluate(trainer, val_loader)
            print(
                f"[epoch {epoch+1}/{epochs}] val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} "
                f"val_f1={val_metrics['f1_macro']:.4f} "
                f"val_pos_f1={val_metrics['f1_pos']:.4f} "
                f"val_ece={val_metrics['ece']:.4f}"
            )
            val_entry = {"epoch": epoch + 1, **{k: float(v) for k, v in val_metrics.items()}}
            val_history.append(val_entry)
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_state = "best"
                best_checkpoint_path = run_dir / "model_best.pt"
                torch.save(trainer.state_dict(), best_checkpoint_path)
                ema_state = trainer.ema_state_dict()
                if ema_state is not None:
                    torch.save(ema_state, run_dir / "model_best_ema.pt")
                best_epoch = epoch

            if scheduler and scheduler_type_lower == "reduce_on_plateau":
                scheduler.step(val_metrics["f1_macro"])

            if (
                early_stopping_patience is not None
                and best_epoch >= 0
                and (epoch - best_epoch) >= early_stopping_patience
            ):
                print(
                    f"[info] Early stopping triggered at epoch {epoch+1}; "
                    f"best epoch was {best_epoch+1}."
                )
                break
        else:
            if scheduler and scheduler_type_lower not in {None, "reduce_on_plateau"}:
                scheduler.step()

    test_records: List[Dict[str, Any]] = []
    if best_state == "best" and best_checkpoint_path is not None and best_checkpoint_path.exists():
        state = torch.load(best_checkpoint_path, map_location=trainer.device)
        trainer.core_model.load_state_dict(state)
        if trainer.is_parallel:
            trainer.model.module.load_state_dict(state)
        else:
            trainer.model.load_state_dict(state)
        if val_history:
            print(f"[info] Restored best checkpoint from {best_checkpoint_path} for final evaluation.")

    if test_loader is not None:
        test_metrics, test_records = evaluate(trainer, test_loader)
        print(
            f"[test] loss={test_metrics['loss']:.4f} "
            f"acc={test_metrics['acc']:.4f} "
            f"f1={test_metrics['f1_macro']:.4f} "
            f"pos_f1={test_metrics['f1_pos']:.4f} "
            f"ece={test_metrics['ece']:.4f}"
        )
    else:
        test_metrics = None

    torch.save(trainer.state_dict(), run_dir / "model_last.pt")
    ema_state = trainer.ema_state_dict()
    if ema_state is not None:
        torch.save(ema_state, run_dir / "model_last_ema.pt")

    def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(run_dir / "train_metrics.csv", train_history)
    _write_csv(run_dir / "val_metrics.csv", val_history)

    summary_payload = {
        "config_path": str(args.config),
        "disable_distill": args.disable_distill,
        "seed": args.seed,
        "train_history": train_history,
        "val_history": val_history,
        "test_metrics": test_metrics,
        "best_state": best_state,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    if test_records:
        preds_path = run_dir / "test_predictions.csv"
        fieldnames = list(test_records[0].keys())
        with preds_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_records)


if __name__ == "__main__":
    main()
