from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import compute_dynamic_distillation, dirichlet_evidential_loss
from ..models.multimodal import DynamicModalDistillationModel, ModalOutputs
from ..models import TextEncoder, VisionEncoder
from ..utils import linear_warmup


@dataclass
class TrainerConfig:
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 1.0
    temperature: float = 2.0
    lambda_feat: float = 0.0
    lambda_kl: float = 1.0
    delta: float = 0.05
    evidence_anneal_steps: int = 0
    distill_warmup_steps: int = 0
    use_ema_teacher: bool = False
    ema_decay: float = 0.999
    teacher_config: Optional[str] = None
    teacher_checkpoint: Optional[str] = None
    distill_start_fraction: float = 0.0
    distill_end_fraction: float = 1.0
    adaptive_temperature: bool = False
    temperature_base: float = 2.0
    temperature_coeff: float = 0.0
    use_confidence_gate: bool = False
    confidence_margin: float = 0.0
    use_uncertainty_ema: bool = False
    uncertainty_ema: float = 0.9
    trace_batches: int = 0
    allow_data_parallel: bool = False
    lambda_fusion_to_text: float = 0.0
    lambda_fusion_to_vision: float = 0.0
    fusion_confidence_margin: float = 0.0
    disable_uncertainty_mask: bool = False
    scheduler_type: Optional[str] = None
    class_weights: Optional[torch.Tensor] = None
    min_event_size_for_distill: int = 0
    event_filter_min_teacher_acc: float = 0.0
    event_filter_min_teacher_conf: float = 0.0
    event_filter_warmup_steps: int = 0
    distill_require_student_mistake: bool = True
    agreement_confidence_gap: float = 0.0
    positive_distill_boost: float = 1.0
    positive_student_conf_margin: float = 0.0
    positive_stage_start_fraction: Optional[float] = None
    positive_stage_boost: Optional[float] = None
    positive_stage_conf_margin: Optional[float] = None
    positive_event_gate_enabled: bool = False
    positive_event_teacher_conf: float = 0.0
    positive_event_student_conf: float = 1.0
    positive_event_min_size: int = 0
    positive_event_only: bool = False
    fusion_teacher_must_match: bool = False
    uncertainty_weight_enabled: bool = False
    uncertainty_weight_scale: float = 0.0
    uncertainty_weight_power: float = 1.0
    uncertainty_weight_clip: Optional[float] = None
    delta_schedule_start_fraction: Optional[float] = None
    delta_schedule_end_fraction: Optional[float] = None
    delta_schedule_end_value: Optional[float] = None
    event_student_focus_enabled: bool = False
    event_student_focus_warmup: int = 0
    event_student_focus_threshold: float = 1.0
    event_student_focus_mode: str = "acc_below"
    event_reweight_enabled: bool = False
    event_reweight_warmup_steps: int = 0
    event_reweight_min_size: int = 0
    event_reweight_scale: float = 0.0
    event_reweight_power: float = 1.0
    event_reweight_focus_positive: bool = False
    event_reweight_clip: Optional[float] = None
    fallback_distill_enabled: bool = False
    fallback_min_pairs_fraction: float = 0.0
    fallback_teacher_conf: float = 0.0
    fallback_student_conf: float = 1.0
    fallback_positive_only: bool = False
    fallback_require_student_mistake: bool = False
    fallback_temperature: Optional[float] = None
    fallback_lambda_scale: float = 1.0
    fallback_start_fraction: Optional[float] = None
    fallback_confidence_scale: float = 0.0
    fallback_confidence_power: float = 1.0
    # soft quota selection (unified, before fallback)
    soft_quota_enabled: bool = False
    soft_quota_min_frac: float = 0.0
    soft_quota_max_frac: float = 0.5
    soft_quota_score_temp_coeff: float = 0.0
    soft_quota_score_kl_scale: float = 0.0
    soft_quota_per_event_cap_frac: float = 1.0
    soft_quota_min_reliability: float = 0.0
    event_calibration_enabled: bool = False
    event_calibration_min_seen: int = 5
    event_calibration_default_reliability: float = 0.7
    event_calibration_temp_scale: float = 1.0
    event_calibration_conf_scale: float = 0.1
    event_calibration_alpha0: float = 1.0
    event_calibration_beta0: float = 1.0
    event_calibration_min_temp: float = 0.7
    event_calibration_max_temp: float = 1.3
    event_layer_enabled: bool = False
    event_layer_metric: str = "min"
    event_layer_min_seen: int = 5
    event_layer_high_threshold: float = 0.92
    event_layer_mid_threshold: float = 0.86
    event_layer_weight_high: float = 1.0
    event_layer_weight_mid: float = 0.7
    event_layer_weight_low: float = 0.4
    event_layer_default_weight: float = 1.0
    consistency_enabled: bool = False
    consistency_lambda: float = 0.0
    consistency_metric: str = "min"
    consistency_min_seen: int = 3
    consistency_max_score: float = 0.85
    consistency_apply_positive: bool = True
    consistency_apply_negative: bool = False
    consistency_default_score: float = 1.0
    weak_teacher_enabled: bool = False
    weak_teacher_lambda: float = 0.0
    weak_teacher_metric: str = "min"
    weak_teacher_min_seen: int = 3
    weak_teacher_threshold: float = 0.85
    weak_teacher_apply_positive: bool = True
    weak_teacher_apply_negative: bool = False
    weak_teacher_temperature: float = 2.2
    weak_teacher_default_score: float = 0.9
    weak_teacher_event_mix: float = 0.5
    weak_teacher_use_ema: bool = False
    weak_teacher_mix_ratio: float = 0.5
    weak_teacher_event_ema_enabled: bool = False
    weak_teacher_event_ema_decay: float = 0.6
    weak_teacher_dynamic_scale: bool = False
    weak_teacher_dynamic_min_score: float = 0.0
    weak_teacher_dynamic_max_score: float = 1.0
    weak_teacher_dynamic_power: float = 1.0
    consensus_gate_enabled: bool = False
    consensus_gate_min_conf: float = 0.9
    consensus_gate_require_agreement: bool = True
    force_distill_min_frac: float = 0.0
    force_distill_max_frac: float = 0.0
    # bootstrap: during early fraction of training, ignore uncertainty gates and label constraints
    distill_bootstrap_fraction: float = 0.0


class DynamicDistillationTrainer:
    """Orchestrates optimisation for the dynamic modal distillation model."""

    def __init__(
        self,
        model: DynamicModalDistillationModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config: Optional[TrainerConfig] = None,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or TrainerConfig()
        self.global_step = 0
        self.scheduler_type_lower = (
            self.config.scheduler_type.lower() if self.config.scheduler_type else None
        )
        self.class_weights: Optional[torch.Tensor] = None
        if self.config.class_weights is not None:
            self.class_weights = self.config.class_weights.to(self.device)
        self.event_teacher_stats = defaultdict(lambda: {"seen": 0, "correct": 0, "conf_sum": 0.0})
        self.event_student_stats = defaultdict(
            lambda: {"seen": 0, "correct": 0, "pos_seen": 0, "pos_correct": 0}
        )
        self.event_calibration_stats = defaultdict(
            lambda: {"seen": 0, "correct": 0, "conf_sum": 0.0}
        )
        self.event_logits_ema = defaultdict(
            lambda: {"count": 0, "logits_text": None, "logits_vision": None, "logits_fusion": None}
        )
        self.event_logits_cluster = defaultdict(
            lambda: {"count": 0, "logits": None}
        )
        self.event_student_focus_mode_lower = (
            self.config.event_student_focus_mode.lower()
            if self.config.event_student_focus_mode
            else "acc_below"
        )

        model = model.to(self.device)
        self.is_parallel = False
        if self.config.allow_data_parallel and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model)
            self.is_parallel = True
        else:
            self.model = model
        self.core_model: DynamicModalDistillationModel = (
            self.model.module if self.is_parallel else self.model
        )

        self.ema_model: Optional[DynamicModalDistillationModel] = None
        if self.config.use_ema_teacher:
            self.ema_model = copy.deepcopy(self.core_model).to(self.device)
            for param in self.ema_model.parameters():
                param.requires_grad_(False)

        self.teacher_model: Optional[DynamicModalDistillationModel] = None
        if self.config.teacher_checkpoint:
            teacher = self._build_teacher_model()
            state = torch.load(self.config.teacher_checkpoint, map_location=self.device)
            teacher.load_state_dict(state)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)
            self.teacher_model = teacher

        self.current_epoch_fraction: Optional[float] = None
        self.prev_uncertainty_text: Optional[torch.Tensor] = None
        self.prev_uncertainty_vision: Optional[torch.Tensor] = None

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                moved[key] = {
                    k: (v.to(self.device) if hasattr(v, "to") else v)
                    for k, v in value.items()
                }
            else:
                moved[key] = value.to(self.device) if hasattr(value, "to") else value
        return moved

    def _current_delta(self) -> float:
        delta = float(self.config.delta)
        if (
            self.config.delta_schedule_end_value is not None
            and self.current_epoch_fraction is not None
        ):
            start = self.config.delta_schedule_start_fraction or 0.0
            end = self.config.delta_schedule_end_fraction or 1.0
            if end <= start:
                end = start + 1e-6
            if self.current_epoch_fraction >= start:
                progress = (self.current_epoch_fraction - start) / (end - start)
                progress = float(min(max(progress, 0.0), 1.0))
                target = float(self.config.delta_schedule_end_value)
                delta = delta + (target - delta) * progress
        return delta

    def _cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            reduction="none",
        )
        if sample_weights is not None:
            weights = sample_weights.to(loss.device)
            loss = loss * weights
            denom = weights.sum().clamp_min(1.0)
            return loss.sum() / denom
        return loss.mean()

    def _compute_sample_weights(
        self,
        labels: torch.Tensor,
        meta: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        if not self.config.event_reweight_enabled:
            return None
        if meta is None or not isinstance(meta, dict) or "event_ids" not in meta:
            return None
        if self.global_step < self.config.event_reweight_warmup_steps:
            return None
        event_ids = meta.get("event_ids")
        if not torch.is_tensor(event_ids):
            return None
        event_sizes = meta.get("event_sizes") if isinstance(meta, dict) else None
        labels_cpu = labels
        weights = torch.ones_like(labels_cpu, dtype=torch.float, device=labels.device)
        scale = max(self.config.event_reweight_scale, 0.0)
        if scale <= 0:
            return None
        power = max(self.config.event_reweight_power, 0.0)
        for idx in range(labels_cpu.size(0)):
            eid = int(event_ids[idx].item())
            if eid < 0:
                continue
            if (
                self.config.event_reweight_min_size > 0
                and event_sizes is not None
                and torch.is_tensor(event_sizes)
                and int(event_sizes[idx].item()) < self.config.event_reweight_min_size
            ):
                continue
            stats = self.event_student_stats[eid]
            if self.config.event_reweight_focus_positive and int(labels_cpu[idx].item()) != 1:
                continue
            seen_key = "pos_seen" if self.config.event_reweight_focus_positive else "seen"
            correct_key = "pos_correct" if self.config.event_reweight_focus_positive else "correct"
            seen = max(stats[seen_key], 0)
            correct = max(stats[correct_key], 0)
            if seen <= 0:
                continue
            acc = correct / max(seen, 1)
            boost = 1.0 - acc
            if boost <= 0:
                continue
            if power != 1.0:
                boost = boost ** power
            weight = 1.0 + scale * boost
            if self.config.event_reweight_clip is not None and self.config.event_reweight_clip > 0:
                weight = min(weight, float(self.config.event_reweight_clip))
            weights[idx] = max(weight, 1.0)
        if torch.allclose(weights, torch.ones_like(weights)):
            return None
        return weights

    def _update_student_stats(
        self,
        labels: torch.Tensor,
        fusion_student_pred: torch.Tensor,
        meta: Optional[Dict[str, Any]],
    ) -> None:
        if meta is None or not isinstance(meta, dict) or "event_ids" not in meta:
            return
        event_ids = meta.get("event_ids")
        if not torch.is_tensor(event_ids):
            return
        labels_cpu = labels.detach().cpu().tolist()
        preds_cpu = fusion_student_pred.detach().cpu().tolist()
        events_cpu = event_ids.detach().cpu().tolist()
        for idx, eid in enumerate(events_cpu):
            if eid is None:
                continue
            if int(eid) < 0:
                continue
            stats = self.event_student_stats[int(eid)]
            stats["seen"] += 1
            if preds_cpu[idx] == labels_cpu[idx]:
                stats["correct"] += 1
            if labels_cpu[idx] == 1:
                stats["pos_seen"] += 1
                if preds_cpu[idx] == labels_cpu[idx]:
                    stats["pos_correct"] += 1

    def _build_fallback_masks(
        self,
        labels: torch.Tensor,
        teacher_pred_text: torch.Tensor,
        teacher_pred_vision: torch.Tensor,
        teacher_conf_text: torch.Tensor,
        teacher_conf_vision: torch.Tensor,
        student_pred_text: torch.Tensor,
        student_pred_vision: torch.Tensor,
        student_conf_text: torch.Tensor,
        student_conf_vision: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = labels.device
        fallback_text = torch.zeros_like(labels, dtype=torch.bool, device=device)
        fallback_vision = torch.zeros_like(labels, dtype=torch.bool, device=device)
        if not self.config.fallback_distill_enabled:
            return fallback_text, fallback_vision

        teacher_conf_threshold = max(self.config.fallback_teacher_conf, 0.0)
        student_conf_threshold = min(self.config.fallback_student_conf, 1.0)
        positive_only = self.config.fallback_positive_only

        label_is_pos = labels == 1

        text_candidates = teacher_pred_text == labels
        vision_candidates = teacher_pred_vision == labels
        if teacher_conf_threshold > 0:
            text_candidates = text_candidates & (teacher_conf_text >= teacher_conf_threshold)
            vision_candidates = vision_candidates & (teacher_conf_vision >= teacher_conf_threshold)
        if student_conf_threshold < 1.0:
            text_candidates = text_candidates & (student_conf_vision <= student_conf_threshold)
            vision_candidates = vision_candidates & (student_conf_text <= student_conf_threshold)
        if positive_only:
            text_candidates = text_candidates & label_is_pos
            vision_candidates = vision_candidates & label_is_pos

        if self.config.fallback_require_student_mistake:
            text_candidates = text_candidates & (student_pred_vision != labels)
            vision_candidates = vision_candidates & (student_pred_text != labels)

        fallback_text = text_candidates
        fallback_vision = vision_candidates
        return fallback_text, fallback_vision

    def _build_force_distill_masks(
        self,
        labels: torch.Tensor,
        teacher_conf_text: torch.Tensor,
        teacher_conf_vision: torch.Tensor,
        student_conf_text: torch.Tensor,
        student_conf_vision: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        min_frac = max(self.config.force_distill_min_frac, 0.0)
        max_frac = max(self.config.force_distill_max_frac, min_frac)
        if min_frac <= 0.0:
            return None
        batch_size = labels.size(0)
        min_pairs = max(int(batch_size * min_frac + 1e-6), 1)
        max_pairs = max(int(batch_size * max_frac + 1e-6), min_pairs)
        device = labels.device
        score_text = (teacher_conf_text - student_conf_vision).detach()
        score_vision = (teacher_conf_vision - student_conf_text).detach()
        combined = torch.maximum(score_text, score_vision)
        if combined.numel() == 0:
            return None
        order = torch.argsort(combined, descending=True)
        mask_text = torch.zeros_like(labels, dtype=torch.bool, device=device)
        mask_vision = torch.zeros_like(labels, dtype=torch.bool, device=device)
        selected = 0
        for idx in order.tolist():
            if selected >= max_pairs:
                break
            choose_text = score_text[idx] >= score_vision[idx]
            if score_text[idx] <= 0 and score_vision[idx] <= 0 and selected >= min_pairs:
                break
            if choose_text:
                mask_text[idx] = True
            else:
                mask_vision[idx] = True
            selected += 1
            if selected >= min_pairs and score_text[idx] <= 0 and score_vision[idx] <= 0:
                break
        if selected == 0:
            return None
        return mask_text, mask_vision

    def _build_teacher_model(self) -> DynamicModalDistillationModel:
        if self.config.teacher_config:
            cfg_path = Path(self.config.teacher_config)
            if not cfg_path.exists():
                raise FileNotFoundError(f"Teacher config not found: {cfg_path}")
            with cfg_path.open("r", encoding="utf-8") as handle:
                teacher_cfg = yaml.safe_load(handle)
            model_cfg = teacher_cfg.get("model", {})
            tokenizer_cfg = teacher_cfg.get("tokenizer", {})
            vision_cfg = teacher_cfg.get("vision_processor", {})
            encoder_dim = model_cfg.get("encoder_dim", 768)
            text_encoder = TextEncoder(
                model_name=model_cfg.get("text_model", "bert-base-uncased"),
                trainable=False,
                projection_dim=encoder_dim,
                local_files_only=tokenizer_cfg.get("local_files_only", False),
            )
            vision_encoder = VisionEncoder(
                model_name=model_cfg.get("vision_model", "google/vit-base-patch16-224-in21k"),
                trainable=False,
                projection_dim=encoder_dim,
                local_files_only=vision_cfg.get("local_files_only", False),
            )
            teacher = DynamicModalDistillationModel(
                num_classes=model_cfg.get("num_classes", self.core_model.num_classes),
                text_encoder=text_encoder,
                vision_encoder=vision_encoder,
                encoder_dim=encoder_dim,
                classifier_hidden=model_cfg.get("classifier_hidden", 512),
                dropout=model_cfg.get("dropout", 0.1),
            )
        else:
            teacher = copy.deepcopy(self.core_model)
        return teacher.to(self.device)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        batch = self._move_batch(batch)
        labels = batch["labels"]
        meta = batch.get("meta") if isinstance(batch, dict) else None
        sample_weights = self._compute_sample_weights(labels, meta)

        trace_this = self.config.trace_batches > 0 and self.global_step < self.config.trace_batches
        if trace_this:
            print(f"[trace] step {self.global_step+1} on device {self.device} forward start", flush=True)

        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(
            text_batch=batch["text"],
            vision_batch=batch["vision"],
        )
        if trace_this:
            print("[trace] forward pass complete", flush=True)

        text_modal: ModalOutputs = outputs["text"]
        vision_modal: ModalOutputs = outputs["vision"]
        fusion_logits: torch.Tensor = outputs["fusion_logits"]

        student_probs_text = F.softmax(text_modal.logits.detach(), dim=-1)
        student_conf_text, student_pred_text = student_probs_text.max(dim=-1)
        student_probs_vision = F.softmax(vision_modal.logits.detach(), dim=-1)
        student_conf_vision, student_pred_vision = student_probs_vision.max(dim=-1)

        teacher_conf_text = torch.softmax(text_modal.logits.detach(), dim=-1).max(dim=-1).values
        teacher_conf_vision = torch.softmax(vision_modal.logits.detach(), dim=-1).max(dim=-1).values

        fusion_student_probs = F.softmax(fusion_logits.detach(), dim=-1)
        fusion_student_conf, fusion_student_pred = fusion_student_probs.max(dim=-1)

        ce_fusion = self._cross_entropy(fusion_logits, labels, sample_weights)
        ce_text = self._cross_entropy(text_modal.logits, labels, sample_weights)
        ce_vision = self._cross_entropy(vision_modal.logits, labels, sample_weights)
        if trace_this:
            print(
                "[trace] classification losses "
                f"(fusion={ce_fusion.item():.4f}, text={ce_text.item():.4f}, vision={ce_vision.item():.4f})",
                flush=True,
            )

        evidence_anneal = linear_warmup(
            current_step=self.global_step,
            warmup_steps=self.config.evidence_anneal_steps,
            max_value=1.0,
        )
        loss_evi_text = dirichlet_evidential_loss(
            alpha=text_modal.alpha,
            targets=labels,
            num_classes=self.core_model.num_classes,
            annealing_coef=evidence_anneal,
        )
        loss_evi_vision = dirichlet_evidential_loss(
            alpha=vision_modal.alpha,
            targets=labels,
            num_classes=self.core_model.num_classes,
            annealing_coef=evidence_anneal,
        )

        if trace_this:
            print(
                "[trace] evidential losses "
                f"(text={loss_evi_text.item():.4f}, vision={loss_evi_vision.item():.4f}, anneal={evidence_anneal:.4f})",
                flush=True,
            )

        gamma_scale = 0.0
        if self.config.gamma > 0:
            if self.config.distill_warmup_steps > 0:
                if self.global_step >= self.config.distill_warmup_steps:
                    gamma_scale = self.config.gamma
            else:
                gamma_scale = self.config.gamma

        if (
            gamma_scale > 0
            and self.config.distill_start_fraction > 0
            and self.current_epoch_fraction is not None
            and self.current_epoch_fraction < self.config.distill_start_fraction
        ):
            gamma_scale = 0.0
        if (
            gamma_scale > 0
            and self.config.distill_end_fraction < 1.0
            and self.current_epoch_fraction is not None
            and self.current_epoch_fraction >= self.config.distill_end_fraction
        ):
            gamma_scale = 0.0

        uncertainty_text = text_modal.uncertainty.detach()
        uncertainty_vision = vision_modal.uncertainty.detach()

        if self.config.use_uncertainty_ema:
            with torch.no_grad():
                if (
                    self.prev_uncertainty_text is None
                    or self.prev_uncertainty_text.shape != uncertainty_text.shape
                ):
                    self.prev_uncertainty_text = uncertainty_text.clone()
                else:
                    self.prev_uncertainty_text = (
                        self.config.uncertainty_ema * self.prev_uncertainty_text
                        + (1 - self.config.uncertainty_ema) * uncertainty_text
                    )
                if (
                    self.prev_uncertainty_vision is None
                    or self.prev_uncertainty_vision.shape != uncertainty_vision.shape
                ):
                    self.prev_uncertainty_vision = uncertainty_vision.clone()
                else:
                    self.prev_uncertainty_vision = (
                        self.config.uncertainty_ema * self.prev_uncertainty_vision
                        + (1 - self.config.uncertainty_ema) * uncertainty_vision
                    )
            uncertainty_text_for_distill = self.prev_uncertainty_text
            uncertainty_vision_for_distill = self.prev_uncertainty_vision
        else:
            uncertainty_text_for_distill = uncertainty_text
            uncertainty_vision_for_distill = uncertainty_vision

        eligible_mask = None
        if (
            not self.config.disable_uncertainty_mask
            and self.config.min_event_size_for_distill > 0
            and isinstance(meta, dict)
            and "event_ids" in meta
            and "event_sizes" in meta
        ):
            event_ids = meta["event_ids"].to(labels.device)
            event_sizes = meta["event_sizes"].to(labels.device)
            eligible_mask = (event_ids >= 0) & (event_sizes >= self.config.min_event_size_for_distill)

        device = labels.device
        consistency_loss = torch.tensor(0.0, device=device)
        consistency_pairs = torch.tensor(0, device=device, dtype=torch.long)
        weak_teacher_loss = torch.tensor(0.0, device=device)
        weak_teacher_pairs = torch.tensor(0, device=device, dtype=torch.long)
        weak_teacher_lambda_factor = 1.0
        event_ids_batch: Optional[torch.Tensor] = None
        event_accs: Optional[torch.Tensor] = None
        event_confs: Optional[torch.Tensor] = None
        event_valid_mask: Optional[torch.Tensor] = None
        event_sizes_batch: Optional[torch.Tensor] = None
        effective_delta = self.config.delta
        positive_boost = self.config.positive_distill_boost
        positive_conf_margin = self.config.positive_student_conf_margin

        if gamma_scale > 0:
            effective_delta = self._current_delta()
            if (
                self.config.positive_stage_start_fraction is not None
                and self.current_epoch_fraction is not None
                and self.current_epoch_fraction >= self.config.positive_stage_start_fraction
            ):
                if self.config.positive_stage_boost is not None:
                    positive_boost = self.config.positive_stage_boost
                if self.config.positive_stage_conf_margin is not None:
                    positive_conf_margin = self.config.positive_stage_conf_margin
            if trace_this:
                print("[trace] entering distillation branch", flush=True)
        teacher_kwargs: Dict[str, torch.Tensor] = {}
        ema_teacher_outputs = None
        need_ema_outputs = (
            self.config.weak_teacher_use_ema
            or self.config.weak_teacher_event_ema_enabled
            or (self.teacher_model is None)
        )
        if need_ema_outputs and self.ema_model is not None:
            with torch.no_grad():
                self.ema_model.eval()
                ema_outputs = self.ema_model(
                    text_batch=batch["text"],
                    vision_batch=batch["vision"],
                )
            ema_teacher_outputs = ema_outputs
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    text_batch=batch["text"],
                    vision_batch=batch["vision"],
                )
            teacher_kwargs = {
                "teacher_logits_text": teacher_outputs["text"].logits,
                "teacher_logits_vision": teacher_outputs["vision"].logits,
                "teacher_penultimate_text": teacher_outputs["text"].penultimate,
                "teacher_penultimate_vision": teacher_outputs["vision"].penultimate,
                "teacher_logits_fusion": teacher_outputs["fusion_logits"],
            }
        elif ema_teacher_outputs is not None:
            teacher_kwargs = {
                "teacher_logits_text": ema_teacher_outputs["text"].logits,
                "teacher_logits_vision": ema_teacher_outputs["vision"].logits,
                "teacher_penultimate_text": ema_teacher_outputs["text"].penultimate,
                "teacher_penultimate_vision": ema_teacher_outputs["vision"].penultimate,
                "teacher_logits_fusion": ema_teacher_outputs["fusion_logits"],
            }

        teacher_logits_text = teacher_kwargs.get("teacher_logits_text", text_modal.logits)
        teacher_logits_vision = teacher_kwargs.get("teacher_logits_vision", vision_modal.logits)
        teacher_penultimate_text = teacher_kwargs.get("teacher_penultimate_text", text_modal.penultimate)
        teacher_penultimate_vision = teacher_kwargs.get("teacher_penultimate_vision", vision_modal.penultimate)
        fusion_teacher_logits = teacher_kwargs.get("teacher_logits_fusion", fusion_logits)

        if self.config.consensus_gate_enabled:
            probs_teacher = torch.softmax(fusion_teacher_logits.detach(), dim=-1)
            conf_teacher, pred_teacher = probs_teacher.max(dim=-1)
            if self.config.consensus_gate_require_agreement and ema_teacher_outputs is not None:
                fusion_ema_logits = ema_teacher_outputs["fusion_logits"]
                probs_ema = torch.softmax(fusion_ema_logits.detach(), dim=-1)
                conf_ema, pred_ema = probs_ema.max(dim=-1)
            else:
                conf_ema, pred_ema = conf_teacher, pred_teacher
            min_conf = max(float(self.config.consensus_gate_min_conf), 0.0)
            consensus_mask = (conf_teacher >= min_conf) & (conf_ema >= min_conf)
            if self.config.consensus_gate_require_agreement:
                consensus_mask = consensus_mask & (pred_teacher == pred_ema)
            if eligible_mask is None:
                eligible_mask = consensus_mask
            else:
                eligible_mask = eligible_mask & consensus_mask

        if (
            self.config.event_calibration_enabled
            and meta is not None
            and isinstance(meta, dict)
            and "event_ids" in meta
        ):
            event_ids_batch = meta["event_ids"].to(labels.device)
            cal_temp = torch.ones_like(labels, dtype=torch.float, device=labels.device)
            default_rel = self.config.event_calibration_default_reliability
            temp_scale = self.config.event_calibration_temp_scale
            alpha0 = max(self.config.event_calibration_alpha0, 1e-3)
            beta0 = max(self.config.event_calibration_beta0, 1e-3)
            min_temp = max(self.config.event_calibration_min_temp, 0.2)
            max_temp = max(min_temp + 1e-3, self.config.event_calibration_max_temp)
            for idx, event_id in enumerate(event_ids_batch):
                eid = int(event_id.item())
                if eid < 0:
                    continue
                stats = self.event_calibration_stats[eid]
                seen = stats["seen"]
                correct = stats["correct"]
                total = seen + alpha0 + beta0
                reliability = (correct + alpha0) / max(total, 1e-6)
                if seen < self.config.event_calibration_min_seen:
                    reliability = default_rel * 0.5 + reliability * 0.5
                delta_rel = reliability - default_rel
                adj = 1.0 - temp_scale * delta_rel
                adj = float(torch.clamp(torch.tensor(adj, device=labels.device), min=min_temp, max=max_temp).item())
                cal_temp[idx] = adj
            cal_temp = cal_temp.unsqueeze(-1)
            teacher_logits_text = teacher_logits_text / cal_temp
            teacher_logits_vision = teacher_logits_vision / cal_temp
            fusion_teacher_logits = fusion_teacher_logits / cal_temp

        with torch.no_grad():
            probs_text_full = F.softmax(teacher_logits_text.detach(), dim=-1)
            probs_vision_full = F.softmax(teacher_logits_vision.detach(), dim=-1)
            teacher_conf_text = probs_text_full.max(dim=-1).values
            teacher_conf_vision = probs_vision_full.max(dim=-1).values
            topk_text = torch.topk(probs_text_full, k=min(2, probs_text_full.shape[-1]), dim=-1).values
            topk_vision = torch.topk(probs_vision_full, k=min(2, probs_vision_full.shape[-1]), dim=-1).values
            if topk_text.shape[1] == 1:
                topk_text = torch.cat([topk_text, torch.zeros_like(topk_text)], dim=1)
            if topk_vision.shape[1] == 1:
                topk_vision = torch.cat([topk_vision, torch.zeros_like(topk_vision)], dim=1)
            fusion_probs = F.softmax(fusion_teacher_logits.detach(), dim=-1)
            fusion_conf, fusion_pred = fusion_probs.max(dim=-1)

            reliable_mask = None
            if (
                (self.config.event_filter_min_teacher_acc > 0
                 or self.config.event_filter_min_teacher_conf > 0)
                and meta is not None
                and isinstance(meta, dict)
                and "event_ids" in meta
            ):
                event_ids_batch = meta["event_ids"].to(labels.device)
                reliable_flags = []
                for idx, event_id in enumerate(event_ids_batch):
                    eid = int(event_id.item())
                    if eid < 0:
                        reliable_flags.append(False)
                        continue
                    stats = self.event_teacher_stats[eid]
                    stats["seen"] += 1
                    stats["correct"] += int((fusion_pred[idx] == labels[idx]).item())
                    stats["conf_sum"] += float(fusion_conf[idx].item())
                    if self.config.event_calibration_enabled:
                        cal_stats = self.event_calibration_stats[eid]
                        cal_stats["seen"] += 1
                        cal_stats["correct"] += int((fusion_pred[idx] == labels[idx]).item())
                        cal_stats["conf_sum"] += float(fusion_conf[idx].item())
                    if stats["seen"] < max(self.config.event_filter_warmup_steps, 1):
                        reliable_flags.append(False)
                        continue
                    acc = stats["correct"] / max(stats["seen"], 1)
                    conf_avg = stats["conf_sum"] / max(stats["seen"], 1)
                    if (
                        acc < self.config.event_filter_min_teacher_acc
                        or conf_avg < self.config.event_filter_min_teacher_conf
                    ):
                        reliable_flags.append(False)
                    else:
                        reliable_flags.append(True)
                reliable_mask = torch.tensor(reliable_flags, device=labels.device, dtype=torch.bool)

            if eligible_mask is not None and reliable_mask is not None:
                eligible_mask = eligible_mask & reliable_mask
            elif reliable_mask is not None:
                eligible_mask = reliable_mask

            if self.config.fusion_teacher_must_match:
                match_mask = fusion_pred == labels
                if eligible_mask is None:
                    eligible_mask = match_mask
                else:
                    eligible_mask = eligible_mask & match_mask

            if (
                self.config.event_student_focus_enabled
                and isinstance(meta, dict)
                and "event_ids" in meta
            ):
                event_ids_batch = meta["event_ids"].to(labels.device)
                focus_flags = []
                warmup = max(self.config.event_student_focus_warmup, 0)
                threshold = float(self.config.event_student_focus_threshold)
                for idx, event_id in enumerate(event_ids_batch):
                    eid = int(event_id.item())
                    if eid < 0:
                        focus_flags.append(True)
                        continue
                    stats = self.event_student_stats[eid]
                    if stats["seen"] < max(warmup, 1):
                        focus_flags.append(True)
                        continue
                    if self.event_student_focus_mode_lower == "pos_recall":
                        seen = max(stats["pos_seen"], 1)
                        metric = stats["pos_correct"] / seen
                    else:
                        seen = max(stats["seen"], 1)
                        metric = stats["correct"] / seen
                    focus_flags.append(metric <= threshold)
                focus_mask = torch.tensor(focus_flags, device=labels.device, dtype=torch.bool)
                if eligible_mask is None:
                    eligible_mask = focus_mask
                else:
                    eligible_mask = eligible_mask & focus_mask

            positive_event_mask: Optional[torch.Tensor] = None
            event_accs: Optional[torch.Tensor] = None
            event_confs: Optional[torch.Tensor] = None
            event_valid_mask: Optional[torch.Tensor] = None
            event_sizes_batch: Optional[torch.Tensor] = None
            positive_event_mask: Optional[torch.Tensor] = None

            if meta is not None and isinstance(meta, dict) and "event_ids" in meta:
                event_ids_batch = meta["event_ids"].to(labels.device)
                event_valid_mask = event_ids_batch >= 0
                if "event_sizes" in meta:
                    event_sizes_batch = meta["event_sizes"].to(labels.device)
                else:
                    event_sizes_batch = torch.ones_like(labels, dtype=torch.long, device=labels.device)
                event_accs = torch.full_like(labels, float(self.config.consistency_default_score), dtype=torch.float32)
                event_confs = torch.full_like(labels, float(self.config.consistency_default_score), dtype=torch.float32)
                for idx_sample, event_id in enumerate(event_ids_batch):
                    eid = int(event_id.item())
                    if eid < 0:
                        continue
                    stats = self.event_teacher_stats[eid]
                    seen = max(stats["seen"], 1)
                    acc = stats["correct"] / seen
                    conf_avg = stats["conf_sum"] / seen
                    event_accs[idx_sample] = acc
                    event_confs[idx_sample] = conf_avg

                if self.config.positive_event_gate_enabled:
                    label_positive = labels == 1
                    size_ok = event_sizes_batch >= self.config.positive_event_min_size
                    teacher_positive = fusion_pred == 1
                    teacher_conf_ok = fusion_conf >= self.config.positive_event_teacher_conf
                    if self.config.positive_event_student_conf < 1.0:
                        student_conf_ok = fusion_student_conf <= self.config.positive_event_student_conf
                    else:
                        student_conf_ok = torch.ones_like(fusion_student_conf, dtype=torch.bool)
                    positive_event_mask = (
                        label_positive
                        & event_valid_mask
                        & size_ok
                        & teacher_positive
                        & teacher_conf_ok
                        & student_conf_ok
                    )
                    if self.config.positive_event_only:
                        if eligible_mask is None:
                            eligible_mask = torch.ones_like(labels, dtype=torch.bool, device=labels.device)
                        eligible_mask = torch.where(label_positive, positive_event_mask, eligible_mask)
                    if positive_event_mask is not None and not positive_event_mask.any():
                        positive_event_mask = None

            if (
                self.config.weak_teacher_event_ema_enabled
                and event_ids_batch is not None
            ):
                self._update_event_logits_ema(
                    event_ids_batch,
                    teacher_logits_text.detach(),
                    teacher_logits_vision.detach(),
                    fusion_teacher_logits.detach(),
                )

            override_weights = None
            event_scores_layer = None
            if (
                self.config.event_layer_enabled
                and event_accs is not None
                and event_confs is not None
                and event_valid_mask is not None
            ):
                weights = torch.full_like(labels, float(self.config.event_layer_default_weight), dtype=torch.float32)
                event_scores_layer = torch.full_like(labels, float(self.config.event_layer_default_weight), dtype=torch.float32)
                metric_layer = self.config.event_layer_metric.lower()
                min_seen_layer = max(self.config.event_layer_min_seen, 1)
                for idx_sample in range(labels.size(0)):
                    if not event_valid_mask[idx_sample]:
                        weights[idx_sample] = float(self.config.event_layer_weight_low)
                        event_scores_layer[idx_sample] = self.config.event_layer_default_weight
                        continue
                    stats = self.event_teacher_stats[int(event_ids_batch[idx_sample].item())]
                    seen = stats["seen"]
                    if seen < min_seen_layer:
                        weight = self.config.event_layer_weight_low
                        score = self.config.event_layer_default_weight
                    else:
                        acc = event_accs[idx_sample].item()
                        conf_avg = event_confs[idx_sample].item()
                        if metric_layer == "acc":
                            score = acc
                        elif metric_layer == "conf":
                            score = conf_avg
                        elif metric_layer == "avg":
                            score = 0.5 * (acc + conf_avg)
                        else:
                            score = min(acc, conf_avg)
                        if score >= self.config.event_layer_high_threshold:
                            weight = self.config.event_layer_weight_high
                        elif score >= self.config.event_layer_mid_threshold:
                            weight = self.config.event_layer_weight_mid
                        else:
                            weight = self.config.event_layer_weight_low
                    weights[idx_sample] = float(weight)
                    event_scores_layer[idx_sample] = float(score)
                override_weights = weights.to(device)

        # base distillation with eligibility masks (event filter / positive gate / confidence gate)
        override_mask_text_arg = None
        override_mask_vision_arg = None
        # bootstrap window: force-enable pairs regardless of uncertainty/eligibility, and ignore label constraints
        in_bootstrap = (
            (self.config.distill_bootstrap_fraction > 0.0)
            and (self.current_epoch_fraction is not None)
            and (self.current_epoch_fraction < self.config.distill_bootstrap_fraction)
        )
        if self.config.disable_uncertainty_mask or in_bootstrap:
            override_mask_text_arg = torch.ones_like(labels, dtype=torch.bool, device=device)
            override_mask_vision_arg = torch.ones_like(labels, dtype=torch.bool, device=device)
        distill = {
            "kl_loss": torch.tensor(0.0, device=device),
            "feature_loss": torch.tensor(0.0, device=device),
            "loss": torch.tensor(0.0, device=device),
            "num_pairs": torch.tensor(0, device=device, dtype=torch.long),
            "mask_text_teacher": torch.zeros(labels.shape[0], device=device, dtype=torch.bool),
            "mask_vision_teacher": torch.zeros(labels.shape[0], device=device, dtype=torch.bool),
            "fusion_loss": torch.tensor(0.0, device=device),
            "fusion_pairs": torch.tensor(0, device=device, dtype=torch.long),
        }
        if gamma_scale > 0:
            distill = compute_dynamic_distillation(
                logits_text=text_modal.logits,
                logits_vision=vision_modal.logits,
                penultimate_text=text_modal.penultimate,
                penultimate_vision=vision_modal.penultimate,
                uncertainty_text=uncertainty_text_for_distill,
                uncertainty_vision=uncertainty_vision_for_distill,
                labels=labels if not (self.config.disable_uncertainty_mask or in_bootstrap) else None,
                temperature=self.config.temperature,
                lambda_kl=self.config.lambda_kl,
                lambda_feat=self.config.lambda_feat,
                delta=effective_delta,
                teacher_logits_text=teacher_logits_text,
                teacher_logits_vision=teacher_logits_vision,
                teacher_penultimate_text=teacher_penultimate_text,
                teacher_penultimate_vision=teacher_penultimate_vision,
                teacher_probs_text=topk_text,
                teacher_probs_vision=topk_vision,
                require_student_mistake=self.config.distill_require_student_mistake,
                adaptive_temperature=self.config.adaptive_temperature,
                temperature_base=self.config.temperature_base,
                temperature_coeff=self.config.temperature_coeff,
                use_confidence_gate=self.config.use_confidence_gate,
                confidence_margin=self.config.confidence_margin,
                eligible_mask=None if in_bootstrap else eligible_mask,
                agreement_confidence_gap=self.config.agreement_confidence_gap,
                positive_distill_boost=positive_boost,
                positive_student_conf_margin=positive_conf_margin,
                positive_focus_mask=positive_event_mask,
                uncertainty_weight_enabled=self.config.uncertainty_weight_enabled,
                uncertainty_weight_scale=self.config.uncertainty_weight_scale,
                uncertainty_weight_power=self.config.uncertainty_weight_power,
                uncertainty_weight_clip=self.config.uncertainty_weight_clip,
                override_weights_text=override_weights,
                override_weights_vision=override_weights,
                override_mask_text=override_mask_text_arg,
                override_mask_vision=override_mask_vision_arg,
            )
            if trace_this or in_bootstrap:
                mt = int(distill["mask_text_teacher"].sum().item()) if "mask_text_teacher" in distill else -1
                mv = int(distill["mask_vision_teacher"].sum().item()) if "mask_vision_teacher" in distill else -1
                pairs = int(distill.get("num_pairs", torch.tensor(0, device=device)).item())
                print(
                    f"[trace] masks after base distill: text={mt} vision={mv} num_pairs={pairs}",
                    flush=True,
                )

            # soft-quota: ensure a minimum coverage using scored selection (before fallback)
            if self.config.soft_quota_enabled and labels.numel() > 0:
                min_frac = max(self.config.soft_quota_min_frac, 0.0)
                max_frac = max(self.config.soft_quota_max_frac, min_frac)
                batch_size = labels.size(0)
                min_pairs = int(batch_size * min_frac + 1e-6)
                max_pairs = int(batch_size * max_frac + 1e-6)
                current_pairs = int(distill["num_pairs"].item())
                if current_pairs < min_pairs:
                    # score = (teacher_conf - student_conf).clamp_min(0) * reliability_flag * class_factor
                    with torch.no_grad():
                        reliability_flag = torch.ones_like(labels, dtype=torch.float, device=device)
                        if reliable_mask is not None:
                            reliability_flag = reliable_mask.float()
                        pos_mask = (labels == 1)
                        class_factor = torch.where(pos_mask, torch.full_like(reliability_flag, 1.2), torch.ones_like(reliability_flag))
                        score_text = (teacher_conf_text - student_conf_vision).clamp_min(0.0) * reliability_flag * class_factor
                        score_vision = (teacher_conf_vision - student_conf_text).clamp_min(0.0) * reliability_flag * class_factor
                        # 禁止已选中的样本再次计入
                        score_text = torch.where(distill["mask_text_teacher"], torch.zeros_like(score_text), score_text)
                        score_vision = torch.where(distill["mask_vision_teacher"], torch.zeros_like(score_vision), score_vision)
                        # 每事件上限
                        if (
                            isinstance(meta, dict)
                            and "event_ids" in meta
                            and self.config.soft_quota_per_event_cap_frac < 1.0
                        ):
                            per_event_cap = max(int(batch_size * self.config.soft_quota_per_event_cap_frac + 1e-6), 1)
                            event_ids_batch = meta["event_ids"].to(labels.device)
                            # 简易抑制：对超过上限的事件强制将其分数裁剪到分位数以下
                            unique_eids = torch.unique(event_ids_batch)
                            for eid in unique_eids:
                                if int(eid.item()) < 0:
                                    continue
                                mask_e = event_ids_batch == eid
                                count_e = int(mask_e.sum().item())
                                if count_e > per_event_cap:
                                    # 将最小的 per_event_cap 个保留，其他分数乘以0.0（禁用）
                                    vals = (score_text[mask_e] + score_vision[mask_e])
                                    if vals.numel() > per_event_cap:
                                        thresh = torch.topk(vals, per_event_cap, largest=True).values.min()
                                        drop = mask_e & ( (score_text + score_vision) < thresh )
                                        score_text = torch.where(drop, torch.zeros_like(score_text), score_text)
                                        score_vision = torch.where(drop, torch.zeros_like(score_vision), score_vision)

                    # 组装候选并选取 top-K（限制到 max_pairs）
                    with torch.no_grad():
                        # 先合并两个方向，优先选择更大的方向
                        choose_text = score_text >= score_vision
                        scores = torch.where(choose_text, score_text, score_vision)
                        if (
                            isinstance(meta, dict)
                            and "event_ids" in meta
                        ):
                            event_ids_batch = meta["event_ids"].to(labels.device)
                            alpha0 = max(self.config.event_calibration_alpha0, 1e-3)
                            beta0 = max(self.config.event_calibration_beta0, 1e-3)
                            default_rel = self.config.event_calibration_default_reliability
                            rel_list = []
                            for idx, eid_val in enumerate(event_ids_batch):
                                eid = int(eid_val.item())
                                if eid < 0:
                                    rel_list.append(default_rel)
                                    continue
                                stats = self.event_calibration_stats[eid]
                                seen = stats["seen"]
                                correct = stats["correct"]
                                rel = (correct + alpha0) / max(seen + alpha0 + beta0, 1e-6)
                                if seen < self.config.event_calibration_min_seen:
                                    rel = 0.5 * default_rel + 0.5 * rel
                                rel_list.append(rel)
                            reliability_scores = torch.tensor(rel_list, device=labels.device, dtype=scores.dtype)
                            min_rel = max(self.config.soft_quota_min_reliability, 0.0)
                            scores = torch.where(reliability_scores >= min_rel, scores, torch.zeros_like(scores))
                        order = torch.argsort(scores, descending=True)
                        target_pairs = max(min_pairs - current_pairs, 0)
                        target_pairs = min(target_pairs, max(0, max_pairs - current_pairs))
                        if target_pairs > 0:
                            pick_idx = order[:target_pairs]
                            add_text_mask = torch.zeros_like(choose_text)
                            add_vision_mask = torch.zeros_like(choose_text)
                            add_text_mask[pick_idx] = choose_text[pick_idx]
                            add_vision_mask[pick_idx] = ~choose_text[pick_idx]
                            # 样本级权重（1 + score_scale * score^power）
                            score_scale = max(self.config.soft_quota_score_kl_scale, 0.0)
                            temp_coeff = max(self.config.soft_quota_score_temp_coeff, 0.0)
                            weights_text = None
                            weights_vision = None
                            if score_scale > 0 or temp_coeff > 0:
                                base = scores
                                if score_scale > 0:
                                    scale_vec = 1.0 + score_scale * base
                                else:
                                    scale_vec = torch.ones_like(base)
                                # 这里只作为权重放大；温度可在下次重算时通过 override（此处不改变全局温度，以降低复杂性）
                                weights_text = scale_vec
                                weights_vision = scale_vec

                    if target_pairs > 0:
                        # 用覆写掩码与样本权重重算蒸馏
                        combined_text_mask = distill["mask_text_teacher"] | add_text_mask
                        combined_vision_mask = distill["mask_vision_teacher"] | add_vision_mask
                        distill = compute_dynamic_distillation(
                            logits_text=text_modal.logits,
                            logits_vision=vision_modal.logits,
                            penultimate_text=text_modal.penultimate,
                            penultimate_vision=vision_modal.penultimate,
                            uncertainty_text=uncertainty_text_for_distill,
                            uncertainty_vision=uncertainty_vision_for_distill,
                            labels=labels if not self.config.disable_uncertainty_mask else None,
                            temperature=self.config.temperature,
                            lambda_kl=self.config.lambda_kl,
                            lambda_feat=self.config.lambda_feat,
                            delta=effective_delta,
                            teacher_logits_text=teacher_logits_text,
                            teacher_logits_vision=teacher_logits_vision,
                            teacher_penultimate_text=teacher_penultimate_text,
                            teacher_penultimate_vision=teacher_penultimate_vision,
                            teacher_probs_text=topk_text,
                            teacher_probs_vision=topk_vision,
                            require_student_mistake=self.config.distill_require_student_mistake,
                            adaptive_temperature=self.config.adaptive_temperature,
                            temperature_base=self.config.temperature_base,
                            temperature_coeff=self.config.temperature_coeff,
                            use_confidence_gate=self.config.use_confidence_gate,
                            confidence_margin=self.config.confidence_margin,
                            eligible_mask=None,
                            agreement_confidence_gap=self.config.agreement_confidence_gap,
                            positive_distill_boost=positive_boost,
                            positive_student_conf_margin=positive_conf_margin,
                            positive_focus_mask=positive_event_mask,
                            uncertainty_weight_enabled=self.config.uncertainty_weight_enabled,
                            uncertainty_weight_scale=self.config.uncertainty_weight_scale,
                            uncertainty_weight_power=self.config.uncertainty_weight_power,
                            uncertainty_weight_clip=self.config.uncertainty_weight_clip,
                            override_mask_text=combined_text_mask,
                            override_mask_vision=combined_vision_mask,
                            override_weights_text=weights_text,
                            override_weights_vision=weights_vision,
                        )
            if (
                self.config.force_distill_min_frac > 0
                and labels.numel() > 0
                and distill["num_pairs"].item() == 0
            ):
                masks = self._build_force_distill_masks(
                    labels=labels,
                    teacher_conf_text=teacher_conf_text,
                    teacher_conf_vision=teacher_conf_vision,
                    student_conf_text=student_conf_text,
                    student_conf_vision=student_conf_vision,
                )
                if masks is not None:
                    force_text_mask, force_vision_mask = masks
                    if force_text_mask.any() or force_vision_mask.any():
                        if trace_this or self.config.force_distill_min_frac > 0:
                            print(
                                f"[trace] force distill triggered "
                                f"(text_pairs={int(force_text_mask.sum().item())}, "
                                f"vision_pairs={int(force_vision_mask.sum().item())})",
                                flush=True,
                            )
                        distill = compute_dynamic_distillation(
                            logits_text=text_modal.logits,
                            logits_vision=vision_modal.logits,
                            penultimate_text=text_modal.penultimate,
                            penultimate_vision=vision_modal.penultimate,
                            uncertainty_text=uncertainty_text_for_distill,
                            uncertainty_vision=uncertainty_vision_for_distill,
                            labels=labels if not self.config.disable_uncertainty_mask else None,
                            temperature=self.config.temperature,
                            lambda_kl=self.config.lambda_kl,
                            lambda_feat=self.config.lambda_feat,
                            delta=effective_delta,
                            teacher_logits_text=teacher_logits_text,
                            teacher_logits_vision=teacher_logits_vision,
                            teacher_penultimate_text=teacher_penultimate_text,
                            teacher_penultimate_vision=teacher_penultimate_vision,
                            teacher_probs_text=topk_text,
                            teacher_probs_vision=topk_vision,
                            require_student_mistake=False,
                            adaptive_temperature=self.config.adaptive_temperature,
                            temperature_base=self.config.temperature_base,
                            temperature_coeff=self.config.temperature_coeff,
                            use_confidence_gate=False,
                            confidence_margin=0.0,
                            eligible_mask=None,
                            agreement_confidence_gap=self.config.agreement_confidence_gap,
                            positive_distill_boost=positive_boost,
                            positive_student_conf_margin=positive_conf_margin,
                            positive_focus_mask=None,
                            uncertainty_weight_enabled=self.config.uncertainty_weight_enabled,
                            uncertainty_weight_scale=self.config.uncertainty_weight_scale,
                            uncertainty_weight_power=self.config.uncertainty_weight_power,
                            uncertainty_weight_clip=self.config.uncertainty_weight_clip,
                            override_mask_text=force_text_mask,
                            override_mask_vision=force_vision_mask,
                        )
            initial_pairs = distill["num_pairs"].clone()
            fallback_pairs_tensor = torch.tensor(0, device=device, dtype=torch.long)
            if self.config.fallback_distill_enabled and labels.numel() > 0:
                start_fraction = self.config.fallback_start_fraction or 0.0
                if self.current_epoch_fraction is None or self.current_epoch_fraction >= start_fraction:
                    min_frac = max(self.config.fallback_min_pairs_fraction, 0.0)
                    min_pairs = int(labels.size(0) * min_frac)
                    if min_frac > 0.0 and min_pairs == 0:
                        min_pairs = 1
                    if distill["num_pairs"].item() < min_pairs:
                        fallback_text_mask, fallback_vision_mask = self._build_fallback_masks(
                            labels=labels,
                            teacher_pred_text=teacher_logits_text.argmax(dim=-1),
                            teacher_pred_vision=teacher_logits_vision.argmax(dim=-1),
                            teacher_conf_text=teacher_conf_text,
                            teacher_conf_vision=teacher_conf_vision,
                            student_pred_text=student_pred_text,
                            student_pred_vision=student_pred_vision,
                            student_conf_text=student_conf_text,
                            student_conf_vision=student_conf_vision,
                        )
                        combined_text_mask = distill["mask_text_teacher"] | fallback_text_mask
                        combined_vision_mask = distill["mask_vision_teacher"] | fallback_vision_mask
                        if combined_text_mask.any() or combined_vision_mask.any():
                            fallback_temp = (
                                self.config.fallback_temperature
                                if self.config.fallback_temperature is not None
                                else self.config.temperature
                            )
                            base_lambda = self.config.lambda_kl * max(self.config.fallback_lambda_scale, 0.0)
                            fallback_lambda = base_lambda
                            conf_scale = max(self.config.fallback_confidence_scale, 0.0)
                            fallback_weights_text = None
                            fallback_weights_vision = None
                            if conf_scale > 0:
                                power = max(self.config.fallback_confidence_power, 0.0)
                                if combined_text_mask.any():
                                    gap_text = (teacher_conf_text - student_conf_vision).clamp_min(0.0)
                                    weights_text = torch.ones_like(gap_text, device=gap_text.device)
                                    weights_text = weights_text + conf_scale * (gap_text ** power)
                                    fallback_weights_text = weights_text
                                if combined_vision_mask.any():
                                    gap_vision = (teacher_conf_vision - student_conf_text).clamp_min(0.0)
                                    weights_vision = torch.ones_like(gap_vision, device=gap_vision.device)
                                    weights_vision = weights_vision + conf_scale * (gap_vision ** power)
                                    fallback_weights_vision = weights_vision
                            distill = compute_dynamic_distillation(
                                logits_text=text_modal.logits,
                                logits_vision=vision_modal.logits,
                                penultimate_text=text_modal.penultimate,
                                penultimate_vision=vision_modal.penultimate,
                                uncertainty_text=uncertainty_text_for_distill,
                                uncertainty_vision=uncertainty_vision_for_distill,
                                labels=labels if not self.config.disable_uncertainty_mask else None,
                                temperature=fallback_temp,
                                lambda_kl=fallback_lambda,
                                lambda_feat=self.config.lambda_feat,
                                delta=effective_delta,
                                teacher_logits_text=teacher_logits_text,
                                teacher_logits_vision=teacher_logits_vision,
                                teacher_penultimate_text=teacher_penultimate_text,
                                teacher_penultimate_vision=teacher_penultimate_vision,
                                teacher_probs_text=topk_text,
                                teacher_probs_vision=topk_vision,
                                require_student_mistake=self.config.fallback_require_student_mistake,
                                adaptive_temperature=self.config.adaptive_temperature,
                                temperature_base=self.config.temperature_base,
                                temperature_coeff=self.config.temperature_coeff,
                                use_confidence_gate=self.config.use_confidence_gate,
                                confidence_margin=self.config.confidence_margin,
                                eligible_mask=None,
                                agreement_confidence_gap=self.config.agreement_confidence_gap,
                                positive_distill_boost=positive_boost,
                            positive_student_conf_margin=positive_conf_margin,
                            positive_focus_mask=positive_event_mask,
                            uncertainty_weight_enabled=self.config.uncertainty_weight_enabled,
                                uncertainty_weight_scale=self.config.uncertainty_weight_scale,
                                uncertainty_weight_power=self.config.uncertainty_weight_power,
                                uncertainty_weight_clip=self.config.uncertainty_weight_clip,
                                override_mask_text=combined_text_mask,
                                override_mask_vision=combined_vision_mask,
                                override_weights_text=fallback_weights_text,
                                override_weights_vision=fallback_weights_vision,
                            )
                        fallback_pairs_tensor = distill["num_pairs"] - initial_pairs
                        fallback_pairs_tensor = torch.clamp_min(fallback_pairs_tensor, 0)
            distill["fallback_pairs"] = fallback_pairs_tensor
            if trace_this:
                print(
                    "[trace] distillation losses "
                    f"(kl={distill['kl_loss'].item():.4f}, feat={distill['feature_loss'].item():.4f}, "
                    f"pairs={distill['num_pairs'].item()})",
                    flush=True,
                )

            scores_for_consistency: Optional[torch.Tensor] = None
            if event_ids_batch is not None and event_valid_mask is not None:
                scores_for_consistency = self._compute_event_scores(
                    labels=labels,
                    metric=self.config.consistency_metric,
                    min_seen=self.config.consistency_min_seen,
                    default_score=self.config.consistency_default_score,
                    event_ids=event_ids_batch,
                    valid_mask=event_valid_mask,
                    event_accs=event_accs,
                    event_confs=event_confs,
                )

            if (
                self.config.consistency_enabled
                and self.config.consistency_lambda > 0.0
                and scores_for_consistency is not None
            ):
                scores_cons = scores_for_consistency
                mask_cons = event_valid_mask & (scores_cons <= self.config.consistency_max_score)
                if self.config.consistency_apply_positive and not self.config.consistency_apply_negative:
                    mask_cons = mask_cons & (labels == 1)
                elif self.config.consistency_apply_negative and not self.config.consistency_apply_positive:
                    mask_cons = mask_cons & (labels == 0)
                elif not self.config.consistency_apply_positive and not self.config.consistency_apply_negative:
                    mask_cons = mask_cons & torch.zeros_like(labels, dtype=torch.bool, device=device)
                if mask_cons.any():
                    logp_text = F.log_softmax(text_modal.logits[mask_cons], dim=-1)
                    logp_vision = F.log_softmax(vision_modal.logits[mask_cons], dim=-1)
                    probs_text = logp_text.exp()
                    probs_vision = logp_vision.exp()
                    kl_tv = F.kl_div(logp_text, probs_vision, reduction="batchmean")
                    kl_vt = F.kl_div(logp_vision, probs_text, reduction="batchmean")
                    consistency_loss = 0.5 * (kl_tv + kl_vt)
                    consistency_pairs = torch.tensor(int(mask_cons.sum().item()), device=device, dtype=torch.long)
            if trace_this:
                print(f"[trace] consistency_loss={consistency_loss.item():.4f}", flush=True)

            if (
                self.config.weak_teacher_enabled
                and self.config.weak_teacher_lambda > 0.0
                and event_ids_batch is not None
                and event_valid_mask is not None
            ):
                scores_weak = self._compute_event_scores(
                    labels=labels,
                    metric=self.config.weak_teacher_metric,
                    min_seen=self.config.weak_teacher_min_seen,
                    default_score=self.config.weak_teacher_default_score,
                    event_ids=event_ids_batch,
                    valid_mask=event_valid_mask,
                    event_accs=event_accs,
                    event_confs=event_confs,
                )
                mask_weak = event_valid_mask & (scores_weak <= self.config.weak_teacher_threshold)
                if self.config.weak_teacher_apply_positive and not self.config.weak_teacher_apply_negative:
                    mask_weak = mask_weak & (labels == 1)
                elif self.config.weak_teacher_apply_negative and not self.config.weak_teacher_apply_positive:
                    mask_weak = mask_weak & (labels == 0)
                elif not self.config.weak_teacher_apply_positive and not self.config.weak_teacher_apply_negative:
                    mask_weak = mask_weak & torch.zeros_like(labels, dtype=torch.bool, device=device)
                if mask_weak.any():
                    weak_teacher_logits_fusion = fusion_teacher_logits
                    if self.config.weak_teacher_use_ema and ema_teacher_outputs is not None:
                        mix_ratio = max(0.0, min(1.0, float(self.config.weak_teacher_mix_ratio)))
                        ema_fusion = ema_teacher_outputs["fusion_logits"]
                        weak_teacher_logits_fusion = mix_ratio * fusion_teacher_logits + (1.0 - mix_ratio) * ema_fusion
                    weak_logits_batch = weak_teacher_logits_fusion.clone()
                    if (
                        self.config.weak_teacher_event_ema_enabled
                        and self.config.weak_teacher_event_mix > 0.0
                    ):
                        event_mix = max(0.0, min(1.0, float(self.config.weak_teacher_event_mix)))
                        if event_mix > 0.0:
                            indices = mask_weak.nonzero(as_tuple=False).squeeze(-1)
                            for idx_sample in indices:
                                eid = int(event_ids_batch[idx_sample].item())
                                event_state = self.event_logits_ema[eid]
                                event_logits = event_state["logits_fusion"]
                                if event_logits is None:
                                    continue
                                weak_logits_batch[idx_sample] = (
                                    event_mix * weak_logits_batch[idx_sample]
                                    + (1.0 - event_mix) * event_logits.to(device)
                                )
                    temp_w = max(self.config.weak_teacher_temperature, 1e-6)
                    teacher_soft = F.softmax(weak_logits_batch[mask_weak] / temp_w, dim=-1)
                    logp_text_w = F.log_softmax(text_modal.logits[mask_weak] / temp_w, dim=-1)
                    logp_vision_w = F.log_softmax(vision_modal.logits[mask_weak] / temp_w, dim=-1)
                    kl_text_w = F.kl_div(logp_text_w, teacher_soft, reduction="batchmean") * (temp_w ** 2)
                    kl_vision_w = F.kl_div(logp_vision_w, teacher_soft, reduction="batchmean") * (temp_w ** 2)
                    weak_teacher_loss = 0.5 * (kl_text_w + kl_vision_w)
                    weak_teacher_pairs = torch.tensor(int(mask_weak.sum().item()), device=device, dtype=torch.long)
                    if self.config.weak_teacher_dynamic_scale:
                        min_score = self.config.weak_teacher_dynamic_min_score
                        max_score = max(self.config.weak_teacher_dynamic_max_score, min_score + 1e-6)
                        selected_scores = scores_weak[mask_weak].clamp(min_score, max_score)
                        rng = max_score - min_score
                        if rng <= 0:
                            norm = torch.ones_like(selected_scores)
                        else:
                            norm = (max_score - selected_scores) / rng
                        power = max(self.config.weak_teacher_dynamic_power, 1e-6)
                        norm = norm.clamp(0.0, 1.0) ** power
                        weak_teacher_lambda_factor = max(norm.mean().item(), 1e-3)
                    if trace_this:
                        print(
                            f"[trace] weak_teacher_loss={weak_teacher_loss.item():.4f}"
                            f" lambda_factor={weak_teacher_lambda_factor:.4f}",
                            flush=True,
                        )

            fusion_distill_loss = torch.tensor(0.0, device=device)
            fusion_pairs = torch.tensor(0, device=device, dtype=torch.long)
            if (
                gamma_scale > 0
                and (self.config.lambda_fusion_to_text > 0 or self.config.lambda_fusion_to_vision > 0)
            ):
                with torch.no_grad():
                    probs_fusion = F.softmax(fusion_teacher_logits.detach(), dim=-1)
                    fusion_confidence, fusion_pred = probs_fusion.max(dim=-1)
                    fusion_mask = fusion_pred == labels
                    if self.config.fusion_confidence_margin > 0:
                        fusion_mask = fusion_mask & (fusion_confidence >= self.config.fusion_confidence_margin)
                    if eligible_mask is not None:
                        fusion_mask = fusion_mask & eligible_mask
                    fusion_indices = fusion_mask.nonzero(as_tuple=False).squeeze(-1)
                if fusion_indices.numel() > 0:
                    temp = self.config.temperature
                    teacher_scaled = fusion_teacher_logits[fusion_indices] / temp
                    teacher_probs = F.softmax(teacher_scaled.detach(), dim=-1)
                    student_pred_text = text_modal.logits.argmax(dim=-1)
                    student_pred_vision = vision_modal.logits.argmax(dim=-1)
                    if self.config.lambda_fusion_to_text > 0:
                        text_wrong = (student_pred_text[fusion_indices] != labels[fusion_indices]).nonzero(as_tuple=False).squeeze(-1)
                        if text_wrong.numel() > 0:
                            idx = fusion_indices[text_wrong]
                            student_log_probs_text = F.log_softmax(text_modal.logits[idx] / temp, dim=-1)
                            teacher_probs_text = teacher_probs[text_wrong]
                            kl_text = (
                                F.kl_div(
                                    student_log_probs_text,
                                    teacher_probs_text,
                                    reduction="batchmean",
                                )
                                * (temp ** 2)
                            )
                            fusion_distill_loss = fusion_distill_loss + self.config.lambda_fusion_to_text * kl_text
                    if self.config.lambda_fusion_to_vision > 0:
                        vision_wrong = (student_pred_vision[fusion_indices] != labels[fusion_indices]).nonzero(as_tuple=False).squeeze(-1)
                        if vision_wrong.numel() > 0:
                            idx = fusion_indices[vision_wrong]
                            student_log_probs_vision = F.log_softmax(vision_modal.logits[idx] / temp, dim=-1)
                            teacher_probs_vision = teacher_probs[vision_wrong]
                            kl_vision = (
                                F.kl_div(
                                    student_log_probs_vision,
                                    teacher_probs_vision,
                                    reduction="batchmean",
                                )
                                * (temp ** 2)
                            )
                            fusion_distill_loss = fusion_distill_loss + self.config.lambda_fusion_to_vision * kl_vision
                    fusion_pairs = torch.tensor(fusion_indices.numel(), device=device, dtype=torch.long)
                distill["fusion_loss"] = fusion_distill_loss
                distill["fusion_pairs"] = fusion_pairs
                distill["loss"] = distill["loss"] + fusion_distill_loss
                distill["num_pairs"] = distill["num_pairs"] + fusion_pairs

        if self.config.event_student_focus_enabled or self.config.event_reweight_enabled:
            self._update_student_stats(labels, fusion_student_pred, meta)

        total_loss = (
            ce_fusion
            + self.config.alpha * (ce_text + ce_vision)
            + self.config.beta * (loss_evi_text + loss_evi_vision)
            + gamma_scale * distill["loss"]
            + self.config.consistency_lambda * consistency_loss
            + self.config.weak_teacher_lambda * weak_teacher_lambda_factor * weak_teacher_loss
        )
        if trace_this:
            print(
                f"[trace] total loss={total_loss.item():.4f} (gamma_scale={gamma_scale:.4f})",
                flush=True,
            )

        total_loss.backward()
        if trace_this:
            print("[trace] backward complete", flush=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        self._update_ema()
        if self.scheduler is not None and self.scheduler_type_lower not in {"reduce_on_plateau"}:
            self.scheduler.step()
        if trace_this:
            print("[trace] optimizer/scheduler step complete", flush=True)

        stats = {
            "loss_total": total_loss.item(),
            "loss_fusion": ce_fusion.item(),
            "loss_cls_text": ce_text.item(),
            "loss_cls_vision": ce_vision.item(),
            "loss_evi_text": loss_evi_text.item(),
            "loss_evi_vision": loss_evi_vision.item(),
            "loss_distill": distill["loss"].item() * float(gamma_scale > 0),
            "num_distill_pairs": distill["num_pairs"].item(),
            "loss_distill_fusion": distill.get("fusion_loss", torch.tensor(0.0, device=device)).item()
            * float(gamma_scale > 0),
            "num_fusion_pairs": distill.get("fusion_pairs", torch.tensor(0, device=device, dtype=torch.long)).item(),
            "fallback_pairs": distill.get("fallback_pairs", torch.tensor(0, device=device, dtype=torch.long)).item(),
            "loss_consistency": (self.config.consistency_lambda * consistency_loss).item()
            if self.config.consistency_enabled and self.config.consistency_lambda > 0
            else 0.0,
            "consistency_pairs": consistency_pairs.item(),
            "loss_weak_teacher": (
                self.config.weak_teacher_lambda * weak_teacher_lambda_factor * weak_teacher_loss
            ).item()
            if self.config.weak_teacher_enabled and self.config.weak_teacher_lambda > 0
            else 0.0,
            "weak_teacher_pairs": weak_teacher_pairs.item(),
            "weak_teacher_lambda_factor": weak_teacher_lambda_factor,
        }

        self.global_step += 1
        return stats

    def _compute_event_scores(
        self,
        labels: torch.Tensor,
        metric: str,
        min_seen: int,
        default_score: float,
        event_ids: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        event_accs: Optional[torch.Tensor],
        event_confs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = torch.full_like(labels, float(default_score), dtype=torch.float32)
        if event_ids is None or valid_mask is None:
            return scores
        metric_lower = (metric or "min").lower()
        min_seen = max(min_seen, 1)
        for idx_sample in range(labels.size(0)):
            if not valid_mask[idx_sample]:
                continue
            eid = int(event_ids[idx_sample].item())
            stats = self.event_teacher_stats[eid]
            seen = stats["seen"]
            if seen < min_seen:
                score_val = default_score
            else:
                if event_accs is not None:
                    acc_val = float(event_accs[idx_sample].item())
                else:
                    acc_val = stats["correct"] / max(seen, 1)
                if event_confs is not None:
                    conf_val = float(event_confs[idx_sample].item())
                else:
                    conf_val = stats["conf_sum"] / max(seen, 1)
                if metric_lower == "acc":
                    score_val = acc_val
                elif metric_lower == "conf":
                    score_val = conf_val
                elif metric_lower == "avg":
                    score_val = 0.5 * (acc_val + conf_val)
                else:
                    score_val = min(acc_val, conf_val)
            scores[idx_sample] = float(score_val)
        return scores

    def update_epoch(self, epoch: int, total_epochs: int) -> None:
        if total_epochs > 0:
            self.current_epoch_fraction = (epoch + 1) / total_epochs
        else:
            self.current_epoch_fraction = None

    def _update_ema(self) -> None:
        if self.ema_model is None:
            return
        decay = self.config.ema_decay
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.core_model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
            for ema_buffer, buffer in zip(self.ema_model.buffers(), self.core_model.buffers()):
                ema_buffer.data.copy_(buffer.data)

    def _update_event_logits_ema(
        self,
        event_ids: torch.Tensor,
        logits_text: torch.Tensor,
        logits_vision: torch.Tensor,
        logits_fusion: torch.Tensor,
    ) -> None:
        decay = float(self.config.weak_teacher_event_ema_decay)
        for idx, eid_val in enumerate(event_ids):
            eid = int(eid_val.item())
            if eid < 0:
                continue
            info = self.event_logits_ema[eid]
            info["logits_fusion"] = self._ema_tensor(info["logits_fusion"], logits_fusion[idx], decay)

    @staticmethod
    def _ema_tensor(stored: Optional[torch.Tensor], new_value: torch.Tensor, decay: float) -> torch.Tensor:
        new_cpu = new_value.detach().cpu()
        if stored is None:
            return new_cpu
        decay = max(0.0, min(1.0, decay))
        return stored * decay + new_cpu * (1.0 - decay)

    @property
    def num_classes(self) -> int:
        return self.core_model.num_classes

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.core_model.state_dict()

    def ema_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.ema_model is None:
            return None
        return self.ema_model.state_dict()
