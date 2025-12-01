from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _kl_div(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    temperature: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    temp = temperature.unsqueeze(-1)
    scaled_teacher = logits_teacher / temp
    scaled_student = logits_student / temp

    teacher_probs = F.softmax(scaled_teacher, dim=-1)
    student_log_probs = F.log_softmax(scaled_student, dim=-1)
    kl_per_sample = (teacher_probs * (torch.log(teacher_probs + 1e-8) - student_log_probs)).sum(dim=-1)
    kl_per_sample = kl_per_sample * (temperature ** 2)
    if weights is not None:
        weights = weights.to(kl_per_sample.device)
        weighted_sum = (kl_per_sample * weights).sum()
        denom = weights.sum().clamp_min(1.0)
        return weighted_sum / denom
    return kl_per_sample.mean()


def _feature_mse(
    feats_teacher: torch.Tensor,
    feats_student: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = (feats_student - feats_teacher.detach()) ** 2
    mse_per_sample = diff.view(diff.size(0), -1).mean(dim=-1)
    if weights is not None:
        weights = weights.to(mse_per_sample.device)
        weighted_sum = (mse_per_sample * weights).sum()
        denom = weights.sum().clamp_min(1.0)
        return weighted_sum / denom
    return mse_per_sample.mean()


def compute_dynamic_distillation(
    logits_text: torch.Tensor,
    logits_vision: torch.Tensor,
    penultimate_text: torch.Tensor,
    penultimate_vision: torch.Tensor,
    uncertainty_text: torch.Tensor,
    uncertainty_vision: torch.Tensor,
    labels: torch.Tensor | None = None,
    temperature: float = 2.0,
    lambda_kl: float = 1.0,
    lambda_feat: float = 0.0,
    delta: float = 0.05,
    teacher_logits_text: torch.Tensor | None = None,
    teacher_logits_vision: torch.Tensor | None = None,
    teacher_penultimate_text: torch.Tensor | None = None,
    teacher_penultimate_vision: torch.Tensor | None = None,
    teacher_probs_text: torch.Tensor | None = None,
    teacher_probs_vision: torch.Tensor | None = None,
    adaptive_temperature: bool = False,
    temperature_base: float = 2.0,
    temperature_coeff: float = 0.0,
    use_confidence_gate: bool = False,
    confidence_margin: float = 0.0,
    eligible_mask: torch.Tensor | None = None,
    require_student_mistake: bool = True,
    agreement_confidence_gap: float = 0.0,
    positive_distill_boost: float = 1.0,
    positive_student_conf_margin: float = 0.0,
    positive_focus_mask: torch.Tensor | None = None,
    uncertainty_weight_enabled: bool = False,
    uncertainty_weight_scale: float = 0.0,
    uncertainty_weight_power: float = 1.0,
    uncertainty_weight_clip: float | None = None,
    override_mask_text: torch.Tensor | None = None,
    override_mask_vision: torch.Tensor | None = None,
    override_weights_text: torch.Tensor | None = None,
    override_weights_vision: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Computes dynamic teacher-student distillation losses."""
    if uncertainty_text.dim() > 1:
        u_text = uncertainty_text.squeeze(-1)
    else:
        u_text = uncertainty_text
    if uncertainty_vision.dim() > 1:
        u_vision = uncertainty_vision.squeeze(-1)
    else:
        u_vision = uncertainty_vision

    mask_text_teacher = (u_text + delta < u_vision)
    mask_vision_teacher = (u_vision + delta < u_text)

    if override_mask_text is not None:
        mask_text_teacher = override_mask_text.to(mask_text_teacher.device)
    if override_mask_vision is not None:
        mask_vision_teacher = override_mask_vision.to(mask_vision_teacher.device)

    teacher_logits_text = teacher_logits_text if teacher_logits_text is not None else logits_text
    teacher_logits_vision = teacher_logits_vision if teacher_logits_vision is not None else logits_vision
    teacher_pred_text = teacher_logits_text.argmax(dim=-1)
    teacher_pred_vision = teacher_logits_vision.argmax(dim=-1)
    student_pred_text = logits_text.argmax(dim=-1)
    student_pred_vision = logits_vision.argmax(dim=-1)
    teacher_penultimate_text = (
        teacher_penultimate_text if teacher_penultimate_text is not None else penultimate_text
    )
    teacher_penultimate_vision = (
        teacher_penultimate_vision if teacher_penultimate_vision is not None else penultimate_vision
    )

    losses = {
        "kl_loss": torch.tensor(0.0, device=logits_text.device),
        "feature_loss": torch.tensor(0.0, device=logits_text.device),
        "num_pairs": torch.tensor(0, device=logits_text.device, dtype=torch.long),
        "mask_text_teacher": mask_text_teacher,
        "mask_vision_teacher": mask_vision_teacher,
    }

    kl_total = torch.tensor(0.0, device=logits_text.device)
    feat_total = torch.tensor(0.0, device=logits_text.device)
    pairs = 0

    teacher_conf_text = torch.softmax(teacher_logits_text.detach(), dim=-1).max(dim=-1).values
    teacher_conf_vision = torch.softmax(teacher_logits_vision.detach(), dim=-1).max(dim=-1).values
    student_conf_text = torch.softmax(logits_text.detach(), dim=-1).max(dim=-1).values
    student_conf_vision = torch.softmax(logits_vision.detach(), dim=-1).max(dim=-1).values

    if eligible_mask is not None:
        eligible_mask = eligible_mask.to(mask_text_teacher.device)
        mask_text_teacher = mask_text_teacher & eligible_mask
        mask_vision_teacher = mask_vision_teacher & eligible_mask

    def _apply_uncertainty_weight(
        base: torch.Tensor,
        gap: torch.Tensor,
    ) -> torch.Tensor:
        if not uncertainty_weight_enabled or uncertainty_weight_scale <= 0:
            return base
        adjusted = torch.clamp(gap, min=0.0)
        if uncertainty_weight_power != 1.0:
            power = max(uncertainty_weight_power, 0.0)
            adjusted = adjusted ** power
        adjusted = 1.0 + uncertainty_weight_scale * adjusted
        if uncertainty_weight_clip is not None and uncertainty_weight_clip > 0:
            adjusted = torch.clamp(adjusted, max=uncertainty_weight_clip)
        return base * adjusted.to(base.device)

    if mask_text_teacher.any():
        idx = mask_text_teacher.clone()
        if positive_focus_mask is not None:
            idx = idx & positive_focus_mask.to(idx.device)
        if use_confidence_gate and teacher_probs_text is not None:
            confidence_gap = teacher_probs_text[:, 0] - teacher_probs_text[:, 1]
            idx = idx & (confidence_gap >= confidence_margin)
        if labels is not None:
            idx = idx & (teacher_pred_text == labels)
            if require_student_mistake:
                idx = idx & (student_pred_vision != labels)
            else:
                neg_mask = labels == 0
                idx = idx & (~neg_mask | (student_pred_vision != labels))
                if positive_student_conf_margin > 0:
                    pos_mask = labels == 1
                    pos_condition = (student_pred_vision != labels) | (
                        student_conf_vision <= positive_student_conf_margin
                    )
                    idx = idx & (~pos_mask | pos_condition)
                if agreement_confidence_gap > 0:
                    confidence_delta = teacher_conf_text - student_conf_vision
                    idx = idx & (confidence_delta >= agreement_confidence_gap)

        if idx.any():
            temp_vec = torch.full_like(uncertainty_text[idx].squeeze(-1), temperature)
            if adaptive_temperature:
                delta_u = (uncertainty_vision[idx] - uncertainty_text[idx]).squeeze(-1).clamp_min(0.0)
                temp_vec = temperature_base + temperature_coeff * delta_u
            weights = torch.ones_like(temp_vec)
            weights = _apply_uncertainty_weight(
                weights,
                (uncertainty_vision[idx] - uncertainty_text[idx]).squeeze(-1),
            )
            if labels is not None and positive_distill_boost != 1.0:
                label_subset = labels[idx]
                boost_value = torch.full_like(weights, positive_distill_boost)
                weights = torch.where(label_subset == 1, boost_value, weights)
            if override_weights_text is not None:
                weights = weights * override_weights_text[idx]
            kl_total = kl_total + _kl_div(
                logits_teacher=teacher_logits_text[idx].detach(),
                logits_student=logits_vision[idx],
                temperature=temp_vec,
                weights=weights,
            )
            if lambda_feat > 0:
                feat_total = feat_total + _feature_mse(
                    feats_teacher=teacher_penultimate_text[idx],
                    feats_student=penultimate_vision[idx],
                    weights=weights,
                )
            pairs += idx.sum()
            mask_text_teacher = idx
        else:
            mask_text_teacher = idx
        
    if mask_vision_teacher.any():
        idx = mask_vision_teacher.clone()
        if positive_focus_mask is not None:
            idx = idx & positive_focus_mask.to(idx.device)
        if use_confidence_gate and teacher_probs_vision is not None:
            confidence_gap = teacher_probs_vision[:, 0] - teacher_probs_vision[:, 1]
            idx = idx & (confidence_gap >= confidence_margin)
        if labels is not None:
            idx = idx & (teacher_pred_vision == labels)
            if require_student_mistake:
                idx = idx & (student_pred_text != labels)
            else:
                neg_mask = labels == 0
                idx = idx & (~neg_mask | (student_pred_text != labels))
                if positive_student_conf_margin > 0:
                    pos_mask = labels == 1
                    pos_condition = (student_pred_text != labels) | (
                        student_conf_text <= positive_student_conf_margin
                    )
                    idx = idx & (~pos_mask | pos_condition)
                if agreement_confidence_gap > 0:
                    confidence_delta = teacher_conf_vision - student_conf_text
                    idx = idx & (confidence_delta >= agreement_confidence_gap)

        if idx.any():
            temp_vec = torch.full_like(uncertainty_vision[idx].squeeze(-1), temperature)
            if adaptive_temperature:
                delta_u = (uncertainty_text[idx] - uncertainty_vision[idx]).squeeze(-1).clamp_min(0.0)
                temp_vec = temperature_base + temperature_coeff * delta_u
            weights = torch.ones_like(temp_vec)
            weights = _apply_uncertainty_weight(
                weights,
                (uncertainty_text[idx] - uncertainty_vision[idx]).squeeze(-1),
            )
            if labels is not None and positive_distill_boost != 1.0:
                label_subset = labels[idx]
                boost_value = torch.full_like(weights, positive_distill_boost)
                weights = torch.where(label_subset == 1, boost_value, weights)
            if override_weights_vision is not None:
                weights = weights * override_weights_vision[idx]
            kl_total = kl_total + _kl_div(
                logits_teacher=teacher_logits_vision[idx].detach(),
                logits_student=logits_text[idx],
                temperature=temp_vec,
                weights=weights,
            )
            if lambda_feat > 0:
                feat_total = feat_total + _feature_mse(
                    feats_teacher=teacher_penultimate_vision[idx],
                    feats_student=penultimate_text[idx],
                    weights=weights,
                )
            pairs += idx.sum()
            mask_vision_teacher = idx
        else:
            mask_vision_teacher = idx

    losses["kl_loss"] = kl_total * lambda_kl
    losses["feature_loss"] = feat_total * lambda_feat
    losses["num_pairs"] = torch.as_tensor(pairs, device=logits_text.device, dtype=torch.long)
    losses["loss"] = losses["kl_loss"] + losses["feature_loss"]
    return losses
