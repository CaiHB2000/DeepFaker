import torch

from dynamic_distill.src.losses.distillation import compute_dynamic_distillation


def test_dynamic_teacher_selection():
    logits_text = torch.tensor([[2.0, -1.0], [0.1, 0.2]])
    logits_vision = torch.tensor([[0.5, -0.2], [3.0, -2.0]])
    pen_text = torch.randn(2, 4)
    pen_vision = torch.randn(2, 4)
    u_text = torch.tensor([[0.2], [0.4]])
    u_vision = torch.tensor([[0.6], [0.1]])

    losses = compute_dynamic_distillation(
        logits_text=logits_text,
        logits_vision=logits_vision,
        penultimate_text=pen_text,
        penultimate_vision=pen_vision,
        uncertainty_text=u_text,
        uncertainty_vision=u_vision,
        temperature=1.0,
        lambda_kl=1.0,
        lambda_feat=0.0,
        delta=0.05,
    )

    mask_text_teacher = losses["mask_text_teacher"]
    mask_vision_teacher = losses["mask_vision_teacher"]

    assert mask_text_teacher.tolist() == [True, False]
    assert mask_vision_teacher.tolist() == [False, True]
    assert losses["num_pairs"].item() == 2
    assert losses["kl_loss"] > 0
    assert losses["loss"] > 0
