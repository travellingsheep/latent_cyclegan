from __future__ import annotations

import torch
import torch.nn.functional as F


def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake


def g_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def style_reconstruction_loss(pred_style: torch.Tensor, target_style: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_style, target_style)


def diversity_sensitive_loss(
    fake_1: torch.Tensor,
    fake_2: torch.Tensor,
    margin: float | None = None,
) -> torch.Tensor:
    """Diversity sensitive loss.

    Note: In training we subtract this term (i.e. encourage distance). When margin is set,
    we clamp the distance so once it exceeds the margin the gradient from this term becomes 0.
    """

    dist = F.l1_loss(fake_1, fake_2)
    if margin is None:
        return dist
    margin_v = float(margin)
    if margin_v <= 0.0:
        return dist
    return torch.clamp(dist, max=margin_v)


def cycle_consistency_loss(reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(reconstructed, original)


def r1_penalty(real_logits: torch.Tensor, real_inputs: torch.Tensor) -> torch.Tensor:
    gradients = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_inputs,
        create_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return gradients.pow(2).sum(dim=1).mean()
