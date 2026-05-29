from __future__ import annotations

import torch
import torch.nn.functional as F


def cox_loss(pred, data, model, lambda_1: float) -> torch.Tensor:
    theta = pred.view(-1)
    risk_matrix = (data.deadtime.unsqueeze(1) <= data.deadtime.unsqueeze(0)).float()
    exp_theta = torch.exp(theta)
    partial = theta - torch.log(torch.sum(exp_theta * risk_matrix, dim=1))
    observed_events = data.dead.sum().clamp_min(1.0)
    negative_log_likelihood = -torch.sum(partial * data.dead) / observed_events
    l1_reg = sum(torch.abs(parameter).sum() for parameter in model.parameters())
    return negative_log_likelihood + lambda_1 * l1_reg


def compute_jsd_batch(p, q, epsilon=1e-8):
    mean_distribution = 0.5 * (p + q)
    log_p = torch.log(p + epsilon)
    log_q = torch.log(q + epsilon)
    log_m = torch.log(mean_distribution + epsilon)
    return 0.5 * ((p * (log_p - log_m)).sum(-1) + (q * (log_q - log_m)).sum(-1))


def contrastive_jsd_loss(share, distinct, tau=0.07):
    batch_size, hidden_dim, num_views = share.shape
    device = share.device

    share_prob = F.softmax(share, dim=1)
    distinct_prob = F.softmax(distinct, dim=1)

    all_share = share_prob.permute(0, 2, 1).reshape(batch_size * num_views, hidden_dim)
    all_distinct = distinct_prob.permute(0, 2, 1).reshape(batch_size * num_views, hidden_dim)

    pos_jsd = compute_jsd_batch(all_share.unsqueeze(1), all_share.unsqueeze(0))
    mask_same_instance = torch.block_diag(*[torch.ones(num_views, num_views, device=device)] * batch_size)
    mask_different_view = ~torch.eye(num_views, device=device).repeat(batch_size, batch_size).bool()
    pos_mask = (mask_same_instance * mask_different_view).bool()

    neg_jsd = compute_jsd_batch(all_share.unsqueeze(1), all_distinct.unsqueeze(0))
    neg_mask = torch.eye(batch_size * num_views, device=device).bool()

    pos_sim = torch.exp(-pos_jsd / tau) * pos_mask
    neg_sim = torch.exp(-neg_jsd / tau) * neg_mask

    pos_sum = pos_sim.sum(dim=1).reshape(batch_size, num_views).sum(dim=1).repeat_interleave(num_views)
    neg_sum = neg_sim.sum(dim=1)
    losses = -torch.log(pos_sum / (pos_sum + neg_sum))
    return losses.view(batch_size, num_views).mean(dim=1).mean()
