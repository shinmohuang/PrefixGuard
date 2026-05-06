from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


def soft_target_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: list[tuple[str, str, str]],
    temperature: float,
) -> torch.Tensor:
    if features.size(0) < 2:
        return features.new_tensor(0.0)

    label_to_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        label_to_indices[label].append(index)

    valid_anchor_indices = [
        index
        for index, label in enumerate(labels)
        if len(label_to_indices[label]) > 1
    ]
    if not valid_anchor_indices:
        return features.new_tensor(0.0)

    normalized = F.normalize(features, dim=-1)
    similarity = normalized @ normalized.T / temperature
    similarity = similarity - torch.eye(
        similarity.size(0),
        device=similarity.device,
        dtype=similarity.dtype,
    ) * 1e9
    log_prob = similarity - torch.logsumexp(similarity, dim=-1, keepdim=True)

    losses = []
    for anchor in valid_anchor_indices:
        positive_indices = [
            index
            for index in label_to_indices[labels[anchor]]
            if index != anchor
        ]
        positive_log_prob = log_prob[anchor, positive_indices]
        losses.append(-positive_log_prob.mean())
    return torch.stack(losses).mean()


def compactness_loss(
    probs: torch.Tensor,
    marginal_weight: float = 1.0,
) -> torch.Tensor:
    probs = probs.clamp_min(1e-8)
    per_sample_entropy = -(probs * probs.log()).sum(dim=-1).mean()
    marginal = probs.mean(dim=0).clamp_min(1e-8)
    marginal_entropy = -(marginal * marginal.log()).sum()
    return per_sample_entropy - marginal_weight * marginal_entropy
