import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):
    """
    AM-Softmax（Additive Margin Softmax）损失函数。
    """

    def __init__(self, in_dim: int, num_cls: int = 2, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(num_cls, in_dim))

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        W = F.normalize(self.W, dim=1)
        x = F.normalize(embeds, dim=1)
        cos = (x @ W.T).clamp(-1 + 1e-7, 1 - 1e-7)
        one_hot = F.one_hot(labels, num_classes=self.W.shape[0]).float()
        cos_m = cos - self.m * one_hot
        return F.cross_entropy(self.s * cos_m, labels)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(embeds, dim=1)
        sim = (z @ z.T) / self.temp
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.T).float()
        mask_self = torch.eye(z.size(0), device=z.device)
        mask_pos = mask_pos - mask_self

        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values)
        denom = (exp_sim * (1 - mask_self)).sum(dim=1, keepdim=True)
        log_prob = sim - torch.log(denom + 1e-8)
        loss = -(mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
        return loss.mean()