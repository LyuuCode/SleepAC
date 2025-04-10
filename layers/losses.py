import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets) # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss) # 计算预测的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss # 根据Focal Loss公式计算Focal Loss
        return focal_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=10):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, labels, temperature=0.2):
        device = inputs.device
        B, T = inputs.size(0), inputs.size(1)
        if T == 1:
            return inputs.new_tensor(0.)

        # 正则化输入
        inputs = F.normalize(inputs, p=2, dim=1)
        # 计算相似度
        sim = (torch.mm(inputs, inputs.T) / temperature).to(device)  # B x B
        # 计算 log softmax
        logits = F.log_softmax(sim, dim=-1).to(device)

        # 生成正样本 mask
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
        diag_indices = torch.arange(pos_mask.size(0))
        pos_mask[diag_indices, diag_indices] = 0

        # 生成负样本 mask
        neg_mask = torch.zeros((labels.size(0), labels.size(0)), dtype=int).to(device)
        category_pairs = [(0, 1), (0, 2), (2, 3), (1, 4), (1, 2), (2, 4)]
        category_pairs_set = set(category_pairs)
        labels_i = labels.unsqueeze(0)  # 形状为 (1, n)
        labels_j = labels.unsqueeze(1)  # 形状为 (n, 1)

        for pair in category_pairs_set:
            mask = ((labels_i == pair[0]) & (labels_j == pair[1])) | ((labels_i == pair[1]) & (labels_j == pair[0]))
            neg_mask[mask] = 1

        # 提取正样本和负样本的 logits
        pos_logits = logits * pos_mask
        neg_logits = logits * neg_mask

        # 计算正样本和负样本的损失
        pos_divisor = torch.sum(pos_mask, dim=-1).clamp(min=1e-8)
        neg_divisor = torch.sum(neg_mask, dim=-1).clamp(min=1e-8)

        pos_loss = -torch.sum(pos_logits, dim=-1) / pos_divisor
        neg_loss = -torch.sum(neg_logits, dim=-1) / neg_divisor

        pos_loss = pos_loss / pos_loss.max()
        neg_loss = neg_loss / neg_loss.max()
        # print(pos_loss.mean().item(), neg_loss.mean().item())
        neg_loss = 1 - neg_loss
        # neg_max = torch.max(neg_loss).item()
        # neg_loss = (neg_max-neg_loss).clamp(min=1e-8)
        # print(pos_loss.mean().item(), neg_loss.mean().item())
        loss = torch.mean(pos_loss + neg_loss)
        # loss = neg_loss.mean()
        return loss


# B = 7
# N = 128  # 输入的大小
# inputs = torch.randn(B, N, requires_grad=True)
# labels = torch.tensor([0, 1, 2, 2, 0, 1, 1])  # 随机生成一些标签
# contrastive_loss(inputs, labels)
