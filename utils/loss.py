import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
import random

def sim_loss(fea_a, fea_v, S):
    fea_a = F.normalize(fea_a, dim=1)
    fea_v = F.normalize(fea_v, dim=1)
    Omega = 0.5 * fea_a @ fea_v.T
    loss_sim = - torch.mean(S * Omega - torch.log1p(torch.exp(Omega)))
    return loss_sim

def triplet_cosine_loss(out_1, out_2):
    bz = out_1.size(0)
    out_1 = out_1 / torch.norm(out_1, dim=-1, keepdim=True)
    out_2 = out_2 / torch.norm(out_2, dim=-1, keepdim=True)
    scores = out_1.mm(out_2.T)
    # print(scores)
    # compute image-sentence score matrix
    diagonal = scores.diag().view(bz, 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column: caption retrieval
    cost_s = (1 + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row: image retrieval
    cost_im = (1 + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > 0.5
    mask = mask.to(cost_s.device)
    cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

    # maximum and mean
    # cost_s_max, cost_im_max = cost_s.max(1)[0], cost_im.max(0)[0]
    cost_s_mean, cost_im_mean = cost_s.mean(1), cost_im.mean(0)

    return cost_s_mean.sum() + cost_im_mean.sum()


def exp_map_poincare(a, c=1.0, z=None, eps=1e-5):
    """
    映射欧几里得向量 a 到双曲空间 D^n 中，使用 Poincare ball 模型。

    参数:
        a: Tensor, 欧几里得输入向量，形状为 [B, D]
        c: float, 曲率常数 > 0
        z: 参考点（默认使用原点）
        eps: 避免除零的小值

    返回:
        双曲空间中的嵌入，形状仍为 [B, D]
    """
    # 默认参考点 z = 0
    if z is None:
        z = torch.zeros_like(a)
    c = torch.tensor(c, dtype=a.dtype, device=a.device)
    norm_a = torch.norm(a, dim=-1, keepdim=True).clamp(min=eps)  # [B, 1]
    lambda_z = 2. / (1. - c * (z ** 2).sum(dim=-1, keepdim=True).clamp(min=eps))  # conformal factor

    # 计算 Möbius 加法的部分项
    scaled = torch.tanh(torch.sqrt(c) * lambda_z * norm_a / 2) * (a / (torch.sqrt(c) * norm_a))

    # Möbius 加法（z + scaled），这里 z 默认为 0，则就是 scaled
    return z + scaled  # 如果 z=0，可以省略

def compute_pfc_loss(fv, fm):
    """
    计算 PFC 损失，fv 和 fm 为两个模态的双曲嵌入，形状都为 [B, D]

    参数:
        fv: Tensor, 第一个模态（如视觉）特征 [B, D]
        fm: Tensor, 第二个模态（如文本）特征 [B, D]

    返回:
        pfc_loss: Tensor, 标量损失
    """
    # 点积： [B]
    sim_scores = (fv * fm).sum(dim=-1)

    # sigmoid 然后取 log
    log_sigmoid_sim = torch.log(torch.sigmoid(sim_scores) + 1e-6)  # 加 1e-6 避免 log(0)

    # 负号 + 平均
    loss = -log_sigmoid_sim.mean()
    return loss

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def metric_loss(feas, y):
    """
    calcaute feature metric loss for representations
    """
    y_np = y.detach().cpu().numpy()
    cls_num = np.unique(y_np)
    query_list = []
    support_list = []
    for c in cls_num:
        idx = np.where(y_np == c)[0]
        sel_idx = random.choice(idx)
        rest_idx = [id for id in idx if id != sel_idx]
        query_list.append(feas[sel_idx])
        support_list.append(feas[rest_idx].mean(0))

    labels = torch.arange(len(cls_num)).long().cuda()
    query_list, support_list = torch.stack(query_list), torch.stack(support_list)
    logits = euclidean_metric(query_list, support_list)
    loss = F.cross_entropy(logits, labels)
    return loss


class FocalLoss(nn.Module):
    """
    Implementation of focal loss
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target!=0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0,select.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)** self.gamma * logpt
        
        return loss

def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon= 1e-20):

    N, C = predict_prob.shape
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob * torch.log(predict_prob + epsilon) - (1 - predict_prob)*torch.log(1- predict_prob + epsilon)

    return entropy.sum(1)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)