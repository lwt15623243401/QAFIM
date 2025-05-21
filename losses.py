# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
import torch.nn as nn

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


















class FocalLoss(nn.Module):
    r"""
    Focal Loss for multi-class classification.

    公式来源：
        Lin et al., Focal Loss for Dense Object Detection (ICCV 2017):
        FL(p_t) = - (1 - p_t)^gamma * log(p_t)               :contentReference[oaicite:0]{index=0}

    参数:
        weight (Tensor, optional): 手动为每个类别设置的权重 (C,)；对应于
            `torch.nn.CrossEntropyLoss(weight=...)` 中的 weight。 :contentReference[oaicite:1]{index=1}
        gamma (float, optional): Focal Loss 的聚焦参数，常用默认值 2.0。 :contentReference[oaicite:2]{index=2}
        ignore_index (int, optional): 忽略某个类别索引，不计入损失。 :contentReference[oaicite:3]{index=3}
        reduction (str, optional): 指定如何聚合损失，可选 `'none' | 'mean' | 'sum'`，
            行为同 `CrossEntropyLoss`。 :contentReference[oaicite:4]{index=4}
        label_smoothing (float, optional): 标签平滑参数，0 表示不平滑。 :contentReference[oaicite:5]{index=5}
    """
    __constants__ = ['weight', 'ignore_index', 'reduction', 'label_smoothing', 'gamma']

    def __init__(self,
                 weight: torch.Tensor = torch.tensor([1.0, 10.0]), # 类别权重，形状为 (C,)，C是类别数
                 gamma: float = 2.0, # 聚焦参数，控制易例权重衰减速度
                 ignore_index: int = -100,  # 忽略的类别索引（如背景类）
                 reduction: str = 'mean',  # 损失聚合方式：'none'/'mean'/'sum'
                 label_smoothing: float = 0.0):  # 标签平滑强度（0~1）
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input:  (N, C, ...) 未经过 Softmax 的 logits
            target: (N, ...)            LongTensor，每个值在 [0, C-1] 或 == ignore_index
        Returns:
            标量损失（若 reduction != 'none'），或与 input 同形的逐元素损失。
        """
        # 1. 先计算普通的交叉熵（带 weight、ignore_index、label_smoothing）
        ce_loss = F.cross_entropy(
            input, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing
        )  # :contentReference[oaicite:6]{index=6}

        # 2. 计算 p_t = exp(-CE)
        pt = torch.exp(-ce_loss)

        # 3. 应用聚焦因子 (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce_loss

        # 4. 根据 reduction 决定输出
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss











class CoulombLoss(nn.Module):
    r"""
    基于库仑定律的斥力损失函数

    参数:
        reduction (str, optional): 指定如何聚合损失，可选 `'none' | 'mean' | 'sum'`。
    """
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, K, 2) 锚点坐标
        Returns:
            标量损失（若 reduction != 'none'），或与输入同形的逐元素损失。
        """
        b, k, _ = coords.shape
        loss = 0
        for i in range(k):
            for j in range(i + 1, k):
                dist = torch.norm(coords[:, i] - coords[:, j], dim=1)
                loss += 1 / (dist + 1e-6)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss