# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm  # 可选，用于进度条显示（非必需，但提升体验）




def train_one_epoch(lr_scheduler, model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    num_steps = len(data_loader)
    idx = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        '''
        使用 for 循环遍历数据加载器中的每个批次。metric_logger.log_every 会在每 print_freq 个批次打印一次日志信息，header 是日志的头部信息。
        samples：当前批次的输入样本。
        targets：当前批次的目标标签。
        '''

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        # lr_scheduler.step_update(epoch * num_steps + idx)
        # idx += 1

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #在分布式训练中，同步所有进程中的指标信息
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    #返回一个字典，包含所有指标的全局平均值


# @torch.no_grad()
# def evaluate(data_loader, model, device):
#     criterion = torch.nn.CrossEntropyLoss()
#
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'
#
#     # switch to evaluation mode
#     model.eval()
#
#     for images, target in metric_logger.log_every(data_loader, 10, header):
#         #每 10 个批次打印一次进度日志（如 Test: [10/100] Loss: 0.321）
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)
#
#         # compute output
#         with torch.cuda.amp.autocast():
#             output = model(images)
#             loss = criterion(output, target)
#
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#
#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
#
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}







@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    # 新增：用于存储所有真实标签和预测标签
    all_targets = []
    all_preds = []

    metric_logger.init_binary_metrics()  # 新增：显式初始化灵敏度/特异性指标
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # 计算预测类别（二分类中为0或1）
        pred = output.argmax(dim=1)
        #在 dim=1（类别维度）上，对每个样本的所有类别得分取最大值的 索引，返回形状为 (batch_size,) 的一维张量，每个元素是对应样本的预测类别索引

        # 保存到全局列表（转换为CPU上的numpy数组）
        all_targets.append(target.cpu())
        all_preds.append(pred.cpu())

        acc1= accuracy(output, target, topk=(1,))[0]
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # 合并所有批次的标签（避免内存碎片）
    target = torch.cat(all_targets)
    pred = torch.cat(all_preds)
    metric_logger.update_binary_metrics(target, pred)  # 调用类内方法更新指标


    # 同步多进程统计量（分布式环境）
    metric_logger.synchronize_between_processes()

    # 打印新增指标（在原有日志基础上追加）
    # 打印新增指标（在原有日志基础上追加）
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}  Sensitivity: {sen:.4f}  Specificity: {spe:.4f}'
          .format(
              top1=metric_logger.acc1,
              losses=metric_logger.loss,
              sen=metric_logger.sensitivity.global_avg,
              spe=metric_logger.specificity.global_avg
          ))

    # 返回结果包含新增指标
    results = {
        k: meter.global_avg for k, meter in metric_logger.meters.items()
    }
    return results


