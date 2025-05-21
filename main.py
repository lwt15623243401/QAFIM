# modified from https://github.com/facebookresearch/deit
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
'''
NativeScaler：用于自动混合精度训练，可在不损失太多精度的前提下加速训练过程。
get_state_dict：用于获取模型的状态字典，常用于保存和加载模型的参数。
ModelEma：指数移动平均（EMA）模型，通过对模型参数进行指数加权平均，可提升模型的泛化能力。
'''

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
import utils
import os

#import models
from RMT import RMT_T, RMT_S, RMT_B, RMT_L

archs = {
            'RMT_T': RMT_T,
            'RMT_S': RMT_S,
            'RMT_B': RMT_B,
            'RMT_L': RMT_L
         }

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    #argparse.ArgumentParser 是 argparse 模块中用于创建命令行参数解析器的类   DeiT（数据高效图像变换器）的训练与评估脚本
    parser.add_argument('--early-conv', action='store_true')
    parser.add_argument('--conv-pos', action='store_true')
    parser.add_argument('--use-ortho', action='store_true')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    #Drop path 是一种用于随机丢弃网络中某些路径的技术，类似于 Dropout，但作用于网络的路径级别。在训练过程中，以指定的概率随机丢弃某些层或模块的输出，从而提高模型的泛化能力。

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    #model-ema 表示是否使用模型的指数移动平均（Exponential Moving Average）。EMA 是一种对模型参数进行平滑处理的技术，通过对模型参数的历史值进行加权平均，得到一个更稳定的模型参数。在训练过程中，EMA 模型通常比原始模型在测试集上有更好的泛化性能。
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    #模型 EMA 的衰减率。在计算 EMA 时，会根据当前模型参数和之前的 EMA 参数进行加权平均，衰减率决定了当前参数和历史参数的权重分配。衰减率越接近 1，历史参数的权重越大，EMA 模型的更新就越缓慢；反之，衰减率越接近 0，当前参数的权重越大，EMA 模型的更新就越迅速
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    #是否强制将模型的 EMA 部分在 CPU 上运行。在某些情况下，由于内存或计算资源的限制，可能需要将 EMA 模型放在 CPU 上运行，以避免 GPU 内存不足的问题

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    #该参数用于指定优化器中一阶矩估计和二阶矩估计的指数衰减率。在 Adam 系列的优化器中，betas 是一个包含两个元素的元组 (beta1, beta2)，beta1 控制一阶矩估计（即梯度的移动平均）的衰减率，beta2 控制二阶矩估计（即梯度平方的移动平均）的衰减率
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    #该参数用于控制权重衰减的强度。权重衰减是一种正则化技术，它在损失函数中加入了一个与模型参数的平方和成正比的项，使得模型在训练过程中倾向于学习较小的权重，从而防止模型过拟合
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    #该参数用于设置随机梯度下降（SGD）优化器的动量值。动量是一种在优化过程中引入惯性的技术，它会考虑之前梯度的累积影响。具体来说，在更新参数时，当前的梯度不仅会影响参数的更新方向和步长，还会加上一个与之前累积梯度相关的项（乘以动量系数）。较大的动量值（如默认的0.9）会使优化器在更新参数时更倾向于沿着之前的方向移动，从而加快收敛速度，同时也有助于在平坦区域（梯度较小的地方）保持更新的稳定性，避免陷入局部最优解。
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    #含义：这是权重衰减（L2 正则化）的系数。在训练神经网络时，权重衰减通过在损失函数中添加一个与模型参数平方和成正比的项，来惩罚较大的权重值。其目的是防止模型过拟合，使模型的参数尽量小，从而让模型具有更好的泛化能力。例如，对于一个神经网络的参数W，损失函数L在加上权重衰减项后变为L' = L + weight_decay * ||W||^2，其中||W||^2是参数W的平方和。
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    #该参数用于指定学习率调度器（Learning Rate Scheduler）的类型。学习率调度器的作用是在训练过程中动态调整学习率的大小。默认的'cosine'表示使用余弦退火（Cosine Annealing）调度器，它会在训练过程中按照余弦函数的形状逐渐降低学习率，这种方法在许多情况下能够有效地平衡模型的收敛速度和最终性能。其他常见的学习率调度器类型可能还包括'step'（按固定步数降低学习率）等
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    #lr是学习率（Learning Rate）的缩写，它控制着模型在训练过程中参数更新的步长。较大的学习率可能会使模型在训练时收敛速度更快，但也更容易跳过最优解，导致模型无法收敛或出现振荡；较小的学习率则会使训练过程更加稳定，但收敛速度可能会很慢。因此，选择合适的学习率对于模型的训练效果至关重要。
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    #含义：该参数用于控制学习率噪声（Learning Rate Noise）的开启和关闭以及相关的 epoch 百分比设置。学习率噪声是一种在训练过程中对学习率进行随机扰动的技术，旨在增加模型的鲁棒性和泛化能力。当设置为具体的浮点数对时，例如[start_epoch_pct, end_epoch_pct]，表示在start_epoch_pct到end_epoch_pct（占总训练 epoch 的百分比）的范围内开启学习率噪声
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    #该参数指定学习率噪声的限制百分比。当开启学习率噪声时，学习率的随机扰动幅度会被限制在当前学习率的一定百分比范围内，这个百分比就是由--lr-noise-pct参数控制的。例如，当--lr-noise-pct为0.67时，学习率的随机变化幅度不会超过当前学习率的67%。
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    #该参数用于设置学习率噪声的标准差（Standard Deviation）。在对学习率进行随机扰动时，通常会使用正态分布来生成噪声值，而--lr-noise-std就是这个正态分布的标准差。较大的标准差会使学习率的扰动范围更大，从而增加学习率的变化程度；较小的标准差则会使学习率的变化相对较小，更加稳定。
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    #该参数指定在训练初期的热身学习率（Warmup Learning Rate）。在训练开始时，使用较小的学习率进行热身可以帮助模型更好地收敛，避免一开始就使用较大学习率导致模型参数更新不稳定。在热身阶段结束后，学习率会逐渐调整到由--lr参数指定的学习率或按照学习率调度器的规则进行调整。
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    #该参数用于设置学习率的下限。对于一些循环学习率调度器（如余弦退火等），在学习率下降到接近 0 时，为了防止学习率过低导致模型训练停滞不前，会设置一个最小学习率。当学习率下降到这个下限值时，就不会再继续下降，而是保持在这个最小值进行训练。
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    #该参数指定了学习率（LR，Learning Rate）衰减的 epoch 间隔。在训练过程中，学习率通常会随着训练的进行而逐渐减小，以帮助模型更好地收敛到最优解。--decay-epochs 表示每隔多少个 epoch 对学习率进行一次衰减操作。例如，如果 --decay-epochs 设置为 30，那么每经过 30 个 epoch，学习率就会按照相应的衰减规则进行调整
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    #该参数用于指定学习率热身阶段的 epoch 数。在训练开始时，通常会使用一个较小的学习率进行热身，让模型先在参数空间中找到一个较好的起始位置，然后再逐渐增大学习率进行正常训练。--warmup-epochs 就是用来控制这个热身阶段持续的 epoch 数量。例如，当 --warmup-epochs 设置为 5 时，模型会在前 5 个 epoch 内使用热身学习率（由 --warmup-lr 参数指定），之后再按照学习率调度器的规则调整学习率。
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    #该参数指定了在循环学习率调度器结束后，学习率在最小学习率（--min-lr 参数指定）下冷却的 epoch 数。当使用一些循环学习率调度器（如余弦退火等）时，在循环结束后，为了让模型在一个相对稳定的学习率下继续训练一段时间，以进一步优化模型性能，会设置一个冷却阶段。--cooldown-epochs 就是用来控制这个冷却阶段的持续时间，即学习率保持在最小学习率的 epoch 数量。
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    #该参数是针对 Plateau 学习率调度器设置的耐心 epoch 数。Plateau 调度器会监控某个指标（如验证集上的损失或准确率），当该指标在连续的 --patience-epochs 个 epoch 内没有显著改善时，就会降低学习率。例如，如果 --patience-epochs 设置为 10，那么当验证集指标在连续 10 个 epoch 内都没有变好时，学习率就会按照调度器的规则进行衰减。
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    #该参数指定了学习率的衰减率。当进行学习率衰减时，新的学习率会按照当前学习率乘以衰减率的方式进行计算。例如，如果当前学习率为 lr，衰减率 --decay-rate 为 0.1，那么衰减后的学习率就是 lr * 0.1。这个参数与 --decay-epochs 等参数配合使用，共同决定了学习率在训练过程中的变化方式。

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    #设置颜色抖动因子，并且默认值是 0.4
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    #用于设置 AutoAugment 策略，并且可以是 "v0"、"original" 等值，默认是 rand-m9-mstd0.5-inc1。
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    #用于标签平滑，默认值是 0.1
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    #设置训练插值方法，可以是 random（随机）、bilinear（双线性）、bicubic（双三次），默认是 bicubic
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    #用于设置随机擦除的概率，默认值是 0.25
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    #用于设置随机擦除的模式，默认模式是 pixel
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    #用于设置随机擦除的次数，默认值是 1
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    #作用是在第一次（干净的）数据增强分割时不进行随机擦除操作

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    #该参数是 mixup 操作的 alpha 值，并且当该值大于 0 时启用 mixup 操作，默认值为 0.8
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    #该参数是 cutmix 操作的 alpha 值，当该值大于 0 时启用 cutmix 操作，默认值为 1.0
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    #该参数用于设置 cutmix 的最小和最大比例，如果设置了该参数，将覆盖 cutmix 的 alpha 值并启用 cutmix 操作，默认值为 None
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    #设置当 mixup 或 cutmix 启用时执行它们的概率
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    #设置当 mixup 和 cutmix 都启用时切换到 cutmix 的概率
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    #设置应用 mixup 和 cutmix 参数的模式，可以是 batch（按批次）、pair（按对）或 elem（按元素）。

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    #用于指定要训练的教师模型的名称，并且默认值为 'regnety_160'
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    #设置知识蒸馏的类型，默认情况下不进行知识蒸馏
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    #设置知识蒸馏中的温度参数 tau

    # * Finetuning params
    #微调参数
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    #用户在命令行中可以通过 --finetune 来指定从哪个检查点（checkpoint）进行微调
    # Dataset parameters
    #数据集参数部分
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    #用于指定数据集的路径
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    #设置数据集的类型，默认是 IMNET，取值只能是 ['CIFAR', 'IMNET', 'INAT', 'INAT19'] 中的一个，用于区分不同的数据集
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    #当数据集为 INAT 或 INAT19 时，该参数用于指定语义粒度（semantic granularity），默认值为 'name'，可选值有 ['kingdom', 'phylum', 'class', 'order','supercategory', 'family', 'genus', 'name']

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    #指定保存结果的路径，默认值为空字符串，表示如果不指定路径则不保存结果
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    #设置用于训练和测试的设备，默认是 'cuda'，即使用 GPU（如果可用），也可以指定其他设备
    parser.add_argument('--seed', default=0, type=int)
    #设置随机种子，默认值为 0，类型为整数，用于控制随机数生成的结果，以保证实验的可重复性
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    #指定从哪个检查点恢复训练，默认值为空字符串，即不进行恢复操作
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    #设置开始训练的 epoch 数，默认值为 0，类型为整数
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    #当用户在命令行中指定该参数时（即加上 --eval 选项），程序只进行评估操作，不进行训练
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    #用于启用分布式评估，默认值为 False
    parser.add_argument('--num_workers', default=10, type=int)
    #设置数据加载器（DataLoader）使用的工作进程数，默认值为 10，类型为整数
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    #用于指定分布式训练进程的数量
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    #用于设置分布式训练的 URL
    return parser


def main(args):
    utils.init_distributed_mode(args)
    #调用 utils 模块中的 init_distributed_mode 函数，该函数的作用是初始化分布式训练模式。

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")
    #这是一个条件判断语句，用于检查特定的配置组合是否支持。具体来说，如果 args.distillation_type 不等于 'none'（表示使用了知识蒸馏技术），并且 args.finetune 为 True（表示要进行微调操作），同时 args.eval 为 False（表示不是只进行评估操作），则抛出一个 NotImplementedError 异常，提示当前程序还不支持这种配置下的微调操作
    device = torch.device(args.device)
    #这行代码根据 args.device 的值创建一个 torch.device 对象，用于指定模型和数据要运行的设备。args.device 通常可以是 'cuda'（表示使用 GPU）或 'cpu'（表示使用 CPU）

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    #这里将 args.seed 与 utils.get_rank() 的结果相加得到一个新的种子值。utils.get_rank() 通常用于获取当前进程在分布式训练中的排名，不同的进程使用不同的种子值可以确保每个进程的随机性是独立的
    torch.manual_seed(seed)
    #设置 PyTorch 的随机种子，这样在使用 PyTorch 进行随机操作（如初始化模型参数、随机数据采样等）时，每次运行程序得到的结果都是相同的
    np.random.seed(seed)
    #设置 NumPy 的随机种子，NumPy 在数据处理中经常会用到随机操作，设置种子可以保证这些操作的结果可重复
    # random.seed(seed)

    cudnn.benchmark = True
    #cudnn 是 NVIDIA 提供的用于深度神经网络的 GPU 加速库。cudnn.benchmark = True 表示让 CuDNN 在运行时自动寻找最优的卷积算法，以提高卷积操作的性能。不过，这可能会增加一些额外的初始化时间

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    #调用 build_dataset 函数构建训练数据集，is_train=True 表示构建的是训练集。函数返回两个值，第一个是训练数据集对象 dataset_train，第二个是数据集中的类别数量，将其赋值给 args.nb_classes
    dataset_val, _ = build_dataset(is_train=False, args=args)
    #调用 build_dataset 函数构建验证数据集，is_train=False 表示构建的是验证集。只取返回的第一个值作为验证数据集对象 dataset_val，第二个值用 _ 占位表示忽略

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        #调用 utils 模块中的 get_world_size 函数，获取分布式训练中的进程总数，即参与训练的设备或节点的数量，将结果存储在 num_tasks 变量中
        global_rank = utils.get_rank()
        #调用 utils 模块中的 get_rank 函数，获取当前进程在所有进程中的全局排名，将结果存储在 global_rank 变量中。
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            #如果 args.repeated_aug 为 True，表示使用重复数据增强（Repeated Augmentation）策略。此时使用 RASampler 作为训练集的采样器。RASampler 是自定义的采样器，用于实现重复数据增强，它会在每个 epoch 中对数据进行多次采样，以增加数据的多样性。num_replicas 参数指定了参与训练的进程总数，rank 参数指定了当前进程的排名，shuffle 参数设置为 True 表示在每个 epoch 开始时对数据进行打乱
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        #如果 args.repeated_aug 为 False，则使用 torch.utils.data.DistributedSampler 作为训练集的采样器。DistributedSampler 是 PyTorch 提供的用于分布式训练的采样器，它会将数据集划分到不同的进程中，每个进程只处理数据集中的一部分数据，从而实现分布式训练。同样，num_replicas、rank 和 shuffle 参数的含义与上面相同。
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        #如果 args.dist_eval 为 True，表示要进行分布式评估。首先检查验证集的样本数量是否能被进程总数整除，如果不能整除，会打印一个警告信息，提示由于需要在每个进程中分配相等数量的样本，可能会添加额外的重复条目，这会稍微改变验证结果。然后使用 torch.utils.data.DistributedSampler 作为验证集的采样器，shuffle 参数设置为 False 表示在评估过程中不打乱数据顺序
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        #如果 args.dist_eval 为 False，则使用 torch.utils.data.SequentialSampler 作为验证集的采样器。SequentialSampler 会按照数据集的顺序依次采样，用于非分布式评估
    else:
    #非分布式训练模式（else 分支）
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #对于训练集，使用 torch.utils.data.RandomSampler 作为采样器，它会随机地从数据集中采样数据，以增加训练数据的随机性。
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    #对于验证集，使用 torch.utils.data.SequentialSampler 作为采样器，按照数据集的顺序依次采样数据，保证评估结果的一致性。

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        #dataset_train：传入之前构建好的训练集数据集对象，DataLoader 会从这个数据集中获取数据
        #sampler=sampler_train：指定使用之前根据不同条件选择好的训练集采样器 sampler_train，用于决定如何从数据集中选取样本
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        #num_workers=args.num_workers：指定用于数据加载的子进程数量，同样从命令行参数 args 中获取。使用多个子进程可以并行加载数据，加快数据加载速度
        pin_memory=args.pin_mem,
        #pin_memory=args.pin_mem：如果 args.pin_mem 为 True，则会将数据加载到固定内存（页锁定内存）中，这样可以加快数据从 CPU 到 GPU 的传输速度
        drop_last=True,
        #如果数据集的样本数量不能被 batch_size 整除，最后一个不完整的批次将被丢弃，保证每个批次的数据量都是 batch_size
        persistent_workers=True
        #persistent_workers=True：设置为 True 时，数据加载器在多个 epoch 之间会保持子进程的存活，避免了每个 epoch 重新创建和销毁子进程的开销，提高了效率
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size / 2),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    #判断是否需要使用 Mixup 或 CutMix 数据增强策略。如果 args.mixup 大于 0、args.cutmix 大于 0 或者 args.cutmix_minmax 不为 None，则认为需要使用这些增强策略
    if mixup_active:
        mixup_fn = Mixup(
            #如果需要使用增强策略，则创建一个 Mixup 类的实例
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    '''
    mixup_alpha=args.mixup：Mixup 操作的 alpha 参数，控制 Mixup 中样本混合的程度。
    cutmix_alpha=args.cutmix：CutMix 操作的 alpha 参数，控制 CutMix 中裁剪和粘贴的程度。
    cutmix_minmax=args.cutmix_minmax：CutMix 的最小和最大比例，用于更精细地控制 CutMix 操作。
    prob=args.mixup_prob：执行 Mixup 或 CutMix 操作的概率。
    switch_prob=args.mixup_switch_prob：当同时启用 Mixup 和 CutMix 时，切换到 CutMix 的概率。
    mode=args.mixup_mode：指定应用 Mixup 或 CutMix 参数的模式，如 'batch'、'pair' 或 'elem'。
    label_smoothing=args.smoothing：标签平滑的系数，用于缓解过拟合。
    num_classes=args.nb_classes：数据集中的类别数量
    '''

    print(f"Creating model: {args.model}")
    #打印出正在创建的模型名称，该名称从命令行参数 args 中获取，方便开发者确认当前要创建的模型

    model = archs[args.model](args)
    #通过 archs[args.model] 选取对应的模型构造函数，并传入 args 参数来创建模型实例

    print(model)
    #直接打印模型实例，会输出模型的具体结构，包括各个层的信息，方便开发者查看和调试
    model.eval()
    #将模型设置为评估模式
    flops = FlopCountAnalysis(model, torch.rand(1, 3, args.input_size, args.input_size))
    #FlopCountAnalysis：这是一个用于计算模型浮点运算次数（FLOPs）的工具。torch.rand(1, 3, args.input_size, args.input_size) 创建了一个随机输入张量，模拟模型的输入。通过 FlopCountAnalysis 对模型和这个随机输入进行分析，得到模型的 FLOPs 信息
    print(flop_count_table(flops))
    #将计算得到的 FLOPs 信息以表格形式输出，方便查看

    if args.finetune:
        #args.finetune 是一个命令行参数，如果该参数不为空，则表示要进行模型微调
        checkpoint = torch.load(args.finetune, map_location='cpu')
        #从指定的文件路径加载模型的检查点（checkpoint），并将其映射到 CPU 上
        model.load_state_dict(checkpoint['model'], strict=True)
        #将加载的模型参数加载到当前模型中。strict=True 表示严格匹配模型的参数，即要求加载的参数和当前模型的参数完全一致

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #将模型中的普通批量归一化层（BatchNorm）转换为同步批量归一化层（SyncBatchNorm）。在分布式训练中，同步批量归一化可以确保不同设备上的批量归一化统计信息一致。
    model.to(device)
    #将模型移动到指定的设备（如 GPU）上

    model_ema = None
    #args.model_ema 是一个命令行参数，如果该参数为 True，则表示要使用指数移动平均（EMA）模型
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema_decay = args.model_ema_decay ** (args.batch_size * utils.get_world_size() / 512.0)
        #计算 EMA 模型的衰减率。它是根据命令行参数 args.model_ema_decay 以及当前的批量大小和分布式训练的进程数进行调整的
        model_ema = ModelEma(
            model,
            decay=model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        #ModelEma：这是一个自定义的类，用于创建 EMA 模型

    model_without_ddp = model
    if args.distributed:
        #如果要进行分布式训练
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        #将模型包装成分布式数据并行（DDP）模型，以支持在多个设备上并行训练。device_ids=[args.gpu] 指定了当前设备的 ID，find_unused_parameters=False 表示不查找未使用的参数，以提高训练效率
        model_without_ddp = model.module
        #在使用 DDP 包装模型后，通过 model.module 可以获取原始的模型实例
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    #通过 sum 函数计算所有可训练参数的元素数量，并打印输出

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    #utils.get_world_size() 函数返回分布式训练中的进程总数，也就是参与训练的设备数量
    #通过将基础学习率乘以批次大小和进程总数，再除以 512，得到线性缩放后的学习率 linear_scaled_lr。这种缩放方式是为了在分布式训练或者增大批次大小时，让学习率能与之适配，保持训练的稳定性
    args.lr = linear_scaled_lr
    #最后将缩放后的学习率重新赋值给 args.lr，以便后续使用
    optimizer = create_optimizer(args, model_without_ddp)
    #create_optimizer 是一个自定义函数，用于根据传入的参数 args 和模型 model_without_ddp 创建优化器
    loss_scaler = NativeScaler()
    #NativeScaler 是 PyTorch 中用于混合精度训练的损失缩放器。在混合精度训练中，使用半精度浮点数（如 FP16）来减少内存占用和加速计算

    lr_scheduler, _ = create_scheduler(args, optimizer)
    #create_scheduler 是自定义函数，根据传入的参数 args 和优化器 optimizer 创建学习率调度器。
    # lr_scheduler = build_scheduler(args, optimizer, len(data_loader_train))

    criterion = LabelSmoothingCrossEntropy()
    #首先，将损失函数初始化为 LabelSmoothingCrossEntropy，这是一种带有标签平滑的交叉熵损失函数，标签平滑可以防止模型过拟合

    if args.mixup > 0.:
        #如果 args.mixup > 0，即使用了 Mixup 数据增强技术，那么将损失函数替换为 SoftTargetCrossEntropy。Mixup 会将不同样本混合，生成软标签，因此需要使用适合软标签的损失函数
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        #如果 args.smoothing 为 True，则使用带有指定平滑系数的 LabelSmoothingCrossEntropy
    else:
        criterion = torch.nn.CrossEntropyLoss()
        #如果以上条件都不满足，就使用 PyTorch 提供的标准交叉熵损失函数 torch.nn.CrossEntropyLoss

    teacher_model = None
    #创建教师模型（知识蒸馏）
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    #这段代码将原有的损失函数 criterion 封装成一个支持知识蒸馏的损失函数
    '''
    teacher_model：教师模型，在知识蒸馏中，教师模型的输出会作为软标签，辅助学生模型（即当前正在训练的模型）的学习。
    args.distillation_type：指定知识蒸馏的类型，例如可以是 'soft'、'hard' 等不同的蒸馏策略。
    args.distillation_alpha：蒸馏损失的权重系数，用于平衡基础损失和蒸馏损失的比重。
    args.distillation_tau：温度参数，在软蒸馏中，温度参数用于调整教师模型输出的软标签的平滑程度。
    '''

    max_accuracy = 0.0
    #用于记录训练过程中的最大准确率
    output_dir = Path(args.output_dir)
    #使用 pathlib.Path 来处理输出目录
    # ipdb.set_trace()
    if args.resume == '':
        #如果命令行参数 args.resume 为空，说明没有指定恢复训练的检查点路径
        tmp = f"{args.output_dir}/checkpoint.pth"
        #构造一个默认的检查点路径，即输出目录下的 checkpoint.pth 文件
        if os.path.exists(tmp):
            #检查默认的检查点文件是否存在，如果存在，则将其路径赋值给 args.resume，以便后续恢复训练
            args.resume = tmp
    flag = os.path.exists(args.resume)
    #如果指定了恢复训练的检查点路径，并且该文件确实存在，则执行恢复训练的操作
    if args.resume and flag:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        #如果 args.resume 以 https 开头，说明检查点文件存放在网络上。使用 torch.hub.load_state_dict_from_url 函数从指定的 URL 加载检查点。map_location='cpu' 表示将加载的张量数据映射到 CPU 上，避免在 GPU 显存不足时出现错误；check_hash=True 表示会检查下载文件的哈希值，确保文件的完整性
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        #从本地加载：如果 args.resume 不是以 https 开头，说明检查点文件存放在本地。使用 torch.load 函数从本地文件加载检查点，同样将数据映射到 CPU 上
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        #model_without_ddp：这是原始的模型对象（未经过分布式数据并行包装）
        #checkpoint['model']：检查点中保存的模型参数的状态字典
        if args.model_ema:                
            model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=True)
        #加载 EMA 模型参数（如果使用了 EMA）
        #加载优化器、学习率调度器和损失缩放器状态（如果不是仅评估模式）
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        #加载最大准确率
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    if args.eval:
        #if args.resume == '':
        tmp = f"{args.output_dir}/downtarget.pth"
        #构建一个默认的检查点文件路径
        if os.path.exists(tmp):
            args.resume = tmp
        flag = os.path.exists(args.resume)
        #再次检查 args.resume 指定的检查点文件是否存在，并将结果存储在 flag 变量中
        if args.resume and flag:
            #如果 args.resume 不为空且对应的检查点文件存在
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            if args.model_ema:                
                model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=True)
        test_stats = evaluate(data_loader_val, model, device)
        #调用 evaluate 函数对模型进行评估。data_loader_val 是验证集的数据加载器，model 是待评估的模型，device 是指定的计算设备（如 GPU 或 CPU）。该函数返回一个包含评估指标的字典 test_stats
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        #印模型在验证集上的准确率（test_stats['acc1']），并显示验证集的样本数量
        if model_ema is not None:
            #检查是否使用了 EMA 模型。如果存在 EMA 模型，则对其进行评估
            test_stats_ema = evaluate(data_loader_val, model_ema.ema, device)
            print(f"Accuracy of the network_ema on the {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
            #打印 EMA 模型在验证集上的准确率，并显示验证集的样本数量
        return

    print(f"Start training for {args.epochs} epochs")
    #表明训练即将开始，并且会进行 args.epochs 个轮次（epochs）的训练
    start_time = time.time()
    #time.time() 函数会返回当前时间的时间戳
    max_accuracy = 0.0
    #用于记录在训练过程中模型在验证集上达到的最大准确率
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            #分布式训练时设置采样器的 epoch

        train_stats = train_one_epoch(
            #调用 train_one_epoch 函数进行一轮的训练
            lr_scheduler, model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune==''  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)
        #调用 lr_scheduler.step(epoch) 方法，根据当前的训练轮次 epoch 更新学习率

        #保存主要检查点文件
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                if args.model_ema:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'max_accuracy': max_accuracy,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'max_accuracy': max_accuracy,
                    }, checkpoint_path)
            if epoch % 10 == 0:
                #每 10 个轮次保存备份检查点文件
                if args.model_ema:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'max_accuracy': max_accuracy,
                    }, f"{args.output_dir}/backup{epoch}.pth")
                else:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'max_accuracy': max_accuracy,
                    }, f"{args.output_dir}/backup{epoch}.pth")

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if model_ema is not None:
            test_stats_ema = evaluate(data_loader_val, model_ema.ema, device)
            print(f"Accuracy of the network_ema on the {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        if model_ema is not None:
            max_accuracy = max(max_accuracy, test_stats_ema['acc1'])
        print('Max accuracy: {:.2f}%'.format(max_accuracy))
        #保存最佳模型
        if (max_accuracy == test_stats["acc1"]) or (max_accuracy == test_stats_ema['acc1']):
            if args.model_ema:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'max_accuracy': max_accuracy,
                    }, f"{args.output_dir}/best.pth")
            else:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'max_accuracy': max_accuracy,
                }, f"{args.output_dir}/best.pth")
        #格式化训练和测试统计信息
        train_f = {'train_{}'.format(k): v for k, v in train_stats.items()}
        test_f = {'test_{}'.format(k): v for k, v in test_stats.items()}
        if model_ema is not None:
            test_ema_f = {'test_ema_{}'.format(k): v for k, v in test_stats_ema.items()}
        # 合并日志统计信息
        log_stats = dict({'epoch': epoch,
                          'n_parameters': n_parameters}, **train_f, **test_f, **test_ema_f)
        #记录日志信息
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    #计算并输出训练总时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
