# modified from https://github.com/facebookresearch/LeViT
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
from sklearn.metrics import confusion_matrix

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item() if self.deque else 0.0  # 示例修复

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item() if self.deque else 0.0  # 示例修复

    @property
    def global_avg(self):
        # return self.total / self.count
        return self.total / self.count if self.count != 0 else 0.0  # 关键修改行

    @property
    def max(self):
        return max(self.deque) if self.deque else 0.0  # 示例修复

    @property
    def value(self):
        return self.deque[-1] if self.deque else 0.0  # 关键修复行

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue) #字典，键为指标名称（如'loss'、'acc1'），值为对应的SmoothedValue实例
        self.delimiter = delimiter #控制日志输出格式的分隔符

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        允许通过属性访问直接获取指标跟踪器（语法糖）
        例如：logger.acc1 等价于 logger.meters['acc1']
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    # 使用示例：
    # print(logger.acc1.global_avg)  # 直接通过属性访问全局平均值

    def __str__(self):
        """
        生成指标日志字符串，格式为"指标名: SmoothedValue字符串表示"的组合
        """
        loss_str = []
        for name, meter in self.meters.items():
            ## 每个指标的字符串由SmoothedValue的__str__方法生成（包含中位数、全局平均等）
            loss_str.append(
                "{}: {}".format(name, str(meter)) # 如"loss: 0.2345 (0.2567)"
            )
        return self.delimiter.join(loss_str)   # 用分隔符连接所有指标字符串

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        手动添加自定义的指标跟踪器（默认自动创建SmoothedValue，此方法用于特殊需求）
        :param name: 指标名称（字符串）
        :param meter: 跟踪器实例（需兼容SmoothedValue接口）
        """
        self.meters[name] = meter ## 覆盖默认的SmoothedValue创建逻辑
        # 使用场景：
        # 若需使用非SmoothedValue的跟踪器（如自定义的统计类），可通过此方法添加


    def init_binary_metrics(self):
        """初始化二分类专用指标（灵敏度、特异性）"""
        self.add_meter('sensitivity', SmoothedValue(fmt='{value:.4f}'))
        self.add_meter('specificity', SmoothedValue(fmt='{value:.4f}'))


    def update_binary_metrics(self, target, pred):
        """
        基于真实标签和预测标签更新灵敏度和特异性（需在所有批次处理后调用）
        :param target: 真实标签张量（CPU上的1D张量，如target.cpu()）
        :param pred: 预测标签张量（CPU上的1D张量，如pred.cpu()）
        """
        # 转换为numpy数组（确保在CPU上）
        target_np = target.numpy()
        pred_np = pred.numpy()
        # 计算混淆矩阵
        cm = confusion_matrix(target_np, pred_np, labels=[0, 1])  # 假设0=负例，1=正例
        TN, FP = cm[0, 0], cm[0, 1]
        FN, TP = cm[1, 0], cm[1, 1]
        # 计算指标（处理除零情况）
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
        # 更新指标
        self.sensitivity.update(sensitivity)
        self.specificity.update(specificity)

    def log_every(self, iterable, print_freq, header=None):
        """
        迭代数据加载器并按指定频率打印进度日志，支持时间统计、指标显示和CUDA内存监控
        :param iterable: 可迭代对象（如DataLoader）
        :param print_freq: 打印频率（每print_freq次迭代打印一次）
        :param header: 日志前缀（如'Train:'或'Val:'）
        :return: 生成器，逐个yield iterable中的元素
        """
        i = 0  # 迭代计数器
        if not header:
            header = '' # 处理空header
        start_time = time.time() # 记录整个迭代过程的开始时间
        end = time.time()  # 初始化上一次迭代结束时间
        # 初始化时间统计器（仅跟踪平均值，窗口大小使用SmoothedValue默认值20）
        iter_time = SmoothedValue(fmt='{avg:.4f}')  # 记录每次迭代处理时间（含前向传播等）
        data_time = SmoothedValue(fmt='{avg:.4f}')  # 记录数据加载时间（从DataLoader获取数据的时间）

        # 动态生成迭代索引的格式字符串（确保数字位数一致，如总迭代1000次时显示000/1000）
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        # 定义日志模板（包含动态替换的占位符）
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]', # 迭代进度（当前/总次数）
            'eta: {eta}',  # 预计剩余时间（Estimated Time of Arrival）
            '{meters}',  # 指标日志（来自MetricLogger的__str__输出）
            'time: {time}',  # 平均迭代时间
            'data: {data}'  # 平均数据加载时间
        ]

        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')  # 添加CUDA内存监控（仅GPU环境）
        log_msg = self.delimiter.join(log_msg)  # 用分隔符拼接成完整日志模板
        #使用 self.delimiter 将列表中的各个部分连接成一个完整的日志消息
        MB = 1024.0 * 1024.0
        #用于将字节转换为兆字节（MB）

        # 核心迭代逻辑（生成器函数，通过yield返回数据
        for obj in iterable:
            # 记录数据加载时间（从DataLoader获取数据到进入循环的时间）
            data_time.update(time.time() - end) # 当前数据加载耗时 = 当前时间 - 上次迭代结束时间
            yield obj # 将数据对象交给调用者（如训练循环中的模型处理）

            # 当前迭代耗时 = 当前时间 - 上次迭代结束时间
            iter_time.update(time.time() - end)

            # 满足打印条件时输出日志
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 计算ETA：剩余迭代次数 × 平均迭代时间
                #ETA（Estimated Time of Arrival） 指的是预计剩余时间
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))  # 转换为HH:MM:SS格式
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1  # 迭代计数器递增
            end = time.time()  # 更新上次迭代结束时间为当前时间
        # 迭代结束后打印总时间统计
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


#分布式环境初始化
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    '''
    检测分布式环境变量

    条件1：检查环境变量中是否存在RANK和WORLD_SIZE。
    
    若存在，表示通过标准的分布式启动工具（如torch.distributed.launch）启动。
    
    设置args.rank（进程全局唯一ID）、args.world_size（总进程数）、args.gpu（当前进程的本地GPU编号）。
    
    条件2：若未满足条件1，但存在SLURM_PROCID，表示通过SLURM集群管理系统启动。
    
    设置args.rank为SLURM_PROCID（SLURM分配的进程ID）。
    
    根据rank和当前节点的GPU数量，计算args.gpu（通过取余实现GPU分配）。
    
    条件3：若上述条件均不满足，标记为非分布式模式并退出。
    '''

    args.distributed = True
    #启用分布式模式
    torch.cuda.set_device(args.gpu)
    #绑定当前进程使用的GPU
    args.dist_backend = 'nccl'
    #设置通信后端为nccl（NVIDIA GPU专用通信库）
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    #打印初始化信息，包含当前进程的rank和通信地址dist_url
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    '''
    调用torch.distributed.init_process_group初始化进程组：

    backend: 通信后端（此处为nccl）。
    
    init_method: 进程组发现URL（如tcp://IP:PORT或共享文件路径）。
    
    world_size: 总进程数。
    
    rank: 当前进程的全局ID
    '''
    torch.distributed.barrier()
    #同步所有进程，确保初始化完成
    setup_for_distributed(args.rank == 0)
    #仅在主进程（rank=0）执行特定初始化（如日志、模型保存）


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.Conv2d):
            child.bias = torch.nn.Parameter(torch.zeros(child.weight.size(0)))
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


def replace_layernorm(net):
    import apex
    for child_name, child in net.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            setattr(net, child_name, apex.normalization.FusedLayerNorm(
                child.weight.size(0)))
        else:
            replace_layernorm(child)



