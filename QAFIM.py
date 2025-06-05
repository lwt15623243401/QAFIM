import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import init
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
from typing import Tuple, Union
from functools import partial
from einops import einsum


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class AnchorGenerator(nn.Module):  # 锚点生成器架构：一个轻量级网络，根据输入特征动态生成锚点坐标，替代原有的固定可学习参数
    def __init__(self, embed_dim, num_anchors):
        super().__init__()
        self.num_anchors = num_anchors
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局特征压缩
        # 生成复数坐标，每个坐标由实部和虚部组成
        self.coord_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * num_anchors),  # 生成 (num_anchors, 4) ，分别对应实部和虚部
            nn.Tanh()  # 限制坐标在 [-1, 1] 范围（与像素坐标一致）
        )

    def forward(self, x):
        b, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2)
        global_feat = self.global_pool(x).reshape(b, c)  # (b, c)
        # 生成复数坐标
        complex_coords = self.coord_proj(global_feat).reshape(b, self.num_anchors, 4)  # (b, K, 4)
        return complex_coords


class PhaseRotation(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # 相位参数生成网络
        self.phase_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)  # 输出embed_dim维相位参数
        )

        # 复数特征生成器：将实数特征转换为复数表示（保持维度不变）
        self.real_imag_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # 最终投影层：将复数特征（实部+虚部）合并回embed_dim维
        self.final_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        输入: 实数特征 x (b, n, embed_dim)
        输出: 相位旋转后的实数特征 (b, n, embed_dim)
        """
        b, n, c = x.shape

        # 1. 生成复数特征的实部和虚部（保持维度不变）
        complex_features = self.real_imag_proj(x)  # (b, n, embed_dim)

        # 2. 生成相位参数（基于全局特征均值）
        global_feat = x.mean(dim=1)  # (b, embed_dim)
        phase_params = self.phase_proj(global_feat)  # (b, embed_dim)
        cos_theta = torch.cos(phase_params)  # (b, embed_dim)
        sin_theta = torch.sin(phase_params)  # (b, embed_dim)

        # 3. 相位旋转（复数乘法）
        x_real = complex_features  # 实部
        x_imag = complex_features  # 虚部（使用相同特征，通过不同权重学习虚实部关系）

        x_rot_real = x_real * cos_theta.unsqueeze(1) - x_imag * sin_theta.unsqueeze(1)  # (b, n, embed_dim)
        x_rot_imag = x_real * sin_theta.unsqueeze(1) + x_imag * cos_theta.unsqueeze(1)  # (b, n, embed_dim)

        # 4. 合并实部和虚部，通过投影层回到embed_dim维
        x_rotated = torch.cat([x_rot_real, x_rot_imag], dim=-1)  # (b, n, 2*embed_dim)
        x_rotated = self.final_proj(x_rotated)  # (b, n, embed_dim)

        return x_rotated


class AmplitudeAttenuation(nn.Module):
    def __init__(self, coord_dim=4):  # 指定坐标维度为4
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(coord_dim, coord_dim),  # 输入输出维度对齐坐标的4维
            nn.Sigmoid()
        )

    def forward(self, complex_coords, x):
        """
        complex_coords: (b, K, 4)
        x: 未使用（仅为兼容接口）
        """
        b, K, _ = complex_coords.shape
        # 直接基于坐标生成衰减因子
        attenuation = self.gate(complex_coords)  # (b, K, 4)
        return complex_coords * attenuation  # (b, K, 4)


class LocalPhaseField(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.phase_proj = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        phase_shift = self.phase_proj(x).squeeze(1)  # (b, h, w)
        return phase_shift


class GaugeConnection(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.phase_increment_proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, relative_coords):
        b, N, K, _ = relative_coords.shape
        flat_relative_coords = relative_coords.reshape(-1, 2)
        phase_increments = self.phase_increment_proj(flat_relative_coords).reshape(b, N, K)
        gauge_connection = torch.exp(1j * phase_increments)
        return gauge_connection


class AnchoredFeatureMixing(nn.Module):
    def __init__(self, embed_dim, num_anchors=8, kernel_size=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_anchors = num_anchors

        self.anchor_generator = AnchorGenerator(embed_dim, num_anchors)
        self.anchor_features = nn.Parameter(torch.zeros(num_anchors, embed_dim))

        init.xavier_normal_(self.anchor_features, gain=0.1)

        # 位置编码函数：计算像素与锚点的相对距离（曼哈顿距离）
        self.distance_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 局部细化模块：轻量卷积处理邻域特征
        self.local_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size, padding=kernel_size // 2,
                                    groups=embed_dim)

        self.local_norm = nn.LayerNorm(embed_dim)
        self.anchor_norm = nn.LayerNorm(embed_dim)

        # 新增：层缩放参数（锚点全局分支）
        self.global_scale = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.local_scale = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # 新增：最终残差后归一化
        self.post_layer_norm = nn.LayerNorm(embed_dim)

        self.phase_rotation = PhaseRotation(embed_dim)  # <--- 添加此行
        # 新增：相位投影层（将虚部坐标映射到 embed_dim 维度）
        self.phase_projection = nn.Sequential(
            nn.Linear(2, embed_dim),  # 输入虚部坐标 (2维)
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # 新增：振幅衰减模块
        self.attenuation_module = AmplitudeAttenuation(coord_dim=4)  # <--- 添加此行

        # 新增：局部相位场
        self.local_phase_field = LocalPhaseField(embed_dim)
        # 新增：规范联接
        self.gauge_connection = GaugeConnection(embed_dim)

    def forward(self, x):
        b, h, w, c = x.shape
        pixel_coords = self._get_pixel_coords(h, w, x)  # (N,2)

        # 动态生成复数锚点坐标
        complex_anchor_coords = self.anchor_generator(x)  # (b, K, 4)
        # 测量机制：振幅衰减和相位旋转
        # 这里简单示例，根据特征图均值进行振幅衰减
        # feature_mean = x.mean(dim=(1, 2, 3), keepdim=True) #(b, 1, 1, 1)
        # attenuation_factor = torch.sigmoid(feature_mean) #(b, 1, 1, 1)
        # complex_anchor_coords = complex_anchor_coords * attenuation_factor.squeeze(1) # (b, K, 4)

        # 新代码：使用可学习的振幅衰减模块
        complex_anchor_coords = self.attenuation_module(complex_anchor_coords, x)  # (b, K, 4)

        # 取实部作为最终锚点位置
        dynamic_anchor_coords = complex_anchor_coords[..., :2]  # (b, K, 2)

        anchor_phases = complex_anchor_coords[..., 2:]  # 虚部用于相位

        # 给 pixel_coords 增加批次维度
        pixel_coords = pixel_coords.unsqueeze(0).expand(b, -1, -1)  # (b, N, 2)

        # 计算局部相位场
        x_permuted = x.permute(0, 3, 1, 2)
        phase_shift = self.local_phase_field(x_permuted).unsqueeze(-1)  # (b, h, w, 1)
        pixel_coords = pixel_coords + phase_shift.reshape(b, -1, 1)

        # 计算相对坐标
        relative_coords = pixel_coords[:, :, None, :] - dynamic_anchor_coords[:, None, :, :]  # (b, N, K, 2)

        # 计算规范联接
        gauge_connection = self.gauge_connection(relative_coords)

        # 1. 像素→锚点更新：聚合像素特征到锚点
        # 形状：(b, N, K, 2) → 展平为 (b*N*K, 2) 以适配 nn.Linear
        flat_relative_coords = relative_coords.reshape(-1, 2).abs()

        # 通过距离编码器
        distance_emb = self.distance_encoder(flat_relative_coords)

        # 恢复原始维度：(b*N*K, embed_dim) → (b, N, K, embed_dim)
        distance_emb = distance_emb.reshape(b, h * w, self.num_anchors, self.embed_dim)

        # ====== 插入相位信息 ======
        # 投影虚部相位到特征维度
        anchor_phases = self.phase_projection(anchor_phases.reshape(-1, 2))  # (b*K, c)
        anchor_phases = anchor_phases.reshape(b, self.num_anchors, self.embed_dim)
        # 调整维度并应用相位信息
        distance_emb = distance_emb * anchor_phases.unsqueeze(1)  # (b, N, K, c)
        # ========================

        # 取绝对值后成为曼哈顿距离
        distance_emb = torch.clamp(distance_emb, min=-10, max=10)
        # 将 distance_emb 张量中的所有元素限制在 [-10, 10] 范围内，防止数值溢出
        pixel_features = x.reshape(b, h * w, c)
        pixel_features = nn.functional.layer_norm(pixel_features, (self.embed_dim,))

        # 应用相位旋转（输入形状需为 (b, n, c)）
        pixel_features = self.phase_rotation(pixel_features)  # <--- 修改后的PhaseRotation

        # 在通道维度应用 softmax
        log_distance_emb = distance_emb - distance_emb.max(dim=2, keepdim=True).values
        distance_emb_softmax = torch.exp(log_distance_emb) + 1e-10
        distance_emb_softmax /= distance_emb_softmax.sum(dim=2, keepdim=True)

        # 不变聚合：与规范联接相乘
        distance_emb_softmax = distance_emb_softmax * torch.abs(gauge_connection).unsqueeze(-1)

        # 沿通道维度加权聚合，得到每个像素对锚点的贡献 (b, N, K)
        anchor_update = torch.einsum('bnc,bnkc->bnk', pixel_features, distance_emb_softmax)
        anchor_update = anchor_update / (h * w)

        anchor_update_sum = anchor_update.sum(dim=1)

        # 扩展 anchor_update_sum 到 (b, K, C)
        anchor_update_sum = anchor_update_sum.unsqueeze(-1).expand(-1, -1, c)
        anchor_representation = self.anchor_features[None, :, :] + anchor_update_sum

        anchor_representation = self.anchor_norm(anchor_representation)

        # 锚点到像素的广播，同样使用通道维度的权重
        anchor_to_pixel = torch.einsum('bkc,bnkc->bnc', anchor_representation, distance_emb_softmax)

        global_features = anchor_to_pixel.reshape(b, h, w, c)

        # 3. 局部特征细化
        local_features = self.local_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        local_features = self.local_norm(local_features)

        # 层缩放
        global_features = self.global_scale * global_features
        local_features = self.local_scale * local_features

        # 4. 特征融合：全局+局部
        residual = x + global_features + local_features
        output = self.post_layer_norm(residual)

        return output

    def _get_pixel_coords(self, h, w, x):
        """生成归一化的像素坐标（范围[-1, 1]）"""
        y = torch.linspace(-1, 1, h, device=x.device)
        x_coord = torch.linspace(-1, 1, w, device=x.device)
        grid = torch.meshgrid(y, x_coord, indexing='ij')
        coords = torch.stack(grid, dim=-1).reshape(-1, 2)
        return coords


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x


class RetNetRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        此方法用于生成一个二维的衰减掩码（decay mask），掩码的形状为 (H * W) * (H * W)，其中 H 和 W 分别代表二维空间的高度和宽度。该掩码可能会在模型中用于处理二维数据的衰减或注意力机制等场景
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        '''
        torch.arange(H) 生成一个从 0 到 H - 1 的一维张量，代表高度方向的索引。
        torch.arange(W) 生成一个从 0 到 W - 1 的一维张量，代表宽度方向的索引。
        .to(self.decay) 将这些索引张量移动到与 self.decay 相同的设备（如 CPU 或 GPU）上
        '''
        grid = torch.meshgrid([index_h, index_w])
        # torch.meshgrid([index_h, index_w]) 根据高度和宽度的索引生成二维网格，返回两个张量，分别代表每个点的高度和宽度坐标。
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # (H*W 2)
        # torch.stack(grid, dim=-1) 将这两个张量在最后一个维度上堆叠起来，形成一个形状为 (H, W, 2) 的张量。
        # .reshape(H*W, 2) 将张量重新调整形状为 (H * W, 2)，其中每一行代表一个二维点的坐标，这一步将二维图像的像素坐标展平成了一维序列
        mask = grid[:, None, :] - grid[None, :, :]  # (H*W H*W 2)
        # grid[:, None, :] 将 grid 张量在第二个维度上扩展，形状变为 (H * W, 1, 2)
        # grid[None, :, :] 将 grid 张量在第一个维度上扩展，形状变为 (1, H * W, 2)
        # 两者相减得到一个形状为 (H * W, H * W, 2) 的张量，其中 mask[i, j] 表示第 i 个点和第 j 个点的坐标差值
        mask = (mask.abs()).sum(dim=-1)
        # mask.abs() 计算差值的绝对值。.sum(dim=-1) 在最后一个维度上求和，得到一个形状为 (H * W, H * W) 的张量，其中 mask[i, j] 表示二维图像像素点展平后的序列的第 i 个点和第 j 个点的曼哈顿距离
        mask = mask * self.decay[:, None, None]  # (n H*W H*W)
        # self.decay 是一个一维张量，代表衰减系数
        # self.decay[:, None, None] 将衰减系数张量扩展为形状 (n, 1, 1)，其中 n 是衰减系数的数量
        # 与 mask 相乘得到一个形状为 (n, H * W, H * W) 的张量，其中每个衰减系数对应一个不同的二维衰减掩码。
        '''
        引入多个衰减系数的目的是为了在模型中模拟不同的注意力衰减行为。例如，在多头注意力机制中，每个注意力头可能需要关注不同范围的信息：

        某些注意力头可能更关注局部信息，即对距离较近的像素赋予更高的权重。

        其他注意力头可能需要捕捉全局信息，即对整个图像范围内的像素赋予相对均衡的权重。

        通过为每个注意力头分配不同的衰减系数，可以实现这种差异化的注意力机制。具体而言：

        衰减系数较大的注意力头会对远处的像素赋予较低的权重，强调局部信息。

        衰减系数较小的注意力头则对远处的像素赋予较高的权重，捕捉全局信息。

        这种设计允许模型在处理图像时，同时考虑局部细节和全局结构，从而提高模型的表达能力和性能。
        '''
        return mask

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # (h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # (h w d1)

            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # (h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # (h w d1)
            mask = self.generate_2d_decay(slen[0], slen[1])  # (n l l)
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


class VisionRetentionChunk(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''

        qr_w = qr.transpose(1, 2)  # (b h n w d1)
        kr_w = kr.transpose(1, 2)  # (b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # (b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (b h n w w)
        qk_mat_w = qk_mat_w + mask_w  # (b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1)  # (b h n w w)
        v = torch.matmul(qk_mat_w, v)  # (b h n w d2)

        qr_h = qr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        v = v.permute(0, 3, 2, 1, 4)  # (b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (b w n h h)
        qk_mat_h = qk_mat_h + mask_h  # (b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # (b w n h h)
        output = torch.matmul(qk_mat_h, v)  # (b w n h d2)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output


class VisionRetentionAll(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos

        assert h * w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        qr = theta_shift(q, sin, cos)  # (b n h w d1)
        kr = theta_shift(k, sin, cos)  # (b n h w d1)

        qr = qr.flatten(2, 3)  # (b n l d1)
        kr = kr.flatten(2, 3)  # (b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d2)
        vr = vr.flatten(2, 3)  # (b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2)  # (b n l l)
        qk_mat = qk_mat + mask  # (b n l l)
        qk_mat = torch.softmax(qk_mat, -1)  # (b n l l)
        output = torch.matmul(qk_mat, vr)  # (b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            activation_fn=F.gelu,
            dropout=0.0,
            activation_dropout=0.0,
            layernorm_eps=1e-6,
            subln=False,
            subconv=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


# class RetBlock(nn.Module):
#
#     def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False,
#                  layer_init_values=1e-5):
#         super().__init__()
#         self.layerscale = layerscale
#         self.embed_dim = embed_dim
#         self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
#         assert retention in ['chunk', 'whole']
#         if retention == 'chunk':
#             self.retention = VisionRetentionChunk(embed_dim, num_heads)
#         else:
#             self.retention = VisionRetentionAll(embed_dim, num_heads)
#         self.drop_path = DropPath(drop_path)
#         self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
#         self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
#         self.pos = DWConv2d(embed_dim, 3, 1, 1)
#
#         if layerscale:
#             self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
#             self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
#
#     def forward(
#             self,
#             x: torch.Tensor,
#             incremental_state=None,
#             chunkwise_recurrent=False,
#             retention_rel_pos=None
#     ):
#         x = x + self.pos(x)
#         if self.layerscale:
#             x = x + self.drop_path(
#                 self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent,
#                                               incremental_state))
#             x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
#         else:
#             x = x + self.drop_path(
#                 self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
#             x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
#         return x


class RetBlock(nn.Module):

    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False,
                 layer_init_values=1e-5, num_anchors=8):  # （新增锚点参数）
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = VisionRetentionChunk(embed_dim, num_heads)
        else:
            self.retention = VisionRetentionAll(embed_dim, num_heads)

        # 新增锚点混合模块
        self.anchor_mixing = AnchoredFeatureMixing(embed_dim, num_anchors=num_anchors)  # 锚点数量可配置

        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(
            self,
            x: torch.Tensor,
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
    ):
        x = x + self.pos(x)

        # 新增锚点混合路径
        x_anchor = self.anchor_mixing(x)  # 锚点双向特征交换

        if self.layerscale:
            x = x + self.drop_path(
                self.gamma_1 * x_anchor)  # 融合全局（锚点）+ 局部（保留机制）特征
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(
                x_anchor)
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.permute(0, 3, 1, 2).contiguous()  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (b oh ow oc)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 downsample: PatchMerging = None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5, num_anchors=8):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
        self.Relpos = RetNetRelPos2d(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.ModuleList([
            RetBlock(flag, embed_dim, num_heads, ffn_dim,
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values,
                     num_anchors=num_anchors)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        b, h, w, d = x.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        for blk in self.blocks:
            if self.use_checkpoint:
                tmp_blk = partial(blk, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent,
                                  retention_rel_pos=rel_pos)
                x = checkpoint.checkpoint(tmp_blk, x)
            else:
                x = blk(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent,
                        retention_rel_pos=rel_pos)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# class BasicLayer(nn.Module):
#     """ A basic Swin Transformer layer for one stage.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#         fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
#     """
#
#     def __init__(self, embed_dim, out_dim, depth, num_heads,
#                  init_value: float, heads_range: float,
#                  ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
#                  downsample: PatchMerging = None, use_checkpoint=False,
#                  layerscale=False, layer_init_values=1e-5):
#
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#         self.chunkwise_recurrent = chunkwise_recurrent
#         if chunkwise_recurrent:
#             flag = 'chunk'
#         else:
#             flag = 'whole'
#         self.Relpos = RetNetRelPos2d(embed_dim, num_heads, init_value, heads_range)
#
#         # build blocks
#         self.blocks = nn.ModuleList([
#             RetBlock(flag, embed_dim, num_heads, ffn_dim,
#                      drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
#             for i in range(depth)])
#
#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None
#
#     def forward(self, x):
#         b, h, w, d = x.size()
#         rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 tmp_blk = partial(blk, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent,
#                                   retention_rel_pos=rel_pos)
#                 x = checkpoint.checkpoint(tmp_blk, x)
#             else:
#                 x = blk(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent,
#                         retention_rel_pos=rel_pos)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous()  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)  # (b h w c)
        return x


class VisRetNet(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoints=[False, False, False, False],
                 chunkwise_recurrents=[True, True, False, False], projection=1024,
                 layerscales=[False, False, False, False], layer_init_values=1e-6):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
                                      norm_layer=norm_layer if self.patch_norm else None)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoints[i_layer],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)

        self.proj = nn.Linear(self.num_features, projection)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(projection, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.proj(x)  # (b h w c)
        x = self.norm(x.permute(0, 3, 1, 2)).flatten(2, 3)  # (b c h*w)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def RMT_T3(args):
    model = VisRetNet(
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, True, False, False],
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()
    return model


@register_model
# def RMT_S(args):
def RMT_S(*args, **kwargs):
    num_classes = kwargs.get("num_classes", 1000)  # 如果没有传入，则默认 1000
    model = VisRetNet(
        num_classes=num_classes,  # 添加这一行
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.15,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()
    return model


@register_model
def RMT_M2(args):
    model = VisRetNet(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[5, 5, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.4,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6
    )
    model.default_cfg = _cfg()
    return model


@register_model
def RMT_L6(args):
    model = VisRetNet(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        init_values=[2, 2, 2, 2],
        heads_ranges=[6, 6, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.5,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6
    )
    model.default_cfg = _cfg()
    return model
