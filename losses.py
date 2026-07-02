"""
losses.py - 4D 密度场损失函数
包含：
  - 背景区域损失（强制空白区域无衰减）
  - 前景区域损失（强制晶体区域衰减值匹配）
  - 时间单调性损失（生长不可逆）
  - 空间 TV 损失（抑制噪声）
已移除时间平滑损失和质量守恒损失（不适合生长过程）。
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def normed_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """归一化互相关（保留备用）"""
    mux = x.mean()
    muy = y.mean()
    dx = x - mux
    dy = y - muy
    return torch.sum(dx * dy) / (torch.sqrt(dx.pow(2).sum() * dy.pow(2).sum()) + 1e-8)


# ==============================================================
# 不推荐用于生长过程的损失（保留，但训练时权重置 0）
# ==============================================================

def temporal_smoothness_loss(
    density_field: torch.nn.Module,
    npts: int = 2048,
    n_pairs: int = 5,
    device: Optional[torch.device] = None,
    time_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    时间平滑损失：惩罚相邻时刻密度变化（使密度场趋向静止）。
    【注意】对于枝晶生长等密度单调增加的过程，该损失会抑制变化，请勿使用。
    """
    if device is None:
        device = next(density_field.parameters()).device

    pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
    loss = torch.tensor(0.0, device=device)

    for _ in range(n_pairs):
        t0 = torch.rand(1, device=device) * (time_range[1] - time_range[0] - 0.02) + time_range[0]
        dt = 0.02 * torch.rand(1, device=device) + 0.01
        t1 = t0 + dt
        rho0 = density_field(pos, t0.expand(npts, 1))
        rho1 = density_field(pos, t1.expand(npts, 1))
        loss += F.mse_loss(rho1, rho0)

    return loss / n_pairs


def mass_conservation_loss(
    density_field: torch.nn.Module,
    npts: int = 2048,
    num_times: int = 4,
    device: Optional[torch.device] = None,
    time_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    质量守恒损失：惩罚不同时间的总密度变化。
    【注意】生长过程中质量（或固相比例）增加，该损失会导致模型拒绝生长，请勿使用。
    """
    if device is None:
        device = next(density_field.parameters()).device

    pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
    t0 = torch.rand(num_times - 1, device=device) * (time_range[1] - time_range[0]) + time_range[0]
    dt_min = 0.05
    dt = torch.rand_like(t0) * (1.0 - t0 - dt_min) + dt_min
    t1 = t0 + dt

    loss = torch.tensor(0.0, device=device)
    for i in range(num_times - 1):
        rho0 = density_field(pos, t0[i].view(1, 1).expand(npts, 1))
        rho1 = density_field(pos, t1[i].view(1, 1).expand(npts, 1))
        loss += F.mse_loss(rho1, rho0)

    return loss / (num_times - 1)


# ==============================================================
# 推荐使用的生长过程损失
# ==============================================================

def temporal_monotonicity_loss(
    density_field: torch.nn.Module,
    npts: int = 512,
    device: Optional[torch.device] = None,
    dt: float = 0.1,
) -> torch.Tensor:
    """
    时间单调性损失：惩罚密度随时间下降。
    对于生长过程（密度只增不减），该损失起正向约束作用。
    注意：若 density_field 返回的是 (density, prob)，请修改为提取 prob 再计算。
    """
    if device is None:
        device = next(density_field.parameters()).device

    pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
    t1 = torch.rand(1, device=device) * 0.9
    t2 = t1 + dt

    rho1 = density_field(pos, t1.expand(npts, 1))
    rho2 = density_field(pos, t2.expand(npts, 1))
    # 如果 density_field 返回 (density, prob)，则改为：
    # _, prob1 = density_field(pos, t1.expand(npts,1))
    # _, prob2 = density_field(pos, t2.expand(npts,1))
    # violation = F.relu(prob1 - prob2)
    violation = F.relu(rho1 - rho2)
    return violation.mean()


def spatial_tv_loss(
    density_field: torch.nn.Module,
    npts: int = 2048,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    空间全变分（TV）损失：抑制空间噪声，使密度场平滑。
    适用于任意时刻的密度场。
    """
    if device is None:
        device = next(density_field.parameters()).device

    pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
    pos.requires_grad_(True)
    t = torch.rand(1, device=device).expand(npts, 1)

    density = density_field(pos, t)
    # 若返回元组，取第一项
    if isinstance(density, tuple):
        density = density[0]

    grad_outputs = torch.ones_like(density)
    gradients = torch.autograd.grad(
        outputs=density,
        inputs=pos,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    tv = gradients.abs().mean()
    return tv


def volumetric_self_consistency_loss(
    density_field: torch.nn.Module,
    npts: int = 8192,
    device: Optional[torch.device] = None,
    time: float = 1.0,
) -> torch.Tensor:
    """
    体自洽损失：鼓励最终时刻密度场与稍早时刻密度场高度相关（可选）。
    """
    if device is None:
        device = next(density_field.parameters()).device

    pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
    t1 = torch.full((npts, 1), time, device=device)
    t2 = torch.full((npts, 1), max(0.0, time - 0.05), device=device)

    rho1 = density_field(pos, t1)
    rho2 = density_field(pos, t2)
    if isinstance(rho1, tuple):
        rho1 = rho1[0]
        rho2 = rho2[0]

    ncc = normed_correlation(rho1.squeeze(), rho2.squeeze())
    return 1.0 - ncc


# ==============================================================
# 背景/前景分离损失（推荐用于 X 射线投影重建）
# ==============================================================

def background_mse_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.95,
) -> torch.Tensor:
    """
    背景区域损失：强制预测值接近 1.0（无衰减）。
    背景定义为真值衰减 >= threshold 的像素（空白区域）。
    """
    mask = (gt >= threshold).float()
    diff = (pred - 1.0) * mask
    loss = (diff ** 2).sum() / (mask.sum() + 1e-8)
    return loss


def foreground_mse_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.95,
) -> torch.Tensor:
    """
    前景（有晶体）区域损失：强制预测值匹配真值衰减。
    前景定义为真值衰减 < threshold 的像素。
    """
    mask = (gt < threshold).float()
    diff = (pred - gt) * mask
    loss = (diff ** 2).sum() / (mask.sum() + 1e-8)
    return loss


# ==============================================================
# 可选：投影边缘损失（保留但训练时通常被上述分离损失替代）
# ==============================================================

def projection_edge_loss(
    pred_attenuation: torch.Tensor,   # (H,W) 或 (B,H,W)
    gt_attenuation: torch.Tensor      # 相同形状
) -> torch.Tensor:
    """
    投影边缘损失：比较预测投影与真实投影的边缘（轮廓）。
    使用 Sobel 算子提取梯度幅值，然后计算 MSE。
    """
    if pred_attenuation.dim() == 2:
        pred_attenuation = pred_attenuation.unsqueeze(0).unsqueeze(0)
        gt_attenuation   = gt_attenuation.unsqueeze(0).unsqueeze(0)
    elif pred_attenuation.dim() == 3:
        pred_attenuation = pred_attenuation.unsqueeze(1)
        gt_attenuation   = gt_attenuation.unsqueeze(1)

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=pred_attenuation.dtype, device=pred_attenuation.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=pred_attenuation.dtype, device=pred_attenuation.device
    ).view(1, 1, 3, 3)

    gx_pred = F.conv2d(pred_attenuation, sobel_x, padding=1)
    gy_pred = F.conv2d(pred_attenuation, sobel_y, padding=1)
    edge_pred = torch.sqrt(gx_pred ** 2 + gy_pred ** 2 + 1e-8)

    gx_gt = F.conv2d(gt_attenuation, sobel_x, padding=1)
    gy_gt = F.conv2d(gt_attenuation, sobel_y, padding=1)
    edge_gt = torch.sqrt(gx_gt ** 2 + gy_gt ** 2 + 1e-8)

    return F.mse_loss(edge_pred, edge_gt)