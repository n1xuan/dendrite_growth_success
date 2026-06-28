"""
losses.py - 4D 密度场损失函数（修正形状为标量，增加空间 TV）
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def normed_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mux = x.mean()
    muy = y.mean()
    dx = x - mux
    dy = y - muy
    return torch.sum(dx * dy) / (torch.sqrt(dx.pow(2).sum() * dy.pow(2).sum()) + 1e-8)


def temporal_smoothness_loss(
    density_field: torch.nn.Module,
    npts: int = 2048,
    n_pairs: int = 5,
    device: Optional[torch.device] = None,
    time_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    if device is None:
        device = next(density_field.parameters()).device

    pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
    loss = torch.tensor(0.0, device=device)          # ← 标量初始化

    for _ in range(n_pairs):
        t0 = torch.rand(1, device=device) * (time_range[1] - time_range[0] - 0.02) + time_range[0]
        dt = 0.02 * torch.rand(1, device=device) + 0.01
        t1 = t0 + dt
        rho0 = density_field(pos, t0.expand(npts, 1))
        rho1 = density_field(pos, t1.expand(npts, 1))
        loss += F.mse_loss(rho1, rho0)               # mse_loss 返回标量

    return loss / n_pairs


def mass_conservation_loss(
    density_field: torch.nn.Module,
    npts: int = 2048,
    num_times: int = 4,
    device: Optional[torch.device] = None,
    time_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    if device is None:
        device = next(density_field.parameters()).device

    pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4

    t0 = torch.rand(num_times - 1, device=device) * (time_range[1] - time_range[0]) + time_range[0]
    dt_min = 0.05
    dt = torch.rand_like(t0) * (1.0 - t0 - dt_min) + dt_min
    t1 = t0 + dt

    loss = torch.tensor(0.0, device=device)          # ← 标量初始化
    for i in range(num_times - 1):
        rho0 = density_field(pos, t0[i].view(1, 1).expand(npts, 1))
        rho1 = density_field(pos, t1[i].view(1, 1).expand(npts, 1))
        loss += F.mse_loss(rho1, rho0)

    return loss / (num_times - 1)


def temporal_monotonicity_loss(
    density_field: torch.nn.Module,
    npts: int = 512,
    device: Optional[torch.device] = None,
    dt: float = 0.1,
) -> torch.Tensor:
    if device is None:
        device = next(density_field.parameters()).device

    pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
    t1 = torch.rand(1, device=device) * 0.9
    t2 = t1 + dt

    rho1 = density_field(pos, t1.expand(npts, 1))
    rho2 = density_field(pos, t2.expand(npts, 1))
    violation = F.relu(rho1 - rho2)
    return violation.mean()                          # 已经返回标量


def volumetric_self_consistency_loss(
    density_field: torch.nn.Module,
    npts: int = 8192,
    device: Optional[torch.device] = None,
    time: float = 1.0,
) -> torch.Tensor:
    if device is None:
        device = next(density_field.parameters()).device

    pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4

    t1 = torch.full((npts, 1), time, device=device)
    t2 = torch.full((npts, 1), max(0.0, time - 0.05), device=device)

    rho1 = density_field(pos, t1)
    rho2 = density_field(pos, t2)

    ncc = normed_correlation(rho1.squeeze(), rho2.squeeze())
    return 1.0 - ncc                                   # 返回标量


def spatial_tv_loss(
    density_field: torch.nn.Module,
    npts: int = 2048,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = next(density_field.parameters()).device

    pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
    pos.requires_grad_(True)
    t = torch.rand(1, device=device).expand(npts, 1)

    density = density_field(pos, t)

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
    return tv                                          # 返回标量