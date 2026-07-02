"""
ray_utils.py - 光线生成与采样工具
支持正交相机光线生成和沿光线均匀/分层采样。
完全独立，不依赖 nerfstudio。
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union


# ----------------------------------------------------------------------
# 光线数据结构（可选：用字典或简单类）
# ----------------------------------------------------------------------
class RayBundle:
    """简单的光线束类，用于组织光线参数。"""
    def __init__(
        self,
        origins: torch.Tensor,      # (N, 3)
        directions: torch.Tensor,   # (N, 3)
        pixel_area: Optional[torch.Tensor] = None,  # (N, 1)
        near: Optional[torch.Tensor] = None,        # (N, 1)
        far: Optional[torch.Tensor] = None,         # (N, 1)
        times: Optional[torch.Tensor] = None,       # (N, 1)
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.origins = origins
        self.directions = directions
        self.pixel_area = pixel_area
        self.near = near
        self.far = far
        self.times = times
        self.metadata = metadata or {}

    def __len__(self):
        return self.origins.shape[0]

    def to(self, device: torch.device):
        """将张量移动到指定设备。"""
        self.origins = self.origins.to(device)
        self.directions = self.directions.to(device)
        if self.pixel_area is not None:
            self.pixel_area = self.pixel_area.to(device)
        if self.near is not None:
            self.near = self.near.to(device)
        if self.far is not None:
            self.far = self.far.to(device)
        if self.times is not None:
            self.times = self.times.to(device)
        return self

    def flatten(self):
        """展平所有维度为 (total_rays, ...)。"""
        self.origins = self.origins.reshape(-1, 3)
        self.directions = self.directions.reshape(-1, 3)
        if self.pixel_area is not None:
            self.pixel_area = self.pixel_area.reshape(-1, 1)
        if self.near is not None:
            self.near = self.near.reshape(-1, 1)
        if self.far is not None:
            self.far = self.far.reshape(-1, 1)
        if self.times is not None:
            self.times = self.times.reshape(-1, 1)
        return self


class RaySamples:
    """沿光线采样点的简单容器。"""
    def __init__(
        self,
        positions: torch.Tensor,   # (N, num_samples, 3)
        deltas: torch.Tensor,      # (N, num_samples, 1)
        near: Optional[torch.Tensor] = None,  # (N, 1)
        far: Optional[torch.Tensor] = None,
        times: Optional[torch.Tensor] = None,
    ):
        self.positions = positions
        self.deltas = deltas
        self.near = near
        self.far = far
        self.times = times

    def to(self, device):
        self.positions = self.positions.to(device)
        self.deltas = self.deltas.to(device)
        if self.near is not None:
            self.near = self.near.to(device)
        if self.far is not None:
            self.far = self.far.to(device)
        if self.times is not None:
            self.times = self.times.to(device)
        return self


# ----------------------------------------------------------------------
# 正交相机光线生成（ORTHOPHOTO）
# ----------------------------------------------------------------------
def generate_ortho_rays(
    c2w: torch.Tensor,           # (3, 4) 或 (4, 4)
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    height: int,
    width: int,
    device: Optional[torch.device] = None,
    pixel_offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成正交投影相机的光线原点和方向。

    正交相机模型：
        - 所有光线方向相同，等于 -c2w 的第三列（即相机坐标系的 -Z 方向）在世界的表示。
        - 光线原点：世界坐标系中，根据像素坐标（归一化到 [-1,1] 或按 fx/fy 缩放）得到。
          通常，像素坐标映射到世界坐标：原点 = c2w[:3,3] + R_world @ ( (u - cx)/fx, (v - cy)/fy, 0 )^T
          但为了对齐，我们采用与 nerfstudio 一致的方式：
          对于正交相机，光线原点 = camera_center + R_world @ pixel_world_coord
          其中 pixel_world_coord = ((u - cx)/fx, (v - cy)/fy, 0) ，注意这里假设图像平面在 z=0 处。

    返回:
        origins: (H*W, 3)
        directions: (H*W, 3) 所有光线方向相同。
    """
    if c2w.shape == (4, 4):
        c2w = c2w[:3, :4]
    device = device or c2w.device
    dtype = c2w.dtype

    # 相机中心（世界坐标）
    center = c2w[:, 3]  # (3,)

    # 旋转矩阵
    R = c2w[:, :3]  # (3,3)

    # 方向：相机坐标系下的 -Z 方向（0,0,-1）变换到世界
    # 注意：在 nerfstudio 的 ORTHOPHOTO 中，方向 = R @ (0,0,-1) = -R[:,2]
    direction = -R[:, 2]  # (3,)

    # 生成像素坐标网格 (u, v) 从 0 到 width-1, height-1
    u = torch.arange(width, device=device, dtype=dtype) + pixel_offset
    v = torch.arange(height, device=device, dtype=dtype) + pixel_offset
    u, v = torch.meshgrid(u, v, indexing='xy')  # shape: (H, W)
    u = u.reshape(-1)  # (H*W)
    v = v.reshape(-1)  # (H*W)

    # 图像平面坐标：归一化到世界坐标（假设焦距 fx, fy）
    # 像素坐标 (u,v) 映射到世界坐标： ( (u - cx)/fx, (v - cy)/fy, 0 )
    x = (u - cx) / fx
    y = (v - cy) / fy
    # 注意：在 nerfstudio 中，他们用 (x, y, 0) 然后乘以 R 并加上中心。
    # 但 x 和 y 方向可能与 c2w 的列对应，所以直接计算：
    # 原点 = center + R @ (x, y, 0)^T
    pixel_local = torch.stack([x, y, torch.zeros_like(x)], dim=-1)  # (N, 3)
    origins = center[None, :] + torch.matmul(pixel_local, R.T)  # (N, 3)

    directions = direction[None, :].expand(origins.shape[0], -1)  # (N, 3)

    return origins, directions


# ----------------------------------------------------------------------
# 沿光线采样
# ----------------------------------------------------------------------
def sample_points_uniform(
    origins: torch.Tensor,      # (N, 3)
    directions: torch.Tensor,   # (N, 3)
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 [near, far] 范围内沿光线均匀采样点。

    Args:
        origins: (N, 3) 光线原点
        directions: (N, 3) 光线方向（已归一化）
        near: 近平面距离
        far: 远平面距离
        num_samples: 采样点数
        perturb: 是否在采样区间内添加随机抖动（用于训练）

    Returns:
        positions: (N, num_samples, 3) 采样点在世界空间中的坐标
        deltas: (N, num_samples, 1) 相邻采样点之间的距离
    """
    N = origins.shape[0]
    device = origins.device
    dtype = origins.dtype

    # 生成采样深度值 (N, num_samples)
    bins = torch.linspace(near, far, num_samples + 1, device=device, dtype=dtype)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_centers = bin_centers.expand(N, -1)

    if perturb and (num_samples > 1):
        # 在每个区间内均匀随机采样
        bin_edges = bins.expand(N, -1)
        lower = bin_edges[:, :-1]
        upper = bin_edges[:, 1:]
        # 随机偏移
        offsets = torch.rand_like(lower) * (upper - lower)
        depths = lower + offsets
    else:
        depths = bin_centers

    # 计算 deltas
    deltas = torch.zeros_like(depths)
    if num_samples > 1:
        deltas[:, :-1] = depths[:, 1:] - depths[:, :-1]
        deltas[:, -1] = deltas[:, -2]  # 最后一段用前一段长度

    # 计算采样点位置： origin + depth * direction
    positions = origins[:, None, :] + depths[:, :, None] * directions[:, None, :]  # (N, num_samples, 3)

    return positions, deltas.unsqueeze(-1)  # deltas (N, num_samples, 1)


def sample_points_layered(
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
    num_proposal_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    分层采样（类似 Nerf 的分层采样，但无权重指导，仅均匀 + 小扰动）。
    这里简单实现均匀采样，后续可扩展为概率密度引导。

    目前直接调用 sample_points_uniform，保留此函数名以备将来扩展。
    """
    return sample_points_uniform(origins, directions, near, far, num_samples, perturb)


# ----------------------------------------------------------------------
# 辅助：从相机参数生成光线束
# ----------------------------------------------------------------------
def create_ray_bundle_from_camera(
    c2w: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    height: int,
    width: int,
    time: Optional[float] = None,
    near: float = 0.5,
    far: float = 2.0,
    device: Optional[torch.device] = None,
) -> RayBundle:
    """
    从相机参数创建一个完整的 RayBundle（正交相机）。

    Args:
        c2w: 相机到世界矩阵 (3,4) 或 (4,4)
        fx, fy, cx, cy: 内参
        height, width: 图像尺寸
        time: 时间戳（可选）
        near, far: 近远平面
        device: 设备

    Returns:
        RayBundle 对象
    """
    if device is None:
        device = c2w.device
    origins, directions = generate_ortho_rays(
        c2w, fx, fy, cx, cy, height, width, device
    )
    N = origins.shape[0]
    if time is not None:
        times = torch.full((N, 1), time, device=device, dtype=torch.float32)
    else:
        times = None
    near_t = torch.full((N, 1), near, device=device, dtype=torch.float32)
    far_t = torch.full((N, 1), far, device=device, dtype=torch.float32)
    # pixel_area: 对于正交相机，像素面积通常为 1/(fx*fy) 或简单的 1.0
    pixel_area = torch.full((N, 1), 1.0 / (fx * fy), device=device, dtype=torch.float32)

    return RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=pixel_area,
        near=near_t,
        far=far_t,
        times=times,
    )