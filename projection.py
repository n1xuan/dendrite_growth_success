"""
projection.py - 可微平行束投影算子
完全复现 generate_synthetic_dendrite_v6.py 中的投影几何，支持梯度反向传播。
假设输入体积轴顺序为 [X, Y, Z]，旋转绕 Z 轴进行，沿 X 轴积分。
"""

import torch
import torch.nn.functional as F
import math


def parallel_beam_project_torch(volume, angle_deg, output_size=None):
    """
    对密度体素进行平行束投影，返回线积分图像。

    几何约定（与数据生成脚本完全一致）：
        - 体素位于 [-1, 1]^3 的规则网格中，形状为 [X, Y, Z]（X=Y=Z）
        - 投影方向 (cosθ, sinθ, 0)，θ = angle_deg（度）
        - 通过将体素绕 Z 轴旋转 -θ，使投影方向对齐到 X 轴，然后沿 X 轴求和
        - 求和后经过转置 + 垂直翻转匹配图像坐标系（u 向右，v 向下）

    Args:
        volume: torch.Tensor, 形状 [X, Y, Z], 密度值 >= 0
        angle_deg: float, 投影方位角（度）
        output_size: tuple (H_out, W_out) 或 None，若指定则调整输出尺寸

    Returns:
        projection: torch.Tensor, 形状 [H_out, W_out] 或 [Z, Y]（取决于 output_size）
                    线积分值（未乘以体素尺寸）
    """
    N = volume.shape[0]  # 假设立方体，X=Y=Z=N
    device = volume.device
    dtype = volume.dtype

    # -------------- 1. 旋转体素 --------------
    # 与 scipy.ndimage.rotate(volume, -angle_deg, axes=(0,1), reshape=False) 等价
    theta_rad = math.radians(-angle_deg)  # 注意取负
    cos_a, sin_a = math.cos(theta_rad), math.sin(theta_rad)

    # 逆旋转矩阵（用于 grid_sample 采样原体积）
    # 绕 Z 轴的逆旋转矩阵为：
    #   [ cosθ  sinθ  0]
    #   [-sinθ  cosθ  0]
    #   [   0     0   1]
    inv_rot = torch.tensor([
        [cos_a, sin_a, 0.0],
        [-sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ], device=device, dtype=dtype)

    # 构造采样网格：形状 [1, X, Y, Z, 3]，坐标归一化到 [-1, 1]
    grid = F.affine_grid(
        torch.eye(3, 4, device=device).unsqueeze(0),
        [1, 1, N, N, N],
        align_corners=False
    )

    # 应用逆旋转：将网格点变换到原体积坐标系
    grid_rot = grid @ inv_rot.T  # [1, X, Y, Z, 3]

    # 对体积进行采样（双线性插值，外部补零）
    volume_4d = volume.view(1, 1, N, N, N)  # [1, 1, X, Y, Z]
    rotated = F.grid_sample(
        volume_4d, grid_rot,
        mode='bilinear',
        align_corners=False,
        padding_mode='zeros'
    )
    rotated = rotated[0, 0]  # [X, Y, Z]

    # -------------- 2. 沿 X 轴积分（光束方向）--------------
    line_integral = torch.sum(rotated, dim=0)  # [Y, Z]  (X 轴被消除)

    # -------------- 3. 图像坐标转换 --------------
    # 原生成代码中： projection = line_integral.T[::-1, :]
    # 即先转置（将 [Y, Z] 变为 [Z, Y]），然后垂直翻转行（索引 0 对应最大 z）
    # 最终得到 [Z, Y] 图像（高=Z，宽=Y），其中行向下增加，列向右增加
    projection = line_integral.T.flip(0)  # [Z, Y]

    # -------------- 4. 调整尺寸 --------------
    if output_size is not None:
        # 将 [Z, Y] 视为 [1, 1, Z, Y] 进行缩放
        projection = projection.unsqueeze(0).unsqueeze(0)  # [1, 1, Z, Y]
        projection = F.interpolate(
            projection,
            size=output_size,  # 注意 output_size 应为 (H_out, W_out)
            mode='bilinear',
            align_corners=False
        )
        projection = projection[0, 0]  # [H_out, W_out]

    return projection


def parallel_beam_project_attenuation(volume, angle_deg, voxel_size, atten_k=1.5, output_size=None):
    """
    便捷函数：从密度体素直接计算 Beer-Lambert 衰减图像。

    Args:
        volume: [X, Y, Z] 密度体素
        angle_deg: 投影角度
        voxel_size: 体素边长（如 2.0 / spatial_res）
        atten_k: 衰减系数（默认 1.5，与生成数据一致）
        output_size: 输出图像尺寸

    Returns:
        attenuation: [H_out, W_out] 衰减值，范围 (0, 1]
    """
    proj_line = parallel_beam_project_torch(volume, angle_deg, output_size)
    line_integral = proj_line * voxel_size
    attenuation = torch.exp(-atten_k * line_integral)
    return attenuation