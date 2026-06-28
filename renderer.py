"""
renderer.py - X 射线衰减渲染器
完全独立，不依赖 nerfstudio。
"""

from typing import Union, Optional, Literal, Tuple
import torch
from torch import nn, Tensor

# 简单颜色映射（仅支持常见颜色）
COLORS_DICT = {
    'white': (1.0, 1.0, 1.0),
    'black': (0.0, 0.0, 0.0),
    'red': (1.0, 0.0, 0.0),
    'green': (0.0, 1.0, 0.0),
    'blue': (0.0, 0.0, 1.0),
}

BackgroundColor = Union[
    Literal['white', 'black', 'red', 'green', 'blue'],
    Tuple[float, float, float],
    Tensor
]

class AttenuationRenderer(nn.Module):
    """
    X 射线衰减渲染器。
    根据沿射线的密度积分计算衰减，并可应用平场校正。
    """

    def __init__(self, background_color: BackgroundColor = 'white'):
        super().__init__()
        self.background_color = background_color

    def forward(
        self,
        densities: Tensor,     # [..., num_samples, 1] 或 [..., num_samples]
        deltas: Tensor,        # [..., num_samples, 1] 或 [..., num_samples]
    ) -> Tensor:
        """
        计算衰减。

        Args:
            densities: 沿射线的密度值。
            deltas: 相邻采样点之间的间距（路径长度）。

        Returns:
            衰减值，形状为 [..., 1]（如果输入是 [..., num_samples, 1]）或 [..., 1]。
        """
        # 确保形状为 [..., num_samples, 1]
        if densities.dim() == deltas.dim() - 1:
            densities = densities.unsqueeze(-1)
        if deltas.dim() == densities.dim() - 1:
            deltas = deltas.unsqueeze(-1)

        delta_density = deltas * densities
        acc = torch.sum(delta_density, dim=-2)  # 沿采样点求和
        attenuation = torch.exp(-acc)
        attenuation = torch.nan_to_num(attenuation)  # 处理 NaN/Inf
        return attenuation

    @staticmethod
    def merge_flat_field(
        attenuation: Tensor,
        flat_field: Tensor,
    ) -> Tensor:
        """
        将平场校正应用到衰减。

        Args:
            attenuation: 原始衰减值。
            flat_field: 平场参数（非负）。

        Returns:
            校正后的衰减。
        """
        flat_field = nn.functional.relu(flat_field)      # 确保非负
        flat_field = torch.exp(-flat_field)              # 转换为乘性因子
        return attenuation * flat_field

    @staticmethod
    def _get_color_rgb(color: BackgroundColor) -> Tensor:
        """将颜色规范转换为 RGB 张量。"""
        if isinstance(color, str):
            color = color.lower()
            if color in COLORS_DICT:
                return torch.tensor(COLORS_DICT[color], dtype=torch.float32)
            else:
                raise ValueError(f"Unknown color name: {color}")
        elif isinstance(color, (list, tuple)):
            return torch.tensor(color, dtype=torch.float32)
        elif isinstance(color, Tensor):
            return color.clone().detach().float()
        else:
            raise TypeError(f"Unsupported color type: {type(color)}")

    def blend_background(
        self,
        image: Tensor,          # [..., 3] 或 [..., 4] (RGBA)
        background_color: Optional[BackgroundColor] = None,
    ) -> Tensor:
        """
        将背景颜色混合到图像中（如果图像是 RGBA）。

        Args:
            image: RGB 或 RGBA 图像。
            background_color: 背景颜色（默认使用 self.background_color）。

        Returns:
            混合后的 RGB 图像。
        """
        if image.shape[-1] < 4:
            return image  # 无 alpha 通道，直接返回

        rgb = image[..., :3]
        alpha = image[..., 3:]

        if background_color is None:
            background_color = self.background_color
        bg_rgb = self._get_color_rgb(background_color).to(rgb.device)
        # 扩展形状以匹配 alpha
        bg_rgb = bg_rgb.expand_as(rgb)
        return rgb * alpha + bg_rgb * (1.0 - alpha)

    def blend_background_for_loss(
        self,
        pred_image: Tensor,
        pred_accumulation: Tensor,
        gt_image: Tensor,
        background_color: Optional[BackgroundColor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        为损失计算准备预测图像和真值图像（混合背景）。

        Args:
            pred_image: 预测的 RGB（未混合背景）。
            pred_accumulation: 预测的累积不透明度。
            gt_image: 真值图像（可能是 RGBA 或 RGB）。
            background_color: 用于混合的背景颜色。

        Returns:
            (混合后的预测图像, 混合后的真值图像)
        """
        if background_color is None:
            background_color = self.background_color

        # 混合预测图像
        bg_rgb = self._get_color_rgb(background_color).to(pred_image.device)
        bg_rgb = bg_rgb.expand_as(pred_image)
        pred_mixed = pred_image + bg_rgb * (1.0 - pred_accumulation)

        # 混合真值图像（如果是 RGBA）
        if gt_image.shape[-1] == 4:
            gt_rgb = gt_image[..., :3]
            gt_alpha = gt_image[..., 3:]
            gt_mixed = gt_rgb * gt_alpha + bg_rgb * (1.0 - gt_alpha)
        else:
            gt_mixed = gt_image

        return pred_mixed, gt_mixed