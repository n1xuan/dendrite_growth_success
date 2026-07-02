"""
models.py - 4D 密度场（时间作为输入）
使用 4D 哈希网格或 MLP + 位置编码，直接输出密度值。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# 尝试导入 tinycudann
try:
    import tinycudann as tcnn
    HAS_TCNN = True
except ImportError:
    HAS_TCNN = False
    print("Warning: tinycudann not installed. Falling back to MLP with positional encoding.")


def trunc_exp(x: torch.Tensor) -> torch.Tensor:
    """Exponential with truncation for numerical stability."""
    return torch.exp(torch.clamp(x, -15.0, 15.0))


class TemporalDensityField(nn.Module):
    """
    4D 密度场 f(x, y, z, t) → density
    支持：
      - 4D 哈希网格（优先，需要 tcnn）
      - 回退：MLP + 空间位置编码 + 时间位置编码
    """
    def __init__(
        self,
        aabb: torch.Tensor,                     # [2, 3] 空间范围
        time_range: Tuple[float, float] = (0.0, 1.0),
        hash_config: Optional[dict] = None,
        average_init_density: float = 1.0,
        use_4d_hash: bool = True,              # 是否使用真正的 4D 哈希
        num_time_freqs: int = 6,               # 时间位置编码频率（回退用）
    ):
        super().__init__()
        self.register_buffer('aabb', aabb)
        self.time_min, self.time_max = time_range
        self.average_init_density = average_init_density
        self.use_hash = HAS_TCNN and use_4d_hash

        if hash_config is None:
            hash_config = {}

        if self.use_hash:
            # 4D 哈希网格
            per_level_scale = (hash_config.get('max_res', 2048) / hash_config.get('base_res', 16)) ** (1.0 / (hash_config.get('n_levels', 16) - 1))
            self.encoder = tcnn.Encoding(
                n_input_dims=4,  # (x, y, z, t)
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": hash_config.get('n_levels', 16),
                    "n_features_per_level": hash_config.get('n_features_per_level', 2),
                    "log2_hashmap_size": hash_config.get('log2_hashmap_size', 19),
                    "base_resolution": hash_config.get('base_res', 16),
                    "per_level_scale": per_level_scale,
                }
            )
            self.density_net = tcnn.Network(
                n_input_dims=self.encoder.n_output_dims,
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                    "activation": "ReLU",
                    "output_activation": "None",
                }
            )
        else:
            # 回退：MLP + 位置编码（空间 3D + 时间 1D）
            self.num_time_freqs = num_time_freqs
            spatial_in_dim = 3 * (2 * 6 + 1)       # 3 * 13 = 39
            time_in_dim = 1 * (2 * num_time_freqs + 1)
            total_in_dim = spatial_in_dim + time_in_dim
            self.mlp = nn.Sequential(
                nn.Linear(total_in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

    def forward(self, positions: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [N, 3] 空间坐标 (x, y, z)
            times:     [N, 1] 或 [N]   时间坐标
        Returns:
            density: [N, 1]
        """
        if times.dim() == 1:
            times = times.unsqueeze(-1)

        if self.use_hash:
            # 归一化空间坐标到 [0,1]
            normalized_xyz = (positions - self.aabb[0]) / (self.aabb[1] - self.aabb[0])
            normalized_xyz = torch.clamp(normalized_xyz, 0.0, 1.0)
            # 归一化时间
            normalized_t = (times - self.time_min) / (self.time_max - self.time_min)
            normalized_t = torch.clamp(normalized_t, 0.0, 1.0)
            # 拼接为 4D 输入
            input_4d = torch.cat([normalized_xyz, normalized_t], dim=-1)
            h = self.encoder(input_4d)
            density_before = self.density_net(h)
        else:
            # 空间位置编码（6 个频率）
            pe_spatial = []
            for freq in range(6):
                pe_spatial.append(torch.sin(2.0 ** freq * math.pi * positions))
                pe_spatial.append(torch.cos(2.0 ** freq * math.pi * positions))
            pe_spatial.append(positions)
            pe_spatial = torch.cat(pe_spatial, dim=-1)  # [N, 39]

            # 时间位置编码
            pe_time = []
            for freq in range(self.num_time_freqs):
                pe_time.append(torch.sin(2.0 ** freq * math.pi * times))
                pe_time.append(torch.cos(2.0 ** freq * math.pi * times))
            pe_time.append(times)
            pe_time = torch.cat(pe_time, dim=-1)  # [N, 13]

            features = torch.cat([pe_spatial, pe_time], dim=-1)
            density_before = self.mlp(features)

        density = self.average_init_density * trunc_exp(density_before - 2.0)
        return density