"""
hash_encoder.py - 纯 PyTorch 实现的 4D 哈希网格编码器
不依赖 tinycudann，可直接替代 tcnn.Encoding
"""

import torch
import torch.nn as nn

class HashGridEncoder4D(nn.Module):
    def __init__(
        self,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 1 << log2_hashmap_size

        # 每一层的分辨率按几何级数增长（与 Instant NGP 一致）
        growth_factor = (max_resolution / base_resolution) ** (1.0 / (n_levels - 1))
        resolutions = [int(base_resolution * (growth_factor ** i)) for i in range(n_levels)]
        self.resolutions = torch.tensor(resolutions, dtype=torch.int32)

        # 为每一层创建独立哈希嵌入表
        self.embeddings = nn.ParameterList()
        for _ in range(n_levels):
            emb = nn.Parameter(torch.zeros(self.hashmap_size, n_features_per_level))
            nn.init.uniform_(emb, a=-1e-4, b=1e-4)
            self.embeddings.append(emb)

        # 用于哈希计算的4个大质数
        self.register_buffer('primes', torch.tensor(
            [1, 2654435761, 805459861, 3674653429], dtype=torch.int64
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, 4]，坐标已经归一化到 [0,1]
        返回: [N, n_levels * n_features]
        """
        N = x.shape[0]
        outputs = []
        for level in range(self.n_levels):
            res = float(self.resolutions[level])
            # 连续网格坐标
            pos = x * res
            idx0 = torch.floor(pos).long()
            frac = pos - idx0.float()

            # 确保索引不超过边界
            idx0 = torch.clamp(idx0, 0, int(res) - 2)

            # 16 个顶点的四线性插值权重
            w0 = (1 - frac[:, 0]) * (1 - frac[:, 1]) * (1 - frac[:, 2]) * (1 - frac[:, 3])
            w1 = frac[:, 0] * (1 - frac[:, 1]) * (1 - frac[:, 2]) * (1 - frac[:, 3])
            w2 = (1 - frac[:, 0]) * frac[:, 1] * (1 - frac[:, 2]) * (1 - frac[:, 3])
            w3 = frac[:, 0] * frac[:, 1] * (1 - frac[:, 2]) * (1 - frac[:, 3])
            w4 = (1 - frac[:, 0]) * (1 - frac[:, 1]) * frac[:, 2] * (1 - frac[:, 3])
            w5 = frac[:, 0] * (1 - frac[:, 1]) * frac[:, 2] * (1 - frac[:, 3])
            w6 = (1 - frac[:, 0]) * frac[:, 1] * frac[:, 2] * (1 - frac[:, 3])
            w7 = frac[:, 0] * frac[:, 1] * frac[:, 2] * (1 - frac[:, 3])
            w8 = (1 - frac[:, 0]) * (1 - frac[:, 1]) * (1 - frac[:, 2]) * frac[:, 3]
            w9 = frac[:, 0] * (1 - frac[:, 1]) * (1 - frac[:, 2]) * frac[:, 3]
            w10 = (1 - frac[:, 0]) * frac[:, 1] * (1 - frac[:, 2]) * frac[:, 3]
            w11 = frac[:, 0] * frac[:, 1] * (1 - frac[:, 2]) * frac[:, 3]
            w12 = (1 - frac[:, 0]) * (1 - frac[:, 1]) * frac[:, 2] * frac[:, 3]
            w13 = frac[:, 0] * (1 - frac[:, 1]) * frac[:, 2] * frac[:, 3]
            w14 = (1 - frac[:, 0]) * frac[:, 1] * frac[:, 2] * frac[:, 3]
            w15 = frac[:, 0] * frac[:, 1] * frac[:, 2] * frac[:, 3]

            weights = torch.stack([
                w0, w1, w2, w3, w4, w5, w6, w7,
                w8, w9, w10, w11, w12, w13, w14, w15
            ], dim=1)  # [N, 16]

            # 16 个顶点的索引偏移
            offsets = torch.tensor([
                [0,0,0,0], [1,0,0,0], [0,1,0,0], [1,1,0,0],
                [0,0,1,0], [1,0,1,0], [0,1,1,0], [1,1,1,0],
                [0,0,0,1], [1,0,0,1], [0,1,0,1], [1,1,0,1],
                [0,0,1,1], [1,0,1,1], [0,1,1,1], [1,1,1,1]
            ], device=x.device)

            corner_indices = idx0.unsqueeze(1) + offsets.unsqueeze(0)  # [N,16,4]

            # 哈希计算：每个维度乘以一个大质数并异或
            corner_indices = corner_indices.to(torch.int64)
            hash_val = torch.zeros(N, 16, dtype=torch.int64, device=x.device)
            for d in range(4):
                hash_val ^= corner_indices[..., d] * self.primes[d]
            hash_val = hash_val % self.hashmap_size  # [N,16]

            # 查表并加权求和
            emb = self.embeddings[level]  # [H, F]
            features = emb[hash_val]      # [N,16,F]
            out_level = (features * weights.unsqueeze(-1)).sum(dim=1)  # [N, F]
            outputs.append(out_level)

        return torch.cat(outputs, dim=-1)  # [N, n_levels * F]