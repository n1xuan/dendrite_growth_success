"""
sequence_selection.py - 从候选密度体素中选择最连贯的时间序列
使用动态规划，使相邻帧的密度差异最小，构建具有包络性质的连续生长过程。
"""

import numpy as np
from pathlib import Path
import argparse
import json


def select_optimal_sequence(all_candidates, lambda_vol=0.1, cost_type='mse'):
    """
    从每帧的多候选密度中选出最优序列。

    Args:
        all_candidates: list of lists, 结构为 [num_frames][num_candidates] -> 3D numpy array
                        每个密度体素形状相同 [D, H, W]
        lambda_vol: 体积变化惩罚系数 (默认为0.1，越大越鼓励体积守恒)
        cost_type: 代价类型
            'mse' - 密度体素间的均方误差
            'mae' - 平均绝对误差

    Returns:
        selected_path: list of int, 每帧选中的候选索引
    """
    num_frames = len(all_candidates)
    K = len(all_candidates[0])  # 每帧候选数

    # 动态规划表
    dp = [np.zeros(K) for _ in range(num_frames)]
    parent = [np.zeros(K, dtype=int) for _ in range(num_frames)]

    # 第0帧无前驱，代价为0
    dp[0] = np.zeros(K)
    parent[0] = -np.ones(K, dtype=int)

    for t in range(1, num_frames):
        for j in range(K):
            best_cost = np.inf
            best_i = 0
            rho_curr_flat = all_candidates[t][j].ravel()

            for i in range(K):
                rho_prev_flat = all_candidates[t-1][i].ravel()

                # 密度差异
                if cost_type == 'mse':
                    diff = np.mean((rho_prev_flat - rho_curr_flat) ** 2)
                elif cost_type == 'mae':
                    diff = np.mean(np.abs(rho_prev_flat - rho_curr_flat))
                else:
                    raise ValueError(f"Unknown cost_type: {cost_type}")

                # 体积变化惩罚（可选）
                if lambda_vol > 0:
                    vol_prev = np.sum(rho_prev_flat)
                    vol_curr = np.sum(rho_curr_flat)
                    vol_diff = np.abs(vol_prev - vol_curr) / rho_prev_flat.size
                    cost = diff + lambda_vol * vol_diff
                else:
                    cost = diff

                total = dp[t-1][i] + cost
                if total < best_cost:
                    best_cost = total
                    best_i = i

            dp[t][j] = best_cost
            parent[t][j] = best_i

    # 回溯
    selected_path = np.zeros(num_frames, dtype=int)
    selected_path[-1] = np.argmin(dp[-1])
    for t in range(num_frames - 2, -1, -1):
        selected_path[t] = parent[t+1][selected_path[t+1]]

    return selected_path.tolist()


def load_candidates_from_npz(npz_path):
    """
    从 all_candidates.npz 加载候选密度体素。
    文件格式：键 frame00, frame01, ... 每个值形状 [K, D, H, W]
    返回 list of lists。
    """
    data = np.load(npz_path)
    # 按帧排序
    keys = sorted([k for k in data.files if k.startswith('frame')])
    all_candidates = []
    for key in keys:
        frame_data = data[key]  # [K, D, H, W]
        # 拆成列表
        all_candidates.append([frame_data[i] for i in range(frame_data.shape[0])])
    print(f"从 {npz_path} 加载了 {len(all_candidates)} 帧，每帧 {len(all_candidates[0])} 个候选")
    return all_candidates


def main():
    parser = argparse.ArgumentParser(
        description="从候选密度体素中选择最优演化序列"
    )
    parser.add_argument('--input', type=str, required=True,
                        help='候选密度 npz 文件路径 (如 all_candidates.npz)')
    parser.add_argument('--output', type=str, default='selected_sequence.json',
                        help='输出序列选择结果的文件路径')
    parser.add_argument('--lambda_vol', type=float, default=0.1,
                        help='体积变化惩罚系数 (0 表示不惩罚)')
    parser.add_argument('--cost_type', type=str, default='mse',
                        choices=['mse', 'mae'], help='密度差异度量方式')

    args = parser.parse_args()

    all_candidates = load_candidates_from_npz(args.input)
    selected = select_optimal_sequence(all_candidates,
                                       lambda_vol=args.lambda_vol,
                                       cost_type=args.cost_type)

    print("选出的最优序列:", selected)

    # 保存为 JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({'selected_indices': selected,
                   'lambda_vol': args.lambda_vol,
                   'cost_type': args.cost_type}, f, indent=2)
    print(f"结果已保存至 {output_path}")


if __name__ == "__main__":
    main()