"""
export.py - 导出 4D 密度场体积、渲染图像，并可提取 STL 网格
从检查点加载 TemporalDensityField，支持：
  - 导出 3D 体积 (npz)
  - 渲染 2D 投影图像
  - 导出 STL 网格（需要 scikit-image）
  - 3D 体积评估（PSNR / 3D-SSIM），需提供 ground truth 目录
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import argparse
from PIL import Image

from models import TemporalDensityField
from renderer import AttenuationRenderer
from ray_utils import create_ray_bundle_from_camera, sample_points_uniform, RayBundle
from data_loader import XRayDataset

# 尝试导入 marching_cubes 用于 STL 导出
try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image 未安装，无法导出 STL。请运行: pip install scikit-image")


# ------------------------------------------------------------
# 评估函数
# ------------------------------------------------------------
def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 20.0 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0,
                 K1: float = 0.01, K2: float = 0.03) -> float:
    from scipy.ndimage import uniform_filter
    gt, pred = gt.astype(np.float64), pred.astype(np.float64)
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    mu1 = uniform_filter(pred, size=3, mode='reflect')
    mu2 = uniform_filter(gt, size=3, mode='reflect')
    sigma1_sq = uniform_filter(pred ** 2, 3, mode='reflect') - mu1 ** 2
    sigma2_sq = uniform_filter(gt ** 2, 3, mode='reflect') - mu2 ** 2
    sigma12 = uniform_filter(pred * gt, 3, mode='reflect') - mu1 * mu2
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    return float(ssim_map.mean())


def compute_3d_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 3.0) -> float:
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 20.0 * np.log10(data_range / np.sqrt(mse))


def compute_3d_ssim(gt: np.ndarray, pred: np.ndarray,
                    data_range: float = 3.0,
                    window_size: int = 3) -> float:
    from scipy.ndimage import uniform_filter
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = uniform_filter(pred, size=window_size, mode='reflect')
    mu2 = uniform_filter(gt,   size=window_size, mode='reflect')
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = uniform_filter(pred ** 2, window_size, mode='reflect') - mu1_sq
    sigma2_sq = uniform_filter(gt ** 2,   window_size, mode='reflect') - mu2_sq
    sigma12   = uniform_filter(pred * gt, window_size, mode='reflect') - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return float(ssim_map.mean())


def load_gt_volume(gt_dir: str, time: float) -> np.ndarray:
    gt_path = Path(gt_dir)
    if not gt_path.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    import re
    indices = []
    for f in gt_path.glob("volume_*.npz"):
        m = re.search(r"volume_(\d+)\.npz", f.name)
        if m:
            indices.append(int(m.group(1)))
    if not indices:
        raise RuntimeError(f"No volume_*.npz files found in {gt_dir}")
    n_frames = max(indices) + 1
    frame_idx = int(round(time * (n_frames - 1)))
    frame_idx = max(0, min(frame_idx, n_frames - 1))
    vol_file = gt_path / f"volume_{frame_idx:02d}.npz"
    if not vol_file.exists():
        raise FileNotFoundError(f"Ground truth volume not found: {vol_file}")
    data = np.load(vol_file)
    if 'volume' in data:
        return data['volume']
    return data[list(data.keys())[0]]


# ------------------------------------------------------------
# STL 导出工具
# ------------------------------------------------------------
def write_ascii_stl(vertices, faces, filepath, solid_name="mesh"):
    """将顶点和面片写入 ASCII STL 文件"""
    with open(filepath, 'w') as f:
        f.write(f"solid {solid_name}\n")
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 0.0])
            f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def export_volume_to_stl(volume, filepath, threshold=None):
    """从 3D 体积提取等值面并保存为 STL"""
    if not HAS_SKIMAGE:
        print("错误：需要 scikit-image 库。请运行: pip install scikit-image")
        return
    if threshold is None:
        threshold = volume.max() * 0.4
        print(f"自动设置 STL 阈值 = {threshold:.4f} (最大值的 40%)")
    verts, faces, _, _ = marching_cubes(volume, level=threshold, spacing=(1, 1, 1))
    # 体素坐标映射到世界坐标 [-1,1]
    res = volume.shape[0]
    verts_world = verts * (2.0 / (res - 1)) - 1.0
    write_ascii_stl(verts_world, faces, filepath)
    print(f"STL 文件已保存: {filepath}")


# ------------------------------------------------------------
# 导出器
# ------------------------------------------------------------
class Exporter:
    def __init__(self, model: TemporalDensityField, device, near=1.5, far=4.5,
                 num_samples=128, render_chunk_size=4096):
        self.model = model.to(device).eval()
        self.device = device
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.render_chunk_size = render_chunk_size
        self.renderer = AttenuationRenderer(background_color='white')

    @torch.no_grad()
    def export_volume_grid(self, resolution=128, time=0.0, output_path="volume.npz"):
        lin = torch.linspace(-1, 1, resolution, device=self.device)
        X, Y, Z = torch.meshgrid(lin, lin, lin, indexing='ij')
        positions = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
        batch_size = 16384
        density_list = []
        for i in range(0, positions.shape[0], batch_size):
            pos = positions[i:i+batch_size]
            t = torch.full((pos.shape[0], 1), time, device=self.device)
            density_list.append(self.model(pos, t).cpu().numpy())
        density_grid = np.concatenate(density_list).reshape(resolution, resolution, resolution)
        output_path = Path(output_path)
        if output_path.suffix != '.npz':
            output_path = output_path.with_suffix('.npz')
        np.savez_compressed(output_path, volume=density_grid)
        print(f"体积网格导出到 {output_path}")
        return density_grid

    @torch.no_grad()
    def render_image_from_ray_bundle(self, ray_bundle: RayBundle) -> torch.Tensor:
        N = ray_bundle.origins.shape[0]
        positions, deltas = sample_points_uniform(
            ray_bundle.origins, ray_bundle.directions,
            near=self.near, far=self.far, num_samples=self.num_samples, perturb=False)
        times = ray_bundle.times if ray_bundle.times is not None else torch.zeros(N, 1, device=self.device)
        density = self.model(positions.reshape(-1, 3), times.expand(-1, self.num_samples).reshape(-1, 1))
        density = density.reshape(N, self.num_samples, 1)
        return self.renderer(density, deltas)

    def render_dataset(self, dataset, output_dir, max_frames=None, compute_metrics=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        num_frames = min(len(dataset), max_frames or len(dataset))
        psnr_vals, ssim_vals = [], []
        for idx in range(num_frames):
            sample = dataset[idx]
            c2w = sample['c2w'].to(self.device)
            h, w = sample['height'], sample['width']
            ray_bundle = create_ray_bundle_from_camera(
                c2w, sample['fx'], sample['fy'], sample['cx'], sample['cy'],
                h, w, time=sample['time'], near=self.near, far=self.far, device=self.device)
            ray_bundle.flatten()
            total_rays = ray_bundle.origins.shape[0]
            pred_chunks = []
            for start in range(0, total_rays, self.render_chunk_size):
                end = min(start + self.render_chunk_size, total_rays)
                sub = RayBundle(origins=ray_bundle.origins[start:end],
                                directions=ray_bundle.directions[start:end],
                                times=ray_bundle.times[start:end] if ray_bundle.times is not None else None)
                pred_chunks.append(self.render_image_from_ray_bundle(sub).cpu())
            pred_img = torch.cat(pred_chunks).reshape(h, w).numpy()
            pred_uint8 = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(pred_uint8, mode='L').save(output_dir / f"pred_{idx:04d}.png")

            if sample['image'] is not None:
                gt_img = sample['image'].cpu().numpy().reshape(h, w)
                gt_uint8 = (np.clip(gt_img, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(gt_uint8, mode='L').save(output_dir / f"gt_{idx:04d}.png")

                if compute_metrics:
                    psnr_vals.append(compute_psnr(gt_img, pred_img))
                    ssim_vals.append(compute_ssim(gt_img, pred_img))
                    print(f"帧 {idx:04d} (t={sample['time']:.3f}) | PSNR: {psnr_vals[-1]:.2f} dB | SSIM: {ssim_vals[-1]:.4f}")
            else:
                print(f"帧 {idx:04d} (t={sample['time']:.3f}) -> {output_dir / f'pred_{idx:04d}.png'} (无真值)")

        if psnr_vals:
            avg_psnr = np.mean(psnr_vals)
            std_psnr = np.std(psnr_vals)
            avg_ssim = np.mean(ssim_vals)
            std_ssim = np.std(ssim_vals)
            print(f"\n========== 重建质量 (2D) ==========")
            print(f"PSNR 均值: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
            print(f"SSIM 均值: {avg_ssim:.4f} ± {std_ssim:.4f}")
            print(f"===================================\n")


# ------------------------------------------------------------
# 检查点加载
# ------------------------------------------------------------
def load_checkpoint(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state['config']
    aabb = torch.tensor([config['aabb_min'], config['aabb_max']], dtype=torch.float32)
    model = TemporalDensityField(
        aabb=aabb,
        time_range=(0.0, 1.0),
        hash_config={
            'base_res': config.get('hash_base_res', 16),
            'max_res': config.get('hash_max_res', 2048),
            'n_levels': config.get('hash_n_levels', 16),
            'n_features_per_level': config.get('hash_n_features', 2),
            'log2_hashmap_size': config.get('hash_log2_size', 19),
        },
        average_init_density=config.get('average_init_density', 1.0),
        use_4d_hash=config.get('use_4d_hash', True),
    ).to(device)
    model.load_state_dict({k.replace('model.', ''): v for k, v in state['model_state_dict'].items()})
    return model, config


# ------------------------------------------------------------
# 主程序
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="导出 4D 密度场体积、渲染图像和 STL")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='exports')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--times', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    parser.add_argument('--export_volume', action='store_true')
    parser.add_argument('--export_stl', action='store_true', help='导出 STL 网格')
    parser.add_argument('--stl_threshold', type=float, default=None,
                        help='STL 等值面提取阈值，None 则自动取最大值的 40%')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render_data', default=None)
    parser.add_argument('--render_max_frames', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--downscale', type=float, default=1.0)
    parser.add_argument('--render_chunk_size', type=int, default=4096)
    parser.add_argument('--no_metrics', action='store_true')

    # 3D 评估参数
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Ground truth 体积目录，用于 3D 评估')
    parser.add_argument('--gt_data_range', type=float, default=3.0,
                        help='3D 评估时的信号动态范围')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, config = load_checkpoint(args.checkpoint, device)
    near, far = config.get('near_plane', 1.5), config.get('far_plane', 4.5)

    exporter = Exporter(model, device, near=near, far=far, render_chunk_size=args.render_chunk_size)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 3D 体积导出 ----
    if args.export_volume:
        print("\n=== 导出 3D 体积 ===")
        for t in args.times:
            grid = exporter.export_volume_grid(args.resolution, t, out_root / f"volume_t{t:.3f}.npz")

            # STL 导出
            if args.export_stl and HAS_SKIMAGE:
                stl_path = out_root / f"volume_t{t:.3f}.stl"
                export_volume_to_stl(grid, stl_path, threshold=args.stl_threshold)

            # 3D 评估
            if args.gt_dir:
                try:
                    gt_vol = load_gt_volume(args.gt_dir, t)
                    if grid.shape != gt_vol.shape:
                        from scipy.ndimage import zoom
                        print(f"  预测体积 shape {grid.shape} 与 GT shape {gt_vol.shape} 不一致，正在缩放...")
                        zoom_factors = tuple(g / p for p, g in zip(grid.shape, gt_vol.shape))
                        grid = zoom(grid, zoom_factors, order=1)
                    data_range = args.gt_data_range
                    psnr_val = compute_3d_psnr(gt_vol, grid, data_range=data_range)
                    ssim_val = compute_3d_ssim(gt_vol, grid, data_range=data_range)
                    print(f"  t={t:.3f} | 3D PSNR: {psnr_val:.2f} dB | 3D-SSIM: {ssim_val:.4f}")
                except Exception as e:
                    print(f"  t={t:.3f} 3D 评估失败: {e}")

    # ---- 2D 渲染 ----
    if args.render:
        if not args.render_data:
            print("错误: 未指定 --render_data 参数。")
        else:
            dataset = XRayDataset(
                args.render_data,
                downscale_factor=args.downscale,
                load_images=True,
                preload_images=False
            )
            exporter.render_dataset(
                dataset,
                out_root / "rendered",
                max_frames=args.render_max_frames,
                compute_metrics=not args.no_metrics,
            )

    print("\n所有导出完成。")


if __name__ == "__main__":
    main()