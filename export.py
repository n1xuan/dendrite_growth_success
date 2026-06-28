"""
export.py - 导出 4D 密度场并渲染预测图像
从检查点加载 TemporalDensityField，无变形场，直接查询密度。
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


def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 20.0 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0,
                 K1: float = 0.01, K2: float = 0.03) -> float:
    from scipy.ndimage import convolve
    gt, pred = gt.astype(np.float64), pred.astype(np.float64)
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    window = np.ones((3, 3), dtype=np.float64) / 9.0
    mu1 = convolve(gt, window, mode='reflect')
    mu2 = convolve(pred, window, mode='reflect')
    sigma1_sq = convolve(gt ** 2, window, mode='reflect') - mu1 ** 2
    sigma2_sq = convolve(pred ** 2, window, mode='reflect') - mu2 ** 2
    sigma12 = convolve(gt * pred, window, mode='reflect') - mu1 * mu2
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    return float(ssim_map.mean())


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
        if output_path.suffix == '.npz':
            np.savez_compressed(output_path, volume=density_grid)
        elif output_path.suffix == '.npy':
            np.save(output_path, density_grid)
        else:
            np.savez_compressed(output_path.with_suffix('.npz'), volume=density_grid)
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
            print(f"\n========== 重建质量 ==========")
            print(f"PSNR 均值: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
            print(f"SSIM 均值: {avg_ssim:.4f} ± {std_ssim:.4f}")
            print(f"===============================\n")


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


def main():
    parser = argparse.ArgumentParser(description="导出 4D 密度场并渲染图像")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='exports')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--times', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    parser.add_argument('--export_volume', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render_data', default=None)
    parser.add_argument('--render_max_frames', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--downscale', type=float, default=1.0)
    parser.add_argument('--render_chunk_size', type=int, default=4096)
    parser.add_argument('--no_metrics', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, config = load_checkpoint(args.checkpoint, device)
    near, far = config.get('near_plane', 1.5), config.get('far_plane', 4.5)

    exporter = Exporter(model, device, near=near, far=far, render_chunk_size=args.render_chunk_size)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.export_volume:
        for t in args.times:
            exporter.export_volume_grid(args.resolution, t, out_root / f"volume_t{t:.3f}.npz")

    if args.render:
        data_json = args.render_data or next(Path(args.checkpoint).parent.glob("transforms_*.json"))
        dataset = XRayDataset(data_json, downscale_factor=args.downscale, load_images=True, preload_images=False)
        exporter.render_dataset(dataset, out_root / "rendered", max_frames=args.render_max_frames,
                                compute_metrics=not args.no_metrics)

    print("所有导出完成。")


if __name__ == "__main__":
    main()