"""
train.py - 4D 密度场训练脚本（稀疏演化 + 密集锚点）
模型：TemporalDensityField
损失：背景/前景分离损失 + 时间单调性 + 空间TV
已移除：时间平滑损失、质量守恒损失、Sobel 边缘损失
新增：空白区域（背景）与晶体区域（前景）使用独立损失约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

from models import TemporalDensityField
from renderer import AttenuationRenderer
from data_loader import XRayDataset
from ray_utils import create_ray_bundle_from_camera, sample_points_uniform, RayBundle
from losses import (
    temporal_monotonicity_loss,
    spatial_tv_loss,
    background_mse_loss,      # 新增
    foreground_mse_loss,      # 新增
)

from PIL import Image


# ----------------------------------------------------------------------
# PSNR / SSIM 计算函数
# ----------------------------------------------------------------------
def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 20.0 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    from scipy.ndimage import uniform_filter
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = uniform_filter(pred, size=3, mode='reflect')
    mu2 = uniform_filter(gt, size=3, mode='reflect')
    sigma1_sq = uniform_filter(pred ** 2, 3, mode='reflect') - mu1 ** 2
    sigma2_sq = uniform_filter(gt ** 2, 3, mode='reflect') - mu2 ** 2
    sigma12 = uniform_filter(pred * gt, 3, mode='reflect') - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return float(ssim_map.mean())


# ----------------------------------------------------------------------
# 训练配置
# ----------------------------------------------------------------------
class TrainConfig:
    def __init__(self, **kwargs):
        self.data_json = kwargs.get('data_json', 'data/transforms_00_to_29.json')
        self.downscale_factor = kwargs.get('downscale_factor', 1.0)
        self.aabb_min = kwargs.get('aabb_min', (-1.0, -1.0, -1.0))
        self.aabb_max = kwargs.get('aabb_max', (1.0, 1.0, 1.0))
        # 4D 密度场
        self.hash_base_res = kwargs.get('hash_base_res', 16)
        self.hash_max_res = kwargs.get('hash_max_res', 2048)
        self.hash_n_levels = kwargs.get('hash_n_levels', 16)
        self.hash_n_features = kwargs.get('hash_n_features', 2)
        self.hash_log2_size = kwargs.get('hash_log2_size', 19)
        self.average_init_density = kwargs.get('average_init_density', 1.0)
        self.use_4d_hash = kwargs.get('use_4d_hash', True)
        # 训练
        self.max_iterations = kwargs.get('max_iterations', 3000)
        self.batch_size = kwargs.get('batch_size', 512)
        self.dense_batch_size = kwargs.get('dense_batch_size', 1024)
        self.num_samples_per_ray = kwargs.get('num_samples_per_ray', 128)
        self.near_plane = kwargs.get('near_plane', 1.5)
        self.far_plane = kwargs.get('far_plane', 4.5)
        self.lr = kwargs.get('lr', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 1e-8)
        self.grad_clip = kwargs.get('grad_clip', 1.0)
        self.mixed_precision = kwargs.get('mixed_precision', True)
        # 稀疏/密集
        self.sparse_steps_per_dense = kwargs.get('sparse_steps_per_dense', 4)
        # 损失权重
        self.rgb_loss_weight = kwargs.get('rgb_loss_weight', 1.0)          # 稀疏帧缩放因子
        self.dense_rgb_loss_weight = kwargs.get('dense_rgb_loss_weight', 1.0)  # 密集帧缩放因子（已降低以避免饱和帧干扰）
        self.background_loss_weight = kwargs.get('background_loss_weight', 1.0)
        self.foreground_loss_weight = kwargs.get('foreground_loss_weight', 1.0)
        self.silhouette_threshold = kwargs.get('silhouette_threshold', 0.95)  # 背景阈值
        # 单调性
        self.monotonicity_weight = kwargs.get('monotonicity_weight', 0.1)
        self.monotonicity_start_step = kwargs.get('monotonicity_start_step', 0)  # 从开始就启用
        # 空间 TV 损失
        self.tv_weight = kwargs.get('tv_weight', 1e-5)
        self.tv_start_step = kwargs.get('tv_start_step', 500)
        # 日志与验证
        self.log_dir = kwargs.get('log_dir', 'logs')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints')
        self.save_every = kwargs.get('save_every', 500)
        self.print_every = kwargs.get('print_every', 100)
        self.val_every = kwargs.get('val_every', 200)
        self.val_num_images = kwargs.get('val_num_images', 4)
        self.render_chunk_size = kwargs.get('render_chunk_size', 4096)
        self.resume_checkpoint = kwargs.get('resume_checkpoint', None)
        self.render_output_dir = kwargs.get('render_output_dir', 'renders')

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ----------------------------------------------------------------------
# 训练器
# ----------------------------------------------------------------------
class Trainer:
    def __init__(self, config: TrainConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resume_path = config.resume_checkpoint

        if resume_path:
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            if 'config' in ckpt:
                config = TrainConfig.from_dict(ckpt['config'])

        self.config = config
        self.step = 0
        self.best_psnr = -float('inf')

        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.render_dir = Path(config.render_output_dir)
        self.render_dir.mkdir(parents=True, exist_ok=True)

        self._setup_data()
        self._setup_model()
        self.renderer = AttenuationRenderer(background_color='white')
        self._setup_optimizer()
        self._setup_scheduler()

        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(device_type, enabled=config.mixed_precision)

        if resume_path:
            self._load_checkpoint(resume_path)

        print(f"设备: {self.device}")
        print(f"4D 密度场参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"稀疏帧: {len(self.sparse_indices)}, 密集帧: {len(self.dense_indices)}")
        print(f"背景/前景损失阈值: {self.config.silhouette_threshold}")
        print(f"渲染输出目录: {self.render_dir}")

    def _setup_data(self):
        self.dataset = XRayDataset(
            self.config.data_json,
            downscale_factor=self.config.downscale_factor,
            load_images=True,
            preload_images=True,
        )
        self.sparse_indices = [i for i, t in enumerate(self.dataset.times) if t < 0.99]
        self.dense_indices = [i for i, t in enumerate(self.dataset.times) if t >= 0.99]
        if not self.dense_indices:
            n_dense = max(1, int(len(self.dataset) * 0.2))
            self.dense_indices = list(range(len(self.dataset) - n_dense, len(self.dataset)))
            self.sparse_indices = list(range(len(self.dataset) - n_dense))

    def _setup_model(self):
        aabb = torch.tensor([self.config.aabb_min, self.config.aabb_max], dtype=torch.float32)
        self.model = TemporalDensityField(
            aabb=aabb,
            time_range=(0.0, 1.0),
            hash_config={
                'base_res': self.config.hash_base_res,
                'max_res': self.config.hash_max_res,
                'n_levels': self.config.hash_n_levels,
                'n_features_per_level': self.config.hash_n_features,
                'log2_hashmap_size': self.config.hash_log2_size,
            },
            average_init_density=self.config.average_init_density,
            use_4d_hash=self.config.use_4d_hash,
        ).to(self.device)

    def _setup_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _setup_scheduler(self):
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_iterations,
            eta_min=1e-5,
        )

    def _load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        if 'scaler_state_dict' in state:
            self.scaler.load_state_dict(state['scaler_state_dict'])
        self.step = state.get('step', 0)
        if 'best_psnr' in state:
            self.best_psnr = state['best_psnr']
        print(f"从检查点恢复训练: step {self.step}")

    def save_checkpoint(self, path):
        state = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config.to_dict(),
            'best_psnr': self.best_psnr,
        }
        torch.save(state, path)
        print(f"检查点已保存: {path}")

    def _sample_rays(self, use_dense: bool):
        idx = np.random.choice(self.dense_indices if use_dense else self.sparse_indices)
        sample = self.dataset[idx]
        img = sample['image'].to(self.device)          # [H, W, 1]
        H, W = img.shape[:2]
        ray_bundle = create_ray_bundle_from_camera(
            sample['c2w'].to(self.device),
            sample['fx'], sample['fy'], sample['cx'], sample['cy'],
            H, W,
            time=sample['time'],
            near=self.config.near_plane,
            far=self.config.far_plane,
            device=self.device,
        )
        batch_size = self.config.dense_batch_size if use_dense else self.config.batch_size
        pix_idx = torch.randint(0, H * W, (batch_size,), device=self.device)
        origins = ray_bundle.origins[pix_idx]
        directions = ray_bundle.directions[pix_idx]
        times = ray_bundle.times[pix_idx]
        gt = img.reshape(-1, 1)[pix_idx]
        return origins, directions, times, gt

    def _render_rays(self, origins, directions, times, perturb=True):
        N = origins.shape[0]
        positions, deltas = sample_points_uniform(
            origins, directions,
            near=self.config.near_plane,
            far=self.config.far_plane,
            num_samples=self.config.num_samples_per_ray,
            perturb=perturb,
        )
        density = self.model(
            positions.reshape(-1, 3),
            times.expand(-1, self.config.num_samples_per_ray).reshape(-1, 1),
        )
        density = density.reshape(N, self.config.num_samples_per_ray, 1)
        attenuation = self.renderer(density, deltas)
        return attenuation

    def _train_step(self, use_dense: bool):
        origins, directions, times, gt = self._sample_rays(use_dense)
        self.optimizer.zero_grad()

        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with autocast(device_type, enabled=self.config.mixed_precision):
            attenuation = self._render_rays(origins, directions, times)

            # 背景/前景分离损失
            loss_bg = background_mse_loss(attenuation, gt, threshold=self.config.silhouette_threshold)
            loss_fg = foreground_mse_loss(attenuation, gt, threshold=self.config.silhouette_threshold)
            loss_rgb = (self.config.background_loss_weight * loss_bg +
                        self.config.foreground_loss_weight * loss_fg)

            # 稀疏/密集帧的整体缩放因子（可在命令行调整，默认为 1.0）
            scale = self.config.dense_rgb_loss_weight if use_dense else self.config.rgb_loss_weight
            loss_total = scale * loss_rgb

            # 单调性损失
            loss_mono = torch.tensor(0.0, device=self.device)
            if self.step >= self.config.monotonicity_start_step and self.config.monotonicity_weight > 0:
                loss_mono = temporal_monotonicity_loss(self.model, device=self.device)
                loss_total += self.config.monotonicity_weight * loss_mono

            # 空间 TV 损失
            loss_tv = torch.tensor(0.0, device=self.device)
            if self.step >= self.config.tv_start_step and self.config.tv_weight > 0:
                loss_tv = spatial_tv_loss(self.model, device=self.device)
                loss_total += self.config.tv_weight * loss_tv

        self.scaler.scale(loss_total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if not use_dense:
            self.scheduler.step()

        return {
            'rgb': loss_rgb.item(),
            'bg': loss_bg.item(),
            'fg': loss_fg.item(),
            'mono': loss_mono.item(),
            'tv': loss_tv.item(),
            'total': loss_total.item(),
        }

    @torch.no_grad()
    def _validate_psnr(self, num_images=None):
        self.model.eval()
        if num_images is None:
            num_images = self.config.val_num_images

        indices = np.random.choice(len(self.dataset), min(num_images, len(self.dataset)), replace=False)
        total_psnr = 0.0
        total_ssim = 0.0

        for idx in indices:
            sample = self.dataset[idx]
            img_gt = sample['image']
            H, W = img_gt.shape[:2]

            ray_bundle = create_ray_bundle_from_camera(
                sample['c2w'].to(self.device),
                sample['fx'], sample['fy'], sample['cx'], sample['cy'],
                H, W,
                time=sample['time'],
                near=self.config.near_plane,
                far=self.config.far_plane,
                device=self.device,
            )
            ray_bundle.flatten()

            pred_chunks = []
            N = ray_bundle.origins.shape[0]
            chunk_size = self.config.render_chunk_size
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)
                sub = RayBundle(
                    origins=ray_bundle.origins[i:end],
                    directions=ray_bundle.directions[i:end],
                    times=ray_bundle.times[i:end],
                )
                pred = self._render_rays(sub.origins, sub.directions, sub.times, perturb=False)
                pred_chunks.append(pred.cpu())
            pred_img = torch.cat(pred_chunks).reshape(H, W).numpy()
            gt_img = img_gt.reshape(H, W).numpy()

            # 保存渲染图像
            self._save_projection_image(pred_img, gt_img, idx, sample['time'])

            psnr = compute_psnr(gt_img, pred_img)
            ssim = compute_ssim(gt_img, pred_img)
            total_psnr += psnr
            total_ssim += ssim

        self.model.train()
        avg_psnr = total_psnr / len(indices)
        avg_ssim = total_ssim / len(indices)
        return avg_psnr, avg_ssim

    def _save_projection_image(self, pred_img, gt_img, idx, time):
        """将预测投影和真实投影并排保存为 PNG"""
        pred_uint8 = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
        gt_uint8   = (np.clip(gt_img, 0, 1) * 255).astype(np.uint8)

        # 左真值，右预测，中间加分隔线
        combined = np.hstack([gt_uint8, pred_uint8])
        combined[:, gt_uint8.shape[1]:gt_uint8.shape[1]+2] = 128

        img = Image.fromarray(combined, mode='L')
        fname = f"step{self.step:06d}_idx{idx:03d}_t{time:.3f}.png"
        img.save(self.render_dir / fname)

    def train(self):
        sparse_counter = 0
        while self.step < self.config.max_iterations:
            if sparse_counter < self.config.sparse_steps_per_dense:
                losses = self._train_step(use_dense=False)
                sparse_counter += 1
                step_type = 'sparse'
            else:
                losses = self._train_step(use_dense=True)
                sparse_counter = 0
                step_type = 'dense'

            if self.step % self.config.print_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Step {self.step:06d} [{step_type}] "
                    f"Loss: {losses['total']:.6f} | RGB: {losses['rgb']:.6f} "
                    f"BG: {losses['bg']:.6f} FG: {losses['fg']:.6f} | "
                    f"Mono: {losses['mono']:.6f} TV: {losses['tv']:.6f} | "
                    f"LR: {lr:.2e}"
                )
                for k, v in losses.items():
                    self.writer.add_scalar(f'Loss/{k}', v, self.step)
                self.writer.add_scalar('LR', lr, self.step)

                if self.step % self.config.val_every == 0:
                    val_psnr, val_ssim = self._validate_psnr()
                    self.writer.add_scalar('Metrics/PSNR', val_psnr, self.step)
                    self.writer.add_scalar('Metrics/SSIM', val_ssim, self.step)
                    print(f"  [VAL] PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} (渲染图已保存至 {self.render_dir})")

                    if val_psnr > self.best_psnr:
                        self.best_psnr = val_psnr
                        self.save_checkpoint(self.ckpt_dir / "best_psnr.ckpt")
                        print(f"  >> New best PSNR: {val_psnr:.2f} dB, checkpoint saved.")

            if self.step > 0 and self.step % self.config.save_every == 0:
                self.save_checkpoint(self.ckpt_dir / f"step_{self.step:06d}.ckpt")

            self.step += 1

        self.save_checkpoint(self.ckpt_dir / "final.ckpt")
        self.writer.close()
        print("训练完成。")


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="4D 密度场训练（稀疏演化 + 密集锚点）")
    parser.add_argument('--config', type=str, help='JSON 配置文件')
    parser.add_argument('--data_json', type=str, default='data/transforms_00_to_29.json')
    parser.add_argument('--max_iter', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dense_batch_size', type=int, default=1024)
    parser.add_argument('--sparse_steps_per_dense', type=int, default=4)
    parser.add_argument('--num_samples_per_ray', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--monotonicity_weight', type=float, default=0.1)
    parser.add_argument('--tv_weight', type=float, default=1e-5)
    parser.add_argument('--background_loss_weight', type=float, default=1.0,
                        help='背景（空白）区域损失权重')
    parser.add_argument('--foreground_loss_weight', type=float, default=1.0,
                        help='前景（有晶体）区域损失权重')
    parser.add_argument('--silhouette_threshold', type=float, default=0.95,
                        help='判断空白/晶体的衰减阈值')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--no_4d_hash', action='store_true')
    parser.add_argument('--val_every', type=int, default=200)
    parser.add_argument('--val_num_images', type=int, default=4)
    parser.add_argument('--render_chunk_size', type=int, default=4096)
    parser.add_argument('--render_output_dir', type=str, default='renders',
                        help='保存验证渲染图像的目录')
    args = parser.parse_args()

    config = TrainConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config = TrainConfig.from_dict(json.load(f))

    # 命令行覆盖
    config.data_json = args.data_json
    config.max_iterations = args.max_iter
    config.batch_size = args.batch_size
    config.dense_batch_size = args.dense_batch_size
    config.sparse_steps_per_dense = args.sparse_steps_per_dense
    config.num_samples_per_ray = args.num_samples_per_ray
    config.lr = args.lr
    config.monotonicity_weight = args.monotonicity_weight
    config.tv_weight = args.tv_weight
    config.background_loss_weight = args.background_loss_weight
    config.foreground_loss_weight = args.foreground_loss_weight
    config.silhouette_threshold = args.silhouette_threshold
    config.resume_checkpoint = args.resume
    config.use_4d_hash = not args.no_4d_hash
    config.log_dir = str(Path(args.output_dir) / 'logs')
    config.checkpoint_dir = str(Path(args.output_dir) / 'checkpoints')
    config.val_every = args.val_every
    config.val_num_images = args.val_num_images
    config.render_chunk_size = args.render_chunk_size
    config.render_output_dir = args.render_output_dir

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()