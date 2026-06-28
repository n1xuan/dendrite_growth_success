"""
data_loader.py - 独立数据加载模块
从 transforms.json 读取 X 射线投影数据集，提供时序采样功能。
完全独立，不依赖 nerfstudio。
修改：新增 max_images_in_memory 参数，控制预加载到内存的图像数量。
修复：preload_images=False 时从磁盘直接加载，避免 NoneType 赋值错误。
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from torch.utils.data import Dataset
from PIL import Image


def parse_transforms_json(
    json_path: Union[str, Path],
    downscale_factor: float = 1.0,
) -> Tuple[List[str], List[Dict], List[float]]:
    """
    解析 transforms.json 文件，提取图像路径、相机参数和时间戳。

    Args:
        json_path: transforms.json 文件的路径。
        downscale_factor: 图像下采样因子（例如 2 表示降采样到一半大小），
                          会相应地缩放 fx, fy, cx, cy。

    Returns:
        image_paths: 图像文件路径列表（相对路径或绝对路径）。
        camera_params: 相机参数字典列表，每个字典包含：
            {
                'c2w': np.ndarray (3x4 或 4x4),
                'fx': float,
                'fy': float,
                'cx': float,
                'cy': float,
                'height': int,
                'width': int,
            }
        times: 每个帧对应的时间戳列表。
    """
    json_path = Path(json_path)
    data_dir = json_path.parent
    with open(json_path, 'r') as f:
        meta = json.load(f)

    # 获取全局或每个 frame 的参数
    # 优先使用 frame 内的参数，否则使用全局 meta 中的值
    def get_frame_param(frame, key, default=None):
        if key in frame:
            return frame[key]
        return meta.get(key, default)

    image_paths = []
    camera_params_list = []
    times = []

    for frame in meta['frames']:
        # 图像路径
        fname = frame['file_path']
        # 如果路径是相对路径，则相对于 data_dir
        img_path = data_dir / fname
        image_paths.append(str(img_path))

        # 时间戳
        time = frame.get('time', 0.0)
        times.append(float(time))

        # 相机内参：优先 frame，否则 meta
        fx = get_frame_param(frame, 'fl_x')
        fy = get_frame_param(frame, 'fl_y')
        cx = get_frame_param(frame, 'cx')
        cy = get_frame_param(frame, 'cy')
        h = get_frame_param(frame, 'h')
        w = get_frame_param(frame, 'w')

        # 如果全局未定义，则必须每个 frame 都有
        if fx is None:
            raise ValueError(f"fx not defined for frame {fname}")
        if fy is None:
            fy = fx  # 有时仅提供 fl_x
        if cx is None:
            cx = w / 2.0
        if cy is None:
            cy = h / 2.0

        # 下采样缩放
        if downscale_factor != 1.0:
            fx /= downscale_factor
            fy /= downscale_factor
            cx /= downscale_factor
            cy /= downscale_factor
            h = int(h / downscale_factor)
            w = int(w / downscale_factor)

        # 相机到世界矩阵
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        # 如果提供的是 4x4，取前 3 行
        if c2w.shape == (4, 4):
            c2w = c2w[:3, :]

        camera_params = {
            'c2w': c2w,
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
            'height': int(h),
            'width': int(w),
        }
        camera_params_list.append(camera_params)

    return image_paths, camera_params_list, times


class XRayDataset(Dataset):
    """
    X 射线投影数据集。
    每个样本包含图像、相机参数和时间戳。
    支持预加载所有或部分图像到内存，以平衡内存和速度。
    """

    def __init__(
        self,
        json_path: Union[str, Path],
        downscale_factor: float = 1.0,
        load_images: bool = True,
        preload_images: bool = True,          # 是否预加载图像到内存
        max_images_in_memory: Optional[int] = None,  # 最多预加载多少张（None 表示全部）
    ):
        """
        Args:
            json_path: transforms.json 路径。
            downscale_factor: 图像下采样因子。
            load_images: 如果为 True，在 __getitem__ 中加载图像；否则仅返回路径。
            preload_images: 如果为 True，在初始化时加载图像到内存（仅在 load_images=True 时有效）。
            max_images_in_memory: 预加载图像的最大数量，超出部分在运行时从磁盘读取（仅在 preload_images=True 时有效）。
        """
        self.json_path = Path(json_path)
        self.data_dir = self.json_path.parent
        self.downscale_factor = downscale_factor
        self.load_images = load_images

        # 解析数据
        self.image_paths, self.camera_params, self.times = parse_transforms_json(
            json_path, downscale_factor
        )

        # 获取图像尺寸（假设所有图像尺寸一致）
        if len(self.camera_params) > 0:
            self.height = self.camera_params[0]['height']
            self.width = self.camera_params[0]['width']
        else:
            self.height = self.width = 0

        # 唯一时间戳列表（用于采样）
        self.unique_times = sorted(set(self.times))

        # ---- 预加载图像到内存 ----
        self.images = [None] * len(self.image_paths)  # 初始化占位列表
        if load_images and preload_images:
            # 确定实际要预加载的数量
            n_total = len(self.image_paths)
            n_preload = n_total if max_images_in_memory is None else min(max_images_in_memory, n_total)

            print(f"Preloading {n_preload} out of {n_total} images into memory...")
            for idx in range(n_preload):
                self._load_and_cache_image(idx)
            print(f"Loaded {n_preload} images into memory.")
        else:
            # 不预加载任何图像，所有图像在 __getitem__ 时从磁盘读取
            self.images = None

    def _load_and_cache_image(self, idx: int):
        """从磁盘加载单张图像并缓存到 self.images[idx]（返回张量，形状 [H,W,1]）"""
        path = self.image_paths[idx]
        img = Image.open(path)
        if self.downscale_factor != 1.0:
            new_size = (int(img.width / self.downscale_factor),
                        int(img.height / self.downscale_factor))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
        if img.mode != 'L':
            img = img.convert('L')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(-1)  # [H, W, 1]
        self.images[idx] = img_tensor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        返回:
            image: torch.Tensor (H, W, 1) 归一化到 [0,1]
            c2w: torch.Tensor (3, 4)
            fx, fy, cx, cy: float
            height, width: int
            time: float
            image_path: str (可选)
            idx: int (索引)
        """
        # ---- 获取图像 ----
        if self.load_images and self.images is not None and self.images[idx] is not None:
            # 从缓存读取
            img_tensor = self.images[idx]
        elif self.load_images:
            # 未预加载，直接从磁盘读取（不缓存）
            path = self.image_paths[idx]
            img = Image.open(path)
            if self.downscale_factor != 1.0:
                new_size = (int(img.width / self.downscale_factor),
                            int(img.height / self.downscale_factor))
                img = img.resize(new_size, Image.Resampling.BILINEAR)
            if img.mode != 'L':
                img = img.convert('L')
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.unsqueeze(-1)  # [H, W, 1]
        else:
            img_tensor = torch.zeros((self.height, self.width, 1))  # 占位

        # ---- 相机参数 ----
        params = self.camera_params[idx]
        c2w = torch.from_numpy(params['c2w']).float()
        fx = params['fx']
        fy = params['fy']
        cx = params['cx']
        cy = params['cy']
        h = params['height']
        w = params['width']
        time = self.times[idx]

        return {
            'image': img_tensor,
            'c2w': c2w,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'height': h,
            'width': w,
            'time': time,
            'image_path': self.image_paths[idx],
            'idx': idx,
        }


class TimestampSampler:
    """
    时序采样器：根据训练步数渐进式采样帧。
    用于从所有帧中选择子集，逐步纳入更早的时间步。
    """

    def __init__(
        self,
        unique_times: List[float],
        time_proposal_steps: int = 300,
        max_images_per_timestamp: int = 3,
        max_unique_timestamps: int = 5,
    ):
        """
        Args:
            unique_times: 所有唯一时间戳列表（已排序）。
            time_proposal_steps: 经过多少步训练后，才允许采样最早的时间。
            max_images_per_timestamp: 每个时间戳最多采样多少张图像。
            max_unique_timestamps: 每批次最多包含多少个不同的时间戳。
        """
        self.unique_times = np.array(sorted(unique_times))
        self.min_time = self.unique_times[0]
        self.max_time = self.unique_times[-1]
        self.time_proposal_steps = time_proposal_steps
        self.max_images_per_timestamp = max_images_per_timestamp
        self.max_unique_timestamps = max_unique_timestamps

    def choose_indices(
        self,
        dataset_times: List[float],
        step: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """
        根据当前训练步数选择图像索引。

        Args:
            dataset_times: 数据集中所有图像对应的时间列表（长度与数据集相同）。
            step: 当前训练步数。
            rng: 随机数生成器（可选）。

        Returns:
            选中的图像索引列表。
        """
        if rng is None:
            rng = np.random.default_rng()

        # 计算当前时间截止值
        if self.time_proposal_steps > 0:
            progress = min(step / self.time_proposal_steps, 1.0)
            cutoff_time = self.min_time + progress * (self.max_time - self.min_time)
        else:
            cutoff_time = self.max_time

        # 构建所有索引及其时间
        indices = np.arange(len(dataset_times))
        times_arr = np.array(dataset_times)

        # 只保留 cutoff_time 之前的帧
        valid_mask = times_arr <= cutoff_time
        valid_indices = indices[valid_mask]
        valid_times = times_arr[valid_mask]

        # 按时间分组
        time_groups = {}
        for idx, t in zip(valid_indices, valid_times):
            time_groups.setdefault(t, []).append(idx)

        # 打乱时间顺序，并限制不同时间戳的数量
        time_keys = list(time_groups.keys())
        rng.shuffle(time_keys)
        selected_time_keys = time_keys[:self.max_unique_timestamps]

        # 从每个选中的时间戳中随机选 max_images_per_timestamp 个
        selected_indices = []
        for t in selected_time_keys:
            group = time_groups[t]
            # 如果组内数量大于限制，随机采样
            if len(group) > self.max_images_per_timestamp:
                chosen = rng.choice(group, self.max_images_per_timestamp, replace=False)
            else:
                chosen = group
            selected_indices.extend(chosen)

        return selected_indices