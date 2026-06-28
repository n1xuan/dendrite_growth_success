"""
viser_viewer.py - 4D 密度场交互可视化
从检查点加载 TemporalDensityField，通过 viser 提供实时交互渲染。
"""

import torch
import viser
import numpy as np
from typing import Optional
import time
from pathlib import Path
from export import load_checkpoint, Exporter
from ray_utils import create_ray_bundle_from_camera


class DensityViewer:
    def __init__(
        self,
        checkpoint_path: str,
        render_h: int = 500,
        render_w: int = 500,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"设备: {self.device}")
        print(f"加载检查点: {checkpoint_path}")

        # 加载模型与配置
        self.model, self.config = load_checkpoint(checkpoint_path, self.device)
        self.near = self.config.get('near_plane', 1.5)
        self.far = self.config.get('far_plane', 4.5)
        self.render_h, self.render_w = render_h, render_w

        # 创建导出器
        self.exporter = Exporter(
            self.model, self.device,
            near=self.near, far=self.far,
        )

        self.current_time = 0.0
        self.last_cam_params = None
        self.last_rendered_img = None

        # 启动 viser
        self.server = viser.ViserServer()
        self._setup_gui()

    def _setup_gui(self):
        self.server.gui.add_slider(
            "Time", 0.0, 1.0, 0.01, 0.0
        ).on_update(lambda h: setattr(self, "current_time", h.value))

        self.server.gui.add_button("Render").on_click(lambda _: self.render_current_view())

        self.image_handle = self.server.gui.add_image(
            "X-Ray",
            np.zeros((self.render_h, self.render_w, 3), dtype=np.uint8),
        )

    def get_camera(self):
        clients = self.server.get_clients()
        if not clients:
            return None
        client = list(clients.values())[0]
        if hasattr(client, 'camera'):
            return client.camera
        return getattr(client, 'camera_handle', None)

    def _camera_changed(self, cam) -> bool:
        if cam is None:
            return False
        pos = np.array(cam.position)
        look = np.array(cam.look_at)
        state = np.concatenate([pos, look])
        if self.last_cam_params is None:
            self.last_cam_params = state
            return True
        if not np.allclose(state, self.last_cam_params, atol=1e-4):
            self.last_cam_params = state
            return True
        return False

    def render_current_view(self):
        cam = self.get_camera()
        if cam is None:
            return

        pos = np.array(cam.position)
        look = np.array(cam.look_at)
        forward = look - pos
        forward /= np.linalg.norm(forward) + 1e-8

        up_ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, up_ref)) > 0.99:
            up_ref = np.array([0.0, 0.0, 1.0])

        right = np.cross(forward, up_ref)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-8

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = pos

        h, w = self.render_h, self.render_w
        fx, fy = w / 2.0, w / 2.0
        cx, cy = w / 2.0, h / 2.0

        ray_bundle = create_ray_bundle_from_camera(
            torch.from_numpy(c2w).to(self.device),
            fx, fy, cx, cy, h, w,
            time=self.current_time,
            near=self.near, far=self.far,
            device=self.device,
        )
        ray_bundle.flatten()

        with torch.no_grad():
            att = self.exporter.render_image_from_ray_bundle(ray_bundle)
        img = att.reshape(h, w).cpu().numpy()
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img_rgb = np.stack([img_uint8] * 3, axis=-1)

        if self.last_rendered_img is None or not np.array_equal(self.last_rendered_img, img_rgb):
            self.image_handle.image = img_rgb
            self.last_rendered_img = img_rgb

    def run(self):
        print("Viser 已启动，访问 http://localhost:8080")
        print("拖动视角或调整时间滑块，图像实时更新")
        try:
            while True:
                cam = self.get_camera()
                if cam is not None and self._camera_changed(cam):
                    self.render_current_view()
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("关闭 viewer")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="4D X-Ray 密度场交互可视化")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--render_h", type=int, default=500)
    parser.add_argument("--render_w", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    viewer = DensityViewer(
        args.checkpoint,
        args.render_h,
        args.render_w,
        torch.device(args.device if torch.cuda.is_available() else "cpu"),
    )
    viewer.run()