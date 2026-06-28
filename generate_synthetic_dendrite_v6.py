#!/usr/bin/env python3
"""
Synthetic dendrite solidification — V5 (parallel beam).
30 frames, tree-shaped tips, partial initial structure + new nucleation.

Key change from v4: replaces Go cone-beam renderer with self-contained
parallel beam (ORTHOPHOTO) projector via scipy.ndimage.rotate + sum.
Output format matches prepare_dendrite_data_growth.py exactly, so it
plugs directly into dendrite_growth_xray training pipeline.

MODIFIED: output directory can be specified via --output_dir argument.
"""

import sys, json, yaml, numpy as np, shutil
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data): return True

# 【YAML 报错补丁】：恢复 v4 中的 Numpy 浮点数识别补丁
_NoAliasDumper.add_representer(np.float64, lambda dumper, data: dumper.represent_float(float(data)))
_NoAliasDumper.add_representer(np.float32, lambda dumper, data: dumper.represent_float(float(data)))

def _yaml_dump(data, stream):
    yaml.dump(data, stream, Dumper=_NoAliasDumper, default_flow_style=False)

@dataclass
class DendriteArm:
    origin: np.ndarray
    direction: np.ndarray
    radius: float
    max_length: float
    birth_time: float
    accel_factor: float
    level: int
    coarsen_rate: float
    initial_frac: float  # fraction already grown at birth_time

    def current_length(self, t):
        if t < self.birth_time: return 0.0
        dt = t - self.birth_time
        avail = 1.0 - self.birth_time + 1e-8
        tau = dt / avail
        grown = tau * (1.0 + self.accel_factor * tau * 0.5)
        total = self.initial_frac + (1.0 - self.initial_frac) * min(grown, 1.0)
        return self.max_length * min(total, 1.0)

    def current_radius(self, t, axial_frac=0.0):
        dt = max(0.0, t - self.birth_time)
        local_dt = dt * (1.0 - axial_frac * 0.5)
        return self.radius * (1.0 + self.coarsen_rate * local_dt)

    def tip_position(self, t):
        return self.origin + self.direction * self.current_length(t)

    def to_objects(self, t, rho=3.0):
        L = self.current_length(t)
        r_base = self.current_radius(t, 0.0)
        r_tip = self.current_radius(t, 1.0)
        if L < r_base * 1.8:
            if L > 0 and self.level == 0:
                return [{"type": "sphere", "center": self.origin.tolist(),
                         "radius": max(L*0.4, self.radius*0.3), "rho": rho}]
            return []
        objs = []
        if self.level == 0: n_seg = min(15, max(4, int(L/0.06)))
        elif self.level == 1: n_seg = min(6, max(2, int(L/0.05)))
        else: n_seg = 1
        for i in range(n_seg):
            f0, f1 = i/n_seg, (i+1)/n_seg
            p0 = (self.origin + self.direction * L * f0).tolist()
            p1 = (self.origin + self.direction * L * f1).tolist()
            r = r_base + (r_tip - r_base) * (f0+f1)/2
            objs.append({"type":"cylinder","p0":p0,"p1":p1,"radius":r,"rho":rho})
        tip = self.tip_position(t).tolist()
        objs.append({"type":"sphere","center":tip,"radius":r_tip,"rho":rho})
        return objs


def build_dendrite_arm_table(
    num_dendrites=12,
    primary_radius=0.055,
    secondary_radius=0.033,
    tertiary_radius=0.020,
    secondary_spacing=0.10,
    tertiary_spacing=0.10,
    tip_accel=0.30,
    coarsen_rate=0.65,
    enable_tertiary=True,
    seed=42,
):
    rng = np.random.RandomState(seed)
    arms = []

    # ── Nucleation plan ──
    # Wave 1 (t<0): 5 dendrites already have primary arms at t=0
    # Wave 2 (t=0.06): 4 new dendrites nucleate
    # Wave 3 (t=0.14): 3 new dendrites nucleate
    waves = [
        (-0.15, 5),  # negative birth = already grown at t=0
        (0.06, 4),
        (0.14, 3),
    ]

    positions = []
    for wt, nn in waves:
        for _ in range(nn * 30):
            x, y = rng.uniform(-0.55, 0.55), rng.uniform(-0.55, 0.55)
            if all((x-px)**2+(y-py)**2 > 0.15**2 for px,py,_ in positions):
                positions.append((x, y, wt))
            if sum(1 for _,_,w in positions if w==wt) >= nn:
                break

    print(f"  Nucleation: {len(positions)} sites")

    for px, py, nuc_time in positions:
        crystal_ang = rng.uniform(0, np.pi/4)
        speed_var = 1.0 + rng.uniform(-0.08, 0.08)
        z0 = -0.82 + rng.uniform(-0.02, 0.02)
        tilt = np.array([rng.uniform(-0.03,0.03), rng.uniform(-0.03,0.03), 1.0])
        tilt /= np.linalg.norm(tilt)
        p_max_len = 1.55 + rng.uniform(-0.06, 0.06)

        # Primary arm
        if nuc_time < 0:
            p_birth = 0.0
            # Already grown: estimate how much at t=0
            # If nuc_time=-0.15, the dendrite has been growing for 0.15
            p_init_frac = min(0.35, abs(nuc_time) * 2.0)
        else:
            p_birth = nuc_time
            p_init_frac = 0.0

        arms.append(DendriteArm(
            origin=np.array([px, py, z0]),
            direction=tilt.copy(),
            radius=primary_radius + rng.uniform(-0.003, 0.003),
            max_length=p_max_len,
            birth_time=p_birth,
            accel_factor=tip_accel * speed_var,
            level=0,
            coarsen_rate=coarsen_rate * 0.5,
            initial_frac=p_init_frac,
        ))

        # 4 fixed crystallographic secondary directions
        sec_dirs = [(np.cos(crystal_ang + k*np.pi/2),
                     np.sin(crystal_ang + k*np.pi/2)) for k in range(4)]

        z_first = z0 + secondary_spacing * 0.4
        z_last = z0 + p_max_len - secondary_spacing * 0.2
        z_positions = np.arange(z_first, z_last, secondary_spacing)

        for iz, zp in enumerate(z_positions):
            frac_along = (zp - z0) / p_max_len

            # ── TREE-SHAPED ENVELOPE ──
            # Secondary arm length drops STEEPLY toward tip
            # This creates the conical/fir-tree silhouette
            envelope = (1.0 - frac_along)**1.5  # steeper than linear
            s_max_base = 0.10 + 0.32 * envelope + rng.uniform(-0.02, 0.02)
            s_max_base = max(0.04, s_max_base)

            # Secondary birth: must wait for primary to reach this z
            if nuc_time < 0:
                # Primary already existed before t=0
                primary_frac_at_t0 = p_init_frac
                z_reached_at_t0 = z0 + primary_frac_at_t0 * p_max_len
                if zp <= z_reached_at_t0:
                    # Already passed at t=0, secondary also pre-exists
                    s_birth = 0.0
                    s_init = min(0.6, 0.3 + rng.uniform(0, 0.2))
                else:
                    # Primary hasn't reached here yet at t=0
                    frac_remaining = (frac_along - p_init_frac) / (1.0 - p_init_frac + 1e-8)
                    s_birth = max(0.01, frac_remaining * 0.85)
                    s_init = 0.0
            else:
                s_birth = nuc_time + frac_along * (1.0 - nuc_time) * 0.88
                s_birth = min(s_birth, 0.92)
                s_init = 0.0

            for dx, dy in sec_dirs:
                s_dir = np.array([dx, dy, rng.uniform(-0.02, 0.04)])
                s_dir /= np.linalg.norm(s_dir)
                s_origin = np.array([px, py, z0]) + tilt * (zp - z0)

                arms.append(DendriteArm(
                    origin=s_origin.copy(),
                    direction=s_dir.copy(),
                    radius=secondary_radius + rng.uniform(-0.002, 0.002),
                    max_length=s_max_base,
                    birth_time=s_birth,
                    accel_factor=tip_accel * 0.35,
                    level=1,
                    coarsen_rate=coarsen_rate * (1.2 if s_birth == 0 else 0.6),
                    initial_frac=s_init,
                ))

                if not enable_tertiary or s_max_base < 0.09:
                    continue
                perp = np.cross(s_dir, [0,0,1])
                if np.linalg.norm(perp) < 0.1: continue
                perp /= np.linalg.norm(perp)
                for td in np.arange(tertiary_spacing, s_max_base*0.65, tertiary_spacing):
                    for sign in [1, -1]:
                        t_dir = sign*perp + np.array([0,0,rng.uniform(0.04,0.12)])
                        t_dir /= np.linalg.norm(t_dir)
                        t_origin = s_origin + s_dir * td
                        frac_on_sec = td / s_max_base
                        t_birth_est = s_birth + frac_on_sec * (1.0-s_birth) * 0.8 + 0.03
                        if t_birth_est > 0.92: continue
                        arms.append(DendriteArm(
                            origin=t_origin.copy(), direction=t_dir.copy(),
                            radius=tertiary_radius,
                            max_length=0.045+rng.uniform(0,0.03),
                            birth_time=t_birth_est,
                            accel_factor=tip_accel*0.15, level=2,
                            coarsen_rate=coarsen_rate*0.5, initial_frac=0.0))

    n_lv = [sum(1 for a in arms if a.level==lv) for lv in range(3)]
    n_init = sum(1 for a in arms if a.birth_time==0 and a.initial_frac>0)
    print(f"  Forest: {len(arms)} arms ({n_lv[0]}P {n_lv[1]}S {n_lv[2]}T)")
    print(f"  Pre-existing at t=0: {n_init} arms with initial_frac>0")
    return arms


def create_dendrite_collection(arms, t):
    objs = []
    for arm in arms: objs.extend(arm.to_objects(t))
    if not objs:
        objs.append({"type":"sphere","center":[0,0,-0.8],"radius":0.03,"rho":3.0})
    return {"type":"object_collection","objects":objs}


def rasterize_arms_fast(arms, t, resolution):
    N = resolution
    vol = np.zeros((N,N,N), dtype=np.float32)
    half = N / 2.0
    def w2i(w): return (w+1.0)*half
    for arm in arms:
        L = arm.current_length(t)
        rb = arm.current_radius(t, 0.0)
        rt = arm.current_radius(t, 1.0)
        rm = max(rb, rt)
        if L < rb*1.8:
            if L > 0 and arm.level == 0:
                nr = max(L*0.4, arm.radius*0.3)
                o = arm.origin
                li = [max(0,int(w2i(o[j]-nr))) for j in range(3)]
                hi = [min(N,int(w2i(o[j]+nr))+1) for j in range(3)]
                if all(hi[j]>li[j] for j in range(3)):
                    cx=np.linspace(-1+2*li[0]/N,-1+2*hi[0]/N,hi[0]-li[0],endpoint=False,dtype=np.float32)
                    cy=np.linspace(-1+2*li[1]/N,-1+2*hi[1]/N,hi[1]-li[1],endpoint=False,dtype=np.float32)
                    cz=np.linspace(-1+2*li[2]/N,-1+2*hi[2]/N,hi[2]-li[2],endpoint=False,dtype=np.float32)
                    Xl,Yl,Zl=np.meshgrid(cx,cy,cz,indexing='ij')
                    vol[li[0]:hi[0],li[1]:hi[1],li[2]:hi[2]][
                        (Xl-o[0])**2+(Yl-o[1])**2+(Zl-o[2])**2<=nr**2]=1.0
            continue
        o,d=arm.origin,arm.direction; tip=o+d*L
        lo=np.minimum(o,tip)-rm-0.01; hi=np.maximum(o,tip)+rm+0.01
        ix0=max(0,int(w2i(lo[0]))); iy0=max(0,int(w2i(lo[1]))); iz0=max(0,int(w2i(lo[2])))
        ix1=min(N,int(w2i(hi[0]))+1); iy1=min(N,int(w2i(hi[1]))+1); iz1=min(N,int(w2i(hi[2]))+1)
        if ix1<=ix0 or iy1<=iy0 or iz1<=iz0: continue
        cx=np.linspace(-1+2*ix0/N,-1+2*ix1/N,ix1-ix0,endpoint=False,dtype=np.float32)
        cy=np.linspace(-1+2*iy0/N,-1+2*iy1/N,iy1-iy0,endpoint=False,dtype=np.float32)
        cz=np.linspace(-1+2*iz0/N,-1+2*iz1/N,iz1-iz0,endpoint=False,dtype=np.float32)
        Xl,Yl,Zl=np.meshgrid(cx,cy,cz,indexing='ij')
        dX=Xl-o[0];dY=Yl-o[1];dZ=Zl-o[2]
        ax=dX*d[0]+dY*d[1]+dZ*d[2]
        psq=(dX-ax*d[0])**2+(dY-ax*d[1])**2+(dZ-ax*d[2])**2
        af=np.clip(ax/(L+1e-8),0,1)
        rl=rb+(rt-rb)*af; rl2=rl*rl
        inside=(ax>=0)&(ax<=L)&(psq<=rl2)
        inside|=((ax-L)**2+psq<=rt**2)
        vol[ix0:ix1,iy0:iy1,iz0:iz1][inside]=1.0
    return vol


def compute_growth_decomposition(arms, t, resolution=128):
    vt = rasterize_arms_fast(arms, t, resolution)
    v0 = rasterize_arms_fast(arms, 0.0, resolution)
    g = np.clip(vt-v0, 0, 1)
    return {'volume':vt,'deform_mask':v0,'growth_mask':g,
            'solid_fraction':float(vt.sum()/vt.size),
            'growth_fraction':float(g.sum()/vt.size)}


NUM_FRAMES = 30
RASTER_RESOLUTION = 256      # 体素分辨率（投影用）
IMAGE_RESOLUTION = 500        # 输出图像像素分辨率
SCENE_EXTENT = 2.0            # 场景范围 [-1,1]³ → extent=2
ATTENUATION_K = 1.5           # Beer-Lambert 对比度控制
DENSITY_RHO = 3.0             # 枝晶密度值（与 rasterize 的二值×rho一致）

# ================================================================
# Parallel beam projection (replaces Go cone-beam renderer)
# ================================================================

def parallel_beam_project(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Parallel beam projection of a 3D volume at azimuthal angle (beam in xy-plane).

    Convention matches ORTHOPHOTO camera in nerfstudio:
      - beam direction:     (cos θ, sin θ, 0)
      - camera x (→ col):  (-sin θ, cos θ, 0)
      - camera y (→ row):  (0, 0, 1)  (flipped: row 0 = top = +z)

    Args:
        volume: 3D array [Nx, Ny, Nz] sampled on [-1,1]³, density values.
        angle_deg: azimuthal rotation angle in degrees.

    Returns:
        2D line-integral array [H, W] in image orientation (row=v, col=u).
    """
    from scipy.ndimage import rotate as ndrotate

    N = volume.shape[0]
    voxel_size = SCENE_EXTENT / N

    # Rotate volume by -θ around z-axis so beam direction aligns with x-axis
    rotated = ndrotate(volume, -angle_deg, axes=(0, 1),
                       reshape=False, order=1, mode='constant', cval=0.0)

    # Sum along x-axis (beam direction) → shape (Ny, Nz)
    line_integral = np.sum(rotated, axis=0) * voxel_size

    # Convert to image orientation: [row=z_flipped, col=y]
    # Camera y = (0,0,1) → image v increases downward, but z increases upward → flip z
    projection = line_integral.T[::-1, :]

    return projection.astype(np.float32)


def resize_projection(projection: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize projection to target image size via bilinear interpolation."""
    from scipy.ndimage import zoom
    zoom_h = target_h / projection.shape[0]
    zoom_w = target_w / projection.shape[1]
    return zoom(projection, (zoom_h, zoom_w), order=1)


def save_projection_image(projection: np.ndarray, output_path: Path,
                          global_max: float, attenuation_k: float = ATTENUATION_K):
    """
    Save line-integral as Beer-Lambert attenuated 8-bit PNG.
    Matches AttenuationRenderer.forward() exactly:
        background (integral=0) → exp(0) = 1.0 → white (255)
        material   (integral>0) → exp(-k·norm) → darker
    """
    from PIL import Image

    proj = projection.copy()
    if global_max > 0:
        proj = proj / global_max
    attenuation = np.exp(-attenuation_k * proj)
    img_uint8 = np.clip(attenuation * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode='L').save(str(output_path))


# ================================================================
# ORTHOPHOTO transform matrices (from prepare_dendrite_data_growth.py)
# ================================================================

def build_parallel_beam_transform_matrix(theta_deg: float) -> list:
    """
    Build 4×4 camera-to-world matrix for ORTHOPHOTO parallel beam.

    Convention (matches nerfstudio + prepare_dendrite_data_growth.py):
        col 0: camera right   = (-sin θ, cos θ, 0)
        col 1: camera up      = (0, 0, 1)
        col 2: camera backward = (-cos θ, -sin θ, 0)
        col 3: camera position = D × (-cos θ, -sin θ, 0)
    """
    theta = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    D = 3.0
    return [
        [-sin_t, 0.0, -cos_t, -D * cos_t],
        [ cos_t, 0.0, -sin_t, -D * sin_t],
        [   0.0, 1.0,    0.0,        0.0],
        [   0.0, 0.0,    0.0,        1.0],
    ]


def compute_ortho_camera_params(image_h: int, image_w: int,
                                scene_extent: float = SCENE_EXTENT) -> dict:
    """ORTHOPHOTO focal lengths: pixel ↔ world mapping."""
    fl_x = image_w / scene_extent
    fl_y = image_h / scene_extent
    return {
        'camera_model': 'ORTHOPHOTO',
        'camera_angle_x': float(2.0 * np.arctan(scene_extent / 2.0)),
        'fl_x': float(fl_x),
        'fl_y': float(fl_y),
        'cx': float(image_w / 2.0),
        'cy': float(image_h / 2.0),
        'w': int(image_w),
        'h': int(image_h),
    }


# ================================================================
# Main
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic dendrite data")
    parser.add_argument('--output_dir', type=str, default='data/dendrite_v5',
                        help='Output directory for generated data')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    NF = NUM_FRAMES
    last = NF - 1
    for i in range(NF):
        (out / f"images_{i:02d}").mkdir(exist_ok=True)

    # ── Build dendrite arms (unchanged) ──
    print("Building dendrite forest...")
    arms = build_dendrite_arm_table(seed=42)

    # ── Save analytic geometry YAML (unchanged from v4) ──
    init_coll = create_dendrite_collection(arms, 0.0)
    cfg = out / "dendrite.yaml"
    with open(cfg, 'w') as f:
        _yaml_dump(init_coll, f)
    with open(out / "dendrite_00.yaml", 'w') as f:
        _yaml_dump(init_coll, f)
    final_coll = create_dendrite_collection(arms, 1.0)

    # ── Angle assignments (same logic as v4) ──
    # First/last: 16 equispaced angles for canonical training
    # Intermediate: 4 orthogonal angles (sparse views for vfield training)
    full_angles = np.linspace(0, 180, 16, endpoint=False).tolist()
    sparse_angles = [45.0, 90.0, 135.0, 180.0]  # 4 angles

    # ── Camera parameters (ORTHOPHOTO) ──
    cam_params = compute_ortho_camera_params(IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    print(f"Camera: ORTHOPHOTO, fl_x={cam_params['fl_x']:.1f}, "
          f"image={IMAGE_RESOLUTION}×{IMAGE_RESOLUTION}")

    # ── Pass 1: Rasterize all volumes and project, find global max ──
    print(f"\nPass 1: Rasterizing {NF} volumes and projecting...")
    all_projections = {}   # i → (list_of_2D_arrays, list_of_angles)
    global_proj_max = 0.0

    for i in range(NF):
        t = i / (NF - 1)
        angles = full_angles if i == last else sparse_angles

        # Rasterize volume (binary → multiply by density)
        vol = rasterize_arms_fast(arms, t, RASTER_RESOLUTION) * DENSITY_RHO

        # Project at each angle
        projections = []
        for ang in angles:
            proj = parallel_beam_project(vol, ang)
            proj = resize_projection(proj, IMAGE_RESOLUTION, IMAGE_RESOLUTION)
            projections.append(proj)
            global_proj_max = max(global_proj_max, proj.max())

        all_projections[i] = (projections, angles)
        n_ang = len(angles)
        print(f"  Frame {i:02d} (t={t:.3f}): {n_ang} proj, "
              f"max_integral={max(p.max() for p in projections):.4f}")

    print(f"  Global max line integral: {global_proj_max:.4f}")

    # ── Pass 2: Save images with consistent normalization ──
    print(f"\nPass 2: Saving images (attenuation_k={ATTENUATION_K})...")
    all_transforms = {}

    for i in range(NF):
        t = i / (NF - 1)
        projections, angles = all_projections[i]
        image_dir = out / f"images_{i:02d}"

        frames = []
        for ia, (proj, angle) in enumerate(zip(projections, angles)):
            fname = f"train_{ia:02d}.png"
            save_projection_image(proj, image_dir / fname,
                                  global_max=global_proj_max)
            frames.append({
                'file_path': f"images_{i:02d}/{fname}",
                'transform_matrix': build_parallel_beam_transform_matrix(angle),
                'time': t,
            })

        # Eval image: copy first train image (neural_xray convention)
        eval_src = image_dir / "train_00.png"
        eval_dst = image_dir / "eval_00.png"
        if eval_src.exists():
            shutil.copy2(eval_src, eval_dst)
            eval_frame = frames[0].copy()
            eval_frame['file_path'] = f"images_{i:02d}/eval_00.png"
            frames.append(eval_frame)

        all_transforms[i] = frames

    # ── Assemble transforms JSON files ──
    print("\nAssembling transforms...")

    # transforms_00.json (first frame, full angles)
    t00 = {**cam_params, 'frames': all_transforms[0]}
    with open(out / "transforms_00.json", 'w') as f:
        json.dump(t00, f, indent=2)
    print(f"  ✓ transforms_00.json ({len(all_transforms[0])} frames)")

    # transforms_{last}.json (last frame, full angles)
    tN = {**cam_params, 'frames': all_transforms[last]}
    with open(out / f"transforms_{last:02d}.json", 'w') as f:
        json.dump(tN, f, indent=2)
    print(f"  ✓ transforms_{last:02d}.json ({len(all_transforms[last])} frames)")

    # transforms_00_to_{last}.json (aggregated, all frames)
    all_frames = []
    for i in range(NF):
        all_frames.extend(all_transforms[i])
    t_all = {**cam_params, 'frames': all_frames}
    with open(out / f"transforms_00_to_{last:02d}.json", 'w') as f:
        json.dump(t_all, f, indent=2)
    print(f"  ✓ transforms_00_to_{last:02d}.json ({len(all_frames)} frames)")

    # ── Ground truth volumes (unchanged) ──
    print("\nExporting ground truth volumes...")
    gt = out / "ground_truth"
    gt.mkdir(exist_ok=True)
    for i in range(NF):
        d = compute_growth_decomposition(arms, i / (NF - 1), resolution=256)
        np.savez_compressed(str(gt / f'volume_{i:02d}.npz'), volume=d['volume'])
        np.savez_compressed(str(gt / f'deform_mask_{i:02d}.npz'), mask=d['deform_mask'])
        np.savez_compressed(str(gt / f'growth_mask_{i:02d}.npz'), mask=d['growth_mask'])

    # ── Save analytic geometry for last frame (unchanged from v4) ──
    with open(out / f"dendrite_{last:02d}.yaml", 'w') as f:
        _yaml_dump(final_coll, f)

    print(f"\n{'='*50}")
    print(f"✓ Complete!  Output: {out}")
    print(f"  Camera model:  ORTHOPHOTO (parallel beam)")
    print(f"  Frames:        {NF}")
    print(f"  Image size:    {IMAGE_RESOLUTION}×{IMAGE_RESOLUTION}")
    print(f"  Voxel grid:    {RASTER_RESOLUTION}³")
    print(f"  First/last:    {len(full_angles)} angles")
    print(f"  Intermediate:  {len(sparse_angles)} angles (45.0, 90.0, 135.0, 180.0)")
    print(f"  Attenuation k: {ATTENUATION_K}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()