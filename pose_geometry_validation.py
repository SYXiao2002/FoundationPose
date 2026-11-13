"""First-frame mesh and RGB-D screening, independent of pose estimation."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ValidationConfig:
    # Provisional eu-plug screening thresholds; calibrate against reviewed examples.
    min_mask_depth_points: int = 300
    min_mask_depth_fraction: float = 0.5
    min_overlap_depth_points: int = 200
    min_iou: float = 0.5
    min_mask_coverage: float = 0.7
    max_depth_median_mm: float = 10.0
    max_depth_p90_mm: float = 25.0
    depth_tolerance_mm: float = 10.0
    min_depth_inlier_fraction: float = 0.7
    occlusion_tolerance_mm: float = 10.0
    max_occluded_fraction: float = 0.3


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def evaluate_geometry(mask, depth_mm, rendered_depth_m, config=ValidationConfig()):
    """Compare a visible instance mask and raw depth with the full rendered mesh."""
    mask = np.asarray(mask) > 0
    obs = np.asarray(depth_mm, dtype=np.float64) / 1000.0
    rendered = np.asarray(rendered_depth_m, dtype=np.float64)
    if mask.shape != obs.shape or mask.shape != rendered.shape or mask.ndim != 2:
        raise ValueError("Mask and depth images must have the same 2D shape")
    valid_obs = np.isfinite(obs) & (obs > 0)
    render = np.isfinite(rendered) & (rendered > 0)
    overlap = mask & render
    valid = overlap & valid_obs
    n_mask, n_render, n_overlap = (int(np.count_nonzero(x)) for x in (mask, render, overlap))
    n_mask_depth = int(np.count_nonzero(mask & valid_obs))
    n_valid = int(np.count_nonzero(valid))
    in_render_obs = render & valid_obs
    n_render_obs = int(np.count_nonzero(in_render_obs))
    occluded = in_render_obs & (rendered - obs > config.occlusion_tolerance_mm / 1000)
    n_occluded = int(np.count_nonzero(occluded))
    visible_render = render & ~occluded
    visible_union = int(np.count_nonzero(mask | visible_render))
    errors = (rendered[valid] - obs[valid]) * 1000
    absolute = np.abs(errors)
    metrics = {
        "mask_pixels": n_mask, "render_pixels": n_render, "overlap_pixels": n_overlap,
        "mask_valid_depth_pixels": n_mask_depth,
        "mask_valid_depth_fraction": ratio(n_mask_depth, n_mask),
        "overlap_valid_depth_pixels": n_valid,
        "overlap_valid_depth_fraction_of_mask": ratio(n_valid, n_mask),
        "silhouette_iou": ratio(n_overlap, int(np.count_nonzero(mask | render))),
        "mask_coverage": ratio(n_overlap, n_mask),
        "render_coverage": ratio(n_overlap, n_render),
        "depth_median_mm": float(np.median(absolute)) if n_valid else None,
        "depth_p90_mm": float(np.percentile(absolute, 90)) if n_valid else None,
        "signed_depth_median_mm": float(np.median(errors)) if n_valid else None,
        "depth_inlier_fraction": float(np.mean(absolute <= config.depth_tolerance_mm)) if n_valid else None,
        "render_valid_depth_pixels": n_render_obs,
        "render_unknown_depth_fraction": ratio(n_render - n_render_obs, n_render),
        "occluded_pixels": n_occluded,
        "occluded_fraction_of_render_valid_depth": ratio(n_occluded, n_render_obs),
        "visible_silhouette_iou": ratio(int(np.count_nonzero(mask & visible_render)), visible_union),
    }
    warnings = []
    def warn(code, condition):
        if condition:
            warnings.append(code)
    warn("empty_mask", n_mask == 0)
    warn("empty_render", n_render == 0)
    warn("few_mask_depth_points", n_mask_depth < config.min_mask_depth_points)
    warn("low_mask_depth_fraction", n_mask > 0 and metrics["mask_valid_depth_fraction"] < config.min_mask_depth_fraction)
    warn("few_overlap_depth_points", n_valid < config.min_overlap_depth_points)
    warn("low_iou", metrics["silhouette_iou"] is not None and metrics["silhouette_iou"] < config.min_iou)
    warn("low_mask_coverage", metrics["mask_coverage"] is not None and metrics["mask_coverage"] < config.min_mask_coverage)
    # Do not interpret an error distribution from an insufficient sample.
    if n_valid >= config.min_overlap_depth_points:
        warn("high_depth_median", metrics["depth_median_mm"] > config.max_depth_median_mm)
        warn("high_depth_p90", metrics["depth_p90_mm"] > config.max_depth_p90_mm)
        warn("low_depth_inliers", metrics["depth_inlier_fraction"] < config.min_depth_inlier_fraction)
    warn("high_occlusion", n_render_obs > 0 and metrics["occluded_fraction_of_render_valid_depth"] > config.max_occluded_fraction)
    invalid = n_mask == 0 or n_render == 0
    insufficient = (n_mask_depth < config.min_mask_depth_points or n_valid < config.min_overlap_depth_points
                    or "low_mask_depth_fraction" in warnings)
    suspect = any(w in warnings for w in ("low_iou", "low_mask_coverage", "high_depth_median", "high_depth_p90", "low_depth_inliers"))
    status = "invalid" if invalid else "insufficient_evidence" if insufficient else "suspect" if suspect else "pass_screening"
    return {"status": status, "warnings": warnings, "metrics": metrics}


class MeshDepthRenderer:
    def __init__(self, mesh, K, width, height):
        # Set before importing OpenGL; callers may explicitly choose another backend.
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender
        self.pyrender = pyrender
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        self.node = self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.001, zfar=10.0)
        self.scene.add(camera, pose=np.diag([1., -1., -1., 1.]))

    def render(self, pose):
        self.scene.set_pose(self.node, pose=pose)
        # pyrender's DEPTH_ONLY path unconditionally enables face culling.
        _, depth = self.renderer.render(self.scene, flags=self.pyrender.RenderFlags.SKIP_CULL_FACES)
        return depth

    def close(self):
        self.renderer.delete()


def config_dict(config=ValidationConfig()):
    return asdict(config)
