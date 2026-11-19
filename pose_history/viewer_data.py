"""Adapt an immutable history record to the existing multi-pose viewer."""
import io
import json
import uuid

import cv2
import numpy as np
from PIL import Image

from pose_service.protocol import rigid
from .store import digest_file, ArchiveCorruptError


class HistoryViewer:
    def __init__(self, store, record):
        self.store, self.record = store, record
        self.path = store.record_dir(record)
        self.manifest = json.loads(store.artifact(record, "manifest.json").read_text())

    def read_path(self, name):
        path = self.path / name
        if not path.is_file() or digest_file(path) != self.manifest["files"].get(name):
            raise ArchiveCorruptError(f"missing or corrupt archive input: {name}")
        return path

    def result(self):
        return json.loads(self.read_path("result.json").read_text())

    def info(self):
        result = self.result()
        pose = result.get("T_camera_object")
        if pose is not None:
            pose = rigid(pose).tolist()
        screening = result.get("screening") or {"status": "invalid", "warning": result.get("message"), "metrics": {}}
        metrics = dict(screening.get("metrics", {}))
        # The online screen and offline reference viewer have different metric names.
        metrics.setdefault("overlap_valid_depth_pixels", metrics.get("valid_depth_pixels"))
        metrics.setdefault("depth_inlier_fraction", metrics.get("depth_inlier_ratio"))
        return {"scene": self.record["object_id"], "frame": self.record["history_id"],
                "point_stride": 2, "image_size": self.record["image_size"],
                "K": np.loadtxt(self.read_path("cam_K.txt")).reshape(3, 3).tolist(),
                "mesh_name": self.record["object_id"],
                "validation": {"profile": "archived", "thresholds": self.manifest.get("parameters", {}).get("screening", {})},
                "history": self.record,
                "instances": [{"id": 1, "label": self.record.get("instance_id") or self.record["object_id"],
                               "color": "#168ca3", "pose": pose, "origin": [r[3] for r in pose[:3]] if pose else None,
                               "score": result.get("score"), "validation": {
                                   "status": screening.get("status", "invalid"),
                                   "warnings": screening.get("warnings") or ([screening["warning"]] if screening.get("warning") else []),
                                   "metrics": metrics}}]}

    def cloud(self, stride=2):
        if not 1 <= stride <= 16:
            raise ValueError("stride must be 1..16")
        cache = self.path / "cache" / f'cloud-v1-{self.record["input_sha256"]}-{stride}.bin'
        # Verify original inputs even when a derived cache exists.
        rgb_path, depth_path, k_path = (self.read_path(p) for p in ("RGB.png", "depth_m.npy", "cam_K.txt"))
        if cache.exists():
            return cache.read_bytes()
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))[::stride, ::stride]
        depth = np.load(depth_path, allow_pickle=False)[::stride, ::stride]
        K = np.loadtxt(k_path).reshape(3, 3)
        v, u = np.indices(depth.shape)
        valid = np.isfinite(depth) & (depth > 0)
        rays = np.linalg.inv(K) @ np.stack((u[valid] * stride, v[valid] * stride, np.ones(valid.sum())))
        xyz = (rays * (depth[valid] / rays[2])).T.astype("<f4")
        data = np.asarray([len(xyz)], dtype="<u4").tobytes() + xyz.tobytes() + rgb[valid].astype(np.uint8).tobytes()
        cache.parent.mkdir(exist_ok=True)
        temporary = cache.with_suffix("." + uuid.uuid4().hex)
        try:
            temporary.write_bytes(data)
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
        return data

    def model(self):
        asset_id = self.record["model_asset_id"]
        if not asset_id or len(asset_id) != 64 or any(c not in "0123456789abcdef" for c in asset_id):
            raise ValueError("model snapshot unavailable")
        directory = self.store.root / "models" / asset_id
        self.store.validate_directory(directory)
        with np.load(directory / "mesh.npz", allow_pickle=False) as mesh:
            return {"vertices": mesh["vertices"].tolist(), "faces": mesh["faces"].tolist()}

    def mask_map(self):
        mask = np.asarray(Image.open(self.read_path("mask.png"))) > 0
        labels = np.zeros((*mask.shape, 3), dtype=np.uint8)
        labels[mask, 0] = 1
        output = io.BytesIO()
        Image.fromarray(labels).save(output, format="PNG")
        return output.getvalue()

    def mask_preview(self):
        image = cv2.imread(str(self.read_path("RGB.png")))
        mask = cv2.imread(str(self.read_path("mask.png")), cv2.IMREAD_GRAYSCALE)
        active = mask > 0
        image[active] = (image[active].astype(np.float32) * .47 + np.asarray([163, 140, 22]) * .53).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (163, 140, 22), 2)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("could not encode mask")
        return encoded.tobytes()
