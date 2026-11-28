"""Validate a bounded single-frame FoundationPose request."""

import numpy as np

VERSION = 1


def validate_request(request, model_id, model_sha256):
    if not isinstance(request, dict) or request.get("type") != "register" or request.get("version") != VERSION:
        raise ValueError("expected version 1 register request")
    for key in ("request_id", "observation_id"):
        value = request.get(key)
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise ValueError(f"invalid {key}")
    if request.get("object_id") != model_id or request.get("model_sha256") != model_sha256:
        raise ValueError("unknown object or mismatched model hash")
    algorithm = request.setdefault("algorithm", "foundationpose")
    if algorithm not in ("foundationpose", "wapr"):
        raise ValueError("unsupported pose algorithm")
    rgb, depth, mask = (request.get(k) for k in ("rgb", "depth_m", "mask"))
    if not all(isinstance(v, np.ndarray) for v in (rgb, depth, mask)):
        raise ValueError("images must be arrays")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be HxWx3 uint8")
    h, w = rgb.shape[:2]
    if not 0 < h <= 4096 or not 0 < w <= 4096:
        raise ValueError("invalid image dimensions")
    if depth.shape != (h, w) or depth.dtype != np.float32 or mask.shape != (h, w) or mask.dtype != np.uint8:
        raise ValueError("depth must be float32 and mask uint8 matching RGB")
    if not np.isfinite(depth).all() or (depth < 0).any() or not np.isin(mask, (0, 1)).all():
        raise ValueError("depth must be finite nonnegative and mask binary")
    if np.count_nonzero(mask & (depth >= 0.001)) < 4:
        raise ValueError("insufficient valid depth inside mask")
    K = np.asarray(request.get("K"), dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or abs(K[1, 0]) > 1e-8):
        raise ValueError("invalid camera intrinsics")
    for key in ("capture_clock", "camera_frame", "instance_id", "mask_source", "source"):
        value = request.get(key)
        if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 128):
            raise ValueError(f"invalid {key}")
    stamp = request.get("capture_timestamp_ns")
    if stamp is not None:
        if not isinstance(stamp, (str, int)) or isinstance(stamp, bool) or not str(stamp).isdigit() or not 0 < int(stamp) < 2**63:
            raise ValueError("invalid capture_timestamp_ns")
    return request


def validate_batch_request(request, model_id, model_sha256):
    """Validate a batch_register request carrying multiple instance masks."""
    if not isinstance(request, dict) or request.get("type") != "batch_register" or request.get("version") != VERSION:
        raise ValueError("expected version 1 batch_register request")
    for key in ("request_id",):
        value = request.get(key)
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise ValueError(f"invalid {key}")
    if request.get("object_id") != model_id or request.get("model_sha256") != model_sha256:
        raise ValueError("unknown object or mismatched model hash")
    rgb, depth = (request.get(k) for k in ("rgb", "depth_m"))
    if not all(isinstance(v, np.ndarray) for v in (rgb, depth)):
        raise ValueError("images must be arrays")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be HxWx3 uint8")
    h, w = rgb.shape[:2]
    if not 0 < h <= 4096 or not 0 < w <= 4096:
        raise ValueError("invalid image dimensions")
    if depth.shape != (h, w) or depth.dtype != np.float32:
        raise ValueError("depth must be float32 matching RGB")
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("depth must be finite nonnegative")
    K = np.asarray(request.get("K"), dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or abs(K[1, 0]) > 1e-8):
        raise ValueError("invalid camera intrinsics")
    instances = request.get("instances")
    if not isinstance(instances, list) or len(instances) < 2:
        raise ValueError("batch_register requires at least 2 instances")
    if len(instances) > 64:
        raise ValueError("batch_register supports at most 64 instances")
    seen_ids = set()
    for idx, item in enumerate(instances):
        if not isinstance(item, dict):
            raise ValueError(f"instance {idx} must be an object")
        iid = item.get("instance_id")
        if not isinstance(iid, str) or not 1 <= len(iid) <= 128:
            raise ValueError(f"instance {idx} has invalid instance_id")
        if iid in seen_ids:
            raise ValueError(f"duplicate instance_id {iid}")
        seen_ids.add(iid)
        mask = item.get("mask")
        if not isinstance(mask, np.ndarray) or mask.shape != (h, w) or mask.dtype != np.uint8:
            raise ValueError(f"instance {iid} mask must be HxW uint8")
        if not np.isin(mask, (0, 1)).all():
            raise ValueError(f"instance {iid} mask must be binary")
        if np.count_nonzero(mask & (depth >= 0.001)) < 4:
            raise ValueError(f"instance {iid} has insufficient valid depth inside mask")
    return request


def rigid(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("expected a finite 4x4 transform")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("transform must be a proper rigid transform")
    return matrix

