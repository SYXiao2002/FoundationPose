import numpy as np
import pytest
from pose_service.protocol import validate_request
from pose_service.screening import screen


def test_screening_metrics_and_status():
    mask = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    observed = np.array([[1, 1,], [1, 0]], dtype=np.float32)
    rendered = np.array([[1.002, 1.004], [0, 1]], dtype=np.float32)
    result = screen(mask, observed, rendered, min_valid_pixels=2, min_iou=.4, max_median_m=.005)
    assert result["status"] == "pass_screening"
    assert result["metrics"]["silhouette_iou"] == .5
    assert result["metrics"]["valid_depth_pixels"] == 2
    assert result["metrics"]["depth_median_mm"] == pytest.approx(3, abs=.001)
    assert screen(mask, observed, rendered, min_valid_pixels=3)["status"] == "insufficient_evidence"
    assert screen(mask, observed, np.zeros_like(rendered))["status"] == "invalid"


def test_validate_request_rejects_weak_or_mismatched_data():
    request = {"type": "register", "version": 1, "request_id": "r", "observation_id": "o",
               "object_id": "eu_plug", "model_sha256": "hash", "rgb": np.zeros((2, 2, 3), np.uint8),
               "depth_m": np.ones((2, 2), np.float32), "mask": np.ones((2, 2), np.uint8),
               "K": np.eye(3)}
    validate_request(request, "eu_plug", "hash")
    request["mask"][0, 0] = 0
    with pytest.raises(ValueError, match="insufficient"):
        validate_request(request, "eu_plug", "hash")
    request["mask"][0, 0] = 1
    with pytest.raises(ValueError, match="mismatched"):
        validate_request(request, "eu_plug", "other")

