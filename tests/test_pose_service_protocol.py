import numpy as np
import pytest
from pose_service.runtime import FoundationRuntimeRegistry
from pose_service.protocol import validate_request
from pose_service.server import parse_model_specs
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


def test_model_registry_validates_ids_and_switches_runtime_by_requested_mesh(tmp_path):
    old_mesh = tmp_path / "old.obj"
    new_mesh = tmp_path / "new.obj"
    old_mesh.write_text("old model")
    new_mesh.write_text("new model")

    class Runtime:
        def __init__(self, path, *, object_id, **_kwargs):
            self.object_id = object_id
            self.model_sha256 = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            self.path = path

        def infer(self, request, _debug_dir):
            return {"object_id": self.object_id, "mesh": self.path.name}

    registry = FoundationRuntimeRegistry({"old": old_mesh, "new": new_mesh}, runtime_factory=Runtime)
    assert registry.default_object_id == "old"
    assert set(registry.model_manifest()) == {"old", "new"}

    request = {"type": "register", "version": 1, "request_id": "r", "observation_id": "o",
               "object_id": "new", "model_sha256": registry.models["new"]["model_sha256"],
               "rgb": np.zeros((2, 2, 3), np.uint8), "depth_m": np.ones((2, 2), np.float32),
               "mask": np.ones((2, 2), np.uint8), "K": np.eye(3)}
    registry.validate_request(request)
    assert registry.infer(request, None) == {"object_id": "new", "mesh": "new.obj"}
    assert registry.infer({**request, "object_id": "old",
                           "model_sha256": registry.models["old"]["model_sha256"]}, None) == {
        "object_id": "old", "mesh": "old.obj"}
    with pytest.raises(ValueError, match="unknown object_id"):
        registry.validate_request({**request, "object_id": "not-registered"})


def test_model_spec_parser_supports_multiple_models_and_single_model_compatibility(tmp_path):
    first = str(tmp_path / "first.obj")
    second = str(tmp_path / "second.obj")
    assert parse_model_specs([f"plug={first}", f"plug2={second}"], None, "unused") == {
        "plug": first, "plug2": second}
    assert parse_model_specs([], first, "legacy") == {"legacy": first}
    with pytest.raises(ValueError, match="unique ID=PATH"):
        parse_model_specs([f"plug={first}", f"plug={second}"], None, "unused")

