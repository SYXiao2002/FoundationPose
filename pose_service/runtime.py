"""FoundationPose wrapper; model output uses the original mesh coordinates."""

import hashlib
import gc
import logging
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from .protocol import rigid
from .protocol import validate_request
from .screening import screen

LOG = logging.getLogger(__name__)


class FoundationRuntime:
    def __init__(self, mesh_path, *, object_id="eu_plug", iterations=5, screening=None):
        import nvdiffrast.torch as dr
        import trimesh
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
        from Utils import make_mesh_tensors

        path = Path(mesh_path)
        self.object_id = object_id
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.iterations = iterations
        self.screening = screening or {}
        self.mesh = trimesh.load(str(path), force="mesh", process=False)
        if len(self.mesh.vertices) == 0 or len(self.mesh.faces) == 0:
            raise ValueError("mesh is empty")
        self.context = dr.RasterizeCudaContext()
        self._debug_dir = tempfile.TemporaryDirectory(prefix="pose6d-")
        self.estimator = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals,
                                        mesh=self.mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                                        glctx=self.context, debug=3, debug_dir=self._debug_dir.name)
        # register returns the raw OBJ transform; estimator.mesh_tensors is centered.
        self.raw_mesh_tensors = make_mesh_tensors(self.mesh)

    def infer(self, request, debug_dir):
        import torch
        from Utils import nvdiffrast_render

        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        self.validate_request(request)
        # The upstream crop projection multiplies K by double-precision mesh points.
        K = np.asarray(request["K"], dtype=np.float64)
        # MessagePack arrays are read-only byte views; upstream Torch expects writable storage.
        rgb = np.array(request["rgb"], copy=True, order="C")
        depth = np.array(request["depth_m"], copy=True, order="C")
        mask = np.ascontiguousarray(request["mask"], dtype=bool)
        self.estimator.debug = 3
        self.estimator.debug_dir = str(debug_dir)
        registration_started = time.perf_counter()
        pose = self.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=self.iterations)
        torch.cuda.synchronize()
        registration_ms = (time.perf_counter() - registration_started) * 1000
        # register() returns a guessed translation when filtered depth is sparse.
        if self.estimator.best_score is None:
            raise ValueError("registration did not produce a scored pose")
        pose = rigid(pose)
        score = float(self.estimator.best_score.detach().cpu().item())
        if not np.isfinite(score):
            raise ValueError("registration score is not finite")
        try:
            import cv2
            _, rendered, _ = nvdiffrast_render(
                K=K, H=depth.shape[0], W=depth.shape[1],
                ob_in_cams=torch.as_tensor(pose[None], dtype=torch.float32, device="cuda"),
                glctx=self.context, mesh_tensors=self.raw_mesh_tensors)
            rendered = rendered[0].detach().cpu().numpy()
            result = screen(mask, depth, rendered, **self.screening)
            silhouette = np.isfinite(rendered) & (rendered > 0)
            overlay = rgb.copy()
            colors = np.zeros_like(rgb)
            colors[mask & ~silhouette] = (255, 100, 0)
            colors[silhouette & ~mask] = (0, 140, 255)
            colors[silhouette & mask] = (0, 220, 80)
            active = mask | silhouette
            overlay[active] = ((overlay[active].astype(np.uint16) + colors[active]) // 2).astype(np.uint8)
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if not ok:
                raise ValueError("could not encode screening overlay")
            result["overlay_png"] = encoded.tobytes()
        except Exception as exc:
            result = {"status": "invalid", "warning": f"screening failed: {exc}", "metrics": {}}
        return {"type": "result", "version": 1, "request_id": request["request_id"],
                "observation_id": request["observation_id"], "object_id": self.object_id,
                "model_sha256": self.model_sha256, "T_camera_object": pose.tolist(),
                "timings": {"registration_ms": registration_ms, "inference_ms": (time.perf_counter() - started) * 1000,
                            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()},
                "score": score, "best_candidate_id": int(self.estimator.best_id.detach().cpu().item()), "screening": result}

    def validate_request(self, request):
        return validate_request(request, self.object_id, self.model_sha256)


class FoundationRuntimeRegistry:
    """Select one registered mesh at a time while keeping one FoundationPose model on GPU."""

    def __init__(self, models, *, iterations=5, screening=None, runtime_factory=FoundationRuntime):
        if not models:
            raise ValueError("at least one FoundationPose model must be configured")
        self.iterations = iterations
        self.screening = screening or {}
        self.runtime_factory = runtime_factory
        self.models = {}
        self._active = None
        self._lock = threading.RLock()
        for object_id, path in models.items():
            if not isinstance(object_id, str) or not object_id or len(object_id) > 128:
                raise ValueError("model IDs must be nonempty strings up to 128 characters")
            mesh_path = Path(path).expanduser().resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(f"mesh for {object_id} does not exist: {mesh_path}")
            self.models[object_id] = {
                "path": mesh_path,
                "model_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
                "model_asset_id": None,
            }
        self.default_object_id = next(iter(self.models))
        self.object_id = self.default_object_id
        self.model_sha256 = self.models[self.default_object_id]["model_sha256"]

    def model_manifest(self):
        return {object_id: {"model_sha256": model["model_sha256"]}
                for object_id, model in self.models.items()}

    def prepare_archives(self, store, provenance):
        """Snapshot each CPU mesh so old history remains tied to its original geometry."""
        import trimesh

        for object_id, model in self.models.items():
            mesh = trimesh.load(str(model["path"]), force="mesh", process=False)
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                raise ValueError(f"mesh for {object_id} is empty")
            model["model_asset_id"] = store.snapshot_model(
                mesh, model["model_sha256"],
                {**provenance, "object_id": object_id, "mesh_name": model["path"].name})

    def validate_request(self, request):
        from .protocol import validate_request

        if not isinstance(request, dict):
            raise ValueError("expected register request object")
        object_id = request.get("object_id")
        model = self.models.get(object_id)
        if model is None:
            raise ValueError("unknown object_id")
        return validate_request(request, object_id, model["model_sha256"])

    def model_asset_id(self, object_id):
        try:
            asset_id = self.models[object_id]["model_asset_id"]
        except KeyError as exc:
            raise ValueError("unknown object_id") from exc
        if asset_id is None:
            raise RuntimeError("model archives have not been prepared")
        return asset_id

    def _load(self, object_id):
        with self._lock:
            if object_id not in self.models:
                raise ValueError("unknown object_id")
            if self._active is not None and self._active.object_id == object_id:
                return self._active
            if self._active is not None:
                old = self._active
                self._active = None
                try:
                    temporary = getattr(old, "_debug_dir", None)
                    if temporary is not None:
                        temporary.cleanup()
                except Exception:
                    LOG.debug("could not clean temporary files for %s", old.object_id, exc_info=True)
                del old
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    LOG.debug("could not release the previous model CUDA cache", exc_info=True)
            model = self.models[object_id]
            LOG.info("loading FoundationPose object_id=%s mesh=%s", object_id, model["path"])
            self._active = self.runtime_factory(
                model["path"], object_id=object_id, iterations=self.iterations, screening=self.screening)
            if self._active.model_sha256 != model["model_sha256"]:
                self._active = None
                raise ValueError(f"mesh changed after registry initialization for {object_id}")
            return self._active

    def infer(self, request, debug_dir):
        self.validate_request(request)
        return self._load(request["object_id"]).infer(request, debug_dir)
