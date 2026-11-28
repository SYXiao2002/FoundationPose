"""WAPR adapter for the shared single-frame pose WebSocket and history service."""

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from .protocol import rigid, validate_request
from .screening import screen


class WAPRRuntime:
    algorithm = "wapr"

    def __init__(self, mesh_path, *, object_id="eu_plug", backend="torch", screening=None):
        import nvdiffrast.torch as dr
        import torch
        import trimesh
        from Utils import make_mesh_tensors
        from wapr import WAPREstimator

        if backend not in ("torch", "trt"):
            raise ValueError("WAPR backend must be torch or trt")
        # WAPR resolves this before constructing its networks. Keep that choice
        # local to the service process and explicit in the archived parameters.
        os.environ["WAPR_BACKEND"] = backend
        from wapr import recipe

        recipe.backend = backend
        self.object_id = object_id
        path = Path(mesh_path)
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.screening = screening or {}
        self.backend = backend
        self.score_top_k = 10
        self.mesh = trimesh.load(str(path), force="mesh", process=False)
        if not isinstance(self.mesh, trimesh.Trimesh) or not len(self.mesh.vertices) or not len(self.mesh.faces):
            raise ValueError("mesh is empty or is not a single triangle mesh")
        self.diameter_m = float(np.linalg.norm(self.mesh.extents))
        if not np.isfinite(self.diameter_m) or self.diameter_m <= 0:
            raise ValueError("mesh has zero or invalid extent")

        self.estimator = WAPREstimator(device="cuda:0")
        # WAPR stores a centered copy for its crops and maps its returned pose
        # back to the original OBJ frame. Keep the raw mesh for screening.
        self.pose_mesh = self.estimator.prepare_meshes([self.mesh])[0]
        self._model_center = np.asarray(self.pose_mesh.metadata.get("model_center", np.zeros(3)), dtype=np.float64)
        self.recipe = recipe
        self._recipe = {name: getattr(recipe, name) for name in
                        ("n_view", "n_inplane", "wapr_iters", "sapr_iters", "wapr_max_rot_rad",
                         "sapr_max_rot_rad", "refine_crop_ratio", "wbps_crop_ratio", "crop_px")}
        self.context = dr.RasterizeCudaContext()
        self.raw_mesh_tensors = make_mesh_tensors(self.mesh)
        self.torch = torch

    def validate_request(self, request):
        return validate_request(request, self.object_id, self.model_sha256)

    def close(self):
        """Drop WAPR's process-wide renderer cache before another algorithm loads."""
        renderer = getattr(self.estimator, "renderer", None)
        try:
            from wapr import ogl

            if ogl._RUNTIMES.get(0) is renderer:
                ogl._RUNTIMES.pop(0, None)
        except (ImportError, AttributeError):
            pass
        if renderer is not None:
            for name in ("_meshes", "_mesh_by_obj", "_cache", "_mesh_cache"):
                values = getattr(renderer, name, None)
                if hasattr(values, "clear"):
                    values.clear()
            native = getattr(renderer, "_rt", None)
            clear_meshes = getattr(native, "clear_all_meshes", None)
            if callable(clear_meshes):
                try:
                    clear_meshes()
                except Exception:
                    pass
        self.estimator.renderer = None
        self.estimator.nets.clear()
        self.context = None
        self.raw_mesh_tensors = None
        self.pose_mesh = None

    def infer(self, request, _debug_dir):
        from Utils import nvdiffrast_render
        from .wapr_score_vis import write_wapr_score_image

        self.validate_request(request)
        torch = self.torch
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        rgb = np.array(request["rgb"], copy=True, order="C")
        depth = np.array(request["depth_m"], dtype=np.float32, copy=True, order="C")
        mask = np.ascontiguousarray(request["mask"])
        K = np.asarray(request["K"], dtype=np.float32)

        registration_started = time.perf_counter()
        estimated = self.estimator.estimate_one_category_one_instance(
            rgb, depth, K, self.pose_mesh, self.diameter_m, mask=mask,
            score_top_k=self.score_top_k,
        )
        torch.cuda.synchronize()
        registration_ms = (time.perf_counter() - registration_started) * 1000
        pose = rigid(estimated["pose_4x4"])
        score = float(estimated["score_6d"])
        if not np.isfinite(score):
            raise ValueError("WAPR score_6d is not finite")

        score_vis_started = time.perf_counter()
        score_candidate_count = 0
        if self.score_top_k:
            score_candidate_count = write_wapr_score_image(
                estimated["score_candidates"], score, Path(_debug_dir) / "vis_score.png",
                top_k=self.score_top_k,
            )
        score_vis_ms = (time.perf_counter() - score_vis_started) * 1000

        try:
            import cv2

            _, rendered, _ = nvdiffrast_render(
                K=K, H=depth.shape[0], W=depth.shape[1],
                ob_in_cams=torch.as_tensor(pose[None], dtype=torch.float32, device="cuda"),
                glctx=self.context, mesh_tensors=self.raw_mesh_tensors,
            )
            rendered = rendered[0].detach().cpu().numpy()
            screening = screen(mask, depth, rendered, **self.screening)
            silhouette = np.isfinite(rendered) & (rendered > 0)
            overlay = rgb.copy()
            colors = np.zeros_like(rgb)
            colors[mask.astype(bool) & ~silhouette] = (255, 100, 0)
            colors[silhouette & ~mask.astype(bool)] = (0, 140, 255)
            colors[silhouette & mask.astype(bool)] = (0, 220, 80)
            active = mask.astype(bool) | silhouette
            overlay[active] = ((overlay[active].astype(np.uint16) + colors[active]) // 2).astype(np.uint8)
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if not ok:
                raise ValueError("could not encode screening overlay")
            screening["overlay_png"] = encoded.tobytes()
        except Exception as exc:
            screening = {"status": "invalid", "warning": f"screening failed: {exc}", "metrics": {}}

        return {
            "type": "result", "version": 1,
            "request_id": request["request_id"], "observation_id": request["observation_id"],
            "object_id": self.object_id, "model_sha256": self.model_sha256,
            "algorithm": self.algorithm, "score_type": "wapr_score_6d",
            "T_camera_object": pose.tolist(), "score": score,
            "score_preview_available": score_candidate_count > 0,
            "score_candidate_count": score_candidate_count,
            "screening": screening,
            "parameters": {"backend": self.backend, "renderer": os.environ.get("WAPR_RENDERER", "ogl"),
                           "recipe": self._recipe, "model_center": self._model_center.tolist()},
            "timings": {
                "registration_ms": registration_ms,
                "score_vis_ms": score_vis_ms,
                "inference_ms": (time.perf_counter() - started) * 1000,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
        }

    def infer_batch(self, request, _debug_dir):
        """Batch-infer multiple instances of the same mesh in one WAPR call.

        Uses ``estimate_many_categories_many_instances`` so registration and
        scoring happen on batched GPU tensors.  Instances are processed in
        chunks of ``_BATCH_CHUNK`` to avoid GPU OOM on large scenes.
        Returns per-instance results with aggregate timing.  The request
        carries a shared RGB/depth/K and an ``instances`` list where each
        item has ``instance_id`` and ``mask``.
        """
        from Utils import nvdiffrast_render
        from .wapr_score_vis import write_wapr_score_image

        _BATCH_CHUNK = 30

        instances_payload = request["instances"]
        rgb = np.array(request["rgb"], copy=True, order="C")
        depth = np.array(request["depth_m"], dtype=np.float32, copy=True, order="C")
        K = np.asarray(request["K"], dtype=np.float64)
        torch = self.torch

        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        all_results = []
        registration_ms_total = 0.0
        for chunk_start in range(0, len(instances_payload), _BATCH_CHUNK):
            chunk = instances_payload[chunk_start:chunk_start + _BATCH_CHUNK]
            wapr_instances = []
            for item in chunk:
                wapr_instances.append({
                    "mesh": self.pose_mesh,
                    "diameter_m": self.diameter_m,
                    "mask": np.ascontiguousarray(item["mask"]),
                })
            chunk_started = time.perf_counter()
            chunk_results = self.estimator.estimate_many_categories_many_instances(
                rgb, depth, K, wapr_instances,
                score_top_k=10,
            )
            torch.cuda.synchronize()
            registration_ms_total += (time.perf_counter() - chunk_started) * 1000
            all_results.extend(chunk_results)

        output = []
        for index, (item, result) in enumerate(zip(instances_payload, all_results)):
            if result is None:
                output.append({
                    "instance_id": item["instance_id"],
                    "status": "failed",
                    "reason": "WAPR returned no result for this instance",
                })
                continue

            pose = rigid(result["pose_4x4"])
            score = float(result["score_6d"])
            if not np.isfinite(score):
                output.append({"instance_id": item["instance_id"], "status": "failed",
                               "reason": "WAPR returned a non-finite score"})
                continue
            mask = np.ascontiguousarray(item["mask"])
            try:
                _, rendered, _ = nvdiffrast_render(
                    K=np.asarray(K, dtype=np.float32), H=depth.shape[0], W=depth.shape[1],
                    ob_in_cams=torch.as_tensor(pose[None], dtype=torch.float32, device="cuda"),
                    glctx=self.context, mesh_tensors=self.raw_mesh_tensors,
                )
                rendered = rendered[0].detach().cpu().numpy()
                screening = screen(mask != 0, depth, rendered, **self.screening)
            except Exception as exc:
                screening = {"status": "invalid", "warning": f"screening failed: {exc}", "metrics": {}}

            item_output = {
                "instance_id": item["instance_id"],
                "status": "ok",
                "pose_camera_object": pose.tolist(),
                "score": score,
                "algorithm": "wapr",
                "screening": screening,
            }
            try:
                preview_path = Path(_debug_dir) / f"instance-{index}" / "vis_score.png"
                write_wapr_score_image(result["score_candidates"], score, preview_path, top_k=10)
                item_output["_history_score_preview_path"] = str(preview_path)
            except (KeyError, OSError, ValueError):
                pass
            output.append(item_output)

        peak = torch.cuda.max_memory_allocated()
        return {
            "type": "batch_result",
            "version": 1,
            "request_id": request["request_id"],
            "observation_id": request.get("observation_id"),
            "algorithm": "wapr",
            "score_type": "wapr_score_6d",
            "results": output,
            "instance_count": len(instances_payload),
            "timings": {
                "registration_ms": registration_ms_total,
                "inference_ms": (time.perf_counter() - started) * 1000,
                "peak_gpu_allocated_bytes": peak,
                "chunk_size": _BATCH_CHUNK,
            },
        }
