"""Archive the actual GPU worker's result even after its client disconnects."""
import logging
from pathlib import Path
import time

LOG = logging.getLogger(__name__)


class RecordingRuntime:
    def __init__(self, runtime, store, model_asset_id, parameters=None):
        self.runtime, self.store = runtime, store
        self.model_asset_id = model_asset_id
        self.parameters = parameters or {}
        self.object_id = getattr(runtime, "object_id", None)
        self.model_sha256 = getattr(runtime, "model_sha256", None)
        self.default_object_id = getattr(runtime, "default_object_id", self.object_id)
        self.algorithms = getattr(runtime, "algorithms", ("foundationpose",))
        self.default_algorithm = getattr(runtime, "default_algorithm", "foundationpose")

    def model_manifest(self):
        if hasattr(self.runtime, "model_manifest"):
            return self.runtime.model_manifest()
        return {self.object_id: {"model_sha256": self.model_sha256}}

    def validate_request(self, request):
        if hasattr(self.runtime, "validate_request"):
            return self.runtime.validate_request(request)
        from pose_service.protocol import validate_request
        return validate_request(request, self.object_id, self.model_sha256)

    def validate_batch_request(self, request):
        if hasattr(self.runtime, "validate_batch_request"):
            return self.runtime.validate_batch_request(request)
        raise ValueError("batch inference is not supported by the active runtime")

    def replay(self, request):
        return self.store.replay(request)

    def infer(self, request):
        started = time.perf_counter()
        asset_id = (self.runtime.model_asset_id(request["object_id"])
                    if hasattr(self.runtime, "model_asset_id") else self.model_asset_id)
        algorithm = request.get("algorithm", self.default_algorithm)
        parameters = {**self.parameters, "object_id": request["object_id"], "algorithm": algorithm}
        if algorithm == "wapr":
            parameters.pop("debug", None)
            parameters.pop("iterations", None)
        record, created = self.store.begin(request, asset_id, parameters)
        if not created:
            return self.store.replay(request)
        try:
            path = self.store.save_inputs(record, request)
        except Exception as exc:
            self.store.archive_failure(record, exc)
            raise
        try:
            response = self.runtime.infer(request, path / "debug")
        except Exception as exc:
            LOG.exception("registration failed for %s", record["history_id"])
            response = {"type": "error", "version": 1, "request_id": request["request_id"],
                        "observation_id": request["observation_id"], "code": "inference_failed", "message": str(exc)}
        elapsed = (time.perf_counter() - started) * 1000
        archive_started = time.perf_counter()
        try:
            archived = self.store.publish(record, response, elapsed_ms=elapsed)
            archive_ms = (time.perf_counter() - archive_started) * 1000
            archived.setdefault("timings", {})["archive_ms"] = archive_ms
            LOG.info("history=%s worker_ms=%.1f archive_ms=%.1f", record["history_id"], elapsed, archive_ms)
            # Keep the established binary overlay field on the inference wire.
            if response.get("screening", {}).get("overlay_png"):
                archived["screening"]["overlay_png"] = response["screening"]["overlay_png"]
            return archived
        except Exception as exc:
            self.store.archive_failure(record, exc)
            response.update(history_id=record["history_id"], history_path=record["history_path"],
                            archive_status="failed", archive_error=str(exc))
            return response

    def infer_batch(self, request):
        """Run WAPR once and archive each instance as a normal pose record."""
        import shutil
        import tempfile
        import uuid

        self.validate_batch_request(request)
        algorithm = "wapr"
        asset_id = (self.runtime.model_asset_id(request["object_id"])
                    if hasattr(self.runtime, "model_asset_id") else self.model_asset_id)
        parameters = {**self.parameters, "object_id": request["object_id"], "algorithm": algorithm,
                      "batch": True, "batch_request_id": request["request_id"],
                      "batch_instance_count": len(request["instances"])}
        parameters.pop("debug", None)
        parameters.pop("iterations", None)

        with tempfile.TemporaryDirectory(prefix="pose6d-batch-") as debug_dir:
            records = {}
            for index, instance in enumerate(request["instances"]):
                instance_id = instance["instance_id"]
                instance_request = {
                    "type": "register", "version": request["version"],
                    "request_id": uuid.uuid5(uuid.NAMESPACE_URL,
                                              f"pose6d-batch:{request['request_id']}:{instance_id}").hex,
                    "observation_id": request["observation_id"], "object_id": request["object_id"],
                    "model_sha256": request["model_sha256"], "algorithm": algorithm,
                    "rgb": request["rgb"], "depth_m": request["depth_m"],
                    "K": request["K"], "mask": instance["mask"],
                    "instance_id": instance_id, "mask_source": "sam2",
                }
                for key in ("capture_timestamp_ns", "capture_clock", "camera_frame"):
                    if key in request:
                        instance_request[key] = request[key]
                record, created = self.store.begin(instance_request, asset_id,
                                                   {**parameters, "batch_index": index})
                staging = self.store.staging(record)
                if created:
                    try:
                        staging = self.store.save_inputs(record, instance_request)
                    except Exception as exc:
                        self.store.archive_failure(record, exc)
                        raise
                records[instance_id] = (instance_request, record, staging, created)

            started = time.perf_counter()
            try:
                batch_result = self.runtime.infer_batch(request, debug_dir)
                if not isinstance(batch_result, dict) or batch_result.get("type") != "batch_result":
                    raise ValueError("WAPR returned an invalid batch result")
                returned = {}
                for item in batch_result.get("results", []):
                    instance_id = item.get("instance_id") if isinstance(item, dict) else None
                    if instance_id not in records or instance_id in returned:
                        raise ValueError("WAPR returned duplicate or unknown batch instances")
                    returned[instance_id] = item
            except Exception as exc:
                LOG.exception("WAPR batch registration failed for %s", request["request_id"])
                batch_result = {"type": "batch_result", "version": 1,
                                "request_id": request["request_id"],
                                "observation_id": request["observation_id"],
                                "algorithm": algorithm, "score_type": "wapr_score_6d",
                                "results": []}
                returned = {instance_id: {"instance_id": instance_id, "status": "failed",
                                          "reason": str(exc)} for instance_id in records}

            elapsed_ms = (time.perf_counter() - started) * 1000
            archived_results = []
            for instance in request["instances"]:
                instance_id = instance["instance_id"]
                instance_request, record, staging, created = records[instance_id]
                item = dict(returned.get(instance_id, {
                    "instance_id": instance_id, "status": "failed",
                    "reason": "WAPR omitted this instance from the batch result",
                }))
                score_preview_path = item.pop("_history_score_preview_path", None)
                score_preview_available = bool(score_preview_path and Path(score_preview_path).is_file())
                item["request_id"] = instance_request["request_id"]

                if not created and record["archive_status"] == "complete":
                    item.update(history_id=record["history_id"], history_path=record["history_path"],
                                archive_status="complete",
                                score_preview_available=record["preview_status"] == "complete")
                else:
                    screening = dict(item.get("screening") or {})
                    if item.get("status") == "ok":
                        result = {
                            "type": "result", "version": 1,
                            "request_id": instance_request["request_id"],
                            "observation_id": request["observation_id"],
                            "object_id": request["object_id"],
                            "model_sha256": request["model_sha256"],
                            "algorithm": algorithm, "score_type": "wapr_score_6d",
                            "T_camera_object": item["pose_camera_object"],
                            "score": item["score"], "screening": screening,
                            "score_preview_available": score_preview_available,
                        }
                    else:
                        result = {"type": "error", "version": 1,
                                  "request_id": instance_request["request_id"],
                                  "observation_id": request["observation_id"],
                                  "code": "inference_failed",
                                  "message": item.get("reason", "WAPR batch inference failed"),
                                  "screening": screening}
                    try:
                        if score_preview_available:
                            shutil.copyfile(score_preview_path, staging / "debug/vis_score.png")
                        archived = self.store.publish(record, result, elapsed_ms=elapsed_ms)
                        published_record = self.store.get(record["history_id"])
                        item.update(history_id=archived["history_id"], history_path=archived["history_path"],
                                    archive_status=archived["archive_status"],
                                    score_preview_available=published_record["preview_status"] == "complete")
                    except Exception as exc:
                        self.store.archive_failure(record, exc)
                        item.update(history_id=record["history_id"], history_path=record["history_path"],
                                    archive_status="failed", archive_error=str(exc),
                                    score_preview_available=False)

                item["screening"] = {key: value for key, value in dict(item.get("screening") or {}).items()
                                     if key != "overlay_png"}
                archived_results.append(item)

            batch_result["results"] = archived_results
            batch_result["instance_count"] = len(archived_results)
            return batch_result
