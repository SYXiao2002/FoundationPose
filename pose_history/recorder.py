"""Archive the actual GPU worker's result even after its client disconnects."""
import logging
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

    def model_manifest(self):
        if hasattr(self.runtime, "model_manifest"):
            return self.runtime.model_manifest()
        return {self.object_id: {"model_sha256": self.model_sha256}}

    def validate_request(self, request):
        if hasattr(self.runtime, "validate_request"):
            return self.runtime.validate_request(request)
        from pose_service.protocol import validate_request
        return validate_request(request, self.object_id, self.model_sha256)

    def replay(self, request):
        return self.store.replay(request)

    def infer(self, request):
        started = time.perf_counter()
        asset_id = (self.runtime.model_asset_id(request["object_id"])
                    if hasattr(self.runtime, "model_asset_id") else self.model_asset_id)
        parameters = {**self.parameters, "object_id": request["object_id"]}
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
