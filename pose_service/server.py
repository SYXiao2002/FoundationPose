"""Single-GPU, bounded WebSocket registration service."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import fcntl
import signal

from .wire import MAX_MESSAGE_BYTES, packb, unpackb

from .protocol import validate_request
from .runtime import FoundationRuntimeRegistry

LOG = logging.getLogger("pose6d")


class PoseService:
    def __init__(self, runtime, executor, timeout):
        self.runtime = runtime
        self.executor = executor
        self.timeout = timeout
        self.busy = False

    async def handle(self, websocket, path=None):
        from websockets.exceptions import ConnectionClosed

        await websocket.send(packb({"type": "hello", "version": 1, "ready": True,
                                    "service": "pose6d",
                                    "object_id": self.runtime.object_id,
                                    "model_sha256": self.runtime.model_sha256,
                                    "algorithms": list(getattr(self.runtime, "algorithms", ("foundationpose",))),
                                    "default_algorithm": getattr(self.runtime, "default_algorithm", "foundationpose"),
                                    "default_object_id": getattr(self.runtime, "default_object_id", self.runtime.object_id),
                                    "models": self.runtime.model_manifest()
                                    if hasattr(self.runtime, "model_manifest") else {
                                        self.runtime.object_id: {"model_sha256": self.runtime.model_sha256}}}))
        loop = asyncio.get_running_loop()
        receive = asyncio.create_task(websocket.recv())
        pending, current, deadline = None, None, None
        try:
            while True:
                watch = {receive}
                if pending is not None:
                    watch.add(pending)
                timeout = max(0, deadline - loop.time()) if deadline is not None else None
                done, _ = await asyncio.wait(watch, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    await websocket.send(packb(self.error(current, "timeout", "inference timed out; GPU remains busy until completion")))
                    pending, current, deadline = None, None, None
                    continue
                if receive in done:
                    raw = receive.result()
                    receive = asyncio.create_task(websocket.recv())
                    request = None
                    try:
                        request = unpackb(raw)
                        if isinstance(request, dict) and request.get("type") == "cancel":
                            if pending is None or request.get("request_id") != current["request_id"]:
                                raise ValueError("no matching request to cancel")
                            await websocket.send(packb(self.error(current, "canceled", "result discarded; GPU remains busy until completion")))
                            pending, current, deadline = None, None, None
                            continue
                        is_batch = isinstance(request, dict) and request.get("type") == "batch_register"
                        if is_batch:
                            if hasattr(self.runtime, "validate_batch_request"):
                                self.runtime.validate_batch_request(request)
                            else:
                                raise ValueError("batch inference is not supported")
                        else:
                            if hasattr(self.runtime, "validate_request"):
                                self.runtime.validate_request(request)
                            else:
                                validate_request(request, self.runtime.object_id, self.runtime.model_sha256)
                        if not is_batch and hasattr(self.runtime, "replay"):
                            replay = await asyncio.to_thread(self.runtime.replay, request)
                            if replay is not None:
                                await websocket.send(packb(replay))
                                continue
                        if pending is not None or self.busy:
                            raise ValueError("inference busy")
                        self.busy = True
                        if is_batch:
                            pending = loop.run_in_executor(self.executor, self.runtime.infer_batch, request)
                        else:
                            pending = loop.run_in_executor(self.executor, self.runtime.infer, request)
                        pending.add_done_callback(self._worker_done)
                        current, deadline = request, loop.time() + self.timeout
                    except (ValueError, TypeError, KeyError, OSError) as exc:
                        await websocket.send(packb(self.error(request, "invalid_request", str(exc))))
                if pending is not None and pending in done:
                    try:
                        response = pending.result()
                    except Exception as exc:
                        LOG.exception("registration failed")
                        response = self.error(current, "inference_failed", str(exc))
                    pending, current, deadline = None, None, None
                    await websocket.send(packb(response))
        except ConnectionClosed:
            pass
        finally:
            receive.cancel()

    def _worker_done(self, worker):
        self.busy = False
        if not worker.cancelled():
            worker.exception()

    @staticmethod
    def error(request, code, message):
        request = request if isinstance(request, dict) else {}
        return {"type": "error", "version": 1, "request_id": request.get("request_id"),
                "observation_id": request.get("observation_id"), "code": code, "message": message}


async def main_async(args):
    from websockets.asyncio.server import serve
    from aiohttp import web
    from pose_history.store import HistoryStore, digest_file
    from pose_history.recorder import RecordingRuntime
    from pose_history.http import create_app

    root = Path(args.history_root)
    root.mkdir(parents=True, exist_ok=True)
    # Only the server owning this lock may recover unfinished GPU work.
    with (root / "server.lock").open("a") as lock, ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose6d-gpu") as executor:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        loop = asyncio.get_running_loop()
        store = HistoryStore(root)
        model_specs = parse_model_specs(args.model, args.mesh, args.object_id)
        algorithms = parse_algorithms(args.algorithms)
        if "wapr" in algorithms:
            from wapr.download_assets import check_and_fetch_pack

            await asyncio.to_thread(check_and_fetch_pack, "wapr_sapr_wbps")
        reference = Path(__file__).resolve().parent.parent
        provenance = {"mesh_name": Path(args.mesh).name if args.mesh else None,
                      "algorithms": list(algorithms),
                      "code": {str(p.relative_to(reference)): digest_file(p)
                               for p in [reference / "Utils.py", reference / "estimater.py",
                                         reference / "learning/training/predict_score.py",
                                         reference / "pose_service/runtime.py", reference / "pose_service/runtime_wapr.py",
                                         reference / "pose_service/protocol.py", reference / "pose_service/screening.py"]},
                      "weights": {str(p.relative_to(reference)): digest_file(p)
                                  for p in (reference / "weights").rglob("*.pth")}}
        if "wapr" in algorithms:
            import wapr
            from wapr.resources import weights_dir

            wapr_root = Path(wapr.__file__).resolve().parent
            wapr_files = [path for path in wapr_root.rglob("*") if path.is_file()
                          and path.suffix in {".py", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".cmake"}]
            wapr_weights = Path(weights_dir())
            provenance["wapr"] = {
                "backend": args.wapr_backend,
                "code": {str(p.relative_to(wapr_root.parent)): digest_file(p) for p in wapr_files if p.is_file()},
                "weights": {p.name: digest_file(p) for p in sorted((*wapr_weights.glob("*.pth"),
                                                                      *wapr_weights.glob("*.engine")))},
            }
        screening = {"min_valid_pixels": args.min_valid_pixels,
                     "min_iou": args.min_iou, "max_median_m": args.max_median_m}
        runtime = FoundationRuntimeRegistry(model_specs, iterations=args.iterations, screening=screening,
                                            algorithms=algorithms, default_algorithm=args.default_algorithm,
                                            wapr_backend=args.wapr_backend)
        await asyncio.to_thread(runtime.prepare_archives, store, provenance)
        await loop.run_in_executor(executor, runtime._load, runtime.default_algorithm,
                                   runtime.default_object_id)
        # Load WAPR once before opening the socket so missing dependencies,
        # weights, or incompatible engines fail at service startup. The
        # registry releases it when restoring the configured default runtime.
        if "wapr" in algorithms and runtime.default_algorithm != "wapr":
            await loop.run_in_executor(executor, runtime._load, "wapr", runtime.default_object_id)
            await loop.run_in_executor(executor, runtime._load, runtime.default_algorithm,
                                       runtime.default_object_id)
        recorded = RecordingRuntime(runtime, store, None,
                                    {"debug": 3, "iterations": args.iterations,
                                     "screening": screening, "provenance": provenance,
                                     "algorithms": list(algorithms), "wapr_backend": args.wapr_backend})
        service = PoseService(recorded, executor, args.timeout)
        app = create_app(store, service=service)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            await web.TCPSite(runner, args.host, args.http_port).start()
            async with serve(service.handle, args.host, args.port, max_size=MAX_MESSAGE_BYTES):
                LOG.info("pose6d ready WS=%s:%s history HTTP=%s, debug=3", args.host, args.port, args.http_port)
                stop = asyncio.Event()
                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(sig, stop.set)
                await stop.wait()
        finally:
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", help="single-model compatibility option")
    parser.add_argument("--object-id", default="eu_plug", help="object ID paired with --mesh")
    parser.add_argument("--model", action="append", default=[], metavar="ID=PATH",
                        help="register a model; may be repeated")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--history-root", default=os.environ.get("POSE6D_HISTORY_ROOT", "/data/pose_history"))
    parser.add_argument("--http-port", type=int, default=int(os.environ.get("POSE6D_HISTORY_HTTP_PORT", "8095")))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--algorithms", default="foundationpose",
                        help="comma-separated enabled algorithms: foundationpose,wapr")
    parser.add_argument("--default-algorithm", choices=["foundationpose", "wapr"], default="foundationpose")
    parser.add_argument("--wapr-backend", choices=["torch", "trt"], default=os.environ.get("WAPR_BACKEND", "torch"))
    parser.add_argument("--min-valid-pixels", type=int, default=100)
    parser.add_argument("--min-iou", type=float, default=None)
    parser.add_argument("--max-median-m", type=float, default=None)
    args = parser.parse_args()
    if args.timeout <= 0 or args.iterations < 1 or args.min_valid_pixels < 1:
        parser.error("timeout, iterations and min-valid-pixels must be positive")
    try:
        parse_model_specs(args.model, args.mesh, args.object_id)
        algorithms = parse_algorithms(args.algorithms)
        if args.default_algorithm not in algorithms:
            raise ValueError("default algorithm must be enabled")
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main_async(args))


def parse_model_specs(specifications, mesh, object_id):
    models = {}
    for specification in specifications:
        model_id, separator, path = specification.partition("=")
        if not separator or not model_id or not path or model_id in models:
            raise ValueError("--model must be unique ID=PATH entries")
        models[model_id] = path
    if mesh:
        if models:
            raise ValueError("use either --mesh or repeated --model arguments")
        models[object_id] = mesh
    if not models:
        raise ValueError("provide --mesh or at least one --model ID=PATH")
    return models


def parse_algorithms(value):
    if not isinstance(value, str):
        raise ValueError("--algorithms must be a comma-separated string")
    algorithms = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not algorithms or set(algorithms) - {"foundationpose", "wapr"}:
        raise ValueError("--algorithms supports foundationpose and wapr")
    return algorithms


if __name__ == "__main__":
    main()
