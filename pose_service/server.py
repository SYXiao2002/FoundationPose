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
from .runtime import FoundationRuntime

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
                                    "object_id": self.runtime.object_id,
                                    "model_sha256": self.runtime.model_sha256}))
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
                        validate_request(request, self.runtime.object_id, self.runtime.model_sha256)
                        if hasattr(self.runtime, "replay"):
                            replay = await asyncio.to_thread(self.runtime.replay, request)
                            if replay is not None:
                                await websocket.send(packb(replay))
                                continue
                        if pending is not None or self.busy:
                            raise ValueError("inference busy")
                        self.busy = True
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
        runtime = await loop.run_in_executor(executor, lambda: FoundationRuntime(
            args.mesh, iterations=args.iterations,
            screening={"min_valid_pixels": args.min_valid_pixels,
                       "min_iou": args.min_iou, "max_median_m": args.max_median_m}))
        reference = Path(__file__).resolve().parent.parent
        provenance = {"mesh_name": Path(args.mesh).name,
                      "code": {str(p.relative_to(reference)): digest_file(p)
                               for p in [reference / "estimater.py", reference / "learning/training/predict_score.py"]},
                      "weights": {str(p.relative_to(reference)): digest_file(p)
                                  for p in (reference / "weights").rglob("*.pth")}}
        asset_id = await asyncio.to_thread(store.snapshot_model, runtime.mesh, runtime.model_sha256, provenance)
        recorded = RecordingRuntime(runtime, store, asset_id,
                                    {"debug": 3, "iterations": args.iterations, "screening": runtime.screening, "provenance": provenance})
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
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--history-root", default=os.environ.get("POSE6D_HISTORY_ROOT", "/data/pose_history"))
    parser.add_argument("--http-port", type=int, default=int(os.environ.get("POSE6D_HISTORY_HTTP_PORT", "8095")))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--min-valid-pixels", type=int, default=100)
    parser.add_argument("--min-iou", type=float, default=None)
    parser.add_argument("--max-median-m", type=float, default=None)
    args = parser.parse_args()
    if args.timeout <= 0 or args.iterations < 1 or args.min_valid_pixels < 1:
        parser.error("timeout, iterations and min-valid-pixels must be positive")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
