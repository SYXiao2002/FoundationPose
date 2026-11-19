"""Read-only history app; file/CPU work never runs on the GPU executor."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import logging
from pathlib import Path

from aiohttp import web

from .store import HistoryStore, ArchiveCorruptError
from .preview import score_candidates_preview
from .viewer_data import HistoryViewer

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "/6d_pose_history"
LOG = logging.getLogger(__name__)
ASSET_FILES = {"history.js": "web/pose_history/history.js", "history.css": "web/pose_history/style.css",
               "detail.js": "web/pose_history/detail.js", "viewer.js": "web/multi_pose_viewer/viewer.js",
               "viewer.css": "web/multi_pose_viewer/style.css"}


def create_app(store, *, service=None):
    # A new HTML page must never reuse scripts/styles from an older deployment.
    asset_versions = {name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()[:16]
                      for name, path in ASSET_FILES.items()}

    def version_assets(source):
        for name, version in asset_versions.items():
            source = source.replace(f'{PREFIX}/assets/{name}"', f'{PREFIX}/assets/{name}?v={version}"')
        return source

    @web.middleware
    async def errors(request, handler):
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except FileNotFoundError as exc:
            return web.json_response({"message": str(exc)}, status=404)
        except ValueError as exc:
            return web.json_response({"message": str(exc)}, status=400)
        except (OSError, KeyError, ArchiveCorruptError) as exc:
            LOG.exception("history read failed")
            return web.json_response({"message": f"history unavailable: {exc}"}, status=500)

    app = web.Application(middlewares=[errors])
    # Bound both running and queued requests, separate from the GPU thread.
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="history-http")
    slots = asyncio.Semaphore(4)

    async def work(function, *args, **kwargs):
        try:
            await asyncio.wait_for(slots.acquire(), timeout=.5)
        except asyncio.TimeoutError:
            raise web.HTTPServiceUnavailable(text="history viewer busy; retry shortly")
        task = asyncio.get_running_loop().run_in_executor(pool, partial(function, *args, **kwargs))
        task.add_done_callback(lambda _: slots.release())
        return await asyncio.shield(task)

    async def close(_):
        pool.shutdown(wait=True, cancel_futures=True)

    app.on_cleanup.append(close)

    async def health(request):
        stats = await work(store.stats)
        stats.update(gpu_busy=bool(service and service.busy), debug=3)
        return web.json_response(stats, status=200 if stats["ready"] else 503)

    async def records(request):
        args = {k: request.query.get(v) for k, v in (("before", "before"), ("from_ns", "from"), ("to_ns", "to"),
                                                   ("object_id", "object_id"), ("status", "status"))}
        return web.json_response(await work(store.list_records, limit=int(request.query.get("limit", "30")), **args))

    async def record_for(request):
        return await work(store.get, request.match_info["id"])

    async def detail(request):
        record = await record_for(request)
        if record["archive_status"] == "complete":
            def read():
                path = store.artifact(record, "manifest.json")
                manifest = json.loads(path.read_text())
                manifest["result"] = json.loads(store.artifact(record, "result.json").read_text())
                return manifest
            record = await work(read)
        return web.json_response(record)

    async def artifact(request):
        record = await record_for(request)
        if record["archive_status"] == "writing":
            return web.json_response({"message": "record is still being generated"}, status=409)
        name = request.match_info.get("artifact", request.match_info.get("preview"))
        if name == "score-top30":
            def read_candidates():
                return score_candidates_preview(store.artifact(record, "vis_score.png"))
            payload = await work(read_candidates)
            return web.Response(body=payload, content_type="image/png",
                                headers={"Cache-Control": "private, max-age=3600",
                                         "X-Content-Type-Options": "nosniff"})
        path = await work(store.artifact, record, name)
        return web.FileResponse(path, headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})

    async def viewer(request):
        record = await record_for(request)
        if record["archive_status"] != "complete":
            return web.json_response({"message": f'archive {record["archive_status"]}'}, status=409)
        def render():
            data = HistoryViewer(store, record)
            name = request.match_info["kind"]
            if name == "scene":
                return data.cloud(int(request.query.get("stride", "2"))), "application/octet-stream"
            if name == "mask-preview":
                return data.mask_preview(), "image/jpeg"
            if name == "mask-map":
                return data.mask_map(), "image/png"
            if name in ("info", "model"):
                return json.dumps(getattr(data, name)(), allow_nan=False).encode(), "application/json"
            raise FileNotFoundError("unknown viewer resource")
        payload, mime = await work(render)
        return web.Response(body=payload, content_type=mime, headers={"Cache-Control": "private, max-age=3600"})

    async def assets(request):
        name = request.match_info["asset"]
        if name == "plotly.js":
            from plotly.offline import get_plotlyjs
            return web.Response(text=await work(get_plotlyjs), content_type="application/javascript",
                                headers={"Cache-Control": "public, max-age=86400"})
        if name not in ASSET_FILES:
            raise web.HTTPNotFound()
        return web.FileResponse(ROOT / ASSET_FILES[name], headers={"Cache-Control": "no-cache"})

    async def index(request):
        source = await work((ROOT / "web/pose_history/index.html").read_text)
        return web.Response(text=version_assets(source), content_type="text/html", headers={"Cache-Control": "no-store"})

    async def page(request):
        await record_for(request)
        source = await work((ROOT / "web/multi_pose_viewer/index.html").read_text)
        source = source.replace('href="/style.css"', f'href="{PREFIX}/assets/viewer.css"')
        source = source.replace('src="/plotly.js"', f'src="{PREFIX}/assets/plotly.js"')
        source = source.replace('src="/viewer.js"', f'src="{PREFIX}/assets/viewer.js"')
        source = source.replace('src="/api/mask.jpg"', '')
        source = source.replace('id="show-model" type="checkbox"', 'id="show-model" type="checkbox" checked')
        source = source.replace("<head>", '<head><script>window.POSE_VIEWER_BASE=' + json.dumps(
            f'{PREFIX}/api/records/{request.match_info["id"]}/viewer') + ';</script>'
            + f'<link rel="stylesheet" href="{PREFIX}/assets/history.css"><script defer src="{PREFIX}/assets/detail.js"></script>')
        source = source.replace('第一帧', '归档记录').replace('eu-plug 初始阈值 · 筛查结果需人工复核', '当次归档筛查结果')
        source = source.replace('<header class="topbar">', '<header class="topbar"><a href="/6d_pose_history/">← 历史</a>')
        source = source.replace('<body>', '<body class="history-detail">')
        source = source.replace('</main>', '''<aside id="history-score-candidates" class="history-score-candidates" aria-labelledby="score-candidates-title">
          <div class="score-candidates-heading"><h2 id="score-candidates-title">vis_score · 前 30 候选</h2>
          <p>按评分从高到低排列 · 图内标注候选 ID 与分数</p></div>
          <p id="score-candidates-status" role="status">等待评分候选图</p>
        </aside></main>''', 1)
        source = source.replace('<section class="instances-section"', '<section id="history-images" class="history-images"></section><section class="instances-section"')
        return web.Response(text=version_assets(source), content_type="text/html", headers={"Cache-Control": "no-store"})

    async def redirect(request):
        raise web.HTTPFound(PREFIX + "/")

    app.router.add_get("/healthz", health)
    app.router.add_get(PREFIX, redirect)
    app.router.add_get(PREFIX + "/", index)
    app.router.add_get(PREFIX + "/assets/{asset}", assets)
    app.router.add_get(PREFIX + "/api/records", records)
    app.router.add_get(PREFIX + "/api/records/{id:\\d+}", detail)
    app.router.add_get(PREFIX + "/api/records/{id:\\d+}/preview/{preview:rgb|score|score-top30}", artifact)
    app.router.add_get(PREFIX + "/api/records/{id:\\d+}/files/{artifact}", artifact)
    app.router.add_get(PREFIX + "/api/records/{id:\\d+}/viewer/{kind}", viewer)
    app.router.add_get(PREFIX + "/{id:\\d+}/", page)
    return app


def main():
    parser = argparse.ArgumentParser(description="Browse FoundationPose history without loading a GPU")
    parser.add_argument("--root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()
    web.run_app(create_app(HistoryStore(args.root, recover=False)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
