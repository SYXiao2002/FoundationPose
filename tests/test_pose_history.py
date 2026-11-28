"""Durability, image identity, geometry and HTTP behavior without a GPU."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from io import BytesIO
from pathlib import Path
import threading
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
import numpy as np
from PIL import Image
import pytest
import trimesh
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from pose_history.http import create_app
from pose_history.recorder import RecordingRuntime
from pose_history.preview import score_candidates_preview
from pose_history.store import HistoryStore, ArchiveCorruptError
from pose_history.viewer_data import HistoryViewer
from pose_service.server import PoseService
from pose_service.wire import packb, unpackb


@pytest.fixture
def request_data():
    return {"type": "register", "version": 1, "request_id": "r1", "observation_id": "o1",
            "object_id": "plug", "model_sha256": "hash", "rgb": np.arange(27, dtype=np.uint8).reshape(3, 3, 3),
            "depth_m": np.array([[1.000123, 1, 2], [1, 1, 1], [0, 1, .5]], np.float32),
            "mask": np.ones((3, 3), np.uint8), "K": np.array([[2, 0, 1], [0, 4, 1], [0, 0, 1.]]),
            "capture_timestamp_ns": "1794470462051315677", "instance_id": "sam-257"}


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "history")


def model_asset(store):
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [.01, 0, 0], [0, .01, 0]], faces=[[0, 1, 2]], process=False)
    return store.snapshot_model(mesh, "hash")


def response(request):
    pose = np.eye(4); pose[:3, 3] = [.1, -.2, .5]
    return {"type": "result", "version": 1, "request_id": request["request_id"],
            "observation_id": request["observation_id"], "object_id": "plug", "model_sha256": "hash",
            "T_camera_object": pose.tolist(), "score": 123., "screening": {"status": "suspect", "metrics": {}, "warning": "test"}}


def finish(store, request):
    record, created = store.begin(request, model_asset(store))
    path = store.save_inputs(record, request)
    image = np.arange(210 * 409 * 3, dtype=np.uint8).reshape(210, 409, 3)
    Image.fromarray(image).save(path / "debug/vis_score.png")
    store.publish(record, response(request))
    return store.get(record["history_id"]), image


def test_timestamps_unique_across_concurrent_calls_clock_rollback_and_restart(store, request_data):
    def begin(index):
        return store.begin({**request_data, "request_id": f"r{index}"}, None)[0]["history_id"]
    with patch('pose_history.store.time.time_ns', return_value=1794873600000000000):
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(begin, range(16)))
    assert len(set(ids)) == 16
    restarted = HistoryStore(store.root, recover=False)
    with patch('pose_history.store.time.time_ns', return_value=100):
        record, _ = restarted.begin({**request_data, "request_id": "after-restart"}, None)
    assert int(record["history_id"]) == max(map(int, ids)) + 1
    assert record["received_at_ns"] == '100'
    assert isinstance(record["capture_timestamp_ns"], str)


def test_debug_level_is_recorded_from_runtime_parameters(store, request_data):
    record, created = store.begin({**request_data, "request_id": "debug3"}, model_asset(store), {"debug": 3})
    assert created
    assert record["debug"] == 3
    assert record["parameters"]["debug"] == 3


def test_history_manifest_preserves_algorithm_score_semantics(store, request_data):
    record, _ = store.begin(request_data, model_asset(store), {"algorithm": "wapr"})
    path = store.save_inputs(record, request_data)
    Image.new("RGB", (640, 100), color=(16, 24, 32)).save(path / "debug/vis_score.png")
    result = response(request_data)
    result["score_type"] = "wapr_score_6d"
    store.publish(record, result)
    manifest = json.loads((store.record_dir(store.get(record["history_id"])) / "manifest.json").read_text())
    assert manifest["score_type"] == "wapr_score_6d"


@pytest.mark.parametrize("count", [1, 2, 30, 35])
def test_score_candidates_preview_keeps_complete_sorted_rows_and_caps_at_thirty(tmp_path, count):
    image = np.arange(count * 105 * 409 * 3, dtype=np.uint8).reshape(count * 105, 409, 3)
    source = tmp_path / "vis_score.png"
    Image.fromarray(image).save(source)
    encoded = score_candidates_preview(source)
    with Image.open(BytesIO(encoded)) as preview:
        height = min(count, 30) * 105 - 5
        assert preview.size == (409, height)
        np.testing.assert_array_equal(np.asarray(preview), image[:height])


def test_score_candidates_preview_rejects_incomplete_first_row(tmp_path):
    source = tmp_path / "vis_score.png"
    Image.new("RGB", (409, 99)).save(source)
    with pytest.raises(ValueError, match="shorter than one row"):
        score_candidates_preview(source)


def test_score_candidates_preview_can_limit_history_detail_to_ten_rows(tmp_path):
    count = 30
    image = np.arange(count * 105 * 409 * 3, dtype=np.uint8).reshape(count * 105, 409, 3)
    source = tmp_path / "vis_score.png"
    Image.fromarray(image).save(source)
    encoded = score_candidates_preview(source, top_k=10)
    with Image.open(BytesIO(encoded)) as preview:
        assert preview.size == (409, 1045)
        np.testing.assert_array_equal(np.asarray(preview), image[:1045])


def test_top30_http_preview_supports_existing_archives_without_rewriting_them(store, request_data):
    record, image = finish(store, request_data)
    root = store.record_dir(record)
    manifest_before = (root / "manifest.json").read_bytes()
    source_before = (root / "debug/vis_score.png").read_bytes()

    async def exercise():
        async with TestClient(TestServer(create_app(store))) as client:
            url = '/6d_pose_history/api/records/' + record['history_id'] + '/preview/'
            result = await client.get(url + 'score-top30')
            assert result.status == 200 and result.content_type == 'image/png'
            with Image.open(BytesIO(await result.read())) as preview:
                assert preview.size == (409, 205)
                np.testing.assert_array_equal(np.asarray(preview), image[:205])
            result = await client.get(url + 'score-top10')
            assert result.status == 200 and result.content_type == 'image/png'
            with Image.open(BytesIO(await result.read())) as preview:
                # This fixture has only two rows; history detail caps at ten.
                assert preview.size == (409, 205)
            # The established highest-score preview retains exactly one row.
            result = await client.get(url + 'score')
            with Image.open(BytesIO(await result.read())) as preview:
                np.testing.assert_array_equal(np.asarray(preview), image[:100])
    asyncio.run(exercise())
    assert (root / "manifest.json").read_bytes() == manifest_before
    assert (root / "debug/vis_score.png").read_bytes() == source_before


def test_top30_http_preview_handles_records_without_scorer_images(store, request_data):
    record, _ = store.begin(request_data, model_asset(store))
    store.save_inputs(record, request_data)
    store.publish(record, response(request_data))

    async def exercise():
        async with TestClient(TestServer(create_app(store))) as client:
            url = '/6d_pose_history/api/records/' + record['history_id'] + '/preview/score-top30'
            assert (await client.get(url)).status == 404
    asyncio.run(exercise())


def test_previews_inputs_replay_and_snapshot_independence(store, request_data):
    record, image = finish(store, request_data)
    np.testing.assert_array_equal(np.asarray(Image.open(store.artifact(record, "score"))), image[:100])
    assert Image.open(store.artifact(record, "score")).size == (409, 100)
    np.testing.assert_array_equal(np.asarray(Image.open(store.artifact(record, "RGB.png"))), request_data["rgb"])
    np.testing.assert_array_equal(np.load(store.artifact(record, "depth_m.npy")), request_data["depth_m"])
    assert store.begin(request_data, None)[1] is False
    assert store.replay(request_data)["T_camera_object"] == response(request_data)["T_camera_object"]
    with pytest.raises(ValueError, match="different input"):
        store.lookup({**request_data, "K": np.eye(3)})
    data = HistoryViewer(store, record)
    assert data.info()["instances"][0]["label"] == 'sam-257'
    assert data.info()["instances"][0]["origin"] == [.1, -.2, .5]
    assert data.model()["vertices"][1] == [.01, 0, 0]
    binary = data.cloud(2)
    count = int(np.frombuffer(binary, '<u4', count=1)[0]); assert count == 3
    points = np.frombuffer(binary, '<f4', count=count * 3, offset=4).reshape(-1, 3)
    np.testing.assert_allclose(points, [[-.5000615, -.25003075, 1.000123], [1, -.5, 2], [.25, .125, .5]], rtol=1e-6)
    colors = np.frombuffer(binary, np.uint8, offset=4 + count * 12).reshape(-1, 3)
    np.testing.assert_array_equal(colors, request_data["rgb"][[0, 0, 2], [0, 2, 2]])
    # A damaged input must be detected even if a point cloud cache exists.
    store.artifact(record, "RGB.png").write_bytes(b'corrupt')
    with pytest.raises(ArchiveCorruptError, match="corrupt"):
        data.cloud(2)


def test_recover_crash_before_and_after_rename(store, request_data):
    record, _ = store.begin(request_data, model_asset(store))
    store.save_inputs(record, request_data)
    restarted = HistoryStore(store.root)
    interrupted = restarted.get(record["history_id"])
    assert interrupted["status"] == "interrupted"
    assert restarted.artifact(interrupted, "RGB.png").exists()
    second, _ = restarted.begin({**request_data, "request_id": "r2"}, None)
    restarted.save_inputs(second, request_data)
    with patch.object(restarted, 'update', side_effect=OSError("simulated crash after rename")):
        with pytest.raises(OSError):
            restarted.publish(second, response({**request_data, "request_id": "r2"}))
    recovered = HistoryStore(store.root)
    assert recovered.get(second["history_id"])["status"] == "complete"
    assert recovered.replay({**request_data, "request_id": "r2"})["type"] == "result"


def test_failed_inference_and_failed_archive_have_distinct_states(store, request_data):
    class Runtime:
        object_id, model_sha256 = "plug", "hash"
        def infer(self, request, debug_dir):
            raise ValueError("no scored pose")
    recorded = RecordingRuntime(Runtime(), store, model_asset(store))
    result = recorded.infer(request_data)
    record = store.get(result['history_id'])
    assert record['status'] == 'failed'
    assert record['archive_status'] == 'complete'
    assert record['preview_status'] == 'missing'
    assert store.artifact(record, 'RGB.png').exists()
    with patch.object(store, 'publish', side_effect=OSError('disk full')):
        failed = recorded.infer({**request_data, 'request_id': 'r2'})
    assert failed['archive_status'] == 'failed'
    assert store.stats()['ready'] is False
    with pytest.raises(OSError, match='storage unavailable'):
        recorded.infer({**request_data, 'request_id': 'r3'})


def test_batch_inference_archives_each_instance_with_its_pose_and_score_preview(store, request_data, tmp_path):
    batch = {"type": "batch_register", "version": 1, "request_id": "batch-history",
             "observation_id": request_data["observation_id"], "object_id": request_data["object_id"],
             "model_sha256": request_data["model_sha256"], "rgb": request_data["rgb"],
             "depth_m": request_data["depth_m"], "K": request_data["K"],
             "capture_timestamp_ns": request_data["capture_timestamp_ns"],
             "capture_clock": "ros", "camera_frame": "camera",
             "instances": [{"instance_id": "sam-1", "mask": request_data["mask"]},
                           {"instance_id": "sam-2", "mask": np.fliplr(request_data["mask"]).copy()}]}

    class Runtime:
        object_id, model_sha256 = "plug", "hash"

        def validate_batch_request(self, value):
            assert value["type"] == "batch_register"

        def infer_batch(self, value, debug_dir):
            results = []
            for index, instance in enumerate(value["instances"]):
                preview = tmp_path / f"candidate-{index}.png"
                Image.new("RGB", (640, 100), color=(index, 24, 32)).save(preview)
                pose = np.eye(4)
                pose[0, 3] = index * .1
                results.append({"instance_id": instance["instance_id"], "status": "ok",
                                "pose_camera_object": pose.tolist(), "score": .4 + index * .1,
                                "screening": {"status": "pass_screening", "metrics": {"overlap_pixels": 4}},
                                "_history_score_preview_path": str(preview)})
            return {"type": "batch_result", "request_id": value["request_id"],
                    "observation_id": value["observation_id"], "algorithm": "wapr",
                    "results": results, "timings": {"registration_ms": 20.}}

    recorded = RecordingRuntime(Runtime(), store, model_asset(store))
    result = recorded.infer_batch(batch)
    assert result["instance_count"] == 2
    assert all(item["archive_status"] == "complete" for item in result["results"])
    assert all("_history_score_preview_path" not in item for item in result["results"])

    records = store.list_records(limit=10)["records"]
    assert len(records) == 2
    by_instance = {record["instance_id"]: record for record in records}
    for index, instance_id in enumerate(("sam-1", "sam-2")):
        record = by_instance[instance_id]
        assert record["parameters"]["batch_request_id"] == "batch-history"
        assert record["parameters"]["batch_index"] == index
        archived = json.loads(store.artifact(record, "result.json").read_text())
        assert archived["score_type"] == "wapr_score_6d"
        assert archived["score"] == pytest.approx(.4 + index * .1)
        assert store.artifact(record, "vis_score.png").is_file()
        view = HistoryViewer(store, record).info()
        assert len(view["instances"]) == 1
        assert view["instances"][0]["label"] == instance_id
        assert view["instances"][0]["pose"][0][3] == pytest.approx(index * .1)


def test_http_filters_assets_detail_and_artifact_bounds(store, request_data):
    record, _ = finish(store, request_data)
    async def exercise():
        async with TestClient(TestServer(create_app(store))) as client:
            root = '/6d_pose_history'
            listing = await (await client.get(root + '/api/records')).json()
            assert listing['records'][0]['history_id'] == record['history_id']
            assert isinstance(listing['records'][0]['history_time_ns'], str)
            assert (await client.get(root + '/api/records?limit=101')).status == 400
            assert (await client.get(root + '/api/records?before=nan')).status == 400
            assert not (await (await client.get(root + '/api/records?object_id=other')).json())['records']
            assert (await client.get(root + '/api/records?from=' + record['history_id'])).status == 200
            url = root + '/api/records/' + record['history_id']
            assert (await client.get(url + '/files/history.sqlite3')).status == 404
            assert (await client.get(url + '/viewer/scene?stride=0')).status == 400
            assert len(await (await client.get(url + '/viewer/scene')).read()) == 49
            assert (await (await client.get(url + '/viewer/info')).json())['instances'][0]['pose'] == response(request_data)['T_camera_object']
            asset = await client.get(root + '/assets/viewer.js')
            assert asset.status == 200
            assert asset.headers['Cache-Control'] == 'no-cache'
            page = await (await client.get(record['history_path'])).text()
            assert 'POSE_VIEWER_BASE' in page
            assert 'id="show-model" type="checkbox" checked' in page
            # Existing browser caches must not mix old CSS/JS with the new page.
            import hashlib
            from pose_history.http import ASSET_FILES, ROOT
            for html, names in [(page, ['history.css', 'review.js', 'detail.js', 'viewer.css', 'viewer.js']),
                                (await (await client.get(root + '/')).text(), ['history.css', 'review.js', 'history.js'])]:
                for name in names:
                    version = hashlib.sha256((ROOT / ASSET_FILES[name]).read_bytes()).hexdigest()[:16]
                    assert f'/assets/{name}?v={version}"' in html
                    assert f'/assets/{name}"' not in html
                    assert (await client.get(root + f'/assets/{name}?v={version}')).status == 200
            assert (await client.get(root + '/assets/unknown')).status == 404
            assert (await client.get('/healthz')).status == 200
    asyncio.run(exercise())


def test_manual_review_is_saved_to_a_per_record_file_and_survives_restart(store, request_data):
    record, _ = finish(store, request_data)
    review_file = store.reviews / f'{record["history_id"]}.json'
    assert not review_file.exists()

    async def exercise():
        async with TestClient(TestServer(create_app(store))) as client:
            root = '/6d_pose_history/api/records/' + record['history_id']
            listing = await (await client.get('/6d_pose_history/api/records')).json()
            assert listing['records'][0]['review_status'] == 'unreviewed'
            assert listing['records'][0]['review_updated_at_ns'] is None

            response = await client.post(root + '/review', json={'status': 'correct'})
            assert response.status == 200
            saved = await response.json()
            assert saved['review_status'] == 'correct'
            assert saved['review_updated_at_ns'].isdigit()
            assert json.loads(review_file.read_text()) == {
                'history_id': record['history_id'], 'review_status': 'correct',
                'review_updated_at_ns': saved['review_updated_at_ns'],
            }
            detail = await (await client.get(root)).json()
            assert detail['review_status'] == 'correct'

            listing = await (await client.get('/6d_pose_history/api/records')).json()
            assert listing['records'][0]['review_status'] == 'correct'
            incorrect = await (await client.post(root + '/review', json={'status': 'incorrect'})).json()
            assert incorrect['review_status'] == 'incorrect'
            cleared = await (await client.post(root + '/review', json={'status': 'unreviewed'})).json()
            assert cleared['review_status'] == 'unreviewed'
            assert cleared['review_updated_at_ns'] is None
            assert json.loads(review_file.read_text())['review_status'] == 'unreviewed'
            await client.post(root + '/review', json={'status': 'correct'})
            assert (await client.post(root + '/review', json={'status': 'unsure'})).status == 400
            assert (await client.post(root + '/review', json={'status': []})).status == 400
            assert (await client.post('/6d_pose_history/api/records/999/review',
                                      json={'status': 'incorrect'})).status == 404

    asyncio.run(exercise())
    restarted = HistoryStore(store.root, recover=False)
    assert restarted.get(record['history_id'])['review_status'] == 'correct'


def test_client_disconnect_keeps_worker_and_archive_alive(store, request_data):
    release = threading.Event(); started = threading.Event()
    class Runtime:
        object_id, model_sha256 = 'plug', 'hash'
        def infer(self, request, debug_dir):
            started.set(); release.wait(3)
            return response(request)
    async def exercise():
        recorded = RecordingRuntime(Runtime(), store, model_asset(store))
        with ThreadPoolExecutor(max_workers=1) as executor:
            service = PoseService(recorded, executor, .05)
            async with serve(service.handle, '127.0.0.1', 0) as server:
                uri = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'
                async with connect(uri) as ws:
                    await ws.recv(); await ws.send(packb(request_data))
                    assert unpackb(await ws.recv())['code'] == 'timeout'
                    assert started.is_set() and service.busy
                release.set()
                for _ in range(200):
                    if not service.busy: break
                    await asyncio.sleep(.01)
                assert not service.busy
                assert store.replay(request_data)['type'] == 'result'
                async with connect(uri) as ws:
                    await ws.recv(); await ws.send(packb(request_data))
                    assert unpackb(await ws.recv())['history_id'] == store.lookup(request_data)['history_id']
    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_wire_compatible_with_realman_client_envelope(request_data):
    # Check the established ndarray envelope, rather than importing the robot repo.
    import msgpack
    packed = packb(request_data)
    envelope = msgpack.unpackb(packed, raw=False)
    assert envelope['rgb'] == {'__ndarray__': True, 'dtype': '|u1', 'shape': [3, 3, 3], 'data': request_data['rgb'].tobytes()}
    round_trip = unpackb(msgpack.packb(envelope, use_bin_type=True))
    np.testing.assert_array_equal(round_trip['depth_m'], request_data['depth_m'])
    assert not round_trip['rgb'].flags.writeable


def test_partial_preview_failure_does_not_lose_pose(store, request_data):
    record, _ = store.begin(request_data, model_asset(store))
    path = store.save_inputs(record, request_data)
    (path / 'debug/vis_score.png').write_bytes(b'broken png')
    store.publish(record, response(request_data))
    saved = store.get(record['history_id'])
    assert saved['status'] == 'complete' and saved['preview_status'] == 'error'
    assert store.replay(request_data)['T_camera_object'] == response(request_data)['T_camera_object']
