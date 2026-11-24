"""Durable timestamp-keyed history index and immutable artifact publication."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

import numpy as np
from PIL import Image

from .preview import rgb_preview, score_preview

class ArchiveCorruptError(RuntimeError):
    pass


LOG = logging.getLogger(__name__)
META_KEYS = ("capture_timestamp_ns", "capture_clock", "camera_frame", "instance_id", "mask_source", "source")
ARTIFACTS = {
    "RGB.png": "RGB.png", "depth_m.npy": "depth_m.npy", "mask.png": "mask.png",
    "cam_K.txt": "cam_K.txt", "pose.txt": "pose.txt", "result.json": "result.json",
    "manifest.json": "manifest.json", "vis_score.png": "debug/vis_score.png",
    "vis_refiner.png": "debug/vis_refiner.png", "scene_complete.ply": "debug/scene_complete.ply",
    "screening_overlay.png": "screening_overlay.png", "rgb": "preview/rgb_thumb.jpg",
    "score": "preview/vis_score_top1.png",
}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_digest(request):
    digest = hashlib.sha256()
    for name in ("object_id", "model_sha256", "observation_id"):
        digest.update(json_bytes([name, request[name]]))
    for name in ("rgb", "depth_m", "mask", "K"):
        dtype = {"rgb": "u1", "depth_m": "<f4", "mask": "u1", "K": "<f8"}[name]
        array = np.asarray(request[name], dtype=dtype, order="C")
        digest.update(json_bytes([name, array.shape, array.dtype.str]))
        digest.update(array.tobytes())
    return digest.hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_tree(root):
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        sync_dir(path)
    sync_dir(root)


class HistoryStore:
    def __init__(self, root, *, recover=True):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.error = None
        self.reviews = self.root / "reviews"
        self.reviews.mkdir(exist_ok=True)
        probe = self.root / (".write-check-" + uuid.uuid4().hex)
        probe.write_bytes(b"ok")
        probe.unlink()
        (self.root / "staging").mkdir(exist_ok=True)
        (self.root / "models").mkdir(exist_ok=True)
        with self.connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS high_water (id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL);
                INSERT OR IGNORE INTO high_water VALUES (1, 0);
                CREATE TABLE IF NOT EXISTS pose_history (
                    history_time_ns INTEGER PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
                    observation_id TEXT NOT NULL, input_sha256 TEXT NOT NULL,
                    object_id TEXT NOT NULL, model_sha256 TEXT NOT NULL,
                    model_asset_id TEXT, status TEXT NOT NULL, archive_status TEXT NOT NULL,
                    preview_status TEXT NOT NULL, score REAL, screening_status TEXT,
                    record_relpath TEXT NOT NULL, metadata TEXT NOT NULL, error_code TEXT
                );
                CREATE INDEX IF NOT EXISTS history_observation ON pose_history(observation_id, history_time_ns);
                CREATE INDEX IF NOT EXISTS history_object ON pose_history(object_id, history_time_ns);
                CREATE INDEX IF NOT EXISTS history_status ON pose_history(status, history_time_ns);
            """)
        if recover:
            self.recover()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.root / "history.sqlite3", timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def public(self, row):
        data = dict(row)
        data.update(json.loads(data.pop("metadata")))
        data["history_time_ns"] = str(data["history_time_ns"])
        data["history_id"] = data["history_time_ns"]
        data["review_status"] = "unreviewed"
        data["review_updated_at_ns"] = None
        review_path = self.reviews / f'{data["history_id"]}.json'
        if review_path.is_file():
            try:
                review = json.loads(review_path.read_text())
            except (OSError, ValueError):
                data["review_error"] = "人工判断文件不可读取，可重新选择后保存"
            else:
                updated_at = review.get("review_updated_at_ns") if isinstance(review, dict) else None
                valid_review = (isinstance(review, dict) and review.get("history_id") == data["history_id"]
                                and review.get("review_status") in {"unreviewed", "correct", "incorrect"}
                                and (updated_at is None or str(updated_at).isdigit()))
                if valid_review:
                    data["review_status"] = review["review_status"]
                    data["review_updated_at_ns"] = str(updated_at) if updated_at is not None else None
                else:
                    data["review_error"] = "人工判断文件格式无效，可重新选择后保存"
        data["history_path"] = f'/6d_pose_history/{data["history_id"]}/'
        return data

    def get(self, history_id):
        if not str(history_id).isdigit() or not 0 < int(history_id) < 2**63:
            raise ValueError("invalid history ID")
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM pose_history WHERE history_time_ns=?", (int(history_id),)).fetchone()
        if row is None:
            raise FileNotFoundError("history record does not exist")
        return self.public(row)

    def set_review(self, history_id, status):
        if not str(history_id).isdigit() or not 0 < int(history_id) < 2**63:
            raise ValueError("invalid history ID")
        if not isinstance(status, str) or status not in {"unreviewed", "correct", "incorrect"}:
            raise ValueError("review status must be unreviewed, correct, or incorrect")
        record = self.get(history_id)
        reviewed_at = time.time_ns() if status != "unreviewed" else None
        review = {"history_id": record["history_id"], "review_status": status,
                  "review_updated_at_ns": str(reviewed_at) if reviewed_at is not None else None}
        path = self.reviews / f'{record["history_id"]}.json'
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(json_bytes(review))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            sync_dir(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return self.get(history_id)

    def lookup(self, request):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM pose_history WHERE request_id=?", (request["request_id"],)).fetchone()
        if row is None:
            return None
        if row["input_sha256"] != input_digest(request):
            raise ValueError("request_id already belongs to different input")
        return self.public(row)

    def begin(self, request, model_asset_id, parameters=None):
        if self.error:
            raise OSError(f"history storage unavailable: {self.error}")
        digest = input_digest(request)
        received = time.time_ns()
        metadata = {key: request.get(key) for key in META_KEYS}
        if metadata["capture_timestamp_ns"] is not None:
            metadata["capture_timestamp_ns"] = str(metadata["capture_timestamp_ns"])
        metadata.update(received_at_ns=str(received), started_at_ns=str(received),
                        finished_at_ns=None, image_size=[request["rgb"].shape[1], request["rgb"].shape[0]],
                        schema_version=1, parameters=parameters or {}, debug=(parameters or {}).get("debug", 2))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM pose_history WHERE request_id=?", (request["request_id"],)).fetchone()
            if old:
                if old["input_sha256"] != digest:
                    raise ValueError("request_id already belongs to different input")
                return self.public(old), False
            last = conn.execute("SELECT value FROM high_water WHERE id=1").fetchone()[0]
            stamp = max(received, last + 1)
            day = datetime.fromtimestamp(stamp // 10**9, timezone.utc).strftime("%Y/%m/%d")
            relpath = f"records/{day}/{stamp}"
            conn.execute("UPDATE high_water SET value=? WHERE id=1", (stamp,))
            conn.execute("""INSERT INTO pose_history
                (history_time_ns,request_id,observation_id,input_sha256,object_id,model_sha256,model_asset_id,
                 status,archive_status,preview_status,record_relpath,metadata)
                VALUES (?,?,?,?,?,?,?,'running','writing','pending',?,?)""",
                (stamp, request["request_id"], request["observation_id"], digest, request["object_id"],
                 request["model_sha256"], model_asset_id, relpath, json.dumps(metadata)))
        return self.get(stamp), True

    def update(self, history_id, **fields):
        allowed = {"status", "archive_status", "preview_status", "score", "screening_status", "error_code", "metadata"}
        if not fields or set(fields) - allowed:
            raise ValueError("invalid update fields")
        with self.connect() as conn:
            conn.execute("UPDATE pose_history SET " + ",".join(f"{key}=?" for key in fields) + " WHERE history_time_ns=?",
                         (*fields.values(), int(history_id)))

    def staging(self, record):
        return self.root / "staging" / record["history_id"]

    def save_inputs(self, record, request):
        path = self.staging(record)
        path.mkdir()
        (path / "debug").mkdir()
        (path / "preview").mkdir()
        Image.fromarray(request["rgb"]).save(path / "RGB.png")
        Image.fromarray(request["mask"] * np.uint8(255)).save(path / "mask.png")
        np.save(path / "depth_m.npy", request["depth_m"], allow_pickle=False)
        np.savetxt(path / "cam_K.txt", request["K"], fmt="%.17g")
        rgb_preview(path / "RGB.png", path / "preview/rgb_thumb.jpg")
        return path

    def snapshot_model(self, mesh, identity, provenance=None):
        vertices = np.asarray(mesh.vertices, dtype="<f8")
        faces = np.asarray(mesh.faces, dtype="<i8")
        asset_id = hashlib.sha256(identity.encode() + vertices.tobytes() + faces.tobytes()).hexdigest()
        target = self.root / "models" / asset_id
        if target.exists():
            self.validate_directory(target)
            return asset_id
        temporary = target.with_name(".model-" + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            np.savez_compressed(temporary / "mesh.npz", vertices=vertices, faces=faces)
            # Geometry snapshot preserves the exact original mesh frame, independent of material files.
            mesh.export(temporary / "model.obj", include_texture=False)
            manifest = {"schema_version": 1, "model_sha256": identity, "unit": "m",
                        "coordinates": "original OBJ", "provenance": provenance or {},
                        "files": {p.name: digest_file(p) for p in temporary.iterdir()}}
            (temporary / "manifest.json").write_bytes(json_bytes(manifest))
            sync_tree(temporary)
            temporary.rename(target)
            sync_dir(target.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return asset_id

    def publish(self, record, response, *, elapsed_ms=None):
        path = self.staging(record)
        result = dict(response)
        result["screening"] = dict(result.get("screening") or {})
        overlay = result["screening"].pop("overlay_png", None)
        if overlay:
            (path / "screening_overlay.png").write_bytes(overlay)
        pose = result.get("T_camera_object")
        if pose is not None:
            np.savetxt(path / "pose.txt", pose, fmt="%.17g")
        preview_status = "missing"
        if (path / "debug/vis_score.png").exists():
            try:
                score_preview(path / "debug/vis_score.png", path / "preview/vis_score_top1.png")
                preview_status = "complete"
            except Exception:
                LOG.exception("could not create score preview for %s", record["history_id"])
                preview_status = "error"
        fields = {"status": "complete" if result.get("type") == "result" else "failed",
                  "archive_status": "complete", "preview_status": preview_status,
                  "score": result.get("score"), "screening_status": result["screening"].get("status"),
                  "error_code": result.get("code")}
        result.update(history_id=record["history_id"], history_path=record["history_path"], archive_status="complete")
        (path / "result.json").write_bytes(json_bytes(result))
        manifest = dict(record)
        manifest.update(fields, finished_at_ns=str(time.time_ns()), elapsed_ms=elapsed_ms,
                        depth_unit="m", pose_unit="m", coordinates="RGB optical camera; original OBJ",
                        score_type="raw scorer logit with pairwise selection offset")
        manifest["files"] = {str(p.relative_to(path)): digest_file(p) for p in path.rglob("*") if p.is_file()}
        manifest["bytes"] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        (path / "manifest.json").write_bytes(json_bytes(manifest))
        sync_tree(path)
        destination = self.root / record["record_relpath"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        sync_dir(destination.parent)
        metadata = {k: manifest.get(k) for k in (*META_KEYS, "received_at_ns", "started_at_ns", "finished_at_ns",
                    "image_size", "schema_version", "parameters", "debug", "elapsed_ms", "bytes")}
        self.update(record["history_id"], **fields, metadata=json.dumps(metadata))
        return result

    def archive_failure(self, record, exc):
        self.error = str(exc)
        LOG.exception("history archival failed: %s", record["history_id"])
        try:
            self.update(record["history_id"], archive_status="failed", error_code="archive_failed")
        except sqlite3.Error:
            LOG.exception("could not update history index")

    def record_dir(self, record):
        path = (self.root / record["record_relpath"]).resolve()
        if not path.is_relative_to(self.root / "records"):
            raise ValueError("invalid record location")
        return path

    def artifact(self, record, name):
        relative = ARTIFACTS.get(name)
        if relative is None:
            raise FileNotFoundError("unknown artifact")
        path = self.record_dir(record) / relative
        manifest_path = path.parent if name == "manifest.json" else self.record_dir(record)
        manifest = json.loads((manifest_path / "manifest.json").read_text())
        if not path.is_file():
            raise FileNotFoundError("artifact was not generated")
        if name != "manifest.json" and digest_file(path) != manifest["files"].get(relative):
            raise ArchiveCorruptError("artifact checksum mismatch")
        return path

    @staticmethod
    def validate_directory(path):
        manifest = json.loads((path / "manifest.json").read_text())
        for relative, digest in manifest["files"].items():
            candidate = (path / relative).resolve()
            if not candidate.is_relative_to(path.resolve()) or digest_file(candidate) != digest:
                raise ArchiveCorruptError(f"archive checksum mismatch: {relative}")
        return manifest

    def replay(self, request):
        record = self.lookup(request)
        if record is None:
            return None
        if record["archive_status"] != "complete":
            return {"type": "error", "version": 1, "request_id": request["request_id"],
                    "observation_id": request["observation_id"], "code": "history_unavailable",
                    "message": f'history {record["status"]}; archive {record["archive_status"]}',
                    "history_id": record["history_id"], "history_path": record["history_path"]}
        result = json.loads(self.artifact(record, "result.json").read_text())
        if "screening_overlay.png" in json.loads((self.record_dir(record) / "manifest.json").read_text())["files"]:
            result.setdefault("screening", {})["overlay_png"] = self.artifact(record, "screening_overlay.png").read_bytes()
        return result

    def list_records(self, *, limit=30, before=None, from_ns=None, to_ns=None, object_id=None, status=None):
        if not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        clauses, values = [], []
        for key, op, value in (("history_time_ns", "<", before), ("history_time_ns", ">=", from_ns),
                               ("history_time_ns", "<", to_ns), ("object_id", "=", object_id), ("status", "=", status)):
            if value is not None:
                if key == "history_time_ns":
                    if not str(value).isdigit() or not 0 < int(value) < 2**63:
                        raise ValueError("invalid timestamp filter")
                    value = int(value)
                clauses.append(f"{key} {op} ?")
                values.append(value)
        sql = "SELECT * FROM pose_history" + (" WHERE " + " AND ".join(clauses) if clauses else "")
        with self.connect() as conn:
            rows = conn.execute(sql + " ORDER BY history_time_ns DESC LIMIT ?", (*values, limit + 1)).fetchall()
        return {"records": [self.public(row) for row in rows[:limit]],
                "next_cursor": str(rows[limit - 1]["history_time_ns"]) if len(rows) > limit else None}

    def recover(self):
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM pose_history WHERE status='running'").fetchall()
        for row in rows:
            record = self.public(row)
            directory = self.record_dir(record)
            if directory.exists():
                try:
                    manifest = self.validate_directory(directory)
                    fields = {k: manifest.get(k) for k in ("status", "archive_status", "preview_status", "score", "screening_status", "error_code")}
                    meta = json.loads(row["metadata"])
                    meta.update({k: manifest.get(k) for k in ("finished_at_ns", "elapsed_ms", "bytes")})
                    self.update(record["history_id"], **fields, metadata=json.dumps(meta))
                    continue
                except (OSError, ValueError, KeyError, ArchiveCorruptError):
                    LOG.exception("incomplete published history: %s", record["history_id"])
            # Preserve usable inputs from a worker that died before publication.
            stage = self.staging(record)
            try:
                if stage.exists():
                    self.publish(record, {"type": "error", "version": 1, "request_id": record["request_id"],
                        "observation_id": record["observation_id"], "code": "interrupted", "message": "service restarted before completion"})
                self.update(record["history_id"], status="interrupted", error_code="interrupted")
            except Exception as exc:
                self.archive_failure(record, exc)
        # Keep unknown staging directories for inspection rather than deleting data.
        with self.connect() as conn:
            known = {str(row[0]) for row in conn.execute("SELECT history_time_ns FROM pose_history")}
        for path in (self.root / "staging").iterdir():
            if path.name not in known:
                LOG.warning("orphan history staging directory retained: %s", path)

    def stats(self):
        with self.connect() as conn:
            count = conn.execute("SELECT count(*) FROM pose_history").fetchone()[0]
            size = conn.execute("SELECT coalesce(sum(json_extract(metadata,'$.bytes')),0) FROM pose_history").fetchone()[0]
        disk = shutil.disk_usage(self.root)
        return {"count": count, "record_bytes": size, "disk_free_bytes": disk.free,
                "ready": self.error is None, "error": self.error}
