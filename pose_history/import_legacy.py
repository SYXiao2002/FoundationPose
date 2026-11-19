"""Explicit, idempotent import of existing first-frame instance results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import trimesh

from pose_service.protocol import rigid
from .store import HistoryStore, digest_file


def import_results(store, debug_dir, scene_dir, mesh_file, *, dry_run=True):
    debug_dir, scene_dir, mesh_file = (Path(p).resolve() for p in (debug_dir, scene_dir, mesh_file))
    model_sha = digest_file(mesh_file)
    asset = None
    if not dry_run:
        asset = store.snapshot_model(trimesh.load(mesh_file, force='mesh', process=False), model_sha,
                                     {"source": "legacy_import", "mesh_name": mesh_file.name})
    report = []
    for directory in sorted(debug_dir.glob("instance*")):
        if not directory.is_dir() or not directory.name.removeprefix("instance").isdigit():
            continue
        poses = sorted((directory / "ob_in_cam").glob("*.txt"), key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
        item = {"instance": directory.name}
        try:
            if not poses:
                raise ValueError("no first-frame pose")
            pose_file = poses[0]
            frame = pose_file.stem
            rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{frame}.png").convert("RGB"))
            depth = np.asarray(Image.open(scene_dir / "depth" / f"{frame}.png")).astype(np.float32) * np.float32(.001)
            mask = (np.asarray(Image.open(directory / "ob_mask.png").convert("L")) > 0).astype(np.uint8)
            K = np.loadtxt(scene_dir / "cam_K.txt")
            pose = rigid(np.loadtxt(pose_file))
            if depth.shape != rgb.shape[:2] or mask.shape != depth.shape:
                raise ValueError("RGB/depth/mask dimensions differ")
            if not np.array_equal(rgb, np.asarray(Image.open(directory / "color.png").convert("RGB"))):
                raise ValueError("registration RGB does not match source scene")
            score = json.loads((directory / "register_score.json").read_text())["score"]
            key = hashlib.sha256(str(directory).encode() + pose_file.read_bytes()).hexdigest()
            request = {"type": "register", "version": 1, "request_id": "legacy-" + key,
                       "observation_id": "legacy-" + hashlib.sha256(str(scene_dir).encode()).hexdigest(),
                       "object_id": mesh_file.parent.name, "model_sha256": model_sha,
                       "rgb": rgb, "depth_m": depth, "mask": mask, "K": K,
                       "instance_id": directory.name, "mask_source": "legacy", "source": "legacy_import",
                       "capture_timestamp_ns": frame if frame.isdigit() else None, "capture_clock": "unknown"}
            item.update(frame=frame, status="ready", score=score)
            if not dry_run:
                record, created = store.begin(request, asset, {"source": "legacy_import", "versions": "unknown"})
                item.update(history_id=record["history_id"], status="imported" if created else "already_imported")
                if created:
                    path = store.save_inputs(record, request)
                    for name in ["vis_score.png", "vis_refiner.png", "color.png", "depth.png", "ob_mask.png", "scene_raw.ply", "scene_complete.ply", "init_center.ply"]:
                        if (directory / name).is_file():
                            shutil.copyfile(directory / name, path / "debug" / name)
                    validation_file = debug_dir / "pose_viewer_validation.json"
                    screening = {"status": "insufficient_evidence", "warning": "legacy screening unavailable", "metrics": {}}
                    if validation_file.is_file():
                        info = json.loads(validation_file.read_text())
                        for instance in info.get("instances", []):
                            if instance["id"] == int(directory.name.removeprefix("instance")):
                                screening = instance["validation"]
                                break
                        record["parameters"]["screening"] = info.get("validation", {}).get("thresholds", {})
                    store.publish(record, {"type": "result", "version": 1, "request_id": request["request_id"],
                        "observation_id": request["observation_id"], "object_id": request["object_id"],
                        "model_sha256": model_sha, "T_camera_object": pose.tolist(), "score": score, "screening": screening})
        except (OSError, ValueError, KeyError) as exc:
            item.update(status="unavailable", message=str(exc))
        report.append(item)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--debug-dir", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--mesh-file", required=True)
    parser.add_argument("--execute", action="store_true", help="Without this flag only inspect inputs")
    args = parser.parse_args()
    store = HistoryStore(args.root, recover=False) if args.execute else None
    print(json.dumps(import_results(store, args.debug_dir, args.scene_dir, args.mesh_file, dry_run=not args.execute), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
