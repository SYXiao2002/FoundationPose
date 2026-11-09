"""View first-frame RGB-D and instance poses, in a local web app or standalone HTML."""

from __future__ import annotations

import argparse
import base64
import errno
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

import cv2
import numpy as np
import trimesh
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parent
PALETTE = [
    "#e6534f", "#168ca3", "#5b9e45", "#9a62b2", "#d88722", "#3476c5",
    "#c34e83", "#7c8730", "#e36b35", "#5962bd", "#10a18d", "#a34d3a",
    "#7b68a6", "#6b7d45", "#ba7b33", "#377a7b", "#bd5360",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "results/multi_inbox03")
    parser.add_argument(
        "--scene-dir", type=Path,
        default=ROOT / "demo_data/eu-plug/multi_scene_eu-plug/multi_inbox03_eu-plug/raw",
        help="RGB-D sequence directory containing rgb/, depth/, and cam_K.txt",
    )
    parser.add_argument(
        "--mesh-file", type=Path,
        default=ROOT / "demo_data/eu-plug/mesh_eu-plug/TexturedMeshSimplified.obj",
    )
    parser.add_argument("--point-stride", type=int, default=2, help="Pixel stride used to sample the cloud")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--output", type=Path, help="Export a self-contained HTML file and exit")
    return parser.parse_args()


def build_mask_preview(color_image: np.ndarray, instances: list[dict]) -> bytes:
    overlay = color_image.copy()
    labels = []
    for instance in instances:
        mask = instance["mask"]
        color = tuple(int(instance["rgb"][i]) for i in (2, 1, 0))
        selected = mask > 0
        overlay[selected] = (
            overlay[selected].astype(np.float32) * 0.47
            + np.asarray(color, dtype=np.float32) * 0.53
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)
        moments = cv2.moments(mask)
        if moments["m00"]:
            center = (round(moments["m10"] / moments["m00"]), round(moments["m01"] / moments["m00"]))
            labels.append((str(instance["id"]), center, color))

    for label, center, color in labels:
        cv2.circle(overlay, center, 12, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, center, 12, color, 2, cv2.LINE_AA)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0]
        cv2.putText(overlay, label, (center[0] - text_size[0] // 2, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, color, 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode the instance mask preview")
    return encoded.tobytes()


class ViewerData:
    def __init__(self, debug_dir: Path, scene_dir: Path, mesh_file: Path, point_stride: int):
        self.debug_dir = debug_dir.expanduser().resolve()
        self.scene_dir = scene_dir.expanduser().resolve()
        self.mesh_file = mesh_file.expanduser().resolve()
        self.point_stride = point_stride
        if point_stride < 1:
            raise ValueError("--point-stride must be at least 1")
        for path in (self.debug_dir, self.scene_dir, self.mesh_file):
            if not path.exists():
                raise FileNotFoundError(path)

        self.instances = []
        directories = [path for path in self.debug_dir.glob("instance*")
                       if path.is_dir() and path.name.removeprefix("instance").isdigit()]
        for directory in sorted(directories, key=lambda path: int(path.name.removeprefix("instance"))):
            pose_dir = directory / "ob_in_cam"
            pose_files = sorted(pose_dir.glob("*.txt"), key=lambda path: (0, int(path.stem))
                                if path.stem.isdigit() else (1, path.stem))
            if not pose_files:
                continue
            index = int(directory.name.removeprefix("instance"))
            if not 1 <= index <= 0xFFFFFF:
                raise ValueError(f"Instance ID must be between 1 and 16777215: {directory.name}")
            self.instances.append({
                "id": index,
                "color": PALETTE[(index - 1) % len(PALETTE)],
                "rgb": [int(PALETTE[(index - 1) % len(PALETTE)][i:i + 2], 16) for i in (1, 3, 5)],
                "mask_file": directory / "ob_mask.png",
                "pose_file": pose_files[0],
            })
        if not self.instances:
            raise FileNotFoundError(f"No instance*/ob_in_cam/*.txt files under {self.debug_dir}")

        first_stems = {item["pose_file"].stem for item in self.instances}
        if len(first_stems) != 1:
            raise ValueError(f"Instances have different first frames: {sorted(first_stems)}")
        self.frame_stem = first_stems.pop()
        self.rgb_file = self.scene_dir / "rgb" / f"{self.frame_stem}.png"
        self.depth_file = self.scene_dir / "depth" / f"{self.frame_stem}.png"
        if not self.rgb_file.is_file() or not self.depth_file.is_file():
            raise FileNotFoundError(f"RGB-D frame {self.frame_stem} is missing under {self.scene_dir}")
        self.K = np.loadtxt(self.scene_dir / "cam_K.txt").reshape(3, 3)
        if not np.isfinite(self.K).all() or self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError("Invalid camera intrinsics")
        self.bgr = cv2.imread(str(self.rgb_file), cv2.IMREAD_COLOR)
        self.depth = cv2.imread(str(self.depth_file), cv2.IMREAD_UNCHANGED)
        if self.bgr is None or self.depth is None:
            raise OSError(f"Could not read RGB-D frame {self.frame_stem}")
        if self.depth.ndim != 2 or self.depth.shape != self.bgr.shape[:2]:
            raise ValueError("Depth must be aligned to the RGB image with the same dimensions")
        for item in self.instances:
            item["mask"] = cv2.imread(str(item["mask_file"]), cv2.IMREAD_GRAYSCALE)
            if item["mask"] is None or item["mask"].shape != self.depth.shape:
                raise ValueError(f"Missing or incorrectly sized mask: {item['mask_file']}")
            saved_color = item["pose_file"].parent.parent / "color.png"
            if saved_color.is_file() and not np.array_equal(cv2.imread(str(saved_color)), self.bgr):
                raise ValueError(f"RGB scene does not match the registration image: {saved_color}")

    @lru_cache(maxsize=1)
    def info(self):
        instances = []
        for instance in self.instances:
            pose = np.loadtxt(instance["pose_file"]).reshape(4, 4)
            if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
                raise ValueError(f"Invalid pose: {instance['pose_file']}")
            score_file = instance["pose_file"].parent.parent / "register_score.json"
            score = json.loads(score_file.read_text()).get("score") if score_file.is_file() else None
            instances.append({
                "id": instance["id"],
                "color": instance["color"],
                "pose": pose.astype(float).tolist(),
                "origin": pose[:3, 3].astype(float).tolist(),
                "score": score,
            })
        return {
            "scene": self.debug_dir.name,
            "frame": self.frame_stem,
            "point_stride": self.point_stride,
            "image_size": [self.bgr.shape[1], self.bgr.shape[0]],
            "K": self.K.tolist(),
            "mesh_name": self.mesh_file.name,
            "instances": instances,
        }

    @lru_cache(maxsize=1)
    def scene_cloud(self) -> bytes:
        bgr, depth = self.bgr, self.depth
        stride = self.point_stride
        depth = depth[::stride, ::stride].astype(np.float32) * 0.001
        rgb = cv2.cvtColor(bgr[::stride, ::stride], cv2.COLOR_BGR2RGB)
        height, width = depth.shape
        v, u = np.indices((height, width), dtype=np.float32)
        u *= stride
        v *= stride
        valid = depth > 0
        z = depth[valid]
        x = (u[valid] - self.K[0, 2]) * z / self.K[0, 0]
        y = (v[valid] - self.K[1, 2]) * z / self.K[1, 1]
        xyz = np.column_stack((x, y, z)).astype("<f4", copy=False)
        colors = rgb[valid].astype(np.uint8, copy=False)
        header = np.asarray([len(xyz)], dtype="<u4").tobytes()
        return header + xyz.tobytes() + colors.tobytes()

    @lru_cache(maxsize=1)
    def mask_map(self) -> bytes:
        labels = np.zeros((*self.depth.shape, 3), dtype=np.uint8)
        for item in self.instances:
            index = item["id"]
            if index > 0xFFFFFF:
                raise ValueError("Instance IDs must fit in 24 bits")
            labels[item["mask"] > 0] = [index & 255, (index >> 8) & 255, (index >> 16) & 255]
        ok, image = cv2.imencode(".png", labels[:, :, ::-1])
        if not ok:
            raise OSError("Could not encode the mask map")
        return image.tobytes()

    @lru_cache(maxsize=1)
    def mask_preview(self) -> bytes:
        return build_mask_preview(self.bgr, self.instances)

    @lru_cache(maxsize=1)
    def model(self):
        mesh = trimesh.load(self.mesh_file, force="mesh", process=False)
        if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError(f"Empty or invalid model: {self.mesh_file}")
        return {
            "vertices": np.asarray(mesh.vertices, dtype=float).tolist(),
            "faces": np.asarray(mesh.faces, dtype=int).tolist(),
        }


def make_handler(data: ViewerData, ui_dir: Path):
    plotly_js = get_plotlyjs().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            routes = {
                "/": (ui_dir / "index.html", "text/html; charset=utf-8"),
                "/style.css": (ui_dir / "style.css", "text/css; charset=utf-8"),
                "/viewer.js": (ui_dir / "viewer.js", "text/javascript; charset=utf-8"),
            }
            if self.path in routes:
                path, content_type = routes[self.path]
                if not path.is_file():
                    self.send_error(404)
                    return
                payload = path.read_bytes()
            elif self.path == "/plotly.js":
                payload, content_type = plotly_js, "text/javascript; charset=utf-8"
            elif self.path == "/api/info":
                payload, content_type = json.dumps(data.info()).encode(), "application/json"
            elif self.path == "/api/scene":
                payload, content_type = data.scene_cloud(), "application/octet-stream"
            elif self.path == "/api/model":
                payload, content_type = json.dumps(data.model()).encode(), "application/json"
            elif self.path == "/api/mask.jpg":
                payload, content_type = data.mask_preview(), "image/jpeg"
            elif self.path == "/api/masks.png":
                payload, content_type = data.mask_map(), "image/png"
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            if len(args) < 2 or str(args[1]) not in {"200", "304", "204"}:
                super().log_message(format, *args)

    return Handler


def export_html(data: ViewerData, output: Path):
    ui_dir = ROOT / "web/multi_pose_viewer"
    payload = {
        "info": data.info(),
        "model": data.model(),
        "cloud": base64.b64encode(data.scene_cloud()).decode("ascii"),
        "mask": "data:image/jpeg;base64," + base64.b64encode(data.mask_preview()).decode("ascii"),
        "maskMap": "data:image/png;base64," + base64.b64encode(data.mask_map()).decode("ascii"),
    }
    document = (ui_dir / "index.html").read_text()
    document = document.replace('<link rel="stylesheet" href="/style.css">',
                                "<style>" + (ui_dir / "style.css").read_text() + "</style>")
    document = document.replace('<script defer src="/plotly.js"></script>',
                                "<script>" + get_plotlyjs().replace("</script", "<\\/script") + "</script>")
    document = document.replace('<script defer src="/viewer.js"></script>', "")
    document = document.replace('src="/api/mask.jpg"', 'src="' + payload["mask"] + '"')
    script = (ui_dir / "viewer.js").read_text()
    embedded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    document = document.replace("</body>", "<script>window.POSE_DATA=" + embedded + ";</script>"
                                + "<script>" + script + "</script></body>")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"Exported first-frame viewer: {output}")


def main():
    args = parse_args()
    data = ViewerData(args.debug_dir, args.scene_dir, args.mesh_file, args.point_stride)
    data.info()
    if args.output:
        export_html(data, args.output)
        return
    handler = make_handler(data, ROOT / "web/multi_pose_viewer")
    port = args.port
    while True:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            break
        except OSError as error:
            if error.errno != errno.EADDRINUSE or port >= args.port + 20:
                raise
            port += 1
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"First-frame pose viewer: {url}")
    print(f"Instances: {len(data.instances)}; frame: {data.frame_stem}; point stride: {data.point_stride}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
