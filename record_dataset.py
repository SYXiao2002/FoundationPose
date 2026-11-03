"""Record an Orbbec RGB-D sequence in FoundationPose's YCB-ineoat format.

The output can be passed directly to ``run_demo.py`` after recording::

    <output>/
      rgb/<timestamp>.png       BGR uint8 colour image
      depth/<timestamp>.png     uint16 depth in millimetres
      masks/<timestamp>.png     binary SAM mask (0 or 255)
      cam_K.txt                 3x3 colour-camera intrinsics
      capture_meta.json         capture and SAM metadata

The first usable colour frame is shown in a window. Select the object with the
mouse and press ENTER/SPACE. MobileSAM is then prompted with the selected box
and its centre point for every saved frame. ``--bbox`` can be used for headless
operation or to reuse a known box (x1 y1 x2 y2, in colour-image pixels).

This script intentionally keeps camera and MobileSAM imports lazy. Run it with
the camera environment that provides ``pyorbbecsdk`` and ``mobile_sam``; the
FoundationPose inference environment does not need either package.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = CODE_DIR / "demo_data"
DEFAULT_SAM_MODEL = Path(
    "/home/yixiaoshen/Repo/chaung_wei_2_jiao_fu/backend/models/mobile_sam.pt"
)
KEY_HELP = "[SPACE/r] record/pause   [b] reselect box   [q/ESC] stop   [h] help"


def decode_color(frame, ob):
  """Decode an Orbbec colour frame to a BGR uint8 image."""
  height, width = frame.get_height(), frame.get_width()
  data = np.asarray(frame.get_data())
  fmt = frame.get_format()
  if fmt == ob.OBFormat.RGB:
    return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
  if fmt == ob.OBFormat.BGR:
    return data.reshape(height, width, 3).copy()
  if fmt == ob.OBFormat.MJPG:
    result = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if result is None:
      raise ValueError("Invalid MJPEG colour frame")
    return result
  if fmt == ob.OBFormat.YUYV:
    return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
  raise ValueError(f"Unsupported colour format: {fmt}")


def select_video_profile(profile_list, width, height, fps):
  """Pick the requested colour profile, with a same-resolution fallback."""
  any_resolution = None
  for index in range(profile_list.get_count()):
    try:
      profile = profile_list.get_stream_profile_by_index(index).as_video_stream_profile()
    except Exception:
      continue
    if profile.get_width() != width or profile.get_height() != height:
      continue
    if fps and profile.get_fps() == fps:
      return profile
    if any_resolution is None:
      any_resolution = profile
  return any_resolution or profile_list.get_default_video_stream_profile()


def pick_device(context, ob, serial):
  """Select a camera by serial, or the only connected camera."""
  devices = context.query_devices()
  available = [
      {
          "name": devices.get_device_name_by_index(i),
          "serial": devices.get_device_serial_number_by_index(i),
      }
      for i in range(devices.get_count())
  ]
  if not available:
    raise RuntimeError("No Orbbec camera detected")
  if serial:
    info = next((item for item in available if item["serial"] == serial), None)
    if info is None:
      raise ValueError(f"Camera not found: {serial}; available: {available}")
  elif len(available) == 1:
    info = available[0]
    print(f"[info] using the only camera: {info['name']} ({info['serial']})")
  else:
    raise ValueError(f"Multiple cameras detected, pass --serial: {available}")
  device = devices.get_device_by_serial_number(info["serial"])
  info["firmware"] = device.get_device_info().get_firmware_version()
  return device, info


class MobileSAMSegmentor:
  """MobileSAM box/centre-point segmenter used by the offline grasp tools."""

  def __init__(self, model_path, device=None):
    try:
      import torch
      from mobile_sam import SamPredictor, sam_model_registry
    except ImportError as exc:
      raise RuntimeError(
          "MobileSAM is unavailable. Run this script with the camera environment "
          "that installs mobile-sam and torch."
      ) from exc

    model_path = Path(model_path).expanduser()
    if not model_path.is_file():
      raise FileNotFoundError(f"MobileSAM checkpoint not found: {model_path}")
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_t"](checkpoint=str(model_path))
    sam.to(device=self.device)
    self.predictor = SamPredictor(sam)
    print(f"[info] MobileSAM loaded on {self.device}: {model_path}")

  def predict(self, bgr, bbox):
    """Return a binary uint8 mask for an xyxy box."""
    height, width = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(1, min(width, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(1, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
      raise ValueError(f"Invalid SAM box: {(x1, y1, x2, y2)}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    self.predictor.set_image(rgb)
    center = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
    point_label = np.array([1], dtype=np.int32)
    masks, scores, _ = self.predictor.predict(
        point_coords=center,
        point_labels=point_label,
        box=np.array([x1, y1, x2, y2], dtype=np.float32),
        multimask_output=False,
    )
    mask = np.asarray(masks[0], dtype=bool)
    if mask.shape != (height, width):
      raise ValueError(f"SAM returned {mask.shape}, expected {(height, width)}")
    if not mask.any():
      raise RuntimeError("MobileSAM returned an empty mask")
    return mask.astype(np.uint8) * 255, float(np.asarray(scores).reshape(-1)[0])


def parse_bbox(values):
  if values is None:
    return None
  if len(values) != 4:
    raise ValueError("--bbox requires x1 y1 x2 y2")
  x1, y1, x2, y2 = map(float, values)
  if not (x2 > x1 and y2 > y1):
    raise ValueError("--bbox must satisfy x2 > x1 and y2 > y1")
  return np.array([x1, y1, x2, y2], dtype=np.float32)


def choose_bbox(bgr, initial=None):
  """Interactively choose an object box unless one was supplied on the CLI."""
  if initial is not None:
    return initial.copy()
  print("[info] select the object in the preview, then press ENTER or SPACE")
  x, y, width, height = cv2.selectROI("select object", bgr, fromCenter=False, showCrosshair=True)
  cv2.destroyWindow("select object")
  if width <= 1 or height <= 1:
    raise RuntimeError("No valid object box selected")
  return np.array([x, y, x + width, y + height], dtype=np.float32)


def draw_preview(bgr, depth_mm, mask, bbox, state, saved, fps_live):
  color_view = bgr.copy()
  if mask is not None:
    color_view[mask > 0] = (
        0.55 * color_view[mask > 0] + 0.45 * np.array([0, 190, 0])
    ).astype(np.uint8)
  if bbox is not None:
    x1, y1, x2, y2 = np.rint(bbox).astype(int)
    cv2.rectangle(color_view, (x1, y1), (x2, y2), (0, 255, 255), 2)
  depth_view = cv2.applyColorMap(cv2.convertScaleAbs(depth_mm, alpha=255.0 / 2000.0), cv2.COLORMAP_JET)
  depth_view[depth_mm == 0] = 0
  canvas = np.hstack([color_view, depth_view])
  lines = [f"{state}  saved={saved}  fps={fps_live:.1f}", KEY_HELP]
  for index, text in enumerate(lines):
    y = 24 + index * 24
    cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
  return canvas


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--object", default="object", help="object name in the auto-generated output folder")
  parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="base directory for auto-generated captures")
  parser.add_argument("--output", type=Path, help="explicit output folder")
  parser.add_argument("--overwrite", action="store_true", help="allow writing into an existing output folder")
  parser.add_argument("--serial", help="camera serial; required when multiple cameras are connected")
  parser.add_argument("--list", action="store_true", help="list connected cameras and exit")
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument("--fps", type=int, default=None, help="preferred colour stream FPS")
  parser.add_argument("--max-skew-ms", type=float, default=20.0)
  parser.add_argument("--warmup", type=int, default=15, help="discard this many frames before recording")
  parser.add_argument("--record-fps", type=float, default=0.0, help="save throttle; 0 saves every valid frame")
  parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames; 0 is unlimited")
  parser.add_argument("--duration", type=float, default=0.0, help="headless stop time in seconds; 0 is unlimited")
  parser.add_argument("--min-depth-pixels", type=int, default=100)
  parser.add_argument("--bbox", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="SAM box prompt")
  parser.add_argument("--sam-model", type=Path, default=DEFAULT_SAM_MODEL)
  parser.add_argument("--sam-device", choices=("cuda", "cpu"), default=None)
  parser.add_argument("--no-preview", action="store_true", help="record immediately; requires --bbox")
  args = parser.parse_args()

  try:
    import pyorbbecsdk as ob
  except ImportError as exc:
    raise SystemExit(
        "pyorbbecsdk is unavailable. Run with the Orbbec/MobileSAM environment, "
        "for example /home/yixiaoshen/Repo/chaung_wei_2_jiao_fu/.venv/bin/python record_dataset.py"
    ) from exc

  bbox = parse_bbox(args.bbox)
  if args.no_preview and bbox is None:
    parser.error("--no-preview requires --bbox x1 y1 x2 y2")
  if args.record_fps < 0 or args.max_frames < 0 or args.duration < 0:
    parser.error("record rate, max frames and duration must be non-negative")

  context = ob.Context()
  if args.list:
    devices = context.query_devices()
    print(json.dumps([
        {"name": devices.get_device_name_by_index(i),
         "serial": devices.get_device_serial_number_by_index(i)}
        for i in range(devices.get_count())
    ], indent=2))
    return

  if args.output is not None:
    out_dir = args.output.expanduser()
  else:
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = Path(args.data_dir).expanduser() / f"{stamp}_{args.object}"
  if out_dir.exists() and not args.overwrite:
    raise SystemExit(f"Output exists: {out_dir}; pass --overwrite or choose another output")
  rgb_dir, depth_dir, masks_dir = out_dir / "rgb", out_dir / "depth", out_dir / "masks"
  rgb_dir.mkdir(parents=True, exist_ok=True)
  depth_dir.mkdir(parents=True, exist_ok=True)
  masks_dir.mkdir(parents=True, exist_ok=True)

  device, info = pick_device(context, ob, args.serial)
  pipeline, config = ob.Pipeline(device), ob.Config()
  color_profile = select_video_profile(
      pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR), args.width, args.height, args.fps)
  depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
  config.enable_stream(color_profile)
  config.enable_stream(depth_profile)
  align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
  pipeline.enable_frame_sync()

  segmentor = MobileSAMSegmentor(args.sam_model, device=args.sam_device)
  print(f"[info] output -> {out_dir}")
  print(f"[info] {KEY_HELP}")

  K = None
  depth_scale_mm = None
  recording = args.no_preview
  warmup_left = args.warmup
  saved = 0
  last_save = 0.0
  last_ns = None
  last_mask = None
  last_score = None
  min_interval = 1.0 / args.record_fps if args.record_fps > 0 else 0.0
  recent = deque(maxlen=30)
  started = time.monotonic()
  running = False
  preview_ok = not args.no_preview
  bgr = None

  try:
    pipeline.start(config)
    running = True
    while True:
      frames = pipeline.wait_for_frames(1000)
      if frames is None:
        continue
      aligned = align.process(frames)
      if aligned is None:
        continue
      frame_set = aligned.as_frame_set()
      color, depth = frame_set.get_color_frame(), frame_set.get_depth_frame()
      if color is None or depth is None:
        continue
      if abs(color.get_timestamp_us() - depth.get_timestamp_us()) > args.max_skew_ms * 1000:
        continue

      bgr = decode_color(color, ob)
      raw_depth = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(
          depth.get_height(), depth.get_width()).copy()
      if raw_depth.shape != bgr.shape[:2]:
        continue
      if K is None:
        intr = color.get_stream_profile().as_video_stream_profile().get_intrinsic()
        K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], dtype=np.float64)
        depth_scale_mm = float(depth.get_depth_scale())
        np.savetxt(out_dir / "cam_K.txt", K)
        print(f"[info] native resolution {bgr.shape[1]}x{bgr.shape[0]}, depth scale {depth_scale_mm:g} mm/unit")

      if warmup_left > 0:
        warmup_left -= 1
        continue

      depth_mm = np.clip(np.rint(raw_depth.astype(np.float32) * depth_scale_mm), 0, 65535).astype(np.uint16)
      now = time.time()
      recent.append(now)
      fps_live = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
      enough_depth = int((depth_mm > 0).sum()) >= args.min_depth_pixels

      if bbox is None and not args.no_preview:
        bbox = choose_bbox(bgr)
      throttled = min_interval > 0 and (now - last_save) < min_interval
      capped = args.max_frames > 0 and saved >= args.max_frames
      if recording and enough_depth and not throttled and not capped:
        try:
          mask, score = segmentor.predict(bgr, bbox)
        except Exception as exc:
          print(f"\n[warn] SAM failed for current frame: {exc}")
          mask, score = None, None
        if mask is not None:
          last_mask, last_score = mask, score
          ns = time.time_ns()
          if last_ns is not None and ns <= last_ns:
            ns = last_ns + 1
          last_ns = ns
          name = f"{ns}.png"
          if not cv2.imwrite(str(rgb_dir / name), bgr):
            raise OSError(f"Failed to write {rgb_dir / name}")
          if not cv2.imwrite(str(depth_dir / name), depth_mm):
            raise OSError(f"Failed to write {depth_dir / name}")
          if not cv2.imwrite(str(masks_dir / name), mask):
            raise OSError(f"Failed to write {masks_dir / name}")
          saved += 1
          last_save = now

      if args.no_preview:
        if saved and saved % 30 == 0:
          print(f"\r[rec] saved={saved} fps={fps_live:.1f}", end="", flush=True)
        if capped or (args.duration > 0 and time.monotonic() - started >= args.duration):
          break
        continue

      state = "RECORDING" if recording else ("WARMUP" if warmup_left > 0 else "PAUSED")
      try:
        cv2.imshow("record_dataset", draw_preview(bgr, depth_mm, last_mask, bbox, state, saved, fps_live))
      except cv2.error as exc:
        if args.bbox is None:
          raise RuntimeError("GUI is unavailable and no --bbox was supplied") from exc
        print(f"\n[warn] GUI unavailable ({exc}); continuing headless")
        args.no_preview = True
        recording = True
        preview_ok = False
        continue
      key = cv2.waitKey(1) & 0xFF
      if key in (ord("q"), 27):
        break
      if key in (ord(" "), ord("r")):
        recording = not recording
        print(f"[info] {'RECORDING' if recording else 'PAUSED'}")
      elif key == ord("b"):
        bbox = choose_bbox(bgr)
        last_mask, last_score = None, None
      elif key == ord("h"):
        print(f"[info] {KEY_HELP}")
      if capped or (args.duration > 0 and time.monotonic() - started >= args.duration):
        break
  except KeyboardInterrupt:
    print("\n[info] interrupted, finalising...")
  finally:
    try:
      if running:
        pipeline.stop()
    finally:
      if preview_ok:
        cv2.destroyAllWindows()

  metadata = {
      **info,
      "backend": "pyorbbecsdk2; depth-to-colour aligned",
      "alignment": "depth_to_color",
      "width": int(bgr.shape[1]) if bgr is not None else None,
      "height": int(bgr.shape[0]) if bgr is not None else None,
      "K": K.tolist() if K is not None else None,
      "raw_depth_scale_mm": depth_scale_mm,
      "depth_storage": "uint16 millimetres",
      "color_storage": "bgr uint8",
      "mask_storage": "uint8 binary 0/255 from MobileSAM",
      "frames_saved": saved,
      "sam_model": str(Path(args.sam_model).expanduser()),
      "sam_device": segmentor.device,
      "sam_box_xyxy": bbox.tolist() if bbox is not None else None,
      "last_sam_score": last_score,
  }
  if K is not None:
    (out_dir / "capture_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")

  print("\n" + "=" * 60)
  print(f"Saved {saved} RGB-D + SAM frames to {out_dir}")
  if saved:
    print("Run FoundationPose with:")
    print(f"  python run_demo.py --mesh_file <mesh.obj> --test_scene_dir {out_dir} --debug 0")
  else:
    print("[warn] nothing recorded")
  print("=" * 60)


if __name__ == "__main__":
  main()
