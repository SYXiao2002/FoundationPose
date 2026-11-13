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
DEFAULT_STREAM_WIDTH = 848
DEFAULT_STREAM_HEIGHT = 530
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
  if any_resolution is None:
    raise ValueError(f'Colour resolution {width}x{height} is unavailable; use --list-profiles')
  return any_resolution


def select_depth_profile(profile_list, width, height, fps):
  """Select a depth stream; explicit requests must match exactly."""
  default = profile_list.get_default_video_stream_profile()
  requested_size = width is not None
  width = default.get_width() if width is None else width
  height = default.get_height() if height is None else height
  for index in range(profile_list.get_count()):
    profile = profile_list.get_stream_profile_by_index(index).as_video_stream_profile()
    if (profile.get_width(), profile.get_height(), profile.get_fps()) == (width, height, fps):
      return profile
  if requested_size:
    raise ValueError(f'Depth profile {width}x{height}@{fps} is unavailable; use --list-profiles')
  raise ValueError(f'No default-resolution depth stream at {fps} FPS; use --depth-width/--depth-height/--depth-fps')


def profile_metadata(profile):
  return dict(width=profile.get_width(), height=profile.get_height(),
              fps=profile.get_fps(), format=str(profile.get_format()))


class FirstFrameDepthFusion:
  """Median of recent aligned depth measurements for a stationary first frame."""

  def __init__(self, samples=30, min_valid_fraction=0.8):
    if samples < 1 or not 0 < min_valid_fraction <= 1:
      raise ValueError('First-frame samples must be positive and valid fraction must be within (0, 1]')
    self.samples = samples
    self.min_valid_fraction = min_valid_fraction
    self.frames = deque(maxlen=samples)

  def reset(self):
    self.frames.clear()

  def add(self, depth_mm):
    if self.frames and depth_mm.shape != self.frames[0].shape:
      raise ValueError('Depth dimensions changed during first-frame fusion')
    self.frames.append(depth_mm.copy())
    if len(self.frames) < self.samples:
      return None
    stack = np.stack(self.frames)
    counts = (stack > 0).sum(axis=0)
    # Sort invalid measurements after valid ones. Select the middle valid
    # sample(s), rather than averaging surfaces or including zero-depth holes.
    ordered = np.sort(np.where(stack > 0, stack, np.uint16(65535)), axis=0)
    lower_index = np.maximum(counts - 1, 0) // 2
    upper_index = counts // 2
    lower = np.take_along_axis(ordered, lower_index[None], axis=0)[0].astype(np.float32)
    upper = np.take_along_axis(ordered, upper_index[None], axis=0)[0].astype(np.float32)
    valid = ((counts >= np.ceil(self.samples * self.min_valid_fraction)) & (depth_mm > 0))
    return np.where(valid, np.rint((lower + upper) * 0.5), 0).astype(np.uint16)


class DepthPostProcessor:
  """Apply device-compatible SDK filters before depth-to-colour alignment."""

  def __init__(self, device, ob, mode='first-frame', spatial_alpha=0.5, temporal_weight=0.4):
    self.filters = []
    self.metadata = {'mode': mode, 'stage': 'before_depth_to_color_alignment', 'filters': []}
    if mode == 'none':
      return
    available = {f.get_name(): f for f in
                 device.get_sensor(ob.OBSensorType.DEPTH_SENSOR).get_recommended_filters()}
    settings = [('SpatialAdvancedFilter', {'magnitude': 1, 'alpha': spatial_alpha, 'radius': 0})]
    if mode == 'static':
      settings.append(('TemporalFilter', {'weight': temporal_weight}))
    for name, values in settings:
      if name not in available:
        raise RuntimeError(f'{name} is not supported by this camera; use --depth-filter none')
      filt = available[name]
      for key, value in values.items():
        filt.set_config_value(key, value)
      filt.enable(True)
      self.filters.append(filt)
      params = {item.name: filt.get_config_value(item.name) for item in filt.get_config_schema_vec()}
      self.metadata['filters'].append({'name': name, 'parameters': params})

  def process(self, depth):
    if not self.filters:
      return depth
    # Keep measured invalid pixels invalid; no hole filling or historical depth
    # may invent foreground geometry along occlusions after alignment.
    invalid = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(
        depth.get_height(), depth.get_width()) == 0
    for filt in self.filters:
      result = filt.process(depth)
      if result is None:
        return None
      depth = result.as_depth_frame()
    if invalid.any():
      values = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(
          depth.get_height(), depth.get_width()).copy()
      values[invalid] = 0
      depth.update_data(values.tobytes())
    return depth

  def reset(self):
    for filt in self.filters:
      filt.reset()


def depth_to_millimetres(depth, min_mm=0, max_mm=65535):
  raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(
      depth.get_height(), depth.get_width())
  values = raw.astype(np.float32) * float(depth.get_depth_scale())
  valid = (raw > 0) & np.isfinite(values) & (values >= min_mm) & (values <= max_mm)
  return np.where(valid, np.clip(np.rint(values), 0, 65535), 0).astype(np.uint16)


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
  parser.add_argument("--width", type=int, default=DEFAULT_STREAM_WIDTH, help="colour width (default: 848)")
  parser.add_argument("--height", type=int, default=DEFAULT_STREAM_HEIGHT, help="colour height (default: 530)")
  parser.add_argument("--fps", type=int, default=None, help="preferred colour stream FPS")
  parser.add_argument("--depth-width", type=int, help="native depth width (default: 848); requires --depth-height")
  parser.add_argument("--depth-height", type=int, help="native depth height (default: 530); requires --depth-width")
  parser.add_argument("--depth-fps", type=int, help="native depth FPS (default: selected colour FPS)")
  parser.add_argument("--list-profiles", action="store_true", help="list selected camera profiles and presets, then exit")
  parser.add_argument("--camera-preset", help="optional SDK preset, e.g. 'High Accuracy'; restored when recording ends")
  parser.add_argument("--depth-filter", choices=("none", "spatial", "static", "first-frame"), default="first-frame",
                      help="first-frame (default): spatial filter + multi-frame median for the first saved depth only; static: spatial + temporal for every frame; spatial: every frame; none: no filters")
  parser.add_argument("--first-frame-samples", type=int, default=30, help="stationary measurements fused for the first saved frame")
  parser.add_argument("--first-frame-min-valid", type=float, default=0.8, help="minimum valid fraction per fused pixel (default: 0.8)")
  parser.add_argument("--spatial-alpha", type=float, default=0.5, help="SDK current-pixel weight (0.1..1); smaller smooths more")
  parser.add_argument("--temporal-weight", type=float, default=0.4, help="SDK current-frame weight (0.1..1), for static mode")
  parser.add_argument("--depth-min-mm", type=float, default=0.0, help="discard depths below this distance in millimetres")
  parser.add_argument("--depth-max-mm", type=float, default=65535.0, help="discard depths above this distance in millimetres")
  parser.add_argument("--save-unfiltered-depth", action="store_true", help="also save aligned, unfiltered uint16-mm depth in depth_raw/")
  parser.add_argument("--max-skew-ms", type=float, default=20.0)
  parser.add_argument("--warmup", type=int, default=60, help="discard this many frames before recording (default: 60)")
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
  if args.first_frame_samples < 1 or not 0 < args.first_frame_min_valid <= 1:
    parser.error('first-frame samples must be positive and min-valid must be within (0, 1]')
  if args.warmup < 0:
    parser.error('--warmup must be nonnegative')
  if (args.depth_width is None) != (args.depth_height is None):
    parser.error('--depth-width and --depth-height must be supplied together')
  if args.depth_width is None:
    args.depth_width, args.depth_height = DEFAULT_STREAM_WIDTH, DEFAULT_STREAM_HEIGHT
  if not (0.1 <= args.spatial_alpha <= 1 and 0.1 <= args.temporal_weight <= 1):
    parser.error('spatial alpha and temporal weight must be within 0.1..1')
  if not (0 <= args.depth_min_mm < args.depth_max_mm <= 65535):
    parser.error('depth range must satisfy 0 <= min < max <= 65535 mm')
  if any(value is not None and value <= 0 for value in
         (args.width, args.height, args.fps, args.depth_width, args.depth_height, args.depth_fps)):
    parser.error('stream dimensions and FPS must be positive')

  context = ob.Context()
  if args.list:
    devices = context.query_devices()
    print(json.dumps([
        {"name": devices.get_device_name_by_index(i),
         "serial": devices.get_device_serial_number_by_index(i)}
        for i in range(devices.get_count())
    ], indent=2))
    return

  device, info = pick_device(context, ob, args.serial)
  pipeline, config = ob.Pipeline(device), ob.Config()
  if args.list_profiles:
    for name, sensor in (('colour', ob.OBSensorType.COLOR_SENSOR), ('depth', ob.OBSensorType.DEPTH_SENSOR)):
      profiles = pipeline.get_stream_profile_list(sensor)
      print(f'{name} profiles:')
      for index in range(profiles.get_count()):
        profile = profiles.get_stream_profile_by_index(index).as_video_stream_profile()
        print(' ', profile_metadata(profile))
    presets = device.get_available_preset_list()
    print('presets:', [presets.get_name_by_index(i) for i in range(presets.get_count())])
    print('current preset:', device.get_current_preset_name())
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

  depth_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
  preferred_fps = args.fps or depth_profiles.get_default_video_stream_profile().get_fps()
  color_profile = select_video_profile(
      pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR), args.width, args.height, preferred_fps)
  depth_profile = select_depth_profile(
      depth_profiles, args.depth_width, args.depth_height, args.depth_fps or color_profile.get_fps())
  config.enable_stream(color_profile)
  config.enable_stream(depth_profile)
  align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
  raw_align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM) if args.save_unfiltered_depth else None
  pipeline.enable_frame_sync()

  segmentor = MobileSAMSegmentor(args.sam_model, device=args.sam_device)
  print(f"[info] output -> {out_dir}")
  print(f"[info] {KEY_HELP}")
  selected_profiles = {'color': profile_metadata(color_profile), 'depth': profile_metadata(depth_profile)}
  print(f'[info] stream profiles: {selected_profiles}')
  raw_depth_dir = out_dir / 'depth_raw'
  if args.save_unfiltered_depth:
    raw_depth_dir.mkdir(exist_ok=True)

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
  previous_preset = None
  active_preset = None
  postprocessor = None
  first_frame_fusion = (FirstFrameDepthFusion(args.first_frame_samples, args.first_frame_min_valid)
                        if args.depth_filter == 'first-frame' else None)
  first_frame_quality = None
  last_depth_timestamp = None

  try:
    if args.camera_preset:
      presets = device.get_available_preset_list()
      if not presets.has_preset(args.camera_preset):
        raise ValueError(f'Unsupported camera preset: {args.camera_preset}')
      previous_preset = device.get_current_preset_name()
      device.load_preset(args.camera_preset)
      active_preset = device.get_current_preset_name()
      print(f'[info] camera preset: {active_preset} (will restore {previous_preset})')
    postprocessor = DepthPostProcessor(device, ob, args.depth_filter, args.spatial_alpha, args.temporal_weight)
    print(f'[info] depth processing: {postprocessor.metadata}')
    pipeline.start(config)
    running = True
    while True:
      if args.duration > 0 and time.monotonic() - started >= args.duration:
        break
      frames = pipeline.wait_for_frames(1000)
      if frames is None:
        continue
      native_color, native_depth = frames.get_color_frame(), frames.get_depth_frame()
      if native_color is None or native_depth is None:
        continue
      if abs(native_color.get_timestamp_us() - native_depth.get_timestamp_us()) > args.max_skew_ms * 1000:
        continue
      timestamp = native_depth.get_timestamp_us()
      if timestamp == last_depth_timestamp:
        continue
      last_depth_timestamp = timestamp
      unfiltered_mm = None
      if raw_align is not None:
        unfiltered = raw_align.process(frames)
        if unfiltered is None:
          continue
        unfiltered_depth = unfiltered.as_frame_set().get_depth_frame()
        if unfiltered_depth is None:
          continue
        unfiltered_mm = depth_to_millimetres(unfiltered_depth)
      # Only the registration frame needs fusion in the default mode. All
      # subsequent saved frames use the camera's unfiltered aligned depth.
      filtered_depth = (native_depth if first_frame_fusion is not None and saved > 0
                        else postprocessor.process(native_depth))
      if filtered_depth is None:
        continue
      frames.push_frame(filtered_depth)
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
      if (depth.get_height(), depth.get_width()) != bgr.shape[:2]:
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

      depth_mm = depth_to_millimetres(depth, args.depth_min_mm, args.depth_max_mm)
      now = time.time()
      recent.append(now)
      fps_live = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
      enough_depth = int((depth_mm > 0).sum()) >= args.min_depth_pixels

      if bbox is None and not args.no_preview:
        bbox = choose_bbox(bgr)
        postprocessor.reset()
        if first_frame_fusion is not None:
          first_frame_fusion.reset()
        continue
      throttled = min_interval > 0 and (now - last_save) < min_interval
      capped = args.max_frames > 0 and saved >= args.max_frames
      first_frame_ready = True
      if first_frame_fusion is not None and saved == 0 and recording and enough_depth:
        fused = first_frame_fusion.add(depth_mm)
        first_frame_ready = fused is not None
        if fused is not None:
          depth_mm = fused
          enough_depth = int((depth_mm > 0).sum()) >= args.min_depth_pixels
        elif len(first_frame_fusion.frames) == 1:
          print(f'[info] collecting {args.first_frame_samples} stationary depth frames for the first saved frame')
      if recording and first_frame_ready and enough_depth and not throttled and not capped:
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
          if unfiltered_mm is not None and not cv2.imwrite(str(raw_depth_dir / name), unfiltered_mm):
            raise OSError(f'Failed to write {raw_depth_dir / name}')
          if not cv2.imwrite(str(masks_dir / name), mask):
            raise OSError(f"Failed to write {masks_dir / name}")
          if saved == 0 and first_frame_fusion is not None:
            first_frame_quality = {
                'frame_id': str(ns),
                'method': 'per-pixel median of valid, spatial-filtered aligned depth measurements',
                'samples': len(first_frame_fusion.frames),
                'min_valid_fraction': args.first_frame_min_valid,
                'keep_latest_invalid_pixels': True,
                'mask_valid_depth_fraction': float((depth_mm[mask > 0] > 0).mean()),
                'rgb_reference': 'latest RGB frame in the stationary capture window',
            }
            first_frame_fusion.reset()
            print(f'[info] first depth fused from {args.first_frame_samples} frames; '
                  f'valid depth in SAM mask: {first_frame_quality["mask_valid_depth_fraction"]:.1%}')
          saved += 1
          last_save = now

      if args.no_preview:
        if saved and saved % 30 == 0:
          print(f"\r[rec] saved={saved} fps={fps_live:.1f}", end="", flush=True)
        if capped or (args.duration > 0 and time.monotonic() - started >= args.duration):
          break
        continue

      state = "RECORDING" if recording else ("WARMUP" if warmup_left > 0 else "PAUSED")
      if recording and first_frame_fusion is not None and saved == 0:
        state = f'FIRST DEPTH {len(first_frame_fusion.frames)}/{args.first_frame_samples}'
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
        postprocessor.reset()
        if first_frame_fusion is not None:
          first_frame_fusion.reset()
        print(f"[info] {'RECORDING' if recording else 'PAUSED'}")
      elif key == ord("b"):
        bbox = choose_bbox(bgr)
        last_mask, last_score = None, None
        postprocessor.reset()
        if first_frame_fusion is not None:
          first_frame_fusion.reset()
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
      try:
        if previous_preset is not None:
          device.load_preset(previous_preset)
          print(f'[info] restored camera preset: {previous_preset}')
      finally:
        if preview_ok:
          cv2.destroyAllWindows()

  metadata = {
      **info,
      "backend": "pyorbbecsdk2; depth-to-colour aligned",
      "sdk_version": getattr(ob, '__version__', 'unknown'),
      "stream_profiles": selected_profiles,
      "requested_camera_preset": args.camera_preset,
      "active_camera_preset": active_preset,
      "depth_processing": postprocessor.metadata if postprocessor is not None else None,
      "first_frame_depth": first_frame_quality,
      "depth_range_mm": [args.depth_min_mm, args.depth_max_mm],
      "unfiltered_depth_storage": 'depth_raw/ aligned uint16 millimetres' if args.save_unfiltered_depth else None,
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
