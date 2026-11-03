"""Render saved FoundationPose poses over an RGB sequence and encode a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh

from Utils import draw_posed_3d_box, draw_xyz_axis


def infer_fps(frame_stems):
  timestamps = []
  for stem in frame_stems:
    try:
      timestamps.append(int(stem))
    except ValueError:
      continue
  if len(timestamps) < 2:
    return 30.0
  delta = np.diff(np.asarray(timestamps, dtype=np.float64))
  delta = delta[delta > 0]
  return float(np.clip(1e9 / np.median(delta), 1.0, 60.0)) if len(delta) else 30.0


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scene-dir", type=Path, required=True, help="Captured scene with rgb/, depth/, masks/, cam_K.txt")
  parser.add_argument("--pose-dir", type=Path, required=True, help="FoundationPose ob_in_cam directory")
  parser.add_argument("--mesh-file", type=Path, required=True, help="Same OBJ used for pose estimation")
  parser.add_argument("--output", type=Path, help="Output MP4 (default: <pose-dir>/../pose_video.mp4)")
  parser.add_argument("--track-vis-dir", type=Path, help="Optionally save the rendered frames as PNGs")
  parser.add_argument("--fps", type=float, help="Output frame rate; default is estimated from nanosecond filenames")
  parser.add_argument("--stride", type=int, default=1, help="Use every Nth RGB/pose frame")
  parser.add_argument("--mask-overlay", action="store_true", help="Tint SAM mask pixels in green")
  args = parser.parse_args()

  if args.stride < 1:
    parser.error("--stride must be at least 1")
  scene_dir = args.scene_dir.expanduser()
  pose_dir = args.pose_dir.expanduser()
  mesh_file = args.mesh_file.expanduser()
  output = args.output.expanduser() if args.output else pose_dir.parent / "pose_video.mp4"
  track_vis_dir = args.track_vis_dir.expanduser() if args.track_vis_dir else None
  if track_vis_dir is not None:
    track_vis_dir.mkdir(parents=True, exist_ok=True)

  color_files = sorted((scene_dir / "rgb").glob("*.png"))[::args.stride]
  if not color_files:
    raise FileNotFoundError(f"No RGB PNG frames found under {scene_dir / 'rgb'}")
  K = np.loadtxt(scene_dir / "cam_K.txt").reshape(3, 3)
  mesh = trimesh.load(mesh_file)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

  valid_frames = []
  missing = []
  for color_file in color_files:
    pose_file = pose_dir / f"{color_file.stem}.txt"
    if pose_file.is_file():
      valid_frames.append((color_file, pose_file))
    else:
      missing.append(color_file.stem)
  if not valid_frames:
    raise FileNotFoundError(f"No RGB frames have matching poses in {pose_dir}")

  fps = args.fps or infer_fps([color_file.stem for color_file, _ in valid_frames])
  first = cv2.imread(str(valid_frames[0][0]), cv2.IMREAD_COLOR)
  if first is None:
    raise OSError(f"Could not read {valid_frames[0][0]}")
  height, width = first.shape[:2]
  output.parent.mkdir(parents=True, exist_ok=True)
  writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
  if not writer.isOpened():
    raise RuntimeError(f"Could not open MP4 writer for {output}")

  written = 0
  try:
    for index, (color_file, pose_file) in enumerate(valid_frames):
      image = cv2.imread(str(color_file), cv2.IMREAD_COLOR)
      if image is None:
        raise OSError(f"Could not read {color_file}")
      if image.shape[:2] != (height, width):
        raise ValueError(f"Frame size changed at {color_file}: {image.shape[:2]}")

      if args.mask_overlay:
        mask_file = scene_dir / "masks" / color_file.name
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
          raise FileNotFoundError(f"Missing mask: {mask_file}")
        selected = mask > 0
        image[selected] = (0.55 * image[selected] + 0.45 * np.array([0, 190, 0])).astype(np.uint8)

      pose = np.loadtxt(pose_file).reshape(4, 4)
      centered_pose = pose @ np.linalg.inv(to_origin)
      image = draw_posed_3d_box(K, image, centered_pose, bbox, line_color=(0, 255, 255), linewidth=2)
      image = draw_xyz_axis(image, pose, scale=0.04, K=K, thickness=2, is_input_rgb=False)
      translation = pose[:3, 3]
      cv2.putText(image, f"frame {index + 1}/{len(valid_frames)}  t=[{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}] m",
                  (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
      writer.write(image)
      if track_vis_dir is not None and not cv2.imwrite(str(track_vis_dir / color_file.name), image):
        raise OSError(f"Could not write visualization frame to {track_vis_dir / color_file.name}")
      written += 1
  finally:
    writer.release()

  duration = written / fps
  print(f"Saved {output}")
  print(f"Frames: {written}; FPS: {fps:.2f}; duration: {duration:.1f}s; size: {width}x{height}")
  if missing:
    print(f"Skipped {len(missing)} RGB frames without matching pose files")


if __name__ == "__main__":
  main()
