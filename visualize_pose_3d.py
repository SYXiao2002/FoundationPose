"""View a saved FoundationPose pose against the camera-frame scene cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


ROOT = Path(__file__).resolve().parent


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--debug-dir", type=Path, default=ROOT / "debug/eu-plug-5")
  parser.add_argument("--mesh-file", type=Path, default=ROOT / "demo_data/eu-plug-mesh/TexturedMeshSimplified.obj")
  parser.add_argument("--pose", type=Path, help="Pose matrix; defaults to the first ob_in_cam/*.txt")
  parser.add_argument("--scene-cloud", type=Path, help="Defaults to <debug-dir>/scene_complete.ply")
  parser.add_argument("--crop-padding", type=float, default=0.10, help="Scene crop padding around the posed model, in meters")
  parser.add_argument("--axis-size", type=float, default=0.04, help="Camera and object coordinate frame size, in meters")
  parser.add_argument("--export-dir", type=Path, help="Also export scene crop, posed mesh, and axes as PLY files")
  parser.add_argument("--no-window", action="store_true", help="Only export files; requires --export-dir")
  args = parser.parse_args()

  debug_dir = args.debug_dir.expanduser()
  mesh_file = args.mesh_file.expanduser()
  pose_file = args.pose.expanduser() if args.pose else None
  if pose_file is None:
    pose_files = sorted((debug_dir / "ob_in_cam").glob("*.txt"))
    if not pose_files:
      raise FileNotFoundError(f"No pose matrices found under {debug_dir / 'ob_in_cam'}")
    pose_file = pose_files[0]
  scene_file = args.scene_cloud.expanduser() if args.scene_cloud else debug_dir / "scene_complete.ply"

  if args.no_window and args.export_dir is None:
    parser.error("--no-window requires --export-dir")
  if args.crop_padding < 0 or args.axis_size <= 0:
    parser.error("--crop-padding must be nonnegative and --axis-size must be positive")

  pose = np.loadtxt(pose_file).reshape(4, 4)
  mesh = trimesh.load(mesh_file, force="mesh")
  vertices = trimesh.transform_points(np.asarray(mesh.vertices), pose)

  scene = o3d.io.read_point_cloud(str(scene_file))
  if not scene.has_points():
    raise ValueError(f"No points found in {scene_file}")
  crop = o3d.geometry.AxisAlignedBoundingBox(
      vertices.min(axis=0) - args.crop_padding,
      vertices.max(axis=0) + args.crop_padding,
  )
  scene = scene.crop(crop)
  if not scene.has_points():
    raise ValueError(f"The scene crop is empty; increase --crop-padding (currently {args.crop_padding} m)")

  posed_mesh = o3d.geometry.TriangleMesh(
      o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
      o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
  )
  posed_mesh.compute_vertex_normals()
  posed_mesh.paint_uniform_color([0.95, 0.22, 0.12])
  posed_mesh.transform(pose)

  camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size)
  object_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size)
  object_axes.transform(pose)

  print(f"Pose: {pose_file}")
  print(f"Object origin in camera coordinates (m): {pose[:3, 3]}")
  print(f"Rotation determinant: {np.linalg.det(pose[:3, :3]):.6f}")
  print(f"Scene points shown: {len(scene.points)}")

  if args.export_dir:
    export_dir = args.export_dir.expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(export_dir / "scene_crop.ply"), scene)
    o3d.io.write_triangle_mesh(str(export_dir / "posed_model.ply"), posed_mesh)
    o3d.io.write_triangle_mesh(str(export_dir / "camera_axes.ply"), camera_axes)
    o3d.io.write_triangle_mesh(str(export_dir / "object_axes.ply"), object_axes)
    print(f"Exported PLY files to {export_dir}")

  if not args.no_window:
    o3d.visualization.draw_geometries(
        [scene, posed_mesh, camera_axes, object_axes],
        window_name="FoundationPose 3D alignment",
        zoom=0.75,
    )


if __name__ == "__main__":
  main()
