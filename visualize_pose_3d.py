"""View saved object poses against a camera-frame scene cloud."""

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
  parser.add_argument("--all-poses", action="store_true", help="Show every pose in <debug-dir>/ob_in_cam")
  parser.add_argument("--scene-cloud", type=Path, help="Defaults to <debug-dir>/scene_complete.ply")
  parser.add_argument("--crop-padding", type=float, default=0.10, help="Scene crop padding around the posed model bounds, in meters")
  parser.add_argument("--no-crop", action="store_true", help="Show the complete original scene cloud")
  parser.add_argument("--axis-size", type=float, default=0.04, help="Camera and object coordinate frame size, in meters")
  parser.add_argument("--export-dir", type=Path, help="Also export the displayed cloud, posed meshes, and axes as PLY files")
  parser.add_argument("--no-window", action="store_true", help="Only export files; requires --export-dir")
  args = parser.parse_args()

  debug_dir = args.debug_dir.expanduser()
  mesh_file = args.mesh_file.expanduser()
  if args.pose and args.all_poses:
    parser.error("--pose and --all-poses cannot be used together")
  if args.pose:
    pose_files = [args.pose.expanduser()]
  else:
    pose_files = sorted((debug_dir / "ob_in_cam").glob("*.txt"))
    if not pose_files:
      raise FileNotFoundError(f"No pose matrices found under {debug_dir / 'ob_in_cam'}")
    if not args.all_poses:
      pose_files = pose_files[:1]
  scene_file = args.scene_cloud.expanduser() if args.scene_cloud else debug_dir / "scene_complete.ply"

  if args.no_window and args.export_dir is None:
    parser.error("--no-window requires --export-dir")
  if args.crop_padding < 0 or args.axis_size <= 0:
    parser.error("--crop-padding must be nonnegative and --axis-size must be positive")

  mesh = trimesh.load(mesh_file, force="mesh")
  scene = o3d.io.read_point_cloud(str(scene_file))
  if not scene.has_points():
    raise ValueError(f"No points found in {scene_file}")
  original_scene_points = len(scene.points)
  palette = np.asarray([
      [0.95, 0.25, 0.20], [0.15, 0.65, 0.95], [0.20, 0.80, 0.38],
      [0.78, 0.32, 0.88], [0.98, 0.72, 0.18], [0.12, 0.78, 0.75],
      [0.95, 0.42, 0.62], [0.58, 0.72, 0.18], [0.98, 0.52, 0.16],
      [0.44, 0.52, 0.95],
  ])
  posed_meshes = []
  object_axes = []
  all_vertices = []
  for index, pose_file in enumerate(pose_files):
    pose = np.loadtxt(pose_file).reshape(4, 4)
    vertices = trimesh.transform_points(np.asarray(mesh.vertices), pose)
    all_vertices.append(vertices)
    posed_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    posed_mesh.compute_vertex_normals()
    posed_mesh.paint_uniform_color(palette[index % len(palette)].tolist())
    posed_mesh.transform(pose)
    posed_meshes.append(posed_mesh)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size)
    axes.transform(pose)
    object_axes.append(axes)

    print(f"Pose: {pose_file}")
    print(f"Object origin in camera coordinates (m): {pose[:3, 3]}")
    print(f"Rotation determinant: {np.linalg.det(pose[:3, :3]):.6f}")

  if not args.no_crop:
    bounds = np.concatenate(all_vertices, axis=0)
    crop = o3d.geometry.AxisAlignedBoundingBox(
        bounds.min(axis=0) - args.crop_padding,
        bounds.max(axis=0) + args.crop_padding,
    )
    scene = scene.crop(crop)
    if not scene.has_points():
      raise ValueError(f"The scene crop is empty; increase --crop-padding (currently {args.crop_padding} m)")

  camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size)
  print(f"Scene points shown: {len(scene.points)} / {original_scene_points}")

  if args.export_dir:
    export_dir = args.export_dir.expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)
    cloud_name = "scene_complete.ply" if args.no_crop else "scene_crop.ply"
    o3d.io.write_point_cloud(str(export_dir / cloud_name), scene)
    o3d.io.write_triangle_mesh(str(export_dir / "camera_axes.ply"), camera_axes)
    for index, (posed_mesh, axes) in enumerate(zip(posed_meshes, object_axes), start=1):
      if len(posed_meshes) == 1:
        mesh_name = "posed_model.ply"
        axes_name = "object_axes.ply"
      else:
        mesh_name = f"posed_model_{index:02d}.ply"
        axes_name = f"object_axes_{index:02d}.ply"
      o3d.io.write_triangle_mesh(str(export_dir / mesh_name), posed_mesh)
      o3d.io.write_triangle_mesh(str(export_dir / axes_name), axes)
    print(f"Exported PLY files to {export_dir}")

  if not args.no_window:
    o3d.visualization.draw_geometries(
        [scene, *posed_meshes, camera_axes, *object_axes],
        window_name="FoundationPose 3D alignment",
        zoom=0.75,
    )


if __name__ == "__main__":
  main()
