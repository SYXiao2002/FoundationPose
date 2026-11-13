import unittest

import numpy as np
import trimesh

from pose_geometry_validation import MeshDepthRenderer, ValidationConfig, evaluate_geometry


class GeometryValidationTest(unittest.TestCase):
    def test_overlap_depth_and_occlusion_are_distinct(self):
        mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        depth = np.array([[1000, 1010, 900], [0, 1000, 0]], dtype=np.uint16)
        render = np.array([[1.0, 1.0, 1.0], [1.0, 0, 0]])
        config = ValidationConfig(min_mask_depth_points=1, min_overlap_depth_points=1)
        result = evaluate_geometry(mask, depth, render, config)
        m = result['metrics']
        self.assertEqual(m['mask_valid_depth_pixels'], 2)
        self.assertEqual(m['overlap_valid_depth_pixels'], 2)
        self.assertAlmostEqual(m['silhouette_iou'], 0.75)
        self.assertAlmostEqual(m['depth_median_mm'], 5.0)
        self.assertAlmostEqual(m['occluded_fraction_of_render_valid_depth'], 1 / 3)
        self.assertAlmostEqual(m['render_unknown_depth_fraction'], 1 / 4)
        self.assertIn('high_occlusion', result['warnings'])

    def test_missing_depth_does_not_become_a_perfect_or_bad_depth_score(self):
        result = evaluate_geometry(np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2)))
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertIsNone(result['metrics']['depth_median_mm'])
        self.assertIsNone(result['metrics']['depth_inlier_fraction'])
        self.assertIn('few_overlap_depth_points', result['warnings'])
        self.assertNotIn('high_depth_median', result['warnings'])

    def test_low_overlap_warns_even_when_depth_matches(self):
        mask = np.ones((20, 20))
        render = np.zeros((20, 20))
        render[:5, :5] = 1
        result = evaluate_geometry(mask, np.full((20, 20), 1000), render,
                                   ValidationConfig(min_mask_depth_points=1, min_overlap_depth_points=1))
        self.assertEqual(result['status'], 'suspect')
        self.assertEqual(result['metrics']['depth_median_mm'], 0)
        self.assertIn('low_iou', result['warnings'])
        self.assertIn('low_mask_coverage', result['warnings'])

    def test_renderer_uses_camera_z_and_original_mesh_frame(self):
        mesh = trimesh.Trimesh(vertices=[[-0.1, -0.1, 0], [0.1, -0.1, 0], [0, 0.1, 0]],
                               faces=[[0, 1, 2]], process=False)
        renderer = MeshDepthRenderer(mesh, np.array([[100, 0, 16], [0, 100, 16], [0, 0, 1]]), 32, 32)
        try:
            pose = np.eye(4)
            pose[2, 3] = 1
            depth = renderer.render(pose)
        finally:
            renderer.close()
        self.assertGreater(np.count_nonzero(depth), 100)
        self.assertAlmostEqual(float(depth[16, 16]), 1, places=4)
        self.assertEqual(depth[0, 0], 0)


if __name__ == '__main__':
    unittest.main()
