"""Check camera-frame geometry and instance identity without inference or a GPU."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from visualize_multi_instance_web import ViewerData


class FirstFrameViewerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / 'results'
        self.scene = self.root / 'scene'
        for child in ['rgb', 'depth']:
            (self.scene / child).mkdir(parents=True)
        self.bgr = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
        self.depth = np.array([[1000, 1000, 2000], [1000, 1000, 1000], [0, 1000, 500]], dtype=np.uint16)
        cv2.imwrite(str(self.scene / 'rgb/0001.png'), self.bgr)
        cv2.imwrite(str(self.scene / 'depth/0001.png'), self.depth)
        np.savetxt(self.scene / 'cam_K.txt', [[2, 0, 1], [0, 4, 1], [0, 0, 1]])
        self.mesh = self.root / 'mesh.obj'
        self.mesh.write_text('v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nf 1 2 3\n')
        self.make_instance(1)

    def make_instance(self, index):
        directory = self.results / f'instance{index}'
        (directory / 'ob_in_cam').mkdir(parents=True)
        pose = np.eye(4)
        pose[:3, 3] = [0.01, -0.02, 0.5]
        np.savetxt(directory / 'ob_in_cam/0001.txt', pose)
        # A corrupt later-frame pose must never be read by a first-frame viewer.
        (directory / 'ob_in_cam/0002.txt').write_text('later frame is deliberately invalid')
        mask = np.zeros((3, 3), dtype=np.uint8)
        mask[2, 2] = 255
        cv2.imwrite(str(directory / 'ob_mask.png'), mask)
        cv2.imwrite(str(directory / 'color.png'), self.bgr)
        return directory

    def data(self):
        return ViewerData(self.results, self.scene, self.mesh, 2)

    def test_projection_preserves_meters_stride_and_rgb(self):
        binary = self.data().scene_cloud()
        count = int(np.frombuffer(binary, dtype='<u4', count=1)[0])
        self.assertEqual(count, 3)
        self.assertEqual(len(binary), 4 + count * 15)
        xyz = np.frombuffer(binary, dtype='<f4', count=count * 3, offset=4).reshape(-1, 3)
        np.testing.assert_allclose(xyz, [[-0.5, -0.25, 1], [1, -0.5, 2], [0.25, 0.125, 0.5]])
        colors = np.frombuffer(binary, dtype=np.uint8, offset=4 + count * 12).reshape(-1, 3)
        np.testing.assert_array_equal(colors, self.bgr[[0, 0, 2], [0, 2, 2]][:, ::-1])

    def test_reads_only_first_pose_without_recentering(self):
        info = self.data().info()
        self.assertEqual(info['frame'], '0001')
        np.testing.assert_allclose(info['instances'][0]['origin'], [0.01, -0.02, 0.5])
        self.assertEqual(info['image_size'], [3, 3])

    def test_mask_ids_above_255_round_trip_without_color_confusion(self):
        self.make_instance(257)
        encoded = self.data().mask_map()
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = bgr[:, :, ::-1].astype(np.uint32)
        ids = rgb[:, :, 0] + (rgb[:, :, 1] << 8) + (rgb[:, :, 2] << 16)
        self.assertEqual(int(ids[2, 2]), 257)
        self.assertEqual(int(ids[0, 0]), 0)

    def test_rejects_different_first_frames_and_wrong_rgb(self):
        directory = self.make_instance(2)
        (directory / 'ob_in_cam/0001.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'different first frames'):
            self.data()
        np.savetxt(directory / 'ob_in_cam/0001.txt', np.eye(4))
        cv2.imwrite(str(directory / 'color.png'), np.zeros_like(self.bgr))
        with self.assertRaisesRegex(ValueError, 'RGB scene does not match'):
            self.data()


if __name__ == '__main__':
    unittest.main()
