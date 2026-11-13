"""Exercise batch subprocesses, resumption and failures without GPU inference."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from run_demo_batch import main


FAKE_DEMO = '''
import argparse
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--test_scene_dir', type=Path)
parser.add_argument('--debug_dir', type=Path)
parser.add_argument('--debug', type=int)
args, _ = parser.parse_known_args()
assert 'DISPLAY' not in os.environ and 'WAYLAND_DISPLAY' not in os.environ
print('invoked ' + args.test_scene_dir.name, flush=True)
if args.debug_dir.name == os.environ.get('BATCH_TEST_FAIL'):
    print('deliberate test failure', file=sys.stderr)
    sys.exit(7)
for frame in sorted((args.test_scene_dir / 'rgb').glob('*.png')):
    pose = args.debug_dir / 'ob_in_cam' / (frame.stem + '.txt')
    pose.parent.mkdir(parents=True, exist_ok=True)
    pose.write_text('pose data')
    if args.debug >= 2:
        vis = args.debug_dir / 'track_vis' / frame.name
        vis.parent.mkdir(parents=True, exist_ok=True)
        vis.write_bytes(b'frame data')
if args.debug >= 2:
    (args.debug_dir / 'pose_video.mp4').write_bytes(b'video data')
'''


class BatchDemoTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='batch demo ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scene_root = self.root / 'scene inputs'
        self.output = self.root / 'output results'
        self.mesh = self.root / 'mesh.obj'
        self.mesh.write_text('test mesh')
        self.demo = self.root / 'fake demo.py'
        self.demo.write_text(FAKE_DEMO)
        # Discovery must use numeric order rather than 1, 10, 2.
        for instance_id in (10, 2, 1):
            scene = self.scene_root / f'single_inbox{instance_id}_eu-plug'
            for kind in ('rgb', 'depth', 'masks'):
                (scene / kind).mkdir(parents=True)
            for frame in ('0001', '0002'):
                for kind in ('rgb', 'depth'):
                    (scene / kind / f'{frame}.png').write_bytes(b'input')
            (scene / 'masks/0001.png').write_bytes(b'mask')
            (scene / 'cam_K.txt').write_text('intrinsics')

    def run_batch(self, *extra):
        arguments = ['--scene-root', str(self.scene_root), '--mesh-file', str(self.mesh),
                     '--output-dir', str(self.output), *extra]
        with contextlib.redirect_stdout(io.StringIO()), mock.patch('run_demo_batch.shutil.which', return_value='/usr/bin/ffmpeg'):
            return main(arguments, demo_script=self.demo)

    def summary(self):
        return json.loads((self.output / 'batch_summary.json').read_text())['runs']

    def test_numeric_order_logs_and_resume(self):
        with mock.patch.dict(os.environ, {'DISPLAY': ':99', 'WAYLAND_DISPLAY': 'wayland-99'}):
            self.assertEqual(self.run_batch(), 0)
        records = self.summary()
        self.assertEqual([record['instance_id'] for record in records], [1, 2, 10])
        self.assertEqual([record['status'] for record in records], ['success'] * 3)
        log = self.output / 'logs/instance1.log'
        self.assertIn('invoked single_inbox1_eu-plug', log.read_text())
        before = log.stat().st_mtime_ns
        self.assertEqual(self.run_batch(), 0)
        self.assertEqual([record['status'] for record in self.summary()], ['skipped'] * 3)
        self.assertEqual(log.stat().st_mtime_ns, before)
        # An interrupted video is incomplete even if poses were all written.
        (self.output / 'instance2/pose_video.mp4').write_bytes(b'')
        self.assertEqual(self.run_batch(), 0)
        self.assertEqual([record['status'] for record in self.summary()], ['skipped', 'success', 'skipped'])

    def test_failure_continues_and_returns_nonzero(self):
        with mock.patch.dict(os.environ, {'BATCH_TEST_FAIL': 'instance2'}):
            self.assertEqual(self.run_batch(), 1)
        records = self.summary()
        self.assertEqual([record['status'] for record in records], ['success', 'failed', 'success'])
        self.assertEqual(records[1]['returncode'], 7)
        self.assertIn('deliberate test failure', (self.output / 'logs/instance2.log').read_text())

    def test_dry_run_has_no_output_and_selection_preserves_ids(self):
        self.assertEqual(self.run_batch('--dry-run', '--instances', '10', '2'), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.run_batch('--instances', '10', '2', '--video-fps', '20'), 0)
        self.assertEqual([record['instance_id'] for record in self.summary()], [2, 10])
        self.assertFalse((self.output / 'instance1').exists())
        self.assertIn('--video_fps', self.summary()[0]['command'])
        self.assertEqual(self.run_batch('--instances', '2', '--overwrite'), 0)
        self.assertEqual(self.summary()[0]['status'], 'success')

    def test_preflight_rejects_missing_depth_before_running(self):
        (self.scene_root / 'single_inbox10_eu-plug/depth/0002.png').unlink()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.run_batch()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_rejects_output_symlink_pointing_to_scene(self):
        self.output.mkdir()
        source = self.scene_root / 'single_inbox1_eu-plug'
        (self.output / 'instance1').symlink_to(source, target_is_directory=True)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.run_batch('--instances', '1')
        self.assertEqual(error.exception.code, 2)
        self.assertTrue((source / 'rgb/0001.png').is_file())
        self.assertFalse((self.output / 'logs').exists())


if __name__ == '__main__':
    unittest.main()
