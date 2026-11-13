"""Protect depth units, invalid pixels and stream choice without a camera SDK."""

from types import SimpleNamespace
import unittest

import numpy as np

from record_dataset import (DepthPostProcessor, FirstFrameDepthFusion, depth_to_millimetres,
                            select_depth_profile, select_video_profile)


class DepthFrame:
    def __init__(self, values, scale=0.1):
        self.values = np.asarray(values, dtype=np.uint16)
        self.scale = scale

    def get_data(self):
        return self.values.tobytes()

    def get_height(self):
        return self.values.shape[0]

    def get_width(self):
        return self.values.shape[1]

    def get_depth_scale(self):
        return self.scale

    def as_depth_frame(self):
        return self

    def update_data(self, data):
        self.values = np.frombuffer(data, dtype=np.uint16).copy().reshape(self.values.shape)


class Profile:
    def __init__(self, width, height, fps):
        self.width, self.height, self.fps = width, height, fps

    def as_video_stream_profile(self):
        return self

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_fps(self):
        return self.fps


class DepthQualityTest(unittest.TestCase):
    def test_first_frame_waits_for_samples_and_rejects_outliers(self):
        fusion = FirstFrameDepthFusion(samples=5)
        for value in (300, 301, 299, 2000):
            self.assertIsNone(fusion.add(np.full((2, 2), value, dtype=np.uint16)))
        result = fusion.add(np.full((2, 2), 300, dtype=np.uint16))
        np.testing.assert_array_equal(result, np.full((2, 2), 300, dtype=np.uint16))
        self.assertEqual(result.dtype, np.uint16)
        fusion.reset()
        self.assertIsNone(fusion.add(np.full((2, 2), 500, dtype=np.uint16)))

    def test_fusion_excludes_missing_depth_and_keeps_current_holes(self):
        fusion = FirstFrameDepthFusion(samples=5, min_valid_fraction=0.8)
        for values in ([300, 300, 0, 0], [301, 0, 0, 0],
                       [299, 300, 0, 0], [300, 300, 0, 0]):
            self.assertIsNone(fusion.add(np.array([values], dtype=np.uint16)))
        result = fusion.add(np.array([[0, 300, 0, 400]], dtype=np.uint16))
        # First pixel is missing in the latest RGB-D pair. Second has 4/5
        # valid measurements. Third has none and fourth only 1/5.
        np.testing.assert_array_equal(result, [[0, 300, 0, 0]])

    def test_device_units_are_scaled_once_and_invalid_values_stay_zero(self):
        frame = DepthFrame([[0, 1000, 2999, 3000, 5001, 65535]])
        result = depth_to_millimetres(frame, 100, 500)
        np.testing.assert_array_equal(result, [[0, 100, 300, 300, 0, 0]])
        self.assertEqual(result.dtype, np.uint16)
        # Range selection discards values; it must not clamp them onto the limit.
        np.testing.assert_array_equal(depth_to_millimetres(frame, 300, 500), [[0, 0, 0, 300, 0, 0]])

    def test_historical_depth_cannot_fill_current_frame_holes(self):
        processor = DepthPostProcessor(None, None, mode='none')
        class FillsAllHoles:
            def process(self, frame):
                values = frame.values.copy()
                values[values == 0] = 5000
                return DepthFrame(values)
        processor.filters = [FillsAllHoles()]
        source = DepthFrame([[1000, 0], [0, 3000]])
        result = processor.process(source)
        np.testing.assert_array_equal(result.values, source.values)

    def test_unavailable_filter_fails_with_actionable_message(self):
        device = SimpleNamespace(get_sensor=lambda _: SimpleNamespace(get_recommended_filters=lambda: []))
        ob = SimpleNamespace(OBSensorType=SimpleNamespace(DEPTH_SENSOR='depth'))
        with self.assertRaisesRegex(RuntimeError, '--depth-filter none'):
            DepthPostProcessor(device, ob, mode='spatial')

    def test_depth_fps_matches_colour_without_silent_resolution_fallback(self):
        profiles = [Profile(848, 530, 30), Profile(848, 530, 15), Profile(1280, 800, 30)]
        listing = SimpleNamespace(get_default_video_stream_profile=lambda: profiles[0],
                                  get_count=lambda: len(profiles),
                                  get_stream_profile_by_index=lambda i: profiles[i])
        self.assertIs(select_depth_profile(listing, None, None, 15), profiles[1])
        self.assertIs(select_depth_profile(listing, 1280, 800, 30), profiles[2])
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            select_depth_profile(listing, 1280, 800, 60)
        with self.assertRaisesRegex(ValueError, 'default-resolution'):
            select_depth_profile(listing, None, None, 60)

    def test_colour_profile_never_falls_back_to_another_resolution(self):
        profiles = [Profile(640, 480, 30), Profile(848, 530, 15), Profile(848, 530, 30)]
        listing = SimpleNamespace(get_count=lambda: len(profiles),
                                  get_stream_profile_by_index=lambda i: profiles[i])
        self.assertIs(select_video_profile(listing, 848, 530, 30), profiles[2])
        self.assertIs(select_video_profile(listing, 848, 530, 60), profiles[1])
        with self.assertRaisesRegex(ValueError, 'Colour resolution 1280x720 is unavailable'):
            select_video_profile(listing, 1280, 720, 30)


if __name__ == '__main__':
    unittest.main()
