"""Write H.264 MP4 videos that VS Code's built-in preview can decode."""

import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


class H264VideoWriter:
  """Stream uint8 BGR frames to FFmpeg without an intermediate video."""

  def __init__(self, path, fps, frame_size):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
      raise RuntimeError('H.264 video output requires ffmpeg on PATH (Ubuntu: sudo apt install ffmpeg)')
    if not math.isfinite(fps) or fps <= 0:
      raise ValueError(f'Video FPS must be positive and finite, got {fps}')
    width, height = frame_size
    if width <= 0 or height <= 0:
      raise ValueError(f'Invalid video frame size: {frame_size}')
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._frame_shape = (height, width, 3)
    self._closed = False
    self._stderr = tempfile.TemporaryFile()
    try:
      self._process = subprocess.Popen([
          ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
          '-f', 'rawvideo', '-pixel_format', 'bgr24',
          '-video_size', f'{width}x{height}', '-framerate', str(float(fps)),
          '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
          '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
          '-movflags', '+faststart', str(self.path),
      ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr)
    except Exception:
      self._stderr.close()
      raise

  def write(self, frame):
    if self._closed:
      raise RuntimeError('Cannot write to a closed video writer')
    if frame.shape != self._frame_shape or frame.dtype != np.uint8:
      raise ValueError(f'Expected uint8 BGR frame with shape {self._frame_shape}, got {frame.dtype} {frame.shape}')
    try:
      self._process.stdin.write(frame.tobytes())
    except BrokenPipeError:
      self.release()
      raise

  def release(self):
    if self._closed:
      return
    self._closed = True
    try:
      try:
        self._process.stdin.close()
      except BrokenPipeError:
        pass
      returncode = self._process.wait()
      self._stderr.seek(0)
      error = self._stderr.read().decode('utf-8', errors='replace').strip()
    finally:
      self._stderr.close()
    if returncode != 0:
      raise RuntimeError(f'FFmpeg could not encode {self.path} (exit {returncode}): {error}')
