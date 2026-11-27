"""Lossless highest-score row extraction; no GPU dependency."""
from io import BytesIO
from pathlib import Path
from PIL import Image
from score_vis_layout import SCORE_ROW_HEIGHT, SCORE_ROW_MARGIN


def score_preview(source: Path, target: Path):
    with Image.open(source) as image:
        if image.height < SCORE_ROW_HEIGHT:
            raise ValueError("scorer image is shorter than one row")
        image.crop((0, 0, image.width, SCORE_ROW_HEIGHT)).save(target)


def score_candidates_preview(source: Path, *, top_k=30) -> bytes:
    """Read up to top_k complete, already sorted rows without changing an archive."""
    if type(top_k) is not int or not 1 <= top_k <= 30:
        raise ValueError("top_k must be between 1 and 30")
    with Image.open(source) as image:
        stride = SCORE_ROW_HEIGHT + SCORE_ROW_MARGIN
        count = min(top_k, (image.height + SCORE_ROW_MARGIN) // stride)
        if count < 1:
            raise ValueError("scorer image is shorter than one row")
        height = count * stride - SCORE_ROW_MARGIN
        output = BytesIO()
        image.crop((0, 0, image.width, height)).save(output, format="PNG")
        return output.getvalue()


def rgb_preview(source: Path, target: Path):
    with Image.open(source) as image:
        image.thumbnail((480, 360))
        image.convert("RGB").save(target, quality=88)
