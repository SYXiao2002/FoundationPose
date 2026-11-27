import numpy as np
from PIL import Image

from pose_service.wapr_score_vis import write_wapr_score_image


def test_wapr_score_image_writes_top_ten_rows_with_each_score(tmp_path):
    count = 12
    candidates = {
        "candidate_ids": np.arange(count, dtype=np.int32),
        "within_group": np.linspace(12, 1, count, dtype=np.float32),
        "between_group": np.linspace(1, 12, count, dtype=np.float32),
        "score_A": np.zeros((count, 6, 8, 8), dtype=np.float32),
        "score_B": np.ones((count, 6, 8, 8), dtype=np.float32),
    }
    target = tmp_path / "debug" / "vis_score.png"

    written = write_wapr_score_image(candidates, .445, target, top_k=10)

    assert written == 10
    with Image.open(target) as image:
        assert image.size == (640, 1045)
        pixels = np.asarray(image)
    # Candidate score labels are rendered into the first and last score rows.
    assert np.count_nonzero(pixels[:100] != 0) > 0
    assert np.count_nonzero(pixels[-100:] != 0) > 0
