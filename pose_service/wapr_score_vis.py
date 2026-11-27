"""Render WAPR's highest-ranked WBPS candidate score rows."""

from pathlib import Path

import numpy as np


def _rgb_panel(chw, height=100, width=160):
    import cv2

    rgb = np.asarray(chw, dtype=np.float32).transpose(1, 2, 0)
    if rgb.size and float(np.nanmax(rgb)) <= 1.5:
        rgb = rgb * 255.0
    rgb = np.clip(np.nan_to_num(rgb), 0, 255).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (width, height), interpolation=cv2.INTER_LINEAR)


def _depth_panel(depth, height=100, width=160):
    import cv2

    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        normalized[valid] = (np.clip((depth[valid] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.resize(color, (width, height), interpolation=cv2.INTER_NEAREST)


def write_wapr_score_image(candidates, score_6d, output_path, *, top_k=10):
    """Write rendered/observed RGB-D pairs in descending WAPR within_group order.

    WAPR's ``within_group`` ranks candidates. ``between_group`` is a second
    candidate diagnostic; ``score_6d`` is group-level and therefore repeats
    for every row instead of being presented as a per-candidate score.
    """
    import cv2

    ids = np.asarray(candidates["candidate_ids"]).reshape(-1)
    within = np.asarray(candidates["within_group"], dtype=np.float32).reshape(-1)
    between = np.asarray(candidates["between_group"], dtype=np.float32).reshape(-1)
    score_a = np.asarray(candidates["score_A"], dtype=np.float32)
    score_b = np.asarray(candidates["score_B"], dtype=np.float32)
    count = min(int(top_k), len(ids), len(within), len(between), len(score_a), len(score_b))
    if count < 1 or score_a.ndim != 4 or score_b.ndim != 4:
        raise ValueError("WAPR score candidates are empty or malformed")
    if (score_a.shape[1] < 6 or score_b.shape[1] < 6
            or not np.isfinite(within[:count]).all() or not np.isfinite(between[:count]).all()
            or not np.isfinite(score_6d)):
        raise ValueError("WAPR score candidates contain invalid scores or crop channels")

    rows = []
    for index in range(count):
        row = np.concatenate((
            _rgb_panel(score_a[index, :3]), _depth_panel(score_a[index, 5]),
            _rgb_panel(score_b[index, :3]), _depth_panel(score_b[index, 5]),
        ), axis=1)
        label = "id=%d within=%.3f between=%.3f score_6d=%.4f" % (
            int(ids[index]), float(within[index]), float(between[index]), float(score_6d))
        cv2.putText(row, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .34,
                    (0, 255, 0), 1, cv2.LINE_AA)
        for x, text in ((4, "A rendered"), (324, "B observed")):
            cv2.putText(row, text, (x, 96), cv2.FONT_HERSHEY_SIMPLEX, .29,
                        (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
        if index + 1 < count:
            rows.append(np.full((5, row.shape[1], 3), 255, dtype=np.uint8))
    image = np.concatenate(rows, axis=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError("could not write WAPR vis_score image")
    return count
