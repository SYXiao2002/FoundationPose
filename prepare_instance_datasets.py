"""Split first-frame X-AnyLabeling contours into run_demo instance datasets.

Each matching shape becomes single_inbox<ID>_<label>, in JSON order from 1.
Only the first frame is annotated; run_demo tracks subsequent RGB/depth frames.
Contours exported as linestrip are closed and filled as segmentation regions.
"""

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


def prepare(annotation_path, output_dir=None, label='eu-plug'):
    annotation_path = Path(annotation_path).expanduser().resolve()
    raw = annotation_path.parent
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else raw.parent
    annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
    width, height = annotation['imageWidth'], annotation['imageHeight']
    rgb_files = sorted((raw / 'rgb').glob('*.png'))
    depth_files = sorted((raw / 'depth').glob('*.png'))
    if not rgb_files or [p.name for p in rgb_files] != [p.name for p in depth_files]:
        raise ValueError('RGB/depth frame lists must be nonempty and have matching names')
    first_id = rgb_files[0].stem
    first_rgb = cv2.imread(str(rgb_files[0]), cv2.IMREAD_COLOR)
    image = cv2.imread(str(raw / annotation['imagePath']), cv2.IMREAD_COLOR)
    if first_rgb is None or first_rgb.shape != (height, width, 3):
        raise ValueError('First RGB dimensions must match the annotation dimensions')
    if image is None or not np.array_equal(image, first_rgb):
        raise ValueError('Annotation PNG must be pixel-identical to the first RGB frame')
    camera_matrix = np.loadtxt(raw / 'cam_K.txt').reshape(3, 3)
    if not np.isfinite(camera_matrix).all():
        raise ValueError('Camera matrix contains nonfinite values')

    prepared = []
    for shape_index, shape in enumerate(annotation['shapes']):
        if shape['label'] != label:
            continue
        if shape.get('shape_type') not in ('polygon', 'linestrip'):
            raise ValueError(f'Shape {shape_index} must be a polygon or contour linestrip')
        points = np.asarray(shape['points'], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError(f'Shape {shape_index} has invalid polygon points')
        if ((points < 0).any() or (points[:, 0] >= width).any()
                or (points[:, 1] >= height).any()):
            raise ValueError(f'Shape {shape_index} has points outside the image')
        points = np.rint(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        if not mask.any():
            raise ValueError(f'Shape {shape_index} produces an empty mask')
        instance_id = len(prepared) + 1
        destination = output_dir / f'single_inbox{instance_id}_{label}'
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite {destination}')
        prepared.append((instance_id, shape_index, shape, destination, mask))
    if not prepared:
        raise ValueError(f'No shapes with label {label!r} found')

    source_meta = raw / 'capture_meta.json'
    metadata = json.loads(source_meta.read_text(encoding='utf-8')) if source_meta.exists() else {}
    # The generated masks come from the annotation, so discard the capture's SAM settings.
    metadata = {k: v for k, v in metadata.items() if not k.startswith('sam_') and k != 'last_sam_score'}
    metadata.update(width=width, height=height, K=camera_matrix.tolist(),
                    frames_saved=len(rgb_files),
                    mask_storage=f'uint8 binary 0/255 rasterized from {annotation_path.name}',
                    mask_frame=first_id)

    for instance_id, shape_index, shape, destination, mask in prepared:
        for subdir, files in (('rgb', rgb_files), ('depth', depth_files)):
            (destination / subdir).mkdir(parents=True)
            for source in files:
                shutil.copy2(source, destination / subdir / source.name)
        (destination / 'masks').mkdir()
        if not cv2.imwrite(str(destination / 'masks' / f'{first_id}.png'), mask):
            raise OSError(f'Could not write mask for {destination}')
        shutil.copy2(raw / 'cam_K.txt', destination / 'cam_K.txt')
        instance_meta = dict(metadata, instance_id=instance_id, label=label,
                             annotation_shape_index=shape_index,
                             annotation_shape_type=shape['shape_type'],
                             mask_pixel_count=int(cv2.countNonZero(mask)))
        (destination / 'capture_meta.json').write_text(
            json.dumps(instance_meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'{destination.name}: {len(rgb_files)} frames, {instance_meta["mask_pixel_count"]} mask pixels')
    print(f'Created {len(prepared)} datasets under {output_dir}; mask frame: {first_id}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', type=Path, required=True, help='JSON beside rgb/, depth/, cam_K.txt and the annotated PNG')
    parser.add_argument('--output-dir', type=Path, help='Destination parent (default: parent of raw/)')
    parser.add_argument('--label', default='eu-plug', help='Label to export (default: eu-plug)')
    args = parser.parse_args()
    prepare(args.annotation, args.output_dir, args.label)


if __name__ == '__main__':
    main()
