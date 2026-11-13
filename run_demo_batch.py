"""Run run_demo.py sequentially for prepared single_inbox<ID>_<label> datasets."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def discover_scenes(scene_root, pattern, instance_ids):
    scenes = []
    seen = set()
    for scene in scene_root.glob(pattern):
        if not scene.is_dir():
            continue
        match = re.fullmatch(r'single_inbox(\d+)_.+', scene.name)
        if not match:
            raise ValueError(f'Expected single_inbox<ID>_<label> directory: {scene}')
        instance_id = int(match[1])
        if instance_ids is not None and instance_id not in instance_ids:
            continue
        if instance_id in seen:
            raise ValueError(f'Duplicate instance ID {instance_id}; narrow --pattern to one label')
        seen.add(instance_id)
        frames = sorted(path.stem for path in (scene / 'rgb').glob('*.png'))
        depth_frames = sorted(path.stem for path in (scene / 'depth').glob('*.png'))
        if not frames or frames != depth_frames:
            raise ValueError(f'RGB/depth frame lists must be nonempty and match: {scene}')
        for required in (scene / 'cam_K.txt', scene / 'masks' / f'{frames[0]}.png'):
            if not required.is_file():
                raise FileNotFoundError(f'Missing input: {required}')
        scenes.append((instance_id, scene.resolve(), frames))
    if instance_ids is not None and (missing := set(instance_ids) - seen):
        raise ValueError(f'Instance IDs not found: {sorted(missing)}')
    if not scenes:
        raise ValueError(f'No instance datasets matching {pattern!r} under {scene_root}')
    return sorted(scenes)


def results_complete(output, frames, debug):
    required = [output / 'ob_in_cam' / f'{frame}.txt' for frame in frames]
    if debug >= 2:
        required += [output / 'pose_video.mp4']
        required += [output / 'track_vis' / f'{frame}.png' for frame in frames]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def main(argv=None, demo_script=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-root', type=Path, required=True, help='Parent containing single_inbox<ID>_<label> directories')
    parser.add_argument('--mesh-file', type=Path, required=True, help='Mesh shared by all selected instances')
    parser.add_argument('--output-dir', type=Path, required=True, help='Results parent; each instance gets instance<ID>/')
    parser.add_argument('--pattern', default='single_inbox*_eu-plug', help='Scene directory glob (default: single_inbox*_eu-plug)')
    parser.add_argument('--instances', type=int, nargs='+', help='Only run these instance IDs (default: all)')
    parser.add_argument('--debug', type=int, default=2, help='run_demo debug level (default: 2, saves PNGs and MP4)')
    parser.add_argument('--est-refine-iter', type=int, default=5)
    parser.add_argument('--track-refine-iter', type=int, default=2)
    parser.add_argument('--video-fps', type=float, help='Override the video FPS; otherwise infer from frame timestamps')
    parser.add_argument('--overwrite', action='store_true', help='Rerun even when output files are complete')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without creating files or running inference')
    args = parser.parse_args(argv)

    scene_root = args.scene_root.expanduser().resolve()
    mesh = args.mesh_file.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    demo_script = Path(demo_script).resolve() if demo_script is not None else ROOT / 'run_demo.py'
    if not scene_root.is_dir():
        parser.error(f'Scene root does not exist: {scene_root}')
    if not mesh.is_file():
        parser.error(f'Mesh does not exist: {mesh}')
    if args.debug < 0 or min(args.est_refine_iter, args.track_refine_iter) < 1:
        parser.error('Debug level must be nonnegative and refine iterations must be positive')
    if args.video_fps is not None and not 0 < args.video_fps < float('inf'):
        parser.error('--video-fps must be positive and finite')
    try:
        scenes = discover_scenes(scene_root, args.pattern, args.instances)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    # run_demo clears its debug directory. Keep every target separate from all inputs.
    for instance_id, _, _ in scenes:
        target = (output_root / f'instance{instance_id}').resolve()
        for source in [mesh, ROOT, *(scene for _, scene, _ in scenes)]:
            if target == source or target in source.parents or (source != ROOT and source in target.parents):
                parser.error(f'Output would overlap an input: {target} and {source}')
    if not args.dry_run and args.debug >= 2 and shutil.which('ffmpeg') is None:
        parser.error('MP4 output requires ffmpeg on PATH (Ubuntu: sudo apt install ffmpeg)')

    environment = os.environ.copy()
    environment.pop('DISPLAY', None)
    environment.pop('WAYLAND_DISPLAY', None)
    report = {'scene_root': str(scene_root), 'mesh_file': str(mesh), 'runs': []}
    summary = output_root / 'batch_summary.json'
    if not args.dry_run:
        (output_root / 'logs').mkdir(parents=True, exist_ok=True)

    for index, (instance_id, scene, frames) in enumerate(scenes, 1):
        output = output_root / f'instance{instance_id}'
        log = output_root / 'logs' / f'instance{instance_id}.log'
        command = [sys.executable, '-u', str(demo_script),
                   '--mesh_file', str(mesh), '--test_scene_dir', str(scene),
                   '--debug', str(args.debug), '--debug_dir', str(output),
                   '--est_refine_iter', str(args.est_refine_iter),
                   '--track_refine_iter', str(args.track_refine_iter)]
        if args.video_fps is not None:
            command.extend(['--video_fps', str(args.video_fps)])
        record = {'instance_id': instance_id, 'scene_dir': str(scene),
                  'output_dir': str(output), 'frames': len(frames), 'command': command}
        if not args.overwrite and results_complete(output, frames, args.debug):
            record['status'] = 'skipped'
            print(f'[{index}/{len(scenes)}] instance{instance_id}: skipped (complete outputs)', flush=True)
        elif args.dry_run:
            print(f'[{index}/{len(scenes)}] {shlex.join(command)}', flush=True)
            continue
        else:
            print(f'[{index}/{len(scenes)}] instance{instance_id}: running; log: {log}', flush=True)
            start = time.monotonic()
            record['log_file'] = str(log)
            try:
                with log.open('w', encoding='utf-8') as stream:
                    result = subprocess.run(command, cwd=ROOT, env=environment,
                                            stdout=stream, stderr=subprocess.STDOUT)
                record['returncode'] = result.returncode
                complete = result.returncode == 0 and results_complete(output, frames, args.debug)
                record['status'] = 'success' if complete else 'failed'
                if result.returncode == 0 and not complete:
                    record['error'] = 'run_demo exited successfully but expected output files are missing or empty'
            except KeyboardInterrupt:
                record['status'] = 'interrupted'
            except OSError as error:
                record.update(status='failed', error=str(error))
            record['elapsed_seconds'] = round(time.monotonic() - start, 2)
            print(f'  {record["status"]} ({record["elapsed_seconds"]:.1f}s)', flush=True)
        report['runs'].append(record)
        if not args.dry_run:
            summary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        if record['status'] == 'interrupted':
            return 130

    if args.dry_run:
        return 0
    counts = {status: sum(run['status'] == status for run in report['runs'])
              for status in ('success', 'skipped', 'failed')}
    print(f'Success: {counts["success"]}; skipped: {counts["skipped"]}; failed: {counts["failed"]}')
    print(f'Summary: {summary}')
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
