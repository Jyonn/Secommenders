#!/usr/bin/env python3
import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = ROOT / 'artifacts'
DATASET_STAGES = (
    'formatted',
    'processed',
    'embedded',
    'quantized',
    'clustered',
    'compiled',
    'trained',
)


@dataclass
class DatasetArtifacts:
    dataset: str
    stage_paths: dict[str, Path] = field(default_factory=dict)
    stage_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def total_size(self) -> int:
        return sum(self.stage_sizes.values())


def parse_args():
    parser = argparse.ArgumentParser(
        description='List or delete dataset-scoped artifact directories under artifacts/<stage>/<data>.',
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help='Artifact root directory. Default: artifacts',
    )
    parser.add_argument(
        '--data',
        '--dataset',
        dest='data',
        help='Dataset name to inspect or delete, e.g. mind, rv10, recifadsall.',
    )
    parser.add_argument(
        '--stages',
        help='Comma-separated stage filter. Default: formatted,processed,embedded,quantized,clustered,compiled,trained.',
    )
    parser.add_argument(
        '--all-stages',
        action='store_true',
        help='Use every direct subdirectory under --root as a stage instead of the default dataset artifact stages.',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually delete directories for --data. Without this flag the script only prints a dry-run plan.',
    )
    parser.add_argument(
        '--no-size',
        action='store_true',
        help='Skip recursive size calculation for faster listing on very large artifact trees.',
    )
    return parser.parse_args()


def parse_stage_filter(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    stages = tuple(stage.strip() for stage in raw.split(',') if stage.strip())
    if not stages:
        raise ValueError('--stages was provided but no valid stage names were found')
    return stages


def human_size(size: int | None) -> str:
    if size is None:
        return '-'
    value = float(size)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB', 'PB'):
        if value < 1024.0 or unit == 'PB':
            if unit == 'B':
                return f'{int(value)}B'
            if value >= 100:
                return f'{value:.0f}{unit}'
            if value >= 10:
                return f'{value:.1f}{unit}'
            return f'{value:.2f}{unit}'
        value /= 1024.0
    return f'{size}B'


def dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob('*'):
        try:
            if item.is_file() or item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def stage_names(root: Path, all_stages: bool, stage_filter: tuple[str, ...] | None) -> tuple[str, ...]:
    if stage_filter is not None:
        return stage_filter
    if all_stages:
        if not root.exists():
            return ()
        return tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))
    return DATASET_STAGES


def ensure_safe_dataset_dir(root: Path, path: Path) -> None:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        relative = path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f'unsafe path outside artifact root: {path}') from exc
    if len(relative.parts) != 2:
        raise ValueError(f'unsafe dataset artifact path, expected artifacts/<stage>/<data>: {path}')
    if any(part in {'', '.', '..'} for part in relative.parts):
        raise ValueError(f'unsafe path component in artifact path: {path}')


def collect_artifacts(root: Path, stages: tuple[str, ...], with_size: bool) -> dict[str, DatasetArtifacts]:
    grouped: dict[str, DatasetArtifacts] = {}
    for stage in stages:
        stage_dir = root / stage
        if not stage_dir.is_dir():
            continue
        for data_dir in sorted(stage_dir.iterdir()):
            if not data_dir.is_dir():
                continue
            ensure_safe_dataset_dir(root, data_dir)
            dataset = data_dir.name
            entry = grouped.setdefault(dataset, DatasetArtifacts(dataset=dataset))
            entry.stage_paths[stage] = data_dir
            entry.stage_sizes[stage] = dir_size(data_dir) if with_size else 0
    return grouped


def format_table(rows: list[list[str]]) -> str:
    if not rows:
        return ''
    widths = [0] * len(rows[0])
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = []
    for row_index, row in enumerate(rows):
        lines.append('  '.join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip())
        if row_index == 0:
            lines.append('  '.join('-' * width for width in widths).rstrip())
    return '\n'.join(lines)


def print_summary(root: Path, stages: tuple[str, ...], grouped: dict[str, DatasetArtifacts], with_size: bool) -> None:
    print(f'artifact datasets root={root}')
    if not grouped:
        print('no dataset artifact directories found')
        return
    active_stages = [stage for stage in stages if any(stage in entry.stage_paths for entry in grouped.values())]
    header = ['dataset', 'total', *active_stages]
    rows = [header]
    for dataset in sorted(grouped):
        entry = grouped[dataset]
        total = human_size(entry.total_size) if with_size else '-'
        row = [dataset, total]
        for stage in active_stages:
            row.append(human_size(entry.stage_sizes[stage]) if stage in entry.stage_paths and with_size else '-')
        rows.append(row)
    print(format_table(rows))


def print_delete_plan(data: str, entry: DatasetArtifacts | None, with_size: bool) -> None:
    if entry is None:
        print(f'no artifacts found for dataset={data}')
        return
    print(f'dataset delete plan data={data} total={human_size(entry.total_size) if with_size else "-"}')
    rows = [['stage', 'size', 'path']]
    for stage in sorted(entry.stage_paths):
        size = human_size(entry.stage_sizes[stage]) if with_size else '-'
        rows.append([stage, size, str(entry.stage_paths[stage])])
    print(format_table(rows))


def delete_dataset_artifacts(root: Path, entry: DatasetArtifacts) -> None:
    for stage in sorted(entry.stage_paths):
        path = entry.stage_paths[stage]
        ensure_safe_dataset_dir(root, path)
        shutil.rmtree(path)
        print(f'deleted {stage}: {path}')


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    try:
        stages = stage_names(root, args.all_stages, parse_stage_filter(args.stages))
        grouped = collect_artifacts(root, stages, with_size=not args.no_size)
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    if args.data:
        entry = grouped.get(args.data)
        print_delete_plan(args.data, entry, with_size=not args.no_size)
        if entry is None:
            return 1
        if not args.apply:
            print('dry-run only; rerun with --apply to delete these directories')
            return 0
        delete_dataset_artifacts(root, entry)
        return 0

    print_summary(root, stages, grouped, with_size=not args.no_size)
    if args.apply:
        print('warning: --apply is ignored unless --data is provided')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
