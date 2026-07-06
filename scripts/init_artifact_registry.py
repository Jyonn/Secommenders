import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.artifact_identity import (  # noqa: E402
    TRAINED_INDEX_NAME,
    TRAINED_PHASES,
    TrainedArtifactRegistryConflict,
    canonical_trained_run_dir,
    GENERIC_SPEC_VERSIONS,
    generic_artifact_identity as identity_generic_artifact_identity,
    canonical_generic_artifact_dir,
    generic_index_path as identity_generic_index_path,
    generic_signature_from_meta as identity_generic_signature_from_meta,
    legacy_signature_from_folder,
    load_generic_index as identity_load_generic_index,
    load_trained_index,
    migrate_train_config_dict,
    register_generic_artifact as identity_register_generic_artifact,
    register_trained_artifact,
    save_generic_index as identity_save_generic_index,
    save_trained_index,
    trained_artifact_identity,
    trained_mode,
    trained_seed,
    trained_signature_from_config,
)


GENERIC_STAGES = {'clustered', 'compiled', 'quantized'}
REGISTRY_STAGES = {'trained', *GENERIC_STAGES}
GENERIC_SCHEMA_VERSIONS = GENERIC_SPEC_VERSIONS
PATH_KEY_SUFFIXES = ('_path', '_dir')
PATH_KEYS = {
    'path',
    'processed_dir',
    'processed_items_path',
    'embedding_path',
    'embedding_meta_path',
    'trainer_output_dir',
    'export_dir',
    'checkpoint_dir',
    'codebook_indices_path',
    'quantized_latents_path',
    'item_ids_path',
    'codebooks_path',
    'binary_bits_path',
    'indexer_dir',
}
VOLATILE_SIGN_KEYS = {'seed', 'device'}


ANSI = {
    'reset': '\033[0m',
    'bold': '\033[1m',
    'dim': '\033[2m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'gray': '\033[90m',
}


def use_color():
    if 'NO_COLOR' in os.environ:
        return False
    return sys.stdout.isatty() or os.environ.get('FORCE_COLOR') == '1'


def paint(text: str, *styles: str):
    if not use_color():
        return str(text)
    prefix = ''.join(ANSI[style] for style in styles if style in ANSI)
    return f'{prefix}{text}{ANSI["reset"]}' if prefix else str(text)


def badge(text: str, color: str):
    return paint(f'[{text}]', 'bold', color)


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid json: {exc}') from exc


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def dedupe(values):
    result = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def registry_index_path(stage: str, data: str, root: Path):
    return identity_generic_index_path(stage, data, root=root)


def load_registry_index(stage: str, data: str, root: Path):
    return identity_load_generic_index(stage, data, root=root)


def save_registry_index(stage: str, data: str, index: dict, root: Path):
    identity_save_generic_index(stage, data, index, root=root)


def strip_path_values(value):
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in PATH_KEYS or key_text in VOLATILE_SIGN_KEYS or key_text.endswith(PATH_KEY_SUFFIXES):
                continue
            clean[key_text] = strip_path_values(item)
        return clean
    if isinstance(value, list):
        return [strip_path_values(item) for item in value]
    return value


def generic_data_from_meta(stage: str, meta: dict):
    value = meta.get('data') or meta.get('dataset')
    if not value:
        raise ValueError(f'{stage} meta missing data/dataset')
    return str(value).lower()


def generic_stage_payload(stage: str, meta: dict):
    # Runtime artifact identity now lives in utils.artifact_identity.  Keep this
    # helper only for backward-compatible callers inside this script.
    if stage == 'compiled':
        if not isinstance(meta.get('config'), dict):
            raise ValueError('compiled meta missing config')
        return {
            'version': meta.get('version'),
            'prepare_id': meta.get('prepare_id'),
            'config': strip_path_values(meta.get('config')),
            'model_kind': meta.get('model_kind'),
            'model_max_length': meta.get('model_max_length'),
        }
    if stage == 'clustered':
        return {
            'version': meta.get('version'),
            'prepare_id': meta.get('prepare_id'),
            'levels_spec': meta.get('levels_spec'),
            'resolved_levels': meta.get('resolved_levels'),
            'word2vec': strip_path_values(meta.get('word2vec') or {}),
            'cluster': strip_path_values(meta.get('cluster') or {}),
        }
    if stage == 'quantized':
        export_metrics = meta.get('export_metrics')
        if export_metrics is None and meta.get('checkpoint_metric') is not None:
            export_metrics = [meta.get('checkpoint_metric')]
        return {
            'embedding_model': meta.get('embedding_model'),
            'quantizer_model': meta.get('quantizer_model') or meta.get('hash_model'),
            'quantizer_scheme': meta.get('quantizer_scheme'),
            'representation_family': meta.get('representation_family'),
            'export_metrics': export_metrics,
            'recommended_decoding': meta.get('recommended_decoding'),
            'requested_latent_dim': meta.get('requested_latent_dim'),
            'resolved_latent_dim': meta.get('resolved_latent_dim'),
            'code_shape': meta.get('code_shape'),
            'binary_bits_shape': meta.get('binary_bits_shape'),
            'num_bits_total': meta.get('num_bits_total'),
            'quantizer_config': strip_path_values(meta.get('quantizer_config') or {}),
            'hash_config': strip_path_values(meta.get('hash_config') or {}),
            'trainer_args': strip_path_values(meta.get('trainer_args') or {}),
        }
    raise ValueError(f'unsupported generic stage: {stage}')


def generic_stage_spec(stage: str, meta: dict):
    from utils.artifact_identity import generic_stage_spec as identity_generic_stage_spec

    return identity_generic_stage_spec(stage, meta)


def generic_signature_from_meta(stage: str, meta: dict):
    return identity_generic_signature_from_meta(stage, meta)


def generic_artifact_identity(
    stage: str,
    meta: dict,
    folder: str,
    *,
    aliases: list[str] | None = None,
    migration_status: str = 'current',
):
    return identity_generic_artifact_identity(
        stage,
        meta,
        folder,
        aliases=aliases,
        migration_status=migration_status,
    )


def iter_generic_meta_paths(root: Path, stage: str, data: str | None = None):
    base = root / 'artifacts' / stage
    dataset_dirs = [base / data.lower()] if data else sorted(path for path in base.glob('*') if path.is_dir())
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue
        if stage in {'clustered', 'compiled'}:
            yield from sorted(dataset_dir.glob('*/meta.json'))
        elif stage == 'quantized':
            seen_roots = set()
            for root_meta_path in sorted(dataset_dir.glob('*/*/meta.json')):
                quantized_root = root_meta_path.parent
                seen_roots.add(quantized_root)
                yield root_meta_path
            for meta_path in sorted(dataset_dir.glob('*/*/exports/*/meta.json')):
                quantized_root = meta_path.parents[2]
                if quantized_root in seen_roots:
                    continue
                seen_roots.add(quantized_root)
                yield meta_path


def generic_meta_location(root: Path, stage: str, meta_path: Path):
    relative = meta_path.relative_to(root / 'artifacts' / stage)
    if len(relative.parts) < 3 or relative.parts[-1] != 'meta.json':
        raise ValueError(f'unrecognized {stage} meta path: {meta_path}')
    data = relative.parts[0]
    if stage == 'quantized' and len(relative.parts) >= 6 and relative.parts[-3] == 'exports':
        folder = Path(*relative.parts[1:-3]).as_posix()
        run_dir = meta_path.parents[2]
    else:
        folder = Path(*relative.parts[1:-1]).as_posix()
        run_dir = meta_path.parent
    return {
        'data': data,
        'folder': folder,
        'run_dir': run_dir,
    }


def read_generic_meta(stage: str, meta_path: Path):
    meta = read_json(meta_path)
    if stage != 'quantized':
        return meta

    if meta_path.parent.name == 'exports' or meta_path.parent.parent.name == 'exports':
        root_dir = meta_path.parents[2]
        root_meta = {}
    else:
        root_dir = meta_path.parent
        root_meta = meta if isinstance(meta, dict) else {}
    export_metas = []
    export_metrics = []
    for export_meta_path in sorted((root_dir / 'exports').glob('*/meta.json')):
        export_meta = read_json(export_meta_path)
        if not isinstance(export_meta, dict):
            continue
        export_metas.append(export_meta)
        metric = export_meta.get('checkpoint_metric') or export_meta_path.parent.name
        export_metrics.append(str(metric))
    if not export_metas:
        return meta

    merged = dict(export_metas[0])
    merged.update({key: value for key, value in root_meta.items() if key in {'artifact_identity'}})
    merged['export_metrics'] = sorted(set(export_metrics))
    merged['exports'] = {
        str(export_meta.get('checkpoint_metric') or Path(export_meta.get('export_dir') or '').name): strip_path_values(export_meta)
        for export_meta in export_metas
    }
    return merged


def generic_folder_has_artifacts(dataset_dir: Path, folder: str):
    folder_dir = dataset_dir / str(folder)
    if not folder_dir.exists():
        return False
    for pattern in ('meta.json', '*/meta.json', '*/*/meta.json', 'exports/*/meta.json', '*/exports/*/meta.json'):
        if next(folder_dir.glob(pattern), None) is not None:
            return True
    return False


def register_generic_artifact(stage: str, meta: dict, folder: str, *, aliases: list[str] | None, root: Path):
    return identity_register_generic_artifact(stage, meta, folder, aliases=aliases, root=root)


def iter_trained_meta_paths(root: Path, data: str | None = None):
    base = root / 'artifacts' / 'trained'
    if data:
        dataset_dir = base / data.lower()
        paths = (
            sorted(dataset_dir.glob('*/meta.json'))
            + sorted(dataset_dir.glob('*/*/meta.json'))
            + sorted(dataset_dir.glob('*/*/*/meta.json'))
        )
        yield from paths
        return
    paths = (
        sorted(base.glob('*/*/meta.json'))
        + sorted(base.glob('*/*/*/meta.json'))
        + sorted(base.glob('*/*/*/*/meta.json'))
    )
    yield from paths


def parse_trained_meta_location(root: Path, meta_path: Path):
    try:
        relative = meta_path.relative_to(root / 'artifacts' / 'trained')
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) == 3:
        return {
            'layout': 'legacy-flat',
            'data': parts[0],
            'folder': parts[1],
            'run_dir': meta_path.parent,
        }
    if len(parts) == 4:
        return {
            'layout': 'seed-root',
            'data': parts[0],
            'folder': parts[1],
            'seed': parts[2],
            'run_dir': meta_path.parent,
        }
    if len(parts) == 5:
        return {
            'layout': 'phase-root',
            'data': parts[0],
            'folder': parts[1],
            'seed': parts[2],
            'phase': parts[3],
            'run_dir': meta_path.parent,
        }
    return None


def is_relative_to(path: Path, parent: Path):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def collect_existing_identity_aliases(meta: dict):
    identity = meta.get('artifact_identity')
    if not isinstance(identity, dict):
        return []
    aliases = []
    raw_aliases = identity.get('aliases') or []
    if isinstance(raw_aliases, list):
        aliases.extend(str(alias) for alias in raw_aliases)
    return aliases


def update_trained_meta(meta_path: Path, config, *, apply: bool, aliases: list[str] | None = None):
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError('meta root must be a dict')
    if not isinstance(meta.get('config'), dict):
        raise ValueError('missing meta.config')

    run_dir = meta_path.parent
    aliases = [*(aliases or []), *collect_existing_identity_aliases(meta)]
    legacy_signature = legacy_signature_from_folder(run_dir.name)
    if legacy_signature:
        aliases.append(legacy_signature)
    identity = trained_artifact_identity(
        config,
        run_dir,
        aliases=aliases,
        migration_status='migrated',
    )
    if apply:
        meta['artifact_identity'] = identity
        write_json(meta_path, meta)
    return identity


def checkpoint_exists(setting_dir: Path):
    if not setting_dir.exists():
        return False
    return any(path.name == 'best.pt' for path in setting_dir.rglob('best.pt'))


def summarize_metrics(metrics: dict | None):
    if not isinstance(metrics, dict):
        return {}
    summary = {}
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float, str)):
            summary[key] = value
    return summary


def summarize_run_dir(run_dir: Path):
    run_dir = Path(run_dir)
    meta_path = run_dir / 'meta.json'
    pid_path = run_dir / 'pid.json'
    checkpoint_path = run_dir / 'best.pt'
    summary = {
        'path': str(run_dir),
        'exists': run_dir.exists(),
        'has_meta': meta_path.exists(),
        'has_checkpoint': checkpoint_path.exists(),
        'has_pid': pid_path.exists(),
    }
    if checkpoint_path.exists():
        summary['checkpoint_size_bytes'] = checkpoint_path.stat().st_size
        summary['checkpoint_mtime'] = datetime.fromtimestamp(
            checkpoint_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
    if not meta_path.exists():
        return summary
    try:
        meta = read_json(meta_path)
    except ValueError as exc:
        summary['meta_error'] = str(exc)
        return summary
    if not isinstance(meta, dict):
        summary['meta_error'] = 'meta root is not a dict'
        return summary
    identity = meta.get('artifact_identity') if isinstance(meta.get('artifact_identity'), dict) else {}
    config = meta.get('config') if isinstance(meta.get('config'), dict) else {}
    summary.update(
        {
            'status': meta.get('status'),
            'mode': identity.get('mode'),
            'phase': identity.get('phase') or identity.get('mode'),
            'seed': identity.get('seed') if identity.get('seed') is not None else config.get('seed'),
            'signature': identity.get('signature'),
            'schema_version': identity.get('schema_version'),
            'aliases': identity.get('aliases') or [],
            'best_epoch': meta.get('best_epoch'),
            'main_metric': meta.get('main_metric'),
            'best_valid_metric': meta.get('best_valid_metric'),
            'finished_at': meta.get('finished_at'),
            'started_at': meta.get('started_at'),
            'error': meta.get('error'),
            'failed_at': meta.get('failed_at'),
            'test_metrics': summarize_metrics(meta.get('test_metrics')),
            'valid_metrics': summarize_metrics(meta.get('valid_metrics')),
        }
    )
    if config:
        summary['config_brief'] = {
            key: config.get(key)
            for key in (
                'data',
                'model',
                'repr_type',
                'task_type',
                'seed',
                'batch_size',
                'accumulate_batch',
                'learning_rate',
                'weight_decay',
            )
            if key in config
        }
    return summary


def compact_json(value):
    if value in (None, {}, []):
        return '-'
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def fmt_value(value):
    if value is None or value == '':
        return '-'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def status_badge(status):
    key = str(status or 'unknown').lower()
    if key in {'finished', 'valid_only_finished', 'test_only_finished', 'completed', 'done'}:
        return badge(key, 'green')
    if key in {'failed', 'error'}:
        return badge(key, 'red')
    if key in {'running'}:
        return badge(key, 'yellow')
    return badge(key, 'gray')


def bool_badge(value, true_text='yes', false_text='no'):
    return badge(true_text, 'green') if value else badge(false_text, 'gray')


def section_rule(title: str, color: str = 'cyan', width: int = 96):
    label = f' {title} '
    line_len = max(2, width - len(label))
    left = line_len // 2
    right = line_len - left
    return paint('=' * left, color) + paint(label, 'bold', color) + paint('=' * right, color)


def kv_line(key: str, value, indent: int = 6):
    print(f'{" " * indent}{paint(key.rjust(14), "dim")}: {value}')


def print_run_summary(label: str, summary: dict):
    color = 'blue' if label == 'source' else 'magenta'
    title = f'{label.upper()} run'
    print(f'    {paint("+-- " + title + " " + "-" * max(1, 76 - len(title)), color)}')
    kv_line('status', status_badge(summary.get('status')), indent=6)
    kv_line('mode', fmt_value(summary.get('mode')), indent=6)
    kv_line('phase', fmt_value(summary.get('phase')), indent=6)
    kv_line('seed', fmt_value(summary.get('seed')), indent=6)
    kv_line('checkpoint', bool_badge(summary.get('has_checkpoint')), indent=6)
    kv_line('pid record', bool_badge(summary.get('has_pid')), indent=6)
    kv_line('best valid', fmt_value(summary.get('best_valid_metric')), indent=6)
    kv_line('best epoch', fmt_value(summary.get('best_epoch')), indent=6)
    kv_line('finished at', fmt_value(summary.get('finished_at')), indent=6)
    if summary.get('checkpoint_mtime'):
        checkpoint_text = (
            f'{summary.get("checkpoint_size_bytes")} bytes, '
            f'mtime={summary.get("checkpoint_mtime")}'
        )
        kv_line('ckpt file', checkpoint_text, indent=6)
    if summary.get('config_brief'):
        kv_line('config', compact_json(summary.get('config_brief')), indent=6)
    if summary.get('test_metrics'):
        kv_line('test metrics', compact_json(summary.get('test_metrics')), indent=6)
    if summary.get('valid_metrics'):
        kv_line('valid metrics', compact_json(summary.get('valid_metrics')), indent=6)
    if summary.get('error') or summary.get('failed_at') or summary.get('meta_error'):
        error_text = (
            f'error={summary.get("error") or "-"}; '
            f'failed_at={summary.get("failed_at") or "-"}; '
            f'meta_error={summary.get("meta_error") or "-"}'
        )
        kv_line('problem', paint(error_text, 'red'), indent=6)
    kv_line('path', paint(summary.get('path') or '-', 'dim'), indent=6)
    print(f'    {paint("`--" + "-" * 80, color)}')


def print_metric_comparison(source: dict, target: dict):
    source_metrics = source.get('test_metrics') or {}
    target_metrics = target.get('test_metrics') or {}
    metric_names = sorted(set(source_metrics) | set(target_metrics))
    if not metric_names:
        return
    print(f'    {paint("+-- TEST METRIC COMPARISON " + "-" * 55, "cyan")}')
    print(f'      {paint("metric", "bold"):<24} {paint("source", "bold"):<16} {paint("target", "bold"):<16}')
    print(f'      {"-" * 58}')
    for metric in metric_names:
        source_value = fmt_value(source_metrics.get(metric))
        target_value = fmt_value(target_metrics.get(metric))
        print(f'      {metric:<24} {source_value:<16} {target_value:<16}')
    print(f'    {paint("`--" + "-" * 80, "cyan")}')


def has_result(summary: dict):
    status = str(summary.get('status') or '').lower()
    finished_statuses = {'finished', 'valid_only_finished', 'test_only_finished', 'completed', 'done'}
    return bool(
        summary.get('has_checkpoint')
        or summary.get('test_metrics')
        or summary.get('valid_metrics')
        or summary.get('best_valid_metric') is not None
        or status in finished_statuses
    )


def suggested_resolution(conflict: dict):
    source_has_result = has_result(conflict.get('source') or {})
    target_has_result = has_result(conflict.get('target') or {})
    if source_has_result and not target_has_result:
        return 'source'
    if target_has_result and not source_has_result:
        return 'target'
    return None


def print_choice_help(conflict: dict):
    suggestion = suggested_resolution(conflict)
    if suggestion:
        color = 'blue' if suggestion == 'source' else 'magenta'
        print(f'    {paint("Suggestion", "bold", "yellow")}: choose {badge(suggestion, color)}; only {suggestion} has result')
    print(f'    {paint("Interactive choices", "bold", "yellow")}')
    print(f'      {badge("t", "magenta")} keep target  : delete source; keep canonical target in place')
    print(f'      {badge("s", "blue")} keep source  : delete target; move source into canonical target path')
    print(f'      {badge("k", "gray")} skip         : leave both directories unchanged')
    print(f'      {badge("q", "red")} quit         : stop migration immediately')


def print_conflict_summary(conflict: dict):
    message = conflict.get('error') or conflict.get('action') or 'target run dir already exists'
    stage_label = str(conflict.get('stage') or 'trained').upper()
    print()
    print(section_rule(f'{stage_label} ARTIFACT CONFLICT'))
    print(f'    {paint("setting", "dim")}: {conflict["data"]}/{conflict["folder"]}')
    print(f'    {paint("signature", "dim")}: {conflict.get("signature") or "-"}')
    if conflict.get('seed') is not None:
        print(f'    {paint("seed", "dim")}: {conflict.get("seed")}')
    print(f'    {paint("reason", "dim")}: {paint(message, "yellow")}')
    source = conflict.get('source') or {}
    target = conflict.get('target') or {}
    if source or target:
        print_run_summary('source', source)
        print_run_summary('target', target)
        print_metric_comparison(source, target)


def conflict_report(data: str, folder: str, signature: str, source_dir: Path, target_dir: Path, *, error: str):
    return {
        'data': data,
        'folder': folder,
        'signature': signature,
        'error': error,
        'source': summarize_run_dir(source_dir),
        'target': summarize_run_dir(target_dir),
    }


def registry_folder_run_dir(dataset_dir: Path, folder: str | None, config):
    if not folder:
        return None
    folder_dir = dataset_dir / str(folder)
    seed_name = str(trained_seed(config))
    phase_name = trained_mode(config)
    for candidate in (
        folder_dir / seed_name / phase_name,
        folder_dir / seed_name,
        folder_dir,
    ):
        if candidate.exists():
            return candidate
    meta_paths = sorted(folder_dir.glob('*/meta.json')) + sorted(folder_dir.glob('*/*/meta.json'))
    if meta_paths:
        return meta_paths[0].parent
    return folder_dir


def registry_conflict_report(
    data: str,
    folder: str,
    signature: str,
    source_dir: Path,
    target_dir: Path,
    *,
    dataset_dir: Path,
    config,
    error: Exception,
):
    display_target = target_dir
    if isinstance(error, TrainedArtifactRegistryConflict):
        registry_target = registry_folder_run_dir(dataset_dir, error.existing_folder, config)
        if registry_target is not None:
            display_target = registry_target
    return conflict_report(
        data,
        folder,
        signature,
        source_dir,
        display_target,
        error=str(error),
    )


def generic_registry_folder_run_dir(dataset_dir: Path, folder: str | None):
    if not folder:
        return None
    folder_dir = dataset_dir / str(folder)
    if folder_dir.exists():
        meta_paths = sorted(folder_dir.glob('meta.json')) + sorted(folder_dir.glob('*/meta.json'))
        meta_paths += sorted(folder_dir.glob('*/*/meta.json')) + sorted(folder_dir.glob('exports/*/meta.json'))
        meta_paths += sorted(folder_dir.glob('*/exports/*/meta.json'))
        if meta_paths:
            return meta_paths[0].parent
    return folder_dir


def generic_conflict_report(
    stage: str,
    data: str,
    folder: str,
    signature: str,
    source_dir: Path,
    *,
    dataset_dir: Path,
    error: Exception,
):
    target_dir = dataset_dir / folder
    if isinstance(error, TrainedArtifactRegistryConflict):
        registry_target = generic_registry_folder_run_dir(dataset_dir, error.existing_folder)
        if registry_target is not None:
            target_dir = registry_target
    report = conflict_report(data, folder, signature, source_dir, target_dir, error=str(error))
    report['stage'] = stage
    return report


def choose_conflict_resolution(conflict: dict):
    print_conflict_summary(conflict)
    print_choice_help(conflict)
    while True:
        choice = input(paint('    choose [t/s/k/q]: ', 'bold', 'yellow')).strip().lower()
        if choice in {'t', 'target'}:
            return 'target'
        if choice in {'s', 'source'}:
            return 'source'
        if choice in {'k', 'skip', ''}:
            return 'report'
        if choice in {'q', 'quit', 'exit'}:
            raise KeyboardInterrupt('user quit interactive conflict resolution')
        print(paint('    please enter t, s, k, or q', 'yellow'))


def collect_delete_candidates(root: Path, data: str | None = None):
    base = root / 'artifacts' / 'trained'
    dataset_dirs = [base / data.lower()] if data else sorted(path for path in base.glob('*') if path.is_dir())
    candidates = []
    seen_paths = set()
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue
        meta_paths = (
            sorted(dataset_dir.glob('*/meta.json'))
            + sorted(dataset_dir.glob('*/*/meta.json'))
            + sorted(dataset_dir.glob('*/*/*/meta.json'))
        )
        for meta_path in meta_paths:
            location = parse_trained_meta_location(root, meta_path)
            if location is None:
                continue
            run_dir = meta_path.parent
            run_dir_key = str(run_dir)
            if run_dir_key in seen_paths:
                continue
            seen_paths.add(run_dir_key)
            abnormal_reasons = []
            try:
                meta = read_json(meta_path)
            except ValueError as exc:
                abnormal_reasons.append(str(exc))
                meta = {}
            identity = meta.get('artifact_identity') if isinstance(meta, dict) else None
            mode = identity.get('mode') if isinstance(identity, dict) else None
            if mode is None and isinstance(meta.get('config'), dict):
                try:
                    mode = trained_mode(migrate_train_config_dict(meta['config']))
                except ValueError:
                    mode = None
            status = str(meta.get('status') or '').lower() if isinstance(meta, dict) else ''
            if status in {'failed', 'running'} or meta.get('error') or meta.get('failed_at'):
                abnormal_reasons.append(f'status={status or "unknown"}')

            reasons = []
            if mode in {'valid', 'precheck'}:
                reasons.append('mode=valid/precheck')
                if status:
                    reasons.append(f'status={status}')
            elif mode == 'train' and not checkpoint_exists(run_dir) and abnormal_reasons:
                reasons.extend(abnormal_reasons)

            if reasons:
                candidates.append(
                    {
                        'data': dataset_dir.name,
                        'folder': str(location.get('folder') or run_dir.name),
                        'path': str(run_dir),
                        'reasons': sorted(set(reasons)),
                    }
                )
    return candidates


def prune_empty_trained_parents(path: Path, dataset_dir: Path):
    parent = path.parent
    while parent != dataset_dir and parent.exists() and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def move_run_dir(source_dir: Path, target_dir: Path, dataset_dir: Path):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if source_dir == target_dir:
        return
    if target_dir.parent == source_dir:
        target_dir.mkdir(parents=True, exist_ok=False)
        for child in list(source_dir.iterdir()):
            if child == target_dir:
                continue
            if child.is_dir() and child.name in TRAINED_PHASES:
                continue
            shutil.move(str(child), str(target_dir / child.name))
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))
    prune_empty_trained_parents(source_dir, dataset_dir)


def delete_run_dir(run_dir: Path, dataset_dir: Path):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    shutil.rmtree(run_dir)
    prune_empty_trained_parents(run_dir, dataset_dir)


def delete_source_for_target(source_dir: Path, target_dir: Path, dataset_dir: Path):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not source_dir.exists() or source_dir == target_dir:
        return
    if is_relative_to(target_dir, source_dir):
        for child in list(source_dir.iterdir()):
            if child == target_dir:
                continue
            if child.is_dir() and child.name in TRAINED_PHASES:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        return
    delete_run_dir(source_dir, dataset_dir)


def prune_empty_generic_parents(path: Path, dataset_dir: Path):
    parent = Path(path).parent
    while parent != dataset_dir and parent.exists() and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def move_generic_run_dir(source_dir: Path, target_dir: Path, dataset_dir: Path):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if source_dir == target_dir:
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))
    prune_empty_generic_parents(source_dir, dataset_dir)


def delete_generic_run_dir(run_dir: Path, dataset_dir: Path):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    shutil.rmtree(run_dir)
    prune_empty_generic_parents(run_dir, dataset_dir)


def init_trained_registry(
    root: Path,
    *,
    data: str | None,
    apply: bool,
    delete_abnormal_empty: bool = False,
    interactive: bool = False,
):
    resolved = []
    unresolved = []
    conflicts = []
    moved = []

    for meta_path in iter_trained_meta_paths(root, data=data):
        if not meta_path.exists():
            continue
        location = parse_trained_meta_location(root, meta_path)
        dataset = str(location.get('data') if location else meta_path.parent.parent.name)
        folder = str(location.get('folder') if location else meta_path.parent.name)
        source_run_dir = Path(location.get('run_dir') if location else meta_path.parent)
        dataset_dir = root / 'artifacts' / 'trained' / dataset
        try:
            meta = read_json(meta_path)
            config = migrate_train_config_dict(meta.get('config'))
            signature = trained_signature_from_config(config)
            aliases = []
            legacy_signature = legacy_signature_from_folder(folder)
            if legacy_signature:
                aliases.append(legacy_signature)
            identity = update_trained_meta(meta_path, config, apply=False, aliases=aliases)
            target_run_dir = canonical_trained_run_dir(config, root=root)
            resolved.append(
                {
                    'data': dataset,
                    'folder': folder,
                    'signature': signature,
                    'seed': trained_seed(config),
                    'phase': target_run_dir.name,
                    'target_run_dir': str(target_run_dir),
                    'aliases': identity.get('aliases') or [],
                    'meta_path': str(meta_path),
                }
            )
            if not apply and source_run_dir != target_run_dir and target_run_dir.exists():
                conflict = {
                    'data': dataset,
                    'folder': folder,
                    'signature': signature,
                    'seed': trained_seed(config),
                    'source': summarize_run_dir(source_run_dir),
                    'target': summarize_run_dir(target_run_dir),
                    'resolution': 'dry-run',
                    'error': f'target run dir already exists: {target_run_dir}',
                }
                conflicts.append(conflict)
            if apply:
                try:
                    final_meta_path = meta_path
                    if source_run_dir != target_run_dir:
                        existing_target_conflict = None
                        if target_run_dir.exists():
                            existing_target_conflict = {
                                'data': dataset,
                                'folder': folder,
                                'signature': signature,
                                'seed': trained_seed(config),
                                'source': summarize_run_dir(source_run_dir),
                                'target': summarize_run_dir(target_run_dir),
                                'resolution': 'interactive' if interactive else 'report',
                            }
                            resolution = (
                                choose_conflict_resolution(existing_target_conflict)
                                if interactive
                                else 'report'
                            )
                            existing_target_conflict['resolution'] = resolution
                            if resolution == 'report':
                                existing_target_conflict['error'] = f'target run dir already exists: {target_run_dir}'
                                conflicts.append(existing_target_conflict)
                                continue
                            if resolution == 'target':
                                register_trained_artifact(
                                    config,
                                    target_run_dir,
                                    aliases=identity.get('aliases'),
                                    root=root,
                                )
                                delete_source_for_target(source_run_dir, target_run_dir, dataset_dir)
                                existing_target_conflict['action'] = 'kept target and deleted source'
                                existing_target_conflict['deleted'] = str(source_run_dir)
                                conflicts.append(existing_target_conflict)
                                continue
                            if resolution == 'source':
                                delete_run_dir(target_run_dir, dataset_dir)
                                existing_target_conflict['action'] = 'kept source and deleted target'
                                existing_target_conflict['deleted'] = str(target_run_dir)
                                conflicts.append(existing_target_conflict)
                            else:
                                raise ValueError(f'unsupported conflict resolution: {resolution}')
                        old_run_dir = source_run_dir
                        move_run_dir(old_run_dir, target_run_dir, dataset_dir)
                        final_meta_path = target_run_dir / 'meta.json'
                        moved.append(
                            {
                                'data': dataset,
                                'from': str(old_run_dir),
                                'to': str(target_run_dir),
                            }
                        )
                    try:
                        registered_artifact = register_trained_artifact(
                            config,
                            target_run_dir,
                            aliases=identity.get('aliases'),
                            root=root,
                        )
                    except ValueError as exc:
                        conflicts.append(
                            registry_conflict_report(
                                dataset,
                                folder,
                                signature,
                                source_run_dir,
                                target_run_dir,
                                dataset_dir=dataset_dir,
                                config=config,
                                error=exc,
                            )
                        )
                        continue
                    final_meta = read_json(final_meta_path)
                    final_meta['run_dir'] = str(target_run_dir)
                    final_meta['artifact_identity'] = trained_artifact_identity(
                        config,
                        target_run_dir,
                        aliases=registered_artifact.get('aliases') or identity.get('aliases'),
                        migration_status='migrated',
                    )
                    write_json(final_meta_path, final_meta)
                except ValueError as exc:
                    conflicts.append(
                        registry_conflict_report(
                            dataset,
                            folder,
                            signature,
                            source_run_dir,
                            target_run_dir,
                            dataset_dir=dataset_dir,
                            config=config,
                            error=exc,
                        )
                    )
        except Exception as exc:
            unresolved.append(
                {
                    'data': dataset,
                    'folder': folder,
                    'meta_path': str(meta_path),
                    'reason': str(exc),
                }
            )

    touched_data = sorted({item['data'] for item in resolved})
    if apply:
        for dataset in touched_data:
            index = load_trained_index(dataset, root=root)
            save_trained_index(dataset, index, root=root)

    delete_candidates = collect_delete_candidates(root, data=data)
    deleted = []
    if apply and delete_abnormal_empty:
        for candidate in delete_candidates:
            path = Path(candidate['path'])
            if path.exists():
                shutil.rmtree(path)
                prune_empty_trained_parents(path, root / 'artifacts' / 'trained' / candidate['data'])
                deleted.append(candidate)

    return {
        'stage': 'trained',
        'mode': 'apply' if apply else 'dry-run',
        'resolved_count': len(resolved),
        'unresolved_count': len(unresolved),
        'conflict_count': len(conflicts),
        'moved_count': len(moved),
        'delete_candidate_count': len(delete_candidates),
        'deleted_count': len(deleted),
        'index_name': TRAINED_INDEX_NAME,
        'datasets': touched_data,
        'resolved': resolved,
        'unresolved': unresolved,
        'conflicts': conflicts,
        'moved': moved,
        'delete_candidates': delete_candidates,
        'deleted': deleted,
    }


def init_generic_registry(
    root: Path,
    *,
    stage: str,
    data: str | None,
    apply: bool,
    interactive: bool = False,
):
    if stage not in GENERIC_STAGES:
        raise ValueError(f'unsupported generic stage: {stage}')

    resolved = []
    unresolved = []
    conflicts = []
    moved = []
    touched_data = set()

    for meta_path in iter_generic_meta_paths(root, stage, data=data):
        try:
            location = generic_meta_location(root, stage, meta_path)
            meta = read_generic_meta(stage, meta_path)
            if not isinstance(meta, dict):
                raise ValueError('meta root must be a dict')
            meta_data = generic_data_from_meta(stage, meta)
            if meta_data != location['data'].lower():
                raise ValueError(f'meta data={meta_data} does not match path data={location["data"]}')
            folder = location['folder']
            run_dir = Path(location['run_dir'])
            identity_meta_path = run_dir / 'meta.json' if stage == 'quantized' else meta_path
            dataset_dir = root / 'artifacts' / stage / meta_data
            aliases = collect_existing_identity_aliases(meta)
            legacy_signature = legacy_signature_from_folder(folder)
            if legacy_signature:
                aliases.append(legacy_signature)
            signature = generic_signature_from_meta(stage, meta)
            target_run_dir = canonical_generic_artifact_dir(stage, meta_data, signature, root=root)
            target_folder = target_run_dir.relative_to(dataset_dir).as_posix()
            identity = generic_artifact_identity(
                stage,
                meta,
                target_folder,
                aliases=aliases,
                migration_status='migrated',
            )
            resolved.append(
                {
                    'data': meta_data,
                    'folder': folder,
                    'signature': signature,
                    'target_run_dir': str(target_run_dir),
                    'aliases': identity.get('aliases') or [],
                    'meta_path': str(identity_meta_path),
                }
            )
            touched_data.add(meta_data)
            if not apply:
                if run_dir != target_run_dir and target_run_dir.exists():
                    conflicts.append(
                        conflict_report(
                            meta_data,
                            folder,
                            signature,
                            run_dir,
                            target_run_dir,
                            error=f'target run dir already exists: {target_run_dir}',
                        )
                    )
                continue

            while True:
                try:
                    final_run_dir = run_dir
                    final_meta_path = identity_meta_path
                    if run_dir != target_run_dir:
                        if target_run_dir.exists():
                            conflict = conflict_report(
                                meta_data,
                                folder,
                                signature,
                                run_dir,
                                target_run_dir,
                                error=f'target run dir already exists: {target_run_dir}',
                            )
                            conflict['stage'] = stage
                            resolution = choose_conflict_resolution(conflict) if interactive else 'report'
                            conflict['resolution'] = resolution
                            if resolution == 'report':
                                conflicts.append(conflict)
                                break
                            if resolution == 'target':
                                delete_generic_run_dir(run_dir, dataset_dir)
                                conflict['action'] = 'kept target and deleted source'
                                conflict['deleted'] = str(run_dir)
                                conflicts.append(conflict)
                                final_run_dir = target_run_dir
                                final_meta_path = target_run_dir / 'meta.json'
                            elif resolution == 'source':
                                delete_generic_run_dir(target_run_dir, dataset_dir)
                                conflict['action'] = 'kept source and deleted target'
                                conflict['deleted'] = str(target_run_dir)
                                conflicts.append(conflict)
                            else:
                                raise ValueError(f'unsupported conflict resolution: {resolution}')
                        if final_run_dir == run_dir and run_dir.exists():
                            old_run_dir = run_dir
                            move_generic_run_dir(old_run_dir, target_run_dir, dataset_dir)
                            final_run_dir = target_run_dir
                            final_meta_path = target_run_dir / 'meta.json'
                            moved.append(
                                {
                                    'data': meta_data,
                                    'from': str(old_run_dir),
                                    'to': str(target_run_dir),
                                }
                            )
                    registered = register_generic_artifact(
                        stage,
                        meta,
                        target_folder,
                        aliases=identity.get('aliases'),
                        root=root,
                    )
                    meta['artifact_identity'] = generic_artifact_identity(
                        stage,
                        meta,
                        target_folder,
                        aliases=registered.get('aliases') or identity.get('aliases'),
                        migration_status='migrated',
                    )
                    write_json(final_meta_path, meta)
                    break
                except ValueError as exc:
                    conflict = generic_conflict_report(
                        stage,
                        meta_data,
                        folder,
                        signature,
                        run_dir,
                        dataset_dir=dataset_dir,
                        error=exc,
                    )
                    if not interactive:
                        conflicts.append(conflict)
                        break
                    resolution = choose_conflict_resolution(conflict)
                    conflict['resolution'] = resolution
                    if resolution == 'report':
                        conflicts.append(conflict)
                        break
                    target_path = Path((conflict.get('target') or {}).get('path') or '')
                    if resolution == 'target':
                        delete_generic_run_dir(run_dir, dataset_dir)
                        conflict['action'] = 'kept target and deleted source'
                        conflict['deleted'] = str(run_dir)
                        conflicts.append(conflict)
                        break
                    if resolution == 'source':
                        if target_path.exists() and target_path != run_dir:
                            delete_generic_run_dir(target_path, dataset_dir)
                        conflict['action'] = 'kept source and deleted target'
                        conflict['deleted'] = str(target_path)
                        conflicts.append(conflict)
                        continue
                    raise ValueError(f'unsupported conflict resolution: {resolution}')
        except Exception as exc:
            fallback_data = data or '-'
            fallback_folder = str(meta_path.parent) if 'meta_path' in locals() else '-'
            try:
                location = generic_meta_location(root, stage, meta_path)
                fallback_data = location.get('data') or fallback_data
                fallback_folder = location.get('folder') or fallback_folder
            except Exception:
                pass
            unresolved.append(
                {
                    'data': str(fallback_data),
                    'folder': str(fallback_folder),
                    'meta_path': str(meta_path),
                    'reason': str(exc),
                }
            )

    if apply:
        for dataset in sorted(touched_data):
            index = load_registry_index(stage, dataset, root)
            save_registry_index(stage, dataset, index, root)

    return {
        'stage': stage,
        'mode': 'apply' if apply else 'dry-run',
        'resolved_count': len(resolved),
        'unresolved_count': len(unresolved),
        'conflict_count': len(conflicts),
        'moved_count': len(moved),
        'delete_candidate_count': 0,
        'deleted_count': 0,
        'index_name': TRAINED_INDEX_NAME,
        'datasets': sorted(touched_data),
        'resolved': resolved,
        'unresolved': unresolved,
        'conflicts': conflicts,
        'moved': moved,
        'delete_candidates': [],
        'deleted': [],
    }


def main():
    parser = argparse.ArgumentParser(description='Initialize artifact registry indexes for existing artifacts.')
    parser.add_argument('--stage', default='trained', choices=sorted(REGISTRY_STAGES))
    parser.add_argument('--data', default=None, help='Optional dataset name, e.g. mind.')
    parser.add_argument('--root', default=str(ROOT), help='Algorithm project root. Defaults to this repository.')
    parser.add_argument('--apply', action='store_true', help='Write meta.json artifact_identity fields and .index.json.')
    parser.add_argument(
        '--delete-abnormal-empty',
        action='store_true',
        help='With --apply, delete all precheck/valid-only runs and abnormal train runs without best.pt.',
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='With --apply, prompt for each conflict and immediately apply the selected resolution.',
    )
    parser.add_argument('--json', action='store_true', help='Print the full report as JSON.')
    args = parser.parse_args()
    if args.interactive and not args.apply:
        parser.error('--interactive requires --apply because choices are executed immediately')
    if args.delete_abnormal_empty and args.stage != 'trained':
        parser.error('--delete-abnormal-empty is only supported for --stage trained')

    if args.stage == 'trained':
        report = init_trained_registry(
            Path(args.root),
            data=args.data,
            apply=bool(args.apply),
            delete_abnormal_empty=bool(args.delete_abnormal_empty),
            interactive=bool(args.interactive),
        )
    else:
        report = init_generic_registry(
            Path(args.root),
            stage=args.stage,
            data=args.data,
            apply=bool(args.apply),
            interactive=bool(args.interactive),
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(
        f'artifact registry init stage={report["stage"]} mode={report["mode"]} '
        f'resolved={report["resolved_count"]} unresolved={report["unresolved_count"]} '
        f'conflicts={report["conflict_count"]} moved={report["moved_count"]} '
        f'delete_candidates={report["delete_candidate_count"]} deleted={report["deleted_count"]}'
    )
    if report['datasets']:
        print('datasets: ' + ', '.join(report['datasets']))
    for item in report['resolved'][:20]:
        aliases = ','.join(item['aliases'])
        print(f'  ok {item["data"]}/{item["folder"]} -> {item["signature"]} aliases=[{aliases}]')
    if len(report['resolved']) > 20:
        print(f'  ... {len(report["resolved"]) - 20} more resolved')
    for item in report['unresolved'][:20]:
        print(f'  unresolved {item["data"]}/{item["folder"]}: {item["reason"]}')
    if len(report['unresolved']) > 20:
        print(f'  ... {len(report["unresolved"]) - 20} more unresolved')
    for item in report['conflicts'][:20]:
        source = item.get('source') or {}
        target = item.get('target') or {}
        if source or target:
            print_conflict_summary(item)
            if not item.get('action'):
                print(
                    '    choose: rerun with --apply --interactive to decide and execute this conflict'
                )
        else:
            print(f'  conflict {item["data"]}/{item["folder"]}: {item.get("error") or item.get("action")}')
    for item in report['moved'][:20]:
        print(f'  moved {item["from"]} -> {item["to"]}')
    deleted_paths = {item.get('path') for item in report.get('deleted', [])}
    for item in report['delete_candidates'][:20]:
        reasons = ','.join(item['reasons'])
        marker = 'deleted' if item.get('path') in deleted_paths else 'not deleted'
        print(f'  delete-candidate ({marker}) {item["data"]}/{item["folder"]}: {reasons}')
    if report['delete_candidate_count'] and not report['deleted_count']:
        print('  note: delete candidates are only reported by default; add --apply --delete-abnormal-empty to delete them.')


if __name__ == '__main__':
    main()
