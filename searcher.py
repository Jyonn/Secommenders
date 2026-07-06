import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.artifact_identity import (  # noqa: E402
    TRAIN_CONFIG_DEFAULTS,
    trained_phase_dir_name,
    trained_seed_dir_name,
    trained_signature_from_config,
)


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def coerce_scalar(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    lower = text.lower()
    if lower == 'null':
        return None
    if lower == 'true':
        return True
    if lower == 'false':
        return False
    if text.isdigit() or (text.startswith('-') and text[1:].isdigit()):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return value


def normalize_config(args):
    kwargs = {key: coerce_scalar(value) for key, value in vars(args).items() if value is not None}
    kwargs.pop('root', None)
    kwargs.pop('all_seeds', None)
    kwargs.pop('json', None)
    kwargs.pop('config', None)
    kwargs = normalize_aliases(kwargs)

    config = deepcopy(TRAIN_CONFIG_DEFAULTS)
    for key, value in kwargs.items():
        if key in config or key in {'data', 'model', 'repr_type', 'task_type'}:
            config[key] = value
    if isinstance(config.get('metrics'), str):
        config['metrics'] = [part.strip().lower() for part in config['metrics'].split(',') if part.strip()]

    missing = [key for key in ('data', 'model', 'repr_type', 'task_type') if not config.get(key)]
    if missing:
        raise ValueError(f'missing required fields: {missing}')
    return config


def normalize_aliases(kwargs: dict):
    aliases = {
        'repr': 'repr_type',
        'task': 'task_type',
        'lr': 'learning_rate',
        'wd': 'weight_decay',
        'sid_quantizer_name': 'sid_coder',
        'hash_quantizer_name': 'hash_coder',
    }
    normalized = dict(kwargs)
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    return normalized


def resolve_setting_folder(data: str, signature: str, root: Path):
    index = read_json_if_exists(root / 'artifacts' / 'trained' / data / '.index.json') or {}
    seen = set()
    cursor = signature
    while cursor and cursor not in seen:
        seen.add(cursor)
        entry = index.get(cursor)
        if not isinstance(entry, dict):
            return signature, None
        alias_of = entry.get('alias_of')
        if alias_of:
            cursor = str(alias_of)
            continue
        return cursor, entry
    return signature, None


def summarize_metrics(metrics: dict | None):
    if not isinstance(metrics, dict):
        return {}
    return {
        key: value
        for key, value in sorted(metrics.items())
        if isinstance(value, (int, float, str))
    }


def summarize_run(run_dir: Path):
    meta = read_json_if_exists(run_dir / 'meta.json') or {}
    pid = read_json_if_exists(run_dir / 'pid.json') or {}
    return {
        'path': str(run_dir),
        'exists': run_dir.exists(),
        'status': meta.get('status'),
        'seed': (meta.get('config') or {}).get('seed'),
        'phase': (meta.get('artifact_identity') or {}).get('phase') or run_dir.name,
        'has_checkpoint': (run_dir / 'best.pt').exists(),
        'best_epoch': meta.get('best_epoch'),
        'main_metric': meta.get('main_metric'),
        'best_valid_metric': meta.get('best_valid_metric'),
        'test_metrics': summarize_metrics(meta.get('test_metrics')),
        'valid_metrics': summarize_metrics(meta.get('valid_metrics')),
        'pid': pid.get('pid'),
        'hostname': pid.get('hostname'),
        'started_at': meta.get('started_at'),
        'finished_at': meta.get('finished_at'),
        'failed_at': meta.get('failed_at'),
        'error': meta.get('error'),
    }


def find_runs(config: dict, root: Path, *, all_seeds: bool):
    signature = trained_signature_from_config(config)
    data = str(config['data']).lower()
    dataset_dir = root / 'artifacts' / 'trained' / data
    primary_signature, entry = resolve_setting_folder(data, signature, root)
    folder = str(entry.get('folder')) if isinstance(entry, dict) and entry.get('folder') else signature
    setting_dir = dataset_dir / folder
    seed_name = trained_seed_dir_name(config)
    phase_name = trained_phase_dir_name(config)

    if all_seeds:
        candidate_runs = sorted(setting_dir.glob(f'*/{phase_name}/meta.json'))
        run_dirs = [path.parent for path in candidate_runs]
    else:
        run_dirs = [setting_dir / seed_name / phase_name]

    return {
        'query_signature': signature,
        'primary_signature': primary_signature,
        'folder': folder,
        'setting_dir': str(setting_dir),
        'index_hit': isinstance(entry, dict),
        'runs': [summarize_run(run_dir) for run_dir in run_dirs],
        'canonical_run_dir': str(dataset_dir / signature / seed_name / phase_name),
        'resolved_run_dir': str(setting_dir / seed_name / phase_name),
    }


def print_report(report: dict):
    print(
        f'trained search signature={report["query_signature"]} '
        f'primary={report["primary_signature"]} index_hit={report["index_hit"]}'
    )
    print(f'setting_dir={report["setting_dir"]}')
    print(f'canonical_run_dir={report["canonical_run_dir"]}')
    print(f'resolved_run_dir={report["resolved_run_dir"]}')
    if not report['runs']:
        print('no runs found')
        return
    for run in report['runs']:
        marker = 'FOUND' if run['exists'] else 'MISS'
        print(
            f'{marker} seed={run.get("seed") or "-"} phase={run.get("phase") or "-"} '
            f'status={run.get("status") or "unknown"} checkpoint={"yes" if run["has_checkpoint"] else "no"}'
        )
        print(f'  path={run["path"]}')
        if run.get('best_epoch') is not None:
            print(
                f'  best_epoch={run["best_epoch"]} main_metric={run.get("main_metric") or "-"} '
                f'best_valid={run.get("best_valid_metric")}'
            )
        metrics = run.get('test_metrics') or run.get('valid_metrics')
        if metrics:
            print('  metrics=' + json.dumps(metrics, sort_keys=True))
        if run.get('error'):
            print(f'  error={run["error"]}')


def main():
    parser = argparse.ArgumentParser(description='Search trained artifacts by trainer config fields.')
    parser.add_argument('--root', default=str(ROOT), help='Algorithm repository root.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--repr_type', '--repr', dest='repr_type', required=True)
    parser.add_argument('--task_type', '--task', dest='task_type', required=True)
    parser.add_argument('--repr_source_model')
    parser.add_argument('--sid_export')
    parser.add_argument('--sid_coder')
    parser.add_argument('--hash_coder')
    parser.add_argument('--repr_combine')
    parser.add_argument('--maxitems', type=int)
    parser.add_argument('--model_max_length', type=int)
    parser.add_argument('--item_text_max_tokens', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--accumulate_batch', type=int)
    parser.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float)
    parser.add_argument('--weight_decay', '--wd', dest='weight_decay', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--uid_decoding')
    parser.add_argument('--uid_cluster_levels')
    parser.add_argument('--uid_cluster_topk')
    parser.add_argument('--code_decoding')
    parser.add_argument('--code_beam_width', type=int)
    parser.add_argument('--code_collision_loss_weight', type=float)
    parser.add_argument('--main_metric')
    parser.add_argument('--metrics')
    parser.add_argument('--patience', type=int)
    parser.add_argument('--alignment_weight', '--alignment', dest='alignment_weight', type=float)
    parser.add_argument('--num_gpus', type=int)
    parser.add_argument('--freeze_backbone')
    parser.add_argument('--model_dtype')
    parser.add_argument('--use_lora')
    parser.add_argument('--lora_rank', type=int)
    parser.add_argument('--lora_alpha', type=int)
    parser.add_argument('--lora_dropout', type=float)
    parser.add_argument('--lora_layers')
    parser.add_argument('--lora_target_modules')
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--valid_only')
    parser.add_argument('--test_only')
    parser.add_argument('--overwrite')
    parser.add_argument('--all-seeds', action='store_true', help='List all seed subdirectories for the setting.')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON.')
    args = parser.parse_args()

    config = normalize_config(args)
    report = find_runs(config, Path(args.root), all_seeds=args.all_seeds)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print_report(report)


if __name__ == '__main__':
    main()
