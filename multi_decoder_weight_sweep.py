import json
from pathlib import Path

import torch
from tqdm import tqdm

from core import TrainConfig
from trainer import Trainer
from utils import function
from utils.config_init import ConfigInit
from utils.logging import setup_logging
SWEEP_KEYS = {'uid_weights', 'fusion_methods', 'rrf_k', 'output', 'max_batches'}


def _parse_weights(value):
    values = [float(part.strip()) for part in str(value).split(',') if part.strip()]
    if not values:
        raise ValueError('--uid_weights requires at least one value')
    if any(weight < 0.0 or weight > 1.0 for weight in values):
        raise ValueError('--uid_weights values must be between 0 and 1')
    return list(dict.fromkeys(values))


def _parse_fusion_methods(value):
    aliases = {'score': 'fixed', 'fixed': 'fixed', 'rrf': 'rrf'}
    methods = []
    for part in str(value).split(','):
        key = part.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError('--fusion_methods supports fixed and rrf')
        methods.append(aliases[key])
    if not methods:
        raise ValueError('--fusion_methods requires at least one method')
    return list(dict.fromkeys(methods))


def _format_table(rows, metric_names):
    headers = ['fusion', 'uid_weight', 'sid_weight', *metric_names, 'candidates']
    values = []
    for row in rows:
        values.append([
            row['fusion_method'],
            f'{row["uid_weight"]:.2f}',
            f'{row["sid_weight"]:.2f}',
            *[f'{row.get(metric, 0.0):.4f}' for metric in metric_names],
            f'{row["multi_candidates"]:.1f}',
        ])
    widths = [max(len(headers[i]), *(len(row[i]) for row in values)) for i in range(len(headers))]
    lines = ['  '.join(headers[i].ljust(widths[i]) for i in range(len(headers)))]
    lines.append('  '.join('-' * width for width in widths))
    lines.extend('  '.join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in values)
    return '\n'.join(lines)


def _full_sid_scores(model, pooled, batch, sid_names):
    spaces = []
    for sid_name in sid_names:
        mode = model._sid_decoding_mode(sid_name)
        if mode == 'fast':
            scores = model._sid_fast_item_scores(batch, sid_name)
        elif mode == 'parallel':
            semantic, collision = model._sid_parallel_item_scores(pooled, sid_name)
            scores = semantic + collision
        else:
            raise ValueError(
                'full-catalog fusion sweep requires test_sid_decoding=fast or parallel'
            )
        spaces.append(model._normalize_score_tensor(
            scores, model.config.multi_score_normalization,
        ))
    return torch.stack(spaces).mean(dim=0)


def main():
    setup_logging()
    kwargs = function.argparse()
    sweep = {key: kwargs.pop(key) for key in list(kwargs) if key in SWEEP_KEYS}
    if not kwargs.get('load_ckpt'):
        raise ValueError('--load_ckpt is required')
    kwargs['test_only'] = True
    configurations = ConfigInit(
        [], {'config': 'config/trainer/sid-uid-content-multi-decoder.yaml'}, [],
    ).parse_kwargs(kwargs)
    config = TrainConfig.from_refconfig(configurations)
    if not config.is_multi_task or 'uid' not in config.task_types or 'sid' not in config.task_types:
        raise ValueError('weight sweep requires a SID+UID multi-decoder trainer config')

    weights = _parse_weights(sweep.get('uid_weights', '0,0.25,0.5,0.75,1'))
    fusion_methods = _parse_fusion_methods(sweep.get('fusion_methods', 'fixed,rrf'))
    rrf_k = float(sweep.get('rrf_k', config.multi_rrf_k))
    if rrf_k < 0:
        raise ValueError('--rrf_k must be non-negative')
    max_batches = int(sweep.get('max_batches', 0))
    output_path = Path(sweep.get('output') or f'reports/{config.data}_multi_decoder_weight_sweep.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(config)
    checkpoint = trainer._load_checkpoint_for_eval(config.load_ckpt)
    trainer.build_test_loader_only()
    model = trainer.model_core
    model.eval()
    sid_names = config.compile_config.names_for_kind('sid', targets=True)
    ks = model.ranking_ks()
    totals_by_setting = {
        (method, weight): model._init_ranking_totals(ks)
        for method in fusion_methods
        for weight in weights
    }
    for totals in totals_by_setting.values():
        totals['multi_candidates'] = 0.0

    sample_count = 0
    with torch.inference_mode():
        progress = tqdm(trainer.test_loader, desc='multi-weight-sweep')
        for batch_index, batch in enumerate(progress):
            if max_batches > 0 and batch_index >= max_batches:
                break
            inputs, attention_mask, lengths = model._build_batch_inputs(batch)
            hidden = model.encoder(inputs_embeds=inputs, attention_mask=attention_mask)
            pooled = hidden[torch.arange(len(batch), device=model.device), lengths - 1]
            uid_scores = model._uid_logits(pooled).float()
            sid_scores = _full_sid_scores(model, pooled, batch, sid_names)
            uid_normalized = model._normalize_score_tensor(
                uid_scores, config.multi_score_normalization,
            )
            sid_normalized = model._normalize_score_tensor(
                sid_scores, config.multi_score_normalization,
            )
            uid_ranks = model._rank_score_tensor(uid_scores)
            sid_ranks = model._rank_score_tensor(sid_scores)
            for method in fusion_methods:
                for weight in weights:
                    if method == 'rrf':
                        fused_scores = (
                            weight / (rrf_k + uid_ranks)
                            + (1.0 - weight) / (rrf_k + sid_ranks)
                        )
                    else:
                        fused_scores = (
                            weight * uid_normalized / config.multi_temperature_uid
                            + (1.0 - weight) * sid_normalized / config.multi_temperature_sid
                        )
                    ranking = torch.topk(
                        fused_scores,
                        k=min(config.multi_output_topk, fused_scores.shape[-1]),
                        dim=-1,
                    ).indices
                    totals = totals_by_setting[(method, weight)]
                    totals['multi_candidates'] += float(fused_scores.shape[-1] * len(batch))
                    for sample, ranked in zip(batch, ranking):
                        model._accumulate_ranking_metrics(
                            totals, ks, [int(uid) for uid in ranked.tolist()], sample,
                        )
            sample_count += len(batch)

    rows = []
    for method in fusion_methods:
        for weight in weights:
            row = {
                'fusion_method': method,
                'uid_weight': weight,
                'sid_weight': 1.0 - weight,
                **{
                    key: value / max(sample_count, 1)
                    for key, value in totals_by_setting[(method, weight)].items()
                },
            }
            rows.append(row)
    metric_names = [name for name in ('ndcg@5', 'ndcg@10', 'ndcg@20', 'hr@5', 'hr@10', 'hr@20', 'mrr') if name in rows[0]]
    selection_metric = next(
        (name for name in str(config.main_metric).split('|') if name in rows[0]),
        metric_names[0],
    )
    best = max(rows, key=lambda row: row[selection_metric])
    report = {
        'data': config.data,
        'checkpoint': str(config.load_ckpt),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'samples': sample_count,
        'selection_metric': selection_metric,
        'best_uid_weight': best['uid_weight'],
        'best_sid_weight': best['sid_weight'],
        'best_fusion_method': best['fusion_method'],
        'fusion_methods': fusion_methods,
        'rrf_k': rrf_k,
        'results': rows,
        'note': 'Fast SID and UID are both ranked over the complete item catalog.',
    }
    output_path.write_text(json.dumps(report, indent=2) + '\n')

    print('\n' + '=' * 100)
    print(f'MULTI-DECODER WEIGHT SWEEP: {config.data} | samples={sample_count}')
    print('=' * 100)
    print(_format_table(rows, metric_names))
    print(
        f'\nbest by {selection_metric}: fusion={best["fusion_method"]} '
        f'UID={best["uid_weight"]:.2f} '
        f'SID={best["sid_weight"]:.2f} {selection_metric}={best[selection_metric]:.4f}'
    )
    print('Note: Fast SID and UID are both ranked over the complete item catalog.')
    print(f'JSON: {output_path}')


if __name__ == '__main__':
    main()
