import json
from pathlib import Path

import torch
from tqdm import tqdm

from core import TrainConfig
from trainer import Trainer
from utils import function
from utils.config_init import ConfigInit
from utils.logging import setup_logging
from utils.multi_decoding import fuse_candidate_scores


SWEEP_KEYS = {'uid_weights', 'output', 'max_batches'}


def _parse_weights(value):
    values = [float(part.strip()) for part in str(value).split(',') if part.strip()]
    if not values:
        raise ValueError('--uid_weights requires at least one value')
    if any(weight < 0.0 or weight > 1.0 for weight in values):
        raise ValueError('--uid_weights values must be between 0 and 1')
    return list(dict.fromkeys(values))


def _format_table(rows, metric_names):
    headers = ['uid_weight', 'sid_weight', *metric_names, 'candidates']
    values = []
    for row in rows:
        values.append([
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


def _sweep_scored_unions(model, batch, scored_unions, weights, totals_by_weight, ks):
    for sample, scores in zip(batch, scored_unions):
        for weight in weights:
            fused = fuse_candidate_scores(
                scores['uid_scores'],
                scores['sid_scores'],
                uid_weight=weight,
                score_normalization=model.config.multi_score_normalization,
                temperature_uid=model.config.multi_temperature_uid,
                temperature_sid=model.config.multi_temperature_sid,
                output_topk=model.config.multi_output_topk,
            )
            totals = totals_by_weight[weight]
            totals['multi_candidates'] += float(scores['candidate_count'])
            model._accumulate_ranking_metrics(totals, ks, [uid for uid, _ in fused], sample)


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
    totals_by_weight = {weight: model._init_ranking_totals(ks) for weight in weights}
    for totals in totals_by_weight.values():
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
            scored_unions = model._score_multi_candidate_unions(pooled, batch, sid_names)
            _sweep_scored_unions(model, batch, scored_unions, weights, totals_by_weight, ks)
            sample_count += len(batch)

    rows = []
    for weight in weights:
        row = {
            'uid_weight': weight,
            'sid_weight': 1.0 - weight,
            **{key: value / max(sample_count, 1) for key, value in totals_by_weight[weight].items()},
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
        'results': rows,
        'note': 'Weights 0 and 1 rerank the shared UID/SID candidate union; they are not standalone branch retrieval.',
    }
    output_path.write_text(json.dumps(report, indent=2) + '\n')

    print('\n' + '=' * 100)
    print(f'MULTI-DECODER WEIGHT SWEEP: {config.data} | samples={sample_count}')
    print('=' * 100)
    print(_format_table(rows, metric_names))
    print(
        f'\nbest by {selection_metric}: UID={best["uid_weight"]:.2f} '
        f'SID={best["sid_weight"]:.2f} {selection_metric}={best[selection_metric]:.4f}'
    )
    print('Note: endpoint weights rerank the shared candidate union, not standalone branch retrieval.')
    print(f'JSON: {output_path}')


if __name__ == '__main__':
    main()
