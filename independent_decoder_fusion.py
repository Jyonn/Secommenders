import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from tqdm import tqdm

from core import TrainConfig
from core.dataset import CompiledTestSampleDataset
from trainer import Trainer
from utils.artifact_identity import migrate_train_config_dict
from utils.logging import setup_logging
from utils.multi_decoding import fuse_candidate_scores


def _parse_weights(value):
    weights = [float(part.strip()) for part in str(value).split(',') if part.strip()]
    if not weights or any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError('--uid-weights must contain values in [0, 1]')
    return list(dict.fromkeys(weights))


def _load_config(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    meta_path = checkpoint_path.parent / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'checkpoint meta.json not found: {meta_path}')
    meta = json.loads(meta_path.read_text())
    config = TrainConfig(**migrate_train_config_dict(meta.get('config')))
    return replace(
        config,
        load_ckpt=str(checkpoint_path),
        test_only=True,
        device=device,
        num_gpus=1,
    )


def _load_model(checkpoint_path, device, expected_task):
    config = _load_config(checkpoint_path, device)
    if config.is_multi_task or config.task_type != expected_task:
        raise ValueError(
            f'{expected_task} checkpoint must be a standalone {expected_task}-only model; '
            f'got task_type={config.task_type}'
        )
    trainer = Trainer(config)
    checkpoint = trainer._load_checkpoint_for_eval(checkpoint_path)
    trainer.model_core.eval()
    return trainer, checkpoint


def _raw_ids(compiled, local_uids):
    return tuple(str(compiled.uid_raw_items[int(uid)]) for uid in local_uids)


def _sample_key(compiled, sample):
    return (
        str(sample['uid']),
        tuple(sorted(_raw_ids(compiled, sample['ground_truth_uids']))),
        int(sample['target_pos']),
    )


def _pair_samples(uid_compiled, sid_compiled, split):
    uid_samples = CompiledTestSampleDataset(getattr(uid_compiled, split))
    sid_samples = CompiledTestSampleDataset(getattr(sid_compiled, split))
    sid_by_key = defaultdict(list)
    for sample in sid_samples:
        sid_by_key[_sample_key(sid_compiled, sample)].append(sample)

    pairs = []
    for sample in uid_samples:
        key = _sample_key(uid_compiled, sample)
        matches = sid_by_key.get(key)
        if not matches:
            raise ValueError(f'SID {split} artifacts do not contain UID sample {key}')
        pairs.append((sample, matches.pop()))
    leftovers = sum(len(values) for values in sid_by_key.values())
    if leftovers:
        raise ValueError(f'{split} artifacts differ: {leftovers} unmatched SID samples remain')
    return pairs


def _encode_context(model, sample):
    inputs, attention_mask, lengths = model._build_batch_inputs([sample])
    hidden = model.encoder(inputs_embeds=inputs, attention_mask=attention_mask)
    return hidden[torch.arange(1, device=model.device), lengths - 1]


def _sid_retrieval(model, pooled, sample, sid_name, topk):
    if model._sid_decoding_mode(sid_name) == 'parallel':
        semantic, collision = model._sid_parallel_item_scores(pooled, sid_name)
        scores = (semantic + collision)[0]
        indices = torch.topk(scores, k=min(topk, len(scores))).indices.tolist()
        return [int(uid) for uid in indices], scores

    beams = model._beam_search_sid_items_batch_with_kv_cache([sample], sid_name)
    if beams is None:
        beams = model._beam_search_sid_items_batch([sample], sid_name)
    decoded = model._decode_sid_beams_to_items(beams[0], sid_name)[:topk]
    return [int(uid) for uid, _, _ in decoded], None


def _format_table(rows, metrics):
    headers = ['model', 'uid_weight', 'sid_weight', *metrics, 'candidates']
    values = []
    for row in rows:
        values.append([
            row['model'],
            '-' if row.get('uid_weight') is None else f'{row["uid_weight"]:.2f}',
            '-' if row.get('sid_weight') is None else f'{row["sid_weight"]:.2f}',
            *[f'{row.get(metric, 0.0):.4f}' for metric in metrics],
            f'{row.get("candidates", 0.0):.1f}',
        ])
    widths = [max(len(headers[i]), *(len(row[i]) for row in values)) for i in range(len(headers))]
    lines = ['  '.join(headers[i].ljust(widths[i]) for i in range(len(headers)))]
    lines.append('  '.join('-' * width for width in widths))
    lines.extend('  '.join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in values)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Fuse independently trained UID-only and SID-only recommenders.',
    )
    parser.add_argument('--uid-checkpoint', required=True)
    parser.add_argument('--sid-checkpoint', required=True)
    parser.add_argument('--uid-weights', default='0,0.25,0.5,0.75,1')
    parser.add_argument('--split', choices=('valid', 'test'), default='test')
    parser.add_argument('--candidate-topk', type=int, default=20)
    parser.add_argument('--output-topk', type=int, default=20)
    parser.add_argument('--score-normalization', choices=('none', 'zscore', 'minmax'), default='zscore')
    parser.add_argument('--temperature-uid', type=float, default=1.0)
    parser.add_argument('--temperature-sid', type=float, default=1.0)
    parser.add_argument('--uid-device')
    parser.add_argument('--sid-device')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.candidate_topk <= 0 or args.output_topk <= 0:
        raise ValueError('candidate and output top-k must be positive')
    if args.temperature_uid <= 0 or args.temperature_sid <= 0:
        raise ValueError('temperatures must be positive')

    setup_logging()
    weights = _parse_weights(args.uid_weights)
    uid_trainer, uid_checkpoint = _load_model(args.uid_checkpoint, args.uid_device, 'uid')
    sid_trainer, sid_checkpoint = _load_model(args.sid_checkpoint, args.sid_device, 'sid')
    uid_model = uid_trainer.model_core
    sid_model = sid_trainer.model_core
    if uid_trainer.config.data != sid_trainer.config.data:
        raise ValueError(
            f'checkpoint datasets differ: {uid_trainer.config.data} vs {sid_trainer.config.data}'
        )
    if set(map(str, uid_trainer.compiled.uid_raw_items)) != set(map(str, sid_trainer.compiled.uid_raw_items)):
        raise ValueError('UID and SID checkpoints do not use the same item vocabulary')

    uid_raw_to_local = {
        str(raw_id): uid for uid, raw_id in enumerate(uid_trainer.compiled.uid_raw_items)
    }
    sid_raw_to_local = {
        str(raw_id): uid for uid, raw_id in enumerate(sid_trainer.compiled.uid_raw_items)
    }
    pairs = _pair_samples(uid_trainer.compiled, sid_trainer.compiled, args.split)
    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]
    sid_names = sid_trainer.config.compile_config.names_for_kind('sid', targets=True)
    if len(sid_names) != 1:
        raise ValueError(f'SID-only checkpoint must expose exactly one SID target, got {sid_names}')
    sid_name = sid_names[0]
    max_sid_width = sid_model._sid_beam_width(sid_name)
    if args.candidate_topk > max_sid_width:
        graph = sid_trainer.config.representation_graph
        for target in graph['decoder']['targets']:
            if target['representation'] == sid_name:
                target.setdefault('decoding', {})['beam_width'] = args.candidate_topk

    ks = [k for k in (5, 10, 20) if k <= args.output_topk]
    if args.output_topk not in ks:
        ks.append(args.output_topk)
    totals_uid = uid_model._init_ranking_totals(ks)
    totals_sid = uid_model._init_ranking_totals(ks)
    totals_fused = {weight: uid_model._init_ranking_totals(ks) for weight in weights}
    candidate_counts = {weight: 0.0 for weight in weights}

    with torch.inference_mode():
        for uid_sample, sid_sample in tqdm(pairs, desc='independent-fusion'):
            uid_pooled = _encode_context(uid_model, uid_sample)
            uid_logits = uid_model._uid_logits(uid_pooled).float()[0]
            uid_top_local = torch.topk(
                uid_logits, k=min(args.candidate_topk, len(uid_logits)),
            ).indices.tolist()
            uid_top_raw = [str(uid_trainer.compiled.uid_raw_items[uid]) for uid in uid_top_local]

            sid_pooled = _encode_context(sid_model, sid_sample)
            sid_top_local, parallel_scores = _sid_retrieval(
                sid_model, sid_pooled, sid_sample, sid_name, args.candidate_topk,
            )
            sid_top_raw = [str(sid_trainer.compiled.uid_raw_items[uid]) for uid in sid_top_local]
            union_raw = list(dict.fromkeys(uid_top_raw + sid_top_raw))

            union_uid_local = [uid_raw_to_local[raw_id] for raw_id in union_raw]
            uid_scores = {
                uid: float(uid_logits[uid].item())
                for uid in union_uid_local
            }
            sid_union_local = [sid_raw_to_local[raw_id] for raw_id in union_raw]
            if parallel_scores is None:
                local_scores = sid_model._score_sequential_sid_candidates(
                    sid_sample, sid_union_local, sid_name,
                )
                sid_scores = {
                    uid_raw_to_local[raw_id]: local_scores[sid_raw_to_local[raw_id]]
                    for raw_id in union_raw
                }
            else:
                sid_scores = {
                    uid_raw_to_local[raw_id]: float(parallel_scores[sid_raw_to_local[raw_id]].item())
                    for raw_id in union_raw
                }

            uid_model._accumulate_ranking_metrics(
                totals_uid, ks, uid_top_local[:args.output_topk], uid_sample,
            )
            sid_ranked_uid_local = [uid_raw_to_local[raw_id] for raw_id in sid_top_raw[:args.output_topk]]
            uid_model._accumulate_ranking_metrics(totals_sid, ks, sid_ranked_uid_local, uid_sample)
            for weight in weights:
                fused = fuse_candidate_scores(
                    uid_scores,
                    sid_scores,
                    uid_weight=weight,
                    score_normalization=args.score_normalization,
                    temperature_uid=args.temperature_uid,
                    temperature_sid=args.temperature_sid,
                    output_topk=args.output_topk,
                )
                ranked = [uid for uid, _ in fused]
                uid_model._accumulate_ranking_metrics(totals_fused[weight], ks, ranked, uid_sample)
                candidate_counts[weight] += len(union_raw)

    sample_count = len(pairs)
    denominator = max(sample_count, 1)
    metric_names = [f'ndcg@{k}' for k in ks] + [f'hr@{k}' for k in ks] + ['mrr']
    rows = [
        {
            'model': 'UID-only', 'uid_weight': None, 'sid_weight': None,
            **{key: value / denominator for key, value in totals_uid.items()},
            'candidates': float(args.candidate_topk),
        },
        {
            'model': 'SID-only', 'uid_weight': None, 'sid_weight': None,
            **{key: value / denominator for key, value in totals_sid.items()},
            'candidates': float(args.candidate_topk),
        },
    ]
    for weight in weights:
        rows.append({
            'model': 'Independent fusion',
            'uid_weight': weight,
            'sid_weight': 1.0 - weight,
            **{key: value / denominator for key, value in totals_fused[weight].items()},
            'candidates': candidate_counts[weight] / denominator,
        })

    selection_metric = next(
        (name for name in str(uid_trainer.config.main_metric).split('|') if name in rows[-1]),
        'ndcg@10' if 'ndcg@10' in rows[-1] else metric_names[0],
    )
    fusion_rows = rows[2:]
    best = max(fusion_rows, key=lambda row: row.get(selection_metric, 0.0))
    output_path = Path(
        args.output or f'reports/{uid_trainer.config.data}_{args.split}_independent_decoder_fusion.json'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'data': uid_trainer.config.data,
        'split': args.split,
        'uid_checkpoint': str(args.uid_checkpoint),
        'uid_checkpoint_epoch': uid_checkpoint.get('epoch'),
        'sid_checkpoint': str(args.sid_checkpoint),
        'sid_checkpoint_epoch': sid_checkpoint.get('epoch'),
        'samples': sample_count,
        'candidate_topk_per_branch': args.candidate_topk,
        'output_topk': args.output_topk,
        'score_normalization': args.score_normalization,
        'selection_metric': selection_metric,
        'best_uid_weight': best['uid_weight'],
        'best_sid_weight': best['sid_weight'],
        'results': rows,
    }
    output_path.write_text(json.dumps(report, indent=2) + '\n')

    print('\n' + '=' * 100)
    print(
        f'INDEPENDENT DECODER FUSION: {uid_trainer.config.data} '
        f'| split={args.split} | samples={sample_count}'
    )
    print('=' * 100)
    print(_format_table(rows, metric_names))
    print(
        f'\nbest by {selection_metric}: UID={best["uid_weight"]:.2f} '
        f'SID={best["sid_weight"]:.2f} {selection_metric}={best[selection_metric]:.4f}'
    )
    print(f'JSON: {output_path}')


if __name__ == '__main__':
    main()
