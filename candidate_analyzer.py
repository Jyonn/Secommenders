import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core import TrainConfig
from core.dataset import CompiledTestSampleDataset
from trainer import Trainer
from utils import function
from utils.config_init import ConfigInit
from utils.frequency_breakdown import count_finetune_target_frequencies
from utils.logging import setup_logging
from utils.multi_decoding import normalize_candidate_scores


ANALYZER_KEYS = {
    'samples', 'cases', 'topk', 'output', 'collaborative_embedding_dir',
}


def _as_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value if isinstance(value, list) else []


def _cosine_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _max_history_similarity(matrix, candidate_uid, history_uids):
    if matrix is None or not history_uids:
        return None
    candidate_uid = int(candidate_uid)
    valid_history = [int(uid) for uid in history_uids if 0 <= int(uid) < len(matrix)]
    if not (0 <= candidate_uid < len(matrix)) or not valid_history:
        return None
    return float(np.max(matrix[valid_history] @ matrix[candidate_uid]))


def _common_prefix(left, right, limit=None):
    if limit is not None:
        left, right = left[:limit], right[:limit]
    count = 0
    for lhs, rhs in zip(left, right):
        if int(lhs) != int(rhs):
            break
        count += 1
    return count


def _max_sid_prefix(item_codes, candidate_uid, history_uids, semantic_slots):
    if not item_codes or not history_uids:
        return None
    candidate = _as_list(item_codes[int(candidate_uid)])
    return max(
        _common_prefix(candidate, _as_list(item_codes[int(uid)]), semantic_slots)
        for uid in history_uids
    )


def _load_external_embedding_space(directory, raw_item_ids):
    if not directory:
        return None
    directory = Path(directory)
    embeddings_path = directory / 'embeddings.npy'
    item_ids_path = directory / 'item_ids.parquet'
    if not embeddings_path.exists() or not item_ids_path.exists():
        raise FileNotFoundError(
            f'collaborative embedding directory requires embeddings.npy and item_ids.parquet: {directory}'
        )
    frame = pd.read_parquet(item_ids_path)
    if len(frame.columns) != 1:
        raise ValueError(f'expected one item ID column in {item_ids_path}, got {list(frame.columns)}')
    external_ids = [str(value) for value in frame.iloc[:, 0].tolist()]
    external_matrix = np.load(embeddings_path).astype(np.float32)
    if len(external_ids) != len(external_matrix):
        raise ValueError(f'item IDs and embeddings have different lengths in {directory}')
    index = {raw_id: row for row, raw_id in enumerate(external_ids)}
    missing = [raw_id for raw_id in raw_item_ids if str(raw_id) not in index]
    if missing:
        raise ValueError(
            f'collaborative embeddings are missing {len(missing)}/{len(raw_item_ids)} compiled items; '
            f'first missing: {missing[:10]}'
        )
    aligned = external_matrix[[index[str(raw_id)] for raw_id in raw_item_ids]]
    return _cosine_rows(aligned)


def _load_item_metadata(data, raw_item_ids):
    processed_dir = Path('artifacts') / 'processed' / data
    meta_path = processed_dir / 'meta.json'
    items_path = processed_dir / 'items.parquet'
    if not meta_path.exists() or not items_path.exists():
        return [{} for _ in raw_item_ids]
    meta = json.loads(meta_path.read_text())
    frame = pd.read_parquet(items_path)
    item_col = meta['item_col']
    attrs = [attr for attr in meta.get('default_attrs', []) if attr in frame.columns]
    rows = {
        str(row[item_col]): {
            attr: '' if pd.isna(row[attr]) else str(row[attr])
            for attr in attrs
        }
        for _, row in frame.iterrows()
    }
    return [rows.get(str(raw_id), {}) for raw_id in raw_item_ids]


def _short_text(metadata, limit=72):
    text = ' | '.join(value for value in metadata.values() if value)
    if len(text) > limit:
        return text[:limit - 3] + '...'
    return text or '-'


def _rank_map(candidates):
    return {int(uid): rank for rank, uid in enumerate(candidates, start=1)}


def _branch_candidates(model, pooled, sample, sid_names, topk):
    uid_logits = model._uid_logits(pooled).float()[0]
    uid_top = torch.topk(uid_logits, k=min(topk, len(uid_logits))).indices.tolist()
    sid_top_by_name = {}
    parallel_scores = {}
    for sid_name in sid_names:
        if model._sid_decoding_mode(sid_name) == 'parallel':
            semantic, collision = model._sid_parallel_item_scores(pooled, sid_name)
            scores = (semantic + collision)[0]
            indices = torch.topk(scores, k=min(topk, len(scores))).indices.tolist()
            sid_top_by_name[sid_name] = [int(uid) for uid in indices]
            parallel_scores[sid_name] = scores
        else:
            beams = model._beam_search_sid_items_batch_with_kv_cache([sample], sid_name)
            if beams is None:
                beams = model._beam_search_sid_items_batch([sample], sid_name)
            decoded = model._decode_sid_beams_to_items(beams[0], sid_name)[:topk]
            sid_top_by_name[sid_name] = [int(uid) for uid, _, _ in decoded]
            parallel_scores[sid_name] = None

    union = set(int(uid) for uid in uid_top)
    for candidates in sid_top_by_name.values():
        union.update(candidates)
    union = sorted(union)
    uid_scores = {uid: float(uid_logits[uid].item()) for uid in union}
    normalized_sid_spaces = []
    exact_sid_by_name = {}
    for sid_name in sid_names:
        if parallel_scores[sid_name] is None:
            scores = model._score_sequential_sid_candidates(sample, union, sid_name)
        else:
            scores = {uid: float(parallel_scores[sid_name][uid].item()) for uid in union}
        exact_sid_by_name[sid_name] = scores
        normalized_sid_spaces.append(normalize_candidate_scores(scores, model.config.multi_score_normalization))
    sid_scores = {
        uid: sum(scores[uid] for scores in normalized_sid_spaces) / len(normalized_sid_spaces)
        for uid in union
    }
    fused = model._fuse_multi_candidates(uid_scores, sid_scores)
    return {
        'uid_top': [int(uid) for uid in uid_top],
        'sid_top_by_name': sid_top_by_name,
        'uid_scores': uid_scores,
        'sid_scores': sid_scores,
        'exact_sid_by_name': exact_sid_by_name,
        'fused': fused,
        'union': union,
    }


def _candidate_record(uid, source, ranks, scores, context):
    raw_item_ids = context['raw_item_ids']
    sid_codes = context['sid_codes']
    return {
        'uid': int(uid),
        'item_id': str(raw_item_ids[uid]),
        'text': _short_text(context['metadata'][uid]),
        'source': source,
        'uid_rank': ranks['uid'].get(uid),
        'sid_rank': ranks['sid'].get(uid),
        'fused_rank': ranks['fused'].get(uid),
        'uid_score': scores['uid'].get(uid),
        'sid_score': scores['sid'].get(uid),
        'fused_score': scores['fused'].get(uid),
        'content_similarity': _max_history_similarity(
            context['content_matrix'], uid, context['history_uids'],
        ),
        'collaborative_similarity': _max_history_similarity(
            context['collaborative_matrix'], uid, context['history_uids'],
        ),
        'popularity': int(context['frequencies'].get(uid, 0)),
        'sid_prefix': _max_sid_prefix(
            sid_codes, uid, context['history_uids'], context['semantic_slots'],
        ),
        'is_target': uid in context['target_uids'],
    }


def _analyze_sample(model, sample, sid_names, context, topk):
    inputs, attention_mask, lengths = model._build_batch_inputs([sample])
    hidden = model.encoder(inputs_embeds=inputs, attention_mask=attention_mask)
    pooled = hidden[torch.arange(1, device=model.device), lengths - 1]
    branches = _branch_candidates(model, pooled, sample, sid_names, topk)
    sid_candidates = []
    for values in branches['sid_top_by_name'].values():
        sid_candidates.extend(values)
    sid_candidates = sorted(
        set(sid_candidates),
        key=lambda uid: (branches['sid_scores'][uid], -uid),
        reverse=True,
    )[:topk]
    fused_candidates = [uid for uid, _ in branches['fused']]
    uid_candidates = branches['uid_top'][:topk]
    uid_set, sid_set = set(uid_candidates), set(sid_candidates)
    overlap = len(uid_set & sid_set) / max(len(uid_set | sid_set), 1)
    target_uids = set(model._sample_ground_truth_uids(sample))
    local_context = {
        **context,
        'history_uids': sample['history_uids'],
        'target_uids': target_uids,
    }
    ranks = {
        'uid': _rank_map(uid_candidates),
        'sid': _rank_map(sid_candidates),
        'fused': _rank_map(fused_candidates),
    }
    fused_scores = dict(branches['fused'])
    scores = {'uid': branches['uid_scores'], 'sid': branches['sid_scores'], 'fused': fused_scores}
    display_uids = list(dict.fromkeys(uid_candidates + sid_candidates + fused_candidates))
    records = []
    for uid in display_uids:
        in_uid, in_sid = uid in uid_set, uid in sid_set
        source = 'both' if in_uid and in_sid else ('uid-only' if in_uid else 'sid-only')
        records.append(_candidate_record(uid, source, ranks, scores, local_context))
    return {
        'user_id': str(sample['uid']),
        'history_item_ids': [str(context['raw_item_ids'][uid]) for uid in sample['history_uids']],
        'history_texts': [_short_text(context['metadata'][uid]) for uid in sample['history_uids']],
        'target_item_ids': [str(context['raw_item_ids'][uid]) for uid in sorted(target_uids)],
        'topk_overlap_jaccard': overlap,
        'target_recalled_by_uid': bool(target_uids & uid_set),
        'target_recalled_by_sid': bool(target_uids & sid_set),
        'target_recalled_by_fused': bool(target_uids & set(fused_candidates)),
        'uid_candidates': uid_candidates,
        'sid_candidates': sid_candidates,
        'fused_candidates': fused_candidates,
        'candidates': records,
    }


def _fmt(value, digits=3):
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _candidate_table(case, branch, topk):
    rank_key = f'{branch}_rank'
    rows = sorted(
        (row for row in case['candidates'] if row[rank_key] is not None),
        key=lambda row: row[rank_key],
    )[:topk]
    output = []
    for row in rows:
        output.append([
            row[rank_key], '*' if row['is_target'] else '', row['item_id'], row['source'],
            _fmt(row['content_similarity']), _fmt(row['collaborative_similarity']),
            row['popularity'], _fmt(row['sid_prefix'], 0), row['text'],
        ])
    return output


def _render_table(headers, rows):
    widths = [max(len(str(header)), *(len(str(row[i])) for row in rows)) for i, header in enumerate(headers)]
    lines = ['  '.join(str(header).ljust(widths[i]) for i, header in enumerate(headers))]
    lines.append('  '.join('-' * width for width in widths))
    lines.extend('  '.join(str(value).ljust(widths[i]) for i, value in enumerate(row)) for row in rows)
    return '\n'.join(lines)


def _render_case(case, topk):
    lines = [
        f'USER {case["user_id"]} | target={",".join(case["target_item_ids"])} '
        f'| UID/SID Jaccard={case["topk_overlap_jaccard"]:.3f}',
        'History:',
    ]
    for item_id, text in zip(case['history_item_ids'][-10:], case['history_texts'][-10:]):
        lines.append(f'  {item_id}: {text}')
    headers = ['#', 'GT', 'item', 'source', 'content', 'collab', 'pop', 'prefix', 'text']
    for branch in ('uid', 'sid', 'fused'):
        lines.extend(['', f'{branch.upper()} TOP-{topk}', _render_table(headers, _candidate_table(case, branch, topk))])
    return '\n'.join(lines)


def _aggregate(cases):
    def mean(values):
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None

    summary = {
        'case_count': len(cases),
        'mean_uid_sid_jaccard': mean([case['topk_overlap_jaccard'] for case in cases]),
        'uid_target_recall': mean([float(case['target_recalled_by_uid']) for case in cases]),
        'sid_target_recall': mean([float(case['target_recalled_by_sid']) for case in cases]),
        'fused_target_recall': mean([float(case['target_recalled_by_fused']) for case in cases]),
    }
    for source in ('uid-only', 'sid-only', 'both'):
        rows = [row for case in cases for row in case['candidates'] if row['source'] == source]
        summary[source] = {
            'count': len(rows),
            'content_similarity': mean([row['content_similarity'] for row in rows]),
            'collaborative_similarity': mean([row['collaborative_similarity'] for row in rows]),
            'popularity': mean([row['popularity'] for row in rows]),
            'sid_prefix': mean([row['sid_prefix'] for row in rows]),
        }
    return summary


def _markdown(report, cases, topk):
    summary = report['summary']
    lines = [
        f'# UID/SID Candidate Analysis: {report["data"]}', '',
        f'- Checkpoint: `{report["checkpoint"]}`',
        f'- Cases analyzed: {summary["case_count"]}',
        f'- Mean UID/SID Top-{topk} Jaccard: {_fmt(summary["mean_uid_sid_jaccard"])}',
        f'- Target recall: UID={_fmt(summary["uid_target_recall"])}; '
        f'SID={_fmt(summary["sid_target_recall"])}; fused={_fmt(summary["fused_target_recall"])}', '',
    ]
    for index, case in enumerate(cases, start=1):
        lines.extend([f'## Case {index}', '', '```text', _render_case(case, topk), '```', ''])
    return '\n'.join(lines)


def main():
    setup_logging()
    kwargs = function.argparse()
    analyzer = {key: kwargs.pop(key) for key in list(kwargs) if key in ANALYZER_KEYS}
    if not kwargs.get('load_ckpt'):
        raise ValueError('--load_ckpt is required')
    configurations = ConfigInit([], {'config': 'config/trainer/sid-uid-content-multi-decoder.yaml'}, []).parse_kwargs(kwargs)
    config = TrainConfig.from_refconfig(configurations)
    if not config.is_multi_task or 'uid' not in config.task_types or 'sid' not in config.task_types:
        raise ValueError('candidate analysis requires a SID+UID multi-decoder trainer config')

    samples = max(1, int(analyzer.get('samples', 100)))
    case_count = max(1, int(analyzer.get('cases', 5)))
    topk = max(1, int(analyzer.get('topk', 10)))
    output_path = Path(analyzer.get('output') or f'reports/{config.data}_candidate_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(config)
    trainer._load_checkpoint_for_eval(config.load_ckpt)
    model = trainer.model_core
    model.eval()
    dataset = CompiledTestSampleDataset(trainer.compiled.test)
    sid_names = config.compile_config.names_for_kind('sid', targets=True)
    raw_item_ids = trainer.compiled.uid_raw_items
    metadata = _load_item_metadata(config.data, raw_item_ids)
    content_name = config.compile_config.primary_name('embedding')
    content_tensor = trainer.compiled.embedding_matrices.get(content_name)
    content_matrix = _cosine_rows(content_tensor.cpu().numpy()) if content_tensor is not None else None
    collaborative_matrix = _load_external_embedding_space(
        analyzer.get('collaborative_embedding_dir'), raw_item_ids,
    )
    frequencies = count_finetune_target_frequencies(trainer.compiled.finetune)
    primary_sid = sid_names[0]
    sid_codes = trainer.compiled.item_views[primary_sid]
    semantic_slots = int(trainer.compiled.sid_metadata[primary_sid]['base_num_quantizers'])
    context = {
        'raw_item_ids': raw_item_ids,
        'metadata': metadata,
        'content_matrix': content_matrix,
        'collaborative_matrix': collaborative_matrix,
        'frequencies': Counter(frequencies),
        'sid_codes': sid_codes,
        'semantic_slots': semantic_slots,
    }

    cases = []
    with torch.inference_mode():
        for index in range(min(samples, len(dataset))):
            cases.append(_analyze_sample(model, dataset[index], sid_names, context, topk))
    cases.sort(
        key=lambda case: (
            case['topk_overlap_jaccard'],
            not case['target_recalled_by_fused'],
            case['user_id'],
        )
    )
    selected = cases[:min(case_count, len(cases))]
    report = {
        'data': config.data,
        'checkpoint': str(config.load_ckpt),
        'topk': topk,
        'content_representation': content_name,
        'collaborative_embedding_dir': analyzer.get('collaborative_embedding_dir'),
        'summary': _aggregate(cases),
        'cases': cases,
    }
    output_path.write_text(json.dumps(report, indent=2) + '\n')
    markdown_path = output_path.with_suffix('.md')
    markdown_path.write_text(_markdown(report, selected, topk) + '\n')

    print('\n' + '=' * 120)
    print(f'UID/SID CANDIDATE ANALYSIS: {config.data}')
    print('=' * 120)
    print(json.dumps(report['summary'], indent=2))
    for case in selected:
        print('\n' + '-' * 120)
        print(_render_case(case, topk))
    print(f'\nJSON: {output_path}')
    print(f'Markdown: {markdown_path}')


if __name__ == '__main__':
    main()
