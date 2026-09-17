import math


def normalize_candidate_scores(scores, mode):
    if not scores:
        return {}
    values = [float(value) for value in scores.values()]
    if mode == 'zscore':
        center = sum(values) / len(values)
        variance = sum((value - center) ** 2 for value in values) / len(values)
        scale = max(math.sqrt(variance), 1e-6)
        values = [(value - center) / scale for value in values]
    elif mode == 'minmax':
        minimum = min(values)
        scale = max(max(values) - minimum, 1e-6)
        values = [(value - minimum) / scale for value in values]
    elif mode != 'none':
        raise ValueError(f'unsupported multi score normalization: {mode}')
    return {uid: value for uid, value in zip(scores, values)}


def fuse_candidate_scores(
    uid_scores,
    sid_scores,
    *,
    uid_weight,
    score_normalization,
    temperature_uid,
    temperature_sid,
    output_topk,
):
    candidates = set(uid_scores) | set(sid_scores)
    if not candidates:
        return []
    if set(uid_scores) != candidates or set(sid_scores) != candidates:
        missing_uid = len(candidates - set(uid_scores))
        missing_sid = len(candidates - set(sid_scores))
        raise ValueError(
            'multi-decoder fusion requires every candidate to have complete scores; '
            f'missing uid={missing_uid} sid={missing_sid}'
        )
    uid_scores = normalize_candidate_scores(uid_scores, score_normalization)
    sid_scores = normalize_candidate_scores(sid_scores, score_normalization)
    ranked = []
    for uid in candidates:
        uid_score = uid_scores[uid] / float(temperature_uid)
        sid_score = sid_scores[uid] / float(temperature_sid)
        ranked.append((int(uid), float(uid_weight) * uid_score + (1.0 - float(uid_weight)) * sid_score))
    ranked.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return ranked[:int(output_topk)]


def fuse_candidate_ranks(
    uid_candidates,
    sid_candidates,
    *,
    uid_weight,
    rrf_k,
    output_topk,
):
    if float(rrf_k) < 0:
        raise ValueError('RRF k must be non-negative')
    uid_ranks = {int(uid): rank for rank, uid in enumerate(uid_candidates, start=1)}
    sid_ranks = {int(uid): rank for rank, uid in enumerate(sid_candidates, start=1)}
    candidates = set(uid_ranks) | set(sid_ranks)
    ranked = []
    for uid in candidates:
        score = 0.0
        if uid in uid_ranks:
            score += float(uid_weight) / (float(rrf_k) + uid_ranks[uid])
        if uid in sid_ranks:
            score += (1.0 - float(uid_weight)) / (float(rrf_k) + sid_ranks[uid])
        ranked.append((uid, score))
    ranked.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return ranked[:int(output_topk)]
