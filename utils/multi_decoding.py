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


def uid_frequency_gate(frequency, *, mode, uid_weight, threshold, smoothing):
    if mode == 'fixed':
        return float(uid_weight)
    if mode != 'frequency':
        raise ValueError(f'unsupported multi fusion mode: {mode}')
    logit = (
        math.log1p(max(0.0, float(frequency)))
        - math.log1p(max(0.0, float(threshold)))
    ) / float(smoothing)
    return 1.0 / (1.0 + math.exp(-logit))


def fuse_candidate_scores(
    uid_scores,
    sid_scores,
    frequencies,
    *,
    fusion_mode,
    uid_weight,
    score_normalization,
    temperature_uid,
    temperature_sid,
    frequency_threshold,
    frequency_smoothing,
    output_topk,
):
    candidates = set(uid_scores) | set(sid_scores)
    if not candidates:
        return []
    uid_scores = normalize_candidate_scores(uid_scores, score_normalization)
    sid_scores = normalize_candidate_scores(sid_scores, score_normalization)
    uid_floor = min(uid_scores.values(), default=0.0) - 1.0
    sid_floor = min(sid_scores.values(), default=0.0) - 1.0
    ranked = []
    for uid in candidates:
        gate = uid_frequency_gate(
            frequencies.get(uid, 0),
            mode=fusion_mode,
            uid_weight=uid_weight,
            threshold=frequency_threshold,
            smoothing=frequency_smoothing,
        )
        uid_score = uid_scores.get(uid, uid_floor) / float(temperature_uid)
        sid_score = sid_scores.get(uid, sid_floor) / float(temperature_sid)
        ranked.append((int(uid), gate * uid_score + (1.0 - gate) * sid_score))
    ranked.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return ranked[:int(output_topk)]
