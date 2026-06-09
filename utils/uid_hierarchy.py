import math
import re


AUTO_LEVEL_PATTERN = re.compile(r'^auto(?:/(\d+))?$')


def parse_uid_cluster_levels_spec(spec: str | None) -> list[str]:
    if spec is None:
        return []
    parts = [part.strip().lower() for part in str(spec).split(',') if part.strip()]
    return parts


def resolve_uid_cluster_levels(spec: str | None, num_items: int) -> list[int]:
    if num_items <= 0:
        raise ValueError('num_items must be positive when resolving uid cluster levels')

    tokens = parse_uid_cluster_levels_spec(spec)
    if not tokens:
        raise ValueError('uid_cluster_levels is required for hierarchical uid decoding')

    auto_base = max(1, int(round(num_items ** (1.0 / (len(tokens) + 1)))))
    resolved = []
    for token in tokens:
        match = AUTO_LEVEL_PATTERN.fullmatch(token)
        if match:
            divisor = int(match.group(1)) if match.group(1) else 1
            value = max(1, int(round(auto_base / divisor)))
            resolved.append(value)
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                f'Invalid uid_cluster_levels token "{token}". '
                f'Use integers, "auto", or "auto/<divisor>".'
            ) from exc
        if value <= 0:
            raise ValueError(f'uid_cluster_levels values must be positive, got {value}')
        resolved.append(value)
    return resolved


def format_uid_cluster_levels(resolved_levels: list[int]) -> str:
    return 'x'.join(str(int(value)) for value in resolved_levels)


def parse_uid_cluster_topk(spec: str | None, depth: int) -> list[int]:
    if spec is None:
        raise ValueError('uid_cluster_topk is required for hierarchical uid decoding')
    parts = [part.strip().lower() for part in str(spec).split(',') if part.strip()]
    if len(parts) != depth:
        raise ValueError(
            f'uid_cluster_topk expects {depth} comma-separated values for hierarchy depth {depth}, '
            f'got {len(parts)} from "{spec}"'
        )
    topk = []
    for token in parts:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f'Invalid uid_cluster_topk token "{token}"') from exc
        topk.append(value)
    return topk
