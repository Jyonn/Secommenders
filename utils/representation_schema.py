import re
from copy import deepcopy

from oba import NotFound, Obj

from utils.embedding_fusion import normalize_embedding_fusion
from utils.experiment_template import (
    HASH_QUANTIZER_CONFIG_DEFAULTS,
    QUANTIZER_TRAINER_DEFAULTS,
    SID_ENCODER_CONFIG_DEFAULTS,
    SID_QUANTIZER_CONFIG_DEFAULTS,
    merge_defaults,
)
from utils.compile import canonicalize_repr_type, canonicalize_task_type


SCHEMA_VERSION = 'trainer.v4'
SUPPORTED_TYPES = {'uid', 'sid', 'hash', 'text', 'embedding'}
NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_-]*$')


def plain(value):
    if isinstance(value, NotFound):
        return None
    if isinstance(value, Obj):
        value = value.__dict__['__obj__']
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def _names(value, field):
    value = plain(value)
    if isinstance(value, str):
        value = [part.strip() for part in value.split(',') if part.strip()]
    if not isinstance(value, list) or not value:
        raise ValueError(f'{field} must be a non-empty list of representation names')
    names = [str(item).strip().lower() for item in value]
    if len(names) != len(set(names)):
        raise ValueError(f'{field} contains duplicate representation names: {names}')
    return names


def _source_spec(raw):
    sources = plain(raw.get('sources'))
    if not sources:
        raise ValueError('sid/hash/embedding representation requires non-empty sources')
    return normalize_embedding_fusion({
        'sources': sources,
        'fusion': plain((raw.get('fusion') or {}).get('method')) or 'concat',
        'normalize_output': plain((raw.get('fusion') or {}).get('normalize_output')) or False,
    })


def _normalize_representation(name, value):
    name = str(name).strip().lower()
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f'invalid representation name {name!r}; expected lowercase letters, numbers, _ or -'
        )
    raw = plain(value) or {}
    kind = str(raw.get('type') or '').strip().lower()
    if kind not in SUPPORTED_TYPES:
        raise ValueError(f'representation {name} has unsupported type {kind!r}')
    spec = {'type': kind}
    if kind in {'sid', 'hash', 'embedding'}:
        spec['embedding'] = _source_spec(raw)
    if kind == 'sid':
        codec = raw.get('codec') or {}
        quantizer = codec.get('quantizer') or {}
        encoder = codec.get('encoder') or {}
        trainer = codec.get('trainer') or {}
        spec['codec'] = {
            'name': str(codec.get('name') or 'rqvae').strip().lower(),
            'export': str(codec.get('export') or 'coll').strip().lower(),
            'quantizer': merge_defaults(SID_QUANTIZER_CONFIG_DEFAULTS, quantizer),
            'encoder': {
                'name': str(encoder.get('name') or 'mlp').strip().lower(),
                'config': merge_defaults(SID_ENCODER_CONFIG_DEFAULTS, encoder.get('config') or {}),
            },
            'trainer': merge_defaults(QUANTIZER_TRAINER_DEFAULTS, trainer),
        }
    elif kind == 'hash':
        codec = raw.get('codec') or {}
        spec['codec'] = {
            'name': str(codec.get('name') or 'simhash').strip().lower(),
            'quantizer': merge_defaults(HASH_QUANTIZER_CONFIG_DEFAULTS, codec.get('quantizer') or {}),
        }
    elif kind == 'uid' and raw.get('hierarchy'):
        spec['hierarchy'] = deepcopy(raw['hierarchy'])
    return name, spec


def normalize_representation_graph(representations, encoder, decoder):
    raw_catalog = plain(representations)
    if not isinstance(raw_catalog, dict) or not raw_catalog:
        raise ValueError('representations must be a non-empty mapping keyed by stable representation name')
    catalog = dict(_normalize_representation(name, value) for name, value in raw_catalog.items())
    encoder_raw = plain(encoder) or {}
    decoder_raw = plain(decoder) or {}
    encoder_names = _names(encoder_raw.get('representations'), 'encoder.representations')
    raw_targets = plain(decoder_raw.get('targets'))
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError('decoder.targets must be a non-empty list')
    targets = []
    for value in raw_targets:
        entry = {'representation': value} if isinstance(value, str) else dict(value or {})
        name = str(entry.get('representation') or '').strip().lower()
        if not name:
            raise ValueError('each decoder target requires representation')
        entry['representation'] = name
        targets.append(entry)
    target_names = [entry['representation'] for entry in targets]
    if len(target_names) != len(set(target_names)):
        raise ValueError(f'decoder.targets contains duplicates: {target_names}')
    referenced = set(encoder_names + target_names)
    missing = sorted(referenced - set(catalog))
    if missing:
        raise ValueError(f'unknown representation references: {missing}')
    if not set(target_names).issubset(encoder_names):
        raise ValueError('every decoder target must also appear in encoder.representations')
    if encoder_names[:len(target_names)] != target_names:
        raise ValueError('decoder targets must lead encoder.representations in the same order')
    combine = str(encoder_raw.get('combine') or 'concat').strip().lower()
    if combine != 'concat':
        raise ValueError('trainer.v4 named representations currently require encoder.combine=concat')
    active = {name: catalog[name] for name in sorted(referenced)}
    target_types = [active[name]['type'] for name in target_names]
    if len(target_types) > 1 and target_types != ['sid', 'uid']:
        raise ValueError('multiple decoder targets currently support one sid followed by one uid')
    for kind in ('uid', 'sid', 'hash', 'text'):
        names = [name for name in encoder_names if active[name]['type'] == kind]
        if len(names) > 1:
            raise ValueError(f'trainer.v4 currently supports at most one active {kind} representation')
    return {
        'schema_version': SCHEMA_VERSION,
        'representations': active,
        'encoder': {
            'representations': encoder_names,
            'combine': combine,
        },
        'decoder': {
            'targets': targets,
            'multiple': plain(decoder_raw.get('multiple')) or {},
        },
    }


def representation_kind(graph, name):
    return graph['representations'][name]['type']


def names_of_type(graph, kind, *, encoder=True, decoder=False):
    names = []
    if encoder:
        names.extend(graph['encoder']['representations'])
    if decoder:
        names.extend(entry['representation'] for entry in graph['decoder']['targets'])
    return [name for name in dict.fromkeys(names) if representation_kind(graph, name) == kind]


def semantic_graph_contract(graph):
    """Return the model-facing graph contract without user-facing instance names."""
    graph = plain(graph) or {}
    catalog = graph.get('representations') or {}
    encoder = graph.get('encoder') or {}
    decoder = graph.get('decoder') or {}
    encoder_names = list(encoder.get('representations') or [])
    index_by_name = {name: index for index, name in enumerate(encoder_names)}
    targets = []
    for target in decoder.get('targets') or []:
        target = dict(target or {})
        name = target.pop('representation')
        targets.append({'encoder_index': index_by_name[name], **target})
    return {
        'representations': [deepcopy(catalog[name]) for name in encoder_names],
        'encoder': {'combine': encoder.get('combine') or 'concat'},
        'decoder': {
            'targets': targets,
            'multiple': deepcopy(decoder.get('multiple') or {}),
        },
    }


def graph_from_legacy_config(config):
    raw = plain(config) or {}
    task_types = canonicalize_task_type(raw.get('task_type')).split('+')
    encoder_types = canonicalize_repr_type(raw.get('task_type'), raw.get('repr_type')).split('+')
    upstreams = raw.get('upstreams') or {}
    catalog = {}
    for kind in encoder_types:
        if kind in {'uid', 'text'}:
            catalog[kind] = {'type': kind}
        elif kind == 'embedding':
            embedding = raw.get('repr_embedding')
            if not embedding:
                embedding = normalize_embedding_fusion({}, legacy_model=raw.get('repr_source_model'))
            catalog[kind] = {'type': kind, 'embedding': embedding}
        elif kind == 'sid':
            upstream = upstreams.get('sid') or {}
            embedding = upstream.get('embedding') or normalize_embedding_fusion(
                {}, legacy_model=upstream.get('embedding_model') or raw.get('repr_source_model')
            )
            quantizer = upstream.get('quantizer') or {}
            catalog[kind] = {
                'type': kind,
                'embedding': embedding,
                'codec': {
                    'name': quantizer.get('name') or raw.get('sid_coder') or 'rqvae',
                    'export': upstream.get('export') or raw.get('sid_export') or 'coll',
                    'quantizer': merge_defaults(SID_QUANTIZER_CONFIG_DEFAULTS, quantizer.get('config') or {}),
                    'encoder': upstream.get('encoder') or {
                        'name': 'mlp', 'config': deepcopy(SID_ENCODER_CONFIG_DEFAULTS),
                    },
                    'trainer': merge_defaults(QUANTIZER_TRAINER_DEFAULTS, upstream.get('trainer') or {}),
                },
            }
        elif kind == 'hash':
            upstream = upstreams.get('hash') or {}
            embedding = upstream.get('embedding') or normalize_embedding_fusion(
                {}, legacy_model=upstream.get('embedding_model') or raw.get('repr_source_model')
            )
            quantizer = upstream.get('quantizer') or {}
            catalog[kind] = {
                'type': kind,
                'embedding': embedding,
                'codec': {
                    'name': quantizer.get('name') or raw.get('hash_coder') or 'simhash',
                    'quantizer': merge_defaults(HASH_QUANTIZER_CONFIG_DEFAULTS, quantizer.get('config') or {}),
                },
            }
    targets = []
    for kind in task_types:
        target = {'representation': kind}
        if kind == 'uid':
            target['decoding'] = {
                'mode': raw.get('uid_decoding') or 'flat',
                'topk': raw.get('uid_cluster_topk'),
            }
        elif kind == 'sid':
            target['decoding'] = {
                'mode': raw.get('code_decoding') or 'auto',
                'beam_width': int(raw.get('code_beam_width') or 20),
                'beam_chunk_size': raw.get('code_beam_chunk_size'),
                'collision_loss_weight': float(raw.get('code_collision_loss_weight', 0.1)),
            }
        targets.append(target)
    return {
        'schema_version': SCHEMA_VERSION,
        'representations': catalog,
        'encoder': {
            'representations': encoder_types,
            'combine': str(raw.get('repr_combine') or 'concat').lower(),
        },
        'decoder': {'targets': targets, 'multiple': {}},
    }


def upstreams_from_graph(graph):
    catalog = graph['representations']
    names = graph['encoder']['representations']
    upstreams = {}
    sid_names = [name for name in names if catalog[name]['type'] == 'sid']
    if sid_names:
        spec = catalog[sid_names[0]]
        codec = spec['codec']
        upstreams['sid'] = {
            'kind': 'quantized',
            'embedding_model': None,
            'embedding': spec['embedding'],
            'export': codec['export'],
            'quantizer': {'name': codec['name'], 'config': codec['quantizer']},
            'encoder': codec['encoder'],
            'trainer': codec['trainer'],
        }
    hash_names = [name for name in names if catalog[name]['type'] == 'hash']
    if hash_names:
        spec = catalog[hash_names[0]]
        codec = spec['codec']
        upstreams['hash'] = {
            'kind': 'quantized',
            'embedding_model': None,
            'embedding': spec['embedding'],
            'export': 'hash',
            'quantizer': {'name': codec['name'], 'config': codec['quantizer']},
        }
    uid_names = [name for name in names if catalog[name]['type'] == 'uid']
    if uid_names and catalog[uid_names[0]].get('hierarchy'):
        upstreams['uid'] = {'kind': 'clustered', 'clusterer': catalog[uid_names[0]]['hierarchy']}
    return upstreams
