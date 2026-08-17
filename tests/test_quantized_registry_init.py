import json
from pathlib import Path

from scripts.init_artifact_registry import (
    generic_meta_location,
    init_generic_registry,
    iter_generic_meta_paths,
)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + '\n')


def _quantized_export_meta(data='mind'):
    return {
        'dataset': data,
        'embedding_model': 'llama3',
        'quantizer_model': 'rqvae',
        'quantizer_scheme': 'sequential',
        'checkpoint_metric': 'recon',
        'quantizer_config': {},
        'encoder_config': {},
        'trainer_args': {},
    }


def _quantized_root_meta(data='mind'):
    return {
        'stage': 'quantized',
        'dataset': data,
        'embedding_model': 'llama3',
        'quantizer_model': 'rqvae',
        'quantizer_scheme': 'sequential',
        'status': 'completed',
    }


def test_quantized_scanner_supports_canonical_and_legacy_layouts(tmp_path):
    canonical = tmp_path / 'artifacts/quantized/mind/0123456789abcdef'
    legacy = tmp_path / 'artifacts/quantized/mind/llama3/rqvae'
    for directory in (canonical, legacy):
        _write_json(directory / 'meta.json', _quantized_root_meta())
        _write_json(directory / 'exports/recon/meta.json', _quantized_export_meta())

    paths = list(iter_generic_meta_paths(tmp_path, 'quantized'))

    assert paths == [canonical / 'meta.json', legacy / 'meta.json']


def test_quantized_export_meta_resolves_to_its_artifact_root(tmp_path):
    canonical_meta = tmp_path / 'artifacts/quantized/mind/0123456789abcdef/exports/recon/meta.json'
    legacy_meta = tmp_path / 'artifacts/quantized/mind/llama3/rqvae/exports/recon/meta.json'
    _write_json(canonical_meta, _quantized_export_meta())
    _write_json(legacy_meta, _quantized_export_meta())

    canonical = generic_meta_location(tmp_path, 'quantized', canonical_meta)
    legacy = generic_meta_location(tmp_path, 'quantized', legacy_meta)

    assert canonical['folder'] == '0123456789abcdef'
    assert canonical['run_dir'] == canonical_meta.parents[2]
    assert legacy['folder'] == 'llama3/rqvae'
    assert legacy['run_dir'] == legacy_meta.parents[2]


def test_quantized_registry_dry_run_discovers_canonical_artifact(tmp_path):
    run_dir = tmp_path / 'artifacts/quantized/mind/0123456789abcdef'
    _write_json(run_dir / 'meta.json', _quantized_root_meta())
    _write_json(run_dir / 'exports/recon/meta.json', _quantized_export_meta())

    report = init_generic_registry(
        tmp_path,
        stage='quantized',
        data=None,
        apply=False,
    )

    assert report['resolved_count'] == 1
    assert report['unresolved_count'] == 0
    assert report['resolved'][0]['folder'] == '0123456789abcdef'
