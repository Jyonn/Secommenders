from scripts.init_artifact_registry import migrate_legacy_scratch_config


def test_registry_migrates_unversioned_scratch_to_legacy():
    config, migrated = migrate_legacy_scratch_config(
        {'config': {'model': 'scratch'}},
        {'model': 'scratch', 'hidden_size': 256},
    )

    assert migrated is True
    assert config['model'] == 'scratchlegacy'


def test_registry_preserves_versioned_scratch_llama():
    meta = {
        'config': {'model': 'scratch'},
        'artifact_identity': {
            'spec': {'config': {'model': 'scratch', 'backbone_architecture': 'llama-v1'}},
        },
    }

    config, migrated = migrate_legacy_scratch_config(meta, {'model': 'scratch'})

    assert migrated is False
    assert config['model'] == 'scratch'
