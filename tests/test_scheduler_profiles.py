from scheduler import needs_oom_precheck, uses_embedding_path


def _args(profile):
    return {
        'config': f'config/trainer/{profile}',
        'data': 'mindf',
        'model': 'scratch',
    }


def test_scheduler_reads_active_representations_from_profile():
    assert uses_embedding_path(_args('hybrid.yaml'))
    assert not needs_oom_precheck(_args('hybrid.yaml'))
    assert needs_oom_precheck(_args('sid-hybrid.yaml'))
    assert needs_oom_precheck(_args('hash.yaml'))
    assert not uses_embedding_path(_args('uid.yaml'))
