from scheduler import reset_failed_experiments


def test_retry_failed_only_resets_failed_experiments():
    experiments = [
        {'name': 'done', 'status': 'done'},
        {
            'name': 'failed',
            'status': 'failed',
            'phase': 'test',
            'batch_size': 4,
            'ckpt_path': '/tmp/best.pt',
            'last_error': 'boom',
            'finished_at': 'yesterday',
            'report_uploaded_at': 'today',
            'notification_marks': {'failed': True},
        },
    ]

    reset_count = reset_failed_experiments(experiments)

    assert reset_count == 1
    assert experiments[0]['status'] == 'done'
    assert experiments[1]['status'] == 'pending'
    assert experiments[1]['phase'] == 'test'
    assert experiments[1]['batch_size'] == 4
    assert experiments[1]['ckpt_path'] == '/tmp/best.pt'
    assert experiments[1]['finished_at'] is None
    assert experiments[1]['report_uploaded_at'] is None
    assert experiments[1]['notification_marks'] == {}
