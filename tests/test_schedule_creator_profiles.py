import pytest

from utils.schedule_creator import Job


def test_job_accepts_trainer_profile_without_legacy_task_arguments():
    item = (
        Job('hybrid')
        .trainer_config('config/trainer/hybrid.yaml')
        .data('mindf')
        .model('scratch')
        .arg('content_embedding_model', 'llama3')
        .to_plan_item()
    )

    assert item['args']['config'] == 'config/trainer/hybrid.yaml'
    assert item['args']['data'] == 'mindf'
    assert 'task_type' not in item['args']


def test_profile_job_rejects_text_with_scratch_backbone():
    with pytest.raises(ValueError, match='do not support text'):
        (
            Job('invalid')
            .trainer_config('config/trainer/uid-text.yaml')
            .data('mindf')
            .model('scratch')
            .to_plan_item()
        )
